"""Multi-source feature engineering & structured Feature Store catalog for relational credit graphs.

Builds three derived feature layers on top of the base + point-in-time aggregated frame:
- Layer 1 (`domain_slice`): directional / status slice proportions and opposing-slice ratios (e.g. outflow/inflow,
  overdue/total, fail/success) from detail tables;
- Layer 2 (`temporal_ratio`): short-vs-long lookback velocity & concentration ratios (e.g. 7d vs 30d, 30d vs all);
- Layer 3 (`cross_derived`): cross-hub interaction ratios & differences across master tables (order, user) and
  multi-source detail tables (credit bureau, cashflow, behavior), combining deterministic domain templates with
  optional LLM-suggested pairs guarded by a strict column whitelist.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.core.llm import LLMError, get_llm_client

LAYER_LABELS = {
    "base": "主表原生特征",
    "window_agg": "时序窗口聚合",
    "domain_slice": "多源领域切片",
    "temporal_ratio": "时序趋势与激增度",
    "cross_derived": "跨表交叉衍生",
}

_WINDOW_PAIRS = [(7, 30), (30, 90), (30, 0), (7, 0)]
_OPPOSING_SLICES = [
    ("out", "in"),
    ("debit", "credit"),
    ("pay", "income"),
    ("expense", "income"),
    ("overdue", "normal"),
    ("overdue", "settled"),
    ("fail", "success"),
    ("failed", "success"),
    ("reject", "pass"),
]

_LOAN_PATTERNS = ("*loan_amount*", "*apply_amount*", "*principal*", "*borrow_amount*", "amount")
_CAPACITY_PATTERNS = ("*income*", "*salary*", "*asset*", "*credit_limit*", "*quota*")
_DEBT_PATTERNS = ("*debt*", "*overdue_amount*", "*balance*", "*arrears*", "*used_limit*")
_MAX_CROSS_FEATURES = 24


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Computes `num / (|den| + 1.0)` while preserving NaN when both sides are missing."""
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    return n / (d.abs() + 1.0)


def _match_any(name: str, patterns: Sequence[str]) -> bool:
    low = name.lower().split(".")[-1]
    return any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns)


def _derive_temporal_and_slice_ratios(
    df: pd.DataFrame,
    col_meta: Dict[str, Dict[str, Any]],
    kinds: Dict[str, str],
) -> Dict[str, pd.Series]:
    """L2 temporal velocity ratios (short vs long window) + L1 slice proportions & opposing-slice ratios."""
    new_cols: Dict[str, pd.Series] = {}
    tables = sorted({m.get("table") for m in col_meta.values() if m.get("table")})

    for t in tables:
        cat = next((m.get("category") for m in col_meta.values() if m.get("table") == t and m.get("category")), None)
        path = next((m.get("graph_path") for m in col_meta.values() if m.get("table") == t and m.get("graph_path")), t)
        src_tables = next((m.get("source_tables") for m in col_meta.values() if m.get("table") == t and m.get("source_tables")), [t])

        # 1. Window velocity ratios: cnt, sum, avg across (w_short, w_long)
        for w_short, w_long in _WINDOW_PAIRS:
            s_sfx = f"{w_short}d"
            l_sfx = f"{w_long}d" if w_long else "all"
            l_label = f"近{w_long}天" if w_long else "历史全部"
            c_short, c_long = f"{t}.cnt_{s_sfx}", f"{t}.cnt_{l_sfx}"
            if c_short in df.columns and c_long in df.columns:
                fname = f"{t}.cnt_ratio_{s_sfx}_{l_sfx}"
                new_cols[fname] = _safe_div(df[c_short], df[c_long])
                col_meta[fname] = {
                    "table": t,
                    "column": fname,
                    "comment": f"近{w_short}天与{l_label} {t} 记录数激增比",
                    "category": cat,
                    "layer": "temporal_ratio",
                    "formula": f"{c_short} / ({c_long} + 1)",
                    "graph_path": path,
                    "source_tables": src_tables,
                }
                kinds[fname] = "numeric"

            # Find value columns that have both short and long window aggregates
            prefix = f"{t}."
            for col in list(df.columns):
                if not col.startswith(prefix):
                    continue
                for agg, agg_cn in (("sum", "总额占比"), ("avg", "均值变化比")):
                    tail = f"__{agg}_{s_sfx}"
                    if col.endswith(tail) and "__slice_" not in col:
                        base_vcol = col[len(prefix) : -len(tail)]
                        long_col = f"{t}.{base_vcol}__{agg}_{l_sfx}"
                        if long_col in df.columns:
                            fname = f"{t}.{base_vcol}__{agg}_ratio_{s_sfx}_{l_sfx}"
                            new_cols[fname] = _safe_div(df[col], df[long_col])
                            col_meta[fname] = {
                                "table": t,
                                "column": fname,
                                "comment": f"{base_vcol} 近{w_short}天与{l_label}{agg_cn}",
                                "category": cat,
                                "layer": "temporal_ratio",
                                "formula": f"{col} / (|{long_col}| + 1)",
                                "graph_path": path,
                                "source_tables": src_tables,
                            }
                            kinds[fname] = "numeric"

        # 2. Slice count proportions & opposing directional ratios
        slice_cnt_re = re.compile(rf"^{re.escape(t)}\.cnt__slice_([a-zA-Z0-9_]+)_([a-zA-Z0-9]+)_(\d+d|all)$")
        for col in list(df.columns):
            m = slice_cnt_re.match(col)
            if not m:
                continue
            sc, slug, sfx = m.group(1), m.group(2), m.group(3)
            total_cnt_col = f"{t}.cnt_{sfx}"
            if total_cnt_col in df.columns:
                fname = f"{t}.cnt__slice_{sc}_{slug}_prop_{sfx}"
                new_cols[fname] = _safe_div(df[col], df[total_cnt_col])
                col_meta[fname] = {
                    "table": t,
                    "column": fname,
                    "comment": f"{t} {sc}={slug} 笔数占比（{sfx}）",
                    "category": cat,
                    "layer": "domain_slice",
                    "formula": f"{col} / ({total_cnt_col} + 1)",
                    "graph_path": path,
                    "source_tables": src_tables,
                }
                kinds[fname] = "numeric"

        # Opposing slice sum ratios (e.g. out/in, overdue/normal, fail/success)
        slice_sum_re = re.compile(
            rf"^{re.escape(t)}\.([a-zA-Z0-9_]+)__slice_([a-zA-Z0-9_]+)_([a-zA-Z0-9]+)__sum_(\d+d|all)$"
        )
        by_group: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        for col in list(df.columns):
            m = slice_sum_re.match(col)
            if m:
                vcol, sc, slug, sfx = m.group(1), m.group(2), m.group(3), m.group(4)
                by_group.setdefault((vcol, sc, sfx), {})[slug.lower()] = col

        for (vcol, sc, sfx), slug_map in by_group.items():
            for s_num, s_den in _OPPOSING_SLICES:
                if s_num in slug_map and s_den in slug_map:
                    c_num, c_den = slug_map[s_num], slug_map[s_den]
                    fname = f"{t}.{vcol}__slice_{sc}_{s_num}_div_{s_den}_{sfx}"
                    new_cols[fname] = _safe_div(df[c_num], df[c_den])
                    col_meta[fname] = {
                        "table": t,
                        "column": fname,
                        "comment": f"{t}.{vcol} {sc}({s_num}/{s_den}) 比率（{sfx}）",
                        "category": cat,
                        "layer": "domain_slice",
                        "formula": f"{c_num} / (|{c_den}| + 1)",
                        "graph_path": path,
                        "source_tables": src_tables,
                    }
                    kinds[fname] = "numeric"

    return new_cols


def _llm_recommend_cross_pairs(
    numeric_cols: List[str],
    col_meta: Dict[str, Dict[str, Any]],
    max_pairs: int = 6,
) -> List[Dict[str, str]]:
    """Optional LLM recommender for cross-table pairs; strictly whitelisted against `numeric_cols`."""
    client = get_llm_client()
    if not client.available or len(numeric_cols) < 2:
        return []
    allowed = set(numeric_cols)
    listing = "\n".join(
        f"{c}\t{col_meta.get(c, {}).get('table', '')}\t{(col_meta.get(c, {}).get('comment') or '')[:40]}"
        for c in numeric_cols[:50]
    )
    try:
        data = client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是信贷风控特征工程专家。请从给定数值字段列表中挑选至多 "
                        f"{max_pairs} 组跨表交叉衍生特征（如申请金额/收入、负债/收入、近期流水/申请金额）。"
                        "num 与 den 必须完全来自给定字段名，op 仅限 \"div\" 或 \"sub\"。"
                        "只输出 JSON 对象: {\"pairs\": [{\"num\": \"...\", \"den\": \"...\", \"op\": \"div\", \"comment\": \"中文释义\"}]}"
                    ),
                },
                {"role": "user", "content": "字段名\t来源表\t注释\n" + listing},
            ],
            max_tokens=600,
        )
    except LLMError:
        return []
    raw_pairs = (data or {}).get("pairs")
    if not isinstance(raw_pairs, list):
        return []
    valid: List[Dict[str, str]] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        num, den, op = item.get("num"), item.get("den"), item.get("op", "div")
        if (
            isinstance(num, str)
            and isinstance(den, str)
            and num in allowed
            and den in allowed
            and num != den
            and op in ("div", "sub")
        ):
            valid.append({"num": num, "den": den, "op": op, "comment": str(item.get("comment") or f"{num} 与 {den} 交叉衍生")})
    return valid[:max_pairs]


def _derive_cross_features(
    profile: Dict[str, Any],
    df: pd.DataFrame,
    col_meta: Dict[str, Dict[str, Any]],
    kinds: Dict[str, str],
) -> Dict[str, pd.Series]:
    """L3 cross-hub / cross-domain ratios & differences across order, user, credit, cashflow and behavior."""
    numeric_cols = [c for c, k in kinds.items() if k in ("numeric", "numeric_text") and c in df.columns]
    if len(numeric_cols) < 2:
        return {}

    loan_cols = [c for c in numeric_cols if _match_any(c, _LOAN_PATTERNS) and "__" not in c]
    cap_cols = [c for c in numeric_cols if _match_any(c, _CAPACITY_PATTERNS) and "__" not in c]
    debt_cols = [c for c in numeric_cols if _match_any(c, _DEBT_PATTERNS)]
    agg_sum_cols = [
        c for c in numeric_cols if ("__sum_30d" in c or "__sum_all" in c or "__max_all" in c) and not c.startswith("cross.")
    ]

    planned: List[Tuple[str, str, str, str]] = []  # (num, den, op, comment)
    seen_pairs: set = set()

    def add_pair(num: str, den: str, op: str, comment: str) -> None:
        if num == den or (num, den, op) in seen_pairs or len(planned) >= _MAX_CROSS_FEATURES:
            return
        t_num = col_meta.get(num, {}).get("table", "")
        t_den = col_meta.get(den, {}).get("table", "")
        if t_num and t_den and t_num == t_den:
            return
        seen_pairs.add((num, den, op))
        planned.append((num, den, op, comment))

    # Rule 1: Loan amount vs User capacity (income / salary / limit) -> repayment burden
    for lc in loan_cols[:2]:
        for cc in cap_cols[:3]:
            add_pair(lc, cc, "div", f"申请金额({lc})与收入/额度({cc})偿债压力比")

    # Rule 2: Multi-source cashflow / credit / behavior aggregates vs Loan amount & User capacity
    for ac in agg_sum_cols[:8]:
        a_cmt = col_meta.get(ac, {}).get("comment") or ac
        for lc in loan_cols[:1]:
            add_pair(ac, lc, "div", f"{a_cmt} 与申请金额({lc})覆盖比")
        for cc in cap_cols[:2]:
            add_pair(ac, cc, "div", f"{a_cmt} 与收入/额度({cc})比率")

    # Rule 3: Debt / Overdue amount vs User capacity (DTI)
    for dc in debt_cols[:4]:
        for cc in cap_cols[:2]:
            add_pair(dc, cc, "div", f"负债/逾期({dc})与收入/额度({cc})负债比(DTI)")

    # Rule 4: Optional LLM semantic cross pairs (guarded by whitelist)
    if (profile.get("categories") or {}).get("use_llm", True):
        for rec in _llm_recommend_cross_pairs(numeric_cols, col_meta):
            add_pair(rec["num"], rec["den"], rec["op"], rec["comment"])

    new_cols: Dict[str, pd.Series] = {}
    for num, den, op, comment in planned:
        safe_num = num.replace(".", "_")
        safe_den = den.replace(".", "_")
        fname = f"cross.{safe_num}__{op}__{safe_den}"
        if fname in df.columns:
            continue
        if op == "div":
            series = _safe_div(df[num], df[den])
            formula = f"{num} / (|{den}| + 1)"
        else:
            series = pd.to_numeric(df[num], errors="coerce") - pd.to_numeric(df[den], errors="coerce")
            formula = f"{num} - {den}"
        new_cols[fname] = series
        t_num = col_meta.get(num, {}).get("table") or "base"
        t_den = col_meta.get(den, {}).get("table") or "base"
        src = list(dict.fromkeys([t_num, t_den]))
        cat = col_meta.get(num, {}).get("category") or col_meta.get(den, {}).get("category") or "跨表交叉衍生"
        col_meta[fname] = {
            "table": "+".join(src),
            "column": fname,
            "comment": comment,
            "category": cat,
            "layer": "cross_derived",
            "formula": formula,
            "graph_path": f"{t_num} × {t_den}",
            "source_tables": src,
        }
        kinds[fname] = "numeric"

    return new_cols


def derive_all_features(
    profile: Dict[str, Any],
    df: pd.DataFrame,
    col_meta: Dict[str, Dict[str, Any]],
    kinds: Dict[str, str],
) -> pd.DataFrame:
    """Appends L2 temporal velocity ratios and L3 cross-hub features to `df` and updates `col_meta` & `kinds`."""
    fe_cfg = profile.get("feature_engineering") or {}
    auto_enabled = bool((profile.get("auto_join") or {}).get("enabled"))
    if not fe_cfg.get("enabled", auto_enabled):
        return df

    # Tag existing columns that don't yet have a `layer`
    for col, meta in col_meta.items():
        if "layer" not in meta:
            if "__slice_" in col:
                meta["layer"] = "domain_slice"
            elif re.search(r"(\.cnt_(\d+d|all)$|__(sum|avg|max|min)_(\d+d|all)$)", col):
                meta["layer"] = "window_agg"
            else:
                meta["layer"] = "base"

    l2_cols = _derive_temporal_and_slice_ratios(df, col_meta, kinds)
    if l2_cols:
        df = pd.concat([df, pd.DataFrame(l2_cols, index=df.index)], axis=1)

    l3_cols = _derive_cross_features(profile, df, col_meta, kinds)
    if l3_cols:
        df = pd.concat([df, pd.DataFrame(l3_cols, index=df.index)], axis=1)

    return df


def build_feature_catalog(
    mined: Dict[str, Any],
    col_meta: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Constructs the structured Feature Store catalog and per-layer summary."""
    suggested = set(mined.get("suggested_features") or [])
    catalog: List[Dict[str, Any]] = []
    by_layer: Dict[str, Dict[str, Any]] = {}

    for m in mined.get("feature_metrics") or []:
        feat = m["feature"]
        meta = col_meta.get(feat) or {}
        layer = meta.get("layer") or (
            "cross_derived"
            if feat.startswith("cross.")
            else "domain_slice"
            if "__slice_" in feat
            else "temporal_ratio"
            if "_ratio_" in feat
            else "window_agg"
            if re.search(r"(\.cnt_(\d+d|all)$|__(sum|avg|max|min)_(\d+d|all)$)", feat)
            else "base"
        )
        tbl = meta.get("table") or m.get("source_table") or ""
        src_tables = meta.get("source_tables") or ([tbl] if tbl else [])
        is_sel = feat in suggested
        entry = {
            "feature": feat,
            "layer": layer,
            "layer_label": LAYER_LABELS.get(layer, layer),
            "domain": m.get("category") or meta.get("category") or "其他",
            "source_tables": src_tables,
            "graph_path": meta.get("graph_path") or (" -> ".join(src_tables) if src_tables else tbl),
            "formula": meta.get("formula") or feat,
            "comment": m.get("comment") or meta.get("comment") or feat,
            "iv": m["iv"],
            "ks": m["ks"],
            "auc": m["auc"],
            "psi": m["psi"],
            "missing_rate": m["missing_rate"],
            "selected": is_sel,
        }
        catalog.append(entry)

        grp = by_layer.setdefault(
            layer,
            {
                "layer": layer,
                "layer_label": LAYER_LABELS.get(layer, layer),
                "mined": 0,
                "suggested": 0,
                "iv_sum": 0.0,
                "top_features": [],
            },
        )
        grp["mined"] += 1
        grp["iv_sum"] += float(m["iv"])
        if is_sel:
            grp["suggested"] += 1
            if len(grp["top_features"]) < 5:
                grp["top_features"].append(feat)

    order_rank = list(LAYER_LABELS.keys())
    layer_summary = [
        {
            "layer": g["layer"],
            "layer_label": g["layer_label"],
            "mined": g["mined"],
            "suggested": g["suggested"],
            "avg_iv": round(g["iv_sum"] / max(g["mined"], 1), 4),
            "top_features": g["top_features"],
        }
        for g in sorted(by_layer.values(), key=lambda x: order_rank.index(x["layer"]) if x["layer"] in order_rank else 99)
    ]
    return catalog, layer_summary
