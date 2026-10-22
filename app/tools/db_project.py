"""Config-driven, multi-table indicator mining for relational databases.

Real credit databases rarely hold the label next to the features: features sit in user / application tables while
performance (status, overdue days, repayment counts) is derived from rules over the application table. Every
project-specific fact therefore lives in a JSON *project profile* (`config/db_profile.json`, override with env
`DB_PROFILE_PATH`), never in code:

- `sample`        base table, its primary key, user key and the date column used for the out-of-time split;
- `joins`         tables to LEFT JOIN as extra features (`table` or `table_pattern`, `left_key` -> `right_key`);
- `population`    which rows are modelled (e.g. disbursed orders), as a condition tree;
- `label`         `bad` / `good` condition trees (rows matching neither are indeterminate and dropped);
- `params`        named thresholds referenced as `"$name"` inside conditions (e.g. `"$bad_overdue_days"`);
- `exclude`       column patterns that may never become features (post-loan outcomes, PII...), each with a reason;
- `performance_checks`  columns whose direction must agree with the label (sanity check of the label rule);
- `categories`    ordered name / comment rules that classify features, plus a per-table fallback;
- `discovery`     how candidate detail tables are found among the many tables of the database.

Safety is unchanged: statements are SQLAlchemy Core constructs over names validated against the database metadata,
values are bound, conditions only allow a fixed operator set, and the read-only session guard still applies. Columns
excluded by the profile are not even read from the database unless a label rule needs them.
"""

from __future__ import annotations

import fnmatch
import json
import os
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sa

from app.core.llm import LLMError, get_llm_client
from app.tools import db_explorer, db_graph

PROFILE_ENV = "DB_PROFILE_PATH"
_DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "config" / "db_profile.json"

_CMP_OPS = {"==", "!=", ">", ">=", "<", "<="}
_SET_OPS = {"in", "not_in"}
_NULL_OPS = {"is_null", "not_null"}
DEFAULT_CATEGORY = "其他"
_LLM_CATEGORY_BATCH = 60

_discovery_cache: "weakref.WeakKeyDictionary[sa.Engine, List[Dict[str, Any]]]" = weakref.WeakKeyDictionary()


class ProfileNotConfigured(RuntimeError):
    """No project profile file exists (the multi-table feature is off)."""


# ---- profile --------------------------------------------------------------------------------------
def profile_path() -> Path:
    return Path(os.environ.get(PROFILE_ENV) or _DEFAULT_PROFILE)


def is_configured() -> bool:
    return profile_path().is_file()


def _validate_cond(cond: Any, where: str) -> None:
    if not isinstance(cond, dict):
        raise ValueError(f"{where}: 条件必须是对象")
    for group in ("all", "any"):
        if group in cond:
            items = cond[group]
            if not isinstance(items, list) or not items:
                raise ValueError(f"{where}.{group}: 必须是非空数组")
            for i, item in enumerate(items):
                _validate_cond(item, f"{where}.{group}[{i}]")
            return
    if "col" not in cond or "op" not in cond:
        raise ValueError(f"{where}: 叶子条件需要 col 与 op")
    op = cond["op"]
    if op not in _CMP_OPS | _SET_OPS | _NULL_OPS:
        raise ValueError(f"{where}: 不支持的操作符 {op!r}")
    if op not in _NULL_OPS and "value" not in cond:
        raise ValueError(f"{where}: 操作符 {op} 需要 value")
    if op in _SET_OPS and not isinstance(cond["value"], list):
        raise ValueError(f"{where}: 操作符 {op} 的 value 必须是数组")


def load_profile(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else profile_path()
    if not p.is_file():
        raise ProfileNotConfigured(f"项目配置文件不存在: {p}")
    try:
        profile = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"项目配置文件不是合法 JSON ({p.name}): {e}") from e
    sample = profile.get("sample") or {}
    if not sample.get("table") or not sample.get("key"):
        raise ValueError("项目配置缺少 sample.table / sample.key")
    label = profile.get("label") or {}
    for side in ("bad", "good"):
        if side not in label:
            raise ValueError(f"项目配置缺少 label.{side}")
        _validate_cond(label[side], f"label.{side}")
    if profile.get("population") is not None:
        _validate_cond(profile["population"], "population")
    for i, j in enumerate(profile.get("joins") or []):
        if not (j.get("table") or j.get("table_pattern")) or not j.get("left_key") or not j.get("right_key"):
            raise ValueError(f"joins[{i}] 需要 table 或 table_pattern，以及 left_key / right_key")
    for i, rule in enumerate((profile.get("discovery") or {}).get("table_categories") or []):
        if not rule.get("name"):
            raise ValueError(f"discovery.table_categories[{i}] 需要 name")
    return profile


# ---- conditions -----------------------------------------------------------------------------------
def _val(value: Any, params: Dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_val(v, params) for v in value]
    if isinstance(value, str) and value.startswith("$"):
        name = value[1:]
        if name not in params:
            raise ValueError(f"条件引用了未定义的参数 ${name}（请在 params 中配置）")
        return params[name]
    return value


def cond_columns(cond: Dict[str, Any]) -> Set[str]:
    for group in ("all", "any"):
        if group in cond:
            return set().union(*(cond_columns(c) for c in cond[group]))
    return {cond["col"]}


def describe_cond(cond: Dict[str, Any], params: Dict[str, Any]) -> str:
    for group, joiner in (("all", " 且 "), ("any", " 或 ")):
        if group in cond:
            parts = [describe_cond(c, params) for c in cond[group]]
            return parts[0] if len(parts) == 1 else "(" + joiner.join(parts) + ")"
    op = cond["op"]
    if op in _NULL_OPS:
        return f"{cond['col']} {'为空' if op == 'is_null' else '非空'}"
    return f"{cond['col']} {op} {_val(cond['value'], params)}"


def cond_to_sql(cond: Dict[str, Any], table: sa.TableClause, params: Dict[str, Any]) -> Any:
    for group, fn in (("all", sa.and_), ("any", sa.or_)):
        if group in cond:
            return fn(*[cond_to_sql(c, table, params) for c in cond[group]])
    col, op = table.c[cond["col"]], cond["op"]
    if op == "is_null":
        return col.is_(None)
    if op == "not_null":
        return col.is_not(None)
    v = _val(cond["value"], params)
    if op == "in":
        return col.in_(v)
    if op == "not_in":
        return sa.and_(col.is_not(None), col.not_in(v))
    return {"==": col == v, "!=": col != v, ">": col > v, ">=": col >= v, "<": col < v, "<=": col <= v}[op]


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def cond_to_mask(cond: Dict[str, Any], df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    for group, fn in (("all", np.logical_and), ("any", np.logical_or)):
        if group in cond:
            masks = [cond_to_mask(c, df, params) for c in cond[group]]
            out = masks[0]
            for m in masks[1:]:
                out = fn(out, m)
            return pd.Series(out, index=df.index)
    s, op = df[cond["col"]], cond["op"]
    if op == "is_null":
        return s.isna()
    if op == "not_null":
        return s.notna()
    v = _val(cond["value"], params)
    probe = v if isinstance(v, list) else [v]
    if all(_is_num(x) for x in probe):
        s = pd.to_numeric(s, errors="coerce")
    else:
        s = s.astype("string")
    present = s.notna()
    if op == "in":
        return s.isin(v) & present
    if op == "not_in":
        return ~s.isin(v) & present
    cmp = {"==": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]
    return cmp.fillna(False).astype(bool) & present


# ---- exclusion / joins ----------------------------------------------------------------------------
def _excluder(profile: Dict[str, Any], base: str, keys: Set[str]):
    rules = [(r.get("reason", "配置排除"), [p.lower() for p in r.get("patterns", [])]) for r in profile.get("exclude") or []]

    def reason_for(table: str, column: str) -> Optional[str]:
        if table == base and column in keys:
            return "主键 / 关联键"
        names = (column.lower(), f"{table}.{column}".lower())
        for reason, patterns in rules:
            if any(fnmatch.fnmatchcase(n, p) for n in names for p in patterns):
                return reason
        return None

    return reason_for


def _resolve_joins(
    profile: Dict[str, Any], base: str, base_names: Set[str], allowed: Sequence[str], warnings: List[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cfg in profile.get("joins") or []:
        if cfg["left_key"] not in base_names:
            raise ValueError(f"joins: 基础表 {base} 没有关联列 {cfg['left_key']}")
        pattern = cfg.get("table_pattern") or cfg["table"]
        matched = [t for t in allowed if fnmatch.fnmatchcase(t, pattern) and t != base]
        if not matched:
            warnings.append(f"joins: 没有表匹配 `{pattern}`（或不在 DB_ALLOWED_TABLES 内），已跳过。")
        for table in matched:
            cols = db_explorer.describe_table(table)
            if cfg["right_key"] not in {c["name"] for c in cols}:
                warnings.append(f"joins: 表 {table} 没有关联列 {cfg['right_key']}，已跳过。")
                continue
            out.append({"table": table, "left_key": cfg["left_key"], "right_key": cfg["right_key"], "columns": cols})
    return out


# ---- frame builder --------------------------------------------------------------------------------
def build_frame(
    profile: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    max_rows: Optional[int] = None,
    edges: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Reads the population (base + joined tables), applies the label rules and drops excluded columns.

    `edges` are auto-inferred links (`db_graph.plan_edges`): `snapshot` edges are LEFT JOINed, `detail` edges are
    aggregated point-in-time (events before the sample's date_col) inside the database.
    """
    params = {**(profile.get("params") or {}), **{k: v for k, v in (overrides or {}).items() if v is not None}}
    sample, label_cfg = profile["sample"], profile["label"]
    base, date_col = sample["table"], sample.get("date_col")
    label_name = label_cfg.get("name", "risk_label")
    warnings: List[str] = []

    allowed = db_explorer._allowed_tables()
    if base not in allowed:
        raise KeyError(f"table '{base}' not found or not allowed")
    base_cols = db_explorer.describe_table(base)
    base_names = {c["name"] for c in base_cols}

    population = profile.get("population")
    conds = [c for c in (population, label_cfg["bad"], label_cfg["good"]) if c]
    needed = (
        set().union(*(cond_columns(c) for c in conds))
        | {sample["key"]}
        | ({date_col} if date_col else set())
        | {chk["col"] for chk in profile.get("performance_checks") or []}
    )
    missing = sorted(needed - base_names)
    if missing:
        raise ValueError(f"基础表 {base} 不存在配置引用的列: {missing}")
    if label_name in base_names:
        raise ValueError(f"label.name `{label_name}` 与基础表列重名，请换一个名字")

    joins = _resolve_joins(profile, base, base_names, allowed, warnings)
    snapshot_edges = [e for e in edges or [] if e["kind"] == "snapshot"]
    detail_edges = [e for e in edges or [] if e["kind"] == "detail"]
    for e in snapshot_edges:
        joins.append(
            {"table": e["table"], "left_key": e["sample_col"], "right_key": e["link_col"], "columns": e["columns"], "category": e.get("category")}
        )
    if snapshot_edges:
        warnings.append(
            "自动关联的一对一表为当前快照，无法保证字段值早于申请时点，请业务复核：" + "、".join(e["table"] for e in snapshot_edges)
        )
    keys = {sample["key"]} | ({sample["user_key"]} if sample.get("user_key") else set()) | {j["left_key"] for j in joins}
    reason_for = _excluder(profile, base, keys)

    excluded: Dict[str, str] = {}
    drop_after: Set[str] = set()
    col_meta: Dict[str, Dict[str, Any]] = {}
    bt = sa.table(base, *[sa.column(c["name"]) for c in base_cols])
    selects: List[Any] = []
    for c in base_cols:
        name = c["name"]
        col_meta[name] = {"table": base, "column": name, "comment": c.get("comment")}
        reason = reason_for(base, name)
        if reason:
            excluded[name] = reason
            if name not in needed:
                continue
            drop_after.add(name)
        selects.append(bt.c[name])

    source = bt
    match_cols: List[str] = []
    for i, j in enumerate(joins):
        jt = sa.table(j["table"], *[sa.column(c["name"]) for c in j["columns"]]).alias(f"j{i}")
        for c in j["columns"]:
            display = f"{j['table']}.{c['name']}"
            col_meta[display] = {"table": j["table"], "column": c["name"], "comment": c.get("comment"), "category": j.get("category")}
            if c["name"] == j["right_key"]:
                excluded[display] = "主键 / 关联键"
                continue
            reason = reason_for(j["table"], c["name"])
            if reason:
                excluded[display] = reason
                continue
            selects.append(jt.c[c["name"]].label(display))
        match_col = f"__match_{i}"
        selects.append(jt.c[j["right_key"]].label(match_col))
        match_cols.append(match_col)
        source = source.outerjoin(jt, bt.c[j["left_key"]] == jt.c[j["right_key"]])

    cap = min(max_rows or db_explorer._max_rows(), db_explorer._max_rows())
    stmt = sa.select(*selects).select_from(source)
    if population:
        stmt = stmt.where(cond_to_sql(population, bt, params))
    stmt = stmt.order_by(bt.c[sample["key"]]).limit(cap + 1)
    with db_explorer.get_engine().connect() as conn:
        df = pd.read_sql(stmt, conn)
    truncated = len(df) > cap
    df = df.iloc[:cap].copy()
    if truncated:
        warnings.append(f"总体行数超过取样上限 {cap}，只使用前 {cap} 行（按主键升序）；如需更大样本请调大 DB_MAX_ROWS。")
    if df[sample["key"]].duplicated().any():
        raise ValueError(f"关联后 {sample['key']} 出现重复行：关联表的 right_key 在表内不唯一，请改用聚合后的表或唯一键")

    for e in detail_edges:
        key_max = df[sample["key"]].max() if truncated else None
        agg = db_graph.aggregate_detail(
            profile, e, bt, cond_to_sql(population, bt, params) if population else None, reason_for, key_max
        )
        df = df.merge(agg["df"], how="left", left_on=sample["key"], right_on="__k").drop(columns="__k")
        col_meta.update(agg["col_meta"])
        excluded.update(agg["excluded"])
        covered = df[f"{e['table']}.cnt_all"].fillna(0) > 0
        e["coverage"] = round(float(covered.mean()), 4) if len(df) else 0.0
        e["features"] = len(agg["col_meta"])
        if e["coverage"] < 0.05:
            warnings.append(f"明细表 {e['table']} 在申请时点之前的覆盖率仅 {e['coverage']:.0%}，聚合特征大多为空。")

    join_info = []
    for j, mc in zip(joins, match_cols):
        rate = round(float(df[mc].notna().mean()), 4) if len(df) else 0.0
        join_info.append({"table": j["table"], "on": f"{base}.{j['left_key']} = {j['table']}.{j['right_key']}", "match_rate": rate})
        if rate < 0.5:
            warnings.append(f"关联表 {j['table']} 的匹配率仅 {rate:.0%}，请确认关联键是否正确。")
    df = df.drop(columns=match_cols)

    population_rows = len(df)
    bad, good = cond_to_mask(label_cfg["bad"], df, params), cond_to_mask(label_cfg["good"], df, params)
    y = pd.Series(np.nan, index=df.index, dtype=float)
    y[good] = 0.0
    y[bad] = 1.0  # bad wins when both rules match
    label_info: Dict[str, Any] = {
        "bad_count": int(bad.sum()),
        "good_count": int((good & ~bad).sum()),
        "dropped_unlabeled": int(y.isna().sum()),
        "unmapped_values": [],
        "population_rows": population_rows,
    }
    if label_info["bad_count"] == 0 or label_info["good_count"] == 0:
        raise ValueError(
            f"标签规则只得到单一类别（{label_info['bad_count']} 坏 / {label_info['good_count']} 好）；"
            "请检查 population / label 配置或调整 params 中的阈值"
        )

    checks = []
    for chk in profile.get("performance_checks") or []:
        col = chk["col"]
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        good_mean, bad_mean = s[y == 0].mean(), s[y == 1].mean()
        higher_is_good = chk.get("higher_is", "good") == "good"
        ok = bool(good_mean > bad_mean) if higher_is_good else bool(good_mean < bad_mean)
        checks.append(
            {
                "col": col,
                "higher_is": "good" if higher_is_good else "bad",
                "good_mean": None if pd.isna(good_mean) else round(float(good_mean), 4),
                "bad_mean": None if pd.isna(bad_mean) else round(float(bad_mean), 4),
                "ok": ok,
            }
        )
        if not ok:
            warnings.append(
                f"标签方向校验未通过：{col} 在好 / 坏样本中的均值为 {checks[-1]['good_mean']} / {checks[-1]['bad_mean']}，"
                f"与配置的方向（越大越{'好' if higher_is_good else '坏'}）不一致，请复核标签规则。"
            )
    label_info["direction_checks"] = checks
    gray_rule = "既不命中坏样本也不命中好样本的订单（如在途未结清且逾期未达阈值、或逾期后结清但未达坏样本阈值），自动丢弃不污染好坏样本"
    label_info["gray_rule"] = gray_rule
    label_info["rule"] = (
        f"坏样本：{describe_cond(label_cfg['bad'], params)}；"
        f"好样本：{describe_cond(label_cfg['good'], params)}；"
        f"灰样本（自动丢弃）：{gray_rule}"
    )
    label_info["bad_rule"] = describe_cond(label_cfg["bad"], params)
    label_info["good_rule"] = describe_cond(label_cfg["good"], params)

    df = df.drop(columns=sorted(drop_after - ({date_col} if date_col else set())))
    df, label_info = db_explorer.clean_labeled(df, y, label_info, label_name, date_col, truncated)
    if label_info.get("dropped_bad_date"):
        warnings.append(f"日期列 {date_col} 为空或无法解析的 {label_info['dropped_bad_date']} 行已丢弃（这些行本可打上标签）。")
    warnings.extend(profile.get("warnings") or [])
    return {
        "df": df,
        "label_info": label_info,
        "warnings": warnings,
        "label_name": label_name,
        "date_col": date_col,
        "col_meta": col_meta,
        "excluded": excluded,
        "joins": join_info,
        "auto_edges": [{k: v for k, v in e.items() if k != "columns"} for e in edges or []],
        "params": params,
    }


# ---- categorisation -------------------------------------------------------------------------------
def _rule_category(meta: Dict[str, Any], rules: Sequence[Dict[str, Any]]) -> Optional[str]:
    name, comment = meta["column"].lower(), meta.get("comment") or ""
    for rule in rules:
        if any(fnmatch.fnmatchcase(name, p.lower()) for p in rule.get("name_patterns", [])):
            return rule["name"]
        if any(k in comment for k in rule.get("comment_keywords", [])):
            return rule["name"]
    return None


def _llm_categories(items: List[Tuple[str, Dict[str, Any]]], allowed: List[str]) -> Dict[str, str]:
    client = get_llm_client()
    if not client.available or not items:
        return {}
    out: Dict[str, str] = {}
    for start in range(0, len(items), _LLM_CATEGORY_BATCH):
        batch = items[start : start + _LLM_CATEGORY_BATCH]
        listing = "\n".join(f"{name}\t{meta['table']}\t{(meta.get('comment') or '')[:60]}" for name, meta in batch)
        try:
            data = client.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是信贷风控特征分类专家。把每个字段归入给定类别之一，只能使用这些类别名："
                            f"{allowed}；拿不准就用“{DEFAULT_CATEGORY}”。只输出 JSON 对象 {{字段名: 类别名}}。"
                        ),
                    },
                    {"role": "user", "content": "字段名\t来源表\t注释\n" + listing},
                ],
                max_tokens=1200,
            )
        except LLMError:
            continue
        for name, cat in (data or {}).items():
            if cat in allowed:
                out[name] = cat
    return out


def categorize(profile: Dict[str, Any], features: Sequence[str], col_meta: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Feature -> category: ordered name / comment rules, then optional LLM for the rest, then the per-table fallback."""
    cfg = profile.get("categories") or {}
    rules = cfg.get("rules", [])
    fallback = cfg.get("table_fallback", {})
    result: Dict[str, str] = {}
    pending: List[Tuple[str, Dict[str, Any]]] = []
    for f in features:
        meta = col_meta.get(f) or {"table": "", "column": f, "comment": None}
        cat = meta.get("category") or _rule_category(meta, rules)  # auto-linked tables carry their table category
        if cat:
            result[f] = cat
        else:
            pending.append((f, meta))
    allowed = [r["name"] for r in rules] + list(dict.fromkeys(fallback.values())) + [DEFAULT_CATEGORY]
    if cfg.get("use_llm", True):
        for f, cat in _llm_categories(pending, list(dict.fromkeys(allowed))).items():
            result[f] = cat
    for f, meta in pending:
        result.setdefault(f, fallback.get(meta["table"], DEFAULT_CATEGORY))
    return result


# ---- mining ---------------------------------------------------------------------------------------
def mine_project(profile: Dict[str, Any], frame: Dict[str, Any], oot_start_date: Optional[str] = None) -> Dict[str, Any]:
    """Indicator mining on the built frame, plus profile exclusions and feature categories."""
    label = {
        **frame["label_info"],
        "column_comments": {k: m.get("comment") for k, m in frame["col_meta"].items()},
    }
    mined = db_explorer.mine_frame(
        profile["sample"]["table"],
        frame["df"],
        label,
        frame["warnings"],
        frame["label_name"],
        [label["bad_rule"]],
        frame["date_col"],
        oot_start_date,
    )
    mined["not_suggested"] = {**frame["excluded"], **mined["not_suggested"]}
    metrics = mined["feature_metrics"]
    cats = categorize(profile, [m["feature"] for m in metrics], frame["col_meta"])
    suggested = set(mined["suggested_features"])
    summary: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        meta = frame["col_meta"].get(m["feature"], {})
        m["category"] = cats[m["feature"]]
        m["source_table"] = meta.get("table")
        m["comment"] = meta.get("comment")
        s = summary.setdefault(m["category"], {"category": m["category"], "mined": 0, "suggested": 0, "iv_sum": 0.0, "top": []})
        s["mined"] += 1
        s["iv_sum"] += m["iv"]
        if m["feature"] in suggested:
            s["suggested"] += 1
            if len(s["top"]) < 5:
                s["top"].append(m["feature"])  # metrics are IV-sorted
    mined["category_summary"] = sorted(
        (
            {"category": s["category"], "mined": s["mined"], "suggested": s["suggested"],
             "avg_iv": round(s["iv_sum"] / s["mined"], 4), "top_features": s["top"]}
            for s in summary.values()
        ),
        key=lambda s: (-s["suggested"], -s["avg_iv"]),
    )
    mined["joins"] = frame["joins"]
    mined["auto_edges"] = frame.get("auto_edges", [])
    mined["auto_skipped"] = frame.get("auto_skipped", [])
    return mined


# ---- discovery ------------------------------------------------------------------------------------
DEFAULT_TABLE_CATEGORY = "其他业务表"
POST_EVENT_CATEGORY = "贷后还款、催收与表现（仅用于标签）"
_COLUMNS_PREVIEW = 8


def _table_rule_category(item: Dict[str, Any], rules: Sequence[Dict[str, Any]]) -> Optional[str]:
    name, comment = item["table"].lower(), item.get("comment") or ""
    for rule in rules:
        if any(fnmatch.fnmatchcase(name, p.lower()) for p in rule.get("table_patterns", [])):
            return rule["name"]
        if any(k in comment for k in rule.get("comment_keywords", [])):
            return rule["name"]
    return None


def _llm_table_categories(items: List[Dict[str, Any]], allowed: List[str]) -> Dict[str, str]:
    client = get_llm_client()
    if not client.available or not items:
        return {}
    out: Dict[str, str] = {}
    for start in range(0, len(items), _LLM_CATEGORY_BATCH):
        batch = items[start : start + _LLM_CATEGORY_BATCH]
        listing = "\n".join(
            f"{i['table']}\t{(i.get('comment') or '')[:60]}\t{','.join(i['columns_preview'])}" for i in batch
        )
        try:
            data = client.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是信贷风控数据专家。把每张数据库表归入给定类别之一，只能使用这些类别名："
                            f"{allowed}；拿不准就用“{DEFAULT_TABLE_CATEGORY}”。只输出 JSON 对象 {{表名: 类别名}}。"
                        ),
                    },
                    {"role": "user", "content": "表名\t表注释\t前几个列名\n" + listing},
                ],
                max_tokens=1200,
            )
        except LLMError:
            continue
        for name, cat in (data or {}).items():
            if cat in allowed:
                out[name] = cat
    return out


def categorize_tables(profile: Dict[str, Any], candidates: List[Dict[str, Any]]) -> None:
    """Sets `category` on each candidate table: post-event tables first, then name / comment rules, then the LLM."""
    cfg = profile.get("discovery") or {}
    rules = cfg.get("table_categories") or []
    post_category = cfg.get("post_event_category", POST_EVENT_CATEGORY)
    pending: List[Dict[str, Any]] = []
    for item in candidates:
        if item["post_event"]:
            item["category"] = post_category
            continue
        cat = _table_rule_category(item, rules)
        if cat:
            item["category"] = cat
        else:
            pending.append(item)
    if cfg.get("use_llm", True) and pending:
        allowed = list(dict.fromkeys([r["name"] for r in rules] + [DEFAULT_TABLE_CATEGORY]))
        picked = _llm_table_categories(pending, allowed)
        for item in pending:
            if item["table"] in picked:
                item["category"] = picked[item["table"]]
    for item in pending:
        item.setdefault("category", DEFAULT_TABLE_CATEGORY)


def discover_candidate_tables(profile: Dict[str, Any], refresh: bool = False) -> List[Dict[str, Any]]:
    """Tables that carry one of the profile's link columns (user / order keys): candidate sources of extra features.

    Metadata only (no row reads). Results are cached per engine because reflecting hundreds of tables is slow.
    """
    cfg = profile.get("discovery") or {}
    link = {c.lower() for c in cfg.get("link_columns", [])}
    skip = [p.lower() for p in cfg.get("exclude_table_patterns", [])]
    post_event = [p.lower() for p in cfg.get("post_event_table_patterns", [])]
    if not link:
        return []
    engine = db_explorer.get_engine()
    if refresh or engine not in _discovery_cache:
        tables = db_explorer._allowed_tables()
        columns, comments = db_explorer.table_columns(tables), db_explorer.table_comments(tables)
        _discovery_cache[engine] = [{"table": t, "columns": columns[t], "comment": comments.get(t)} for t in tables]
    used = {profile["sample"]["table"], *(j["table"] for j in (profile.get("joins") or []) if j.get("table"))}
    out = []
    for item in _discovery_cache[engine]:
        name = item["table"]
        if name in used or any(fnmatch.fnmatchcase(name.lower(), p) for p in skip):
            continue
        hits = [c for c in item["columns"] if c.lower() in link]
        if hits:
            out.append(
                {
                    "table": name,
                    "comment": item["comment"],
                    "link_columns": hits,
                    "n_columns": len(item["columns"]),
                    "columns_preview": item["columns"][:_COLUMNS_PREVIEW],
                    "post_event": any(fnmatch.fnmatchcase(name.lower(), p) for p in post_event),
                }
            )
    out.sort(key=lambda r: (r["post_event"], -len(r["link_columns"]), r["table"]))
    out = out[: int(cfg.get("max_tables", 50))]
    categorize_tables(profile, out)
    return out


def summarize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    params = profile.get("params") or {}
    label = profile["label"]
    return {
        "configured": True,
        "name": profile.get("name"),
        "description": profile.get("description"),
        "path": str(profile_path()),
        "base_table": profile["sample"]["table"],
        "joins": [j.get("table") or j.get("table_pattern") for j in profile.get("joins") or []],
        "population": describe_cond(profile["population"], params) if profile.get("population") else None,
        "label_rule": (
            f"坏样本：{describe_cond(label['bad'], params)}；"
            f"好样本：{describe_cond(label['good'], params)}；"
            "灰样本（自动丢弃）：既不命中坏样本也不命中好样本的中间状态订单（如刚放款还在账期内、表现期未满），自动丢弃不污染好坏样本"
        ),
        "params": params,
    }
