"""Automatic multi-table graph discovery & point-in-time domain slice aggregation.

Starting from N master tables (by default `sample.table` such as `order_apply` plus joined master hubs such as
`user_info`, or explicit `master_tables`), this module:

1. infers 1-hop and 2-hop join edges across candidate tables (`plan_edges`), using column-name mapping, database
   cardinality & non-empty probes, bridge-key propagation through 1:1 hub/snapshot tables, and optional LLM semantic
   disambiguation guarded by a strict column whitelist;
2. classifies each edge as `snapshot` (1:1 -> LEFT JOIN) or `detail` (1:N -> point-in-time SQL aggregation);
3. aggregates `detail` tables strictly before `sample.date_col` over multiple lookback windows (`7d/30d/90d/all`),
   including automatic **Domain Conditional Slices** (`L1`, e.g. cashflow `in/out`, credit `overdue/normal`, behavior
   `success/fail`) when low-cardinality status/direction columns are present.

All SQL statements are built with SQLAlchemy Core over metadata-validated identifiers and bound parameters under the
read-only session guard.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import sqlalchemy as sa

from app.core.llm import LLMError, get_llm_client
from app.tools import db_explorer

DEFAULT_WINDOWS = [7, 30, 90, 0]
DEFAULT_AGGS = ["sum", "avg", "max", "min"]
DEFAULT_TIME_PATTERNS = ["create_time", "created_at", "add_time", "apply_time", "txn_time", "trans_time", "*_time", "*_date", "*_at"]
_AGG_FUNCS = {"sum": sa.func.sum, "avg": sa.func.avg, "max": sa.func.max, "min": sa.func.min}
_NUMERIC_TOKENS = ("INT", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE", "REAL")
_DATETIME_TOKENS = ("DATETIME", "TIMESTAMP", "DATE")
_ID_NAME = re.compile(r"(^|_)(id|no|code|uuid)$", re.IGNORECASE)
_BRIDGE_KEY_RE = re.compile(r"^[a-z0-9]+_(id|no|code)$", re.IGNORECASE)
_SLICE_COL_RE = re.compile(
    r"(^|_)(direction|flow_type|txn_type|trans_type|io_type|status|loan_status|repay_status|account_status|result|event_type|channel_type)$",
    re.IGNORECASE,
)
_SLUG_CLEAN = re.compile(r"[^a-zA-Z0-9]+")


def enabled(profile: Dict[str, Any]) -> bool:
    return bool((profile.get("auto_join") or {}).get("enabled"))


def _is_numeric(col: Dict[str, Any]) -> bool:
    t = str(col["type"]).upper()
    return not col["is_text"] and any(tok in t for tok in _NUMERIC_TOKENS) and "DATE" not in t and "TIME" not in t


def _is_datetime(col: Dict[str, Any]) -> bool:
    return any(tok in str(col["type"]).upper() for tok in _DATETIME_TOKENS)


def pick_time_col(columns: Sequence[Dict[str, Any]], patterns: Sequence[str]) -> Optional[str]:
    """Event-time column: datetime-typed columns first, text columns only when the name matches a pattern."""
    best: Optional[tuple] = None
    for c in columns:
        name = c["name"].lower()
        rank = next((i for i, p in enumerate(patterns) if fnmatch.fnmatchcase(name, p.lower())), None)
        if _is_datetime(c):
            key = (0, rank if rank is not None else len(patterns))
        elif c["is_text"] and rank is not None:
            key = (1, rank)
        else:
            continue
        if best is None or key < best[0]:
            best = (key, c["name"])
    return best[1] if best else None


def _cardinality(table: str, column: str) -> Dict[str, Any]:
    t = sa.table(table, sa.column(column))
    stmt = sa.select(sa.func.count(t.c[column]).label("n"), sa.func.count(sa.distinct(t.c[column])).label("d")).select_from(t)
    with db_explorer.get_engine().connect() as conn:
        row = conn.execute(stmt).mappings().one()
    n, d = int(row["n"]), int(row["d"])
    return {"rows": n, "distinct_keys": d, "unique": n > 0 and n == d}


def _llm_infer_edge_cols(
    table: str,
    columns: Sequence[Dict[str, Any]],
    sample_cols: Sequence[str],
    need_link: bool,
    need_time: bool,
) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    """Optional LLM disambiguation for non-standard link/time column names, strictly whitelisted against metadata."""
    client = get_llm_client()
    if not client.available or (not need_link and not need_time):
        return None, None
    col_names = {c["name"] for c in columns}
    sample_set = set(sample_cols)
    col_desc = ", ".join(f"{c['name']}({c['type']})" for c in columns[:30])
    try:
        data = client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是数据库图谱关联专家。给定明细表字段与主表可用关联键，推断明细表的关联键(link_col)、"
                        "对应的主表键(sample_col)以及事件时间列(time_col)。只能使用给定字段名，不确定填 null。"
                        "只输出 JSON: {\"link_col\": \"...\", \"sample_col\": \"...\", \"time_col\": \"...\"}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"明细表 {table}: {col_desc}\n主表关联键: {list(sample_set)}",
                },
            ],
            max_tokens=200,
        )
    except LLMError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    lc, sc, tc = data.get("link_col"), data.get("sample_col"), data.get("time_col")
    link = (lc, sc) if (need_link and isinstance(lc, str) and isinstance(sc, str) and lc in col_names and sc in sample_set) else None
    time_col = tc if (need_time and isinstance(tc, str) and tc in col_names) else None
    return link, time_col


def extract_master_tables(profile: Dict[str, Any]) -> List[str]:
    """Returns the N master hub tables configured for the project (sample table + joined master tables + explicit hubs)."""
    base = profile["sample"]["table"]
    hubs = [base]
    for m in profile.get("master_tables") or []:
        t = m["table"] if isinstance(m, dict) else str(m)
        if t not in hubs:
            hubs.append(t)
    for j in profile.get("joins") or []:
        t = j.get("table") or (j.get("table_pattern") or "").rstrip("*")
        if t and t not in hubs:
            hubs.append(t)
    return hubs


def plan_edges(profile: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Infers 1-hop and 2-hop edges for usable (non post-event) candidate tables.

    Returns `{"edges": [...], "skipped": [...], "master_tables": [...]}`.
    """
    cfg = profile.get("auto_join") or {}
    if not cfg.get("enabled"):
        return {"edges": [], "skipped": []}
    base = profile["sample"]["table"]
    sample_cols = {c["name"] for c in db_explorer.describe_table(base)}
    link_map = dict(cfg.get("link_map") or {})
    patterns = cfg.get("time_col_patterns") or DEFAULT_TIME_PATTERNS
    skip_cats = set(cfg.get("skip_categories") or [])
    max_tables = int(cfg.get("max_tables", 20))
    use_llm = bool((profile.get("discovery") or {}).get("use_llm", False))

    edges: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    deferred_for_2hop: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []

    # Pass 1: 1-hop edges directly linking to sample_cols (or via link_map)
    for cand in candidates:
        if cand["post_event"]:
            continue
        table = cand["table"]
        if cand.get("category") in skip_cats:
            skipped.append({"table": table, "reason": f"类别 {cand['category']} 被配置跳过"})
            continue
        columns = db_explorer.describe_table(table)
        link = next(((c, link_map.get(c, c)) for c in cand["link_columns"] if link_map.get(c, c) in sample_cols), None)
        time_col = pick_time_col(columns, patterns)
        if (link is None or time_col is None) and use_llm:
            llm_link, llm_time = _llm_infer_edge_cols(table, columns, list(sample_cols), link is None, time_col is None)
            link = link or llm_link
            time_col = time_col or llm_time
        if link is None:
            deferred_for_2hop.append((cand, columns))
            continue
        if len(edges) >= max_tables:
            skipped.append({"table": table, "reason": f"超过 auto_join.max_tables={max_tables}"})
            continue
        card = _cardinality(table, link[0])
        edge: Dict[str, Any] = {
            "table": table,
            "comment": cand.get("comment"),
            "category": cand.get("category"),
            "link_col": link[0],
            "sample_col": link[1],
            "hop": 1,
            "path": f"{base} -> {table}",
            "path_tables": [base, table],
            "columns": columns,
            "time_col": time_col,
            **card,
        }
        edge["kind"] = "snapshot" if card["unique"] else "detail"
        if edge["kind"] == "detail" and not edge["time_col"]:
            skipped.append({"table": table, "reason": "1:N 明细表没有可用的事件时间列，无法保证申请时点之前，已跳过"})
            continue
        if edge["kind"] == "detail" and not profile["sample"].get("date_col"):
            skipped.append({"table": table, "reason": "样本表未配置 date_col，无法做申请时点聚合，已跳过"})
            continue
        edges.append(edge)

    # Pass 2: 2-hop edges bridged through 1-hop snapshot tables (e.g. order_apply -> credit_report -> credit_loan_detail)
    bridge_map: Dict[str, Dict[str, Any]] = {}
    for e in edges:
        if e["kind"] != "snapshot":
            continue
        for c in e["columns"]:
            cname = c["name"]
            if cname != e["link_col"] and _BRIDGE_KEY_RE.match(cname) and cname.lower() not in bridge_map:
                bridge_map[cname.lower()] = {
                    "via_table": e["table"],
                    "via_link_col": e["link_col"],
                    "via_bridge_col": cname,
                    "sample_col": e["sample_col"],
                }

    for cand, columns in deferred_for_2hop:
        table = cand["table"]
        bridge = next(
            ((c["name"], bridge_map[c["name"].lower()]) for c in columns if c["name"].lower() in bridge_map),
            None,
        )
        if bridge is None:
            skipped.append({"table": table, "reason": "关联列在样本表及桥接主表中没有对应列"})
            continue
        if len(edges) >= max_tables:
            skipped.append({"table": table, "reason": f"超过 auto_join.max_tables={max_tables}"})
            continue
        cand_col, binfo = bridge
        time_col = pick_time_col(columns, patterns)
        card = _cardinality(table, cand_col)
        if not time_col:
            skipped.append({"table": table, "reason": "2跳明细表没有可用的事件时间列，无法保证申请时点之前，已跳过"})
            continue
        if not profile["sample"].get("date_col"):
            skipped.append({"table": table, "reason": "样本表未配置 date_col，无法做申请时点聚合，已跳过"})
            continue
        edges.append(
            {
                "table": table,
                "comment": cand.get("comment"),
                "category": cand.get("category"),
                "link_col": cand_col,
                "sample_col": binfo["sample_col"],
                "hop": 2,
                "via_table": binfo["via_table"],
                "via_link_col": binfo["via_link_col"],
                "via_bridge_col": binfo["via_bridge_col"],
                "path": f"{base} -> {binfo['via_table']} -> {table}",
                "path_tables": [base, binfo["via_table"], table],
                "columns": columns,
                "time_col": time_col,
                "kind": "detail",
                **card,
            }
        )

    return {"edges": edges, "skipped": skipped}


def _shift(col: Any, days: int) -> Any:
    dialect = db_explorer.get_engine().dialect.name
    if dialect == "mysql":
        return sa.func.date_sub(col, sa.text(f"INTERVAL {int(days)} DAY"))
    if dialect == "sqlite":
        return sa.func.datetime(col, f"-{int(days)} days")
    raise ValueError(f"自动聚合暂不支持数据库方言 {dialect}")


def _slug(val: Any) -> str:
    s = _SLUG_CLEAN.sub("_", str(val).strip().lower()).strip("_")
    return s[:16] if s else "v"


def _discover_slices(
    table: str,
    columns: Sequence[Dict[str, Any]],
    skip_cols: set,
    reason_for: Callable[[str, str], Optional[str]],
    max_slice_cols: int = 2,
    max_slice_values: int = 4,
) -> List[Tuple[str, List[Tuple[Any, str]]]]:
    """Finds low-cardinality status/direction/type columns in `table` for domain conditional slice aggregation."""
    candidates = [
        c for c in columns
        if c["name"] not in skip_cols
        and _SLICE_COL_RE.search(c["name"])
        and reason_for(table, c["name"]) is None
    ]
    out: List[Tuple[str, List[Tuple[Any, str]]]] = []
    for c in candidates[:max_slice_cols]:
        col_name = c["name"]
        t = sa.table(table, sa.column(col_name))
        col_expr = t.c[col_name]
        stmt = (
            sa.select(col_expr.label("v"), sa.func.count().label("n"))
            .select_from(t)
            .where(col_expr.is_not(None))
            .group_by(col_expr)
            .order_by(sa.desc("n"))
            .limit(max_slice_values + 1)
        )
        with db_explorer.get_engine().connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        if 2 <= len(rows) <= max_slice_values:
            vals: List[Tuple[Any, str]] = []
            seen_slugs: set = set()
            for r in rows[:3]:
                raw = r["v"]
                if raw is None or str(raw).strip() == "":
                    continue
                sl = _slug(raw)
                if sl not in seen_slugs:
                    seen_slugs.add(sl)
                    vals.append((raw, sl))
            if len(vals) >= 2:
                out.append((col_name, vals))
    return out


def aggregate_detail(
    profile: Dict[str, Any],
    edge: Dict[str, Any],
    base_table: sa.TableClause,
    where: Optional[Any],
    reason_for: Callable[[str, str], Optional[str]],
    key_max: Optional[Any],
) -> Dict[str, Any]:
    """Point-in-time aggregates (window stats + L1 domain slices) of a 1:N table per sample row."""
    cfg = profile.get("auto_join") or {}
    windows = cfg.get("windows_days") or DEFAULT_WINDOWS
    aggs = [a for a in (cfg.get("aggs") or DEFAULT_AGGS) if a in _AGG_FUNCS]
    max_cols = int(cfg.get("max_agg_cols", 10))
    sample = profile["sample"]
    table, link, time_col = edge["table"], edge["link_col"], edge["time_col"]
    path = edge.get("path") or f"{sample['table']} -> {table}"
    path_tables = edge.get("path_tables") or [sample["table"], table]

    dt = sa.table(table, *[sa.column(c["name"]) for c in edge["columns"]])
    bt = base_table
    event = dt.c[time_col]

    if edge.get("via_table"):
        bridge_t = sa.table(
            edge["via_table"],
            sa.column(edge["via_link_col"]),
            sa.column(edge["via_bridge_col"]),
        ).alias("__bridge")
        join_from = bt.outerjoin(bridge_t, bt.c[edge["sample_col"]] == bridge_t.c[edge["via_link_col"]]).outerjoin(
            dt, (dt.c[link] == bridge_t.c[edge["via_bridge_col"]]) & (event < bt.c[sample["date_col"]])
        )
    else:
        on = (dt.c[link] == bt.c[edge["sample_col"]]) & (event < bt.c[sample["date_col"]])
        join_from = bt.outerjoin(dt, on)

    excluded: Dict[str, str] = {}
    col_meta: Dict[str, Dict[str, Any]] = {}
    value_cols: List[Dict[str, Any]] = []
    for c in edge["columns"]:
        name = c["name"]
        if name in (link, time_col) or not _is_numeric(c) or _ID_NAME.search(name):
            continue
        reason = reason_for(table, name)
        if reason:
            excluded[f"{table}.{name}"] = reason
            col_meta[f"{table}.{name}"] = {
                "table": table,
                "column": name,
                "comment": c.get("comment"),
                "category": edge.get("category"),
                "graph_path": path,
                "source_tables": path_tables,
            }
        elif len(value_cols) < max_cols:
            value_cols.append(c)

    exprs: List[Any] = []

    def add(label: str, expr: Any, comment: str, layer: str = "window_agg", formula: Optional[str] = None) -> None:
        exprs.append(expr.label(label))
        col_meta[label] = {
            "table": table,
            "column": label,
            "comment": comment,
            "category": edge.get("category"),
            "layer": layer,
            "formula": formula or label,
            "graph_path": path,
            "source_tables": path_tables,
        }

    for w in windows:
        suffix = f"{w}d" if w else "all"
        in_window = (event >= _shift(bt.c[sample["date_col"]], w)) if w else None
        count_expr = (
            sa.func.sum(sa.case((in_window, 1), else_=0)) if w else sa.func.count(dt.c[link])
        )
        add(
            f"{table}.cnt_{suffix}",
            count_expr,
            f"申请前{w if w else '全部'}天内 {table} 记录数",
            layer="window_agg",
            formula=f"COUNT({table}) IN {suffix}",
        )
        for c in value_cols:
            value = sa.case((in_window, dt.c[c["name"]])) if w else dt.c[c["name"]]
            for agg in aggs:
                add(
                    f"{table}.{c['name']}__{agg}_{suffix}",
                    _AGG_FUNCS[agg](value),
                    f"{c.get('comment') or c['name']}（申请前{w if w else '全部'}天 {agg}）",
                    layer="window_agg",
                    formula=f"{agg.upper()}({table}.{c['name']}) IN {suffix}",
                )

    # L1: Multi-source domain conditional slices (e.g. cashflow in/out, credit overdue/normal, behavior success/fail)
    if cfg.get("enable_slices", True):
        slices = _discover_slices(table, edge["columns"], {link, time_col}, reason_for)
        slice_windows = [w for w in windows if w in (30, 0)] or windows[:2]
        for slice_col, val_pairs in slices:
            for raw_val, slug in val_pairs:
                match_slice = dt.c[slice_col] == raw_val
                for w in slice_windows:
                    suffix = f"{w}d" if w else "all"
                    cond = (event >= _shift(bt.c[sample["date_col"]], w)) & match_slice if w else match_slice
                    add(
                        f"{table}.cnt__slice_{slice_col}_{slug}_{suffix}",
                        sa.func.sum(sa.case((cond, 1), else_=0)),
                        f"申请前{w if w else '全部'}天 {table} {slice_col}={raw_val} 记录数",
                        layer="domain_slice",
                        formula=f"COUNT({table}) WHERE {slice_col}={raw_val!r} IN {suffix}",
                    )
                    for c in value_cols[:2]:
                        add(
                            f"{table}.{c['name']}__slice_{slice_col}_{slug}__sum_{suffix}",
                            sa.func.sum(sa.case((cond, dt.c[c["name"]]), else_=0)),
                            f"{c.get('comment') or c['name']}（{slice_col}={raw_val}，申请前{w if w else '全部'}天 sum）",
                            layer="domain_slice",
                            formula=f"SUM({table}.{c['name']}) WHERE {slice_col}={raw_val!r} IN {suffix}",
                        )

    stmt = sa.select(bt.c[sample["key"]].label("__k"), *exprs).select_from(join_from)
    if where is not None:
        stmt = stmt.where(where)
    if key_max is not None:
        stmt = stmt.where(bt.c[sample["key"]] <= key_max)
    stmt = stmt.group_by(bt.c[sample["key"]])
    with db_explorer.get_engine().connect() as conn:
        df = pd.read_sql(stmt, conn)
    return {"df": df, "col_meta": col_meta, "excluded": excluded}
