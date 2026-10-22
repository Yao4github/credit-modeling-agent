"""Automatic multi-table linking: infer join edges from column names and aggregate 1:N tables point-in-time.

Hundreds of tables cannot be wired by hand. Given the discovery candidates (tables that carry a link column such as
`user_id`), this module

1. maps each link column to a sample-table column (`auto_join.link_map`, default: same name) = an *edge*;
2. checks the edge's cardinality inside the database (is the link column unique in the detail table?);
3. classifies the edge: `snapshot` (unique -> LEFT JOIN like a hand-written join) or `detail` (1:N -> aggregate);
4. builds the point-in-time aggregation for `detail` edges entirely in SQL: only events strictly before the sample's
   `date_col` are used, over several look-back windows, so no future information can leak.

Detail tables without a usable event-time column are skipped (a point-in-time guarantee is impossible). All
identifiers come from database metadata and values are bound; the read-only session guard still applies.
Profile section `auto_join` (all optional): `enabled`, `max_tables`, `link_map`, `windows_days` (0 = all history),
`aggs` (sum/avg/max/min), `max_agg_cols`, `time_col_patterns`, `skip_categories`.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd
import sqlalchemy as sa

from app.tools import db_explorer

DEFAULT_WINDOWS = [7, 30, 90, 0]
DEFAULT_AGGS = ["sum", "avg", "max", "min"]
DEFAULT_TIME_PATTERNS = ["create_time", "created_at", "add_time", "apply_time", "*_time", "*_date", "*_at"]
_AGG_FUNCS = {"sum": sa.func.sum, "avg": sa.func.avg, "max": sa.func.max, "min": sa.func.min}
_NUMERIC_TOKENS = ("INT", "NUMERIC", "DECIMAL", "FLOAT", "DOUBLE", "REAL")
_DATETIME_TOKENS = ("DATETIME", "TIMESTAMP", "DATE")
_ID_NAME = re.compile(r"(^|_)(id|no|code|uuid)$", re.IGNORECASE)


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


def plan_edges(profile: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Infers edges for the usable (non post-event) candidate tables. Returns {"edges": [...], "skipped": [...]}."""
    cfg = profile.get("auto_join") or {}
    if not cfg.get("enabled"):
        return {"edges": [], "skipped": []}
    base = profile["sample"]["table"]
    sample_cols = {c["name"] for c in db_explorer.describe_table(base)}
    link_map = cfg.get("link_map") or {}
    patterns = cfg.get("time_col_patterns") or DEFAULT_TIME_PATTERNS
    skip_cats = set(cfg.get("skip_categories") or [])
    max_tables = int(cfg.get("max_tables", 20))
    edges: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for cand in candidates:
        if cand["post_event"]:
            continue
        table = cand["table"]
        if cand.get("category") in skip_cats:
            skipped.append({"table": table, "reason": f"类别 {cand['category']} 被配置跳过"})
            continue
        link = next(((c, link_map.get(c, c)) for c in cand["link_columns"] if link_map.get(c, c) in sample_cols), None)
        if link is None:
            skipped.append({"table": table, "reason": "关联列在样本表中没有对应列"})
            continue
        if len(edges) >= max_tables:
            skipped.append({"table": table, "reason": f"超过 auto_join.max_tables={max_tables}"})
            continue
        columns = db_explorer.describe_table(table)
        card = _cardinality(table, link[0])
        edge = {
            "table": table,
            "comment": cand.get("comment"),
            "category": cand.get("category"),
            "link_col": link[0],
            "sample_col": link[1],
            "columns": columns,
            "time_col": pick_time_col(columns, patterns),
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
    return {"edges": edges, "skipped": skipped}


def _shift(col: Any, days: int) -> Any:
    dialect = db_explorer.get_engine().dialect.name
    if dialect == "mysql":
        return sa.func.date_sub(col, sa.text(f"INTERVAL {int(days)} DAY"))
    if dialect == "sqlite":
        return sa.func.datetime(col, f"-{int(days)} days")
    raise ValueError(f"自动聚合暂不支持数据库方言 {dialect}")


def aggregate_detail(
    profile: Dict[str, Any],
    edge: Dict[str, Any],
    base_table: sa.TableClause,
    where: Optional[Any],
    reason_for: Callable[[str, str], Optional[str]],
    key_max: Optional[Any],
) -> Dict[str, Any]:
    """Point-in-time aggregates of one 1:N table per sample row. Returns {"df", "col_meta", "excluded"}."""
    cfg = profile.get("auto_join") or {}
    windows = cfg.get("windows_days") or DEFAULT_WINDOWS
    aggs = [a for a in (cfg.get("aggs") or DEFAULT_AGGS) if a in _AGG_FUNCS]
    max_cols = int(cfg.get("max_agg_cols", 10))
    sample = profile["sample"]
    table, link, time_col = edge["table"], edge["link_col"], edge["time_col"]

    dt = sa.table(table, *[sa.column(c["name"]) for c in edge["columns"]])
    bt = base_table
    event = dt.c[time_col]
    on = (dt.c[link] == bt.c[edge["sample_col"]]) & (event < bt.c[sample["date_col"]])

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
            col_meta[f"{table}.{name}"] = {"table": table, "column": name, "comment": c.get("comment"), "category": edge.get("category")}
        elif len(value_cols) < max_cols:
            value_cols.append(c)

    exprs: List[Any] = []

    def add(label: str, expr: Any, comment: str) -> None:
        exprs.append(expr.label(label))
        col_meta[label] = {"table": table, "column": label, "comment": comment, "category": edge.get("category")}

    for w in windows:
        suffix = f"{w}d" if w else "all"
        in_window = (event >= _shift(bt.c[sample["date_col"]], w)) if w else None
        count_expr = (
            sa.func.sum(sa.case((in_window, 1), else_=0)) if w else sa.func.count(dt.c[link])
        )
        add(f"{table}.cnt_{suffix}", count_expr, f"申请前{w if w else '全部'}天内 {table} 记录数")
        for c in value_cols:
            value = sa.case((in_window, dt.c[c["name"]])) if w else dt.c[c["name"]]
            for agg in aggs:
                add(f"{table}.{c['name']}__{agg}_{suffix}", _AGG_FUNCS[agg](value), f"{c.get('comment') or c['name']}（申请前{w if w else '全部'}天 {agg}）")

    stmt = sa.select(bt.c[sample["key"]].label("__k"), *exprs).select_from(bt.outerjoin(dt, on))
    if where is not None:
        stmt = stmt.where(where)
    if key_max is not None:
        stmt = stmt.where(bt.c[sample["key"]] <= key_max)
    stmt = stmt.group_by(bt.c[sample["key"]])
    with db_explorer.get_engine().connect() as conn:
        df = pd.read_sql(stmt, conn)
    return {"df": df, "col_meta": col_meta, "excluded": excluded}
