"""Read-only database explorer for indicator mining.

Safety model (the account used may technically have write rights, so read-only is enforced here):
- every statement is built from SQLAlchemy Core `select()` constructs: no SQL text from callers or from an LLM;
- table / column names must exist in the database metadata (and in `DB_ALLOWED_TABLES` when set); values are bound;
- each connection is switched to read-only (`SET SESSION TRANSACTION READ ONLY` on MySQL, `query_only` on SQLite);
- every fetch is row-capped (`DB_MAX_ROWS`) and server-side time-limited (`DB_QUERY_TIMEOUT_S`).

Division of labour: cheap aggregates (row count, empty rate, distinct count) run inside the database; IV / KS / PSI
are computed locally on a capped sample with the same operators the modeling pipeline uses. Many source tables store
numbers as text (`'100%'`, `''` for missing), so column types are inferred from the sample.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import event

from app.core.config import _load_dotenv
from app.core.paths import get_data_dir
from app.tools.credit_metrics import compute_ks_and_auc, compute_woe_iv_and_psi
from app.tools.data_registry import DataPointerRegistry
from app.tools.dataset_files import save_dataframe_csv
from app.tools.leakage_rules import corpus_blacklists, detect_risky_features

DEFAULT_MAX_ROWS = 200_000
MIN_SPLIT_ROWS = 30
NUMERIC_TEXT_MIN_PARSE_RATE = 0.95
IV_MIN, PSI_MAX, MISSING_MAX, CORR_MAX = 0.02, 0.10, 0.95, 0.90
_ID_TOKENS = frozenset({"id", "uuid", "number", "no", "code"})

_engine: Optional[sa.Engine] = None


class DBNotConfigured(RuntimeError):
    """Database connection parameters (`DB_HOST`, `DB_USER`, `DB_NAME`) are not set."""


# ---- connection -----------------------------------------------------------------------------------
def _max_rows() -> int:
    return max(1, int(os.environ.get("DB_MAX_ROWS") or DEFAULT_MAX_ROWS))


def _timeout_s() -> int:
    return max(1, int(os.environ.get("DB_QUERY_TIMEOUT_S") or 30))


def _build_db_url() -> Optional[sa.URL]:
    _load_dotenv()
    host = (os.environ.get("DB_HOST") or "").strip()
    user = (os.environ.get("DB_USER") or "").strip()
    database = (os.environ.get("DB_NAME") or os.environ.get("DB_DATABASE") or "").strip()
    if not (host and user and database):
        return None
    driver = (os.environ.get("DB_DRIVER") or "mysql+pymysql").strip()
    port_raw = (os.environ.get("DB_PORT") or "3306").strip()
    port = int(port_raw) if port_raw else 3306
    password = os.environ.get("DB_PASSWORD") or ""
    charset = (os.environ.get("DB_CHARSET") or "utf8mb4").strip()
    query = {"charset": charset} if charset and driver.startswith("mysql") else {}
    return sa.URL.create(
        drivername=driver,
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
        query=query,
    )


def is_configured() -> bool:
    return _engine is not None or _build_db_url() is not None


def set_engine(engine: Optional[sa.Engine]) -> None:
    """Injects an engine (tests); the read-only guard is attached here too."""
    global _engine
    if engine is not None:
        _attach_read_only_guard(engine)
    _engine = engine


def _attach_read_only_guard(engine: sa.Engine) -> None:
    @event.listens_for(engine, "connect")
    def _read_only(dbapi_conn, _record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        try:
            if engine.dialect.name == "mysql":
                cursor.execute("SET SESSION TRANSACTION READ ONLY")
                cursor.execute(f"SET SESSION MAX_EXECUTION_TIME={_timeout_s() * 1000}")
            elif engine.dialect.name == "sqlite":
                cursor.execute("PRAGMA query_only=ON")
        finally:
            cursor.close()


def get_engine() -> sa.Engine:
    global _engine
    if _engine is None:
        url = _build_db_url()
        if url is None:
            raise DBNotConfigured("Database connection is not configured (DB_HOST, DB_USER, DB_NAME)")
        kwargs: Dict[str, Any] = {"pool_pre_ping": True}
        if url.drivername.startswith("mysql"):
            t = _timeout_s()
            kwargs["connect_args"] = {"connect_timeout": 10, "read_timeout": t, "write_timeout": t}
        engine = sa.create_engine(url, **kwargs)
        _attach_read_only_guard(engine)
        _engine = engine
    return _engine


# ---- metadata -------------------------------------------------------------------------------------
def _allowed_tables() -> List[str]:
    names = sorted(sa.inspect(get_engine()).get_table_names())
    raw = os.environ.get("DB_ALLOWED_TABLES", "").strip()
    if raw:
        allowed = {t.strip() for t in raw.split(",") if t.strip()}
        names = [n for n in names if n in allowed]
    return names


def _info_schema(name: str, *cols: str) -> sa.TableClause:
    return sa.table(name, *[sa.column(c) for c in cols], schema="information_schema")


def table_comments(tables: Sequence[str]) -> Dict[str, Optional[str]]:
    """Table comments. MySQL: one information_schema query instead of one round trip per table."""
    engine = get_engine()
    if engine.dialect.name == "mysql":
        t = _info_schema("tables", "table_schema", "table_name", "table_comment")
        stmt = sa.select(t.c.table_name.label("t"), t.c.table_comment.label("c")).where(t.c.table_schema == sa.func.database())
        with engine.connect() as conn:
            found = {row.t: (row.c or None) for row in conn.execute(stmt)}
        return {name: found.get(name) for name in tables}
    inspector = sa.inspect(engine)
    out: Dict[str, Optional[str]] = {}
    for name in tables:
        try:
            out[name] = inspector.get_table_comment(name).get("text")
        except Exception:  # noqa: BLE001 - comments are optional metadata
            out[name] = None
    return out


def table_columns(tables: Sequence[str]) -> Dict[str, List[str]]:
    """Column names per table. MySQL: one information_schema query instead of one reflection per table."""
    engine = get_engine()
    if engine.dialect.name == "mysql":
        t = _info_schema("columns", "table_schema", "table_name", "column_name", "ordinal_position")
        stmt = (
            sa.select(t.c.table_name.label("t"), t.c.column_name.label("c"))
            .where(t.c.table_schema == sa.func.database())
            .order_by(t.c.table_name, t.c.ordinal_position)
        )
        grouped: Dict[str, List[str]] = {}
        with engine.connect() as conn:
            for row in conn.execute(stmt):
                grouped.setdefault(row.t, []).append(row.c)
        return {name: grouped.get(name, []) for name in tables}
    inspector = sa.inspect(engine)
    return {name: [c["name"] for c in inspector.get_columns(name)] for name in tables}


def list_tables() -> List[Dict[str, Any]]:
    names = _allowed_tables()
    comments = table_comments(names)
    return [{"table": name, "comment": comments.get(name)} for name in names]


def describe_table(table: str) -> List[Dict[str, Any]]:
    """Columns of an allowed table. KeyError when the table is unknown or not allowed."""
    if table not in _allowed_tables():
        raise KeyError(f"table '{table}' not found or not allowed")
    return [
        {
            "name": c["name"],
            "type": str(c["type"]),
            "is_text": isinstance(c["type"], sa.types.String),
            "comment": c.get("comment"),
        }
        for c in sa.inspect(get_engine()).get_columns(table)
    ]


def _table_clause(table: str, columns: Sequence[Dict[str, Any]]) -> sa.TableClause:
    return sa.table(table, *[sa.column(c["name"]) for c in columns])


def _check_cols(columns: Sequence[Dict[str, Any]], *names: Optional[str]) -> None:
    known = {c["name"] for c in columns}
    for name in names:
        if name is not None and name not in known:
            raise KeyError(f"column '{name}' not found")


# ---- in-database aggregates -----------------------------------------------------------------------
def profile_table(table: str) -> Dict[str, Any]:
    """Row count plus per-column empty rate and distinct count, computed inside the database."""
    cols = describe_table(table)
    t = _table_clause(table, cols)
    exprs: List[Any] = [sa.func.count().label("n_rows")]
    for i, c in enumerate(cols):
        col = t.c[c["name"]]
        missing = col.is_(None) | (col == "") if c["is_text"] else col.is_(None)
        exprs.append(sa.func.sum(sa.case((missing, 1), else_=0)).label(f"m{i}"))
        exprs.append(sa.func.count(sa.distinct(col)).label(f"d{i}"))
    with get_engine().connect() as conn:
        row = conn.execute(sa.select(*exprs).select_from(t)).mappings().one()
    n = int(row["n_rows"])
    return {
        "table": table,
        "row_count": n,
        "columns": [
            {
                "name": c["name"],
                "type": c["type"],
                "missing_rate": round(float(row[f"m{i}"] or 0) / n, 4) if n else 0.0,
                "distinct": int(row[f"d{i}"]),
            }
            for i, c in enumerate(cols)
        ],
    }


def _fetch_sample(table: str, cols: Sequence[Dict[str, Any]], max_rows: int) -> Tuple[pd.DataFrame, bool]:
    """First `max_rows` rows (one extra is read to detect truncation)."""
    t = _table_clause(table, cols)
    stmt = sa.select(*t.c).select_from(t).limit(max_rows + 1)
    with get_engine().connect() as conn:
        df = pd.read_sql(stmt, conn)
    truncated = len(df) > max_rows
    return df.iloc[:max_rows].copy(), truncated


# ---- type inference & labelling -------------------------------------------------------------------
def coerce_numeric_text(s: pd.Series) -> Tuple[pd.Series, str]:
    """Returns (series, kind) with kind in numeric | numeric_text | datetime | empty | text.

    Text is numeric-like when, ignoring empty strings and a trailing '%', at least 95% of values parse as numbers.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float), "numeric"
    if pd.api.types.is_numeric_dtype(s):
        return s, "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, "datetime"
    text = s.astype("string").str.strip()
    text = text.mask(text == "")
    non_empty = int(text.notna().sum())
    if non_empty == 0:
        return pd.to_numeric(text, errors="coerce"), "empty"
    parsed = pd.to_numeric(text.str.rstrip("%").str.strip(), errors="coerce")
    if parsed.notna().sum() / non_empty >= NUMERIC_TEXT_MIN_PARSE_RATE:
        return parsed.astype(float), "numeric_text"
    return s, "text"


def _map_label(
    values: pd.Series, bad_values: Sequence[str], good_values: Optional[Sequence[str]]
) -> Tuple[pd.Series, Dict[str, Any]]:
    """1 = risk event, 0 = good, NaN = dropped. Refuses to guess what 'good' means."""
    text = values.astype("string").str.strip()
    text = text.mask(text == "")
    bad = {str(v).strip() for v in bad_values}
    others = sorted(set(text.dropna().unique()) - bad)
    if others and not good_values:
        raise ValueError(
            "label column has values other than bad_values; pass good_values explicitly "
            f"(found: {others[:10]}{'...' if len(others) > 10 else ''})"
        )
    good = {str(v).strip() for v in (good_values or [])}
    y = pd.Series(np.nan, index=values.index, dtype=float)
    y[text.isin(bad)] = 1.0
    y[text.isin(good - bad)] = 0.0
    info = {
        "bad_count": int((y == 1).sum()),
        "good_count": int((y == 0).sum()),
        "dropped_unlabeled": int(y.isna().sum()),
        "unmapped_values": sorted(set(text.dropna().unique()) - bad - good)[:10],
    }
    if info["bad_count"] == 0 or info["good_count"] == 0:
        raise ValueError(f"label mapping yields a single class ({info['bad_count']} bad / {info['good_count']} good)")
    return y, info


def _load_labeled(
    table: str,
    label_col: str,
    bad_values: Sequence[str],
    good_values: Optional[Sequence[str]],
    date_col: Optional[str],
    max_rows: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """Capped sample with the label mapped to 0/1 (same column name) and numeric text converted; sorted by date."""
    cols = describe_table(table)
    _check_cols(cols, label_col, date_col)
    cap = min(max_rows or _max_rows(), _max_rows())
    df, truncated = _fetch_sample(table, cols, cap)
    warnings: List[str] = []
    if truncated:
        warnings.append(f"表行数超过取样上限 {cap}，只使用前 {cap} 行；如需更大样本请调大 DB_MAX_ROWS。")

    y, label_info = _map_label(df[label_col], bad_values, good_values)
    df, label_info = clean_labeled(df, y, label_info, label_col, date_col, truncated)
    return df, label_info, warnings


def clean_labeled(
    df: pd.DataFrame,
    y: pd.Series,
    label_info: Dict[str, Any],
    label_col: str,
    date_col: Optional[str],
    truncated: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Applies a 0/1 label (NaN = drop) to `df`: date filter + sort, numeric-text conversion, column kinds."""
    keep = y.notna()
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        label_info["dropped_bad_date"] = int((keep & dates.isna()).sum())
        keep &= dates.notna()
        df[date_col] = dates
    df = df[keep].copy()
    df[label_col] = y[keep].astype(int)
    if date_col:
        df = df.sort_values(date_col, kind="stable").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("no usable rows after label / date filtering")
    kinds: Dict[str, str] = {}
    for name in df.columns:
        if name in (label_col, date_col):
            continue
        df[name], kinds[name] = coerce_numeric_text(df[name])
    label_info.update({"rows_used": len(df), "truncated": truncated, "column_kinds": kinds})
    return df, label_info


def _split(
    df: pd.DataFrame, date_col: Optional[str], oot_start_date: Optional[str], train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if date_col and oot_start_date:
        cut = pd.Timestamp(oot_start_date)
        train, oot = df[df[date_col] < cut], df[df[date_col] >= cut]
        mode = f"按日期切分：{oot_start_date} 之前为开发集，之后为跨期集"
    else:
        n = int(len(df) * train_ratio)
        train, oot = df.iloc[:n], df.iloc[n:]
        mode = (
            "按日期排序后切分：前 70% 为开发集，后 30% 为跨期集"
            if date_col
            else "未提供日期列：按行顺序切分前 70% / 后 30%（非严格跨期，PSI 仅供参考）"
        )
    info = {"mode": mode, "strict_oot": bool(date_col), "train_rows": len(train), "oot_rows": len(oot)}
    if date_col:
        info["train_window"] = f"{train[date_col].min():%Y-%m-%d} ~ {train[date_col].max():%Y-%m-%d}" if len(train) else "-"
        info["oot_window"] = f"{oot[date_col].min():%Y-%m-%d} ~ {oot[date_col].max():%Y-%m-%d}" if len(oot) else "-"
    return train, oot, info


def _abs_corr(a: pd.Series, b: pd.Series) -> float:
    """|Pearson r| over rows where both are present; 0.0 when undefined (too few rows or a constant side).

    Sparse columns often overlap on few rows where one side is constant; pandas would then divide by a zero
    standard deviation and emit a RuntimeWarning.
    """
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return 0.0
    return abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])))


def _looks_like_id(name: str, series: pd.Series) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
    non_null = series.dropna()
    return bool(tokens & _ID_TOKENS) and len(non_null) > 0 and non_null.nunique() / len(non_null) >= 0.9


# ---- indicator mining -----------------------------------------------------------------------------
def mine_table(
    table: str,
    label_col: str,
    bad_values: Sequence[str],
    good_values: Optional[Sequence[str]] = None,
    date_col: Optional[str] = None,
    oot_start_date: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Single-variable indicator report (missing rate, IV, KS, PSI) and candidate-feature suggestions."""
    df, label_info, warnings = _load_labeled(table, label_col, bad_values, good_values, date_col, max_rows)
    return mine_frame(table, df, label_info, warnings, label_col, bad_values, date_col, oot_start_date)


def mine_frame(
    table: str,
    df: pd.DataFrame,
    label_info: Dict[str, Any],
    warnings: List[str],
    label_col: str,
    bad_values: Sequence[str],
    date_col: Optional[str] = None,
    oot_start_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Indicator mining on an already labelled frame (shared by single-table and multi-table project mining)."""
    train, oot, split_info = _split(df, date_col, oot_start_date)
    kinds: Dict[str, str] = label_info["column_kinds"]

    if len(train) < MIN_SPLIT_ROWS or train[label_col].nunique() < 2:
        raise ValueError(f"development part has {len(train)} rows or a single label class; cannot compute indicators")
    if len(oot) < MIN_SPLIT_ROWS:
        warnings.append(f"跨期集仅 {len(oot)} 行，PSI 不可靠，已跳过 PSI。")
    if not split_info["strict_oot"]:
        warnings.append("未提供日期列，开发集 / 跨期集按行顺序切分，PSI 不代表真实时间稳定性。")
    if label_info["bad_count"] < MIN_SPLIT_ROWS or label_info["good_count"] < MIN_SPLIT_ROWS:
        warnings.append(
            f"标签样本很少（风险 {label_info['bad_count']} / 非风险 {label_info['good_count']}），指标稳定性差，仅适合验证流程。"
        )

    leakage, compliance = detect_risky_features(
        train[list(kinds) + [label_col]], label_col, list(kinds), *corpus_blacklists()
    )
    y_train = train[label_col]
    metrics: List[Dict[str, Any]] = []
    excluded: Dict[str, str] = {}
    for name, kind in kinds.items():
        if name in compliance:
            excluded[name] = f"合规敏感: {compliance[name]}"
            continue
        if name in leakage:
            excluded[name] = f"疑似穿越: {leakage[name]}"
            continue
        if kind in ("text", "datetime", "empty"):
            excluded[name] = {"text": "非数值文本列", "datetime": "时间列", "empty": "整列为空"}[kind]
            continue
        if _looks_like_id(name, df[name]):
            excluded[name] = "疑似标识符（近乎唯一且命名含 id/number/no/code）"
            continue
        tr_col, oot_col = train[name], oot[name]
        missing_rate = float(df[name].isna().mean())
        if tr_col.dropna().nunique() < 2:
            excluded[name] = "开发集内为常量或全空"
            continue
        _, _, iv, psi, bins = compute_woe_iv_and_psi(tr_col, y_train, oot_col, n_bins=5)
        filled = tr_col.fillna(tr_col.median())
        ks, auc = compute_ks_and_auc(y_train.to_numpy(), filled.to_numpy())
        metrics.append(
            {
                "feature": name,
                "kind": kind,
                "missing_rate": round(missing_rate, 4),
                "iv": iv,
                "ks": ks,
                "auc": round(max(auc, 1 - auc), 4),
                "psi": psi if len(oot) >= MIN_SPLIT_ROWS else None,
                "bins": len(bins),
            }
        )
    metrics.sort(key=lambda m: m["iv"], reverse=True)

    suggested: List[str] = []
    reasons: Dict[str, str] = {}
    for m in metrics:
        f = m["feature"]
        if m["missing_rate"] > MISSING_MAX:
            reasons[f] = f"缺失率 {m['missing_rate']} > {MISSING_MAX}"
        elif m["iv"] < IV_MIN:
            reasons[f] = f"IV {m['iv']} < {IV_MIN}"
        elif m["psi"] is not None and m["psi"] > PSI_MAX:
            reasons[f] = f"PSI {m['psi']} > {PSI_MAX}"
        else:
            partner = next(
                (s for s in suggested if _abs_corr(train[f], train[s]) > CORR_MAX), None  # metrics are IV-sorted
            )
            if partner:
                reasons[f] = f"与更高 IV 的 {partner} 相关系数绝对值 > {CORR_MAX}"
            else:
                suggested.append(f)
    return {
        "table": table,
        "label": {"column": label_col, "bad_values": list(bad_values), **{k: v for k, v in label_info.items() if k != "column_kinds"}},
        "split": split_info,
        "column_kinds": kinds,
        "feature_metrics": metrics,
        "suggested_features": suggested,
        "not_suggested": {**excluded, **reasons},
        "thresholds": {"iv_min": IV_MIN, "psi_max": PSI_MAX, "missing_max": MISSING_MAX, "corr_max": CORR_MAX},
        "warnings": warnings,
    }


# ---- sample -> dataset ----------------------------------------------------------------------------
def sample_to_dataset(
    table: str,
    label_col: str,
    bad_values: Sequence[str],
    uri: str,
    good_values: Optional[Sequence[str]] = None,
    date_col: Optional[str] = None,
    max_rows: Optional[int] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Writes the cleaned, date-sorted sample (label as 0/1) to data/ and registers it as a persisted dataset."""
    if not uri.startswith("ds://"):
        raise ValueError("uri must start with 'ds://'")
    df, info, warnings = _load_labeled(table, label_col, bad_values, good_values, date_col, max_rows)
    if date_col:
        df[date_col] = df[date_col].dt.strftime("%Y-%m-%d %H:%M:%S")
    name = filename or f"db_{table}.csv"
    save_dataframe_csv(name, df)  # validates the name, never overwrites
    try:
        meta = DataPointerRegistry.get_instance().register_from_file(
            uri, name, label_col, date_col=date_col, description=f"数据库表 {table} 抽样（{len(df)} 行）"
        )
    except Exception:
        (get_data_dir() / name).unlink(missing_ok=True)
        raise
    return {
        "uri": uri,
        "filename": name,
        "row_count": meta.row_count,
        "bad_rate": meta.bad_rate,
        "obs_window": f"{meta.obs_start_date} ~ {meta.obs_end_date}" if meta.obs_start_date else "未提供",
        "warnings": warnings,
    }
