"""Tests for the read-only DB explorer / indicator-mining agent (SQLite stands in for MySQL)."""

from __future__ import annotations

import json
import random

import pandas as pd
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from app.api.server import app
from app.tools import db_explorer
from app.tools.data_registry import DataPointerRegistry

BAD, GOOD = "风险用户", "一般用户"


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A SQLite database shaped like the real source table: numbers stored as text, Chinese label, id-like columns."""
    rng = random.Random(3)
    rows = []
    for i in range(800):
        x1, x2, noise = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        bad = x1 + 0.5 * x2 + rng.gauss(0, 1) > 0.3
        rows.append(
            {
                "id": i + 1,
                "bvn_number": f"{22000000000 + i}",
                "income": f"{int(5000 + 1000 * x1):d}.00" if i % 11 else "",  # numeric text with empty strings
                "util_pct": f"{max(0, min(100, int(50 + 20 * x2)))}%",  # percent text
                "noise": round(noise, 4),
                "rating": "A" if i % 3 else "B",  # real text
                "gender": rng.choice(["M", "F"]),
                "user_label": BAD if bad else GOOD,
                "created_at": f"2025-{1 + i // 200:02d}-{1 + i % 28:02d} 10:00:00",
            }
        )
    path = tmp_path / "src.db"
    seed = sa.create_engine(f"sqlite:///{path}")
    pd.DataFrame(rows).to_sql("risk_features", seed, index=False)
    pd.DataFrame({"a": [1]}).to_sql("other_table", seed, index=False)
    seed.dispose()

    engine = sa.create_engine(f"sqlite:///{path}")
    db_explorer.set_engine(engine)
    yield engine
    db_explorer.set_engine(None)
    engine.dispose()


MINE = {"table": "risk_features", "label_col": "user_label", "bad_values": [BAD], "good_values": [GOOD], "date_col": "created_at"}


# ---- safety ---------------------------------------------------------------------------------------
def test_connection_is_read_only(db):
    with db.connect() as conn:
        with pytest.raises(sa.exc.OperationalError):
            conn.execute(sa.text("DELETE FROM risk_features"))
        assert conn.execute(sa.text("SELECT COUNT(*) FROM risk_features")).scalar() == 800


def test_unknown_table_and_column_rejected(db):
    with pytest.raises(KeyError):
        db_explorer.describe_table("risk_features; DROP TABLE x")
    with pytest.raises(KeyError):
        db_explorer.mine_table("risk_features", "no_such_col", [BAD], [GOOD])


def test_allowed_tables_whitelist(db, monkeypatch):
    assert {t["table"] for t in db_explorer.list_tables()} == {"risk_features", "other_table"}
    monkeypatch.setenv("DB_ALLOWED_TABLES", "risk_features")
    assert [t["table"] for t in db_explorer.list_tables()] == ["risk_features"]
    with pytest.raises(KeyError):
        db_explorer.describe_table("other_table")


def test_not_configured_returns_503(monkeypatch):
    for k in ("DB_HOST", "DB_USER", "DB_NAME", "DB_DATABASE"):
        monkeypatch.delenv(k, raising=False)
    db_explorer.set_engine(None)
    client = TestClient(app)
    assert client.get("/api/v1/db/status").json() == {"configured": False}
    assert client.get("/api/v1/db/tables").status_code == 503
    assert client.post("/api/v1/db/mine", json=MINE).status_code == 503


def test_build_db_url_handles_special_chars_in_password(monkeypatch):
    monkeypatch.setenv("DB_HOST", "db.example.com")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_USER", "llm_dev")
    monkeypatch.setenv("DB_PASSWORD", "@Xyc333509#!")
    monkeypatch.setenv("DB_NAME", "merge")
    url = db_explorer._build_db_url()
    assert url is not None
    assert url.host == "db.example.com"
    assert url.port == 3306
    assert url.username == "llm_dev"
    assert url.password == "@Xyc333509#!"
    assert url.database == "merge"


# ---- profile & type inference ---------------------------------------------------------------------
def test_profile_counts_inside_database(db):
    prof = db_explorer.profile_table("risk_features")
    cols = {c["name"]: c for c in prof["columns"]}
    assert prof["row_count"] == 800
    assert cols["income"]["missing_rate"] == pytest.approx(73 / 800, abs=1e-3)  # '' counts as missing
    assert cols["id"]["distinct"] == 800


def test_coerce_numeric_text():
    s, kind = db_explorer.coerce_numeric_text(pd.Series(["100%", "", "50%", "7"]))
    assert kind == "numeric_text" and s.tolist()[0] == 100.0 and pd.isna(s.iloc[1])
    assert db_explorer.coerce_numeric_text(pd.Series(["A", "B", "A"]))[1] == "text"
    assert db_explorer.coerce_numeric_text(pd.Series(["", ""]))[1] == "empty"
    assert db_explorer.coerce_numeric_text(pd.Series([1, 2, 3]))[1] == "numeric"


# ---- mining ---------------------------------------------------------------------------------------
def test_label_other_values_require_good_values(db):
    with pytest.raises(ValueError, match="good_values"):
        db_explorer.mine_table("risk_features", "user_label", [BAD], None, "created_at")
    with pytest.raises(ValueError, match="single class"):
        db_explorer.mine_table("risk_features", "user_label", [BAD, GOOD], [], "created_at")


def test_mine_suggests_real_signals_and_excludes_noise_ids_and_text(db):
    res = db_explorer.mine_table(**MINE)
    assert res["split"]["strict_oot"] is True and res["label"]["rows_used"] == 800
    assert res["column_kinds"]["income"] == "numeric_text" and res["column_kinds"]["util_pct"] == "numeric_text"
    suggested = set(res["suggested_features"])
    assert {"income", "util_pct"} <= suggested
    why = res["not_suggested"]
    assert "标识符" in why["bvn_number"] and "标识符" in why["id"]
    assert "文本" in why["rating"] and "合规" in why["gender"]  # name rule beats the generic text reason
    assert "noise" not in suggested and "created_at" not in res["feature_metrics"]
    m = {x["feature"]: x for x in res["feature_metrics"]}
    assert m["income"]["iv"] > m["noise"]["iv"] and 0 <= m["income"]["ks"] <= 1


def test_mine_without_date_col_flags_row_split(db):
    res = db_explorer.mine_table("risk_features", "user_label", [BAD], [GOOD])
    assert res["split"]["strict_oot"] is False and any("按行顺序" in w for w in res["warnings"])


def test_mine_oot_start_date_split(db):
    res = db_explorer.mine_table(**MINE, oot_start_date="2025-03-01")
    assert res["split"]["train_rows"] + res["split"]["oot_rows"] == 800 and "2025-03-01" in res["split"]["mode"]


def test_max_rows_truncates_and_warns(db):
    res = db_explorer.mine_table(**MINE, max_rows=300)
    assert res["label"]["rows_used"] == 300 and res["label"]["truncated"] and any("取样上限" in w for w in res["warnings"])


def test_small_cohort_falls_back_to_full_sample_instead_of_raising(db):
    res = db_explorer.mine_table(**MINE, max_rows=34)
    assert res["label"]["rows_used"] == 34
    assert res["split"]["train_rows"] == 34 and res["split"]["oot_rows"] == 0
    assert any("已自动回退为全量样本" in w for w in res["warnings"])


# ---- API ------------------------------------------------------------------------------------------
def _events(resp):
    out = []
    for block in resp.text.strip().split("\n\n"):
        name, data = "data", None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        out.append((name, data))
    return out


def test_mine_endpoint_streams_steps_then_result(db):
    resp = TestClient(app).post("/api/v1/db/mine", json=MINE)
    events = _events(resp)
    assert [n for n, _ in events] == ["data", "data", "data", "data", "result"]
    result = events[-1][1]
    assert "指标挖掘报告" in result["report_markdown"] and "income" in result["report_markdown"]
    assert result["mining"]["suggested_features"]


def test_mine_endpoint_error_event_does_not_leak(db):
    events = _events(TestClient(app).post("/api/v1/db/mine", json={**MINE, "table": "missing"}))
    assert events[-1][0] == "error" and "not found" in events[-1][1]["message"]


def test_sample_endpoint_registers_dataset_and_refuses_overwrite(db, tmp_path):
    client = TestClient(app)
    body = {**MINE, "uri": "ds://db_risk"}
    resp = client.post("/api/v1/db/sample", json=body)
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["row_count"] == 800 and out["filename"] == "db_risk_features.csv" and out["obs_window"] != "未提供"
    meta = DataPointerRegistry.get_instance().get_meta("ds://db_risk")
    assert meta.target_col == "user_label" and set(DataPointerRegistry.get_instance().get_df("ds://db_risk")["user_label"]) == {0, 1}
    assert client.post("/api/v1/db/sample", json={**body, "uri": "ds://db_risk2"}).status_code == 409
    assert client.post("/api/v1/db/sample", json={**body, "uri": "bad", "filename": "x.csv"}).status_code == 400


# ---- metadata round trips -------------------------------------------------------------------------
class _FakeMySQLEngine:
    """Records statements; proves list / discovery metadata costs one query each instead of one per table."""

    def __init__(self, rows):
        self.dialect = sa.create_engine("mysql+pymysql://u:p@h/d").dialect
        self.rows, self.statements = rows, []

    def connect(self):
        engine = self

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, stmt):
                engine.statements.append(str(stmt.compile(dialect=engine.dialect)))
                return iter(engine.rows)

        return _Conn()


def test_mysql_table_comments_use_a_single_information_schema_query(monkeypatch):
    fake = _FakeMySQLEngine([("a", "表A"), ("b", "")])
    monkeypatch.setattr(db_explorer, "get_engine", lambda: fake)
    assert db_explorer.table_comments(["a", "b", "c"]) == {"a": "表A", "b": None, "c": None}
    assert len(fake.statements) == 1 and "information_schema.tables" in fake.statements[0]


def test_mysql_table_columns_use_a_single_information_schema_query(monkeypatch):
    fake = _FakeMySQLEngine([("a", "x"), ("a", "y"), ("b", "z")])
    monkeypatch.setattr(db_explorer, "get_engine", lambda: fake)
    assert db_explorer.table_columns(["a", "b", "c"]) == {"a": ["x", "y"], "b": ["z"], "c": []}
    assert len(fake.statements) == 1 and "information_schema.columns" in fake.statements[0]
    assert "database()" in fake.statements[0].lower()  # scoped to the connected schema


def test_sqlite_fallback_matches_inspector(db):
    assert db_explorer.table_columns(["risk_features"])["risk_features"][0] == "id"
    assert db_explorer.table_comments(["risk_features"]) == {"risk_features": None}


# ---- correlation helper ---------------------------------------------------------------------------
def test_abs_corr_is_zero_and_silent_when_undefined():
    import warnings

    import numpy as np

    nan = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        assert db_explorer._abs_corr(pd.Series([1.0, 1, 1, nan, 2, 3]), pd.Series([4.0, 5, 6, 7, nan, nan])) == 0.0  # constant on overlap
        assert db_explorer._abs_corr(pd.Series([1.0, 2, 3, nan, nan, nan]), pd.Series([nan, nan, nan, 4.0, 5, 6])) == 0.0  # no overlap
        assert db_explorer._abs_corr(pd.Series([1.0, 2]), pd.Series([2.0, 4])) == 0.0  # too few rows
        assert db_explorer._abs_corr(pd.Series([1.0, 2, 3, 4]), pd.Series([2.0, 4, 6, 8])) == pytest.approx(1.0)
        assert db_explorer._abs_corr(pd.Series([1.0, 2, 3, 4]), pd.Series([8.0, 6, 4, 2])) == pytest.approx(1.0)  # sign ignored
