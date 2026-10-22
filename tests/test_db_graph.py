"""Tests for automatic multi-table linking and point-in-time aggregation (SQLite stands in for MySQL)."""

from __future__ import annotations

import copy
import random

import pandas as pd
import pytest
import sqlalchemy as sa

from app.tools import db_explorer, db_graph, db_project

PROFILE = {
    "name": "graph-test",
    "sample": {"table": "order_apply", "key": "id", "user_key": "user_id", "date_col": "loan_time"},
    "params": {"bad_overdue_days": 30},
    "population": {"all": [{"col": "status", "op": ">=", "value": 62}]},
    "label": {
        "name": "risk_label",
        "bad": {"all": [{"col": "overdue_days", "op": ">=", "value": "$bad_overdue_days"}]},
        "good": {"all": [{"col": "status", "op": "==", "value": 80}, {"col": "overdue_days", "op": "<=", "value": 0}]},
    },
    "exclude": [{"reason": "post-loan", "patterns": ["status", "overdue_days"]}, {"reason": "pii", "patterns": ["secret_amount"]}],
    "categories": {"use_llm": False, "rules": []},
    "discovery": {"link_columns": ["user_id"], "use_llm": False, "post_event_table_patterns": ["collection*"]},
    "auto_join": {"enabled": True, "windows_days": [30, 0], "aggs": ["sum", "max"]},
}


@pytest.fixture
def db(tmp_path):
    rng = random.Random(5)
    orders, users, events = [], [], []
    for i in range(1, 601):
        x = rng.gauss(0, 1)
        bad = x + rng.gauss(0, 1) > 0.4
        status = 62 if bad else 80
        loan = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i % 120, hours=10)
        orders.append(
            {"id": i, "user_id": i, "status": status, "overdue_days": rng.randint(30, 60) if bad else 0,
             "loan_amount": 1000 + int(300 * x), "loan_time": loan.strftime("%Y-%m-%d %H:%M:%S")}
        )
        users.append({"user_id": i, "income": rng.gauss(0, 1)})
        for days_before in (5, 60):  # two events before the loan: 30d window sees one, all-history sees both
            events.append({"user_id": i, "amount": 100 + 50 * x + rng.gauss(0, 5), "secret_amount": 1.0, "row_id": len(events),
                           "create_time": (loan - pd.Timedelta(days=days_before)).strftime("%Y-%m-%d %H:%M:%S")})
        events.append({"user_id": i, "amount": 10 ** 6, "secret_amount": 1.0, "row_id": len(events),  # after the loan: must never leak
                       "create_time": (loan + pd.Timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")})
    path = tmp_path / "g.db"
    seed = sa.create_engine(f"sqlite:///{path}")
    pd.DataFrame(orders).to_sql("order_apply", seed, index=False)
    pd.DataFrame(users).to_sql("user_snapshot", seed, index=False)
    pd.DataFrame(events).to_sql("user_events", seed, index=False)
    pd.DataFrame({"user_id": [1, 1], "v": [1, 2]}).to_sql("user_notime", seed, index=False)
    pd.DataFrame({"user_id": [1]}).to_sql("collection_record", seed, index=False)
    seed.dispose()
    engine = sa.create_engine(f"sqlite:///{path}")
    db_explorer.set_engine(engine)
    yield engine
    db_explorer.set_engine(None)
    engine.dispose()


def _plan(profile=PROFILE):
    return db_graph.plan_edges(profile, db_project.discover_candidate_tables(profile, refresh=True))


def test_plan_classifies_snapshot_detail_and_skips_unsafe(db):
    plan = _plan()
    kinds = {e["table"]: e["kind"] for e in plan["edges"]}
    assert kinds == {"user_snapshot": "snapshot", "user_events": "detail"}
    assert next(e for e in plan["edges"] if e["table"] == "user_events")["time_col"] == "create_time"
    skipped = {s["table"]: s["reason"] for s in plan["skipped"]}
    assert "事件时间列" in skipped["user_notime"] and "collection_record" not in skipped  # post-event: not an edge at all


def test_disabled_by_default(db):
    p = copy.deepcopy(PROFILE)
    p["auto_join"]["enabled"] = False
    assert _plan(p) == {"edges": [], "skipped": []}


def test_detail_aggregation_is_point_in_time(db):
    fr = db_project.build_frame(PROFILE, edges=_plan()["edges"])
    df = fr["df"]
    assert (df["user_events.cnt_all"] == 2).all() and (df["user_events.cnt_30d"] == 1).all()
    assert df["user_events.amount__max_all"].max() < 1000  # the 10**6 post-loan event never leaks
    assert "user_events.secret_amount__sum_all" not in df.columns  # profile exclusion applies to aggregates
    assert "user_events.row_id__sum_all" not in df.columns  # id-like columns are not aggregated
    assert "user_snapshot.income" in df.columns
    edge = next(e for e in fr["auto_edges"] if e["table"] == "user_events")
    assert edge["coverage"] == 1.0 and edge["features"] > 0
    assert any("快照" in w for w in fr["warnings"])


def test_aggregated_features_carry_table_category_and_signal(db):
    p = copy.deepcopy(PROFILE)
    p["discovery"]["table_categories"] = [{"name": "行为", "table_patterns": ["user_events"]}]
    edges = db_graph.plan_edges(p, db_project.discover_candidate_tables(p, refresh=True))["edges"]
    fr = db_project.build_frame(p, edges=edges)
    mined = db_project.mine_project(p, fr)
    top = {m["feature"]: m for m in mined["feature_metrics"]}
    assert top["user_events.amount__max_all"]["category"] == "行为" and top["user_events.amount__max_all"]["iv"] > 0.02
    assert mined["auto_edges"] and mined["joins"] is not None


def test_detail_edge_requires_sample_date_col(db):
    p = copy.deepcopy(PROFILE)
    p["sample"].pop("date_col")
    plan = _plan(p)
    assert [e["table"] for e in plan["edges"]] == ["user_snapshot"]
    assert any("date_col" in s["reason"] for s in plan["skipped"])
