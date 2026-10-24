"""Tests for automatic multi-table linking and point-in-time aggregation (SQLite stands in for MySQL)."""

from __future__ import annotations

import copy
import json
import random

import httpx
import pandas as pd
import pytest
import sqlalchemy as sa

from app.agents import db_miner
from app.core.config import LLMConfig
from app.core.llm import LLMClient, reset_llm_client
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


def test_multi_hub_2hop_slices_temporal_cross_and_feature_catalog(tmp_path):
    """End-to-end verification of 2-hop graph discovery, domain slices, temporal velocity, cross features & catalog."""
    rng = random.Random(19)
    orders, users, credit_reports, credit_details, cashflows = [], [], [], [], []
    for i in range(1, 401):
        x = rng.gauss(0, 1)
        bad = x + rng.gauss(0, 0.8) > 0.3
        loan = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i % 90, hours=12)
        orders.append(
            {
                "id": i,
                "user_id": i,
                "status": 62 if bad else 80,
                "overdue_days": rng.randint(45, 90) if bad else 0,
                "loan_amount": 2000 + int(500 * x),
                "loan_time": loan.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        users.append({"id": i, "monthly_income": max(3000 - int(600 * x), 500)})
        rep_id = f"rep_{i}"
        credit_reports.append({"user_id": i, "report_id": rep_id, "query_cnt": rng.randint(1, 5)})
        # 2-hop credit loan detail (linked ONLY via report_id, no user_id or order_id!)
        credit_details.append(
            {
                "report_id": rep_id,
                "loan_status": "overdue" if (bad and i % 3 != 0) or (not bad and i % 5 == 0) else "normal",
                "overdue_amount": max(0.0, 300.0 + 150.0 * x + rng.gauss(0, 120)),
                "create_time": (loan - pd.Timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        # 1-hop bank cashflow with direction slice (in / out) and varying counts/amounts across 7d / 30d / all
        for k in range(1 + (i % 3)):
            cashflows.append(
                {
                    "user_id": i,
                    "direction": "out",
                    "amount": max(10.0, 500.0 + 120.0 * x + rng.gauss(0, 80)),
                    "txn_time": (loan - pd.Timedelta(days=2 + k)).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        for k in range(1 + ((i + 1) % 3)):
            cashflows.append(
                {
                    "user_id": i,
                    "direction": "in",
                    "amount": max(50.0, 1200.0 - 200.0 * x + rng.gauss(0, 150)),
                    "txn_time": (loan - pd.Timedelta(days=12 + k * 3)).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        cashflows.append(
            {
                "user_id": i,
                "direction": "in",
                "amount": max(50.0, 900.0 + rng.gauss(0, 200)),
                "txn_time": (loan - pd.Timedelta(days=45 + (i % 10))).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    path = tmp_path / "multihub.db"
    seed = sa.create_engine(f"sqlite:///{path}")
    pd.DataFrame(orders).to_sql("order_apply", seed, index=False)
    pd.DataFrame(users).to_sql("user_info", seed, index=False)
    pd.DataFrame(credit_reports).to_sql("credit_report", seed, index=False)
    pd.DataFrame(credit_details).to_sql("credit_loan_detail", seed, index=False)
    pd.DataFrame(cashflows).to_sql("bank_flow", seed, index=False)
    seed.dispose()

    engine = sa.create_engine(f"sqlite:///{path}")
    db_explorer.set_engine(engine)
    try:
        p = copy.deepcopy(PROFILE)
        p["params"] = {"bad_overdue_days": 45}
        p["joins"] = [{"table": "user_info", "left_key": "user_id", "right_key": "id"}]
        p["auto_join"] = {"enabled": True, "windows_days": [7, 30, 0], "aggs": ["sum", "avg"]}
        p["discovery"]["table_categories"] = [
            {"name": "征信", "table_patterns": ["credit_*"]},
            {"name": "流水", "table_patterns": ["bank_*"]},
        ]

        candidates = db_project.discover_candidate_tables(p, refresh=True)
        cand_names = {c["table"] for c in candidates}
        # 2-hop table `credit_loan_detail` (only has `report_id`) is automatically discovered!
        assert {"credit_report", "credit_loan_detail", "bank_flow"} <= cand_names

        plan = db_graph.plan_edges(p, candidates)
        by_table = {e["table"]: e for e in plan["edges"]}
        assert by_table["credit_report"]["hop"] == 1 and by_table["credit_report"]["kind"] == "snapshot"
        assert by_table["credit_loan_detail"]["hop"] == 2 and by_table["credit_loan_detail"]["via_table"] == "credit_report"
        assert by_table["credit_loan_detail"]["path"] == "order_apply -> credit_report -> credit_loan_detail"

        fr = db_project.build_frame(p, edges=plan["edges"])
        df = fr["df"]
        assert fr["master_tables"] == ["order_apply", "user_info"]

        # L1 Domain Conditional Slices: bank_flow (in/out) & 2-hop credit_loan_detail (overdue/normal)
        assert "bank_flow.amount__slice_direction_in__sum_30d" in df.columns
        assert "bank_flow.amount__slice_direction_out__sum_30d" in df.columns
        assert "bank_flow.amount__slice_direction_out_div_in_30d" in df.columns
        assert "bank_flow.cnt__slice_direction_out_prop_30d" in df.columns
        assert "credit_loan_detail.overdue_amount__slice_loan_status_overdue__sum_all" in df.columns

        # L2 Temporal Velocity & Concentration Ratios (7d vs 30d, 30d vs all)
        assert "bank_flow.cnt_ratio_7d_30d" in df.columns
        assert "bank_flow.amount__sum_ratio_30d_all" in df.columns

        # L3 Cross-Hub Derived Features (order loan_amount vs user_info.monthly_income & cashflow)
        assert "cross.order_apply.loan_amount__div__user_info.monthly_income" in df.columns

        # Mine & verify Feature Store Catalog + Layer Summary + Report
        mined = db_project.mine_project(p, fr)
        layers_found = {item["layer"] for item in mined["layer_summary"]}
        assert {"base", "window_agg", "domain_slice", "temporal_ratio", "cross_derived"} <= layers_found
        catalog_by_name = {c["feature"]: c for c in mined["feature_catalog"]}
        c_entry = catalog_by_name["credit_loan_detail.overdue_amount__slice_loan_status_overdue__sum_all"]
        assert c_entry["layer"] == "domain_slice"
        assert c_entry["graph_path"] == "order_apply -> credit_report -> credit_loan_detail"
        assert c_entry["iv"] > 0.1

        prof = db_explorer.profile_table("order_apply")
        report = db_miner.build_project_report(prof, mined, candidates)
        assert "特征工程层级分布" in report
        assert "order_apply -> credit_report -> credit_loan_detail" in report
    finally:
        db_explorer.set_engine(None)
        engine.dispose()


def test_llm_cross_feature_recommender_whitelist_guardrail(db):
    def handler(req):
        body = json.dumps(
            {
                "pairs": [
                    {"num": "loan_amount", "den": "user_snapshot.income", "op": "sub", "comment": "申请额减收入"},
                    {"num": "loan_amount", "den": "hallucinated_col", "op": "div", "comment": "幻觉列必须拦截"},
                ]
            }
        )
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": body}}], "usage": {}})

    reset_llm_client(LLMClient(LLMConfig("deepseek", "https://x", "m", "k", 5, 0, True), transport=httpx.MockTransport(handler)))
    p = copy.deepcopy(PROFILE)
    p["categories"]["use_llm"] = True
    fr = db_project.build_frame(p, edges=_plan(p)["edges"])
    assert "cross.order_apply.loan_amount__sub__user_snapshot.income" in fr["df"].columns
    assert not any("hallucinated_col" in c for c in fr["df"].columns)


def test_dual_master_tables_order_and_user_graph_and_column_lineage(tmp_path):
    """Explicit 2-hub (`order_apply` + `user_info`) configuration routes order & user tables with `<table.col>` lineage."""
    path = tmp_path / "dual_hub.db"
    seed = sa.create_engine(f"sqlite:///{path}")
    pd.DataFrame(
        [
            {"id": 1, "user_id": 10, "status": 80, "overdue_days": 0, "loan_amount": 1000, "loan_time": "2025-02-01 10:00:00"},
            {"id": 2, "user_id": 20, "status": 62, "overdue_days": 45, "loan_amount": 2000, "loan_time": "2025-02-02 10:00:00"},
        ]
    ).to_sql("order_apply", seed, index=False)
    pd.DataFrame([{"id": 10, "monthly_income": 5000}, {"id": 20, "monthly_income": 2000}]).to_sql("user_info", seed, index=False)
    # `order_approve` has BOTH `user_id` (first column) and `order_id` (second column): must prioritize `order_id` -> `order_apply`!
    pd.DataFrame([{"user_id": 10, "order_id": 1, "approve_score": 85}, {"user_id": 20, "order_id": 2, "approve_score": 40}]).to_sql("order_approve", seed, index=False)
    pd.DataFrame([{"user_id": 10, "edu_level": 3}, {"user_id": 20, "edu_level": 1}]).to_sql("user_base_info", seed, index=False)
    pd.DataFrame(
        [
            {"user_id": 10, "call_duration": 80, "create_time": "2025-01-18 09:00:00"},
            {"user_id": 10, "call_duration": 40, "create_time": "2025-01-20 09:00:00"},
            {"user_id": 20, "call_duration": 15, "create_time": "2025-01-25 09:00:00"},
        ]
    ).to_sql("user_call_record", seed, index=False)
    seed.dispose()

    engine = sa.create_engine(f"sqlite:///{path}")
    db_explorer.set_engine(engine)
    try:
        p = copy.deepcopy(PROFILE)
        # No `table_patterns` configured: hub attribution relies purely on table columns & key specificity!
        p["master_tables"] = [
            {
                "table": "order_apply",
                "role": "订单主表",
                "key": "id",
                "link_columns": ["order_id"],
                "link_map": {"order_id": "order_apply.id"},
            },
            {
                "table": "user_info",
                "role": "用户主表",
                "key": "id",
                "bridge_from_sample": "order_apply.user_id = user_info.id",
                "link_columns": ["user_id"],
                "link_map": {"user_id": "user_info.id"},
            },
        ]
        p["joins"] = [{"table": "user_info", "left_key": "user_id", "right_key": "id"}]
        p["discovery"]["link_columns"] = ["order_id", "user_id"]

        candidates = db_project.discover_candidate_tables(p, refresh=True)
        plan = db_graph.plan_edges(p, candidates)
        by_table = {e["table"]: e for e in plan["edges"]}

        assert by_table["order_approve"]["hub"] == "order_apply"
        assert by_table["order_approve"]["hub_role"] == "订单主表"
        assert by_table["order_approve"]["join_expr"] == "order_apply.id = order_approve.order_id"
        assert by_table["order_approve"]["path"] == "order_apply -> order_approve"

        assert by_table["user_base_info"]["hub"] == "user_info"
        assert by_table["user_base_info"]["hub_role"] == "用户主表"
        assert by_table["user_base_info"]["join_expr"] == "order_apply.user_id = user_info.id -> user_info.id = user_base_info.user_id"
        assert by_table["user_base_info"]["path"] == "order_apply -> user_info -> user_base_info"

        assert by_table["user_call_record"]["hub"] == "user_info"
        assert by_table["user_call_record"]["join_expr"] == "order_apply.user_id = user_info.id -> user_info.id = user_call_record.user_id"
        assert by_table["user_call_record"]["path"] == "order_apply -> user_info -> user_call_record"
        assert by_table["user_call_record"]["sample_col"] == "user_id"

        fr = db_project.build_frame(p, edges=plan["edges"])
        df = fr["df"]
        assert df["order_approve.approve_score"].tolist() == [85, 40]
        assert df["user_base_info.edu_level"].tolist() == [3, 1]
        assert df["user_call_record.call_duration__sum_all"].tolist() == [120, 15]
    finally:
        db_explorer.set_engine(None)
        engine.dispose()
