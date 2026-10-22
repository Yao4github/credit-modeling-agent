"""Tests for config-driven multi-table mining (SQLite stands in for MySQL)."""

from __future__ import annotations

import copy
import json
import random

import httpx
import pandas as pd
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from app.api.server import app
from app.core.config import LLMConfig
from app.core.llm import LLMClient, reset_llm_client
from app.tools import db_explorer, db_project

PROFILE = {
    "name": "unit-test",
    "sample": {"table": "order_apply", "key": "id", "user_key": "user_id", "date_col": "loan_time"},
    "joins": [{"table_pattern": "user_info*", "left_key": "user_id", "right_key": "id"}],
    "params": {"disbursed_status": 62, "settled_status": 80, "bad_overdue_days": 30, "good_max_overdue_days": 0},
    "population": {"all": [{"col": "status", "op": ">=", "value": "$disbursed_status"}]},
    "label": {
        "name": "risk_label",
        "bad": {"all": [{"col": "overdue_days", "op": ">=", "value": "$bad_overdue_days"}]},
        "good": {"all": [
            {"col": "status", "op": "==", "value": "$settled_status"},
            {"col": "overdue_days", "op": "<=", "value": "$good_max_overdue_days"},
        ]},
    },
    "performance_checks": [{"col": "repay_count", "higher_is": "good"}],
    "warnings": ["selection bias"],
    "exclude": [
        {"reason": "post-loan", "patterns": ["status", "overdue_days"]},
        {"reason": "pii", "patterns": ["phone", "user_info.nickname"]},
    ],
    "categories": {
        "use_llm": True,
        "rules": [
            {"name": "收入", "name_patterns": ["income*"], "comment_keywords": ["财务"]},
            {"name": "额度", "name_patterns": ["*amount*"]},
        ],
        "table_fallback": {"order_apply": "订单-其他", "user_info": "用户-其他"},
    },
    "discovery": {
        "link_columns": ["user_id", "order_no"],
        "exclude_table_patterns": ["*_log"],
        "post_event_table_patterns": ["collection*"],
    },
}


@pytest.fixture
def db(tmp_path, monkeypatch):
    rng = random.Random(11)
    orders, users = [], []
    for i in range(1, 701):
        x = rng.gauss(0, 1)
        bad = x + rng.gauss(0, 1) > 0.4
        status = 62 if bad else rng.choice([80, 80, 80, 62])
        overdue = rng.randint(30, 60) if bad else ((-rng.randint(1, 5) if i % 7 == 0 else 0) if status == 80 else rng.randint(1, 10))  # <0 = early settle; 62 + <30d = gray
        if i % 10 == 0:
            status, overdue = 41, 0  # not disbursed: outside the population
        orders.append(
            {
                "id": i,
                "user_id": i,
                "status": status,
                "overdue_days": overdue,
                "repay_count": rng.randint(0, 1) if bad else rng.randint(0, 3),
                "loan_amount": 1000 + int(300 * x) + rng.randint(0, 50),
                "noise_amount": rng.gauss(0, 1),
                "loan_time": f"2025-{1 + (i - 1) // 60:02d}-{1 + i % 28:02d} 10:00:00",
            }
        )
        users.append({"id": i, "income_level": rng.gauss(0, 1), "score": rng.randint(300, 700), "phone": f"080{i:08d}", "nickname": f"u{i}"})
    path = tmp_path / "src.db"
    seed = sa.create_engine(f"sqlite:///{path}")
    pd.DataFrame(orders).to_sql("order_apply", seed, index=False)
    pd.DataFrame(users).to_sql("user_info", seed, index=False)
    pd.DataFrame({"user_id": [1], "x": [1]}).to_sql("order_repay_plan", seed, index=False)
    pd.DataFrame({"user_id": [1], "order_no": ["a"]}).to_sql("collection_record", seed, index=False)
    pd.DataFrame({"user_id": [1]}).to_sql("sys_log", seed, index=False)
    pd.DataFrame({"a": [1]}).to_sql("unrelated", seed, index=False)
    pd.DataFrame({"k": [1, 1, 2]}).to_sql("dup_keys", seed, index=False)
    seed.dispose()

    engine = sa.create_engine(f"sqlite:///{path}")
    db_explorer.set_engine(engine)
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(PROFILE, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv(db_project.PROFILE_ENV, str(profile_file))
    yield engine
    db_explorer.set_engine(None)
    engine.dispose()


def _profile(**changes):
    p = copy.deepcopy(PROFILE)
    p.update(changes)
    return p


# ---- profile validation / conditions --------------------------------------------------------------
def test_missing_profile_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv(db_project.PROFILE_ENV, str(tmp_path / "nope.json"))
    assert not db_project.is_configured()
    with pytest.raises(db_project.ProfileNotConfigured):
        db_project.load_profile()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["sample"].pop("key"),
        lambda p: p["label"].pop("good"),
        lambda p: p["label"]["bad"]["all"][0].update(op="LIKE"),
        lambda p: p["label"]["bad"]["all"][0].pop("value"),
        lambda p: p["joins"][0].pop("right_key"),
    ],
)
def test_invalid_profile_rejected(tmp_path, mutate):
    p = _profile()
    mutate(p)
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(ValueError):
        db_project.load_profile(f)


def test_condition_mask_params_and_operators():
    df = pd.DataFrame({"a": [1, 5, 10, None], "b": ["x", "y", None, "x"]})
    params = {"hi": 5}
    m = lambda c: db_project.cond_to_mask(c, df, params).tolist()  # noqa: E731
    assert m({"col": "a", "op": ">=", "value": "$hi"}) == [False, True, True, False]
    assert m({"col": "a", "op": "!=", "value": 5}) == [True, False, True, False]  # NULL never matches
    assert m({"col": "b", "op": "in", "value": ["x"]}) == [True, False, False, True]
    assert m({"col": "b", "op": "is_null"}) == [False, False, True, False]
    assert m({"any": [{"col": "a", "op": "==", "value": 1}, {"col": "b", "op": "==", "value": "y"}]}) == [True, True, False, False]
    with pytest.raises(ValueError):
        m({"col": "a", "op": ">", "value": "$undefined"})


# ---- frame ----------------------------------------------------------------------------------------
def test_build_frame_label_population_and_exclusions(db):
    fr = db_project.build_frame(PROFILE)
    df, li = fr["df"], fr["label_info"]
    assert set(df["risk_label"].unique()) == {0, 1}
    assert li["bad_count"] > 0 and li["good_count"] > 0 and li["dropped_unlabeled"] > 0  # gray rows dropped
    # population: status 41 rows never read; label-rule / excluded columns never become features
    assert li["population_rows"] < 700
    for forbidden in ("status", "overdue_days", "user_info.phone", "user_info.nickname", "id", "user_id"):
        assert forbidden not in df.columns, forbidden
    assert "user_info.income_level" in df.columns and "loan_amount" in df.columns
    assert "repay_count" in df.columns  # point-in-time history count: a legitimate feature, not an outcome
    assert fr["excluded"]["status"] == "post-loan" and fr["excluded"]["user_info.phone"] == "pii"
    assert fr["joins"][0]["match_rate"] == 1.0
    assert "selection bias" in fr["warnings"]


def test_excluded_columns_are_not_read_from_the_database(db):
    statements = []
    sa.event.listen(db, "before_cursor_execute", lambda conn, cur, stmt, *a: statements.append(stmt))
    db_project.build_frame(PROFILE)
    select = next(s for s in statements if "order_apply" in s and "SELECT" in s.upper())
    assert "phone" not in select and "nickname" not in select


def test_param_override_changes_label_and_direction_check_passes(db):
    base = db_project.build_frame(PROFILE)
    strict = db_project.build_frame(PROFILE, {"bad_overdue_days": 45})
    assert strict["label_info"]["bad_count"] < base["label_info"]["bad_count"]
    chk = base["label_info"]["direction_checks"][0]
    assert chk["col"] == "repay_count" and chk["ok"] and chk["good_mean"] > chk["bad_mean"]


def test_early_settlement_counts_as_good(db):
    """status=80 with overdue_days<0 (settled early) is good; a stricter good rule (<= -1) keeps only those."""
    loose = db_project.build_frame(PROFILE)["label_info"]["good_count"]
    early_only = db_project.build_frame(PROFILE, {"good_max_overdue_days": -1})["label_info"]["good_count"]
    assert loose > early_only > 0


def test_direction_check_failure_warns(db):
    p = _profile(performance_checks=[{"col": "repay_count", "higher_is": "bad"}])
    fr = db_project.build_frame(p)
    assert not fr["label_info"]["direction_checks"][0]["ok"]
    assert any("方向校验未通过" in w for w in fr["warnings"])


def test_single_class_label_rejected(db):
    with pytest.raises(ValueError, match="单一类别"):
        db_project.build_frame(PROFILE, {"bad_overdue_days": 9999})


def test_unknown_column_in_rules_rejected(db):
    p = _profile()
    p["label"]["bad"] = {"all": [{"col": "no_such_col", "op": ">", "value": 1}]}
    with pytest.raises(ValueError, match="no_such_col"):
        db_project.build_frame(p)


def test_non_unique_join_key_rejected(db):
    p = _profile(joins=[{"table": "dup_keys", "left_key": "user_id", "right_key": "k"}])
    with pytest.raises(ValueError, match="重复"):
        db_project.build_frame(p)


def test_missing_join_table_warns_instead_of_failing(db):
    p = _profile(joins=[{"table_pattern": "does_not_exist*", "left_key": "user_id", "right_key": "id"}])
    fr = db_project.build_frame(p)
    assert any("没有表匹配" in w for w in fr["warnings"])


# ---- categories / mining --------------------------------------------------------------------------
def test_categorize_rules_and_table_fallback(db):
    fr = db_project.build_frame(PROFILE)
    cats = db_project.categorize(PROFILE, list(fr["col_meta"]), fr["col_meta"])
    assert cats["user_info.income_level"] == "收入"
    assert cats["loan_amount"] == "额度"
    assert cats["user_info.score"] == "用户-其他"  # fallback by source table (LLM disabled in tests)


def test_llm_categories_are_validated_against_allowed_names(db):
    def handler(req):
        body = json.dumps({"user_info.score": "额度", "loan_amount": "不存在的类别"})
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": body}}], "usage": {}})

    cfg = LLMConfig("deepseek", "https://x", "m", "k", 5, 0, True)
    reset_llm_client(LLMClient(cfg, transport=httpx.MockTransport(handler)))
    meta = {
        "user_info.score": {"table": "user_info", "column": "score", "comment": None},
        "plain_col": {"table": "order_apply", "column": "plain_col", "comment": None},
    }
    cats = db_project.categorize(PROFILE, ["user_info.score", "plain_col"], meta)
    assert cats["user_info.score"] == "额度"  # valid LLM answer accepted
    assert cats["plain_col"] == "订单-其他"  # missing from the answer -> table fallback


def test_mine_project_reports_categories_and_finds_signal(db):
    fr = db_project.build_frame(PROFILE)
    mined = db_project.mine_project(PROFILE, fr)
    by_name = {m["feature"]: m for m in mined["feature_metrics"]}
    assert by_name["loan_amount"]["category"] == "额度" and by_name["loan_amount"]["source_table"] == "order_apply"
    assert by_name["loan_amount"]["iv"] > by_name["noise_amount"]["iv"]
    assert "loan_amount" in mined["suggested_features"]
    assert mined["not_suggested"]["status"] == "post-loan"  # config exclusions are reported
    assert {c["category"] for c in mined["category_summary"]} >= {"额度", "收入"}
    assert mined["split"]["strict_oot"] is True


# ---- discovery ------------------------------------------------------------------------------------
def test_discovery_finds_link_tables_and_flags_post_event(db):
    found = {c["table"]: c for c in db_project.discover_candidate_tables(PROFILE)}
    assert set(found) == {"order_repay_plan", "collection_record"}  # not: base / joined / *_log / unrelated
    assert found["collection_record"]["post_event"] and not found["order_repay_plan"]["post_event"]
    ordered = [c["table"] for c in db_project.discover_candidate_tables(PROFILE)]
    assert ordered == ["order_repay_plan", "collection_record"]  # post-event tables last


def _discovery_profile(**discovery):
    p = _profile()
    p["discovery"] = {**p["discovery"], **discovery}
    return p


def test_candidate_tables_are_categorized_by_rules_and_post_event(db):
    p = _discovery_profile(
        table_categories=[{"name": "还款计划", "table_patterns": ["order_repay*"]}],
        post_event_category="贷后",
    )
    found = {c["table"]: c for c in db_project.discover_candidate_tables(p, refresh=True)}
    assert found["order_repay_plan"]["category"] == "还款计划"
    assert found["collection_record"]["category"] == "贷后"  # post-event wins over any rule


def test_uncategorized_tables_fall_back_to_default_category(db):
    found = {c["table"]: c for c in db_project.discover_candidate_tables(_discovery_profile(), refresh=True)}
    assert found["order_repay_plan"]["category"] == db_project.DEFAULT_TABLE_CATEGORY
    assert found["collection_record"]["category"] == db_project.POST_EVENT_CATEGORY


def test_table_category_by_comment_keyword():
    item = {"table": "xyz", "comment": "用户活体检查日志"}
    rules = [{"name": "认证", "table_patterns": ["auth*"], "comment_keywords": ["活体"]}]
    assert db_project._table_rule_category(item, rules) == "认证"
    assert db_project._table_rule_category({"table": "xyz", "comment": ""}, rules) is None


def test_llm_table_categories_are_validated(db):
    def handler(req):
        body = json.dumps({"order_repay_plan": "还款计划", "collection_record": "乱写的类别"})
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": body}}], "usage": {}})

    reset_llm_client(LLMClient(LLMConfig("deepseek", "https://x", "m", "k", 5, 0, True), transport=httpx.MockTransport(handler)))
    p = _discovery_profile(table_categories=[{"name": "还款计划", "table_patterns": ["nomatch*"]}], post_event_table_patterns=[])
    found = {c["table"]: c for c in db_project.discover_candidate_tables(p, refresh=True)}
    assert found["order_repay_plan"]["category"] == "还款计划"  # valid LLM answer accepted
    assert found["collection_record"]["category"] == db_project.DEFAULT_TABLE_CATEGORY  # invalid answer ignored


def test_invalid_table_category_rule_rejected(tmp_path):
    p = _discovery_profile(table_categories=[{"table_patterns": ["x*"]}])
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(ValueError, match="table_categories"):
        db_project.load_profile(f)


# ---- API ------------------------------------------------------------------------------------------
def _sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        name, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        out.append((name, data))
    return out


def test_api_project_info_and_mine_stream(db):
    client = TestClient(app)
    info = client.get("/api/v1/db/project").json()
    assert info["configured"] and info["base_table"] == "order_apply" and info["params"]["bad_overdue_days"] == 30

    resp = client.post("/api/v1/db/project/mine", json={"params": {"bad_overdue_days": 30}})
    events = _sse(resp.text)
    assert events[-1][0] == "result" and "error" not in [n for n, _ in events]
    agents = [d["agent"] for n, d in events if n == "message"]
    assert agents == ["项目配置", "表发现", "取样与标签", "指标挖掘与分类", "报告生成"]
    report = events[-1][1]["report_markdown"]
    for heading in ("特征分类汇总", "标签口径与总体", "候选明细表", "方向校验", "已按主题自动分类"):
        assert heading in report
    assert "| 类别 | 表数量 | 用途 | 代表表 |" in report and "仅可用于定义标签" in report


def test_api_project_mine_reports_bad_params_as_error_event(db):
    resp = TestClient(app).post("/api/v1/db/project/mine", json={"params": {"bad_overdue_days": 99999}})
    events = _sse(resp.text)
    assert events[-1][0] == "error" and "单一类别" in events[-1][1]["message"]


def test_api_without_profile_or_db(monkeypatch, tmp_path):
    client = TestClient(app)
    monkeypatch.setenv(db_project.PROFILE_ENV, str(tmp_path / "missing.json"))
    assert client.get("/api/v1/db/project").json() == {"configured": False}
    db_explorer.set_engine(sa.create_engine("sqlite://"))
    try:
        assert client.post("/api/v1/db/project/mine", json={}).status_code == 503
    finally:
        db_explorer.set_engine(None)
