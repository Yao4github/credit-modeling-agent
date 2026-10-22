"""Tests for B-Card (behavior) and C-Card (collection) support."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agents.specialists import orchestrator_node
from app.api.server import app
from app.core.cards import detect_card_type, get_profile
from app.core.llm import reset_llm_client
from app.graph.workflow import run_credit_modeling_workflow
from app.tools.data_registry import DataPointerRegistry
from tests.test_llm import _client, _ok

_B_QUERY = "帮我针对存量客群全流程构建一张B卡行为评分模型，目标通过率 75%"
_C_QUERY = "针对逾期账户全流程开发C卡催收评分模型，要求 PSI <= 0.10，目标通过率 70%"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("帮我构建A卡评分模型", "A"),
        ("请做一个行为评分模型", "B"),
        ("开发B卡", "B"),
        ("逾期账户催收评分建模", "C"),
        ("C卡KS要求0.3", "C"),
        ("帮我全流程建模", "A"),
    ],
)
def test_detect_card_type(query, expected):
    assert detect_card_type(query) == expected


def test_unknown_card_falls_back_to_a():
    assert get_profile("Z").card_type == "A"
    assert get_profile(None).card_type == "A"


def test_b_and_c_datasets_registered_with_expected_targets():
    reg = DataPointerRegistry.get_instance()
    for card in ("B", "C"):
        p = get_profile(card)
        for uri in (p.train_uri, p.oot_uri):
            assert reg.get_meta(uri).target_col == p.target_col
            assert 0.05 < reg.get_meta(uri).bad_rate < 0.6


@pytest.mark.parametrize(
    "card,query,leak,drift,title,rate_label",
    [
        ("B", _B_QUERY, "fut_m1_overdue_amt", "login_freq_30d", "行为评分卡 (B-Card)", "目标通过率"),
        ("C", _C_QUERY, "post_collection_repay_amt_d7", "call_connect_rate_7d", "催收评分卡 (C-Card)", "目标策略覆盖率"),
    ],
)
def test_end_to_end_b_and_c_cards(card, query, leak, drift, title, rate_label):
    state = run_credit_modeling_workflow(query, session_id=f"pytest-card-{card}")
    p = get_profile(card)
    assert state["card_type"] == card
    assert state["dataset_refs"]["train_uri"] == p.train_uri
    assert state["constraints"]["ks_min"] == p.default_ks_min
    assert leak in state["quarantined_leakage_features"]
    assert "applicant_gender_code" in state["quarantined_compliance_features"]
    assert state["reflection_count"] == 1 and drift in state["excluded_by_reflection"]
    assert drift not in state["selected_features"]
    assert state["validation_result"]["validation_passed"] is True
    assert state["compliance_passed"] is True
    assert "challenger_result" in state
    report = state["final_report_markdown"]
    assert title in report and rate_label in report


def test_explicit_card_type_overrides_semantics_and_api_returns_it():
    state = run_credit_modeling_workflow("帮我全流程构建评分模型", session_id="pytest-card-force", card_type="B")
    assert state["card_type"] == "B"
    assert state["dataset_refs"]["train_uri"] == "ds://credit_b_card_train"

    client = TestClient(app)
    resp = client.post(
        "/api/v1/modeling/run",
        json={"query": "仅探查数据，检查穿越", "session_id": "api-card-c", "card_type": "C"},
    )
    assert resp.status_code == 200 and resp.json()["card_type"] == "C"
    assert client.post("/api/v1/modeling/run", json={"card_type": "Z"}).status_code == 422


def test_llm_card_type_validated_against_whitelist():
    reset_llm_client(_client(lambda r: _ok(json.dumps({"intent": "full_modeling", "card_type": "C"}))))
    assert orchestrator_node({"user_query": "帮我建一个模型"})["card_type"] == "C"

    reset_llm_client(_client(lambda r: _ok(json.dumps({"intent": "full_modeling", "card_type": "Z"}))))
    assert orchestrator_node({"user_query": "帮我建一个模型"})["card_type"] == "A"

    reset_llm_client(_client(lambda r: _ok(json.dumps({"intent": "full_modeling", "card_type": "C"}))))
    assert orchestrator_node({"user_query": "帮我建一个模型", "card_type": "B"})["card_type"] == "B"

    reset_llm_client(_client(lambda r: httpx.Response(401), max_retries=0))
    assert orchestrator_node({"user_query": "做一张催收评分卡"})["card_type"] == "C"
