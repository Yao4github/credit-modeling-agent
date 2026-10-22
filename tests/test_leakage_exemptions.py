"""Tests for per-feature leakage exemptions (manual confirmation) and their audit history."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.graph.workflow import run_credit_modeling_workflow
from app.tools.credit_metrics import DataInspectionInput, tool_inspect_dataset_and_check_leakage
from app.tools.data_registry import DataPointerRegistry
from app.tools.leakage_exemptions import (
    get_active_exemptions,
    get_exemption_state,
    grant_exemption,
    revoke_exemption,
)

TRAIN, OOT = "ds://ex_train", "ds://ex_oot"
NAME_HIT, STAT_HIT = "next_payment_due_days", "settle_flag_score"


@pytest.fixture
def ext(tmp_path):
    """External dataset with one name-rule hit and one statistical near-perfect label proxy."""
    reg = DataPointerRegistry.get_instance()
    rng = np.random.default_rng(11)
    for name, src, window in (
        ("ex_train", "ds://credit_a_card_train", ("2025-01-01", "2025-06-30")),
        ("ex_oot", "ds://credit_a_card_oot", ("2025-07-01", "2025-09-30")),
    ):
        df = reg.get_df(src).drop(columns=["post_loan_repay_amt_m1", "applicant_gender_code"]).copy()
        df[NAME_HIT] = df["credit_inq_cnt_3m"] * 2 + rng.normal(0, 1, len(df))
        df[STAT_HIT] = df["target_mob3_dpd30"] * 10.0 + rng.normal(0, 0.1, len(df))
        df.to_csv(tmp_path / f"{name}.csv", index=False)
        reg.register_from_file(f"ds://{name}", f"{name}.csv", "target_mob3_dpd30", window[0], window[1])
    return tmp_path


def _gov(train=TRAIN, oot=OOT):
    return tool_inspect_dataset_and_check_leakage(DataInspectionInput(train_uri=train, oot_uri=oot))["governance_audit"]


def test_hits_are_marked_exemptable_before_any_exemption(ext):
    gov = _gov()
    assert set(gov["quarantined_leakage_features"]) == {NAME_HIT, STAT_HIT}
    assert set(gov["exemptable_features"]) == {NAME_HIT, STAT_HIT}
    assert gov["exempted_features"] == [] and gov["ignored_exemptions"] == []


def test_name_rule_exemption_moves_feature_to_candidates_and_is_reported(ext):
    record = grant_exemption(TRAIN, NAME_HIT, "下次还款日在申请时点已知", "alice")
    assert record["original_reason"].startswith("命名规则") and record["operator"] == "alice"

    gov = _gov()
    assert NAME_HIT in gov["clean_candidate_features"] and NAME_HIT not in gov["quarantined_leakage_features"]
    assert [e["feature"] for e in gov["exempted_features"]] == [NAME_HIT]
    assert STAT_HIT in gov["quarantined_leakage_features"]  # other hits are unaffected

    state = run_credit_modeling_workflow("仅探查数据，检查穿越", "pytest-ex-1", train_uri=TRAIN, oot_uri=OOT)
    report = state["final_report_markdown"]
    assert "人工豁免字段" in report and "alice" in report and "下次还款日在申请时点已知" in report
    assert "风险提示" not in report  # name-rule exemption carries no statistical warning
    assert "可申请人工豁免" in report  # the remaining hit is marked exemptable


def test_statistical_exemption_adds_inflated_metrics_warning(ext):
    grant_exemption(TRAIN, STAT_HIT, "该字段为前置规则引擎评分，申请时点可得", "bob")
    state = run_credit_modeling_workflow("仅探查数据，检查穿越", "pytest-ex-2", train_uri=TRAIN, oot_uri=OOT)
    report = state["final_report_markdown"]
    assert "人工豁免字段" in report and "风险提示" in report and "虚高" in report


def test_blacklist_and_compliance_features_can_never_be_exempted():
    with pytest.raises(ValueError, match="blacklist"):
        grant_exemption("ds://credit_a_card_train", "post_loan_repay_amt_m1", "x", "alice")
    with pytest.raises(ValueError, match="compliance"):
        grant_exemption("ds://credit_a_card_train", "applicant_gender_code", "x", "alice")


def test_grant_validation_errors(ext):
    with pytest.raises(ValueError, match="reason"):
        grant_exemption(TRAIN, NAME_HIT, "   ", "alice")
    with pytest.raises(ValueError, match="operator"):
        grant_exemption(TRAIN, NAME_HIT, "ok", "")
    with pytest.raises(KeyError):
        grant_exemption("ds://does_not_exist", NAME_HIT, "ok", "alice")
    with pytest.raises(ValueError, match="not found in dataset"):
        grant_exemption(TRAIN, "no_such_feature", "ok", "alice")
    with pytest.raises(ValueError, match="nothing to exempt"):
        grant_exemption(TRAIN, "credit_inq_cnt_3m", "ok", "alice")
    grant_exemption(TRAIN, NAME_HIT, "ok", "alice")
    with pytest.raises(ValueError, match="already exempted"):
        grant_exemption(TRAIN, NAME_HIT, "ok again", "alice")


def test_revoke_requantines_and_history_is_append_only(ext):
    grant_exemption(TRAIN, NAME_HIT, "理由A", "alice")
    event = revoke_exemption(TRAIN, NAME_HIT, "复核后确认存在穿越", "carol")
    assert event["event"] == "revoke" and event["operator"] == "carol"
    assert NAME_HIT in _gov()["quarantined_leakage_features"]
    assert get_active_exemptions(TRAIN) == {}

    history = get_exemption_state(TRAIN)["history"]
    assert [(h["event"], h["operator"]) for h in history] == [("grant", "alice"), ("revoke", "carol")]
    with pytest.raises(KeyError):
        revoke_exemption(TRAIN, NAME_HIT, "again", "carol")


def test_corrupt_store_means_no_exemptions_and_is_moved_aside(ext):
    store = ext / "leakage_exemptions.json"
    store.write_text("{not json", encoding="utf-8")
    assert get_active_exemptions(TRAIN) == {}
    assert NAME_HIT in _gov()["quarantined_leakage_features"]  # safe default: still quarantined

    grant_exemption(TRAIN, NAME_HIT, "理由", "alice")
    assert (ext / "leakage_exemptions.json.corrupt").read_text(encoding="utf-8") == "{not json"
    assert get_active_exemptions(TRAIN)[NAME_HIT]["operator"] == "alice"


def test_stale_exemption_is_ignored_once_feature_is_blacklisted(ext):
    grant_exemption(TRAIN, STAT_HIT, "理由", "alice")
    assert STAT_HIT not in _gov()["quarantined_leakage_features"]

    extra = [{"doc_id": "EXT-LEAK", "category": "data_catalog", "title": "新增穿越字段", "content": "x",
              "metadata": {"leakage_blacklist": [STAT_HIT]}}]
    (ext / "knowledge_extra.json").write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")

    gov = _gov()
    assert STAT_HIT in gov["quarantined_leakage_features"] and STAT_HIT not in gov["exemptable_features"]
    assert gov["exempted_features"] == []
    assert gov["ignored_exemptions"][0]["feature"] == STAT_HIT and "黑名单" in gov["ignored_exemptions"][0]["why"]

    state = run_credit_modeling_workflow("仅探查数据，检查穿越", "pytest-ex-3", train_uri=TRAIN, oot_uri=OOT)
    assert "失效豁免" in state["final_report_markdown"]


def test_exemption_api_flow_and_status_codes(ext):
    client = TestClient(app)
    body = {"uri": TRAIN, "feature": NAME_HIT, "reason": "申请时点可得", "operator": "alice"}

    ok = client.post("/api/v1/datasets/exemptions", json=body)
    assert ok.status_code == 200 and ok.json()["feature"] == NAME_HIT
    assert client.post("/api/v1/datasets/exemptions", json=body).status_code == 400  # duplicate
    assert client.post("/api/v1/datasets/exemptions", json={**body, "uri": "ds://nope"}).status_code == 404
    assert client.post("/api/v1/datasets/exemptions", json={**body, "feature": "ghost"}).status_code == 400
    assert client.post("/api/v1/datasets/exemptions", json={**body, "reason": ""}).status_code == 422
    assert client.post("/api/v1/datasets/exemptions", json={**body, "reason": "  "}).status_code == 400
    blk = client.post(
        "/api/v1/datasets/exemptions",
        json={**body, "uri": "ds://credit_a_card_train", "feature": "applicant_gender_code"},
    )
    assert blk.status_code == 400

    state = client.get("/api/v1/datasets/exemptions", params={"uri": TRAIN}).json()
    assert [r["feature"] for r in state["active"]] == [NAME_HIT] and len(state["history"]) == 1

    params = {"uri": TRAIN, "feature": NAME_HIT, "reason": "复核撤销", "operator": "carol"}
    assert client.delete("/api/v1/datasets/exemptions", params=params).status_code == 200
    assert client.delete("/api/v1/datasets/exemptions", params=params).status_code == 404
    assert client.delete("/api/v1/datasets/exemptions", params={"uri": TRAIN}).status_code == 422
    assert client.get("/api/v1/datasets/exemptions", params={"uri": TRAIN}).json()["active"] == []
