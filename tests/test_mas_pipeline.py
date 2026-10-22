"""Comprehensive Test Suite for Credit-Modeling Multi-Agent System."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.server import app
from app.graph.workflow import run_credit_modeling_workflow
from app.tools.data_registry import DataPointerRegistry
from app.tools.rag_retriever import HybridCreditRAG
from eval.evaluators import run_benchmark_suite


def test_data_pointer_registry_and_synthetic_warehouse() -> None:
    registry = DataPointerRegistry.get_instance()
    datasets = registry.list_datasets()
    uris = {d["uri"] for d in datasets}
    assert "ds://credit_a_card_train" in uris
    assert "ds://credit_a_card_oot" in uris

    train_df = registry.get_df("ds://credit_a_card_train")
    assert len(train_df) == 4000
    assert "target_mob3_dpd30" in train_df.columns
    assert "post_loan_repay_amt_m1" in train_df.columns


def test_hybrid_rag_retriever() -> None:
    rag = HybridCreditRAG()
    hits = rag.retrieve("什么是数据穿越 post_loan_repay_amt_m1", top_k=2)
    assert len(hits) >= 1
    assert hits[0]["doc_id"] == "CAT-002"


def test_end_to_end_full_modeling_with_self_reflection() -> None:
    state = run_credit_modeling_workflow(
        user_query="帮我针对首贷客群全流程构建一张A卡评分模型，要求 KS >= 0.35，PSI <= 0.10，目标通过率 75%，并输出合规报告",
        session_id="pytest-full-001",
    )
    assert state["intent_type"] == "full_modeling"
    # Verify data leakage and compliance quarantine
    assert "post_loan_repay_amt_m1" in state["quarantined_leakage_features"]
    assert "applicant_gender_code" in state["quarantined_compliance_features"]

    # Verify Self-Reflection was triggered to catch OOT PSI drift on `night_apply_ratio_7d`
    assert state["reflection_count"] == 1
    assert "night_apply_ratio_7d" in state["excluded_by_reflection"]
    assert "night_apply_ratio_7d" not in state["selected_features"]

    # Verify final model passed all validation and compliance gates
    assert state["validation_result"]["validation_passed"] is True
    assert state["validation_result"]["metrics"]["oot_ks"] >= 0.35
    assert state["validation_result"]["metrics"]["score_psi"] <= 0.10
    assert state["compliance_passed"] is True


def test_agent_eval_benchmark_suite_achieves_100_percent() -> None:
    results = run_benchmark_suite()
    summary = results["summary"]
    assert summary["task_completion_rate"] == 1.0
    assert summary["tool_calling_ast_accuracy"] == 1.0
    assert summary["hallucination_rate"] == 0.0


def test_fastapi_microservice_endpoints() -> None:
    client = TestClient(app)

    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    resp_ds = client.get("/api/v1/datasets")
    assert resp_ds.status_code == 200
    assert len(resp_ds.json()["datasets"]) >= 2

    resp_run = client.post(
        "/api/v1/modeling/run",
        json={
            "query": "请帮我仅探查首贷A卡数据集，执行防穿越检查和合规敏感字段筛查",
            "session_id": "test-api-audit",
        },
    )
    assert resp_run.status_code == 200
    body = resp_run.json()
    assert body["intent_type"] == "data_audit_only"
    assert body["compliance_passed"] is True
