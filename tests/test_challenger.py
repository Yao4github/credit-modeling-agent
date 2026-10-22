"""Tests for the Champion-Challenger (LR vs monotonic GBDT) comparison."""

from __future__ import annotations

from app.graph.workflow import run_credit_modeling_workflow
from app.tools.credit_metrics import (
    CHALLENGER_KS_MARGIN,
    ChallengerTrainInput,
    tool_train_challenger_model,
)

_FULL_QUERY = "帮我针对首贷客群全流程构建一张A卡评分模型，要求 KS >= 0.35，PSI <= 0.10，目标通过率 75%，并输出合规报告"


def _run_full(session_id: str):
    return run_credit_modeling_workflow(user_query=_FULL_QUERY, session_id=session_id)


def test_challenger_trained_once_after_gate_passes_and_champion_unchanged():
    state = _run_full("pytest-champ-001")
    names = [c["tool_name"] for c in state["tool_calls"]]
    assert names.count("tool_train_challenger_model") == 1
    assert names.index("tool_train_challenger_model") > max(
        i for i, n in enumerate(names) if n == "tool_evaluate_scorecard_and_strategy"
    )

    ch = state["challenger_result"]
    assert ch["champion_model_id"] == state["model_training_result"]["model_id"]
    assert ch["champion_metrics"]["oot_ks"] == state["validation_result"]["metrics"]["oot_ks"]
    assert ch["ks_gap"] == round(ch["challenger_metrics"]["oot_ks"] - ch["champion_metrics"]["oot_ks"], 4)
    assert ch["challenger_significantly_better"] == (ch["ks_gap"] >= CHALLENGER_KS_MARGIN)
    assert {d["feature"] for d in ch["feature_importance"]} == set(state["selected_features"])
    assert state["validation_result"]["validation_passed"] is True


def test_challenger_is_reproducible():
    state = _run_full("pytest-champ-002")
    inp = ChallengerTrainInput(
        champion_model_id=state["model_training_result"]["model_id"],
        train_woe_uri=state["dataset_refs"]["train_woe_uri"],
        oot_woe_uri=state["dataset_refs"]["oot_woe_uri"],
        selected_features=state["selected_features"],
    )
    assert tool_train_challenger_model(inp) == tool_train_challenger_model(inp)


def test_report_contains_challenger_section():
    state = _run_full("pytest-champ-003")
    report = state["final_report_markdown"]
    ch = state["challenger_result"]
    assert "冠军挑战者对比" in report
    assert str(ch["challenger_metrics"]["oot_ks"]) in report
    assert str(ch["ks_gap"]) in report


def test_no_challenger_when_gate_never_passes():
    state = run_credit_modeling_workflow(
        user_query="全流程构建A卡评分模型，要求 KS >= 0.99，PSI <= 0.10",
        session_id="pytest-champ-004",
        max_reflections=0,
    )
    assert state["validation_result"]["validation_passed"] is False
    assert "challenger_result" not in state
    assert all(c["tool_name"] != "tool_train_challenger_model" for c in state["tool_calls"])
