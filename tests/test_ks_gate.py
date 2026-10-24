"""Tests for the Train/OOT KS-degradation gate and the no-actionable-culprit rollback stop."""

from __future__ import annotations

from app.graph.workflow import run_credit_modeling_workflow
from app.tools.credit_metrics import ModelValidationInput, tool_evaluate_scorecard_and_strategy

_QUERY = "帮我针对首贷客群全流程构建一张A卡评分模型，要求 KS >= 0.35，PSI <= 0.10，目标通过率 75%，并输出合规报告"


def _evaluate(model_id: str, **kw):
    return tool_evaluate_scorecard_and_strategy(ModelValidationInput(model_id=model_id, ks_threshold=0.30, **kw))


def test_ks_drop_gate_threshold_and_boundary():
    state = run_credit_modeling_workflow(_QUERY, session_id="pytest-ksgate-1")
    model_id = state["model_training_result"]["model_id"]
    drop = state["validation_result"]["metrics"]["ks_degradation"]
    assert state["constraints"]["ks_drop_max"] == 0.05 and 0 < drop < 0.05

    assert _evaluate(model_id)["validation_passed"] is True
    assert _evaluate(model_id, ks_drop_threshold=drop)["validation_passed"] is True  # boundary: equal passes

    strict = _evaluate(model_id, ks_drop_threshold=round(drop - 0.001, 4))
    assert strict["validation_passed"] is False
    assert any("KS degradation" in a for a in strict["anomalies"])


def test_report_shows_gate_conclusion_when_passed():
    state = run_credit_modeling_workflow(_QUERY, session_id="pytest-ksgate-2")
    report = state["final_report_markdown"]
    assert "跨期门禁结论**：`PASSED`" in report and "ks_degradation" in report


def test_no_futile_rollback_when_failure_has_no_actionable_culprit():
    state = run_credit_modeling_workflow(
        "全流程构建A卡评分模型，要求 KS >= 0.99，PSI <= 0.10", session_id="pytest-ksgate-3", max_reflections=2
    )
    names = [c["tool_name"] for c in state["tool_calls"]]
    # pass 1 fails on the drifting trap feature (rollback 1); pass 2 fails on KS only with no culprit feature, so the
    # validator switches binning (rollback 2); pass 3 still fails and the budget is exhausted -> report.
    assert names.count("tool_evaluate_scorecard_and_strategy") == 3
    assert state["binning_strategy"] == "quantile10" and state["reflection_count"] == 2
    assert state["stop_reflection"] is True
    assert state["validation_result"]["validation_passed"] is False
    assert "challenger_result" not in state

    report = state["final_report_markdown"]
    assert "跨期门禁结论**：`FAILED`" in report
    assert "OOT KS" in report
