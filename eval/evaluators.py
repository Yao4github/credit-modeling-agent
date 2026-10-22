"""Quantitative Evaluators for the Credit Risk Modeling Multi-Agent System (Agent Eval).

Metrics evaluated:
1. Tool Calling AST Accuracy (Sequence Match + Argument Subset AST Match)
2. Numerical & Knowledge Hallucination Rate (Zero-Error Sandbox Metric Verification + RAG Faithfulness)
3. End-to-End Task Completion Rate (Constraint Satisfaction & Self-Reflection Verification)
4. Latency (P50 / P95 ms) & Token Cost Tracking
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from app.core.cards import get_profile
from app.graph.state import CreditModelingState
from app.graph.workflow import run_credit_modeling_workflow


def evaluate_tool_calling_ast(
    state: CreditModelingState,
    benchmark_case: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluates Tool Calling Sequence accuracy and AST Argument matching."""
    actual_calls = state.get("tool_calls", [])
    actual_names = [c["tool_name"] for c in actual_calls]
    expected_names = benchmark_case.get("expected_tools", [])

    seq_match = actual_names == expected_names

    # Evaluate argument subset AST accuracy
    expected_arg_subsets: Dict[str, Dict[str, Any]] = benchmark_case.get("expected_arg_subsets", {})
    total_arg_checks = 0
    matched_arg_checks = 0
    arg_mismatches: List[str] = []

    for tool_name, expected_kv in expected_arg_subsets.items():
        matching_records = [c for c in actual_calls if c["tool_name"] == tool_name]
        if not matching_records:
            total_arg_checks += len(expected_kv)
            arg_mismatches.append(f"Missing tool call: {tool_name}")
            continue
        # Inspect latest call of that tool (post-reflection if applicable)
        last_args = matching_records[-1]["tool_args"]
        for k, exp_v in expected_kv.items():
            total_arg_checks += 1
            act_v = last_args.get(k)
            if isinstance(exp_v, float) and isinstance(act_v, (int, float)):
                if abs(float(act_v) - exp_v) < 1e-4:
                    matched_arg_checks += 1
                else:
                    arg_mismatches.append(f"{tool_name}.{k}: expected {exp_v}, got {act_v}")
            elif act_v == exp_v:
                matched_arg_checks += 1
            else:
                arg_mismatches.append(f"{tool_name}.{k}: expected {exp_v}, got {act_v}")

    arg_accuracy = matched_arg_checks / total_arg_checks if total_arg_checks > 0 else 1.0
    combined_score = 0.5 * (1.0 if seq_match else 0.0) + 0.5 * arg_accuracy

    return {
        "sequence_exact_match": seq_match,
        "actual_tools": actual_names,
        "expected_tools": expected_names,
        "arg_ast_accuracy": round(arg_accuracy, 4),
        "tool_calling_score": round(combined_score, 4),
        "arg_mismatches": arg_mismatches,
    }


def evaluate_hallucination_rate(
    state: CreditModelingState,
    benchmark_case: Dict[str, Any],
) -> Dict[str, Any]:
    """Verifies zero numerical hallucination between Sandbox JSON outputs and final Markdown report."""
    report = state.get("final_report_markdown", "")
    intent = state.get("intent_type", "")
    checks_total = 0
    hallucinations_found = 0
    details: List[str] = []

    # 1. Check required RAG keywords if specified
    for kw in benchmark_case.get("must_contain_keywords", []):
        checks_total += 1
        if str(kw) not in report:
            hallucinations_found += 1
            details.append(f"Missing authoritative RAG keyword/citation: {kw}")

    # 2. Check numerical consistency for full_modeling
    if intent == "full_modeling":
        val_metrics = state.get("validation_result", {}).get("metrics", {})
        strat = state.get("validation_result", {}).get("cutoff_strategy", {})
        samp = state.get("data_inspection_result", {}).get("sample_summary", {})

        ground_truth_numbers = {
            "train_ks": val_metrics.get("train_ks"),
            "oot_ks": val_metrics.get("oot_ks"),
            "train_auc": val_metrics.get("train_auc"),
            "oot_auc": val_metrics.get("oot_auc"),
            "score_psi": val_metrics.get("score_psi"),
            "recommended_cutoff_score": strat.get("recommended_cutoff_score"),
            "post_cutoff_bad_rate": strat.get("post_cutoff_bad_rate"),
            "train_rows": samp.get("train_rows"),
            "oot_rows": samp.get("oot_rows"),
        }
        challenger = state.get("challenger_result") or {}
        if challenger:
            ground_truth_numbers["challenger_oot_ks"] = challenger.get("challenger_metrics", {}).get("oot_ks")
            ground_truth_numbers["challenger_ks_gap"] = challenger.get("ks_gap")
        for metric_name, val in ground_truth_numbers.items():
            if val is not None:
                checks_total += 1
                if str(val) not in report:
                    hallucinations_found += 1
                    details.append(f"Numerical mismatch/omission for {metric_name}={val} in report")

        # Verify compliance & leakage blacklist never entered selected_features
        selected = set(state.get("selected_features", []))
        forbidden = set(get_profile(state.get("card_type")).prohibited_features)
        checks_total += 1
        if selected & forbidden:
            hallucinations_found += 1
            details.append(f"Forbidden feature leaked into model: {selected & forbidden}")

    elif intent == "data_audit_only":
        gov = state.get("data_inspection_result", {}).get("governance_audit", {})
        for field_name in benchmark_case.get("expected_quarantined_leakage", []):
            checks_total += 1
            if field_name not in gov.get("quarantined_leakage_features", []):
                hallucinations_found += 1
                details.append(f"Failed to quarantine leakage feature: {field_name}")
        for field_name in benchmark_case.get("expected_quarantined_compliance", []):
            checks_total += 1
            if field_name not in gov.get("quarantined_compliance_features", []):
                hallucinations_found += 1
                details.append(f"Failed to quarantine compliance feature: {field_name}")

    hallucination_rate = hallucinations_found / max(checks_total, 1)
    return {
        "checks_total": checks_total,
        "hallucinations_found": hallucinations_found,
        "hallucination_rate": round(hallucination_rate, 4),
        "faithfulness_score": round(1.0 - hallucination_rate, 4),
        "details": details,
    }


def evaluate_task_completion(
    state: CreditModelingState,
    benchmark_case: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluates whether the multi-agent workflow completed its business goal and satisfied all gates."""
    intent_ok = state.get("intent_type") == benchmark_case.get("expected_intent")
    if "expected_card_type" in benchmark_case:
        intent_ok = intent_ok and state.get("card_type") == benchmark_case["expected_card_type"]
    report_non_empty = len(state.get("final_report_markdown", "")) > 100
    compliance_ok = bool(state.get("compliance_passed", False))

    reflection_ok = True
    if benchmark_case.get("require_reflection_triggered"):
        reflection_ok = state.get("reflection_count", 0) >= 1 and len(state.get("reflection_history", [])) >= 1

    validation_ok = True
    if benchmark_case.get("require_validation_passed"):
        validation_ok = bool(state.get("validation_result", {}).get("validation_passed", False))

    completed = all([intent_ok, report_non_empty, compliance_ok, reflection_ok, validation_ok])
    return {
        "task_completed": completed,
        "intent_matched": intent_ok,
        "reflection_verified": reflection_ok,
        "validation_gate_passed": validation_ok,
        "compliance_passed": compliance_ok,
    }


def run_benchmark_suite(benchmark_path: str | Path | None = None) -> Dict[str, Any]:
    """Runs the complete Golden Benchmark suite and aggregates L1/L2/L3 metrics."""
    if benchmark_path is None:
        benchmark_path = Path(__file__).parent / "benchmark_dataset.json"
    cases = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))

    case_results: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx, case in enumerate(cases):
        state = run_credit_modeling_workflow(
            user_query=case["query"],
            session_id=f"eval-{case['case_id']}",
        )
        tool_eval = evaluate_tool_calling_ast(state, case)
        halluc_eval = evaluate_hallucination_rate(state, case)
        task_eval = evaluate_task_completion(state, case)

        logs = state.get("agent_logs", [])
        e2e_latency_ms = round(sum(l["latency_ms"] for l in logs), 2)
        p_tokens = sum(l["prompt_tokens"] for l in logs)
        c_tokens = sum(l["completion_tokens"] for l in logs)

        latencies_ms.append(e2e_latency_ms)
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens

        case_results.append(
            {
                "case_id": case["case_id"],
                "level": case["level"],
                "query": case["query"],
                "task_completed": task_eval["task_completed"],
                "tool_calling_score": tool_eval["tool_calling_score"],
                "hallucination_rate": halluc_eval["hallucination_rate"],
                "faithfulness_score": halluc_eval["faithfulness_score"],
                "reflection_count": state.get("reflection_count", 0),
                "e2e_latency_ms": e2e_latency_ms,
                "tokens_used": p_tokens + c_tokens,
                "tool_eval_detail": tool_eval,
                "halluc_eval_detail": halluc_eval,
                "task_eval_detail": task_eval,
            }
        )

    n = max(len(case_results), 1)
    task_completion_rate = sum(1 for r in case_results if r["task_completed"]) / n
    avg_tool_accuracy = sum(r["tool_calling_score"] for r in case_results) / n
    avg_hallucination_rate = sum(r["hallucination_rate"] for r in case_results) / n

    # Pricing estimate based on standard enterprise model ($0.002 / 1K prompt, $0.006 / 1K completion)
    est_cost_usd = round((total_prompt_tokens * 0.002 + total_completion_tokens * 0.006) / 1000.0, 5)

    return {
        "summary": {
            "total_cases": len(case_results),
            "task_completion_rate": round(task_completion_rate, 4),
            "tool_calling_ast_accuracy": round(avg_tool_accuracy, 4),
            "hallucination_rate": round(avg_hallucination_rate, 4),
            "faithfulness_rate": round(1.0 - avg_hallucination_rate, 4),
            "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "estimated_cost_usd": est_cost_usd,
        },
        "case_results": case_results,
    }
