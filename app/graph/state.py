"""LangGraph State Definition for the Credit Risk Data Modeling Multi-Agent System."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class ToolCallRecord(TypedDict):
    agent: str
    tool_name: str
    tool_args: Dict[str, Any]
    status: str
    latency_ms: float


class AgentStepLog(TypedDict):
    step_id: int
    agent: str
    action: str
    thought: str
    summary: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


class CreditModelingState(TypedDict, total=False):
    # Input & Intent
    session_id: str
    user_query: str
    intent_type: str  # "full_modeling" | "data_audit_only" | "feature_selection_only" | "knowledge_qa"
    card_type: str  # "A" (application) | "B" (behavior) | "C" (collection)
    constraints: Dict[str, float]  # {"ks_min": 0.30, "psi_max": 0.10, "iv_min": 0.02, "vif_max": 5.0, "target_approval_rate": 0.75}

    # Planning & Memory
    plan_dag: List[Dict[str, Any]]
    recalled_memories: List[Dict[str, Any]]
    rag_context: List[Dict[str, Any]]

    # Data Pointer & Governance State
    dataset_refs: Dict[str, str]
    data_inspection_result: Dict[str, Any]
    clean_candidate_features: List[str]
    quarantined_leakage_features: List[str]
    quarantined_compliance_features: List[str]

    # Feature Engineering State
    excluded_by_reflection: List[str]
    feature_selection_result: Dict[str, Any]
    selected_features: List[str]

    # Model Training & Validation State
    model_training_result: Dict[str, Any]
    validation_result: Dict[str, Any]
    challenger_result: Dict[str, Any]
    reflection_count: int
    max_reflections: int
    stop_reflection: bool  # set by the validator when a failure has no actionable culprit (skip futile rollbacks)
    reflection_history: Annotated[List[str], operator.add]
    binning_strategy: str  # "quantile5" (default) | "quantile10" | "tree"; switched by the validator's reflection
    binning_tried: Annotated[List[str], operator.add]  # strategies already attempted (never retried)

    # Compliance & Final Deliverable
    compliance_passed: bool
    final_report_markdown: str

    # Observability & Evaluation Trace
    tool_calls: Annotated[List[ToolCallRecord], operator.add]
    agent_logs: Annotated[List[AgentStepLog], operator.add]
    memory_governance_stats: Dict[str, Any]
