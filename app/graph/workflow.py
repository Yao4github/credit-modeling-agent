"""LangGraph Multi-Agent Workflow Orchestrator with Conditional Routing & Self-Reflection."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.specialists import (
    compliance_reporter_node,
    data_explorer_node,
    feature_engineer_node,
    model_trainer_node,
    orchestrator_node,
    validator_node,
)
from app.core.cards import get_profile
from app.graph.state import CreditModelingState


def route_after_orchestrator(
    state: CreditModelingState,
) -> Literal["data_explorer", "compliance_reporter"]:
    if state.get("intent_type") == "knowledge_qa":
        return "compliance_reporter"
    return "data_explorer"


def route_after_data_explorer(
    state: CreditModelingState,
) -> Literal["feature_engineer", "compliance_reporter"]:
    if state.get("intent_type") == "data_audit_only":
        return "compliance_reporter"
    return "feature_engineer"


def route_after_feature_engineer(
    state: CreditModelingState,
) -> Literal["model_trainer", "compliance_reporter"]:
    if state.get("intent_type") == "feature_selection_only":
        return "compliance_reporter"
    return "model_trainer"


def route_after_validator(
    state: CreditModelingState,
) -> Literal["feature_engineer", "compliance_reporter"]:
    """Self-Reflection Conditional Edge: Rolls back to Feature Engineering if OOT validation fails."""
    val = state.get("validation_result", {})
    passed = val.get("validation_passed", False)
    refl_cnt = state.get("reflection_count", 0)
    max_refl = state.get("max_reflections", 2)

    if not passed and refl_cnt < max_refl and not state.get("stop_reflection", False):
        return "feature_engineer"
    return "compliance_reporter"


def build_credit_modeling_graph(with_checkpointer: bool = True):
    """Constructs and compiles the 6-Agent LangGraph StateGraph."""
    builder = StateGraph(CreditModelingState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("data_explorer", data_explorer_node)
    builder.add_node("feature_engineer", feature_engineer_node)
    builder.add_node("model_trainer", model_trainer_node)
    builder.add_node("validator", validator_node)
    builder.add_node("compliance_reporter", compliance_reporter_node)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "data_explorer": "data_explorer",
            "compliance_reporter": "compliance_reporter",
        },
    )
    builder.add_conditional_edges(
        "data_explorer",
        route_after_data_explorer,
        {
            "feature_engineer": "feature_engineer",
            "compliance_reporter": "compliance_reporter",
        },
    )
    builder.add_conditional_edges(
        "feature_engineer",
        route_after_feature_engineer,
        {
            "model_trainer": "model_trainer",
            "compliance_reporter": "compliance_reporter",
        },
    )
    builder.add_edge("model_trainer", "validator")
    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "feature_engineer": "feature_engineer",
            "compliance_reporter": "compliance_reporter",
        },
    )
    builder.add_edge("compliance_reporter", END)

    checkpointer = MemorySaver() if with_checkpointer else None
    return builder.compile(checkpointer=checkpointer)


def run_credit_modeling_workflow(
    user_query: str,
    session_id: str = "default-session",
    max_reflections: int = 2,
    train_uri: Optional[str] = None,
    oot_uri: Optional[str] = None,
    card_type: Optional[str] = None,
) -> CreditModelingState:
    """Executes the compiled multi-agent workflow end-to-end and returns the final state."""
    graph = build_credit_modeling_graph(with_checkpointer=True)
    initial_state: CreditModelingState = {
        "session_id": session_id,
        "user_query": user_query,
        "reflection_count": 0,
        "max_reflections": max_reflections,
        "excluded_by_reflection": [],
        "reflection_history": [],
        "tool_calls": [],
        "agent_logs": [],
    }
    if card_type:
        initial_state["card_type"] = card_type
    if train_uri or oot_uri:
        profile = get_profile(card_type)
        initial_state["dataset_refs"] = {
            "train_uri": train_uri or profile.train_uri,
            "oot_uri": oot_uri or profile.oot_uri,
        }
    config: Dict[str, Any] = {"configurable": {"thread_id": session_id}}
    final_state = graph.invoke(initial_state, config=config)
    return final_state
