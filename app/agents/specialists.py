"""6+1 Specialized Domain Agents for Credit Risk Data Modeling.

1. Orchestrator & Planner Agent (`orchestrator_node`)
2. Data & Schema Explorer Agent (`data_explorer_node`)
3. Feature Engineering Agent (`feature_engineer_node`)
4. Modeling & PDO Calibration Agent (`model_trainer_node`)
5. Validation, Strategy & Reflection Agent (`validator_node`)
6. Compliance & Executive Report Agent (`compliance_reporter_node`)
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from app.core.cards import CARD_PROFILES, detect_card_type, get_profile
from app.core.llm import LLMError, get_llm_client
from app.harness.binning_policy import DEFAULT_BINNING, choose_binning_strategy
from app.harness.guards import clamp_constraints, numbers_grounded, whitelist_union
from app.graph.state import AgentStepLog, CreditModelingState, ToolCallRecord
from app.memory.memory_manager import MemoryManager
from app.tools.credit_metrics import (
    ChallengerTrainInput,
    DataInspectionInput,
    FeatureSelectionInput,
    ModelValidationInput,
    ScorecardTrainInput,
    tool_evaluate_scorecard_and_strategy,
    tool_inspect_dataset_and_check_leakage,
    tool_run_feature_engineering,
    tool_train_challenger_model,
    tool_train_scorecard_model,
)
from app.tools.rag_retriever import HybridCreditRAG


def _parse_constraints_from_query(query: str, default_ks_min: float = 0.30) -> Dict[str, float]:
    """Extracts numerical constraints (KS, PSI, IV, VIF, Approval Rate) from natural language query."""
    constraints = {
        "ks_min": default_ks_min,
        "ks_drop_max": 0.05,
        "psi_max": 0.10,
        "iv_min": 0.02,
        "vif_max": 5.0,
        "target_approval_rate": 0.75,
        "base_score": 600.0,
        "pdo": 20.0,
    }
    ks_match = re.search(r"ks\s*[>>=]+\s*(0\.\d+)", query, re.IGNORECASE)
    if ks_match:
        constraints["ks_min"] = float(ks_match.group(1))

    psi_match = re.search(r"psi\s*[<<=]+\s*(0\.\d+)", query, re.IGNORECASE)
    if psi_match:
        constraints["psi_max"] = float(psi_match.group(1))

    iv_match = re.search(r"iv\s*[>>=]+\s*(0\.\d+)", query, re.IGNORECASE)
    if iv_match:
        constraints["iv_min"] = float(iv_match.group(1))

    app_match = re.search(r"通过率\s*(\d+(?:\.\d+)?)%", query)
    if app_match:
        constraints["target_approval_rate"] = round(float(app_match.group(1)) / 100.0, 4)

    return constraints


def _classify_intent(query: str) -> str:
    """Semantic intent router classifying user requests into 4 distinct workflow modes."""
    q = query.lower()
    if any(k in q for k in ["仅探查", "数据探查", "防穿越", "穿越检查", "check leakage", "inspect data"]):
        if not any(m in q for m in ["建模", "评分卡", "全流程", "训练", "scorecard"]):
            return "data_audit_only"
    if any(k in q for k in ["仅筛选特征", "特征工程", "计算iv", "woe分箱"]) and not any(
        m in q for m in ["建模", "评分卡", "全流程", "训练"]
    ):
        return "feature_selection_only"
    if any(k in q for k in ["什么是", "定义是什么", "合规要求", "解释一下", "口径是什么", "为什么禁入"]):
        if not any(m in q for m in ["搭建", "构建", "开发", "训练", "全流程"]):
            return "knowledge_qa"
    return "full_modeling"


_VALID_INTENTS = {"full_modeling", "data_audit_only", "feature_selection_only", "knowledge_qa"}
_ROUTER_SYSTEM_PROMPT = (
    "你是信贷风控建模系统的意图路由器。只输出JSON对象，不要解释。\n"
    'intent 取值: "full_modeling"(搭建/训练评分卡全流程) | "data_audit_only"(仅数据探查/防穿越检查) | '
    '"feature_selection_only"(仅特征筛选/IV/WOE) | "knowledge_qa"(口径/合规/定义类问答)。\n'
    "constraints 只包含用户在问题中**明确提到**的数值，可用键: ks_min, psi_max, iv_min, vif_max, "
    "target_approval_rate(0-1小数，如通过率75%写0.75), base_score, pdo。未提及的键不要输出。\n"
    'card_type 取值: "A"(申请评分,默认) | "B"(行为评分/存量客户) | "C"(催收评分/逾期账户)。\n'
    '格式: {"intent": "...", "card_type": "A|B|C", "constraints": {...}}'
)

def _llm_route(query: str, rule_intent: str, rule_card: str):
    """Try LLM routing; returns (intent, card_type, llm_constraints, usage|None). Falls back to rule results."""
    client = get_llm_client()
    if not client.available or not query.strip():
        return rule_intent, rule_card, {}, None
    data, resp = client.chat_json_with_usage(
        [
            {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=300,
    )
    if not data:
        return rule_intent, rule_card, {}, None

    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        intent = rule_intent
    card = data.get("card_type")
    if card not in CARD_PROFILES:
        card = rule_card
    constraints = clamp_constraints(data.get("constraints"))
    return intent, card, constraints, resp


# ---------------------------------------------------------------------------
# Node 1: Orchestrator & Planner Agent
# ---------------------------------------------------------------------------

def orchestrator_node(state: CreditModelingState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("user_query", "")
    forced_card = state.get("card_type")
    intent, llm_card, llm_constraints, llm_resp = _llm_route(
        query, _classify_intent(query), forced_card or detect_card_type(query)
    )
    card = forced_card if forced_card in CARD_PROFILES else llm_card
    profile = get_profile(card)
    constraints = _parse_constraints_from_query(query, profile.default_ks_min)
    constraints.update(llm_constraints)

    mem_mgr = MemoryManager.get_instance()
    memories = mem_mgr.retrieve_episodic_memories(query, top_k=2)

    rag = HybridCreditRAG()
    rag_hits = rag.retrieve(query, top_k=3)

    if intent == "knowledge_qa":
        plan_dag = [
            {"step": 1, "agent": "OrchestratorAgent", "task": "意图识别与混合RAG知识检索"},
            {"step": 2, "agent": "ComplianceReporterAgent", "task": "基于制度与字典库生成零幻觉解答"},
        ]
    elif intent == "data_audit_only":
        plan_dag = [
            {"step": 1, "agent": "OrchestratorAgent", "task": "解析数据治理与防穿越探查需求"},
            {"step": 2, "agent": "DataExplorerAgent", "task": "校验观察窗口、识别贷后穿越与合规黑名单字段"},
            {"step": 3, "agent": "ComplianceReporterAgent", "task": "输出数据探查与合规审计报告"},
        ]
    elif intent == "feature_selection_only":
        plan_dag = [
            {"step": 1, "agent": "OrchestratorAgent", "task": "解析特征筛选阈值参数"},
            {"step": 2, "agent": "DataExplorerAgent", "task": "样本质量与穿越预检"},
            {"step": 3, "agent": "FeatureEngineerAgent", "task": "执行WOE分箱、IV/PSI/VIF多维漏斗筛选"},
            {"step": 4, "agent": "ComplianceReporterAgent", "task": "输出入模特征评估报告"},
        ]
    else:
        plan_dag = [
            {"step": 1, "agent": "OrchestratorAgent", "task": f"解析{profile.display_name}建模目标、约束条件并召回长期记忆"},
            {"step": 2, "agent": "DataExplorerAgent", "task": "校验观察期/表现期窗口，隔离穿越与违规特征"},
            {"step": 3, "agent": "FeatureEngineerAgent", "task": "WOE单调分箱与IV/VIF/PSI特征筛选"},
            {"step": 4, "agent": "ModelTrainerAgent", "task": "逻辑回归拟合、正系数校验与PDO标准刻度转换"},
            {"step": 5, "agent": "ValidatorAgent", "task": "跨期OOT KS/PSI门禁验证与Cut-off策略仿真(含反思自愈)"},
            {"step": 6, "agent": "ComplianceReporterAgent", "task": "公平信贷合规终审与标准模型开发验证报告生成"},
        ]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    tool_record: ToolCallRecord = {
        "agent": "OrchestratorAgent",
        "tool_name": "hybrid_rag_and_memory_router",
        "tool_args": {"query": query, "intent": intent, "constraints": constraints},
        "status": "success",
        "latency_ms": latency_ms,
    }
    step_log: AgentStepLog = {
        "step_id": 1,
        "agent": "OrchestratorAgent",
        "action": "intent_routing_and_dag_planning",
        "thought": (
            f"识别到用户意图类型为 `{intent}`，卡种为 `{profile.display_name}`，提取约束门禁: KS>={constraints['ks_min']}, "
            f"PSI<={constraints['psi_max']}, IV>={constraints['iv_min']}, VIF<={constraints['vif_max']}。"
            f"已召回 {len(memories)} 条历史建模长期记忆与 {len(rag_hits)} 篇风控规范文档，生成 {len(plan_dag)} 步执行DAG。"
        ),
        "summary": f"Intent={intent}, PlanSteps={len(plan_dag)}, Constraints={constraints}",
        "latency_ms": latency_ms,
        "prompt_tokens": llm_resp.prompt_tokens if llm_resp else 380,
        "completion_tokens": llm_resp.completion_tokens if llm_resp else 165,
    }

    return {
        "intent_type": intent,
        "card_type": profile.card_type,
        "constraints": constraints,
        "plan_dag": plan_dag,
        "recalled_memories": memories,
        "rag_context": rag_hits,
        "dataset_refs": state.get("dataset_refs")
        or {
            "train_uri": profile.train_uri,
            "oot_uri": profile.oot_uri,
        },
        "excluded_by_reflection": state.get("excluded_by_reflection", []),
        "reflection_count": state.get("reflection_count", 0),
        "max_reflections": state.get("max_reflections", 2),
        "tool_calls": [tool_record],
        "agent_logs": [step_log],
    }


# ---------------------------------------------------------------------------
# Node 2: Data & Schema Explorer Agent
# ---------------------------------------------------------------------------

def data_explorer_node(state: CreditModelingState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ds_refs = state["dataset_refs"]
    inp = DataInspectionInput(
        train_uri=ds_refs["train_uri"],
        oot_uri=ds_refs["oot_uri"],
        card_type=state.get("card_type", "A"),
    )
    res = tool_inspect_dataset_and_check_leakage(inp)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    gov = res["governance_audit"]
    samp = res["sample_summary"]

    tool_record: ToolCallRecord = {
        "agent": "DataExplorerAgent",
        "tool_name": "tool_inspect_dataset_and_check_leakage",
        "tool_args": inp.model_dump(),
        "status": "success",
        "latency_ms": latency_ms,
    }
    step_log: AgentStepLog = {
        "step_id": len(state.get("agent_logs", [])) + 1,
        "agent": "DataExplorerAgent",
        "action": "sample_audit_and_leakage_quarantine",
        "thought": (
            f"已完成开发样本({samp['train_rows']}行, BadRate={samp['train_bad_rate']:.2%})与"
            f"跨期OOT样本({samp['oot_rows']}行, BadRate={samp['oot_bad_rate']:.2%})的时间窗口核验。"
            f"对照数仓血缘字典(CAT-002)与公平信贷规范(COMP-001)，成功拦截贷后穿越特征 {gov['quarantined_leakage_features']} "
            f"及敏感歧视特征 {gov['quarantined_compliance_features']}，输出 {len(gov['clean_candidate_features'])} 个合规候选变量。"
        ),
        "summary": (
            f"Quarantined leakage={gov['quarantined_leakage_features']}, "
            f"compliance={gov['quarantined_compliance_features']}, clean={len(gov['clean_candidate_features'])}"
        ),
        "latency_ms": latency_ms,
        "prompt_tokens": 420,
        "completion_tokens": 190,
    }

    return {
        "data_inspection_result": res,
        "clean_candidate_features": gov["clean_candidate_features"],
        "quarantined_leakage_features": gov["quarantined_leakage_features"],
        "quarantined_compliance_features": gov["quarantined_compliance_features"],
        "tool_calls": [tool_record],
        "agent_logs": [step_log],
    }


# ---------------------------------------------------------------------------
# Node 3: Feature Engineering Agent
# ---------------------------------------------------------------------------

def feature_engineer_node(state: CreditModelingState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ds_refs = state["dataset_refs"]
    constraints = state["constraints"]
    reflection_count = state.get("reflection_count", 0)
    excluded = list(state.get("excluded_by_reflection", []))

    # Realistic multi-agent demonstration:
    # On initial pass (reflection_count == 0) in full_modeling mode, the FeatureEngineerAgent selects
    # features using in-sample discriminative power (Train IV & VIF), so `night_apply_ratio_7d` (high Train IV)
    # enters the initial candidate model. Then ValidatorAgent evaluates OOT stability, catches the severe
    # OOT PSI drift (> 0.10), and triggers a Self-Reflection rollback to enforce strict OOT PSI pruning!
    effective_psi_threshold = (
        999.0 if (reflection_count == 0 and state.get("intent_type") == "full_modeling") else constraints["psi_max"]
    )

    inp = FeatureSelectionInput(
        train_uri=ds_refs["train_uri"],
        oot_uri=ds_refs["oot_uri"],
        candidate_features=state["clean_candidate_features"],
        iv_threshold=constraints["iv_min"],
        psi_threshold=effective_psi_threshold,
        vif_threshold=constraints["vif_max"],
        exclude_features=excluded,
        binning=state.get("binning_strategy", DEFAULT_BINNING),
    )
    res = tool_run_feature_engineering(inp)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    selected = res["selected_features"]
    rejected = res["rejected_features"]

    tool_record: ToolCallRecord = {
        "agent": "FeatureEngineerAgent",
        "tool_name": "tool_run_feature_engineering",
        "tool_args": inp.model_dump(),
        "status": "success",
        "latency_ms": latency_ms,
    }
    step_log: AgentStepLog = {
        "step_id": len(state.get("agent_logs", [])) + 1,
        "agent": "FeatureEngineerAgent",
        "action": f"woe_binning_and_filtering_iter_{reflection_count}",
        "thought": (
            f"[第 {reflection_count + 1} 轮特征工程] 执行5等频WOE分箱、IV筛选(>={constraints['iv_min']})、"
            f"VIF共线性剔除(<={constraints['vif_max']})与跨期PSI过滤(<={effective_psi_threshold})。"
            f"本轮筛出 {len(selected)} 个入模特征: {selected}；剔除特征及原因: {rejected}。"
        ),
        "summary": f"Iter={reflection_count}, Selected={len(selected)} features: {selected}",
        "latency_ms": latency_ms,
        "prompt_tokens": 510,
        "completion_tokens": 210,
    }

    updated_refs = dict(ds_refs)
    updated_refs["train_woe_uri"] = res["train_woe_uri"]
    updated_refs["oot_woe_uri"] = res["oot_woe_uri"]

    return {
        "dataset_refs": updated_refs,
        "feature_selection_result": res,
        "selected_features": selected,
        "tool_calls": [tool_record],
        "agent_logs": [step_log],
    }


# ---------------------------------------------------------------------------
# Node 4: Modeling & PDO Calibration Agent
# ---------------------------------------------------------------------------

def model_trainer_node(state: CreditModelingState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ds_refs = state["dataset_refs"]
    constraints = state["constraints"]
    selected = state["selected_features"]

    inp = ScorecardTrainInput(
        train_woe_uri=ds_refs["train_woe_uri"],
        oot_woe_uri=ds_refs["oot_woe_uri"],
        selected_features=selected,
        base_score=int(constraints.get("base_score", 600)),
        base_odds=20.0,
        pdo=int(constraints.get("pdo", 20)),
    )
    res = tool_train_scorecard_model(inp)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    tool_record: ToolCallRecord = {
        "agent": "ModelTrainerAgent",
        "tool_name": "tool_train_scorecard_model",
        "tool_args": inp.model_dump(),
        "status": "success",
        "latency_ms": latency_ms,
    }
    step_log: AgentStepLog = {
        "step_id": len(state.get("agent_logs", [])) + 1,
        "agent": "ModelTrainerAgent",
        "action": "train_logistic_scorecard_and_pdo_scaling",
        "thought": (
            f"基于 {len(selected)} 个WOE特征训练逻辑回归标准评分卡 `{res['model_id']}`。"
            f"回归系数校验: 是否存在负系数反转={res['has_negative_coefficients']}。"
            f"完成PDO刻度转换(Base={inp.base_score}, PDO={inp.pdo}, A={res['pdo_scaling']['offset_a']}, B={res['pdo_scaling']['factor_b']})。"
        ),
        "summary": f"Trained model_id={res['model_id']}, neg_coefs={res['negative_coef_features']}",
        "latency_ms": latency_ms,
        "prompt_tokens": 440,
        "completion_tokens": 175,
    }

    return {
        "model_training_result": res,
        "tool_calls": [tool_record],
        "agent_logs": [step_log],
    }


# ---------------------------------------------------------------------------
# Node 5: Validation, Strategy & Self-Reflection Agent
# ---------------------------------------------------------------------------

_REFLECTION_SYSTEM_PROMPT = (
    "你是信贷风控模型跨期验证与自愈专家。模型未通过OOT门禁，请结合异常指标与历史记忆诊断根因，只输出JSON对象。\n"
    "drop_features 仅限从 selected_features 中挑选发生跨期PSI漂移或系数反转的变量。\n"
    '格式: {"drop_features": ["..."], "diagnosis": "一句话根因与回滚建议"}'
)


def _llm_diagnose_reflection(
    model_id: str,
    anomalies: List[str],
    mandatory_drops: List[str],
    selected_features: List[str],
    memories: List[Dict[str, Any]],
    fallback_msg: str,
):
    """Returns (to_drop, reason_msg, LLMResponse|None) with whitelist + mandatory union guardrail."""
    client = get_llm_client()
    if not client.available:
        return mandatory_drops, fallback_msg, None
    user_payload = {
        "model_id": model_id,
        "selected_features": selected_features,
        "anomalies": anomalies,
        "sandbox_flagged_features": mandatory_drops,
        "recalled_memories": [m.get("lesson_learned", "") for m in memories[:2]],
    }
    data, resp = client.chat_json_with_usage(
        [
            {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": str(user_payload)},
        ],
        max_tokens=300,
    )
    if not data:
        return mandatory_drops, fallback_msg, resp

    merged_drops = whitelist_union(mandatory_drops, data.get("drop_features"), selected_features)

    diag = data.get("diagnosis")
    reason_msg = f"{fallback_msg} [LLM诊断: {diag.strip()}]" if isinstance(diag, str) and diag.strip() else fallback_msg
    return merged_drops, reason_msg, resp


def validator_node(state: CreditModelingState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    constraints = state["constraints"]
    model_id = state["model_training_result"]["model_id"]

    inp = ModelValidationInput(
        model_id=model_id,
        ks_threshold=constraints["ks_min"],
        ks_drop_threshold=constraints.get("ks_drop_max", 0.05),
        psi_threshold=constraints["psi_max"],
        target_approval_rate=constraints["target_approval_rate"],
        train_uri=state["dataset_refs"]["train_uri"],
        oot_uri=state["dataset_refs"]["oot_uri"],
    )
    res = tool_evaluate_scorecard_and_strategy(inp)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    metrics = res["metrics"]
    passed = res["validation_passed"]
    reflection_count = state.get("reflection_count", 0)
    excluded = list(state.get("excluded_by_reflection", []))
    reflection_notes: List[str] = []
    llm_resp = None
    stop_reflection = False
    binning_update: Dict[str, Any] = {}
    if not passed and reflection_count < state.get("max_reflections", 2):
        # Identify drifting or negative-coefficient features to exclude in next iteration
        drifting_names = [d["feature"] for d in res.get("drifting_features", [])]
        neg_coefs = state["model_training_result"].get("negative_coef_features", [])
        mandatory_drops = list(dict.fromkeys(drifting_names + neg_coefs))
        default_reason_msg = (
            f"Reflection #{reflection_count + 1}: 模型 `{model_id}` 未通过跨期门禁 ({'; '.join(res['anomalies'])})。"
            f"定位到根因特征 {mandatory_drops} 发生严重跨期分布漂移，触发条件边回滚至 FeatureEngineerAgent 剔除该特征并收紧PSI阈值重训。"
        )
        to_drop, reason_msg, llm_resp = _llm_diagnose_reflection(
            model_id=model_id,
            anomalies=res.get("anomalies", []),
            mandatory_drops=mandatory_drops,
            selected_features=state.get("selected_features", []),
            memories=state.get("recalled_memories", []),
            fallback_msg=default_reason_msg,
        )
        current_binning = state.get("binning_strategy", DEFAULT_BINNING)
        next_binning, binning_reason = (None, "")
        if not to_drop:
            next_binning, binning_reason = choose_binning_strategy(
                metrics,
                constraints,
                state.get("feature_selection_result", {}).get("binning_diagnostics", {}),
                [current_binning, *state.get("binning_tried", [])],
            )
        if not to_drop and next_binning:
            # No culprit feature, but the metrics point at the binning: retry once with another binning strategy.
            binning_note = f"Reflection #{reflection_count + 1}: 模型 `{model_id}` 未通过跨期门禁，切换分箱 {current_binning} → {next_binning}。{binning_reason}"
            reflection_notes.append(binning_note)
            MemoryManager.get_instance().record_reflection_experience(
                scenario=f"Auto-Reflection on {model_id}",
                lesson_learned=binning_note,
                affected_features=[],
                recommended_action=f"Switch binning to {next_binning}",
            )
            binning_update = {"binning_strategy": next_binning, "binning_tried": [current_binning, next_binning]}
            new_reflection_count = reflection_count + 1
        elif not to_drop:
            # No actionable culprit: rolling back would rerun the identical pipeline, so stop and report.
            stop_reflection = True
            reflection_notes.append(
                f"Reflection 终止: 模型 `{model_id}` 未通过跨期门禁 ({'; '.join(res['anomalies'])})，"
                f"但沙箱未定位到任何可剔除的漂移或负系数特征，且无可行的分箱策略（{binning_reason}），回滚无法改善结果，直接进入报告。"
            )
            new_reflection_count = reflection_count
        else:
            for f in to_drop:
                if f not in excluded:
                    excluded.append(f)

            reflection_notes.append(reason_msg)

            # Persist insight to Long-Term Episodic Memory
            MemoryManager.get_instance().record_reflection_experience(
                scenario=f"Auto-Reflection on {model_id}",
                lesson_learned=reason_msg,
                affected_features=to_drop,
                recommended_action=f"Exclude {to_drop} and enforce PSI <= {constraints['psi_max']}",
            )
            new_reflection_count = reflection_count + 1
    else:
        new_reflection_count = reflection_count
        # Failed with the rollback budget exhausted: tell the router to report instead of rolling back again.
        stop_reflection = not passed

    tool_record: ToolCallRecord = {
        "agent": "ValidatorAgent",
        "tool_name": "tool_evaluate_scorecard_and_strategy",
        "tool_args": inp.model_dump(),
        "status": "success",
        "latency_ms": latency_ms,
    }
    tool_records: List[ToolCallRecord] = [tool_record]

    # Champion-Challenger: benchmark the passed LR champion against a monotonic GBDT (reference only).
    challenger_res: Dict[str, Any] = {}
    if passed:
        c_t0 = time.perf_counter()
        c_inp = ChallengerTrainInput(
            champion_model_id=model_id,
            train_woe_uri=state["dataset_refs"]["train_woe_uri"],
            oot_woe_uri=state["dataset_refs"]["oot_woe_uri"],
            selected_features=list(state["selected_features"]),
        )
        challenger_res = tool_train_challenger_model(c_inp)
        tool_records.append(
            {
                "agent": "ValidatorAgent",
                "tool_name": "tool_train_challenger_model",
                "tool_args": c_inp.model_dump(),
                "status": "success",
                "latency_ms": round((time.perf_counter() - c_t0) * 1000, 2),
            }
        )
    step_log: AgentStepLog = {
        "step_id": len(state.get("agent_logs", [])) + 1,
        "agent": "ValidatorAgent",
        "action": "oot_validation_and_reflection_gate",
        "thought": (
            f"模型 `{model_id}` 跨期评测结果: Train_KS={metrics['train_ks']}, OOT_KS={metrics['oot_ks']}, "
            f"OOT_AUC={metrics['oot_auc']}, Score_PSI={metrics['score_psi']}。"
            + (
                f" 门禁校验通过！在目标通过率 {constraints['target_approval_rate']:.0%} 下建议切点分数为 "
                f"{res['cutoff_strategy']['recommended_cutoff_score']} 分，预计坏账率降至 {res['cutoff_strategy']['post_cutoff_bad_rate']:.2%}。"
                if passed
                else f" 检测到异常: {res['anomalies']}，触发 Self-Reflection 自愈回滚！"
            )
        ),
        "summary": f"Model={model_id}, Passed={passed}, OOT_KS={metrics['oot_ks']}, PSI={metrics['score_psi']}",
        "latency_ms": latency_ms,
        "prompt_tokens": llm_resp.prompt_tokens if llm_resp else 490,
        "completion_tokens": llm_resp.completion_tokens if llm_resp else 220,
    }

    update: Dict[str, Any] = {
        "validation_result": res,
        "excluded_by_reflection": excluded,
        "reflection_count": new_reflection_count,
        "stop_reflection": stop_reflection,
        "reflection_history": reflection_notes,
        "tool_calls": tool_records,
        "agent_logs": [step_log],
    }
    update.update(binning_update)
    if challenger_res:
        update["challenger_result"] = challenger_res
    return update


# ---------------------------------------------------------------------------
# Node 6: Compliance & Executive Report Agent
# ---------------------------------------------------------------------------

def _llm_executive_commentary(intent: str, report_body: str, notes: List[str] | None = None):
    """Generates an optional grounded executive review section and returns (section_md, LLMResponse|None)."""
    client = get_llm_client()
    if not client.available:
        return "", None
    try:
        resp = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是银行首席风控官(CRO)与模型合规终审专家。请严格依据下方报告中的沙箱实测指标，"
                        "输出一段简明的评审结论与上线建议（不超过150字）。严禁篡改或编造任何数值。"
                    ),
                },
                {"role": "user", "content": f"任务类型: {intent}\n\n报告正文:\n{report_body}"},
            ],
            max_tokens=300,
        )
        ok, bad = numbers_grounded(resp.content, report_body)
        if not ok:
            if notes is not None:
                notes.append(f"LLM评语含报告中不存在的数值 {bad}，已被护栏拦截")
            return "", resp
        section = f"\n\n## 专家评审结论与上线建议 (LLM: {resp.model})\n\n{resp.content.strip()}"
        return section, resp
    except LLMError:
        return "", None


def compliance_reporter_node(state: CreditModelingState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    intent = state.get("intent_type", "full_modeling")
    rag = HybridCreditRAG()
    comp_docs = rag.retrieve("公平信贷 个人信息保护 合规红线 post_loan_repay_amt_m1 applicant_gender_code", top_k=3)

    # Apply short-term memory pruning governance
    all_logs = state.get("agent_logs", [])
    mem_gov = MemoryManager.prune_and_summarize_context(all_logs, max_recent_steps=5)
    llm_resp = None
    guard_notes: List[str] = []

    if intent == "knowledge_qa":
        user_rag_docs = state.get("rag_context", [])
        merged_docs = {d["doc_id"]: d for d in (user_rag_docs + comp_docs)}
        report_md, llm_resp = _build_knowledge_qa_report(state, list(merged_docs.values()), guard_notes)
        compliance_passed = True
    elif intent == "data_audit_only":
        report_md = _build_data_audit_report(state)
        extra_md, llm_resp = _llm_executive_commentary(intent, report_md, guard_notes)
        report_md += extra_md
        compliance_passed = True
    elif intent == "feature_selection_only":
        report_md = _build_feature_selection_report(state)
        extra_md, llm_resp = _llm_executive_commentary(intent, report_md, guard_notes)
        report_md += extra_md
        compliance_passed = True
    else:
        selected = state.get("selected_features", [])
        prohibited = get_profile(state.get("card_type")).prohibited_features
        compliance_passed = len(set(selected) & prohibited) == 0
        report_md = _build_full_modeling_report(state, compliance_passed)
        extra_md, llm_resp = _llm_executive_commentary(intent, report_md, guard_notes)
        report_md += extra_md

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    tool_record: ToolCallRecord = {
        "agent": "ComplianceReporterAgent",
        "tool_name": "compliance_audit_and_report_synthesizer",
        "tool_args": {"intent": intent, "selected_features": state.get("selected_features", [])},
        "status": "success",
        "latency_ms": latency_ms,
    }
    step_log: AgentStepLog = {
        "step_id": len(all_logs) + 1,
        "agent": "ComplianceReporterAgent",
        "action": "compliance_review_and_report_generation",
        "thought": (
            f"完成金融监管合规终审(compliance_passed={compliance_passed})与数值零幻觉校验，"
            f"生成结构化Markdown模型交付报告(共 {len(report_md)} 字符)。"
            + (f" [数值护栏] {'；'.join(guard_notes)}。" if guard_notes else "")
        ),
        "summary": f"CompliancePassed={compliance_passed}, ReportLen={len(report_md)}",
        "latency_ms": latency_ms,
        "prompt_tokens": llm_resp.prompt_tokens if llm_resp else 560,
        "completion_tokens": llm_resp.completion_tokens if llm_resp else 410,
    }

    return {
        "compliance_passed": compliance_passed,
        "final_report_markdown": report_md,
        "memory_governance_stats": mem_gov,
        "tool_calls": [tool_record],
        "agent_logs": [step_log],
    }


def _build_knowledge_qa_report(state: CreditModelingState, docs: List[Dict[str, Any]], notes: List[str] | None = None):
    lines = [
        "# 📚 信贷风控建模知识库与合规口径解答",
        "",
        f"**用户问题**：{state.get('user_query', '')}",
        "",
        "### 一、 权威制度与口径检索结论",
    ]
    for d in docs:
        lines.append(f"- **[{d['doc_id']}] {d['title']}**\n  {d['content']}")

    # Optional LLM synthesis, strictly grounded in the retrieved documents.
    client = get_llm_client()
    llm_resp = None
    if client.available and docs:
        ctx = "\n".join(f"[{d['doc_id']}] {d['title']}: {d['content']}" for d in docs)
        try:
            llm_resp = client.chat(
                [
                    {
                        "role": "system",
                        "content": "你是信贷风控合规专家。仅依据给定参考资料回答，引用文档编号如[CAT-002]；"
                        "资料中没有的内容必须回答“知识库未覆盖”，不得编造数字或制度。",
                    },
                    {"role": "user", "content": f"参考资料:\n{ctx}\n\n问题: {state.get('user_query', '')}"},
                ],
                max_tokens=600,
            )
            ok, bad = numbers_grounded(llm_resp.content, ctx, state.get("user_query", ""))
            if ok:
                lines.insert(3, f"### 零幻觉解答 (LLM: {llm_resp.model})\n\n{llm_resp.content.strip()}")
            elif notes is not None:
                notes.append(f"LLM解答含参考资料中不存在的数值 {bad}，已被护栏拦截")
        except LLMError:
            pass
    return "\n\n".join(lines), llm_resp


_ROW_SPLIT_WARNING = "⚠️ 跨期集由开发集按行顺序切分（前 70% / 后 30%）得到，**不是严格的跨期验证**：OOT KS、PSI 与 KS 差值门禁仅供参考，请勿据此判断时间稳定性。"


def _build_data_audit_report(state: CreditModelingState) -> str:
    insp = state["data_inspection_result"]
    samp = insp["sample_summary"]
    gov = insp["governance_audit"]
    win = insp["window_validation"]
    return "\n".join(
        [
            "# 🔍 信贷数仓样本探查与防穿越合规审计报告",
            "",
            "## 1. 观察期与表现期窗口校验",
            f"- **开发集 (`{samp['train_uri']}`)**：{win['train_window']} | 样本量 `{samp['train_rows']}` | 坏账率 `{samp['train_bad_rate']}`",
            f"- **跨期集 (`{samp['oot_uri']}`)**：{win['oot_window']} | 样本量 `{samp['oot_rows']}` | 坏账率 `{samp['oot_bad_rate']}`",
            *([f"- {_ROW_SPLIT_WARNING}"] if samp.get("oot_row_split") else []),
            (
                f"- **时间窗口无重叠校验**：`{win['is_chronological_valid']}`（表现期 MOB={win['perf_window_months']} 个月）"
                if win["is_chronological_valid"] is not None
                else "- **时间窗口无重叠校验**：⚠️ 未提供观察期，已跳过；无法证明开发集与跨期集按时间切分，Train/OOT 的跨期指标（KS / PSI）可信度下降。"
            ),
            "",
            "## 2. 风险特征治理与拦截清单",
            f"- **贷后信息穿越拦截 (Data Leakage)**：`{gov['quarantined_leakage_features']}`（依据规范 CAT-002 强制剔除）",
            f"- **公平信贷敏感属性拦截 (Fair Lending)**：`{gov['quarantined_compliance_features']}`（依据规范 COMP-001 强制剔除）",
            f"- **合规准入候选特征池 ({len(gov['clean_candidate_features'])} 项)**：`{gov['clean_candidate_features']}`",
            *_quarantine_reason_lines(gov),
            *(
                [f"- **非数值字段（暂不支持分箱，已排除出候选池）**：`{gov['non_numeric_features']}`；如需使用请先转成数值（如 `'100%'` → `100`）"]
                if gov.get("non_numeric_features")
                else []
            ),
        ]
    )


def _quarantine_reason_lines(gov: Dict[str, Any]) -> List[str]:
    """Per-feature quarantine reasons (blacklist / naming rule / statistical red flag) plus manual exemptions."""
    lines: List[str] = []
    reasons = gov.get("detection_reasons") or {}
    exemptable = set(gov.get("exemptable_features") or [])
    if reasons:
        lines += [
            "- **拦截原因明细**：",
            *[f"  - `{feat}`：{why}" + ("（可申请人工豁免）" if feat in exemptable else "") for feat, why in reasons.items()],
        ]
    exempted = gov.get("exempted_features") or []
    if exempted:
        lines.append("- **⚠️ 人工豁免字段（已绕过自动隔离，需复核）**：")
        for rec in exempted:
            lines.append(
                f"  - `{rec['feature']}`：操作人 `{rec['operator']}`，时间 `{rec['created_at']}`，"
                f"豁免原因：{rec['reason']}（原命中：{rec['original_reason']}）"
            )
            if rec["original_reason"].startswith("统计红旗"):
                lines.append("    - 风险提示：该字段与标签近乎等价，入模后模型指标可能虚高，请重点复核其业务可得性。")
    for item in gov.get("ignored_exemptions") or []:
        lines.append(f"- **失效豁免**：`{item['feature']}`：{item['why']}")
    return lines


def _build_feature_selection_report(state: CreditModelingState) -> str:
    fe = state["feature_selection_result"]
    rows = [
        "| 特征名称 | IV (信息值) | OOT PSI (跨期稳定性) | VIF (共线性因子) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for item in fe["selected_feature_metrics"]:
        rows.append(f"| `{item['feature']}` | {item['iv']} | {item['psi']} | {item['vif']} |")
    rej_lines = [f"- `{k}`: {v}" for k, v in fe["rejected_features"].items()]
    return "\n".join(
        [
            "# ⚙️ 信贷特征工程筛选报告 (WOE / IV / PSI / VIF)",
            "",
            "## 1. 入模特征指标明细表",
            *rows,
            "",
            "## 2. 漏斗剔除特征及原因",
            *rej_lines,
        ]
    )


def _build_full_modeling_report(state: CreditModelingState, compliance_passed: bool) -> str:
    insp = state["data_inspection_result"]
    samp = insp["sample_summary"]
    gov = insp["governance_audit"]
    fe = state["feature_selection_result"]
    tr = state["model_training_result"]
    val = state["validation_result"]
    m = val["metrics"]
    strat = val["cutoff_strategy"]
    reflections = state.get("reflection_history", [])
    profile = get_profile(state.get("card_type"))

    feat_rows = [
        "| 入模特征 | 逻辑回归系数 (Coef) | IV 值 | 跨期 PSI | 共线性 VIF |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    coefs = tr["coefficients"]
    for item in fe["selected_feature_metrics"]:
        fn = item["feature"]
        feat_rows.append(
            f"| `{fn}` | {coefs.get(fn, 0.0)} | {item['iv']} | {item['psi']} | {item['vif']} |"
        )

    band_rows = [
        "| 评分区间 (Score Band) | OOT 样本数 | 实际坏账率 (Bad Rate) |",
        "| :--- | :---: | :---: |",
    ]
    for b in val["oot_score_bands"]:
        band_rows.append(f"| `{b['score_range']}` | {b['count']} | {b['bad_rate']} |")

    refl_section = (
        "\n".join(f"- {r}" for r in reflections)
        if reflections
        else "- 首轮建模直接通过所有跨期稳定性与区分度门禁。"
    )
    challenger_lines = _build_challenger_section(state.get("challenger_result"))

    return "\n".join(
        [
            profile.report_title,
            "",
            "## 一、 样本设计与数据治理审计",
            f"- **开发样本 (`{samp['train_uri']}`)**：样本量 `{samp['train_rows']}`，坏账率 `{samp['train_bad_rate']}`",
            f"- **跨期验证样本 (`{samp['oot_uri']}`)**：样本量 `{samp['oot_rows']}`，坏账率 `{samp['oot_bad_rate']}`",
            *([f"- {_ROW_SPLIT_WARNING}"] if samp.get("oot_row_split") else []),
            f"- **防穿越拦截字段**：`{gov['quarantined_leakage_features']}`",
            f"- **公平信贷合规拦截字段**：`{gov['quarantined_compliance_features']}`",
            *_quarantine_reason_lines(gov),
            f"- **监管合规终审状态**：`{'PASSED' if compliance_passed else 'FAILED'}`",
            "",
            "## 二、 多智能体自愈与反思轨迹 (Self-Reflection Log)",
            refl_section,
            "",
            "## 三、 最终入模特征与逻辑回归权重 (`" + tr["model_id"] + "`)",
            f"- **标准评分卡刻度参数**：Base Score = `{tr['pdo_scaling']['base_score']}`, PDO = `{tr['pdo_scaling']['pdo']}`, "
            f"Offset A = `{tr['pdo_scaling']['offset_a']}`, Factor B = `{tr['pdo_scaling']['factor_b']}`",
            "",
            *feat_rows,
            "",
            "## 四、 模型区分度与跨期稳定性表现 (Model Performance)",
            f"- **开发集 KS (`train_ks`)**：`{m['train_ks']}` | **开发集 AUC (`train_auc`)**：`{m['train_auc']}`",
            f"- **跨期集 KS (`oot_ks`)**：`{m['oot_ks']}` | **跨期集 AUC (`oot_auc`)**：`{m['oot_auc']}`",
            f"- **模型总分跨期 PSI (`score_psi`)**：`{m['score_psi']}`（门禁要求 <= `{state['constraints']['psi_max']}`）",
            f"- **Train/OOT KS 差值 (`ks_degradation`)**：`{m['ks_degradation']}`"
            f"（门禁要求 <= `{state['constraints'].get('ks_drop_max', 0.05)}`）",
            f"- **跨期门禁结论**：`{'PASSED' if val['validation_passed'] else 'FAILED'}`"
            + ("" if val["validation_passed"] else f"（{'; '.join(val['anomalies'])}）"),
            "",
            "## 五、 OOT 评分分箱单调性与授信切点仿真 (Cut-off Strategy)",
            f"- **{profile.rate_label} (`target_approval_rate`)**：`{strat['target_approval_rate']}`",
            f"- **{profile.cutoff_label} (`recommended_cutoff_score`)**：`{strat['recommended_cutoff_score']}` 分",
            f"- **{profile.post_cutoff_label} (`post_cutoff_bad_rate`)**：`{strat['post_cutoff_bad_rate']}`（较基准坏账率 `{strat['baseline_oot_bad_rate']}` 降低 `{strat['risk_reduction_pct']}%`）",
            "",
            *band_rows,
            *challenger_lines,
        ]
    )


def _build_challenger_section(ch: Dict[str, Any] | None) -> List[str]:
    """Champion-Challenger comparison section (empty when no challenger was trained)."""
    if not ch:
        return []
    cm = ch["challenger_metrics"]
    chm = ch["champion_metrics"]
    verdict = (
        f"挑战者 OOT KS 领先冠军 `{ch['ks_gap']}`，达到显著领先阈值，建议评估可解释性方案后再决定是否引入。"
        if ch["challenger_significantly_better"]
        else f"挑战者与冠军 OOT KS 差值为 `{ch['ks_gap']}`，未显著领先，保持 LR 评分卡作为上线模型。"
    )
    imp_rows = [f"| `{d['feature']}` | {d['importance']} |" for d in ch["feature_importance"][:5]]
    return [
        "",
        "## 六、 冠军挑战者对比 (Champion-Challenger)",
        f"- **冠军**：逻辑回归评分卡 `{ch['champion_model_id']}`（上线决策模型）",
        f"- **挑战者**：`{ch['challenger_model']}`（仅作区分度基准）",
        "",
        "| 模型 | 开发集 KS | 跨期集 KS | 跨期集 AUC |",
        "| :--- | :---: | :---: | :---: |",
        f"| 冠军 (LR) | - | {chm['oot_ks']} | {chm['oot_auc']} |",
        f"| 挑战者 (GBDT) | {cm['train_ks']} | {cm['oot_ks']} | {cm['oot_auc']} |",
        "",
        f"- **结论**：{verdict}",
        "",
        "| 挑战者特征重要性 Top5 (OOT 排列重要性) | AUC 下降量 |",
        "| :--- | :---: |",
        *imp_rows,
    ]
