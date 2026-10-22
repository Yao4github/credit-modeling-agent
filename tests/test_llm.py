import json

import httpx

from app.agents.specialists import (
    _llm_diagnose_reflection,
    compliance_reporter_node,
    orchestrator_node,
)
from app.core.config import LLMConfig
from app.core.llm import LLMClient, reset_llm_client


def _cfg(**kw):
    base = dict(
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key="sk-test",
        timeout_s=5,
        max_retries=1,
        enabled=True,
    )
    base.update(kw)
    return LLMConfig(**base)


def _client(handler, **kw):
    return LLMClient(_cfg(**kw), transport=httpx.MockTransport(handler))


def _ok(content):
    return httpx.Response(
        200,
        json={
            "model": "deepseek-chat",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def test_config_defaults_to_deepseek(monkeypatch):
    for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_ENABLED", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    cfg = LLMConfig.from_env()
    assert (cfg.provider, cfg.model, cfg.base_url) == ("deepseek", "deepseek-chat", "https://api.deepseek.com")
    assert cfg.usable


def test_config_switch_provider(monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "1")
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "qwen-max")
    cfg = LLMConfig.from_env()
    assert cfg.provider == "qwen" and cfg.model == "qwen-max" and "dashscope" in cfg.base_url


def test_chat_parses_usage_and_url():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        return _ok("hi")

    r = _client(handler).chat([{"role": "user", "content": "x"}])
    assert r.content == "hi" and r.prompt_tokens == 11 and r.completion_tokens == 7
    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["auth"] == "Bearer sk-test"


def test_retry_on_5xx_then_success():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else _ok("ok")

    assert _client(handler).chat([{"role": "user", "content": "x"}]).content == "ok"
    assert calls["n"] == 2


def test_chat_json_bad_output_returns_none():
    assert _client(lambda r: _ok("not json")).chat_json([{"role": "user", "content": "x"}]) is None


def test_unavailable_without_key():
    assert not _client(lambda r: _ok("x"), api_key=None).available


def test_orchestrator_uses_llm_and_clamps_constraints():
    payload = {"intent": "full_modeling", "constraints": {"ks_min": 0.35, "psi_max": 9, "bogus": 1}}
    reset_llm_client(_client(lambda r: _ok(json.dumps(payload))))
    out = orchestrator_node({"user_query": "帮我做个A卡，KS至少0.35"})
    assert out["intent_type"] == "full_modeling"
    assert out["constraints"]["ks_min"] == 0.35
    assert out["constraints"]["psi_max"] == 0.10  # out-of-range value rejected
    assert "bogus" not in out["constraints"]
    assert out["agent_logs"][0]["prompt_tokens"] == 11


def test_orchestrator_falls_back_on_llm_failure():
    reset_llm_client(_client(lambda r: httpx.Response(401), max_retries=0))
    out = orchestrator_node({"user_query": "仅探查数据，检查穿越"})
    assert out["intent_type"] == "data_audit_only"


def test_validator_reflection_llm_guardrails_and_tokens():
    payload = {
        "drop_features": ["night_apply_ratio_7d", "hallucinated_feature_xyz"],
        "diagnosis": "渠道投放切换引起夜间申请特征严重跨期漂移",
    }
    reset_llm_client(_client(lambda r: _ok(json.dumps(payload))))
    drops, msg, resp = _llm_diagnose_reflection(
        model_id="sc_lr_001",
        anomalies=["OOT PSI > 0.10"],
        mandatory_drops=["night_apply_ratio_7d"],
        selected_features=["credit_inq_cnt_3m", "night_apply_ratio_7d"],
        memories=[{"lesson_learned": "night_apply_ratio_7d 易漂移"}],
        fallback_msg="Fallback reason",
    )
    assert drops == ["night_apply_ratio_7d"]
    assert "hallucinated_feature_xyz" not in drops
    assert "[LLM诊断: 渠道投放切换引起夜间申请特征严重跨期漂移]" in msg
    assert resp is not None and resp.prompt_tokens == 11 and resp.completion_tokens == 7


def test_compliance_reporter_injects_llm_commentary_and_tokens():
    reset_llm_client(_client(lambda r: _ok("各项KS与PSI指标均符合巴塞尔协议与内部风控红线，准予投产。")))
    out = compliance_reporter_node(
        {
            "intent_type": "knowledge_qa",
            "user_query": "什么是数据穿越？",
            "rag_context": [],
            "agent_logs": [],
        }
    )
    assert "零幻觉解答 (LLM: deepseek-chat)" in out["final_report_markdown"]
    assert "准予投产" in out["final_report_markdown"]
    assert out["agent_logs"][0]["prompt_tokens"] == 11
    assert out["agent_logs"][0]["completion_tokens"] == 7


def test_compliance_reporter_executive_commentary_and_fallback():
    state_audit = {
        "intent_type": "data_audit_only",
        "data_inspection_result": {
            "sample_summary": {"train_uri": "ds://train", "oot_uri": "ds://oot", "train_rows": 4000, "oot_rows": 2000, "train_bad_rate": 0.05, "oot_bad_rate": 0.05},
            "governance_audit": {"quarantined_leakage_features": ["post_loan_repay_amt_m1"], "quarantined_compliance_features": ["applicant_gender_code"], "clean_candidate_features": ["f1"]},
            "window_validation": {"train_window": "2025Q1", "oot_window": "2025Q2", "is_chronological_valid": True, "perf_window_months": 3},
        },
        "agent_logs": [],
    }
    reset_llm_client(_client(lambda r: _ok("已拦截全部穿越特征，数据治理合规通过。")))
    out_ok = compliance_reporter_node(state_audit)
    assert "专家评审结论与上线建议 (LLM: deepseek-chat)" in out_ok["final_report_markdown"]
    assert out_ok["agent_logs"][0]["prompt_tokens"] == 11

    reset_llm_client(_client(lambda r: httpx.Response(500), max_retries=0))
    out_fb = compliance_reporter_node(state_audit)
    assert "专家评审结论与上线建议" not in out_fb["final_report_markdown"]
    assert out_fb["agent_logs"][0]["prompt_tokens"] == 560


