import json

from app.agents.specialists import compliance_reporter_node
from app.core.llm import reset_llm_client
from app.harness.guards import clamp_constraints, numbers_grounded, whitelist_union
from tests.test_llm import _client, _ok


def test_clamp_constraints():
    assert clamp_constraints({"ks_min": 0.35, "psi_max": 9, "bogus": 1, "iv_min": True}) == {"ks_min": 0.35}
    assert clamp_constraints("x") == {}


def test_whitelist_union_keeps_mandatory_and_filters_unknown():
    assert whitelist_union(["a"], ["b", "zzz", 3], ["a", "b"]) == ["a", "b"]
    assert whitelist_union(["a"], None, ["a"]) == ["a"]


def test_numbers_grounded():
    assert numbers_grounded("KS 为 0.35，共 4000 行", "ks=0.350 rows 4000") == (True, [])
    assert numbers_grounded("KS 为 0.99", "ks=0.35") == (False, ["0.99"])


def _audit_state():
    return {
        "intent_type": "data_audit_only",
        "data_inspection_result": {
            "sample_summary": {"train_uri": "ds://train", "oot_uri": "ds://oot", "train_rows": 4000, "oot_rows": 2000, "train_bad_rate": 0.05, "oot_bad_rate": 0.05},
            "governance_audit": {"quarantined_leakage_features": [], "quarantined_compliance_features": [], "clean_candidate_features": ["f1"]},
            "window_validation": {"train_window": "2025Q1", "oot_window": "2025Q2", "is_chronological_valid": True, "perf_window_months": 3},
        },
        "agent_logs": [],
    }


def test_commentary_with_fabricated_number_is_blocked_and_logged():
    reset_llm_client(_client(lambda r: _ok("开发集 4000 行，坏账率 12.34%，建议投产。")))
    out = compliance_reporter_node(_audit_state())
    assert "专家评审结论" not in out["final_report_markdown"]
    assert "数值护栏" in out["agent_logs"][0]["thought"]


def test_commentary_with_grounded_number_kept():
    reset_llm_client(_client(lambda r: _ok("开发集 4000 行，数据治理合规。")))
    out = compliance_reporter_node(_audit_state())
    assert "专家评审结论" in out["final_report_markdown"]
