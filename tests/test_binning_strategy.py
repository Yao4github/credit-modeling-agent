import numpy as np
import pandas as pd

from app.agents import specialists
from app.harness.binning_policy import choose_binning_strategy
from app.tools.credit_metrics import compute_woe_iv_and_psi

C = {"ks_min": 0.30, "ks_drop_max": 0.05}
D = {"non_monotonic_ratio": 0.0, "min_bin_share": 0.2}


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(x * 1.2 - 2)))).astype(int)
    return pd.Series(x), pd.Series(y)


def test_default_quantile_unchanged_and_methods_differ():
    x, y = _data()
    base = compute_woe_iv_and_psi(x, y, x)
    assert len(base[4]) == 5
    assert len(compute_woe_iv_and_psi(x, y, x, n_bins=10)[4]) == 10
    tree = compute_woe_iv_and_psi(x, y, x, method="tree")
    rates = [b["train_bad_rate"] for b in tree[4]]
    assert 2 <= len(rates) <= 5 and (rates == sorted(rates) or rates == sorted(rates, reverse=True))


def test_tree_edges_fit_on_train_only():
    x, y = _data()
    oot = pd.Series(np.full(100, 1e6))  # extreme OOT must not move the edges
    a = compute_woe_iv_and_psi(x, y, x, method="tree")
    b = compute_woe_iv_and_psi(x, y, oot, method="tree")
    assert [d["bin"] for d in a[4]] == [d["bin"] for d in b[4]]


def test_policy_overfit_picks_tree_then_gives_up():
    m = {"train_ks": 0.50, "oot_ks": 0.35}
    assert choose_binning_strategy(m, C, D, ["quantile5"])[0] == "tree"
    assert choose_binning_strategy(m, C, D, ["quantile5", "tree"])[0] is None


def test_policy_underfit_picks_10_bins_unless_bins_too_small():
    m = {"train_ks": 0.20, "oot_ks": 0.18}
    assert choose_binning_strategy(m, C, D, ["quantile5"])[0] == "quantile10"
    assert choose_binning_strategy(m, C, {**D, "min_bin_share": 0.04}, ["quantile5"])[0] is None
    assert choose_binning_strategy(m, C, D, ["quantile5", "quantile10"])[0] is None


def test_policy_non_monotonic_tiebreak_and_no_action():
    ok = {"train_ks": 0.40, "oot_ks": 0.38}
    assert choose_binning_strategy(ok, C, {**D, "non_monotonic_ratio": 0.5}, ["quantile5"])[0] == "tree"
    assert choose_binning_strategy(ok, C, D, ["quantile5"])[0] is None


def _validator_state():
    return {
        "constraints": {"ks_min": 0.30, "ks_drop_max": 0.05, "psi_max": 0.10, "target_approval_rate": 0.75},
        "model_training_result": {"model_id": "m1", "negative_coef_features": []},
        "dataset_refs": {"train_uri": "t", "oot_uri": "o"},
        "selected_features": ["f1"],
        "feature_selection_result": {"binning_diagnostics": D},
        "reflection_count": 0,
        "max_reflections": 2,
        "agent_logs": [],
    }


def test_validator_switches_binning_instead_of_stopping(monkeypatch):
    res = {
        "validation_passed": False,
        "anomalies": ["KS gap"],
        "drifting_features": [],
        "metrics": {"train_ks": 0.5, "oot_ks": 0.35, "oot_auc": 0.7, "score_psi": 0.01},
    }
    monkeypatch.setattr(specialists, "tool_evaluate_scorecard_and_strategy", lambda inp: res)
    monkeypatch.setattr(specialists.MemoryManager, "get_instance", staticmethod(lambda: type("M", (), {"record_reflection_experience": lambda *a, **k: None})()))
    out = specialists.validator_node(_validator_state())
    assert out["binning_strategy"] == "tree"
    assert out["stop_reflection"] is False and out["reflection_count"] == 1
    assert "切换分箱" in out["reflection_history"][0]

    st = _validator_state() | {"binning_strategy": "tree", "binning_tried": ["quantile5", "tree"]}
    out2 = specialists.validator_node(st)
    assert out2["stop_reflection"] is True and "binning_strategy" not in out2


def test_validator_stops_when_rollback_budget_exhausted(monkeypatch):
    res = {"validation_passed": False, "anomalies": ["KS gap"], "drifting_features": [], "metrics": {"train_ks": 0.5, "oot_ks": 0.35, "oot_auc": 0.7, "score_psi": 0.01}}
    monkeypatch.setattr(specialists, "tool_evaluate_scorecard_and_strategy", lambda inp: res)
    out = specialists.validator_node(_validator_state() | {"reflection_count": 2})
    assert out["stop_reflection"] is True and "binning_strategy" not in out
