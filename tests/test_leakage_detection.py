"""Tests for automatic leakage / compliance feature detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.cards import get_profile
from app.graph.workflow import run_credit_modeling_workflow
from app.tools.credit_metrics import DataInspectionInput, tool_inspect_dataset_and_check_leakage
from app.tools.data_registry import DataPointerRegistry
from app.tools.leakage_rules import _proxy_reason, detect_risky_features

_REG = dict(obs_start_date="2025-01-01", obs_end_date="2025-06-30")


def _detect(columns, df=None, target="y"):
    df = df if df is not None else pd.DataFrame({c: [0.0, 1.0, 2.0, 3.0] for c in columns} | {target: [0, 1, 0, 1]})
    return detect_risky_features(df, target, columns, [], [])


@pytest.mark.parametrize(
    "column,kind",
    [
        ("post_loan_repay_amt_m1", "leak"),
        ("fut_m1_overdue_amt", "leak"),
        ("next_month_dpd", "leak"),
        ("repay_after_call", "leak"),
        ("Future_Balance", "leak"),
        ("applicant_gender_code", "comp"),
        ("SEX", "comp"),
        ("ethnic_group", "comp"),
        ("native_place_code", "comp"),
    ],
)
def test_name_rules_flag_by_whole_token(column, kind):
    leak, comp = _detect([column])
    assert (column in leak) == (kind == "leak") and (column in comp) == (kind == "comp")


@pytest.mark.parametrize(
    "column", ["postal_code", "essex_region", "debt_to_income_ratio", "credit_inq_cnt_3m", "nextgen_score", "sexton_idx"]
)
def test_name_rules_do_not_misfire_on_substrings(column):
    assert _detect([column]) == ({}, {})


def test_blacklist_takes_priority_and_is_reported():
    df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "y": [0, 1, 0, 1]})
    leak, comp = detect_risky_features(df, "y", ["a"], ["a"], [])
    assert leak == {"a": "知识库黑名单"} and comp == {}


def test_proxy_reason_edge_cases():
    y = pd.Series([0, 1] * 50)
    assert _proxy_reason(pd.Series([1.0] * 100), y) is None  # constant column
    assert _proxy_reason(pd.Series(np.arange(100.0)), pd.Series([1] * 100)) is None  # single-class label
    nan_x = pd.Series([np.nan] * 5 + list(y[5:] * 10.0))
    assert "统计红旗" in _proxy_reason(nan_x, y)  # NaNs are ignored, perfect proxy still caught
    assert _proxy_reason(pd.Series(np.random.default_rng(0).normal(size=100)), y) is None


def test_non_numeric_columns_are_skipped_by_statistical_check():
    df = pd.DataFrame({"city": ["a", "b"] * 50, "y": [0, 1] * 50})
    assert detect_risky_features(df, "y", ["city"], [], []) == ({}, {})


def test_synthetic_cards_keep_their_quarantine_results_with_reasons():
    reg = DataPointerRegistry.get_instance()
    expected = {
        "A": ("post_loan_repay_amt_m1", "applicant_gender_code"),
        "B": ("fut_m1_overdue_amt", "applicant_gender_code"),
        "C": ("post_collection_repay_amt_d7", "applicant_gender_code"),
    }
    for card, (leak, comp) in expected.items():
        p = get_profile(card)
        gov = tool_inspect_dataset_and_check_leakage(
            DataInspectionInput(train_uri=p.train_uri, oot_uri=p.oot_uri, card_type=card)
        )["governance_audit"]
        assert gov["quarantined_leakage_features"] == [leak]
        assert gov["quarantined_compliance_features"] == [comp]
        assert set(gov["detection_reasons"]) == {leak, comp}
        assert len(gov["clean_candidate_features"]) == gov["raw_feature_count"] - 2
    assert reg.get_meta("ds://credit_a_card_train").target_col == "target_mob3_dpd30"


def test_external_dataset_statistical_proxy_is_quarantined_and_reported(tmp_path):
    reg = DataPointerRegistry.get_instance()
    rng = np.random.default_rng(7)
    for name, src in (("ext_leak_train", "ds://credit_a_card_train"), ("ext_leak_oot", "ds://credit_a_card_oot")):
        df = reg.get_df(src).drop(columns=["post_loan_repay_amt_m1", "applicant_gender_code"]).copy()
        tgt = df["target_mob3_dpd30"]
        df["settle_flag_score"] = tgt * 10.0 + rng.normal(0, 0.1, len(df))  # near-perfect label proxy
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    reg.register_from_file("ds://ext_leak_train", "ext_leak_train.csv", "target_mob3_dpd30", **_REG)
    reg.register_from_file(
        "ds://ext_leak_oot", "ext_leak_oot.csv", "target_mob3_dpd30",
        obs_start_date="2025-07-01", obs_end_date="2025-09-30",
    )

    gov = tool_inspect_dataset_and_check_leakage(
        DataInspectionInput(train_uri="ds://ext_leak_train", oot_uri="ds://ext_leak_oot")
    )["governance_audit"]
    assert gov["quarantined_leakage_features"] == ["settle_flag_score"]
    assert "统计红旗" in gov["detection_reasons"]["settle_flag_score"]
    assert "credit_inq_cnt_3m" in gov["clean_candidate_features"]  # strong legit feature (AUC ~0.87) is kept

    state = run_credit_modeling_workflow(
        "仅探查数据，检查穿越", session_id="pytest-leak-ext", train_uri="ds://ext_leak_train", oot_uri="ds://ext_leak_oot"
    )
    report = state["final_report_markdown"]
    assert "拦截原因明细" in report and "settle_flag_score" in report and "统计红旗" in report
