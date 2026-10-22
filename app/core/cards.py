"""Card-type profiles (A: application, B: behavior, C: collection scorecards).

Label polarity is unified across cards: target == 1 is the adverse event (A/B: default; C: not recovered),
so WOE/LR/PDO/KS/cut-off machinery is shared and a higher score always means lower risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class CardProfile:
    card_type: str
    display_name: str
    report_title: str
    train_uri: str
    oot_uri: str
    target_col: str
    default_ks_min: float
    prohibited_features: FrozenSet[str]
    rag_query: str
    keywords: Tuple[str, ...]
    rate_label: str
    cutoff_label: str
    post_cutoff_label: str


CARD_PROFILES = {
    "A": CardProfile(
        card_type="A",
        display_name="A卡(申请评分)",
        report_title="# 🏦 信贷风控申请评分卡 (A-Card) 模型开发与验证报告",
        train_uri="ds://credit_a_card_train",
        oot_uri="ds://credit_a_card_oot",
        target_col="target_mob3_dpd30",
        default_ks_min=0.30,
        prohibited_features=frozenset({"applicant_gender_code", "post_loan_repay_amt_m1"}),
        rag_query="数据穿越 post_loan_repay_amt_m1 合规黑名单 applicant_gender_code",
        keywords=(),
        rate_label="目标通过率",
        cutoff_label="建议准入切点分",
        post_cutoff_label="切点后预计通过坏账率",
    ),
    "B": CardProfile(
        card_type="B",
        display_name="B卡(行为评分)",
        report_title="# 🏦 信贷风控行为评分卡 (B-Card) 模型开发与验证报告",
        train_uri="ds://credit_b_card_train",
        oot_uri="ds://credit_b_card_oot",
        target_col="target_fut3m_dpd30",
        default_ks_min=0.30,
        prohibited_features=frozenset({"applicant_gender_code", "fut_m1_overdue_amt"}),
        rag_query="B卡 行为评分 存量客户 fut_m1_overdue_amt 数据穿越 合规黑名单",
        keywords=("b卡", "行为评分", "behavior"),
        rate_label="目标通过率",
        cutoff_label="建议准入切点分",
        post_cutoff_label="切点后预计通过坏账率",
    ),
    "C": CardProfile(
        card_type="C",
        display_name="C卡(催收评分)",
        report_title="# 🏦 信贷风控催收评分卡 (C-Card) 模型开发与验证报告",
        train_uri="ds://credit_c_card_train",
        oot_uri="ds://credit_c_card_oot",
        target_col="target_unrecovered_d30",
        default_ks_min=0.25,
        prohibited_features=frozenset({"applicant_gender_code", "post_collection_repay_amt_d7"}),
        rag_query="C卡 催收评分 逾期账户 post_collection_repay_amt_d7 数据穿越 合规黑名单",
        keywords=("c卡", "催收", "collection"),
        rate_label="目标策略覆盖率",
        cutoff_label="建议策略切点分",
        post_cutoff_label="切点后覆盖账户预计未回款率",
    ),
}


def get_profile(card_type: Optional[str]) -> CardProfile:
    """Returns the profile for a card type; unknown/empty values fall back to the A-card."""
    return CARD_PROFILES.get((card_type or "A").upper(), CARD_PROFILES["A"])


def detect_card_type(query: str) -> str:
    """Rule-based card detection from natural language (C is checked before B); default is A."""
    q = query.lower()
    for card in ("C", "B"):
        if any(k in q for k in CARD_PROFILES[card].keywords):
            return card
    return "A"
