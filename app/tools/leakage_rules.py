"""Automatic detection of data-leakage and compliance-risk features.

Detection layers (all hits are quarantined):
1. Knowledge-base blacklists (exact column names).
2. Column-name rules: whole-token matches after splitting on non-alphanumerics, so `postal_code` is NOT
   treated as `post`, while `post_loan_repay_amt_m1` / `fut_m1_overdue_amt` are.
3. Statistical red flag: a feature that is a near-perfect proxy of the label (single-feature AUC or absolute
   Pearson correlation on the training set at or above the thresholds below).

Known limits: subtle leakage (single-feature AUC < PROXY_AUC_THRESHOLD) and leakage that can only be judged from
event timing are NOT detected; declare those in the knowledge base via `leakage_blacklist`.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LEAKAGE_TOKENS = frozenset({"post", "fut", "future", "after", "next", "label", "outcome"})
COMPLIANCE_TOKENS = frozenset({"gender", "sex", "race", "ethnic", "religion", "nationality", "native"})
BLACKLIST_REASON = "知识库黑名单"

# Real feature strengths in credit data rarely approach these values; near-perfect proxies of the label do.
PROXY_AUC_THRESHOLD = 0.95
PROXY_CORR_THRESHOLD = 0.90


def _tokens(column: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", column.lower()) if t]


def _name_hit(column: str, vocabulary: frozenset) -> str | None:
    for token in _tokens(column):
        if token in vocabulary:
            return token
    return None


def _proxy_reason(x: pd.Series, y: pd.Series) -> str | None:
    """Returns a reason when `x` is a near-perfect proxy of the binary label `y`, else None."""
    mask = x.notna() & y.notna()
    x, y = x[mask].astype(float), y[mask]
    if y.nunique() < 2 or x.nunique() < 2:
        return None
    auc = roc_auc_score(y, x)
    auc = max(auc, 1.0 - auc)
    corr = abs(float(np.corrcoef(x, y)[0, 1]))
    if auc >= PROXY_AUC_THRESHOLD:
        return f"统计红旗: 单特征 AUC {auc:.3f} >= {PROXY_AUC_THRESHOLD}，疑似标签代理"
    if corr >= PROXY_CORR_THRESHOLD:
        return f"统计红旗: 与标签相关系数 {corr:.3f} >= {PROXY_CORR_THRESHOLD}，疑似标签代理"
    return None


def detect_risky_features(
    df: pd.DataFrame,
    target_col: str,
    features: Iterable[str],
    leakage_blacklist: Iterable[str],
    compliance_blacklist: Iterable[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Returns (leakage {feature: reason}, compliance {feature: reason}) in the order of `features`."""
    leak_bl, comp_bl = set(leakage_blacklist), set(compliance_blacklist)
    leakage: Dict[str, str] = {}
    compliance: Dict[str, str] = {}
    for col in features:
        if col in comp_bl:
            compliance[col] = BLACKLIST_REASON
        elif (tok := _name_hit(col, COMPLIANCE_TOKENS)) is not None:
            compliance[col] = f"命名规则: 敏感词元 `{tok}`"

        if col in leak_bl:
            leakage[col] = BLACKLIST_REASON
        elif (tok := _name_hit(col, LEAKAGE_TOKENS)) is not None:
            leakage[col] = f"命名规则: 穿越词元 `{tok}`"
        elif col not in compliance and pd.api.types.is_numeric_dtype(df[col]):
            reason = _proxy_reason(df[col], df[target_col])
            if reason:
                leakage[col] = reason
    return leakage, compliance


def is_exemptable(reason: str) -> bool:
    """Only name-rule / statistical leakage hits can be exempted; blacklist hits are institutional red lines."""
    return reason != BLACKLIST_REASON


def corpus_blacklists() -> Tuple[List[str], List[str]]:
    """Leakage and compliance blacklists aggregated from the whole knowledge corpus (built-in + extra)."""
    from app.tools.rag_retriever import KNOWLEDGE_CORPUS, load_extra_knowledge

    leakage: List[str] = []
    compliance: List[str] = []
    for chunk in KNOWLEDGE_CORPUS + load_extra_knowledge():
        leakage.extend(chunk.metadata.get("leakage_blacklist", []))
        compliance.extend(chunk.metadata.get("compliance_blacklist", []))
    return leakage, compliance
