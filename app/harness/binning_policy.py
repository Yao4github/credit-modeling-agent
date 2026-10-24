"""Deterministic choice of the next binning strategy when validation fails with no feature to drop."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

DEFAULT_BINNING = "quantile5"
NON_MONOTONIC_RATIO_MAX = 0.3  # above this share of non-monotonic selected features, binning noise is suspected
MIN_BIN_SHARE_AFTER_SPLIT = 0.03  # a 10-bin split must keep every bin at least this share of Train


def choose_binning_strategy(
    metrics: Dict[str, Any],
    constraints: Dict[str, float],
    diagnostics: Dict[str, Any],
    tried: Iterable[str],
) -> Tuple[Optional[str], str]:
    """Returns (next_strategy | None, reason). Each strategy is tried at most once; None means give up."""
    tried_set = set(tried)
    train_ks, oot_ks = float(metrics["train_ks"]), float(metrics["oot_ks"])
    ks_min = constraints["ks_min"]
    gap = train_ks - oot_ks
    gap_max = constraints.get("ks_drop_max", 0.05)
    non_mono = float(diagnostics.get("non_monotonic_ratio", 0.0))
    min_share = float(diagnostics.get("min_bin_share", 1.0))
    evidence = f"Train_KS={train_ks}, OOT_KS={oot_ks}, 差值={gap:.4f}, 非单调特征占比={non_mono:.0%}, 最小箱占比={min_share:.2%}"

    if gap > gap_max and oot_ks < train_ks:
        if "tree" not in tried_set:
            return "tree", f"过拟合（Train/OOT KS 差值 > {gap_max}），改用决策树单调分箱。{evidence}"
    elif train_ks < ks_min and oot_ks < ks_min:
        if "quantile10" not in tried_set:
            if min_share / 2 >= MIN_BIN_SHARE_AFTER_SPLIT:
                return "quantile10", f"欠拟合（Train/OOT KS 均低于 {ks_min}），改用10等频分箱。{evidence}"
            return None, f"欠拟合但最小箱样本占比过低，不宜加细分箱。{evidence}"
    elif non_mono > NON_MONOTONIC_RATIO_MAX and "tree" not in tried_set:
        return "tree", f"分箱噪声（非单调特征占比 > {NON_MONOTONIC_RATIO_MAX:.0%}），改用决策树单调分箱。{evidence}"
    return None, f"无可行的分箱策略。{evidence}"
