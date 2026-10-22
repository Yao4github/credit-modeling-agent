"""Quantitative Credit Risk Modeling Sandbox Tools (WOE/IV, PSI, VIF, Logistic Scorecard, KS/AUC, Cutoff Strategy).

All tools operate on dataset URI pointers (`ds://...`) registered in `DataPointerRegistry`
and return structured JSON summaries suitable for LLM Function Calling and deterministic verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from app.core.cards import get_profile
from app.tools.data_registry import DataPointerRegistry
from app.tools.leakage_exemptions import get_active_exemptions
from app.tools.leakage_rules import detect_risky_features, is_exemptable
from app.tools.rag_retriever import HybridCreditRAG


# ---------------------------------------------------------------------------
# Pydantic Schemas for Tool Calling Validation
# ---------------------------------------------------------------------------

class DataInspectionInput(BaseModel):
    train_uri: str = Field(default="ds://credit_a_card_train", description="开发集数据指针 URI")
    oot_uri: str = Field(default="ds://credit_a_card_oot", description="跨期验证集(OOT)数据指针 URI")
    card_type: str = Field(default="A", description="卡种 A/B/C，用于选取知识库检索词")


class FeatureSelectionInput(BaseModel):
    train_uri: str = Field(default="ds://credit_a_card_train", description="开发集数据指针 URI")
    oot_uri: str = Field(default="ds://credit_a_card_oot", description="跨期验证集数据指针 URI")
    candidate_features: List[str] = Field(description="待评估的候选特征名列表")
    iv_threshold: float = Field(default=0.02, description="最小IV准入阈值 (默认0.02)")
    psi_threshold: float = Field(default=0.10, description="最大单特征跨期PSI准入阈值 (默认0.10)")
    vif_threshold: float = Field(default=5.0, description="最大多重共线性VIF阈值 (默认5.0)")
    exclude_features: List[str] = Field(default_factory=list, description="强制剔除的黑名单或漂移特征列表")


class ScorecardTrainInput(BaseModel):
    train_woe_uri: str = Field(default="ds://credit_a_card_train_woe", description="WOE转换后的开发集指针")
    oot_woe_uri: str = Field(default="ds://credit_a_card_oot_woe", description="WOE转换后的OOT集指针")
    selected_features: List[str] = Field(description="入模特征列表")
    base_score: int = Field(default=600, description="评分卡基准分 (默认600)")
    base_odds: float = Field(default=20.0, description="基准好坏比 Good:Bad (默认20.0)")
    pdo: int = Field(default=20, description="好坏比翻倍分数 PDO (默认20)")


class ModelValidationInput(BaseModel):
    model_id: str = Field(description="已训练评分卡模型的唯一ID")
    ks_threshold: float = Field(default=0.30, description="最低KS达标阈值")
    ks_drop_threshold: float = Field(default=0.05, description="Train与OOT的KS最大允许差值 (SOP-001: 0.05)")
    psi_threshold: float = Field(default=0.10, description="最大模型/特征PSI稳定阈值")
    target_approval_rate: float = Field(default=0.75, description="目标通过率 (用于测算最佳Cut-off切点)")
    train_uri: str = Field(default="ds://credit_a_card_train", description="开发集原始数据指针 URI (用于特征级PSI复核)")
    oot_uri: str = Field(default="ds://credit_a_card_oot", description="跨期验证集原始数据指针 URI (用于特征级PSI复核)")


class ChallengerTrainInput(BaseModel):
    champion_model_id: str = Field(description="已通过门禁的冠军(LR评分卡)模型ID")
    train_woe_uri: str = Field(description="WOE转换后的开发集指针")
    oot_woe_uri: str = Field(description="WOE转换后的OOT集指针")
    selected_features: List[str] = Field(description="与冠军模型一致的入模特征列表")


# ---------------------------------------------------------------------------
# Core Mathematical Operators for Credit Risk Modeling
# ---------------------------------------------------------------------------

def compute_woe_iv_and_psi(
    train_series: pd.Series,
    train_target: pd.Series,
    oot_series: pd.Series,
    n_bins: int = 5,
) -> Tuple[pd.Series, pd.Series, float, float, List[Dict[str, Any]]]:
    """Computes quantile bin edges on Train, calculates WOE/IV on Train, and applies edges to OOT for PSI."""
    eps = 1e-6
    _, bin_edges = pd.qcut(train_series.rank(method="first"), q=n_bins, retbins=True, duplicates="drop")
    # Map back to actual value quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(train_series.dropna(), quantiles))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    train_bins = pd.cut(train_series, bins=edges, include_lowest=True)
    oot_bins = pd.cut(oot_series, bins=edges, include_lowest=True)

    total_bad = max(float(train_target.sum()), 1.0)
    total_good = max(float((1 - train_target).sum()), 1.0)
    total_train = float(len(train_series))
    total_oot = float(len(oot_series))

    woe_map: Dict[Any, float] = {}
    iv_total = 0.0
    psi_total = 0.0
    bin_details: List[Dict[str, Any]] = []

    for cat in train_bins.cat.categories:
        mask_tr = train_bins == cat
        mask_oot = oot_bins == cat
        n_tr = float(mask_tr.sum())
        n_oot = float(mask_oot.sum())
        bad_tr = float(train_target[mask_tr].sum())
        good_tr = n_tr - bad_tr

        dist_bad = max(bad_tr / total_bad, eps)
        dist_good = max(good_tr / total_good, eps)
        woe_val = float(np.log(dist_bad / dist_good))
        iv_bin = (dist_bad - dist_good) * woe_val
        iv_total += iv_bin

        prop_tr = max(n_tr / total_train, eps)
        prop_oot = max(n_oot / total_oot, eps)
        psi_bin = (prop_oot - prop_tr) * float(np.log(prop_oot / prop_tr))
        psi_total += psi_bin

        woe_map[cat] = woe_val
        bin_details.append(
            {
                "bin": str(cat),
                "train_cnt": int(n_tr),
                "train_bad_rate": round(bad_tr / max(n_tr, 1.0), 4),
                "woe": round(woe_val, 4),
                "iv_contrib": round(iv_bin, 4),
                "psi_contrib": round(psi_bin, 4),
            }
        )

    train_woe = train_bins.map(woe_map).astype(float).fillna(0.0)
    oot_woe = oot_bins.map(woe_map).astype(float).fillna(0.0)
    return train_woe, oot_woe, round(iv_total, 4), round(psi_total, 4), bin_details


def compute_vif_matrix(df_woe: pd.DataFrame) -> Dict[str, float]:
    """Calculates Variance Inflation Factor (VIF) for each feature in df_woe using correlation matrix inversion."""
    cols = list(df_woe.columns)
    if len(cols) <= 1:
        return {c: 1.0 for c in cols}
    corr = df_woe.corr().to_numpy()
    # Add tiny ridge jitter for numerical stability
    corr_reg = corr + np.eye(len(cols)) * 1e-6
    try:
        inv_corr = np.linalg.inv(corr_reg)
        vifs = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        vifs = np.ones(len(cols))
    return {col: round(max(float(v), 1.0), 2) for col, v in zip(cols, vifs)}


def compute_ks_and_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Computes Kolmogorov-Smirnov (KS) statistic and ROC-AUC."""
    auc = float(roc_auc_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(np.abs(tpr - fpr)))
    return round(ks, 4), round(auc, 4)


def compute_score_psi(train_scores: np.ndarray, oot_scores: np.ndarray, n_bins: int = 10) -> float:
    """Computes PSI on calibrated credit scorecard scores between Train and OOT."""
    eps = 1e-6
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(train_scores, quantiles))
    edges[0] = -np.inf
    edges[-1] = np.inf

    tr_counts, _ = np.histogram(train_scores, bins=edges)
    oot_counts, _ = np.histogram(oot_scores, bins=edges)

    tr_prop = np.maximum(tr_counts / max(len(train_scores), 1), eps)
    oot_prop = np.maximum(oot_counts / max(len(oot_scores), 1), eps)
    psi = float(np.sum((oot_prop - tr_prop) * np.log(oot_prop / tr_prop)))
    return round(psi, 4)


# ---------------------------------------------------------------------------
# High-Level Sandbox Tool Functions Exposed to Agents
# ---------------------------------------------------------------------------

def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def tool_inspect_dataset_and_check_leakage(params: DataInspectionInput) -> Dict[str, Any]:
    """Tool 1: Inspects Train & OOT datasets, validates observation windows, and flags leakage/compliance risks."""
    registry = DataPointerRegistry.get_instance()
    rag = HybridCreditRAG()

    tr_meta = registry.get_meta(params.train_uri)
    oot_meta = registry.get_meta(params.oot_uri)
    tr_df = registry.get_df(params.train_uri)
    oot_df = registry.get_df(params.oot_uri)

    # Retrieve data catalog & compliance rules (citations) and aggregate blacklists from the whole corpus
    rag_docs = rag.retrieve(get_profile(params.card_type).rag_query, top_k=3)
    leakage_blacklist: List[str] = []
    compliance_blacklist: List[str] = []
    for chunk in rag.corpus:
        leakage_blacklist.extend(chunk.metadata.get("leakage_blacklist", []))
        compliance_blacklist.extend(chunk.metadata.get("compliance_blacklist", []))

    id_and_target = {"apply_id", tr_meta.target_col, tr_meta.tags.get("date_col")}
    raw_features = [c for c in tr_df.columns if c not in id_and_target]

    flagged_leakage_reasons, flagged_compliance_reasons = detect_risky_features(
        tr_df, tr_meta.target_col, raw_features, leakage_blacklist, compliance_blacklist
    )

    # Manual exemptions (name-rule / statistical hits only); re-validated on every run so a later blacklist
    # entry or a changed detection result can never be bypassed by a stale exemption.
    exempted_features: List[Dict[str, Any]] = []
    ignored_exemptions: List[Dict[str, str]] = []
    for feat, rec in get_active_exemptions(params.train_uri).items():
        current = flagged_leakage_reasons.get(feat)
        if feat in flagged_compliance_reasons:
            ignored_exemptions.append({"feature": feat, "why": "已属于合规敏感字段，豁免不再生效"})
        elif current is None:
            ignored_exemptions.append({"feature": feat, "why": "当前未被识别为穿越字段，豁免无需生效"})
        elif not is_exemptable(current):
            ignored_exemptions.append({"feature": feat, "why": "已进入知识库黑名单，豁免不再生效"})
        else:
            del flagged_leakage_reasons[feat]
            exempted_features.append({**rec, "current_reason": current})

    exemptable_features = [f for f, why in flagged_leakage_reasons.items() if is_exemptable(why)]
    flagged_leakage = list(flagged_leakage_reasons)
    flagged_compliance = list(flagged_compliance_reasons)
    detection_reasons = {**flagged_leakage_reasons, **flagged_compliance_reasons}
    # WOE binning needs numeric columns; strings / datetimes would crash it, so they never reach the candidate pool
    non_numeric = [c for c in raw_features if not _is_numeric_series(tr_df[c])]
    clean_candidates = [
        c
        for c in raw_features
        if c not in flagged_leakage and c not in flagged_compliance and c not in non_numeric
    ]

    # None = not verifiable (observation window not provided for at least one dataset)
    window_valid = (
        tr_meta.obs_end_date < oot_meta.obs_start_date
        if tr_meta.obs_end_date and oot_meta.obs_start_date
        else None
    )

    return {
        "status": "success",
        "window_validation": {
            "is_chronological_valid": window_valid,
            "train_window": f"{tr_meta.obs_start_date} ~ {tr_meta.obs_end_date}" if tr_meta.obs_start_date and tr_meta.obs_end_date else "未提供",
            "oot_window": f"{oot_meta.obs_start_date} ~ {oot_meta.obs_end_date}" if oot_meta.obs_start_date and oot_meta.obs_end_date else "未提供",
            "perf_window_months": tr_meta.perf_window_months,
        },
        "sample_summary": {
            "train_uri": params.train_uri,
            "train_rows": tr_meta.row_count,
            "train_bad_rate": tr_meta.bad_rate,
            "oot_uri": params.oot_uri,
            "oot_rows": oot_meta.row_count,
            "oot_bad_rate": oot_meta.bad_rate,
            "oot_row_split": "row_split_of" in oot_meta.tags,
        },
        "governance_audit": {
            "raw_feature_count": len(raw_features),
            "non_numeric_features": [c for c in non_numeric if c not in flagged_leakage and c not in flagged_compliance],
            "quarantined_leakage_features": flagged_leakage,
            "quarantined_compliance_features": flagged_compliance,
            "detection_reasons": detection_reasons,
            "exemptable_features": exemptable_features,
            "exempted_features": exempted_features,
            "ignored_exemptions": ignored_exemptions,
            "clean_candidate_features": clean_candidates,
        },
        "rag_citations": [d["doc_id"] for d in rag_docs],
    }


def tool_run_feature_engineering(params: FeatureSelectionInput) -> Dict[str, Any]:
    """Tool 2: Runs WOE binning, IV calculation,跨期 PSI calculation, and VIF multicollinearity filtering."""
    registry = DataPointerRegistry.get_instance()
    tr_df = registry.get_df(params.train_uri)
    oot_df = registry.get_df(params.oot_uri)
    target_col = registry.get_meta(params.train_uri).target_col

    y_train = tr_df[target_col]
    y_oot = oot_df[target_col]

    tr_woe_df = pd.DataFrame(index=tr_df.index)
    oot_woe_df = pd.DataFrame(index=oot_df.index)
    tr_woe_df[target_col] = y_train
    oot_woe_df[target_col] = y_oot

    feature_metrics: Dict[str, Dict[str, Any]] = {}
    rejected_reasons: Dict[str, str] = {}
    iv_psi_passed: List[str] = []

    for feat in params.candidate_features:
        if feat in params.exclude_features:
            rejected_reasons[feat] = "Excluded by Reflection/Compliance Blacklist"
            continue
        if feat not in tr_df.columns:
            rejected_reasons[feat] = "Column not found in dataset"
            continue
        if not _is_numeric_series(tr_df[feat]):
            rejected_reasons[feat] = "Non-numeric column is not supported by WOE binning"
            continue
        if tr_df[feat].dropna().empty:
            rejected_reasons[feat] = "Column is entirely empty in Train"
            continue

        tr_woe, oot_woe, iv_val, psi_val, bins = compute_woe_iv_and_psi(
            tr_df[feat], y_train, oot_df[feat], n_bins=5
        )
        tr_woe_df[feat] = tr_woe
        oot_woe_df[feat] = oot_woe
        feature_metrics[feat] = {
            "iv": iv_val,
            "psi": psi_val,
            "bins_count": len(bins),
        }

        if iv_val < params.iv_threshold:
            rejected_reasons[feat] = f"Low IV ({iv_val} < {params.iv_threshold})"
        elif psi_val > params.psi_threshold:
            rejected_reasons[feat] = f"High PSI Drift ({psi_val} > {params.psi_threshold})"
        else:
            iv_psi_passed.append(feat)

    # Step 2: Iterative VIF Multicollinearity Filtering (drop highest VIF feature if VIF > threshold, favoring higher IV)
    active_features = list(iv_psi_passed)
    while len(active_features) > 1:
        vifs = compute_vif_matrix(tr_woe_df[active_features])
        for f, v in vifs.items():
            feature_metrics[f]["vif"] = v
        max_vif_feat = max(vifs, key=lambda k: vifs[k])
        if vifs[max_vif_feat] > params.vif_threshold:
            # Find the most correlated partner and drop the one with lower IV
            corr_matrix = tr_woe_df[active_features].corr().abs()
            corr_series = corr_matrix[max_vif_feat].drop(labels=[max_vif_feat])
            partner_feat = str(corr_series.idxmax())
            drop_candidate = (
                max_vif_feat
                if feature_metrics[max_vif_feat]["iv"] <= feature_metrics[partner_feat]["iv"]
                else partner_feat
            )
            rejected_reasons[drop_candidate] = (
                f"High Multicollinearity VIF ({vifs[max_vif_feat]} > {params.vif_threshold}, collinear with "
                f"{partner_feat if drop_candidate == max_vif_feat else max_vif_feat})"
            )
            active_features.remove(drop_candidate)
        else:
            break

    # Final VIF computation for selected features
    final_vifs = compute_vif_matrix(tr_woe_df[active_features]) if active_features else {}
    for f, v in final_vifs.items():
        feature_metrics[f]["vif"] = v

    # Register WOE-transformed datasets in DataPointerRegistry
    train_woe_uri = f"{params.train_uri}_woe"
    oot_woe_uri = f"{params.oot_uri}_woe"
    registry.register_dataset(
        uri=train_woe_uri,
        df=tr_woe_df[active_features + [target_col]],
        description="WOE编码后的开发集",
        obs_start_date="2025-01-01",
        obs_end_date="2025-06-30",
        target_col=target_col,
    )
    registry.register_dataset(
        uri=oot_woe_uri,
        df=oot_woe_df[active_features + [target_col]],
        description="WOE编码后的跨期验证集",
        obs_start_date="2025-07-01",
        obs_end_date="2025-09-30",
        target_col=target_col,
    )

    selected_summary = [
        {
            "feature": f,
            "iv": feature_metrics[f]["iv"],
            "psi": feature_metrics[f]["psi"],
            "vif": feature_metrics[f].get("vif", 1.0),
        }
        for f in sorted(active_features, key=lambda x: -feature_metrics[x]["iv"])
    ]

    return {
        "status": "success",
        "train_woe_uri": train_woe_uri,
        "oot_woe_uri": oot_woe_uri,
        "thresholds_applied": {
            "iv_min": params.iv_threshold,
            "psi_max": params.psi_threshold,
            "vif_max": params.vif_threshold,
        },
        "selected_features": [item["feature"] for item in selected_summary],
        "selected_feature_metrics": selected_summary,
        "rejected_features": rejected_reasons,
        "all_evaluated_metrics": feature_metrics,
    }


def tool_train_scorecard_model(params: ScorecardTrainInput) -> Dict[str, Any]:
    """Tool 3: Fits Logistic Regression on WOE features and calibrates standard PDO Scorecard points."""
    registry = DataPointerRegistry.get_instance()
    tr_woe = registry.get_df(params.train_woe_uri)
    oot_woe = registry.get_df(params.oot_woe_uri)
    target_col = registry.get_meta(params.train_woe_uri).target_col

    features = params.selected_features
    X_train = tr_woe[features].to_numpy()
    y_train = tr_woe[target_col].to_numpy()
    X_oot = oot_woe[features].to_numpy()
    y_oot = oot_woe[target_col].to_numpy()

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    clf.fit(X_train, y_train)

    coefs = {feat: round(float(c), 4) for feat, c in zip(features, clf.coef_[0])}
    intercept = round(float(clf.intercept_[0]), 4)
    negative_coef_features = [f for f, c in coefs.items() if c <= 0]

    # PDO Scorecard Calibration: Score = A - B * logit(p)
    factor_b = params.pdo / np.log(2.0)
    offset_a = params.base_score + factor_b * np.log(params.base_odds)

    tr_prob = clf.predict_proba(X_train)[:, 1]
    oot_prob = clf.predict_proba(X_oot)[:, 1]

    tr_logit = intercept + X_train @ clf.coef_[0]
    oot_logit = intercept + X_oot @ clf.coef_[0]

    tr_scores = np.clip(np.round(offset_a - factor_b * tr_logit), 300, 900)
    oot_scores = np.clip(np.round(offset_a - factor_b * oot_logit), 300, 900)

    model_id = f"scorecard_lr_{len(features)}f"
    registry.register_model(
        model_id=model_id,
        payload={
            "model_id": model_id,
            "features": features,
            "coefficients": coefs,
            "intercept": intercept,
            "negative_coef_features": negative_coef_features,
            "pdo_params": {
                "base_score": params.base_score,
                "base_odds": params.base_odds,
                "pdo": params.pdo,
                "offset_a": round(float(offset_a), 2),
                "factor_b": round(float(factor_b), 2),
            },
            "y_train": y_train,
            "tr_prob": tr_prob,
            "tr_scores": tr_scores,
            "y_oot": y_oot,
            "oot_prob": oot_prob,
            "oot_scores": oot_scores,
        },
    )

    return {
        "status": "success",
        "model_id": model_id,
        "feature_count": len(features),
        "coefficients": coefs,
        "intercept": intercept,
        "has_negative_coefficients": len(negative_coef_features) > 0,
        "negative_coef_features": negative_coef_features,
        "pdo_scaling": {
            "base_score": params.base_score,
            "pdo": params.pdo,
            "offset_a": round(float(offset_a), 2),
            "factor_b": round(float(factor_b), 2),
        },
    }


def tool_evaluate_scorecard_and_strategy(params: ModelValidationInput) -> Dict[str, Any]:
    """Tool 4: Evaluates Train/OOT KS, AUC, Score PSI, Bad-Rate Monotonicity, and Cutoff Strategy."""
    registry = DataPointerRegistry.get_instance()
    model_data = registry.get_model(params.model_id)

    y_train = model_data["y_train"]
    tr_prob = model_data["tr_prob"]
    tr_scores = model_data["tr_scores"]
    y_oot = model_data["y_oot"]
    oot_prob = model_data["oot_prob"]
    oot_scores = model_data["oot_scores"]

    train_ks, train_auc = compute_ks_and_auc(y_train, tr_prob)
    oot_ks, oot_auc = compute_ks_and_auc(y_oot, oot_prob)
    score_psi = compute_score_psi(tr_scores, oot_scores, n_bins=10)
    ks_drop = round(abs(train_ks - oot_ks), 4)

    # Check individual feature PSI in OOT vs raw Train to detect hidden feature drift
    tr_raw = registry.get_df(params.train_uri)
    oot_raw = registry.get_df(params.oot_uri)
    raw_target_col = registry.get_meta(params.train_uri).target_col
    drifting_features: List[Dict[str, Any]] = []
    for feat in model_data["features"]:
        _, _, _, f_psi, _ = compute_woe_iv_and_psi(tr_raw[feat], tr_raw[raw_target_col], oot_raw[feat])
        if f_psi > params.psi_threshold:
            drifting_features.append({"feature": feat, "oot_psi": f_psi})

    # Strategy Cutoff Simulation at target_approval_rate
    cutoff_score = float(np.quantile(oot_scores, 1.0 - params.target_approval_rate))
    approved_mask = oot_scores >= cutoff_score
    actual_approval_rate = round(float(approved_mask.mean()), 4)
    approved_bad_rate = round(float(y_oot[approved_mask].mean()), 4) if approved_mask.sum() > 0 else 0.0
    baseline_bad_rate = round(float(y_oot.mean()), 4)
    bad_reduction_pct = round((baseline_bad_rate - approved_bad_rate) / max(baseline_bad_rate, 1e-6) * 100, 2)

    # 5-bin Score Band Table on OOT
    oot_df_eval = pd.DataFrame({"score": oot_scores, "bad": y_oot})
    oot_df_eval["band"] = pd.qcut(oot_df_eval["score"].rank(method="first"), q=5)
    band_table = []
    for band, grp in oot_df_eval.groupby("band", observed=False):
        band_table.append(
            {
                "score_range": f"[{int(grp['score'].min())}, {int(grp['score'].max())}]",
                "count": int(len(grp)),
                "bad_rate": round(float(grp["bad"].mean()), 4),
            }
        )

    # Validation Gate Decision
    anomalies: List[str] = []
    if oot_ks < params.ks_threshold:
        anomalies.append(f"OOT KS ({oot_ks}) below threshold ({params.ks_threshold})")
    if ks_drop > params.ks_drop_threshold:
        anomalies.append(
            f"KS degradation |Train-OOT| ({ks_drop}) exceeds threshold ({params.ks_drop_threshold})"
        )
    if score_psi > params.psi_threshold:
        anomalies.append(f"Score PSI ({score_psi}) exceeds stability threshold ({params.psi_threshold})")
    if drifting_features:
        drift_str = ", ".join(f"{d['feature']}(PSI={d['oot_psi']})" for d in drifting_features)
        anomalies.append(f"In-model features exceed PSI threshold: {drift_str}")
    if model_data["negative_coef_features"]:
        anomalies.append(f"Negative LR coefficients detected: {model_data['negative_coef_features']}")

    passed_gate = len(anomalies) == 0

    return {
        "status": "success",
        "model_id": params.model_id,
        "validation_passed": passed_gate,
        "anomalies": anomalies,
        "drifting_features": drifting_features,
        "metrics": {
            "train_ks": train_ks,
            "oot_ks": oot_ks,
            "ks_degradation": ks_drop,
            "train_auc": train_auc,
            "oot_auc": oot_auc,
            "score_psi": score_psi,
        },
        "cutoff_strategy": {
            "target_approval_rate": params.target_approval_rate,
            "recommended_cutoff_score": int(round(cutoff_score)),
            "actual_approval_rate": actual_approval_rate,
            "baseline_oot_bad_rate": baseline_bad_rate,
            "post_cutoff_bad_rate": approved_bad_rate,
            "risk_reduction_pct": bad_reduction_pct,
        },
        "oot_score_bands": band_table,
    }


# Challenger must beat the champion's OOT KS by at least this margin to be flagged as "significantly ahead".
CHALLENGER_KS_MARGIN = 0.03


def tool_train_challenger_model(params: ChallengerTrainInput) -> Dict[str, Any]:
    """Tool 5: Trains a monotonic GBDT challenger on the champion's WOE features and compares OOT KS/AUC.

    The LR scorecard stays the champion (explainability); the challenger is a benchmark only.
    """
    registry = DataPointerRegistry.get_instance()
    tr_woe = registry.get_df(params.train_woe_uri)
    oot_woe = registry.get_df(params.oot_woe_uri)
    target_col = registry.get_meta(params.train_woe_uri).target_col
    champion = registry.get_model(params.champion_model_id)

    features = params.selected_features
    X_train = tr_woe[features].to_numpy()
    y_train = tr_woe[target_col].to_numpy()
    X_oot = oot_woe[features].to_numpy()
    y_oot = oot_woe[target_col].to_numpy()

    # WOE is defined as ln(Bad_i/Bad_total) - ln(Good_i/Good_total): risk must be non-decreasing in WOE.
    gbdt = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=150,
        learning_rate=0.05,
        monotonic_cst=[1] * len(features),
        random_state=42,
    )
    gbdt.fit(X_train, y_train)

    train_ks, train_auc = compute_ks_and_auc(y_train, gbdt.predict_proba(X_train)[:, 1])
    oot_ks, oot_auc = compute_ks_and_auc(y_oot, gbdt.predict_proba(X_oot)[:, 1])
    champion_oot_ks, champion_oot_auc = compute_ks_and_auc(champion["y_oot"], champion["oot_prob"])

    perm = permutation_importance(gbdt, X_oot, y_oot, scoring="roc_auc", n_repeats=5, random_state=42)
    importance = sorted(
        ({"feature": f, "importance": round(float(v), 4)} for f, v in zip(features, perm.importances_mean)),
        key=lambda d: d["importance"],
        reverse=True,
    )

    ks_gap = round(oot_ks - champion_oot_ks, 4)
    return {
        "status": "success",
        "challenger_model": "HistGradientBoosting(monotonic, depth=3)",
        "champion_model_id": params.champion_model_id,
        "challenger_metrics": {"train_ks": train_ks, "oot_ks": oot_ks, "train_auc": train_auc, "oot_auc": oot_auc},
        "champion_metrics": {"oot_ks": champion_oot_ks, "oot_auc": champion_oot_auc},
        "ks_gap": ks_gap,
        "challenger_significantly_better": ks_gap >= CHALLENGER_KS_MARGIN,
        "feature_importance": importance,
    }
