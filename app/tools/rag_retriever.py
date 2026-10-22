"""Hybrid RAG Retriever (BM25 + Semantic TF-IDF + Reciprocal Rank Fusion).

Provides domain knowledge retrieval over:
1. Credit Data Warehouse Schema & Feature Lineage Catalog (prevents data leakage & clarifies definitions)
2. Regulatory Compliance & Risk Modeling SOP Knowledge Base (Basel III, PIPL, Fair Lending, Scorecard SOP)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.paths import get_data_dir


@dataclass
class KnowledgeChunk:
    doc_id: str
    category: str  # "data_catalog" | "compliance_sop" | "metric_definition"
    title: str
    content: str
    metadata: Dict[str, Any]


KNOWLEDGE_CORPUS: List[KnowledgeChunk] = [
    KnowledgeChunk(
        doc_id="CAT-001",
        category="data_catalog",
        title="申请评分卡(A卡)核心特征字典与时间窗口血缘规范",
        content=(
            "表名: dws_credit_a_card_features | 观察点(Observation Point): 用户提交授信申请时刻 T0。\n"
            "字段明细:\n"
            "1. credit_inq_cnt_3m: 近3个月人行征信机构硬查询次数（贷前T0可获取，合规可用，高区分度）。\n"
            "2. multi_platform_loan_cnt_6m: 近6个月多头借贷申请平台数（贷前T0可获取，合规可用）。\n"
            "3. hist_max_dpd_days_12m: 近12个月历史信贷最大逾期天数（贷前T0可获取，合规可用）。\n"
            "4. debt_to_income_ratio: 认定负债收入比 DTI（贷前T0可获取，合规可用）。\n"
            "5. credit_card_utilization: 近6个月信用卡平均额度使用率（贷前T0可获取，合规可用）。\n"
            "6. verified_income_monthly: 公积金/社保核实月收入（贷前T0可获取，合规可用，反向风险指标）。\n"
            "7. night_apply_ratio_7d: 近7天夜间(0:00-5:00)行为占比（贷前T0可获取，但受营销渠道投放影响大，跨期PSI极易漂移）。\n"
            "8. multi_loan_composite_idx: 外部多头借贷衍生指数（与 multi_platform_loan_cnt_6m 高度共线，VIF>10）。"
        ),
        metadata={"table": "dws_credit_a_card_features", "stage": "A-Card"},
    ),
    KnowledgeChunk(
        doc_id="CAT-002",
        category="data_catalog",
        title="数据穿越(Data Leakage)高危字段与贷后表现期标签定义",
        content=(
            "目标标签定义: target_mob3_dpd30 指放款后账龄满3个月(MOB3)内是否发生逾期超过30天(DPD30+)的坏样本事件。\n"
            "【严禁入模字段 - 数据穿越预警】:\n"
            "- post_loan_repay_amt_m1: 贷后首月(M1)实际还款金额。该字段采集于观察点 T0 之后的放款表现期内，"
            "属于典型的未来信息穿越(Data Leakage)特征，在A卡申请评分模型中必须100%剔除，严禁入模！"
        ),
        metadata={"leakage_blacklist": ["post_loan_repay_amt_m1"], "target": "target_mob3_dpd30"},
    ),
    KnowledgeChunk(
        doc_id="COMP-001",
        category="compliance_sop",
        title="金融信贷风控模型公平信贷与个人信息保护合规红线(PIPL & Fair Lending)",
        content=(
            "根据《个人信息保护法》(PIPL)、《商业银行互联网贷款管理暂行办法》及公平信贷原则(Fair Lending):\n"
            "1. 严禁使用性别、民族、宗教信仰、特定地域歧视等敏感属性作为授信准入或评分卡入模变量。\n"
            "2. 合规黑名单字段: applicant_gender_code (申请人性别编码) 属于算法歧视禁止特征，必须强制剔除。\n"
            "3. 模型可解释性要求: 标准A卡逻辑回归模型对所有WOE转换后的入模特征，其回归系数(Coefficient)必须全部为正数"
            "（当WOE定义为 ln(Bad_i/Bad_total) - ln(Good_i/Good_total) 时），严禁出现系数符号反转的业务悖论。"
        ),
        metadata={"compliance_blacklist": ["applicant_gender_code"], "regulation": "PIPL_FairLending"},
    ),
    KnowledgeChunk(
        doc_id="SOP-001",
        category="metric_definition",
        title="信贷评分卡特征筛选、稳定性(PSI)与区分度(KS/IV)准入标准",
        content=(
            "标准信贷评分卡(Scorecard)上线准入硬性指标门禁:\n"
            "1. 信息值 IV (Information Value): 入模特征需满足 IV >= 0.02；IV < 0.02 视为无预测能力噪声特征（如 app_device_screen_dpi）。\n"
            "2. 群体稳定性指标 PSI (Population Stability Index): 单特征跨期(Train vs OOT) PSI 必须 <= 0.10，模型总分 PSI 必须 <= 0.10。"
            "若 PSI > 0.10 说明发生显著客群漂移，必须触发反思回滚剔除高漂移特征。\n"
            "3. 多重共线性 VIF (Variance Inflation Factor): 入模特征两两相关性及 VIF 必须 <= 5.0。\n"
            "4. 区分度 KS (Kolmogorov-Smirnov): A卡在开发集(Train)与跨期集(OOT)上的 KS 均需 >= 0.30，且 |Train_KS - OOT_KS| <= 0.05。\n"
            "5. 标准刻度转换 PDO (Points to Double the Odds): 默认基准分 Base_Score = 600, 基准好坏比 Base_Odds = 20:1 (即坏账率约4.76%), "
            "翻倍分数 PDO = 20。转换公式: Factor B = PDO / ln(2), Offset A = Base_Score - B * ln(Base_Odds)。"
        ),
        metadata={
            "iv_min": 0.02,
            "psi_max": 0.10,
            "vif_max": 5.0,
            "ks_min": 0.30,
            "base_score": 600,
            "pdo": 20,
        },
    ),
    KnowledgeChunk(
        doc_id="CAT-003",
        category="data_catalog",
        title="行为评分卡(B卡)存量客户特征字典与未来信息字段规范",
        content=(
            "B卡适用对象: 存量授信客户; 观察点为额度管理决策时刻; 标签 target_fut3m_dpd30 指观察点后3个月内发生逾期30天以上(1=坏)。\n"
            "可用行为特征: avg_util_3m(近3个月平均额度使用率)、max_dpd_cnt_6m(近6个月逾期次数)、repay_ratio_3m(近3个月还款率)、"
            "cash_advance_cnt_3m(近3个月取现次数)、txn_amt_trend_3m(近3个月交易金额趋势)。\n"
            "稳定性风险: login_freq_30d(近30天登录频次)受App改版影响，跨期漂移显著。\n"
            "共线风险: avg_util_6m 与 avg_util_3m 高度共线。\n"
            "【严禁入模字段 - 未来信息】: fut_m1_overdue_amt 为观察点之后下一个月的逾期金额，B卡必须剔除。"
        ),
        metadata={"leakage_blacklist": ["fut_m1_overdue_amt"], "target": "target_fut3m_dpd30", "stage": "B-Card"},
    ),
    KnowledgeChunk(
        doc_id="CAT-004",
        category="data_catalog",
        title="催收评分卡(C卡)逾期账户特征字典与催收后信息字段规范",
        content=(
            "C卡适用对象: 已逾期账户; 观察点为进入催收队列时刻; 标签 target_unrecovered_d30 指催收后30天内未回款(1=未回款，分数越高回款可能性越大)。\n"
            "可用特征: overdue_days(当前逾期天数)、overdue_amount(逾期金额)、prior_collection_cnt(历史催收次数)、"
            "promise_to_pay_cnt(承诺还款次数)、debt_to_income_ratio(负债收入比)。\n"
            "稳定性风险: call_connect_rate_7d(近7天外呼接通率)受外呼供应商切换影响，跨期漂移显著。\n"
            "共线风险: overdue_amt_bucket 与 overdue_amount 高度共线。\n"
            "【严禁入模字段 - 催收后信息】: post_collection_repay_amt_d7 为催收决策之后7天内的回款金额，C卡必须剔除。"
        ),
        metadata={"leakage_blacklist": ["post_collection_repay_amt_d7"], "target": "target_unrecovered_d30", "stage": "C-Card"},
    ),
    KnowledgeChunk(
        doc_id="SOP-002",
        category="metric_definition",
        title="B卡与C卡区分度准入差异说明",
        content=(
            "B卡(行为评分)区分度要求 KS >= 0.30; C卡(催收评分)场景区分度天然较低，KS >= 0.25。\n"
            "C卡报告中的通过率口径改为策略覆盖率，即分数不低于切点的账户占比。"
        ),
        metadata={"ks_min_b": 0.30, "ks_min_c": 0.25},
    ),
]


def _tokenize_zh_en(text: str) -> List[str]:
    """Hybrid tokenizer splitting English identifiers/numbers and Chinese character bigrams/unigrams."""
    text = text.lower()
    en_tokens = re.findall(r"[a-z0-9_]+", text)
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    zh_bigrams = [zh_chars[i] + zh_chars[i + 1] for i in range(len(zh_chars) - 1)]
    return en_tokens + zh_chars + zh_bigrams


def load_extra_knowledge() -> List[KnowledgeChunk]:
    """Loads optional user-provided chunks from `<data_dir>/knowledge_extra.json`; invalid entries are skipped."""
    path = get_data_dir() / "knowledge_extra.json"
    if not path.is_file():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    builtin_ids = {c.doc_id for c in KNOWLEDGE_CORPUS}
    chunks: List[KnowledgeChunk] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not all(isinstance(it.get(k), str) for k in ("doc_id", "category", "title", "content")):
            continue
        if it["doc_id"] in builtin_ids:
            continue
        meta = it.get("metadata")
        chunks.append(
            KnowledgeChunk(
                doc_id=it["doc_id"],
                category=it["category"],
                title=it["title"],
                content=it["content"],
                metadata=meta if isinstance(meta, dict) else {},
            )
        )
    return chunks


class HybridCreditRAG:
    """Combines BM25 sparse lexical matching and TF-IDF n-gram semantic vector similarity via RRF."""

    def __init__(self, corpus: Optional[List[KnowledgeChunk]] = None) -> None:
        self.corpus = corpus or (KNOWLEDGE_CORPUS + load_extra_knowledge())
        self.doc_texts = [f"{c.title}\n{c.content}" for c in self.corpus]
        self.tokenized_docs = [_tokenize_zh_en(t) for t in self.doc_texts]
        self.avgdl = sum(len(d) for d in self.tokenized_docs) / max(len(self.tokenized_docs), 1)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        q_tokens = _tokenize_zh_en(query)
        scores = np.zeros(len(self.corpus), dtype=float)
        N = len(self.corpus)
        for qt in set(q_tokens):
            df = sum(1 for doc in self.tokenized_docs if qt in doc)
            if df == 0:
                continue
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            for idx, doc in enumerate(self.tokenized_docs):
                tf = doc.count(qt)
                if tf > 0:
                    denom = tf + k1 * (1.0 - b + b * len(doc) / self.avgdl)
                    scores[idx] += idf * (tf * (k1 + 1.0)) / denom
        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        bm25 = self._bm25_scores(query)
        q_vec = self.vectorizer.transform([query])
        dense_sim = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Reciprocal Rank Fusion (RRF)
        bm25_ranks = np.argsort(np.argsort(-bm25)) + 1
        dense_ranks = np.argsort(np.argsort(-dense_sim)) + 1
        rrf_scores = 1.0 / (10 + bm25_ranks) + 1.0 / (10 + dense_ranks)

        results = []
        for idx in np.argsort(-rrf_scores):
            chunk = self.corpus[int(idx)]
            if category_filter and chunk.category != category_filter:
                continue
            results.append(
                {
                    "doc_id": chunk.doc_id,
                    "category": chunk.category,
                    "title": chunk.title,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "rrf_score": round(float(rrf_scores[idx]), 4),
                }
            )
            if len(results) >= top_k:
                break
        return results
