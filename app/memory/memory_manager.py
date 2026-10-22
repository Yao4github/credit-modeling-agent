"""Dual-Layer Memory Manager (Short-Term Context Pruning + Long-Term Episodic Experience Store).

Addresses JD Requirements:
1. Short-Term Memory: Session context governance, sliding window message pruning, and structured
   error-trace summarization to prevent LLM context window overflow.
2. Long-Term Memory: Episodic experience store capturing domain lessons from historical modeling
   sessions and self-reflection rollbacks.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.paths import get_data_dir
from app.tools.rag_retriever import _tokenize_zh_en


@dataclass
class EpisodicMemoryRecord:
    memory_id: str
    scenario: str
    lesson_learned: str
    affected_features: List[str]
    recommended_action: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryManager:
    """Manages Long-Term Episodic Memory and Short-Term Context Governance."""

    _instance: Optional["MemoryManager"] = None

    def __init__(self) -> None:
        self._episodic_store: List[EpisodicMemoryRecord] = [
            EpisodicMemoryRecord(
                memory_id="MEM-001",
                scenario="首贷客群A卡建模 - 跨期稳定性治理",
                lesson_learned="夜间申请行为占比特征(night_apply_ratio_7d)在单期Train样本上IV较高(>0.20)，但极易受营销投放渠道切换影响，跨季度OOT PSI经常突破0.25。",
                affected_features=["night_apply_ratio_7d"],
                recommended_action="当模型OOT PSI超标(>0.10)时，优先排查并剔除渠道敏感型行为时间特征，收紧单特征PSI阈值至0.10。",
            ),
            EpisodicMemoryRecord(
                memory_id="MEM-002",
                scenario="征信与多头借贷衍生特征共线性治理",
                lesson_learned="外部多头综合指数(multi_loan_composite_idx)与近6个月多头借贷平台数(multi_platform_loan_cnt_6m)高度线性相关(VIF>15)，同时入模会导致逻辑回归系数符号反转。",
                affected_features=["multi_loan_composite_idx", "multi_platform_loan_cnt_6m"],
                recommended_action="启用VIF<=5.0漏斗筛选，在共线特征对中保留物理含义更清晰、IV更高的基础统计特征。",
            ),
        ]
        self._builtin_count = len(self._episodic_store)
        self._load_persisted_memories()
        self._session_checkpoints: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def _memory_file() -> Path:
        return get_data_dir() / "memory_episodic.json"

    def _load_persisted_memories(self) -> None:
        """Loads previously persisted reflection experiences; a missing/corrupt file keeps built-ins only."""
        path = self._memory_file()
        if not path.is_file():
            return
        try:
            for item in json.loads(path.read_text(encoding="utf-8")):
                self._episodic_store.append(EpisodicMemoryRecord(**item))
        except (OSError, ValueError, TypeError):
            del self._episodic_store[self._builtin_count :]

    def _persist_memories(self) -> None:
        """Atomically writes non-built-in memories to disk."""
        path = self._memory_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        records = [asdict(r) for r in self._episodic_store[self._builtin_count :]]
        tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)

    @classmethod
    def get_instance(cls) -> "MemoryManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve_episodic_memories(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieves relevant historical modeling lessons using lexical overlap."""
        q_tokens = set(_tokenize_zh_en(query))
        scored = []
        for rec in self._episodic_store:
            doc_tokens = set(_tokenize_zh_en(f"{rec.scenario} {rec.lesson_learned} {' '.join(rec.affected_features)}"))
            overlap = len(q_tokens & doc_tokens) / max(len(q_tokens), 1)
            scored.append((overlap, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "memory_id": r.memory_id,
                "scenario": r.scenario,
                "lesson_learned": r.lesson_learned,
                "affected_features": r.affected_features,
                "recommended_action": r.recommended_action,
            }
            for _, r in scored[:top_k]
        ]

    def record_reflection_experience(
        self,
        scenario: str,
        lesson_learned: str,
        affected_features: List[str],
        recommended_action: str,
    ) -> str:
        """Persists a newly discovered self-reflection insight into Long-Term Episodic Memory."""
        mem_id = f"MEM-{len(self._episodic_store) + 1:03d}"
        self._episodic_store.append(
            EpisodicMemoryRecord(
                memory_id=mem_id,
                scenario=scenario,
                lesson_learned=lesson_learned,
                affected_features=affected_features,
                recommended_action=recommended_action,
            )
        )
        self._persist_memories()
        return mem_id

    @staticmethod
    def prune_and_summarize_context(
        agent_logs: List[Dict[str, Any]],
        max_recent_steps: int = 6,
    ) -> Dict[str, Any]:
        """Short-Term Memory Governance: Compresses older execution steps while keeping recent steps intact."""
        if len(agent_logs) <= max_recent_steps:
            return {
                "compressed_history_summary": None,
                "active_logs": agent_logs,
                "token_saved_ratio": 0.0,
            }
        older = agent_logs[:-max_recent_steps]
        recent = agent_logs[-max_recent_steps:]
        summary_lines = [
            f"[{item.get('agent', 'sys')} -> {item.get('action', '')}]: {str(item.get('summary', ''))[:80]}"
            for item in older
        ]
        raw_len = sum(len(str(x)) for x in agent_logs)
        pruned_len = sum(len(s) for s in summary_lines) + sum(len(str(x)) for x in recent)
        saved_ratio = round(max(0.0, (raw_len - pruned_len) / max(raw_len, 1)), 4)
        return {
            "compressed_history_summary": " | ".join(summary_lines),
            "active_logs": recent,
            "token_saved_ratio": saved_ratio,
        }
