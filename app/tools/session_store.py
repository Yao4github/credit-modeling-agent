"""Persistence of finished modeling runs (latest `MAX_SESSIONS` kept) for the workbench's history page."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.paths import get_data_dir

MAX_SESSIONS = 50
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_valid_session_id(session_id: str) -> bool:
    return bool(_ID_RE.fullmatch(session_id))


def _safe_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:64]
    return cleaned or "session"


def _dir():
    return get_data_dir() / "sessions"


def save_session(session_id: str, query: str, result: Dict[str, Any]) -> str:
    """Writes the run result atomically and prunes the oldest sessions beyond the cap; returns the stored id."""
    sid = _safe_id(session_id)
    directory = _dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": sid,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "summary": {
            "card_type": result.get("card_type"),
            "intent_type": result.get("intent_type"),
            "validation_passed": result.get("validation_passed"),
            "compliance_passed": result.get("compliance_passed"),
        },
        "result": result,
    }
    path = directory / f"{sid}.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp_path, path)
    _prune(directory)
    return sid


def _read(path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and "saved_at" in data else None
    except (OSError, ValueError):
        return None


def _prune(directory) -> None:
    files = sorted(directory.glob("*.json"), key=lambda p: (_read(p) or {}).get("saved_at", ""), reverse=True)
    for stale in files[MAX_SESSIONS:]:
        stale.unlink(missing_ok=True)


def list_sessions() -> List[Dict[str, Any]]:
    """Newest first; unreadable files are skipped."""
    directory = _dir()
    if not directory.is_dir():
        return []
    items = []
    for path in directory.glob("*.json"):
        data = _read(path)
        if data:
            items.append(
                {
                    "session_id": data["session_id"],
                    "saved_at": data["saved_at"],
                    "query": str(data.get("query", ""))[:80],
                    **data.get("summary", {}),
                }
            )
    return sorted(items, key=lambda i: i["saved_at"], reverse=True)


def delete_session(session_id: str) -> bool:
    """Removes a stored session; False for an invalid id or a session that does not exist."""
    if not is_valid_session_id(session_id):
        return False
    path = _dir() / f"{session_id}.json"
    if not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Returns the stored payload, or None for an invalid id / missing / unreadable session."""
    if not is_valid_session_id(session_id):
        return None
    path = _dir() / f"{session_id}.json"
    return _read(path) if path.is_file() else None
