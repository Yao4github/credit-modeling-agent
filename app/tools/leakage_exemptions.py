"""Per-dataset, per-feature exemptions for auto-detected leakage quarantines (with an append-only audit history).

Rules:
- Only name-rule / statistical leakage hits are exemptable. Knowledge-base blacklist hits and compliance-sensitive
  features can never be exempted.
- `operator` is a caller-supplied label and is NOT identity-verified (the project has no auth layer).
- A missing or corrupt store is treated as "no exemptions" (the safe default); a corrupt file is moved aside
  (`*.corrupt`) before the next write so history is never silently overwritten.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from app.core.paths import get_data_dir
from app.tools.data_registry import DataPointerRegistry
from app.tools.leakage_rules import corpus_blacklists, detect_risky_features, is_exemptable

_lock = threading.Lock()


def _store_path():
    return get_data_dir() / "leakage_exemptions.json"


def _empty() -> Dict[str, Any]:
    return {"active": {}, "history": []}


def _load() -> Dict[str, Any]:
    path = _store_path()
    if not path.is_file():
        return _empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("active"), dict) and isinstance(data.get("history"), list):
            return data
    except (OSError, ValueError, AttributeError):
        pass
    return _empty()


def _save(data: Dict[str, Any]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            os.replace(path, path.with_suffix(".json.corrupt"))
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_text(value: str, label: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def get_active_exemptions(uri: str) -> Dict[str, Dict[str, Any]]:
    return dict(_load()["active"].get(uri, {}))


def get_exemption_state(uri: str) -> Dict[str, Any]:
    data = _load()
    return {
        "uri": uri,
        "active": list(data["active"].get(uri, {}).values()),
        "history": [e for e in data["history"] if e["uri"] == uri],
    }


def grant_exemption(uri: str, feature: str, reason: str, operator: str) -> Dict[str, Any]:
    """Exempts one quarantined leakage feature. Raises KeyError (unknown dataset) or ValueError (not allowed)."""
    reason, operator = _require_text(reason, "reason"), _require_text(operator, "operator")
    registry = DataPointerRegistry.get_instance()
    meta = registry.get_meta(uri)  # KeyError if the dataset is unknown
    df = registry.get_df(uri)
    candidates = [c for c in df.columns if c not in {"apply_id", meta.target_col}]
    if feature not in candidates:
        raise ValueError(f"feature '{feature}' not found in dataset '{uri}'")

    leak_bl, comp_bl = corpus_blacklists()
    leakage, compliance = detect_risky_features(df, meta.target_col, candidates, leak_bl, comp_bl)
    if feature in compliance:
        raise ValueError(f"feature '{feature}' is a compliance-sensitive field and can never be exempted")
    if feature not in leakage:
        raise ValueError(f"feature '{feature}' is not quarantined; nothing to exempt")
    if not is_exemptable(leakage[feature]):
        raise ValueError(f"feature '{feature}' is on the knowledge-base blacklist and can never be exempted")

    with _lock:
        data = _load()
        if feature in data["active"].get(uri, {}):
            raise ValueError(f"feature '{feature}' is already exempted")
        record = {
            "feature": feature,
            "original_reason": leakage[feature],
            "reason": reason,
            "operator": operator,
            "created_at": _now(),
        }
        data["active"].setdefault(uri, {})[feature] = record
        data["history"].append({"event": "grant", "uri": uri, "at": record["created_at"], **record})
        _save(data)
    return record


def revoke_exemption(uri: str, feature: str, reason: str, operator: str) -> Dict[str, Any]:
    """Revokes an active exemption. Raises KeyError when there is none, ValueError for empty reason/operator."""
    reason, operator = _require_text(reason, "reason"), _require_text(operator, "operator")
    with _lock:
        data = _load()
        record = data["active"].get(uri, {}).pop(feature, None)
        if record is None:
            raise KeyError(f"no active exemption for '{feature}' on '{uri}'")
        if not data["active"][uri]:
            del data["active"][uri]
        event = {
            "event": "revoke",
            "uri": uri,
            "feature": feature,
            "original_reason": record["original_reason"],
            "reason": reason,
            "operator": operator,
            "at": _now(),
        }
        data["history"].append(event)
        _save(data)
    return event
