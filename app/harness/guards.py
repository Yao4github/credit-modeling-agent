"""Pure guard functions applied to LLM output before it can influence the pipeline or the report."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

# Sane (min, max) bounds so a hallucinated LLM value can never loosen a risk gate absurdly.
CONSTRAINT_BOUNDS = {
    "ks_min": (0.0, 1.0),
    "ks_drop_max": (0.0, 1.0),
    "psi_max": (0.0, 1.0),
    "iv_min": (0.0, 1.0),
    "vif_max": (1.0, 50.0),
    "target_approval_rate": (0.01, 1.0),
    "base_score": (100.0, 1000.0),
    "pdo": (5.0, 100.0),
}

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def clamp_constraints(raw: Any) -> Dict[str, float]:
    """Keeps only known constraint keys whose numeric value lies inside CONSTRAINT_BOUNDS."""
    out: Dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if k not in CONSTRAINT_BOUNDS or isinstance(v, bool):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        lo, hi = CONSTRAINT_BOUNDS[k]
        if lo <= fv <= hi:
            out[k] = fv
    return out


def whitelist_union(mandatory: List[str], proposed: Any, allowed: Iterable[str]) -> List[str]:
    """Sandbox-flagged items always stay; LLM-proposed items are added only if inside `allowed`."""
    allowed_set = set(allowed)
    extra = [f for f in proposed if isinstance(f, str) and f in allowed_set] if isinstance(proposed, list) else []
    return list(dict.fromkeys(list(mandatory) + extra))


def _numbers(text: str) -> set:
    return {float(m) for m in _NUM_RE.findall(text)}


def numbers_grounded(llm_text: str, *sources: str) -> Tuple[bool, List[str]]:
    """True iff every number in `llm_text` also appears (numerically) in the sources. Returns (ok, ungrounded)."""
    known: set = set()
    for s in sources:
        known |= _numbers(s)
    bad = sorted({m for m in _NUM_RE.findall(llm_text) if float(m) not in known})
    return not bad, bad
