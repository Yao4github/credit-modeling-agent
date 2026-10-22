"""Filesystem locations for external datasets and persisted memory/knowledge stores."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def get_data_dir() -> Path:
    """Returns the data directory (override with env `CREDIT_DATA_DIR`, e.g. in tests)."""
    return Path(os.environ.get("CREDIT_DATA_DIR") or _DEFAULT_DATA_DIR).resolve()
