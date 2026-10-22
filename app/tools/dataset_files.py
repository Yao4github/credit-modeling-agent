"""Safe dataset-file upload into the data directory.

Guarantees: only `.csv` / `.parquet`, plain file names (no path tricks), never overwrites an existing file,
size-capped while streaming (never buffers the whole body in memory), atomic publication via hard link.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import AsyncIterator, Dict, List

from app.core.paths import get_data_dir

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = (".csv", ".parquet")
_NAME_RE = re.compile(r"^[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff._-]{0,99}$")


class UploadTooLarge(Exception):
    pass


def validate_filename(name: str) -> str:
    """Returns the file name if acceptable; raises ValueError otherwise."""
    if not _NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError("invalid file name: use letters, digits, Chinese characters, '.', '_' or '-' (max 100 chars, no paths)")
    if not name.lower().endswith(ALLOWED_SUFFIXES):
        raise ValueError("only .csv and .parquet files are supported")
    return name


async def save_upload(filename: str, chunks: AsyncIterator[bytes], max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Streams `chunks` into `<data_dir>/<filename>`; returns the byte count.

    Raises ValueError (bad name / empty body), FileExistsError (name taken), UploadTooLarge (over `max_bytes`).
    """
    validate_filename(filename)
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / filename
    if dest.exists():
        raise FileExistsError(f"file '{filename}' already exists")

    size = 0
    fd, tmp_name = tempfile.mkstemp(dir=data_dir, prefix=".upload-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp:
            async for chunk in chunks:
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLarge(f"file exceeds {max_bytes // (1024 * 1024)} MB limit")
                tmp.write(chunk)
        if size == 0:
            raise ValueError("empty file")
        try:
            os.link(tmp_name, dest)  # atomic and fails if the name appeared meanwhile (no overwrite)
        except FileExistsError:
            raise FileExistsError(f"file '{filename}' already exists") from None
    finally:
        os.unlink(tmp_name)
    return size


def list_files() -> List[Dict[str, object]]:
    """Uploadable-type files directly under the data directory (name + size), sorted by name."""
    data_dir = get_data_dir()
    if not data_dir.is_dir():
        return []
    return [
        {"filename": p.name, "size_bytes": p.stat().st_size}
        for p in sorted(data_dir.iterdir())
        if p.is_file() and _NAME_RE.fullmatch(p.name) and p.name.lower().endswith(ALLOWED_SUFFIXES)
    ]


def delete_file(filename: str) -> None:
    """Deletes `<data_dir>/<filename>`. ValueError for a bad name, FileNotFoundError when absent."""
    validate_filename(filename)
    path = get_data_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(f"file '{filename}' not found")
    path.unlink()


def save_dataframe_csv(filename: str, df) -> None:
    """Writes a DataFrame as `<data_dir>/<filename>` (UTF-8 CSV) with the same name rules and no-overwrite guarantee."""
    validate_filename(filename)
    if not filename.lower().endswith(".csv"):
        raise ValueError("only .csv output is supported")
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / filename
    if dest.exists():
        raise FileExistsError(f"file '{filename}' already exists")
    fd, tmp_name = tempfile.mkstemp(dir=data_dir, prefix=".upload-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as tmp:
            df.to_csv(tmp, index=False)
        try:
            os.link(tmp_name, dest)
        except FileExistsError:
            raise FileExistsError(f"file '{filename}' already exists") from None
    finally:
        os.unlink(tmp_name)
