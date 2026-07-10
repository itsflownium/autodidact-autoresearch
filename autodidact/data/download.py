"""Resumable, content-addressed TinyStories downloads."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

from autodidact.data.config import SOURCE_FILES, SourceFile

CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_INTERVAL = 128 * 1024 * 1024


class SourceIntegrityError(RuntimeError):
    """Raised when an upstream or cached source does not match its pinned digest."""


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: Path, source: SourceFile) -> None:
    if not path.is_file():
        raise SourceIntegrityError(f"missing source file: {path}")
    actual_size = path.stat().st_size
    if actual_size != source.size:
        raise SourceIntegrityError(
            f"source size mismatch for {source.filename}: expected {source.size}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != source.sha256:
        raise SourceIntegrityError(
            f"source hash mismatch for {source.filename}: "
            f"expected {source.sha256}, got {actual_hash}"
        )


def _make_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def download_source(
    source: SourceFile,
    raw_dir: Path,
    *,
    opener: Callable[..., object] = urlopen,
) -> Path:
    """Download one pinned source, resuming a partial transfer when supported."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / source.filename
    if destination.exists():
        verify_source(destination, source)
        _make_read_only(destination)
        return destination

    partial = raw_dir / f"{source.filename}.part"
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == source.size:
        if sha256_file(partial) == source.sha256:
            os.replace(partial, destination)
            _make_read_only(destination)
            return destination
        partial.unlink()
        offset = 0
    elif offset > source.size:
        partial.unlink()
        offset = 0

    headers = {"User-Agent": "autodidact-data/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    response = opener(Request(source.url, headers=headers))
    status = getattr(response, "status", None) or response.getcode()
    append = offset > 0 and status == 206
    if not append:
        offset = 0

    digest = hashlib.sha256()
    if append:
        with partial.open("rb") as existing:
            while chunk := existing.read(CHUNK_SIZE):
                digest.update(chunk)

    mode = "ab" if append else "wb"
    downloaded = offset
    next_progress = ((downloaded // PROGRESS_INTERVAL) + 1) * PROGRESS_INTERVAL
    with partial.open(mode) as output, response:
        while chunk := response.read(CHUNK_SIZE):
            output.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if downloaded >= next_progress:
                percent = 100 * downloaded / source.size
                print(
                    f"{source.filename}: {downloaded:,}/{source.size:,} bytes ({percent:.1f}%)",
                    file=sys.stderr,
                )
                next_progress += PROGRESS_INTERVAL
        output.flush()
        os.fsync(output.fileno())

    if downloaded != source.size:
        raise SourceIntegrityError(
            f"download size mismatch for {source.filename}: "
            f"expected {source.size}, got {downloaded}"
        )
    actual_hash = digest.hexdigest()
    if actual_hash != source.sha256:
        raise SourceIntegrityError(
            f"download hash mismatch for {source.filename}: "
            f"expected {source.sha256}, got {actual_hash}"
        )

    os.replace(partial, destination)
    _make_read_only(destination)
    return destination


def ensure_sources(raw_dir: Path) -> dict[str, Path]:
    """Download and verify every source, keyed by its manifest role."""

    return {source.role: download_source(source, raw_dir) for source in SOURCE_FILES}


def verify_pinned_sources(raw_dir: Path) -> dict[str, Path]:
    """Verify already downloaded source files without network access."""

    paths: dict[str, Path] = {}
    for source in SOURCE_FILES:
        path = raw_dir / source.filename
        verify_source(path, source)
        paths[source.role] = path
    return paths
