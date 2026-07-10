"""Authenticated download and safe extraction for the prepared dataset archive."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import zstandard as zstd
from huggingface_hub import hf_hub_download

from autodidact.data.config import (
    PREPARED_DATASET_ARCHIVE,
    PreparedDatasetArchive,
)
from autodidact.data.download import sha256_file
from autodidact.data.integrity import seal_dataset_tree, verify_dataset

COPY_BUFFER_SIZE = 8 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 512


class PreparedArchiveError(RuntimeError):
    """Raised when a prepared archive cannot be trusted or extracted safely."""


def verify_prepared_archive(
    path: Path,
    archive: PreparedDatasetArchive = PREPARED_DATASET_ARCHIVE,
) -> None:
    path = path.expanduser()
    if not path.is_file():
        raise PreparedArchiveError(f"prepared archive does not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != archive.size:
        raise PreparedArchiveError(
            f"prepared archive size mismatch: expected {archive.size}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != archive.sha256:
        raise PreparedArchiveError(
            f"prepared archive hash mismatch: expected {archive.sha256}, got {actual_hash}"
        )


def _relative_member_path(member: tarfile.TarInfo, root_name: str) -> Path | None:
    member_path = PurePosixPath(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise PreparedArchiveError(f"unsafe path in prepared archive: {member.name}")
    if not member_path.parts or member_path.parts[0] != root_name:
        raise PreparedArchiveError(
            f"archive member is outside the expected {root_name!r} root: {member.name}"
        )
    if len(member_path.parts) == 1:
        if not member.isdir():
            raise PreparedArchiveError("prepared archive root must be a directory")
        return None
    return Path(*member_path.parts[1:])


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, directories, files in os.walk(root):
        Path(directory).chmod(0o700)
        for name in directories:
            (Path(directory) / name).chmod(0o700)
        for name in files:
            (Path(directory) / name).chmod(0o600)


def extract_prepared_archive(
    archive_path: Path,
    output_root: Path,
    *,
    archive: PreparedDatasetArchive = PREPARED_DATASET_ARCHIVE,
) -> dict[str, Any]:
    """Verify, safely extract, validate, seal, and atomically install an archive."""

    archive_path = archive_path.expanduser().resolve()
    verify_prepared_archive(archive_path, archive)
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"dataset root already exists: {output_root}; verify it instead of replacing it"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.fetch-", dir=output_root.parent))
    regular_files = 0
    expanded_bytes = 0
    seen_files: set[Path] = set()
    try:
        with archive_path.open("rb") as compressed:
            decompressor = zstd.ZstdDecompressor(max_window_size=1 << 27)
            with (
                decompressor.stream_reader(compressed) as reader,
                tarfile.open(fileobj=reader, mode="r|") as tar,
            ):
                for member_number, member in enumerate(tar, start=1):
                    if member_number > MAX_ARCHIVE_MEMBERS:
                        raise PreparedArchiveError("prepared archive has too many members")
                    relative = _relative_member_path(member, archive.root_name)
                    if relative is None:
                        continue
                    target = staging / relative
                    if member.isdir():
                        if target.exists() and not target.is_dir():
                            raise PreparedArchiveError(
                                f"archive directory conflicts with a file: {member.name}"
                            )
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isreg():
                        raise PreparedArchiveError(
                            f"unsupported archive member type: {member.name}"
                        )
                    if relative in seen_files or target.exists():
                        raise PreparedArchiveError(
                            f"duplicate file in prepared archive: {member.name}"
                        )
                    regular_files += 1
                    expanded_bytes += member.size
                    if regular_files > archive.regular_file_count:
                        raise PreparedArchiveError("prepared archive has too many files")
                    if expanded_bytes > archive.expanded_file_bytes:
                        raise PreparedArchiveError("prepared archive expands beyond its contract")
                    source = tar.extractfile(member)
                    if source is None:
                        raise PreparedArchiveError(f"cannot read archive member: {member.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination, length=COPY_BUFFER_SIZE)
                        destination.flush()
                        os.fsync(destination.fileno())
                    if target.stat().st_size != member.size:
                        raise PreparedArchiveError(
                            f"extracted size mismatch for archive member: {member.name}"
                        )
                    seen_files.add(relative)

        if regular_files != archive.regular_file_count:
            raise PreparedArchiveError(
                f"prepared archive contains {regular_files} files; "
                f"expected {archive.regular_file_count}"
            )
        if expanded_bytes != archive.expanded_file_bytes:
            raise PreparedArchiveError(
                f"prepared archive expands to {expanded_bytes} bytes; "
                f"expected {archive.expanded_file_bytes}"
            )
        verify_dataset(staging, scope="all", require_read_only=False)
        seal_dataset_tree(staging)
        manifest = verify_dataset(staging, scope="all", require_read_only=True)
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        _make_tree_writable(staging)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def download_prepared_archive(
    cache_dir: Path,
    *,
    archive: PreparedDatasetArchive = PREPARED_DATASET_ARCHIVE,
) -> Path:
    """Download the pinned private Hub artifact using saved or environment credentials."""

    try:
        downloaded = hf_hub_download(
            repo_id=archive.repo_id,
            filename=archive.filename,
            revision=archive.revision,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    except Exception as error:
        raise PreparedArchiveError(
            "cannot download the private prepared dataset; authenticate with `hf auth login` "
            "or set HF_TOKEN"
        ) from error
    return Path(downloaded)


def fetch_prepared_dataset(
    output_root: Path,
    *,
    archive_path: Path | None = None,
    archive: PreparedDatasetArchive = PREPARED_DATASET_ARCHIVE,
) -> dict[str, Any]:
    """Install the pinned prepared tree without retaining an extra archive copy."""

    output_root = output_root.expanduser()
    if output_root.exists():
        return verify_dataset(output_root, scope="all")
    if archive_path is not None:
        return extract_prepared_archive(archive_path, output_root, archive=archive)
    with tempfile.TemporaryDirectory(prefix="autodidact-hub-download-") as temporary:
        downloaded = download_prepared_archive(Path(temporary), archive=archive)
        return extract_prepared_archive(downloaded, output_root, archive=archive)
