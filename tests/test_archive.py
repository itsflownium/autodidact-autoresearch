from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path

import pytest
import zstandard as zstd

from autodidact.data.archive import (
    PreparedArchiveError,
    download_prepared_archive,
    extract_prepared_archive,
    fetch_prepared_dataset,
    verify_prepared_archive,
)
from autodidact.data.config import (
    PREPARED_DATASET_ARCHIVE,
    PreparedDatasetArchive,
)
from autodidact.data.download import sha256_file
from autodidact.data.integrity import verify_dataset
from tests.conftest import make_tree_writable


def _archive_tree(
    source: Path,
    destination: Path,
    *,
    root_name: str = "tinystories-v1",
) -> PreparedDatasetArchive:
    with destination.open("wb") as compressed:
        compressor = zstd.ZstdCompressor(level=1)
        with (
            compressor.stream_writer(compressed, closefd=False) as writer,
            tarfile.open(fileobj=writer, mode="w|") as tar,
        ):
            tar.add(source, arcname=root_name)
    files = [path for path in source.rglob("*") if path.is_file()]
    return PreparedDatasetArchive(
        repo_id="tests/prepared-dataset",
        revision="test-revision",
        filename=destination.name,
        root_name=root_name,
        size=destination.stat().st_size,
        sha256=sha256_file(destination),
        expanded_file_bytes=sum(path.stat().st_size for path in files),
        regular_file_count=len(files),
    )


def _single_member_archive(
    destination: Path,
    *,
    name: str,
    data: bytes = b"x",
    member_type: bytes = tarfile.REGTYPE,
) -> PreparedDatasetArchive:
    with destination.open("wb") as compressed:
        compressor = zstd.ZstdCompressor(level=1)
        with (
            compressor.stream_writer(compressed, closefd=False) as writer,
            tarfile.open(fileobj=writer, mode="w|") as tar,
        ):
            member = tarfile.TarInfo(name)
            member.type = member_type
            member.size = len(data) if member_type == tarfile.REGTYPE else 0
            member.linkname = "outside" if member_type == tarfile.SYMTYPE else ""
            tar.addfile(member, io.BytesIO(data) if member.size else None)
    return PreparedDatasetArchive(
        repo_id="tests/prepared-dataset",
        revision="test-revision",
        filename=destination.name,
        root_name="tinystories-v1",
        size=destination.stat().st_size,
        sha256=sha256_file(destination),
        expanded_file_bytes=len(data) if member_type == tarfile.REGTYPE else 0,
        regular_file_count=1 if member_type == tarfile.REGTYPE else 0,
    )


def test_production_archive_contract_is_content_addressed() -> None:
    archive = PREPARED_DATASET_ARCHIVE
    assert archive.repo_id == "Flownium/autodidact-dataset"
    assert archive.revision == "1123f36219fdeb261212a73df750be6278a697bb"
    assert archive.filename == "data/autodidact-tinystories-v1.tar.zst"
    assert archive.size == 454_752_409
    assert archive.sha256 == ("49fa417804c3e905cf986392d2397ec58e55317925e31021c7cb128417e153ac")
    assert archive.expanded_file_bytes == 1_183_154_998
    assert archive.regular_file_count == 120


def test_archive_extracts_atomically_and_preserves_dataset(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "prepared.tar.zst"
    archive = _archive_tree(prepared_dataset, archive_path)
    output_root = tmp_path / "installed"
    try:
        manifest = extract_prepared_archive(
            archive_path,
            output_root,
            archive=archive,
        )
        verified = verify_dataset(output_root, scope="all")
        assert manifest == verified
        assert manifest["tokenizer"] == verified["tokenizer"]
        assert set(manifest["splits"]) == {
            "train",
            "dev",
            "promotion",
            "sealed_final",
        }
    finally:
        make_tree_writable(output_root)
        shutil.rmtree(output_root, ignore_errors=True)


def test_corrupt_archive_is_rejected_before_extraction(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "prepared.tar.zst"
    archive = _archive_tree(prepared_dataset, archive_path)
    with archive_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))

    output_root = tmp_path / "installed"
    with pytest.raises(PreparedArchiveError, match="hash mismatch"):
        extract_prepared_archive(archive_path, output_root, archive=archive)
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("name", "member_type", "message"),
    [
        ("tinystories-v1/../escape.txt", tarfile.REGTYPE, "unsafe path"),
        ("tinystories-v1/link", tarfile.SYMTYPE, "unsupported archive member"),
    ],
)
def test_unsafe_archive_members_are_rejected(
    tmp_path: Path,
    name: str,
    member_type: bytes,
    message: str,
) -> None:
    archive_path = tmp_path / "unsafe.tar.zst"
    archive = _single_member_archive(
        archive_path,
        name=name,
        member_type=member_type,
    )
    output_root = tmp_path / "installed"

    with pytest.raises(PreparedArchiveError, match=message):
        extract_prepared_archive(archive_path, output_root, archive=archive)
    assert not output_root.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_download_uses_pinned_private_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded = tmp_path / "prepared.tar.zst"
    downloaded.write_bytes(b"archive")
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr("autodidact.data.archive.hf_hub_download", fake_download)
    result = download_prepared_archive(tmp_path, archive=PREPARED_DATASET_ARCHIVE)

    assert result == downloaded
    assert calls == [
        {
            "repo_id": "Flownium/autodidact-dataset",
            "filename": "data/autodidact-tinystories-v1.tar.zst",
            "revision": "1123f36219fdeb261212a73df750be6278a697bb",
            "repo_type": "dataset",
            "cache_dir": tmp_path,
        }
    ]


def test_fetch_verifies_an_existing_output_without_downloading(
    prepared_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("download should not run")

    monkeypatch.setattr("autodidact.data.archive.download_prepared_archive", unexpected_download)
    manifest = fetch_prepared_dataset(prepared_dataset)
    assert set(manifest["splits"]) == {
        "train",
        "dev",
        "promotion",
        "sealed_final",
    }


def test_verify_prepared_archive_checks_size(tmp_path: Path) -> None:
    path = tmp_path / "archive.tar.zst"
    path.write_bytes(b"small")
    with pytest.raises(PreparedArchiveError, match="size mismatch"):
        verify_prepared_archive(path)
