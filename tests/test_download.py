from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.request import Request

import pytest

from autodidact.data.config import SourceFile
from autodidact.data.download import SourceIntegrityError, download_source, verify_source


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, status: int) -> None:
        super().__init__(payload)
        self.status = status

    def getcode(self) -> int:
        return self.status


def test_source_verification_uses_size_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    payload = b"fixed source contents"
    path.write_bytes(payload)
    source = SourceFile(
        filename=path.name,
        url="https://example.invalid/source.txt",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        role="train",
    )
    verify_source(path, source)

    path.write_bytes(b"changed")
    with pytest.raises(SourceIntegrityError, match="source size mismatch"):
        verify_source(path, source)


def test_download_source_writes_and_verifies_content(tmp_path: Path) -> None:
    payload = b"a complete synthetic source"
    source = SourceFile(
        filename="source.txt",
        url="https://example.invalid/source.txt",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        role="train",
    )

    def opener(request: Request) -> FakeResponse:
        assert request.full_url == source.url
        return FakeResponse(payload, 200)

    destination = download_source(source, tmp_path, opener=opener)
    assert destination.read_bytes() == payload
    assert not destination.stat().st_mode & 0o222


def test_download_source_resumes_partial_transfer(tmp_path: Path) -> None:
    payload = b"resume this deterministic transfer"
    source = SourceFile(
        filename="source.txt",
        url="https://example.invalid/source.txt",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        role="train",
    )
    partial_size = 11
    (tmp_path / "source.txt.part").write_bytes(payload[:partial_size])

    def opener(request: Request) -> FakeResponse:
        assert request.get_header("Range") == f"bytes={partial_size}-"
        return FakeResponse(payload[partial_size:], 206)

    destination = download_source(source, tmp_path, opener=opener)
    assert destination.read_bytes() == payload


def test_download_source_finalizes_complete_partial(tmp_path: Path) -> None:
    payload = b"already complete"
    source = SourceFile(
        filename="source.txt",
        url="https://example.invalid/source.txt",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        role="train",
    )
    (tmp_path / "source.txt.part").write_bytes(payload)

    def opener(_request: Request) -> FakeResponse:
        raise AssertionError("network should not be used for a complete partial file")

    destination = download_source(source, tmp_path, opener=opener)
    assert destination.read_bytes() == payload
