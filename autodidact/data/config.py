"""Versioned constants for the TinyStories data contract."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATASET_BASE_URL = (
    f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{DATASET_REVISION}"
)
END_OF_TEXT_TOKEN = "<|endoftext|>"
UNKNOWN_TOKEN = "<|unk|>"
STORY_DELIMITER = END_OF_TEXT_TOKEN


@dataclass(frozen=True)
class SourceFile:
    """A content-addressed upstream dataset file."""

    filename: str
    url: str
    size: int
    sha256: str
    role: str

    def as_manifest(self) -> dict[str, str | int]:
        return asdict(self)


SOURCE_FILES = (
    SourceFile(
        filename="TinyStories-train.txt",
        url=f"{DATASET_BASE_URL}/TinyStories-train.txt",
        size=1_924_281_556,
        sha256="c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
        role="train",
    ),
    SourceFile(
        filename="TinyStories-valid.txt",
        url=f"{DATASET_BASE_URL}/TinyStories-valid.txt",
        size=19_447_282,
        sha256="94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4",
        role="validation_source",
    ),
)


@dataclass(frozen=True)
class PreparedDatasetArchive:
    """A content-addressed prepared dataset published outside Git history."""

    repo_id: str
    revision: str
    filename: str
    root_name: str
    size: int
    sha256: str
    expanded_file_bytes: int
    regular_file_count: int


PREPARED_DATASET_ARCHIVE = PreparedDatasetArchive(
    repo_id="Flownium/autodidact-dataset",
    revision="1123f36219fdeb261212a73df750be6278a697bb",
    filename="data/autodidact-tinystories-v1.tar.zst",
    root_name="tinystories-v1",
    size=454_752_409,
    sha256="49fa417804c3e905cf986392d2397ec58e55317925e31021c7cb128417e153ac",
    expanded_file_bytes=1_183_154_998,
    regular_file_count=120,
)


@dataclass(frozen=True)
class PipelineConfig:
    """All values that affect tokenizer, split, or shard contents."""

    schema_version: int = 1
    pipeline_version: str = "tinystories-v1"
    vocab_size: int = 1_792
    min_token_frequency: int = 2
    shard_token_limit: int = 10_000_000
    split_modulus: int = 10_000
    dev_upper_bound: int = 5_000
    promotion_upper_bound: int = 7_500
    split_namespace: str = "autodidact/tinystories/evaluation/v1"
    normalization: str = "strip-surrounding-whitespace"

    def validate(self) -> None:
        if self.vocab_size >= 2**16:
            raise ValueError("vocab_size must fit in uint16 shards")
        if self.vocab_size < 258:
            raise ValueError("vocab_size must fit the byte alphabet and special tokens")
        if self.shard_token_limit <= 0:
            raise ValueError("shard_token_limit must be positive")
        if not 0 < self.dev_upper_bound < self.promotion_upper_bound < self.split_modulus:
            raise ValueError("evaluation split thresholds must be strictly ordered")

    def as_manifest(self) -> dict[str, str | int]:
        return asdict(self)


DEFAULT_CONFIG = PipelineConfig()


def default_raw_dir() -> Path:
    configured = os.environ.get("AUTODIDACT_RAW_DATA")
    return Path(configured).expanduser() if configured else Path.home() / ".cache/autodidact/raw"


def default_output_root() -> Path:
    configured = os.environ.get("AUTODIDACT_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache/autodidact/tinystories-v1"
