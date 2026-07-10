from __future__ import annotations

import os
from pathlib import Path

import pytest

from autodidact.data.config import DEFAULT_CONFIG
from autodidact.data.integrity import (
    DatasetIntegrityError,
    ProtectedPathError,
    assert_research_paths_allowed,
    load_policy,
    verify_dataset,
)
from autodidact.data.reader import open_evaluator_split, open_public_split


def test_policy_allows_only_the_training_file() -> None:
    assert_research_paths_allowed(["train.py"])
    with pytest.raises(ProtectedPathError, match="prepare.py"):
        assert_research_paths_allowed(["train.py", "prepare.py"])
    with pytest.raises(ProtectedPathError, match="autodidact/data/config.py"):
        assert_research_paths_allowed(["autodidact/data/config.py"])
    with pytest.raises(ProtectedPathError, match="invalid repository path"):
        assert_research_paths_allowed(["../tokenizer.json"])


def test_default_pipeline_matches_the_fixed_data_policy() -> None:
    contract = load_policy()["dataset"]["fixed_contract"]
    assert DEFAULT_CONFIG.vocab_size == contract["vocab_size"] == 1_792
    assert contract == {
        "attention": "dense-causal",
        "context_length": 256,
        "position_encoding": "rope",
        "vocab_size": 1_792,
    }


def test_public_reader_refuses_evaluator_only_splits(prepared_dataset: Path) -> None:
    train = open_public_split(prepared_dataset, "train")
    assert len(train.token_shard(0)) > 0
    assert len(train.document_index(0)) > 0
    with pytest.raises(PermissionError, match="not a public split"):
        open_public_split(prepared_dataset, "promotion")

    promotion = open_evaluator_split(prepared_dataset, "promotion")
    assert len(promotion.token_shard(0)) > 0


def test_tampering_is_detected(prepared_dataset: Path) -> None:
    manifest = verify_dataset(prepared_dataset, scope="public")
    shard_record = manifest["splits"]["train"]["shards"][0]["tokens"]
    shard = prepared_dataset / "public" / shard_record["path"]
    shard.chmod(0o600)
    with shard.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())

    with pytest.raises(DatasetIntegrityError, match="artifact hash mismatch"):
        verify_dataset(prepared_dataset, scope="public", require_read_only=False)


def test_writable_artifact_is_rejected(prepared_dataset: Path) -> None:
    manifest = verify_dataset(prepared_dataset, scope="public")
    policy = prepared_dataset / "public" / manifest["policy"]["path"]
    policy.chmod(0o644)
    with pytest.raises(DatasetIntegrityError, match="artifact is writable"):
        verify_dataset(prepared_dataset, scope="public")
