from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path

import numpy as np

from autodidact.data.config import END_OF_TEXT_TOKEN
from autodidact.data.formats import INDEX_DTYPE, TOKEN_DTYPE
from autodidact.data.integrity import sha256_file, verify_dataset
from autodidact.data.pipeline import (
    build_dataset,
    evaluation_split,
    iter_stories,
)
from tests.conftest import make_tree_writable


def test_iter_stories_handles_delimiters_across_chunks(tmp_path: Path) -> None:
    source = tmp_path / "stories.txt"
    source.write_text(
        f"  first story  \n{END_OF_TEXT_TOKEN}\nsecond story\n{END_OF_TEXT_TOKEN}\n third story ",
        encoding="utf-8",
    )

    assert list(iter_stories(source, chunk_chars=17)) == [
        "first story",
        "second story",
        "third story",
    ]


def test_evaluation_split_is_deterministic_and_content_based(test_config) -> None:
    story = "A duplicate story always stays in one evaluation split."
    assert evaluation_split(story, test_config) == evaluation_split(story, test_config)
    assert evaluation_split(story, test_config) in {"dev", "promotion", "sealed_final"}


def test_build_creates_isolated_splits_and_read_only_artifacts(
    prepared_dataset: Path,
    test_config,
) -> None:
    full = verify_dataset(prepared_dataset, scope="all")
    public = verify_dataset(prepared_dataset, scope="public")

    assert set(public["splits"]) == {"train", "dev"}
    assert set(full["splits"]) == {"train", "dev", "promotion", "sealed_final"}
    assert public["tokenizer"]["vocab_size"] == test_config.vocab_size
    assert "shards" not in public["protected_split_commitments"]["promotion"]
    assert "shards" not in public["protected_split_commitments"]["sealed_final"]
    serialized_public = json.dumps(public)
    assert "protected/promotion" not in serialized_public
    assert "protected/sealed_final" not in serialized_public

    evaluation_story_count = sum(
        full["splits"][name]["stories"] for name in ("dev", "promotion", "sealed_final")
    )
    assert evaluation_story_count == 240
    assert all(full["splits"][name]["stories"] > 0 for name in full["splits"])

    train_shard = full["splits"]["train"]["shards"][0]
    token_path = prepared_dataset / "public" / train_shard["tokens"]["path"]
    index_path = prepared_dataset / "public" / train_shard["index"]["path"]
    tokens = np.memmap(token_path, mode="r", dtype=TOKEN_DTYPE)
    index = np.load(index_path, mmap_mode="r", allow_pickle=False)
    assert tokens.dtype == TOKEN_DTYPE
    assert index.dtype == INDEX_DTYPE
    assert int(index["token_count"].sum()) == len(tokens)
    assert not token_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert not (prepared_dataset / "protected").stat().st_mode & stat.S_IWUSR


def test_build_is_byte_for_byte_reproducible(
    tmp_path: Path,
    source_files: tuple[Path, Path],
    test_config,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    try:
        build_dataset(*source_files, first, config=test_config)
        build_dataset(*source_files, second, config=test_config)
        assert sha256_file(first / "public/manifest.json") == sha256_file(
            second / "public/manifest.json"
        )
        assert sha256_file(first / "protected/manifest.json") == sha256_file(
            second / "protected/manifest.json"
        )
        first_manifest = json.loads((first / "protected/manifest.json").read_text(encoding="utf-8"))
        second_manifest = json.loads(
            (second / "protected/manifest.json").read_text(encoding="utf-8")
        )
        assert first_manifest == second_manifest
    finally:
        for root in (first, second):
            make_tree_writable(root)
            shutil.rmtree(root, ignore_errors=True)
