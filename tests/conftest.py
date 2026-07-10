from __future__ import annotations

import os
import random
import shutil
import string
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from autodidact.data.config import DEFAULT_CONFIG, END_OF_TEXT_TOKEN
from autodidact.data.pipeline import build_dataset


def write_stories(path: Path, stories: list[str]) -> None:
    path.write_text(
        f"\n{END_OF_TEXT_TOKEN}\n".join(stories) + f"\n{END_OF_TEXT_TOKEN}\n",
        encoding="utf-8",
    )


def make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, directories, files in os.walk(root):
        Path(directory).chmod(0o700)
        for name in directories:
            (Path(directory) / name).chmod(0o700)
        for name in files:
            (Path(directory) / name).chmod(0o600)


@pytest.fixture
def source_files(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.txt"
    valid_path = tmp_path / "valid.txt"
    subjects = ["fox", "owl", "rabbit", "whale", "child", "robot"]
    places = ["forest", "garden", "village", "river", "hill", "house"]
    actions = ["helped", "found", "carried", "shared", "built", "visited"]
    train_stories = [
        (
            f"Once upon a time story {index}, a {subjects[index % len(subjects)]} "
            f"{actions[(index // 3) % len(actions)]} a friend near the "
            f"{places[(index // 7) % len(places)]}. They talked, played, and learned "
            f"lesson number {index}: kindness makes every day brighter."
        )
        for index in range(600)
    ]
    valid_stories = [
        (
            f"Evaluation story {index}: Mina and Taro explored path {index * 17}. "
            f"They solved a small problem, returned safely, and thanked each other."
        )
        for index in range(240)
    ]
    write_stories(train_path, train_stories)
    write_stories(valid_path, valid_stories)
    return train_path, valid_path


@pytest.fixture
def test_config():
    return replace(DEFAULT_CONFIG, vocab_size=320, shard_token_limit=4_000)


@pytest.fixture
def prepared_dataset(
    tmp_path: Path,
    source_files: tuple[Path, Path],
    test_config,
) -> Iterator[Path]:
    root = tmp_path / "prepared"
    build_dataset(*source_files, root, config=test_config)
    try:
        yield root
    finally:
        make_tree_writable(root)
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def baseline_dataset(
    tmp_path: Path,
    source_files: tuple[Path, Path],
) -> Iterator[Path]:
    root = tmp_path / "baseline-prepared"
    train_path = tmp_path / "baseline-train.txt"
    random_source = random.Random(2_026)
    alphabet = string.ascii_lowercase
    stories = [
        " ".join(
            "".join(random_source.choices(alphabet, k=random_source.randint(4, 12)))
            for _word in range(32)
        )
        for _story in range(2_000)
    ]
    write_stories(train_path, stories)
    config = replace(
        DEFAULT_CONFIG,
        min_token_frequency=1,
        shard_token_limit=100_000,
    )
    build_dataset(train_path, source_files[1], root, config=config)
    try:
        yield root
    finally:
        make_tree_writable(root)
        shutil.rmtree(root, ignore_errors=True)
