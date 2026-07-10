"""Deterministic TinyStories tokenization, splitting, and sharding."""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from autodidact.data.config import (
    DATASET_REVISION,
    DEFAULT_CONFIG,
    END_OF_TEXT_TOKEN,
    SOURCE_FILES,
    STORY_DELIMITER,
    UNKNOWN_TOKEN,
    PipelineConfig,
    SourceFile,
)
from autodidact.data.download import sha256_file, verify_source
from autodidact.data.formats import INDEX_DTYPE, TOKEN_DTYPE
from autodidact.data.integrity import (
    canonical_json_bytes,
    policy_bytes,
    policy_sha256,
    seal_dataset_tree,
    verify_dataset,
    write_json,
)

ENCODE_BATCH_SIZE = 1_024


class DatasetBuildError(RuntimeError):
    """Raised when deterministic dataset preparation cannot complete."""


def iter_stories(path: Path, *, chunk_chars: int = 1 << 20) -> Iterator[str]:
    """Yield normalized stories without loading a multi-gigabyte source into memory."""

    if chunk_chars <= len(STORY_DELIMITER):
        raise ValueError("chunk_chars must be larger than the story delimiter")
    buffer = ""
    with path.open("r", encoding="utf-8", newline="") as source:
        while chunk := source.read(chunk_chars):
            buffer += chunk
            while True:
                boundary = buffer.find(STORY_DELIMITER)
                if boundary < 0:
                    break
                story = buffer[:boundary].strip()
                buffer = buffer[boundary + len(STORY_DELIMITER) :]
                if story:
                    yield story
        final_story = buffer.strip()
        if final_story:
            yield final_story


def story_content_hash(story: str) -> bytes:
    return hashlib.sha256(story.encode("utf-8")).digest()


def evaluation_split(story: str, config: PipelineConfig = DEFAULT_CONFIG) -> str:
    """Assign duplicate content to the same deterministic evaluation split."""

    namespace = config.split_namespace.encode("utf-8") + b"\0"
    digest = hashlib.sha256(namespace + story.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % config.split_modulus
    if bucket < config.dev_upper_bound:
        return "dev"
    if bucket < config.promotion_upper_bound:
        return "promotion"
    return "sealed_final"


def _batched(values: Iterable[str], size: int) -> Iterator[list[str]]:
    iterator = iter(values)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def train_tokenizer(
    train_source: Path,
    destination: Path,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> Tokenizer:
    """Train the fixed byte-level BPE tokenizer on training stories only."""

    config.validate()
    tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN))
    byte_level = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.pre_tokenizer = byte_level
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_token_frequency,
        special_tokens=[UNKNOWN_TOKEN, END_OF_TEXT_TOKEN],
        initial_alphabet=sorted(ByteLevel.alphabet()),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_stories(train_source), trainer=trainer)
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != config.vocab_size:
        raise DatasetBuildError(
            f"tokenizer produced {actual_vocab_size} tokens; expected exactly {config.vocab_size}"
        )
    tokenizer.save(str(destination), pretty=True)
    return tokenizer


def _file_record(path: Path, relative_to: Path) -> dict[str, str | int]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


@dataclass
class _ShardState:
    path: Path
    handle: Any
    token_count: int
    stories: int
    utf8_bytes: int
    index_rows: list[tuple[int, int, int, bytes]]


class ShardWriter:
    """Write document-preserving uint16 token shards and binary indexes."""

    def __init__(
        self,
        root: Path,
        *,
        split_name: str,
        visibility: str,
        token_limit: int,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.split_name = split_name
        self.visibility = visibility
        self.token_limit = token_limit
        self._state: _ShardState | None = None
        self._shards: list[dict[str, Any]] = []
        self._stories = 0
        self._tokens = 0
        self._utf8_bytes = 0
        self._content_digest = hashlib.sha256()

    def _open_shard(self) -> _ShardState:
        number = len(self._shards)
        path = self.root / f"shard-{number:05d}.bin"
        return _ShardState(
            path=path,
            handle=path.open("wb"),
            token_count=0,
            stories=0,
            utf8_bytes=0,
            index_rows=[],
        )

    def add(self, story: str, token_ids: Sequence[int]) -> None:
        if not token_ids:
            raise DatasetBuildError("refusing to write an empty tokenized story")
        if max(token_ids) >= 2**16 or min(token_ids) < 0:
            raise DatasetBuildError("token ID does not fit the uint16 shard format")
        if self._state is None:
            self._state = self._open_shard()
        elif self._state.token_count and (
            self._state.token_count + len(token_ids) > self.token_limit
        ):
            self._close_shard()
            self._state = self._open_shard()

        state = self._state
        assert state is not None
        encoded = np.asarray(token_ids, dtype=TOKEN_DTYPE)
        state.handle.write(encoded.tobytes(order="C"))

        content_hash = story_content_hash(story)
        byte_count = len(story.encode("utf-8"))
        state.index_rows.append((state.token_count, len(token_ids), byte_count, content_hash))
        state.token_count += len(token_ids)
        state.stories += 1
        state.utf8_bytes += byte_count
        self._stories += 1
        self._tokens += len(token_ids)
        self._utf8_bytes += byte_count
        self._content_digest.update(content_hash)

    def _close_shard(self) -> None:
        state = self._state
        if state is None:
            return
        state.handle.flush()
        os.fsync(state.handle.fileno())
        state.handle.close()

        index_path = state.path.with_suffix(".idx.npy")
        index = np.asarray(state.index_rows, dtype=INDEX_DTYPE)
        np.save(index_path, index, allow_pickle=False)
        self._shards.append(
            {
                "index": _file_record(index_path, self.root.parent),
                "stories": state.stories,
                "token_count": state.token_count,
                "tokens": _file_record(state.path, self.root.parent),
                "utf8_bytes": state.utf8_bytes,
            }
        )
        self._state = None

    def finish(self) -> dict[str, Any]:
        self._close_shard()
        if self._stories == 0:
            raise DatasetBuildError(f"split {self.split_name} received no stories")
        return {
            "content_sha256": self._content_digest.hexdigest(),
            "name": self.split_name,
            "shards": self._shards,
            "stories": self._stories,
            "token_count": self._tokens,
            "utf8_bytes": self._utf8_bytes,
            "visibility": self.visibility,
        }


def _encode_stories(
    stories: Iterable[str],
    tokenizer: Tokenizer,
    writer_for_story: Any,
    end_of_text_id: int,
) -> None:
    for batch in _batched(stories, ENCODE_BATCH_SIZE):
        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        for story, encoding in zip(batch, encodings, strict=True):
            writer = writer_for_story(story)
            writer.add(story, [*encoding.ids, end_of_text_id])


def _source_manifest(
    train_source: Path,
    validation_source: Path,
    expected_sources: Sequence[SourceFile] | None,
) -> dict[str, Any]:
    by_role = {source.role: source for source in expected_sources or ()}
    records = []
    for role, path in (("train", train_source), ("validation_source", validation_source)):
        expected = by_role.get(role)
        record: dict[str, Any] = {
            "filename": path.name,
            "role": role,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        if expected is not None:
            record["url"] = expected.url
        records.append(record)
    return {
        "files": records,
        "license": "CDLA-Sharing-1.0",
        "name": "TinyStories",
        "repository": "roneneldan/TinyStories",
        "revision": DATASET_REVISION if expected_sources else "local-source",
    }


def _manifest_base(
    *,
    config: PipelineConfig,
    dataset: dict[str, Any],
    tokenizer_record: dict[str, Any],
    policy_record: dict[str, Any],
) -> dict[str, Any]:
    config_manifest = config.as_manifest()
    return {
        "dataset": dataset,
        "pipeline": {
            "config": config_manifest,
            "config_sha256": hashlib.sha256(canonical_json_bytes(config_manifest)).hexdigest(),
            "numpy_version": np.__version__,
            "tokenizers_version": importlib.metadata.version("tokenizers"),
        },
        "policy": policy_record,
        "schema_version": config.schema_version,
        "split_policy": {
            "algorithm": "sha256-content-bucket",
            "dev": [0, config.dev_upper_bound],
            "modulus": config.split_modulus,
            "namespace": config.split_namespace,
            "promotion": [config.dev_upper_bound, config.promotion_upper_bound],
            "sealed_final": [config.promotion_upper_bound, config.split_modulus],
            "source": "validation_source",
        },
        "tokenizer": tokenizer_record,
    }


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> str:
    path = directory / "manifest.json"
    write_json(path, manifest)
    digest = sha256_file(path)
    (directory / "manifest.sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _commitment(split: dict[str, Any]) -> dict[str, Any]:
    return {
        "content_sha256": split["content_sha256"],
        "stories": split["stories"],
        "token_count": split["token_count"],
        "utf8_bytes": split["utf8_bytes"],
    }


def build_dataset(
    train_source: Path,
    validation_source: Path,
    output_root: Path,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
    expected_sources: Sequence[SourceFile] | None = None,
) -> dict[str, Any]:
    """Build all four splits atomically and seal the completed artifact tree."""

    config.validate()
    train_source = train_source.expanduser().resolve()
    validation_source = validation_source.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"dataset root already exists: {output_root}; verify it or choose a new versioned root"
        )
    for source in expected_sources or ():
        source_path = train_source if source.role == "train" else validation_source
        verify_source(source_path, source)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        public_root = staging / "public"
        protected_root = staging / "protected"
        public_root.mkdir()
        protected_root.mkdir()

        tokenizer_path = public_root / "tokenizer.json"
        tokenizer = train_tokenizer(train_source, tokenizer_path, config)
        end_of_text_id = tokenizer.token_to_id(END_OF_TEXT_TOKEN)
        unknown_id = tokenizer.token_to_id(UNKNOWN_TOKEN)
        if end_of_text_id is None or unknown_id is None:
            raise DatasetBuildError("tokenizer is missing required special tokens")

        copied_policy = public_root / "data_policy.json"
        copied_policy.write_bytes(policy_bytes())
        policy_record = _file_record(copied_policy, public_root)
        if policy_record["sha256"] != policy_sha256():
            raise DatasetBuildError("copied data policy hash changed unexpectedly")

        writers = {
            "train": ShardWriter(
                public_root / "train",
                split_name="train",
                visibility="public",
                token_limit=config.shard_token_limit,
            ),
            "dev": ShardWriter(
                public_root / "dev",
                split_name="dev",
                visibility="public",
                token_limit=config.shard_token_limit,
            ),
            "promotion": ShardWriter(
                protected_root / "promotion",
                split_name="promotion",
                visibility="protected",
                token_limit=config.shard_token_limit,
            ),
            "sealed_final": ShardWriter(
                protected_root / "sealed_final",
                split_name="sealed_final",
                visibility="protected",
                token_limit=config.shard_token_limit,
            ),
        }

        _encode_stories(
            iter_stories(train_source), tokenizer, lambda _story: writers["train"], end_of_text_id
        )
        _encode_stories(
            iter_stories(validation_source),
            tokenizer,
            lambda story: writers[evaluation_split(story, config)],
            end_of_text_id,
        )
        splits = {name: writer.finish() for name, writer in writers.items()}

        tokenizer_record = {
            "artifact": _file_record(tokenizer_path, public_root),
            "end_of_text_id": end_of_text_id,
            "end_of_text_token": END_OF_TEXT_TOKEN,
            "model": "byte-level-bpe",
            "unknown_id": unknown_id,
            "unknown_token": UNKNOWN_TOKEN,
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        }
        dataset_manifest = _source_manifest(train_source, validation_source, expected_sources)
        base = _manifest_base(
            config=config,
            dataset=dataset_manifest,
            tokenizer_record=tokenizer_record,
            policy_record=policy_record,
        )

        full_manifest = {**base, "splits": splits}
        protected_manifest_sha256 = _write_manifest(protected_root, full_manifest)
        public_manifest = {
            **base,
            "protected_manifest_sha256": protected_manifest_sha256,
            "protected_split_commitments": {
                name: _commitment(splits[name]) for name in ("promotion", "sealed_final")
            },
            "splits": {name: splits[name] for name in ("train", "dev")},
        }
        _write_manifest(public_root, public_manifest)

        verify_dataset(staging, scope="all", require_read_only=False)
        os.replace(staging, output_root)
        seal_dataset_tree(output_root)
        return verify_dataset(output_root, scope="all", require_read_only=True)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def build_pinned_dataset(
    raw_dir: Path,
    output_root: Path,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    by_role = {source.role: source for source in SOURCE_FILES}
    return build_dataset(
        raw_dir / by_role["train"].filename,
        raw_dir / by_role["validation_source"].filename,
        output_root,
        config=config,
        expected_sources=SOURCE_FILES,
    )
