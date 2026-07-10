"""Artifact verification and research-plane path enforcement."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import stat
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from autodidact.data.config import SOURCE_FILES
from autodidact.data.download import sha256_file
from autodidact.data.formats import INDEX_DTYPE, TOKEN_DTYPE


class DatasetIntegrityError(RuntimeError):
    """Raised when a prepared artifact no longer matches its manifest."""


class ProtectedPathError(RuntimeError):
    """Raised when a research-plane change touches a protected path."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def policy_bytes() -> bytes:
    return resources.files("autodidact.data").joinpath("policy.json").read_bytes()


def load_policy() -> dict[str, Any]:
    return json.loads(policy_bytes())


def policy_sha256() -> str:
    return hashlib.sha256(policy_bytes()).hexdigest()


def _normalize_repo_path(path: str | Path) -> str:
    candidate = PurePosixPath(str(path).replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ProtectedPathError(f"invalid repository path: {path}")
    normalized = candidate.as_posix().removeprefix("./")
    if not normalized or normalized == ".":
        raise ProtectedPathError("repository path cannot be empty")
    return normalized


def assert_research_paths_allowed(paths: list[str | Path]) -> None:
    """Reject a proposed change unless every path matches the allowlist."""

    policy = load_policy()["research_agent"]
    allowed_patterns: list[str] = policy["allowed_repository_paths"]
    rejected: list[str] = []
    for path in paths:
        normalized = _normalize_repo_path(path)
        if not any(fnmatch.fnmatchcase(normalized, pattern) for pattern in allowed_patterns):
            rejected.append(normalized)
    if rejected:
        joined = ", ".join(sorted(rejected))
        raise ProtectedPathError(f"research change touches protected path(s): {joined}")


def _safe_artifact_path(base: Path, relative: str) -> Path:
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise DatasetIntegrityError(f"unsafe artifact path in manifest: {relative}")
    base_resolved = base.resolve()
    candidate = (base / Path(*relative_path.parts)).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise DatasetIntegrityError(f"artifact escapes its visibility root: {relative}")
    return candidate


def _assert_read_only(path: Path) -> None:
    writable = path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    if writable:
        raise DatasetIntegrityError(f"artifact is writable: {path}")


def _verify_file(base: Path, record: dict[str, Any], *, require_read_only: bool) -> Path:
    path = _safe_artifact_path(base, record["path"])
    if not path.is_file():
        raise DatasetIntegrityError(f"missing artifact: {path}")
    expected_size = record["size"]
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise DatasetIntegrityError(
            f"artifact size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != record["sha256"]:
        raise DatasetIntegrityError(
            f"artifact hash mismatch for {path}: expected {record['sha256']}, got {actual_hash}"
        )
    if require_read_only:
        _assert_read_only(path)
    return path


def _read_verified_manifest(directory: Path, *, require_read_only: bool) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    digest_path = directory / "manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise DatasetIntegrityError(f"missing manifest or digest under {directory}")
    expected = digest_path.read_text(encoding="ascii").strip()
    actual = sha256_file(manifest_path)
    if expected != actual:
        raise DatasetIntegrityError(
            f"manifest hash mismatch under {directory}: expected {expected}, got {actual}"
        )
    if require_read_only:
        _assert_read_only(manifest_path)
        _assert_read_only(digest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _visibility_root(dataset_root: Path, visibility: str) -> Path:
    if visibility not in {"public", "protected"}:
        raise DatasetIntegrityError(f"unknown artifact visibility: {visibility}")
    return dataset_root / visibility


def _verify_split(
    dataset_root: Path,
    split: dict[str, Any],
    *,
    require_read_only: bool,
) -> None:
    base = _visibility_root(dataset_root, split["visibility"])
    shard_stories = 0
    shard_tokens = 0
    shard_bytes = 0
    for shard in split["shards"]:
        _verify_file(base, shard["tokens"], require_read_only=require_read_only)
        _verify_file(base, shard["index"], require_read_only=require_read_only)
        token_path = _safe_artifact_path(base, shard["tokens"]["path"])
        index_path = _safe_artifact_path(base, shard["index"]["path"])
        if token_path.stat().st_size != shard["token_count"] * TOKEN_DTYPE.itemsize:
            raise DatasetIntegrityError(
                f"token byte length does not match token count for {token_path}"
            )
        try:
            index = np.load(index_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise DatasetIntegrityError(
                f"cannot read document index {index_path}: {error}"
            ) from error
        if index.dtype != INDEX_DTYPE:
            raise DatasetIntegrityError(f"unexpected document index dtype for {index_path}")
        if len(index) != shard["stories"]:
            raise DatasetIntegrityError(f"document count does not match index for {index_path}")
        token_counts = np.asarray(index["token_count"], dtype=np.uint64)
        offsets = np.asarray(index["offset"], dtype=np.uint64)
        expected_offsets = np.zeros(len(index), dtype=np.uint64)
        if len(index) > 1:
            expected_offsets[1:] = np.cumsum(token_counts[:-1], dtype=np.uint64)
        if not np.array_equal(offsets, expected_offsets):
            raise DatasetIntegrityError(f"document offsets are not contiguous in {index_path}")
        if int(token_counts.sum(dtype=np.uint64)) != shard["token_count"]:
            raise DatasetIntegrityError(f"token counts do not match index for {index_path}")
        index_bytes = int(np.asarray(index["utf8_bytes"], dtype=np.uint64).sum())
        if index_bytes != shard["utf8_bytes"]:
            raise DatasetIntegrityError(f"UTF-8 byte counts do not match index for {index_path}")
        shard_stories += shard["stories"]
        shard_tokens += shard["token_count"]
        shard_bytes += shard["utf8_bytes"]
    expected = (split["stories"], split["token_count"], split["utf8_bytes"])
    actual = (shard_stories, shard_tokens, shard_bytes)
    if actual != expected:
        raise DatasetIntegrityError(
            f"split totals do not match shard totals for {split['name']}: "
            f"expected {expected}, got {actual}"
        )


def _split_commitment(split: dict[str, Any]) -> dict[str, Any]:
    return {
        "content_sha256": split["content_sha256"],
        "stories": split["stories"],
        "token_count": split["token_count"],
        "utf8_bytes": split["utf8_bytes"],
    }


def _verify_pinned_source_records(dataset: dict[str, Any]) -> None:
    if dataset["revision"] == "local-source":
        return
    records = {record["role"]: record for record in dataset["files"]}
    if set(records) != {source.role for source in SOURCE_FILES}:
        raise DatasetIntegrityError("dataset manifest does not contain the pinned source roles")
    for source in SOURCE_FILES:
        record = records[source.role]
        expected = (source.filename, source.size, source.sha256, source.url)
        actual = (record["filename"], record["size"], record["sha256"], record["url"])
        if actual != expected:
            raise DatasetIntegrityError(f"pinned source record mismatch for {source.role}")


def _verify_tokenizer_contract(public_root: Path, manifest: dict[str, Any]) -> None:
    record = manifest["tokenizer"]
    tokenizer_path = _safe_artifact_path(public_root, record["artifact"]["path"])
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as error:
        raise DatasetIntegrityError(f"cannot load tokenizer {tokenizer_path}: {error}") from error
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    configured_vocab_size = manifest["pipeline"]["config"]["vocab_size"]
    policy_vocab_size = load_policy()["dataset"]["fixed_contract"]["vocab_size"]
    if not actual_vocab_size == record["vocab_size"] == configured_vocab_size:
        raise DatasetIntegrityError("tokenizer vocabulary does not match its manifest config")
    if manifest["dataset"]["revision"] != "local-source" and actual_vocab_size != policy_vocab_size:
        raise DatasetIntegrityError("tokenizer vocabulary does not match the fixed data policy")
    if tokenizer.token_to_id(record["end_of_text_token"]) != record["end_of_text_id"]:
        raise DatasetIntegrityError("end-of-text token ID does not match the tokenizer")
    if tokenizer.token_to_id(record["unknown_token"]) != record["unknown_id"]:
        raise DatasetIntegrityError("unknown token ID does not match the tokenizer")


def verify_dataset(
    dataset_root: Path,
    *,
    scope: str = "all",
    require_read_only: bool = True,
) -> dict[str, Any]:
    """Verify manifests and every artifact in the requested trust scope."""

    if scope not in {"public", "all"}:
        raise ValueError("scope must be 'public' or 'all'")
    dataset_root = dataset_root.expanduser()
    public_root = dataset_root / "public"
    public_manifest = _read_verified_manifest(public_root, require_read_only=require_read_only)
    if public_manifest.get("schema_version") != 1:
        raise DatasetIntegrityError("unsupported public manifest schema")

    _verify_file(
        public_root, public_manifest["tokenizer"]["artifact"], require_read_only=require_read_only
    )
    policy_path = _verify_file(
        public_root, public_manifest["policy"], require_read_only=require_read_only
    )
    if public_manifest["policy"]["sha256"] != policy_sha256():
        raise DatasetIntegrityError("prepared policy differs from the repository policy")
    if json.loads(policy_path.read_text(encoding="utf-8")) != load_policy():
        raise DatasetIntegrityError("prepared policy contents do not match the repository policy")
    _verify_pinned_source_records(public_manifest["dataset"])
    _verify_tokenizer_contract(public_root, public_manifest)
    if set(public_manifest["splits"]) != {"train", "dev"}:
        raise DatasetIntegrityError("public manifest must contain exactly train and dev")
    if set(public_manifest["protected_split_commitments"]) != {
        "promotion",
        "sealed_final",
    }:
        raise DatasetIntegrityError("public manifest has incomplete protected split commitments")
    for split in public_manifest["splits"].values():
        if split["visibility"] != "public":
            raise DatasetIntegrityError("public manifest references a protected split")
        _verify_split(dataset_root, split, require_read_only=require_read_only)

    if scope == "public":
        return public_manifest

    protected_root = dataset_root / "protected"
    full_manifest = _read_verified_manifest(protected_root, require_read_only=require_read_only)
    protected_manifest_hash = sha256_file(protected_root / "manifest.json")
    if protected_manifest_hash != public_manifest["protected_manifest_sha256"]:
        raise DatasetIntegrityError("protected manifest does not match its public commitment")
    for key in (
        "dataset",
        "pipeline",
        "policy",
        "schema_version",
        "split_policy",
        "tokenizer",
    ):
        if full_manifest[key] != public_manifest[key]:
            raise DatasetIntegrityError(f"public and protected manifests disagree on {key}")
    expected_visibility = {
        "train": "public",
        "dev": "public",
        "promotion": "protected",
        "sealed_final": "protected",
    }
    for name, split in full_manifest["splits"].items():
        if split["visibility"] != expected_visibility.get(name):
            raise DatasetIntegrityError(f"split has incorrect visibility: {name}")
        _verify_split(dataset_root, split, require_read_only=require_read_only)
        if split["visibility"] == "protected":
            expected = public_manifest["protected_split_commitments"].get(name)
            if expected != _split_commitment(split):
                raise DatasetIntegrityError(f"protected split commitment mismatch for {name}")
    if set(full_manifest["splits"]) != {"train", "dev", "promotion", "sealed_final"}:
        raise DatasetIntegrityError("full manifest does not contain the required four splits")

    if require_read_only:
        for path in (dataset_root, public_root, protected_root):
            _assert_read_only(path)
    return full_manifest


def seal_dataset_tree(dataset_root: Path) -> None:
    """Remove write bits after an atomic build has completed."""

    public_root = dataset_root / "public"
    protected_root = dataset_root / "protected"

    for directory, directories, files in os.walk(public_root, topdown=False):
        for filename in files:
            (Path(directory) / filename).chmod(0o444)
        for name in directories:
            (Path(directory) / name).chmod(0o555)
        Path(directory).chmod(0o555)

    for directory, directories, files in os.walk(protected_root, topdown=False):
        for filename in files:
            (Path(directory) / filename).chmod(0o400)
        for name in directories:
            (Path(directory) / name).chmod(0o500)
        Path(directory).chmod(0o500)

    dataset_root.chmod(0o555)
