"""Small, model-agnostic integrity helpers used by the control plane."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any


class ProtectedPathError(RuntimeError):
    """Raised when a research change touches a path outside its declared surface."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for hashes and append-only commitments."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def normalize_repository_path(value: str, *, field: str = "repository path") -> str:
    if not isinstance(value, str) or not value:
        raise ProtectedPathError(f"{field} must be nonempty text")
    path = PurePosixPath(value.replace("\\", "/"))
    normalized = path.as_posix().removeprefix("./")
    if path.is_absolute() or ".." in path.parts or normalized in {"", "."}:
        raise ProtectedPathError(f"{field} must be a safe repository-relative path")
    return normalized


def assert_paths_allowed(paths: list[str], allowed_paths: tuple[str, ...]) -> None:
    """Reject a research patch unless every changed path is explicitly editable."""

    allowed = {normalize_repository_path(path, field="allowed path") for path in allowed_paths}
    normalized = [normalize_repository_path(path) for path in paths]
    rejected = sorted(path for path in normalized if path not in allowed)
    if rejected:
        raise ProtectedPathError(
            "research change touches protected path(s): " + ", ".join(rejected)
        )
