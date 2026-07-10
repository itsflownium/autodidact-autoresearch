"""Read-only accessors for prepared token shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autodidact.data.formats import TOKEN_DTYPE
from autodidact.data.integrity import verify_dataset


@dataclass(frozen=True)
class PreparedSplit:
    """A manifest-backed view over one immutable split."""

    dataset_root: Path
    manifest: dict[str, Any]

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def visibility(self) -> str:
        return self.manifest["visibility"]

    def _base(self) -> Path:
        return self.dataset_root / self.visibility

    def token_shard(self, index: int) -> np.memmap:
        record = self.manifest["shards"][index]["tokens"]
        return np.memmap(self._base() / record["path"], dtype=TOKEN_DTYPE, mode="r")

    def document_index(self, index: int) -> np.ndarray:
        record = self.manifest["shards"][index]["index"]
        return np.load(self._base() / record["path"], allow_pickle=False, mmap_mode="r")


def open_public_split(
    dataset_root: Path,
    name: str,
    *,
    verify: bool = True,
) -> PreparedSplit:
    """Open train or development data without exposing evaluator-only records."""

    if name not in {"train", "dev"}:
        raise PermissionError(f"{name} is not a public split")
    dataset_root = dataset_root.expanduser()
    if verify:
        manifest = verify_dataset(dataset_root, scope="public")
    else:
        manifest = json.loads((dataset_root / "public/manifest.json").read_text(encoding="utf-8"))
    return PreparedSplit(dataset_root, manifest["splits"][name])


def open_evaluator_split(
    dataset_root: Path,
    name: str,
    *,
    verify: bool = True,
) -> PreparedSplit:
    """Open evaluator-only data from an explicitly privileged call site."""

    if name not in {"promotion", "sealed_final"}:
        raise ValueError(f"{name} is not an evaluator-only split")
    dataset_root = dataset_root.expanduser()
    if verify:
        manifest = verify_dataset(dataset_root, scope="all")
    else:
        manifest = json.loads(
            (dataset_root / "protected/manifest.json").read_text(encoding="utf-8")
        )
    return PreparedSplit(dataset_root, manifest["splits"][name])
