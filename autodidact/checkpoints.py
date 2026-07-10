"""Canonical fingerprints for checkpoint contents and files."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _digest_length(digest: Any, length: int) -> None:
    digest.update(struct.pack(">Q", length))


def _mapping_key_bytes(value: Any) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b1" if value else b"b0"
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"f" + struct.pack(">d", value)
    if isinstance(value, str):
        return b"s" + value.encode("utf-8")
    raise TypeError(f"unsupported checkpoint mapping key: {type(value).__name__}")


def _update_semantic_digest(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(b"I")
        _digest_length(digest, len(encoded))
        digest.update(encoded)
    elif isinstance(value, float):
        digest.update(b"F" + struct.pack(">d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S")
        _digest_length(digest, len(encoded))
        digest.update(encoded)
    elif isinstance(value, bytes):
        digest.update(b"Y")
        _digest_length(digest, len(value))
        digest.update(value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(b"T")
        _update_semantic_digest(digest, str(tensor.dtype))
        _update_semantic_digest(digest, list(tensor.shape))
        _digest_length(digest, len(raw))
        digest.update(raw)
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        raw = array.tobytes()
        digest.update(b"A")
        _update_semantic_digest(digest, array.dtype.str)
        _update_semantic_digest(digest, list(array.shape))
        _digest_length(digest, len(raw))
        digest.update(raw)
    elif isinstance(value, np.generic):
        _update_semantic_digest(digest, value.item())
    elif isinstance(value, Mapping):
        digest.update(b"M")
        items = sorted(value.items(), key=lambda item: _mapping_key_bytes(item[0]))
        _digest_length(digest, len(items))
        for key, item in items:
            _update_semantic_digest(digest, key)
            _update_semantic_digest(digest, item)
    elif isinstance(value, tuple):
        digest.update(b"U")
        _digest_length(digest, len(value))
        for item in value:
            _update_semantic_digest(digest, item)
    elif isinstance(value, list):
        digest.update(b"L")
        _digest_length(digest, len(value))
        for item in value:
            _update_semantic_digest(digest, item)
    else:
        raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def checkpoint_state_sha256(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"autodidact-checkpoint-state-v1")
    _update_semantic_digest(digest, payload)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
