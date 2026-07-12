"""Versioned contracts for targets supplied by Autodidact users."""

from __future__ import annotations

import hashlib
import json
import math
import string
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from autodidact.data.integrity import canonical_json_bytes

PLUGIN_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
_TOP_LEVEL_KEYS = frozenset(
    {
        "commands",
        "data_config_sha256",
        "editable_paths",
        "evaluator_path",
        "metric",
        "plugin_id",
        "plugin_version",
        "schema_version",
        "tokenizer_sha256",
        "trainer_path",
    }
)
_COMMAND_KEYS = frozenset({"evaluate", "inspect", "train"})
_METRIC_KEYS = frozenset({"direction", "name", "objective_offset", "objective_scale"})
_COMMON_PLACEHOLDERS = frozenset(
    {"evaluator", "parameter_cap", "python", "repository_root", "trainer"}
)
_TRAIN_PLACEHOLDERS = _COMMON_PLACEHOLDERS | frozenset(
    {
        "batch_size",
        "checkpoint",
        "device",
        "eval_batch_size",
        "metrics",
        "public_data_root",
        "seed",
        "stage",
        "token_budget",
    }
)
_EVALUATE_PLACEHOLDERS = _COMMON_PLACEHOLDERS | frozenset(
    {
        "batch_size",
        "checkpoint",
        "data_root",
        "device",
        "eval_tokens",
        "seed",
        "split",
        "stage",
    }
)


class TargetPluginError(RuntimeError):
    """Raised when an external target contract is invalid."""


class MetricDirection(StrEnum):
    LOWER = "lower"
    HIGHER = "higher"


def _portable_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TargetPluginError(f"{field} must be a nonempty repository-relative path")
    path = PurePosixPath(value.replace("\\", "/"))
    normalized = path.as_posix().removeprefix("./")
    if path.is_absolute() or ".." in path.parts or not normalized or normalized == ".":
        raise TargetPluginError(f"{field} must be a safe repository-relative path")
    return normalized


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TargetPluginError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _strict_keys(value: Any, expected: frozenset[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TargetPluginError(f"{field} must be an object")
    unknown = frozenset(value) - expected
    missing = expected - value.keys()
    if unknown:
        raise TargetPluginError(f"unknown {field} keys: {sorted(unknown)}")
    if missing:
        raise TargetPluginError(f"{field} is missing keys: {sorted(missing)}")
    return value


def _command_template(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TargetPluginError(f"commands.{name} must be a nonempty argument array")
    allowed = {
        "inspect": _COMMON_PLACEHOLDERS,
        "train": _TRAIN_PLACEHOLDERS,
        "evaluate": _EVALUATE_PLACEHOLDERS,
    }[name]
    result = []
    placeholders: set[str] = set()
    formatter = string.Formatter()
    for argument in value:
        if not isinstance(argument, str) or not argument or "\0" in argument:
            raise TargetPluginError(f"commands.{name} arguments must be nonempty text")
        try:
            parsed = tuple(formatter.parse(argument))
        except ValueError as error:
            raise TargetPluginError(f"commands.{name} contains invalid formatting") from error
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if field_name not in allowed or format_spec or conversion:
                raise TargetPluginError(
                    f"commands.{name} uses unsupported placeholder: {field_name}"
                )
            placeholders.add(field_name)
        result.append(argument)
    required = {
        "inspect": {"evaluator", "parameter_cap", "python", "trainer"},
        "train": {
            "checkpoint",
            "metrics",
            "public_data_root",
            "python",
            "seed",
            "token_budget",
            "trainer",
        },
        "evaluate": {"checkpoint", "data_root", "evaluator", "python", "trainer"},
    }[name]
    missing = required - placeholders
    if missing:
        raise TargetPluginError(f"commands.{name} is missing placeholders: {sorted(missing)}")
    expected_prefix = {
        "inspect": ("{python}", "{evaluator}"),
        "train": ("{python}", "{trainer}"),
        "evaluate": ("{python}", "{evaluator}"),
    }[name]
    if tuple(result[:2]) != expected_prefix:
        raise TargetPluginError(
            f"commands.{name} must start with {' '.join(expected_prefix)}"
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class MetricContract:
    name: str
    direction: MetricDirection
    objective_offset: float
    objective_scale: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 128:
            raise TargetPluginError("metric.name must be nonempty text of at most 128 characters")
        try:
            direction = MetricDirection(self.direction)
        except (TypeError, ValueError) as error:
            raise TargetPluginError("metric.direction must be lower or higher") from error
        object.__setattr__(self, "direction", direction)
        for field, value in (
            ("objective_offset", self.objective_offset),
            ("objective_scale", self.objective_scale),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TargetPluginError(f"metric.{field} must be finite numeric data")
            if not math.isfinite(value):
                raise TargetPluginError(f"metric.{field} must be finite numeric data")
        if self.objective_offset < 0.0:
            raise TargetPluginError("metric.objective_offset must be nonnegative")
        if self.objective_scale <= 0.0:
            raise TargetPluginError("metric.objective_scale must be positive")

    def canonical_objective(self, raw_value: Any) -> float:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TargetPluginError("target evaluator metric must be finite numeric data")
        raw = float(raw_value)
        if not math.isfinite(raw):
            raise TargetPluginError("target evaluator metric must be finite numeric data")
        sign = 1.0 if self.direction is MetricDirection.LOWER else -1.0
        objective = self.objective_offset + sign * self.objective_scale * raw
        if not math.isfinite(objective) or objective < 0.0:
            raise TargetPluginError(
                "metric transform produced a negative canonical objective; adjust its offset"
            )
        return objective


@dataclass(frozen=True, slots=True)
class TargetPluginSpec:
    plugin_id: str
    plugin_version: str
    trainer_path: str
    evaluator_path: str
    editable_paths: tuple[str, ...]
    metric: MetricContract
    inspect_command: tuple[str, ...]
    train_command: tuple[str, ...]
    evaluate_command: tuple[str, ...]
    data_config_sha256: str
    tokenizer_sha256: str

    def __post_init__(self) -> None:
        identities = (("plugin_id", self.plugin_id), ("plugin_version", self.plugin_version))
        for field, value in identities:
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise TargetPluginError(f"{field} must be nonempty text of at most 128 characters")
        trainer = _portable_path(self.trainer_path, field="trainer_path")
        evaluator = _portable_path(self.evaluator_path, field="evaluator_path")
        editable = tuple(
            _portable_path(path, field="editable_paths") for path in self.editable_paths
        )
        if not editable or len(set(editable)) != len(editable):
            raise TargetPluginError("editable_paths must be a nonempty unique sequence")
        if trainer not in editable:
            raise TargetPluginError("editable_paths must include trainer_path")
        if evaluator in editable:
            raise TargetPluginError("the protected evaluator cannot be research-editable")
        object.__setattr__(self, "trainer_path", trainer)
        object.__setattr__(self, "evaluator_path", evaluator)
        object.__setattr__(self, "editable_paths", editable)
        _sha256(self.data_config_sha256, field="data_config_sha256")
        _sha256(self.tokenizer_sha256, field="tokenizer_sha256")

    @classmethod
    def from_mapping(cls, value: Any) -> TargetPluginSpec:
        mapping = _strict_keys(value, _TOP_LEVEL_KEYS, field="target plugin")
        if mapping["schema_version"] != PLUGIN_SCHEMA_VERSION:
            raise TargetPluginError("target plugin schema version is unsupported")
        commands = _strict_keys(mapping["commands"], _COMMAND_KEYS, field="commands")
        metric = _strict_keys(mapping["metric"], _METRIC_KEYS, field="metric")
        editable = mapping["editable_paths"]
        if not isinstance(editable, list):
            raise TargetPluginError("editable_paths must be an array")
        return cls(
            plugin_id=mapping["plugin_id"],
            plugin_version=mapping["plugin_version"],
            trainer_path=mapping["trainer_path"],
            evaluator_path=mapping["evaluator_path"],
            editable_paths=tuple(editable),
            metric=MetricContract(
                name=metric["name"],
                direction=metric["direction"],
                objective_offset=metric["objective_offset"],
                objective_scale=metric["objective_scale"],
            ),
            inspect_command=_command_template(commands["inspect"], name="inspect"),
            train_command=_command_template(commands["train"], name="train"),
            evaluate_command=_command_template(commands["evaluate"], name="evaluate"),
            data_config_sha256=_sha256(
                mapping["data_config_sha256"], field="data_config_sha256"
            ),
            tokenizer_sha256=_sha256(mapping["tokenizer_sha256"], field="tokenizer_sha256"),
        )

    @classmethod
    def from_path(cls, path: Path) -> TargetPluginSpec:
        try:
            return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise TargetPluginError(f"cannot read target plugin contract: {error}") from error

    def contract_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_mapping())).hexdigest()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "commands": {
                "evaluate": list(self.evaluate_command),
                "inspect": list(self.inspect_command),
                "train": list(self.train_command),
            },
            "data_config_sha256": self.data_config_sha256,
            "editable_paths": list(self.editable_paths),
            "evaluator_path": self.evaluator_path,
            "metric": {
                "direction": self.metric.direction.value,
                "name": self.metric.name,
                "objective_offset": self.metric.objective_offset,
                "objective_scale": self.metric.objective_scale,
            },
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "schema_version": PLUGIN_SCHEMA_VERSION,
            "tokenizer_sha256": self.tokenizer_sha256,
            "trainer_path": self.trainer_path,
        }

    def render_command(self, name: str, values: dict[str, object]) -> list[str]:
        template = {
            "evaluate": self.evaluate_command,
            "inspect": self.inspect_command,
            "train": self.train_command,
        }.get(name)
        if template is None:
            raise TargetPluginError(f"unknown target command: {name}")
        try:
            return [argument.format_map(values) for argument in template]
        except KeyError as error:
            raise TargetPluginError(
                f"target command {name} is missing placeholder value: {error.args[0]}"
            ) from error


def resolve_repository_path(repository_root: Path, relative: str) -> Path:
    root = repository_root.resolve()
    path = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not path.is_relative_to(root):
        raise TargetPluginError("target plugin path escapes the repository")
    return path
