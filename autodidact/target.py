"""Versioned configuration for the model under autoresearch."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from autodidact.evaluator import EvaluationError, inspect_trainer
from autodidact.records import DEFAULT_PARAMETER_CAP

TARGET_SCHEMA_VERSION = 1
DEFAULT_TARGET_CONFIG_PATH = Path("artifacts/control/target.json")
_TARGET_KEYS = frozenset(
    {
        "data_root",
        "device",
        "estimated_accelerator_hour_usd",
        "execution_location",
        "max_parameter_count",
        "name",
        "schema_version",
        "trainer_path",
    }
)


class TargetError(RuntimeError):
    """Raised when the target-model contract is invalid."""


class ExecutionLocation(StrEnum):
    LOCAL = "local"
    GPU_HOST = "gpu_host"


@dataclass(frozen=True, slots=True)
class TargetConfig:
    name: str
    data_root: Path
    device: str = "auto"
    trainer_path: str = "train.py"
    max_parameter_count: int = DEFAULT_PARAMETER_CAP
    execution_location: ExecutionLocation = ExecutionLocation.LOCAL
    estimated_accelerator_hour_usd: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 128:
            raise TargetError("target name must be nonempty text of at most 128 characters")
        if not isinstance(self.data_root, Path):
            if not isinstance(self.data_root, str):
                raise TargetError("data_root must be a filesystem path")
            object.__setattr__(self, "data_root", Path(self.data_root))
        if not isinstance(self.trainer_path, str):
            raise TargetError("trainer_path must be portable text")
        trainer = PurePosixPath(self.trainer_path.replace("\\", "/"))
        if trainer.as_posix() != "train.py":
            raise TargetError("target schema version 1 requires trainer_path to be train.py")
        if not isinstance(self.device, str) or not self.device.strip() or len(self.device) > 64:
            raise TargetError("device must be nonempty portable text")
        if type(self.max_parameter_count) is not int or self.max_parameter_count <= 0:
            raise TargetError("max_parameter_count must be a positive integer")
        if self.max_parameter_count > DEFAULT_PARAMETER_CAP:
            raise TargetError(
                f"target schema version 1 caps models at {DEFAULT_PARAMETER_CAP} parameters"
            )
        try:
            location = ExecutionLocation(self.execution_location)
        except (TypeError, ValueError) as error:
            raise TargetError("execution_location is invalid") from error
        object.__setattr__(self, "execution_location", location)
        price = self.estimated_accelerator_hour_usd
        if price is not None and (
            type(price) not in {int, float} or not math.isfinite(price) or price < 0
        ):
            raise TargetError("estimated accelerator price must be nonnegative")

    @classmethod
    def from_mapping(cls, value: Any) -> TargetConfig:
        if not isinstance(value, dict):
            raise TargetError("target configuration must be an object")
        unknown = frozenset(value) - _TARGET_KEYS
        if unknown:
            raise TargetError(f"unknown target configuration keys: {sorted(unknown)}")
        if value.get("schema_version") != TARGET_SCHEMA_VERSION:
            raise TargetError("target configuration schema version is unsupported")
        required = {"name", "data_root"}
        missing = required - value.keys()
        if missing:
            raise TargetError(f"target configuration is missing keys: {sorted(missing)}")
        data_root = value["data_root"]
        if not isinstance(data_root, str):
            raise TargetError("target data_root must be a string")
        return cls(
            name=value["name"],
            data_root=Path(data_root),
            device=value.get("device", "auto"),
            trainer_path=value.get("trainer_path", "train.py"),
            max_parameter_count=value.get("max_parameter_count", DEFAULT_PARAMETER_CAP),
            execution_location=value.get("execution_location", ExecutionLocation.LOCAL),
            estimated_accelerator_hour_usd=value.get("estimated_accelerator_hour_usd"),
        )

    @classmethod
    def from_path(cls, path: Path) -> TargetConfig:
        try:
            return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise TargetError(f"cannot read target configuration: {error}") from error

    def to_mapping(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "device": self.device,
            "estimated_accelerator_hour_usd": self.estimated_accelerator_hour_usd,
            "execution_location": self.execution_location.value,
            "max_parameter_count": self.max_parameter_count,
            "name": self.name,
            "schema_version": TARGET_SCHEMA_VERSION,
            "trainer_path": self.trainer_path,
        }

    def resolved_data_root(self, repository_root: Path) -> Path:
        root = self.data_root.expanduser()
        return root.resolve() if root.is_absolute() else (repository_root / root).resolve()


def _resolve_device(requested: str) -> str:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    try:
        device = torch.device(normalized)
    except RuntimeError as error:
        raise TargetError(f"invalid target device: {requested}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TargetError("CUDA was requested but is unavailable on this host")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise TargetError("MPS was requested but is unavailable on this host")
    return str(device)


def _write_config(path: Path, config: TargetConfig, *, force: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise TargetError(f"target configuration already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(config.to_mapping(), output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure the model under autoresearch.")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="write a local target-model contract")
    initialize.add_argument("--name", required=True)
    initialize.add_argument("--config", type=_path, default=DEFAULT_TARGET_CONFIG_PATH)
    initialize.add_argument("--data-root", type=_path, required=True)
    initialize.add_argument("--device", default="auto")
    initialize.add_argument("--trainer-path", default="train.py")
    initialize.add_argument("--max-parameter-count", type=int, default=DEFAULT_PARAMETER_CAP)
    initialize.add_argument(
        "--execution-location",
        type=ExecutionLocation,
        choices=tuple(ExecutionLocation),
        default=ExecutionLocation.LOCAL,
    )
    initialize.add_argument("--estimated-accelerator-hour-usd", type=float)
    initialize.add_argument("--force", action="store_true")
    doctor = commands.add_parser("doctor", help="inspect the target without training it")
    doctor.add_argument("--config", type=_path, default=DEFAULT_TARGET_CONFIG_PATH)
    doctor.add_argument("--repository-root", type=_path, default=Path.cwd())
    commands.add_parser("show", help="print the target-model contract").add_argument(
        "--config", type=_path, default=DEFAULT_TARGET_CONFIG_PATH
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            config = TargetConfig(
                name=args.name,
                data_root=args.data_root,
                device=args.device,
                trainer_path=args.trainer_path,
                max_parameter_count=args.max_parameter_count,
                execution_location=args.execution_location,
                estimated_accelerator_hour_usd=args.estimated_accelerator_hour_usd,
            )
            _write_config(args.config, config, force=args.force)
            payload = {"config_path": str(args.config.resolve()), **config.to_mapping()}
        else:
            config = TargetConfig.from_path(args.config)
            payload = config.to_mapping()
            if args.command == "doctor":
                repository = args.repository_root.resolve()
                trainer = repository / config.trainer_path
                inspection = inspect_trainer(
                    trainer,
                    parameter_cap=config.max_parameter_count,
                )
                payload = {
                    **payload,
                    "data_root": str(config.resolved_data_root(repository)),
                    "inspection": inspection,
                    "ready": True,
                    "resolved_device": _resolve_device(config.device),
                }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (EvaluationError, OSError, TargetError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
