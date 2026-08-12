"""Versioned configuration for the model under autoresearch."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from autodidact.checkpoints import file_sha256
from autodidact.records import DEFAULT_PARAMETER_CAP
from autodidact.target_plugins import (
    TargetPluginError,
    TargetPluginSpec,
    resolve_repository_path,
)

TARGET_SCHEMA_VERSION = 3
SUPPORTED_TARGET_SCHEMA_VERSIONS = frozenset({2, TARGET_SCHEMA_VERSION})
DEFAULT_TARGET_CONFIG_PATH = Path("artifacts/control/target.json")
_TARGET_KEYS = frozenset(
    {
        "data_root",
        "device",
        "estimated_accelerator_hour_usd",
        "execution_location",
        "max_parameter_count",
        "name",
        "plugin_spec_path",
        "public_data_root",
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
    plugin_spec_path: Path | None = None
    public_data_root: Path | None = None

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
        if trainer.is_absolute() or ".." in trainer.parts or trainer.as_posix() in {"", "."}:
            raise TargetError("trainer_path must be a safe repository-relative path")
        if self.plugin_spec_path is None and trainer.as_posix() != "train.py":
            raise TargetError("target configuration requires a plugin_spec_path")
        if not isinstance(self.device, str) or not self.device.strip() or len(self.device) > 64:
            raise TargetError("device must be nonempty portable text")
        if type(self.max_parameter_count) is not int or self.max_parameter_count <= 0:
            raise TargetError("max_parameter_count must be a positive integer")
        for field in ("plugin_spec_path", "public_data_root"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, Path):
                if not isinstance(value, str):
                    raise TargetError(f"{field} must be a filesystem path")
                object.__setattr__(self, field, Path(value))
        if self.plugin_spec_path is None:
            raise TargetError("target configuration requires a plugin_spec_path")
        if self.public_data_root is None:
            raise TargetError("target configuration requires public_data_root")
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
        schema_version = value.get("schema_version")
        if schema_version not in SUPPORTED_TARGET_SCHEMA_VERSIONS:
            raise TargetError("target configuration schema version is unsupported")
        required = {
            "data_root",
            "max_parameter_count",
            "name",
            "plugin_spec_path",
            "public_data_root",
            "trainer_path",
        }
        missing = required - value.keys()
        if missing:
            raise TargetError(f"target configuration is missing keys: {sorted(missing)}")
        data_root = value["data_root"]
        if not isinstance(data_root, str):
            raise TargetError("target data_root must be a string")
        plugin_spec = value.get("plugin_spec_path")
        public_data = value.get("public_data_root")
        if plugin_spec is not None and not isinstance(plugin_spec, str):
            raise TargetError("plugin_spec_path must be a string")
        if public_data is not None and not isinstance(public_data, str):
            raise TargetError("public_data_root must be a string")
        return cls(
            name=value["name"],
            data_root=Path(data_root),
            device=value.get("device", "auto"),
            trainer_path=value["trainer_path"],
            max_parameter_count=value["max_parameter_count"],
            execution_location=value.get("execution_location", ExecutionLocation.LOCAL),
            estimated_accelerator_hour_usd=value.get("estimated_accelerator_hour_usd"),
            plugin_spec_path=None if plugin_spec is None else Path(plugin_spec),
            public_data_root=None if public_data is None else Path(public_data),
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
            "plugin_spec_path": (
                None if self.plugin_spec_path is None else str(self.plugin_spec_path)
            ),
            "public_data_root": (
                None if self.public_data_root is None else str(self.public_data_root)
            ),
            "schema_version": TARGET_SCHEMA_VERSION,
            "trainer_path": self.trainer_path,
        }

    def resolved_data_root(self, repository_root: Path) -> Path:
        root = self.data_root.expanduser()
        return root.resolve() if root.is_absolute() else (repository_root / root).resolve()

    def resolved_public_data_root(self, repository_root: Path) -> Path | None:
        if self.public_data_root is None:
            return None
        root = self.public_data_root.expanduser()
        return root.resolve() if root.is_absolute() else (repository_root / root).resolve()

    def resolved_plugin_spec_path(self, repository_root: Path) -> Path | None:
        if self.plugin_spec_path is None:
            return None
        path = self.plugin_spec_path.expanduser()
        return path.resolve() if path.is_absolute() else (repository_root / path).resolve()

    def load_plugin(self, repository_root: Path) -> TargetPluginSpec:
        path = self.resolved_plugin_spec_path(repository_root)
        assert path is not None
        plugin = TargetPluginSpec.from_path(path)
        repository = repository_root.resolve()
        if path.is_relative_to(repository):
            relative_spec = path.relative_to(repository).as_posix()
            if relative_spec in plugin.editable_paths:
                raise TargetError("target plugin contract cannot be research-editable")
        if plugin.trainer_path != self.trainer_path:
            raise TargetError("target trainer_path differs from its plugin contract")
        full = self.resolved_data_root(repository_root)
        public = self.resolved_public_data_root(repository_root)
        if public is None or public == full:
            raise TargetError("external target public and protected data roots must be distinct")
        return plugin


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
    initialize.add_argument("--trainer-path", required=True)
    initialize.add_argument("--plugin-spec", type=_path, required=True)
    initialize.add_argument("--public-data-root", type=_path, required=True)
    initialize.add_argument("--max-parameter-count", type=int, required=True)
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
                plugin_spec_path=args.plugin_spec,
                public_data_root=args.public_data_root,
            )
            _write_config(args.config, config, force=args.force)
            payload = {"config_path": str(args.config.resolve()), **config.to_mapping()}
        else:
            config = TargetConfig.from_path(args.config)
            payload = config.to_mapping()
            if args.command == "doctor":
                repository = args.repository_root.resolve()
                plugin = config.load_plugin(repository)
                trainer = resolve_repository_path(repository, config.trainer_path)
                evaluator = resolve_repository_path(repository, plugin.evaluator_path)
                command = plugin.render_command(
                    "inspect",
                    {
                        "device": config.device,
                        "evaluator": evaluator,
                        "parameter_cap": config.max_parameter_count,
                        "python": sys.executable,
                        "repository_root": repository,
                        "trainer": trainer,
                    },
                )
                completed = subprocess.run(
                    command,
                    cwd=repository,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    raise TargetError(
                        "target inspection failed: "
                        + (completed.stderr.strip() or f"exit code {completed.returncode}")
                    )
                lines = [line for line in completed.stdout.splitlines() if line.strip()]
                if not lines:
                    raise TargetError("target inspection emitted no JSON")
                try:
                    inspection = json.loads(lines[-1])
                except json.JSONDecodeError as error:
                    raise TargetError("target inspection emitted invalid JSON") from error
                if (
                    not isinstance(inspection, dict)
                    or inspection.get("event") != "target_inspection"
                    or type(inspection.get("parameter_count")) is not int
                    or inspection["parameter_count"] <= 0
                    or inspection["parameter_count"] > config.max_parameter_count
                    or inspection.get("trainer_sha256") != file_sha256(trainer)
                ):
                    raise TargetError("target inspection contract mismatch")
                payload = {
                    **payload,
                    "data_root": str(config.resolved_data_root(repository)),
                    "inspection": inspection,
                    "plugin": plugin.to_mapping(),
                    "ready": True,
                    "requested_device": config.device,
                }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, TargetError, TargetPluginError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
