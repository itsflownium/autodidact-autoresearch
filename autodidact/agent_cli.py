"""Configure and diagnose native research-agent integrations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

from autodidact.agent_compat import repair_provider
from autodidact.researcher import (
    DEFAULT_CONFIG_PATH,
    RESEARCHER_CONFIG_SCHEMA_VERSION,
    ResearcherConfig,
    ResearcherError,
    ResearcherProvider,
    default_researcher_executable,
)
from autodidact.researcher_providers import probe_provider

_PROVIDER_ALIASES = {
    "codex": ResearcherProvider.CODEX,
    "claude-code": ResearcherProvider.CLAUDE_CODE,
    "hermes-agent": ResearcherProvider.HERMES_AGENT,
}


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _atomic_create_config(path: Path, config: ResearcherConfig, *, force: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise ResearcherError(f"researcher configuration already exists: {path}")
    content = json.dumps(config.to_mapping(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not force:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise ResearcherError(f"researcher configuration already exists: {path}") from error
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            if os.name != "nt":
                path.chmod(0o600)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    if hasattr(os, "fchmod"):
        os.fchmod(descriptor, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            path.chmod(0o600)
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _config_from_setup(args: argparse.Namespace) -> ResearcherConfig:
    provider = _PROVIDER_ALIASES[args.provider]
    if provider is ResearcherProvider.HERMES_AGENT:
        if (args.backend_provider is None) != (args.model is None):
            raise ResearcherError(
                "Hermes setup requires --backend-provider and --model together, or neither"
            )
    elif args.backend_provider is not None:
        raise ResearcherError("--backend-provider is supported only by Hermes Agent")
    if args.profile is not None and provider is not ResearcherProvider.CODEX:
        raise ResearcherError("--profile is supported only by Codex")
    return ResearcherConfig(
        provider=provider,
        executable=args.executable or default_researcher_executable(provider),
        model=args.model,
        profile=args.profile,
        reasoning_effort=args.reasoning_effort,
        backend_provider=args.backend_provider,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        timeout_seconds=args.timeout_seconds,
    )


def _add_setup_arguments(parser: argparse.ArgumentParser, *, include_fix: bool) -> None:
    parser.add_argument("--provider", choices=sorted(_PROVIDER_ALIASES), required=True)
    parser.add_argument("--config", type=_path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--executable")
    parser.add_argument("--model")
    parser.add_argument("--profile")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--backend-provider")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-budget-usd", type=float)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--force", action="store_true")
    if include_fix:
        parser.add_argument("--fix", action="store_true")
        parser.add_argument("--repair-timeout-seconds", type=float, default=300.0)


def _repair_config_permissions(path: Path) -> dict[str, object] | None:
    if os.name == "nt" or not path.exists():
        return None
    if path.stat().st_mode & 0o077 == 0:
        return None
    path.chmod(0o600)
    return {
        "code": "config_permissions",
        "command": None,
        "message": "Restricted researcher configuration permissions to 0600.",
        "status": "fixed",
        "stderr": None,
        "stdout": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure a native research proposal agent without storing credentials."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="validate an agent CLI and write local config")
    _add_setup_arguments(setup, include_fix=False)

    bootstrap = commands.add_parser(
        "bootstrap",
        help="configure an agent CLI and optionally repair common installation failures",
    )
    _add_setup_arguments(bootstrap, include_fix=True)

    doctor = commands.add_parser("doctor", help="check the configured CLI without an agent call")
    doctor.add_argument("--config", type=_path, default=DEFAULT_CONFIG_PATH)
    doctor.add_argument("--workspace", type=_path, default=Path.cwd())
    doctor.add_argument("--fix", action="store_true")
    doctor.add_argument("--repair-timeout-seconds", type=float, default=300.0)

    commands.add_parser("providers", help="list native provider identifiers")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "providers":
            payload = {
                "providers": [
                    {"setup_name": name, "provider": provider.value}
                    for name, provider in sorted(_PROVIDER_ALIASES.items())
                ],
                "schema_version": RESEARCHER_CONFIG_SCHEMA_VERSION,
            }
        elif args.command == "doctor":
            config = ResearcherConfig.from_path(args.config)
            actions: list[dict[str, object]] = []
            if args.fix:
                permission_action = _repair_config_permissions(args.config)
                if permission_action is not None:
                    actions.append(permission_action)
                original = config
                config, repair_actions, probe = repair_provider(
                    config,
                    workspace=args.workspace,
                    timeout_seconds=args.repair_timeout_seconds,
                )
                actions.extend(repair_actions)
                if config != original and probe["ready"]:
                    _atomic_create_config(args.config, config, force=True)
            else:
                probe = probe_provider(config, workspace=args.workspace)
            payload = {
                **probe,
                "config_path": str(args.config.expanduser().resolve()),
                "repair_actions": actions,
            }
            if not payload["ready"]:
                print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
                return 2
        else:
            config = _config_from_setup(args)
            requested_config = config
            config_path = args.config.expanduser().resolve()
            if args.command == "bootstrap" and config_path.exists() and not args.force:
                existing = ResearcherConfig.from_path(config_path)
                if existing != config:
                    raise ResearcherError(
                        "existing researcher configuration differs; use --force to replace it"
                    )
                config = existing
                requested_config = existing
            actions = []
            if args.command == "bootstrap" and args.fix:
                config, actions, probe = repair_provider(
                    config,
                    workspace=Path.cwd(),
                    timeout_seconds=args.repair_timeout_seconds,
                )
            else:
                probe = probe_provider(config, workspace=Path.cwd())
            if not probe["ready"]:
                payload = {**probe, "repair_actions": actions}
                print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
                return 2
            repaired_existing = (
                args.command == "bootstrap"
                and args.fix
                and config_path.exists()
                and config != requested_config
            )
            if (
                args.command == "setup"
                or not config_path.exists()
                or args.force
                or repaired_existing
            ):
                _atomic_create_config(
                    config_path,
                    config,
                    force=args.force or repaired_existing,
                )
            payload = {
                "config_path": str(config_path),
                "model": config.model,
                "profile": config.profile,
                "provider": config.provider.value,
                "ready": True,
                "repair_actions": actions,
                "version": probe["version"],
            }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ResearcherError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
