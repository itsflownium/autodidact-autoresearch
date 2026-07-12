"""Configure and diagnose native research-agent integrations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

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
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    os.fchmod(descriptor, 0o600)
    try:
        content = json.dumps(config.to_mapping(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if force:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure a native research proposal agent without storing credentials."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="validate an agent CLI and write local config")
    setup.add_argument("--provider", choices=sorted(_PROVIDER_ALIASES), required=True)
    setup.add_argument("--config", type=_path, default=DEFAULT_CONFIG_PATH)
    setup.add_argument("--executable")
    setup.add_argument("--model")
    setup.add_argument("--profile")
    setup.add_argument("--reasoning-effort")
    setup.add_argument("--backend-provider")
    setup.add_argument("--max-turns", type=int)
    setup.add_argument("--max-budget-usd", type=float)
    setup.add_argument("--timeout-seconds", type=float, default=900.0)
    setup.add_argument("--force", action="store_true")

    doctor = commands.add_parser("doctor", help="check the configured CLI without an agent call")
    doctor.add_argument("--config", type=_path, default=DEFAULT_CONFIG_PATH)
    doctor.add_argument("--workspace", type=_path, default=Path.cwd())

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
            payload = probe_provider(config, workspace=args.workspace)
            if not payload["ready"]:
                print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
                return 2
        else:
            config = _config_from_setup(args)
            probe = probe_provider(config, workspace=Path.cwd())
            if not probe["ready"]:
                raise ResearcherError(
                    f"provider executable failed its version probe: {probe['error']}"
                )
            _atomic_create_config(args.config, config, force=args.force)
            payload = {
                "config_path": str(args.config.expanduser().resolve()),
                "model": config.model,
                "profile": config.profile,
                "provider": config.provider.value,
                "ready": True,
                "version": probe["version"],
            }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ResearcherError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
