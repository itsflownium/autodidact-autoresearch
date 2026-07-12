"""Diagnose and repair native research-agent CLI installations."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from autodidact.researcher import (
    ResearcherConfig,
    ResearcherProvider,
    default_researcher_executable,
)
from autodidact.researcher_providers import probe_provider

_INSTALL_HELP = {
    ResearcherProvider.CODEX: (
        "Install Codex with `npm install -g @openai/codex`: "
        "https://github.com/openai/codex/blob/main/README.md"
    ),
    ResearcherProvider.CLAUDE_CODE: (
        "Install Claude Code with `npm install -g @anthropic-ai/claude-code`: "
        "https://docs.anthropic.com/en/docs/claude-code/getting-started"
    ),
    ResearcherProvider.HERMES_AGENT: (
        "Use the official Hermes Agent installer for this operating system: "
        "https://github.com/NousResearch/hermes-agent/blob/main/website/docs/"
        "getting-started/installation.md"
    ),
}


def _candidate_executables(config: ResearcherConfig) -> list[str]:
    assert config.executable is not None
    name = default_researcher_executable(config.provider)
    candidates = [config.executable]
    discovered = shutil.which(name)
    if discovered is not None:
        candidates.append(discovered)
    home = Path.home()
    common = [
        home / ".local" / "bin" / name,
        home / ".npm-global" / "bin" / name,
        Path("/usr/local/bin") / name,
        Path("/opt/homebrew/bin") / name,
    ]
    if os.name == "nt":
        suffixes = (".exe", ".cmd", ".bat", "")
        roots = [
            Path(os.environ.get("APPDATA", "")) / "npm",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / name,
        ]
        common.extend(root / f"{name}{suffix}" for root in roots for suffix in suffixes)
    candidates.extend(str(path) for path in common if path.is_file())
    return list(dict.fromkeys(candidates))


def _repair_command(config: ResearcherConfig, probe: dict[str, Any]) -> list[str] | None:
    installed = probe["resolved_executable"] is not None
    if config.provider is ResearcherProvider.CODEX:
        if installed:
            return [str(probe["resolved_executable"]), "--upgrade"]
        return ["npm", "install", "-g", "@openai/codex"]
    if config.provider is ResearcherProvider.CLAUDE_CODE:
        if installed:
            return [str(probe["resolved_executable"]), "update"]
        return ["npm", "install", "-g", "@anthropic-ai/claude-code"]
    if config.provider is ResearcherProvider.HERMES_AGENT and installed:
        return [str(probe["resolved_executable"]), "update"]
    return None


def _action(
    code: str,
    status: str,
    message: str,
    *,
    command: list[str] | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "command": command,
        "message": message,
        "status": status,
        "stderr": stderr,
        "stdout": stdout,
    }


def repair_provider(
    config: ResearcherConfig,
    *,
    workspace: Path,
    timeout_seconds: float = 300.0,
) -> tuple[ResearcherConfig, list[dict[str, Any]], dict[str, Any]]:
    """Repair discoverable native CLI failures without invoking an inference request."""
    initial = probe_provider(config, workspace=workspace)
    if initial["ready"]:
        return config, [], initial

    actions: list[dict[str, Any]] = []
    for executable in _candidate_executables(config):
        candidate = replace(config, executable=executable)
        candidate_probe = probe_provider(candidate, workspace=workspace)
        if candidate_probe["ready"]:
            if executable != config.executable:
                actions.append(
                    _action(
                        "select_compatible_executable",
                        "fixed",
                        f"Selected compatible {config.provider.value} executable.",
                    )
                )
            return candidate, actions, candidate_probe

    command = _repair_command(config, initial)
    if command is None:
        actions.append(
            _action(
                "manual_install_required",
                "manual",
                _INSTALL_HELP[config.provider],
            )
        )
        return config, actions, initial

    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        actions.append(
            _action(
                "repair_command",
                "failed",
                f"Automatic repair could not run. {_INSTALL_HELP[config.provider]}",
                command=command,
                stderr=str(error)[:4_000],
            )
        )
        return config, actions, probe_provider(config, workspace=workspace)

    final = probe_provider(config, workspace=workspace)
    repaired = config
    if not final["ready"]:
        repaired = replace(
            config,
            executable=default_researcher_executable(config.provider),
        )
        final = probe_provider(repaired, workspace=workspace)
    actions.append(
        _action(
            "repair_command",
            "fixed" if completed.returncode == 0 and final["ready"] else "failed",
            (
                "Provider repair completed and passed compatibility checks."
                if completed.returncode == 0 and final["ready"]
                else (
                    "Provider repair did not restore compatibility. "
                    + _INSTALL_HELP[config.provider]
                )
            ),
            command=command,
            stdout=completed.stdout[-4_000:],
            stderr=completed.stderr[-4_000:],
        )
    )
    return repaired, actions, final
