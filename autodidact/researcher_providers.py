"""Native CLI integrations for supported research proposal agents."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from autodidact.integrity import canonical_json_bytes
from autodidact.researcher import (
    CommandResearcherAdapter,
    InvocationResult,
    ResearcherAdapter,
    ResearcherConfig,
    ResearcherError,
    ResearcherProvider,
    ResearcherUsage,
    ResearchRequest,
    run_researcher_process,
)

_RESPONSE_KEYS = frozenset({"status", "proposal", "failure_reason", "usage"})
_PROPOSAL_TEXT_FIELDS = (
    "title",
    "hypothesis",
    "mechanism",
    "change",
    "resource_risk",
    "failure_signal",
    "interaction_risk",
)
_COMPATIBILITY_SCHEMA_VERSION = 1
_CAPABILITY_CONTRACTS: dict[ResearcherProvider, tuple[tuple[str, tuple[str, ...]], ...]] = {
    ResearcherProvider.CODEX: (
        ("ephemeral execution", ("--ephemeral",)),
        ("isolated user configuration", ("--ignore-user-config",)),
        ("workspace sandbox", ("--sandbox",)),
        ("JSON event output", ("--json",)),
        ("response schema", ("--output-schema",)),
        ("final response file", ("--output-last-message",)),
    ),
    ResearcherProvider.CLAUDE_CODE: (
        ("non-interactive execution", ("--print",)),
        ("JSON output", ("--output-format",)),
        ("response schema", ("--json-schema",)),
        ("ephemeral session", ("--no-session-persistence",)),
        ("isolated MCP configuration", ("--strict-mcp-config",)),
        ("permission mode", ("--permission-mode",)),
        ("tool restriction", ("--tools",)),
        ("allowed tools", ("--allowedTools", "--allowed-tools")),
    ),
    ResearcherProvider.HERMES_AGENT: (
        ("one-shot execution", ("--oneshot", "-z")),
        ("usage report", ("--usage-file",)),
        ("toolset restriction", ("--toolsets",)),
        ("non-interactive approval", ("--ignore-rules",)),
    ),
}


def response_json_schema() -> dict[str, Any]:
    proposal_properties: dict[str, Any] = {
        name: {"type": "string"} for name in _PROPOSAL_TEXT_FIELDS
    }
    proposal_properties.update(
        {
            "expected_effect": {"type": "number"},
            "minimum_useful_gain": {"type": "number"},
        }
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "failure_reason": {"type": ["string", "null"]},
            "proposal": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": proposal_properties,
                        "required": sorted(proposal_properties),
                    },
                    {"type": "null"},
                ]
            },
            "status": {"type": "string", "enum": ["proposed", "no_change", "failed"]},
            "usage": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "input_tokens": {"type": "integer"},
                    "output_tokens": {"type": "integer"},
                },
                "required": ["input_tokens", "output_tokens"],
            },
        },
        "required": sorted(_RESPONSE_KEYS),
    }


def _usage(value: Any) -> ResearcherUsage | None:
    if not isinstance(value, dict):
        return None
    input_tokens = value.get("input_tokens")
    output_tokens = value.get("output_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        return None
    if type(output_tokens) is not int or output_tokens < 0:
        return None
    return ResearcherUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _probe_version(
    executable: str,
    *,
    workspace: Path,
    environment: dict[str, str],
) -> tuple[str | None, bytes | None]:
    try:
        completed = subprocess.run(
            [executable, "--version"],
            cwd=workspace,
            env=environment,
            capture_output=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, str(error).encode("utf-8", errors="replace")
    output = (completed.stdout or completed.stderr).decode("utf-8", errors="replace").strip()
    if completed.returncode != 0 or not output:
        detail = completed.stderr or completed.stdout or b"version probe failed"
        return None, detail
    return output[:1_000], None


def _has_flag(help_text: str, flag: str) -> bool:
    return (
        re.search(rf"(?<![A-Za-z0-9_-]){re.escape(flag)}(?![A-Za-z0-9_-])", help_text) is not None
    )


def _capability_command(provider: ResearcherProvider, executable: str) -> list[str]:
    if provider is ResearcherProvider.CODEX:
        return [executable, "exec", "--help"]
    return [executable, "--help"]


def _probe_capabilities(
    provider: ResearcherProvider,
    executable: str,
    *,
    workspace: Path,
    environment: dict[str, str],
) -> tuple[dict[str, str], list[str], str | None]:
    try:
        completed = subprocess.run(
            _capability_command(provider, executable),
            cwd=workspace,
            env=environment,
            capture_output=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {}, [], str(error)
    raw = completed.stdout or completed.stderr
    help_text = raw.decode("utf-8", errors="replace")
    if completed.returncode != 0 or not help_text.strip():
        detail = help_text.strip() or "capability probe failed without output"
        return {}, [], detail[:4_000]
    selected: dict[str, str] = {}
    missing: list[str] = []
    for name, alternatives in _CAPABILITY_CONTRACTS[provider]:
        selected_flag = next((flag for flag in alternatives if _has_flag(help_text, flag)), None)
        if selected_flag is None:
            missing.append(name)
        else:
            selected[name] = selected_flag
    return selected, missing, None


def _issue(code: str, message: str, remediation: str) -> dict[str, str]:
    return {"code": code, "message": message, "remediation": remediation}


def probe_provider(config: ResearcherConfig, *, workspace: Path) -> dict[str, Any]:
    if config.provider is ResearcherProvider.COMMAND:
        executable = config.command[0]
    else:
        assert config.executable is not None
        executable = config.executable
    environment = {key: os.environ[key] for key in config.inherit_environment if key in os.environ}
    environment.update(dict(config.environment))
    resolved = shutil.which(executable, path=environment.get("PATH"))
    if resolved is None and Path(executable).is_file():
        resolved = str(Path(executable).resolve())
    probe_executable = resolved or executable
    issues: list[dict[str, str]] = []
    version, error = _probe_version(
        probe_executable,
        workspace=workspace.resolve(),
        environment=environment,
    )
    selected_flags: dict[str, str] = {}
    missing_capabilities: list[str] = []
    capability_error: str | None = None
    if error is not None:
        code = "executable_not_found" if resolved is None else "version_probe_failed"
        issues.append(
            _issue(
                code,
                error.decode("utf-8", errors="replace")[:4_000],
                "Run autodidact-agent doctor --fix or install a supported provider CLI.",
            )
        )
    elif config.provider is not ResearcherProvider.COMMAND:
        selected_flags, missing_capabilities, capability_error = _probe_capabilities(
            config.provider,
            probe_executable,
            workspace=workspace.resolve(),
            environment=environment,
        )
        if capability_error is not None:
            issues.append(
                _issue(
                    "capability_probe_failed",
                    capability_error,
                    "Run autodidact-agent doctor --fix or upgrade the provider CLI.",
                )
            )
        if missing_capabilities:
            issues.append(
                _issue(
                    "missing_capabilities",
                    "CLI help is missing required capabilities: " + ", ".join(missing_capabilities),
                    "Upgrade to a provider CLI version that supports every listed capability.",
                )
            )
    return {
        "compatibility_schema_version": _COMPATIBILITY_SCHEMA_VERSION,
        "executable": executable,
        "resolved_executable": resolved,
        "provider": config.provider.value,
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "ready": not issues,
        "version": version,
        "selected_flags": selected_flags,
        "missing_capabilities": missing_capabilities,
        "issues": issues,
        "error": None if not issues else issues[0]["message"],
    }


def _trusted_or_failed(result: InvocationResult) -> InvocationResult:
    if result.returncode == 0 and result.trusted_usage is None:
        return InvocationResult(
            returncode=65,
            response_bytes=result.response_bytes,
            stdout_bytes=result.stdout_bytes,
            stderr_bytes=(
                result.stderr_bytes + b"\nnative provider did not emit a trusted token-usage record"
            ),
            timed_out=result.timed_out,
            provider=result.provider,
            cli_version=result.cli_version,
            resolved_model=result.resolved_model,
            inference_provider=result.inference_provider,
            trusted_usage=None,
        )
    return result


class NativeResearcherAdapter(CommandResearcherAdapter):
    provider: ResearcherProvider

    def __init__(self, config: ResearcherConfig) -> None:
        if config.provider is not self.provider:
            raise ResearcherError(f"adapter requires provider {self.provider.value}")
        super().__init__(config)

    @property
    def executable(self) -> str:
        assert self.config.executable is not None
        return self.config.executable

    def _compatibility_or_failure(
        self,
        request: ResearchRequest,
        workspace: Path,
    ) -> tuple[dict[str, Any] | None, InvocationResult | None]:
        probe = probe_provider(self.config, workspace=workspace)
        if probe["ready"]:
            return probe, None
        detail = json.dumps(probe["issues"], sort_keys=True).encode("utf-8")
        return None, InvocationResult(
            returncode=None,
            response_bytes=b"",
            stdout_bytes=b"",
            stderr_bytes=detail,
            timed_out=False,
            provider=self.provider,
        )


class CodexResearcherAdapter(NativeResearcherAdapter):
    provider = ResearcherProvider.CODEX

    @staticmethod
    def _events(stdout: bytes) -> tuple[ResearcherUsage | None, str | None]:
        usage = None
        model = None
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if isinstance(event.get("model"), str):
                model = event["model"]
            event_usage = _usage(event.get("usage"))
            if event_usage is not None:
                usage = event_usage
        return usage, model

    def _invoke(
        self,
        request: ResearchRequest,
        workspace: Path,
        prompt: str,
    ) -> InvocationResult:
        compatibility, failure = self._compatibility_or_failure(request, workspace)
        if failure is not None:
            return failure
        assert compatibility is not None
        with tempfile.TemporaryDirectory(prefix="autodidact-codex-") as temporary:
            temporary_root = Path(temporary)
            schema_path = temporary_root / "response-schema.json"
            response_path = temporary_root / "response.json"
            schema_path.write_bytes(canonical_json_bytes(response_json_schema()))
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--sandbox",
                "workspace-write",
                "--json",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(response_path),
            ]
            if self.config.profile is not None:
                command.extend(["--profile", self.config.profile])
            if self.config.model is not None:
                command.extend(["--model", self.config.model])
            if self.config.reasoning_effort is not None:
                command.extend(
                    ["--config", f'model_reasoning_effort="{self.config.reasoning_effort}"']
                )
            command.append("-")
            returncode, stdout, stderr, timed_out = run_researcher_process(
                command,
                workspace=workspace,
                environment=self._environment(request),
                stdin_bytes=prompt.encode("utf-8"),
                timeout_seconds=self.config.timeout_seconds,
            )
            response = response_path.read_bytes() if response_path.is_file() else b""
        usage, resolved_model = self._events(stdout)
        result = InvocationResult(
            returncode=returncode,
            response_bytes=response,
            stdout_bytes=stdout,
            stderr_bytes=stderr,
            timed_out=timed_out,
            provider=self.provider,
            cli_version=compatibility["version"],
            resolved_model=resolved_model or self.config.model,
            inference_provider="openai",
            trusted_usage=usage,
        )
        return _trusted_or_failed(result)


class ClaudeCodeResearcherAdapter(NativeResearcherAdapter):
    provider = ResearcherProvider.CLAUDE_CODE

    @staticmethod
    def _decode(stdout: bytes) -> tuple[bytes, ResearcherUsage | None, str | None, bool]:
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            return b"", None, None, False
        if not isinstance(envelope, dict):
            return b"", None, None, False
        usage = _usage(envelope.get("usage"))
        model = envelope.get("model") if isinstance(envelope.get("model"), str) else None
        if set(envelope) == _RESPONSE_KEYS:
            response: Any = envelope
        else:
            response = envelope.get("structured_output", envelope.get("result"))
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return b"", usage, model, bool(envelope.get("is_error"))
        if not isinstance(response, dict):
            return b"", usage, model, bool(envelope.get("is_error"))
        return (
            canonical_json_bytes(response),
            usage,
            model,
            bool(envelope.get("is_error")),
        )

    def _invoke(
        self,
        request: ResearchRequest,
        workspace: Path,
        prompt: str,
    ) -> InvocationResult:
        compatibility, failure = self._compatibility_or_failure(request, workspace)
        if failure is not None:
            return failure
        assert compatibility is not None
        allowed_tools_flag = compatibility["selected_flags"]["allowed tools"]
        command = [
            self.executable,
            "--print",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(response_json_schema(), sort_keys=True, separators=(",", ":")),
            "--no-session-persistence",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--no-chrome",
            "--setting-sources",
            "",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Read,Edit,Bash",
            allowed_tools_flag,
            "Read",
            "Edit",
            "Bash(git diff *)",
            "Bash(git status *)",
            "Bash(uv run *)",
            "Bash(python *)",
        ]
        if self.config.model is not None:
            command.extend(["--model", self.config.model])
        if self.config.reasoning_effort is not None:
            command.extend(["--effort", self.config.reasoning_effort])
        if self.config.max_turns is not None:
            command.extend(["--max-turns", str(self.config.max_turns)])
        if self.config.max_budget_usd is not None:
            command.extend(["--max-budget-usd", str(self.config.max_budget_usd)])
        returncode, stdout, stderr, timed_out = run_researcher_process(
            command,
            workspace=workspace,
            environment=self._environment(request),
            stdin_bytes=prompt.encode("utf-8"),
            timeout_seconds=self.config.timeout_seconds,
        )
        response, usage, resolved_model, is_error = self._decode(stdout)
        if returncode == 0 and is_error:
            returncode = 65
        result = InvocationResult(
            returncode=returncode,
            response_bytes=response,
            stdout_bytes=stdout,
            stderr_bytes=stderr,
            timed_out=timed_out,
            provider=self.provider,
            cli_version=compatibility["version"],
            resolved_model=resolved_model or self.config.model,
            inference_provider="anthropic",
            trusted_usage=usage,
        )
        return _trusted_or_failed(result)


class HermesAgentResearcherAdapter(NativeResearcherAdapter):
    provider = ResearcherProvider.HERMES_AGENT

    def _invoke(
        self,
        request: ResearchRequest,
        workspace: Path,
        prompt: str,
    ) -> InvocationResult:
        compatibility, failure = self._compatibility_or_failure(request, workspace)
        if failure is not None:
            return failure
        assert compatibility is not None
        one_shot_flag = compatibility["selected_flags"]["one-shot execution"]
        request_path = workspace / f".autodidact-request-{request.request_id}.json"
        if request_path.exists():
            raise ResearcherError("temporary native request path already exists")
        with tempfile.TemporaryDirectory(prefix="autodidact-hermes-") as temporary:
            usage_path = Path(temporary) / "usage.json"
            request_path.write_text(prompt, encoding="utf-8")
            command = [
                self.executable,
                one_shot_flag,
                (
                    f"Read {request_path.name}, carry out that research request, then return "
                    "only its required JSON response."
                ),
                "--usage-file",
                str(usage_path),
                "--toolsets",
                "terminal",
                "--ignore-rules",
            ]
            if self.config.backend_provider is not None:
                command.extend(["--provider", self.config.backend_provider])
            if self.config.model is not None:
                command.extend(["--model", self.config.model])
            try:
                returncode, stdout, stderr, timed_out = run_researcher_process(
                    command,
                    workspace=workspace,
                    environment=self._environment(request),
                    stdin_bytes=b"",
                    timeout_seconds=self.config.timeout_seconds,
                )
            finally:
                request_path.unlink(missing_ok=True)
            try:
                usage_report = json.loads(usage_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                usage_report = None
        usage = _usage(usage_report)
        resolved_model = (
            usage_report.get("model")
            if isinstance(usage_report, dict) and isinstance(usage_report.get("model"), str)
            else self.config.model
        )
        inference_provider = (
            usage_report.get("provider")
            if isinstance(usage_report, dict) and isinstance(usage_report.get("provider"), str)
            else self.config.backend_provider
        )
        if (
            returncode == 0
            and isinstance(usage_report, dict)
            and usage_report.get("failed") is True
        ):
            returncode = 65
        result = InvocationResult(
            returncode=returncode,
            response_bytes=stdout,
            stdout_bytes=stdout,
            stderr_bytes=stderr,
            timed_out=timed_out,
            provider=self.provider,
            cli_version=compatibility["version"],
            resolved_model=resolved_model,
            inference_provider=inference_provider,
            trusted_usage=usage,
        )
        return _trusted_or_failed(result)


def build_researcher_adapter(config: ResearcherConfig) -> ResearcherAdapter:
    if config.provider is ResearcherProvider.COMMAND:
        return CommandResearcherAdapter(config)
    if config.provider is ResearcherProvider.CODEX:
        return CodexResearcherAdapter(config)
    if config.provider is ResearcherProvider.CLAUDE_CODE:
        return ClaudeCodeResearcherAdapter(config)
    if config.provider is ResearcherProvider.HERMES_AGENT:
        return HermesAgentResearcherAdapter(config)
    raise ResearcherError(f"unsupported researcher provider: {config.provider}")
