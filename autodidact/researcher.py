"""Generic, auditable process boundary for research proposal agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
from contextlib import suppress
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from autodidact.data.integrity import (
    ProtectedPathError,
    assert_research_paths_allowed,
    canonical_json_bytes,
)
from autodidact.records import PatchProposal

RESEARCHER_SCHEMA_VERSION = 1
RESEARCHER_CONFIG_SCHEMA_VERSION = 1
DEFAULT_RESEARCHER_TOKEN_ALLOWANCE = 1_000_000
DEFAULT_CONFIG_PATH = Path("artifacts/control/researcher.json")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/researcher")
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_DEFAULT_INHERITED_ENVIRONMENT = (
    "HOME",
    "LANG",
    "PATH",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
    "COMSPEC",
    "PATHEXT",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
)
_PROPOSAL_KEYS = frozenset(
    {
        "title",
        "hypothesis",
        "mechanism",
        "change",
        "expected_effect_bpb",
        "minimum_useful_gain_bpb",
        "resource_risk",
        "failure_signal",
        "interaction_risk",
    }
)
_RESPONSE_KEYS = frozenset({"status", "proposal", "failure_reason", "usage"})
_USAGE_KEYS = frozenset({"input_tokens", "output_tokens"})
_CONFIG_KEYS = frozenset(
    {
        "backend_provider",
        "command",
        "executable",
        "timeout_seconds",
        "max_output_bytes",
        "max_diff_bytes",
        "inherit_environment",
        "environment",
        "max_budget_usd",
        "max_turns",
        "model",
        "profile",
        "provider",
        "reasoning_effort",
        "schema_version",
    }
)
_REQUEST_KEYS = frozenset(
    {
        "request_id",
        "parent_commit",
        "proposal_number",
        "program_path",
        "previous_results",
        "allowed_paths",
        "maximum_total_tokens",
    }
)
_TRANSCRIPT_KEYS = frozenset(
    {
        "changed_paths",
        "diff",
        "diff_sha256",
        "failure_reason",
        "prompt",
        "prompt_sha256",
        "provider",
        "provider_configuration",
        "request_id",
        "response",
        "response_raw",
        "response_sha256",
        "inference_provider",
        "resolved_model",
        "returncode",
        "schema_version",
        "status",
        "stderr",
        "stdout",
        "timed_out",
        "cli_version",
        "usage_verified",
    }
)
_PROVIDER_CONFIGURATION_KEYS = frozenset(
    {
        "backend_provider",
        "max_budget_usd",
        "max_turns",
        "model",
        "profile",
        "reasoning_effort",
    }
)


class ResearcherError(RuntimeError):
    """Raised when a researcher invocation or contract is invalid."""


class ResearchStatus(StrEnum):
    PROPOSED = "proposed"
    NO_CHANGE = "no_change"
    FAILED = "failed"


class ResearcherProvider(StrEnum):
    COMMAND = "command"
    CODEX = "codex"
    CLAUDE_CODE = "claude_code"
    HERMES_AGENT = "hermes_agent"


_DEFAULT_EXECUTABLES = {
    ResearcherProvider.CODEX: "codex",
    ResearcherProvider.CLAUDE_CODE: "claude",
    ResearcherProvider.HERMES_AGENT: "hermes",
}


def _strict_keys(value: dict[str, Any], expected: frozenset[str], *, name: str) -> None:
    keys = frozenset(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ResearcherError(f"{name} keys differ; missing={missing}, extra={extra}")


def _required_text(name: str, value: Any, *, maximum: int = 4_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ResearcherError(f"{name} must be nonempty text of at most {maximum} characters")
    return value


def _nonnegative_integer(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ResearcherError(f"{name} must be a nonnegative integer")
    return value


def _git(repository: Path, *arguments: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=text,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode("utf-8", errors="replace")
        raise ResearcherError(f"git {' '.join(arguments)} failed: {stderr.strip()}")
    return completed.stdout


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    path.chmod(0o600)


def default_researcher_executable(provider: ResearcherProvider) -> str:
    try:
        return _DEFAULT_EXECUTABLES[provider]
    except KeyError as error:
        raise ResearcherError("command provider has no native executable") from error


@dataclass(frozen=True, slots=True)
class ProposalDraft:
    title: str
    hypothesis: str
    mechanism: str
    change: str
    expected_effect_bpb: float
    minimum_useful_gain_bpb: float
    resource_risk: str
    failure_signal: str
    interaction_risk: str

    def __post_init__(self) -> None:
        for name in (
            "title",
            "hypothesis",
            "mechanism",
            "change",
            "resource_risk",
            "failure_signal",
            "interaction_risk",
        ):
            _required_text(name, getattr(self, name))
        if not isinstance(self.expected_effect_bpb, (int, float)) or not math.isfinite(
            self.expected_effect_bpb
        ):
            raise ResearcherError("expected_effect_bpb must be finite")
        if (
            not isinstance(self.minimum_useful_gain_bpb, (int, float))
            or not math.isfinite(self.minimum_useful_gain_bpb)
            or self.minimum_useful_gain_bpb <= 0.0
        ):
            raise ResearcherError("minimum_useful_gain_bpb must be finite and positive")

    @classmethod
    def from_mapping(cls, value: Any) -> ProposalDraft:
        if not isinstance(value, dict):
            raise ResearcherError("proposal must be an object")
        _strict_keys(value, _PROPOSAL_KEYS, name="proposal")
        return cls(**value)

    def to_record(self, *, proposal_id: str, parent_commit: str) -> PatchProposal:
        return PatchProposal(
            proposal_id=proposal_id,
            parent_commit=parent_commit,
            **asdict(self),
        )


@dataclass(frozen=True, slots=True)
class ResearcherUsage:
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        _nonnegative_integer("input_tokens", self.input_tokens)
        _nonnegative_integer("output_tokens", self.output_tokens)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_mapping(cls, value: Any) -> ResearcherUsage:
        if not isinstance(value, dict):
            raise ResearcherError("usage must be an object")
        _strict_keys(value, _USAGE_KEYS, name="usage")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class StructuredResearchResponse:
    status: ResearchStatus
    proposal: ProposalDraft | None
    failure_reason: str | None
    usage: ResearcherUsage

    @classmethod
    def from_json(cls, raw: str) -> StructuredResearchResponse:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ResearcherError("researcher stdout must be exactly one JSON object") from error
        if not isinstance(value, dict):
            raise ResearcherError("researcher response must be an object")
        _strict_keys(value, _RESPONSE_KEYS, name="response")
        try:
            status = ResearchStatus(value["status"])
        except (TypeError, ValueError) as error:
            raise ResearcherError("response status is invalid") from error
        proposal = (
            None if value["proposal"] is None else ProposalDraft.from_mapping(value["proposal"])
        )
        reason = value["failure_reason"]
        if reason is not None:
            reason = _required_text("failure_reason", reason)
        usage = ResearcherUsage.from_mapping(value["usage"])
        if status is ResearchStatus.PROPOSED:
            if proposal is None or reason is not None:
                raise ResearcherError("proposed responses require a proposal and no failure reason")
        elif proposal is not None:
            raise ResearcherError("non-proposal responses cannot include a proposal")
        if status is ResearchStatus.FAILED and reason is None:
            raise ResearcherError("failed responses require a failure reason")
        return cls(status=status, proposal=proposal, failure_reason=reason, usage=usage)


@dataclass(frozen=True, slots=True)
class ResearchRequest:
    request_id: str
    parent_commit: str
    proposal_number: int
    program_text: str
    previous_results: tuple[dict[str, Any], ...]
    maximum_total_tokens: int = DEFAULT_RESEARCHER_TOKEN_ALLOWANCE
    allowed_paths: tuple[str, ...] = ("train.py",)

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.request_id):
            raise ResearcherError("request_id must be a portable structured ID")
        if not _COMMIT_PATTERN.fullmatch(self.parent_commit):
            raise ResearcherError("parent_commit must be a full lowercase Git commit")
        if type(self.proposal_number) is not int or self.proposal_number <= 0:
            raise ResearcherError("proposal_number must be positive")
        if type(self.maximum_total_tokens) is not int or self.maximum_total_tokens <= 0:
            raise ResearcherError("maximum_total_tokens must be positive")
        _required_text("program_text", self.program_text, maximum=200_000)
        if len(self.previous_results) > 200:
            raise ResearcherError("previous_results contains too many entries")
        try:
            encoded = canonical_json_bytes(list(self.previous_results))
        except (TypeError, ValueError) as error:
            raise ResearcherError("previous_results must contain canonical JSON values") from error
        if len(encoded) > 1_000_000:
            raise ResearcherError("previous_results is too large")
        if not self.allowed_paths or len(set(self.allowed_paths)) != len(self.allowed_paths):
            raise ResearcherError("allowed_paths must be a nonempty unique sequence")
        if self.allowed_paths == ("train.py",):
            try:
                assert_research_paths_allowed(list(self.allowed_paths))
            except (ProtectedPathError, ValueError) as error:
                raise ResearcherError(str(error)) from error
        else:
            for value in self.allowed_paths:
                if not isinstance(value, str):
                    raise ResearcherError("allowed_paths must contain portable text")
                path = PurePosixPath(value.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
                    raise ResearcherError("allowed_paths must be safe repository-relative paths")

    def prompt(self) -> str:
        payload = {
            "allowed_paths": list(self.allowed_paths),
            "contract": {
                "edit_scope": "Modify only the allowed files in the supplied workspace.",
                "evaluation": "Do not grade, schedule, reject, or promote the proposal.",
                "output": (
                    "Write exactly one response object matching the supplied schema to stdout."
                ),
            },
            "output_schema": {
                "failure_reason": "string or null",
                "proposal": {key: "required" for key in sorted(_PROPOSAL_KEYS)},
                "status": [status.value for status in ResearchStatus],
                "usage": {"input_tokens": "integer", "output_tokens": "integer"},
            },
            "parent_commit": self.parent_commit,
            "maximum_total_tokens": self.maximum_total_tokens,
            "previous_results": list(self.previous_results),
            "program_md": self.program_text,
            "proposal_number": self.proposal_number,
            "request_id": self.request_id,
            "schema_version": RESEARCHER_SCHEMA_VERSION,
        }
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class ResearcherConfig:
    command: tuple[str, ...] = ()
    provider: ResearcherProvider = ResearcherProvider.COMMAND
    executable: str | None = None
    model: str | None = None
    profile: str | None = None
    reasoning_effort: str | None = None
    backend_provider: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    timeout_seconds: float = 900.0
    max_output_bytes: int = 1_000_000
    max_diff_bytes: int = 2_000_000
    inherit_environment: tuple[str, ...] = _DEFAULT_INHERITED_ENVIRONMENT
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        try:
            provider = ResearcherProvider(self.provider)
        except (TypeError, ValueError) as error:
            raise ResearcherError("researcher provider is invalid") from error
        object.__setattr__(self, "provider", provider)
        if provider is ResearcherProvider.COMMAND:
            if not self.command or any(
                not isinstance(part, str) or not part for part in self.command
            ):
                raise ResearcherError("command provider requires a nonempty command sequence")
            if self.executable is not None:
                raise ResearcherError("command provider cannot set executable")
            if any(
                value is not None
                for value in (
                    self.model,
                    self.profile,
                    self.reasoning_effort,
                    self.backend_provider,
                    self.max_turns,
                    self.max_budget_usd,
                )
            ):
                raise ResearcherError("command provider cannot set native provider options")
        else:
            if self.command:
                raise ResearcherError("native providers cannot set command")
            if self.executable is None or not self.executable.strip():
                raise ResearcherError("native providers require an executable")
        for name in (
            "executable",
            "model",
            "profile",
            "reasoning_effort",
            "backend_provider",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or "\x00" in value
                or len(value) > 512
            ):
                raise ResearcherError(f"{name} must be nonempty portable text")
        if self.profile is not None and provider is not ResearcherProvider.CODEX:
            raise ResearcherError("profile is supported only by the codex provider")
        if self.backend_provider is not None and provider is not ResearcherProvider.HERMES_AGENT:
            raise ResearcherError("backend_provider is supported only by the hermes_agent provider")
        if provider is ResearcherProvider.HERMES_AGENT and (
            (self.backend_provider is None) != (self.model is None)
        ):
            raise ResearcherError("hermes_agent requires backend_provider and model together")
        if self.reasoning_effort is not None and provider not in {
            ResearcherProvider.CODEX,
            ResearcherProvider.CLAUDE_CODE,
        }:
            raise ResearcherError("reasoning_effort is not supported by this provider")
        if self.max_turns is not None and (type(self.max_turns) is not int or self.max_turns <= 0):
            raise ResearcherError("max_turns must be a positive integer")
        if self.max_turns is not None and provider is not ResearcherProvider.CLAUDE_CODE:
            raise ResearcherError("max_turns is not supported by this provider")
        if self.max_budget_usd is not None and (
            not isinstance(self.max_budget_usd, (int, float))
            or not math.isfinite(self.max_budget_usd)
            or self.max_budget_usd <= 0.0
        ):
            raise ResearcherError("max_budget_usd must be finite and positive")
        if self.max_budget_usd is not None and provider is not ResearcherProvider.CLAUDE_CODE:
            raise ResearcherError("max_budget_usd is supported only by claude_code")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0.0
        ):
            raise ResearcherError("timeout_seconds must be finite and positive")
        for name in ("max_output_bytes", "max_diff_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ResearcherError(f"{name} must be a positive integer")
        if any(not isinstance(key, str) for key in self.inherit_environment):
            raise ResearcherError("inherit_environment must contain strings")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            for item in self.environment
        ):
            raise ResearcherError("environment must contain string key-value pairs")
        if len(set(self.inherit_environment)) != len(self.inherit_environment):
            raise ResearcherError("inherit_environment contains duplicates")
        explicit_keys = [key for key, _value in self.environment]
        if len(set(explicit_keys)) != len(explicit_keys):
            raise ResearcherError("environment contains duplicate keys")
        for key in (*self.inherit_environment, *explicit_keys):
            if not _ENVIRONMENT_KEY_PATTERN.fullmatch(key):
                raise ResearcherError(f"invalid environment key: {key}")

    @classmethod
    def from_path(cls, path: Path) -> ResearcherConfig:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResearcherError(f"cannot read researcher configuration: {error}") from error
        if not isinstance(value, dict):
            raise ResearcherError("researcher configuration must be an object")
        unknown = frozenset(value) - _CONFIG_KEYS
        if unknown:
            raise ResearcherError(f"unknown researcher configuration keys: {sorted(unknown)}")
        schema_version = value.get("schema_version", RESEARCHER_CONFIG_SCHEMA_VERSION)
        if schema_version != RESEARCHER_CONFIG_SCHEMA_VERSION:
            raise ResearcherError("researcher configuration schema version is unsupported")
        environment = value.get("environment", {})
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in environment.items()
        ):
            raise ResearcherError("environment must map strings to strings")
        command = value.get("command", [])
        inherited = value.get("inherit_environment", list(_DEFAULT_INHERITED_ENVIRONMENT))
        if not isinstance(command, list) or not isinstance(inherited, list):
            raise ResearcherError("command and inherit_environment must be arrays")
        provider_value = value.get("provider")
        if provider_value is None:
            provider_value = ResearcherProvider.COMMAND.value
        try:
            provider = ResearcherProvider(provider_value)
        except (TypeError, ValueError) as error:
            raise ResearcherError("researcher provider is invalid") from error
        executable = value.get("executable")
        if executable is None and provider is not ResearcherProvider.COMMAND:
            executable = default_researcher_executable(provider)
        return cls(
            command=tuple(command),
            provider=provider,
            executable=executable,
            model=value.get("model"),
            profile=value.get("profile"),
            reasoning_effort=value.get("reasoning_effort"),
            backend_provider=value.get("backend_provider"),
            max_turns=value.get("max_turns"),
            max_budget_usd=value.get("max_budget_usd"),
            timeout_seconds=value.get("timeout_seconds", 900.0),
            max_output_bytes=value.get("max_output_bytes", 1_000_000),
            max_diff_bytes=value.get("max_diff_bytes", 2_000_000),
            inherit_environment=tuple(inherited),
            environment=tuple(sorted(environment.items())),
        )

    def to_mapping(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "environment": dict(self.environment),
            "inherit_environment": list(self.inherit_environment),
            "max_diff_bytes": self.max_diff_bytes,
            "max_output_bytes": self.max_output_bytes,
            "provider": self.provider.value,
            "schema_version": RESEARCHER_CONFIG_SCHEMA_VERSION,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.provider is ResearcherProvider.COMMAND:
            value["command"] = list(self.command)
        else:
            value["executable"] = self.executable
        for name in (
            "model",
            "profile",
            "reasoning_effort",
            "backend_provider",
            "max_turns",
            "max_budget_usd",
        ):
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        return value


@dataclass(frozen=True, slots=True)
class InvocationResult:
    returncode: int | None
    response_bytes: bytes
    stdout_bytes: bytes
    stderr_bytes: bytes
    timed_out: bool
    provider: ResearcherProvider = ResearcherProvider.COMMAND
    cli_version: str | None = None
    resolved_model: str | None = None
    inference_provider: str | None = None
    trusted_usage: ResearcherUsage | None = None


@dataclass(frozen=True, slots=True)
class ResearchAttempt:
    request_id: str
    status: ResearchStatus
    response: StructuredResearchResponse | None
    failure_reason: str | None
    changed_paths: tuple[str, ...]
    diff_sha256: str | None
    prompt_sha256: str
    response_sha256: str
    transcript_path: Path
    returncode: int | None
    timed_out: bool
    provider: ResearcherProvider = ResearcherProvider.COMMAND
    cli_version: str | None = None
    resolved_model: str | None = None
    inference_provider: str | None = None
    usage_verified: bool = False

    @property
    def proposal(self) -> ProposalDraft | None:
        return None if self.response is None else self.response.proposal

    @property
    def usage(self) -> ResearcherUsage:
        if self.response is None:
            return ResearcherUsage(0, 0)
        return self.response.usage


class ResearcherAdapter(Protocol):
    def run(
        self,
        request: ResearchRequest,
        *,
        workspace: Path,
        artifact_root: Path,
    ) -> ResearchAttempt: ...


def _changed_paths(workspace: Path) -> tuple[str, ...]:
    tracked = _git(workspace, "diff", "--name-only", "-z", "HEAD", text=False)
    untracked = _git(
        workspace,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        text=False,
    )
    assert isinstance(tracked, bytes) and isinstance(untracked, bytes)
    paths = {
        item.decode("utf-8") for raw in (tracked, untracked) for item in raw.split(b"\0") if item
    }
    return tuple(sorted(paths))


def load_research_attempt(
    transcript_path: Path,
    request: ResearchRequest,
    *,
    workspace: Path,
) -> ResearchAttempt:
    """Recover a completed invocation from its transcript without calling the researcher."""
    try:
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearcherError(f"cannot read researcher transcript: {error}") from error
    if not isinstance(transcript, dict):
        raise ResearcherError("researcher transcript must be an object")
    _strict_keys(transcript, _TRANSCRIPT_KEYS, name="transcript")
    if transcript["schema_version"] != RESEARCHER_SCHEMA_VERSION:
        raise ResearcherError("researcher transcript schema is unsupported")
    if transcript["request_id"] != request.request_id:
        raise ResearcherError("researcher transcript belongs to another request")

    workspace = workspace.resolve()
    head = str(_git(workspace, "rev-parse", "HEAD")).strip()
    if head != request.parent_commit:
        raise ResearcherError("research workspace moved after its recorded invocation")
    changed_paths = _changed_paths(workspace)
    if transcript["changed_paths"] != list(changed_paths):
        raise ResearcherError("researcher transcript changed paths differ from the workspace")
    diff_bytes = _git(workspace, "diff", "--binary", "HEAD", text=False)
    assert isinstance(diff_bytes, bytes)
    diff = diff_bytes.decode("utf-8", errors="replace")
    diff_hash = hashlib.sha256(diff_bytes).hexdigest() if diff_bytes else None
    if transcript["diff"] != diff or transcript["diff_sha256"] != diff_hash:
        raise ResearcherError("researcher transcript diff differs from the workspace")

    prompt = request.prompt()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if transcript["prompt"] != prompt or transcript["prompt_sha256"] != prompt_hash:
        raise ResearcherError("researcher transcript prompt differs from the request")
    for name in ("stdout", "stderr", "response_raw"):
        if not isinstance(transcript[name], str):
            raise ResearcherError(f"researcher transcript {name} must be text")
    response_raw = transcript["response_raw"]
    response_hash = hashlib.sha256(response_raw.encode("utf-8")).hexdigest()
    if transcript["response_sha256"] != response_hash:
        raise ResearcherError("researcher transcript response hash is invalid")
    try:
        status = ResearchStatus(transcript["status"])
    except (TypeError, ValueError) as error:
        raise ResearcherError("researcher transcript status is invalid") from error
    response = (
        None
        if transcript["response"] is None
        else StructuredResearchResponse.from_json(response_raw)
    )
    if response is not None:
        if json.loads(response_raw) != transcript["response"] or response.status is not status:
            raise ResearcherError("researcher transcript structured response is inconsistent")
    elif status is not ResearchStatus.FAILED:
        raise ResearcherError("successful researcher transcript is missing its response")
    failure_reason = transcript["failure_reason"]
    if failure_reason is not None:
        failure_reason = _required_text("failure_reason", failure_reason)
    if status is ResearchStatus.FAILED and failure_reason is None:
        raise ResearcherError("failed researcher transcript is missing its reason")
    returncode = transcript["returncode"]
    if returncode is not None and type(returncode) is not int:
        raise ResearcherError("researcher transcript returncode is invalid")
    if type(transcript["timed_out"]) is not bool:
        raise ResearcherError("researcher transcript timed_out must be boolean")
    try:
        provider = ResearcherProvider(transcript["provider"])
    except (TypeError, ValueError) as error:
        raise ResearcherError("researcher transcript provider is invalid") from error
    provider_configuration = transcript["provider_configuration"]
    if not isinstance(provider_configuration, dict):
        raise ResearcherError("researcher transcript provider_configuration must be an object")
    _strict_keys(
        provider_configuration,
        _PROVIDER_CONFIGURATION_KEYS,
        name="provider_configuration",
    )
    try:
        canonical_json_bytes(provider_configuration)
    except (TypeError, ValueError) as error:
        raise ResearcherError("researcher transcript provider_configuration is invalid") from error
    optional_text = {}
    for name in ("cli_version", "resolved_model", "inference_provider"):
        value = transcript[name]
        optional_text[name] = None if value is None else _required_text(name, value, maximum=1_000)
    usage_verified = transcript["usage_verified"]
    if type(usage_verified) is not bool:
        raise ResearcherError("researcher transcript usage_verified must be boolean")
    if (
        provider is not ResearcherProvider.COMMAND
        and response is not None
        and (optional_text["cli_version"] is None or not usage_verified)
    ):
        raise ResearcherError("native researcher transcript lacks trusted provider evidence")
    return ResearchAttempt(
        request_id=request.request_id,
        status=status,
        response=response,
        failure_reason=failure_reason,
        changed_paths=changed_paths,
        diff_sha256=diff_hash,
        prompt_sha256=prompt_hash,
        response_sha256=response_hash,
        transcript_path=transcript_path.resolve(),
        returncode=returncode,
        timed_out=transcript["timed_out"],
        provider=provider,
        cli_version=optional_text["cli_version"],
        resolved_model=optional_text["resolved_model"],
        inference_provider=optional_text["inference_provider"],
        usage_verified=usage_verified,
    )


def _process_group_options(*, windows: bool | None = None) -> dict[str, Any]:
    windows = os.name == "nt" if windows is None else windows
    if windows:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)}
    return {"start_new_session": True}


def _kill_process_group(
    process: subprocess.Popen[bytes],
    *,
    windows: bool | None = None,
) -> None:
    windows = os.name == "nt" if windows is None else windows
    if windows:
        try:
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
            process.wait(timeout=2.0)
            return
        except (OSError, subprocess.TimeoutExpired):
            with suppress(OSError):
                process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


def run_researcher_process(
    command: list[str],
    *,
    workspace: Path,
    environment: dict[str, str],
    stdin_bytes: bytes,
    timeout_seconds: float,
) -> tuple[int | None, bytes, bytes, bool]:
    try:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_process_group_options(),
        )
    except OSError as error:
        return None, b"", str(error).encode("utf-8", errors="replace"), False
    try:
        stdout, stderr = process.communicate(stdin_bytes, timeout=timeout_seconds)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr, True


class CommandResearcherAdapter:
    """Run a configured researcher command with no ledger capability or decision schema."""

    def __init__(self, config: ResearcherConfig) -> None:
        self.config = config

    def _environment(self, request: ResearchRequest) -> dict[str, str]:
        environment = {
            key: os.environ[key] for key in self.config.inherit_environment if key in os.environ
        }
        environment.update(dict(self.config.environment))
        environment.update(
            {
                "AUTODIDACT_ALLOWED_PATHS": json.dumps(list(request.allowed_paths)),
                "AUTODIDACT_REQUEST_ID": request.request_id,
                "AUTODIDACT_RESEARCHER_TOKEN_BUDGET": str(request.maximum_total_tokens),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return environment

    def _invoke(
        self,
        request: ResearchRequest,
        workspace: Path,
        prompt: str,
    ) -> InvocationResult:
        returncode, stdout, stderr, timed_out = run_researcher_process(
            list(self.config.command),
            workspace=workspace,
            environment=self._environment(request),
            stdin_bytes=prompt.encode("utf-8"),
            timeout_seconds=self.config.timeout_seconds,
        )
        return InvocationResult(
            returncode=returncode,
            response_bytes=stdout,
            stdout_bytes=stdout,
            stderr_bytes=stderr,
            timed_out=timed_out,
        )

    def run(
        self,
        request: ResearchRequest,
        *,
        workspace: Path,
        artifact_root: Path,
    ) -> ResearchAttempt:
        workspace = workspace.resolve()
        artifact_root = artifact_root.resolve()
        transcript_path = artifact_root / f"{request.request_id}.json"
        if transcript_path.exists():
            raise ResearcherError(
                "researcher transcript already exists; recover it instead of rerunning"
            )
        head = str(_git(workspace, "rev-parse", "HEAD")).strip()
        if head != request.parent_commit:
            raise ResearcherError("research workspace is not at the requested parent commit")
        if _changed_paths(workspace):
            raise ResearcherError("research workspace must be clean before invocation")

        prompt = request.prompt()
        invocation = self._invoke(
            request,
            workspace,
            prompt,
        )
        response_bytes = invocation.response_bytes
        if invocation.trusted_usage is not None:
            try:
                native_response = json.loads(response_bytes)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(native_response, dict):
                    native_response["usage"] = asdict(invocation.trusted_usage)
                    response_bytes = canonical_json_bytes(native_response)
        response_text = response_bytes.decode("utf-8", errors="replace")
        stdout = invocation.stdout_bytes.decode("utf-8", errors="replace")
        stderr = invocation.stderr_bytes.decode("utf-8", errors="replace")
        changed_paths = _changed_paths(workspace)
        ending_head = str(_git(workspace, "rev-parse", "HEAD")).strip()
        diff_bytes = _git(workspace, "diff", "--binary", "HEAD", text=False)
        assert isinstance(diff_bytes, bytes)
        diff = diff_bytes.decode("utf-8", errors="replace")

        failure_reason: str | None = None
        response: StructuredResearchResponse | None = None
        if invocation.timed_out:
            failure_reason = "researcher command timed out"
        elif invocation.returncode is None:
            failure_reason = "researcher command could not start"
        elif invocation.returncode != 0:
            failure_reason = f"researcher command exited with status {invocation.returncode}"
        elif (
            len(invocation.stdout_bytes) + len(invocation.stderr_bytes) + len(response_bytes)
            > self.config.max_output_bytes
        ):
            failure_reason = "researcher output exceeded the configured byte limit"
        elif ending_head != head:
            failure_reason = "researcher changed Git history; the controller owns candidate commits"
        elif len(diff_bytes) > self.config.max_diff_bytes:
            failure_reason = "researcher diff exceeded the configured byte limit"
        else:
            unexpected = tuple(path for path in changed_paths if path not in request.allowed_paths)
            if unexpected:
                failure_reason = f"researcher changed protected paths: {list(unexpected)}"
            else:
                try:
                    response = StructuredResearchResponse.from_json(response_text)
                except ResearcherError as error:
                    failure_reason = str(error)

        if response is not None:
            if response.usage.total_tokens > request.maximum_total_tokens:
                failure_reason = (
                    f"researcher usage {response.usage.total_tokens} tokens exceeded assigned "
                    f"token budget {request.maximum_total_tokens}; increase the per-proposal "
                    "and campaign researcher-token limits"
                )
                response = None
            elif response.status is ResearchStatus.PROPOSED and not changed_paths:
                failure_reason = "researcher proposed a patch without changing an allowed file"
                response = None
            elif response.status is not ResearchStatus.PROPOSED and changed_paths:
                failure_reason = "researcher changed files without returning a proposal"
                response = None

        status = ResearchStatus.FAILED if failure_reason is not None else response.status
        if response is not None and response.status is ResearchStatus.FAILED:
            failure_reason = response.failure_reason
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        response_hash = hashlib.sha256(response_bytes).hexdigest()
        diff_hash = hashlib.sha256(diff_bytes).hexdigest() if diff_bytes else None
        transcript = {
            "changed_paths": list(changed_paths),
            "diff": diff,
            "diff_sha256": diff_hash,
            "failure_reason": failure_reason,
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "provider": invocation.provider.value,
            "provider_configuration": {
                "backend_provider": self.config.backend_provider,
                "max_budget_usd": self.config.max_budget_usd,
                "max_turns": self.config.max_turns,
                "model": self.config.model,
                "profile": self.config.profile,
                "reasoning_effort": self.config.reasoning_effort,
            },
            "request_id": request.request_id,
            "response": None if response is None else json.loads(response_text),
            "response_raw": response_text,
            "response_sha256": response_hash,
            "inference_provider": invocation.inference_provider,
            "resolved_model": invocation.resolved_model,
            "returncode": invocation.returncode,
            "schema_version": RESEARCHER_SCHEMA_VERSION,
            "status": status.value,
            "stderr": stderr,
            "stdout": stdout,
            "timed_out": invocation.timed_out,
            "cli_version": invocation.cli_version,
            "usage_verified": invocation.trusted_usage is not None,
        }
        _atomic_write(
            transcript_path,
            json.dumps(transcript, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        return ResearchAttempt(
            request_id=request.request_id,
            status=status,
            response=response,
            failure_reason=failure_reason,
            changed_paths=changed_paths,
            diff_sha256=diff_hash,
            prompt_sha256=prompt_hash,
            response_sha256=response_hash,
            transcript_path=transcript_path,
            returncode=invocation.returncode,
            timed_out=invocation.timed_out,
            provider=invocation.provider,
            cli_version=invocation.cli_version,
            resolved_model=invocation.resolved_model,
            inference_provider=invocation.inference_provider,
            usage_verified=invocation.trusted_usage is not None,
        )


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _load_request(path: Path) -> ResearchRequest:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearcherError(f"cannot read research request: {error}") from error
    if not isinstance(value, dict):
        raise ResearcherError("research request must be an object")
    keys = frozenset(value)
    required = _REQUEST_KEYS - {"maximum_total_tokens"}
    missing = sorted(required - keys)
    extra = sorted(keys - _REQUEST_KEYS)
    if missing or extra:
        raise ResearcherError(f"request keys differ; missing={missing}, extra={extra}")
    program_path = Path(_required_text("program_path", value.pop("program_path"))).expanduser()
    try:
        program_text = program_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ResearcherError(f"cannot read research program: {error}") from error
    previous = value["previous_results"]
    allowed = value["allowed_paths"]
    if not isinstance(previous, list) or any(not isinstance(item, dict) for item in previous):
        raise ResearcherError("previous_results must be an array of objects")
    if not isinstance(allowed, list) or any(not isinstance(item, str) for item in allowed):
        raise ResearcherError("allowed_paths must be an array of strings")
    return ResearchRequest(
        request_id=value["request_id"],
        parent_commit=value["parent_commit"],
        proposal_number=value["proposal_number"],
        program_text=program_text,
        previous_results=tuple(previous),
        maximum_total_tokens=value.get(
            "maximum_total_tokens",
            DEFAULT_RESEARCHER_TOKEN_ALLOWANCE,
        ),
        allowed_paths=tuple(allowed),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a configured research proposal agent.")
    parser.add_argument("--config", type=_path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--workspace", type=_path, required=True)
    parser.add_argument("--artifact-root", type=_path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--request", type=_path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = _load_request(args.request)
        from autodidact.researcher_providers import build_researcher_adapter

        adapter = build_researcher_adapter(ResearcherConfig.from_path(args.config))
        attempt = adapter.run(
            request,
            workspace=args.workspace,
            artifact_root=args.artifact_root,
        )
        payload = {
            "changed_paths": list(attempt.changed_paths),
            "diff_sha256": attempt.diff_sha256,
            "failure_reason": attempt.failure_reason,
            "inference_provider": attempt.inference_provider,
            "provider": attempt.provider.value,
            "request_id": attempt.request_id,
            "resolved_model": attempt.resolved_model,
            "status": attempt.status.value,
            "transcript_path": str(attempt.transcript_path),
            "usage": asdict(attempt.usage),
            "usage_verified": attempt.usage_verified,
            "cli_version": attempt.cli_version,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0 if attempt.status is not ResearchStatus.FAILED else 2
    except (OSError, ResearcherError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
