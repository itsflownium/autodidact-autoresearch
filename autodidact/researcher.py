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
from pathlib import Path
from typing import Any, Protocol

from autodidact.data.integrity import (
    ProtectedPathError,
    assert_research_paths_allowed,
    canonical_json_bytes,
)
from autodidact.records import PatchProposal

RESEARCHER_SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = Path("artifacts/control/researcher.json")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/researcher")
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
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
        "command",
        "timeout_seconds",
        "max_output_bytes",
        "max_diff_bytes",
        "inherit_environment",
        "environment",
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
    }
)


class ResearcherError(RuntimeError):
    """Raised when a researcher invocation or contract is invalid."""


class ResearchStatus(StrEnum):
    PROPOSED = "proposed"
    NO_CHANGE = "no_change"
    FAILED = "failed"


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
    allowed_paths: tuple[str, ...] = ("train.py",)

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.request_id):
            raise ResearcherError("request_id must be a portable structured ID")
        if not _COMMIT_PATTERN.fullmatch(self.parent_commit):
            raise ResearcherError("parent_commit must be a full lowercase Git commit")
        if type(self.proposal_number) is not int or self.proposal_number <= 0:
            raise ResearcherError("proposal_number must be positive")
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
        try:
            assert_research_paths_allowed(list(self.allowed_paths))
        except (ProtectedPathError, ValueError) as error:
            raise ResearcherError(str(error)) from error

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
            "previous_results": list(self.previous_results),
            "program_md": self.program_text,
            "proposal_number": self.proposal_number,
            "request_id": self.request_id,
            "schema_version": RESEARCHER_SCHEMA_VERSION,
        }
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class ResearcherConfig:
    command: tuple[str, ...]
    timeout_seconds: float = 900.0
    max_output_bytes: int = 1_000_000
    max_diff_bytes: int = 2_000_000
    inherit_environment: tuple[str, ...] = ("HOME", "LANG", "PATH", "TMPDIR")
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.command or any(not isinstance(part, str) or not part for part in self.command):
            raise ResearcherError("command must be a nonempty argument sequence")
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
        environment = value.get("environment", {})
        if not isinstance(environment, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in environment.items()
        ):
            raise ResearcherError("environment must map strings to strings")
        command = value.get("command")
        inherited = value.get("inherit_environment", ["HOME", "LANG", "PATH", "TMPDIR"])
        if not isinstance(command, list) or not isinstance(inherited, list):
            raise ResearcherError("command and inherit_environment must be arrays")
        return cls(
            command=tuple(command),
            timeout_seconds=value.get("timeout_seconds", 900.0),
            max_output_bytes=value.get("max_output_bytes", 1_000_000),
            max_diff_bytes=value.get("max_diff_bytes", 2_000_000),
            inherit_environment=tuple(inherited),
            environment=tuple(sorted(environment.items())),
        )


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


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


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
                "PYTHONUNBUFFERED": "1",
            }
        )
        return environment

    def _invoke(
        self,
        request: ResearchRequest,
        workspace: Path,
        prompt: str,
    ) -> tuple[int | None, bytes, bytes, bool]:
        try:
            process = subprocess.Popen(
                list(self.config.command),
                cwd=workspace,
                env=self._environment(request),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            return None, b"", str(error).encode("utf-8", errors="replace"), False
        try:
            stdout, stderr = process.communicate(
                prompt.encode("utf-8"),
                timeout=self.config.timeout_seconds,
            )
            return process.returncode, stdout, stderr, False
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
            stdout, stderr = process.communicate()
            return process.returncode, stdout, stderr, True

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
        returncode, stdout_bytes, stderr_bytes, timed_out = self._invoke(
            request,
            workspace,
            prompt,
        )
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        changed_paths = _changed_paths(workspace)
        ending_head = str(_git(workspace, "rev-parse", "HEAD")).strip()
        diff_bytes = _git(workspace, "diff", "--binary", "HEAD", text=False)
        assert isinstance(diff_bytes, bytes)
        diff = diff_bytes.decode("utf-8", errors="replace")

        failure_reason: str | None = None
        response: StructuredResearchResponse | None = None
        if timed_out:
            failure_reason = "researcher command timed out"
        elif returncode != 0:
            failure_reason = f"researcher command exited with status {returncode}"
        elif len(stdout_bytes) + len(stderr_bytes) > self.config.max_output_bytes:
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
                    response = StructuredResearchResponse.from_json(stdout)
                except ResearcherError as error:
                    failure_reason = str(error)

        if response is not None:
            if response.status is ResearchStatus.PROPOSED and not changed_paths:
                failure_reason = "researcher proposed a patch without changing an allowed file"
                response = None
            elif response.status is not ResearchStatus.PROPOSED and changed_paths:
                failure_reason = "researcher changed files without returning a proposal"
                response = None

        status = ResearchStatus.FAILED if failure_reason is not None else response.status
        if response is not None and response.status is ResearchStatus.FAILED:
            failure_reason = response.failure_reason
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        response_hash = hashlib.sha256(stdout_bytes).hexdigest()
        diff_hash = hashlib.sha256(diff_bytes).hexdigest() if diff_bytes else None
        transcript = {
            "changed_paths": list(changed_paths),
            "diff": diff,
            "diff_sha256": diff_hash,
            "failure_reason": failure_reason,
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "request_id": request.request_id,
            "response": None if response is None else json.loads(stdout),
            "response_sha256": response_hash,
            "returncode": returncode,
            "schema_version": RESEARCHER_SCHEMA_VERSION,
            "status": status.value,
            "stderr": stderr,
            "stdout": stdout,
            "timed_out": timed_out,
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
            returncode=returncode,
            timed_out=timed_out,
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
    _strict_keys(value, _REQUEST_KEYS, name="request")
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
        adapter = CommandResearcherAdapter(ResearcherConfig.from_path(args.config))
        attempt = adapter.run(
            request,
            workspace=args.workspace,
            artifact_root=args.artifact_root,
        )
        payload = {
            "changed_paths": list(attempt.changed_paths),
            "diff_sha256": attempt.diff_sha256,
            "failure_reason": attempt.failure_reason,
            "request_id": attempt.request_id,
            "status": attempt.status.value,
            "transcript_path": str(attempt.transcript_path),
            "usage": asdict(attempt.usage),
        }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0 if attempt.status is not ResearchStatus.FAILED else 2
    except (OSError, ResearcherError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
