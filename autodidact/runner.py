"""Protected, paired parent-versus-candidate experiment execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from autodidact.checkpoints import file_sha256
from autodidact.data.config import default_output_root
from autodidact.data.integrity import (
    ProtectedPathError,
    assert_research_paths_allowed,
    canonical_json_bytes,
    policy_sha256,
    verify_dataset,
)
from autodidact.ledger import (
    ExperimentLedger,
    LedgerError,
    WriterRole,
    resource_constraint_failures,
)
from autodidact.records import (
    DEFAULT_PARAMETER_CAP,
    ArtifactManifest,
    ArtifactRef,
    ArtifactRetention,
    CandidateRecord,
    ComputeRecord,
    ExperimentStage,
    PairedResult,
    PatchProposal,
    ResourceLimits,
    RunArm,
    RunResult,
    RunStatus,
    TrialSpec,
    build_paired_result,
    record_to_envelope,
)
from autodidact.target import TargetConfig, TargetError
from autodidact.target_plugins import (
    TargetPluginError,
    TargetPluginSpec,
    resolve_repository_path,
)

RUNNER_SCHEMA_VERSION = 3
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments")
DEFAULT_LEDGER_PATH = Path("artifacts/ledger/experiments.sqlite3")
MAX_SEED = 2**32 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class RunnerError(RuntimeError):
    """Raised when an experiment cannot be executed or trusted."""


class RunnerCancelled(RunnerError):
    """Raised after an interrupted arm has been recorded as cancelled."""


@dataclass(frozen=True, slots=True)
class StageDefaults:
    training_mode: str
    token_budget: int
    eval_tokens: int | None
    evaluator_split: str


STAGE_DEFAULTS = {
    ExperimentStage.CHEAP: StageDefaults("cheap", 2_000_000, 250_000, "dev"),
    ExperimentStage.INTERMEDIATE: StageDefaults("intermediate", 6_000_000, 1_000_000, "dev"),
    ExperimentStage.FULL: StageDefaults("full", 20_000_000, None, "dev"),
    ExperimentStage.PROMOTION: StageDefaults("full", 20_000_000, None, "promotion"),
    ExperimentStage.SEALED_FINAL: StageDefaults("full", 20_000_000, None, "sealed_final"),
}


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    returncode: int | None
    wall_seconds: float
    peak_process_rss_bytes: int | None
    timed_out: bool = False
    cancelled: bool = False


ProcessRunner = Callable[..., ProcessOutcome]


@dataclass(frozen=True, slots=True)
class CandidateValidation:
    parent_commit: str
    candidate_commit: str
    changed_paths: tuple[str, ...]
    diff_sha256: str


@dataclass(frozen=True, slots=True)
class ExperimentRequest:
    repository_root: Path
    ledger_path: Path
    data_root: Path
    output_root: Path
    proposal_id: str
    candidate_commit: str
    stage: ExperimentStage
    seeds: tuple[int, ...]
    assignment_seed: int
    token_budget: int
    eval_tokens: int | None
    batch_size: int
    eval_batch_size: int
    timeout_seconds: int
    device: str
    limits: ResourceLimits
    estimated_accelerator_hour_usd: float | None = None
    target_config_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise RunnerError("seeds must be a nonempty unique predetermined sequence")
        if any(type(seed) is not int or seed < 0 or seed > MAX_SEED for seed in self.seeds):
            raise RunnerError(f"seeds must be integers between 0 and {MAX_SEED}")
        if self.assignment_seed < 0 or self.assignment_seed > MAX_SEED:
            raise RunnerError(f"assignment_seed must be between 0 and {MAX_SEED}")
        if self.token_budget <= 0:
            raise RunnerError("token_budget must be positive")
        if self.eval_tokens is not None and self.eval_tokens <= 0:
            raise RunnerError("eval_tokens must be positive or None")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise RunnerError("batch sizes must be positive")
        if self.timeout_seconds <= 0:
            raise RunnerError("timeout_seconds must be positive")
        if self.estimated_accelerator_hour_usd is not None and (
            not math.isfinite(self.estimated_accelerator_hour_usd)
            or self.estimated_accelerator_hour_usd < 0.0
        ):
            raise RunnerError("estimated accelerator hourly cost must be finite and nonnegative")


def _stable_id(prefix: str, *parts: object) -> str:
    payload = canonical_json_bytes([str(part) for part in parts])
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _git(
    repository_root: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        text=text,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr if text else completed.stderr.decode("utf-8", errors="replace")
        raise RunnerError(f"Git command failed: {stderr.strip() or arguments[0]}")
    return completed.stdout


def validate_candidate_patch(
    repository_root: Path,
    *,
    parent_commit: str,
    candidate_commit: str,
    allowed_paths: Sequence[str] = ("train.py",),
    trainer_path: str = "train.py",
) -> CandidateValidation:
    repository_root = repository_root.resolve()
    for name, value in (("parent", parent_commit), ("candidate", candidate_commit)):
        if not _GIT_COMMIT_PATTERN.fullmatch(value):
            raise RunnerError(f"{name} must be a full lowercase Git commit")
        _git(repository_root, "cat-file", "-e", f"{value}^{{commit}}")
    ancestry = (
        str(_git(repository_root, "rev-list", "--parents", "-n", "1", candidate_commit))
        .strip()
        .split()
    )
    if len(ancestry) != 2 or ancestry[1] != parent_commit:
        raise RunnerError("candidate must be one atomic, non-merge commit on its declared parent")
    changed_raw = _git(
        repository_root,
        "diff",
        "--name-only",
        "-z",
        parent_commit,
        candidate_commit,
        text=False,
    )
    assert isinstance(changed_raw, bytes)
    changed_paths = tuple(item.decode("utf-8") for item in changed_raw.split(b"\0") if item)
    if not changed_paths:
        raise RunnerError("candidate commit does not contain a patch")
    allowed = frozenset(allowed_paths)
    if not allowed or trainer_path not in allowed:
        raise RunnerError("target editable paths must include its trainer")
    if allowed == {"train.py"}:
        try:
            assert_research_paths_allowed(list(changed_paths))
        except (ProtectedPathError, ValueError) as error:
            raise RunnerError(str(error)) from error
    else:
        rejected = sorted(set(changed_paths) - allowed)
        if rejected:
            raise RunnerError(f"research change touches protected path(s): {', '.join(rejected)}")
    diff_check = subprocess.run(
        ["git", "diff", "--check", parent_commit, candidate_commit],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if diff_check.returncode != 0:
        raise RunnerError("candidate patch fails git diff --check")
    patch = _git(
        repository_root,
        "diff",
        "--binary",
        "--full-index",
        parent_commit,
        candidate_commit,
        "--",
        *sorted(allowed),
        text=False,
    )
    assert isinstance(patch, bytes)
    candidate_kind = str(
        _git(repository_root, "cat-file", "-t", f"{candidate_commit}:{trainer_path}")
    ).strip()
    if candidate_kind != "blob":
        raise RunnerError(f"candidate {trainer_path} must be a regular Git blob")
    return CandidateValidation(
        parent_commit=parent_commit,
        candidate_commit=candidate_commit,
        changed_paths=changed_paths,
        diff_sha256=hashlib.sha256(patch).hexdigest(),
    )


def _link_tree(source: Path, destination: Path) -> None:
    for directory, directories, files in os.walk(source):
        relative = Path(directory).relative_to(source)
        target_directory = destination / relative
        target_directory.mkdir(parents=True, exist_ok=True)
        for name in directories:
            (target_directory / name).mkdir(exist_ok=True)
        for name in files:
            source_file = Path(directory) / name
            if source_file.is_symlink():
                raise RunnerError("public dataset view refuses symbolic links")
            try:
                os.link(source_file, target_directory / name)
            except OSError as error:
                raise RunnerError(
                    "public data view requires output and dataset roots on one filesystem"
                ) from error


def _seal_tree(root: Path) -> None:
    for directory, directories, files in os.walk(root, topdown=False):
        for name in files:
            (Path(directory) / name).chmod(0o444)
        for name in directories:
            (Path(directory) / name).chmod(0o555)
        Path(directory).chmod(0o555)
    root.chmod(0o555)


def prepare_public_data_view(data_root: Path, view_root: Path) -> Path:
    data_root = data_root.resolve()
    view_root = view_root.resolve()
    if view_root.is_relative_to(data_root):
        raise RunnerError("public data view must be outside the immutable dataset root")
    verify_dataset(data_root, scope="public")
    if view_root.exists():
        if (view_root / "protected").exists():
            raise RunnerError("public data view unexpectedly exposes protected data")
        verify_dataset(view_root, scope="public")
        _seal_tree(view_root)
        return view_root
    view_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".public-data-", dir=view_root.parent))
    try:
        _link_tree(data_root / "public", staging / "public")
        verify_dataset(staging, scope="public")
        _seal_tree(staging)
        os.replace(staging, view_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return view_root


@contextmanager
def isolated_worktrees(
    repository_root: Path,
    worktree_parent: Path,
    *,
    parent_commit: str,
    candidate_commit: str,
    trainer_path: str = "train.py",
) -> Iterator[dict[RunArm, Path]]:
    worktree_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="paired-", dir=worktree_parent))
    paths = {RunArm.PARENT: root / "parent", RunArm.CANDIDATE: root / "candidate"}
    added: list[Path] = []
    try:
        for arm, commit in (
            (RunArm.PARENT, parent_commit),
            (RunArm.CANDIDATE, candidate_commit),
        ):
            _git(repository_root, "worktree", "add", "--detach", str(paths[arm]), commit)
            added.append(paths[arm])
            trainer = paths[arm] / trainer_path
            if not trainer.is_file() or trainer.is_symlink():
                raise RunnerError(f"{arm.value} worktree has an invalid {trainer_path}")
        yield paths
    finally:
        for path in reversed(added):
            with suppress(RunnerError):
                _git(repository_root, "worktree", "remove", "--force", str(path))
        shutil.rmtree(root, ignore_errors=True)
        with suppress(RunnerError):
            _git(repository_root, "worktree", "prune")


def assign_execution_orders(
    seeds: Sequence[int],
    *,
    assignment_seed: int,
) -> tuple[tuple[RunArm, RunArm], ...]:
    if not seeds:
        raise RunnerError("at least one predetermined seed is required")
    randomizer = random.Random(assignment_seed)
    first_parent = bool(randomizer.getrandbits(1))
    orders = []
    for index in range(len(seeds)):
        parent_first = first_parent if index % 2 == 0 else not first_parent
        orders.append(
            (RunArm.PARENT, RunArm.CANDIDATE) if parent_first else (RunArm.CANDIDATE, RunArm.PARENT)
        )
    randomizer.shuffle(orders)
    return tuple(orders)


def _sample_process_rss(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(("VmHWM:", "VmRSS:")):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1_024
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return int(value) * 1_024 if value.isdigit() else None


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> ProcessOutcome:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak_rss: int | None = None
    timed_out = False
    cancelled = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        except OSError as error:
            stderr.write(f"process launch failed: {type(error).__name__}\n".encode())
            return ProcessOutcome(
                returncode=126,
                wall_seconds=time.perf_counter() - started,
                peak_process_rss_bytes=None,
            )
        try:
            while process.poll() is None:
                sample = _sample_process_rss(process.pid)
                if sample is not None:
                    peak_rss = sample if peak_rss is None else max(peak_rss, sample)
                if time.perf_counter() - started >= timeout_seconds:
                    timed_out = True
                    _terminate_process(process)
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            cancelled = True
            _terminate_process(process)
        returncode = process.poll()
    return ProcessOutcome(
        returncode=returncode,
        wall_seconds=time.perf_counter() - started,
        peak_process_rss_bytes=peak_rss,
        timed_out=timed_out,
        cancelled=cancelled,
    )


def sanitized_environment(seed: int, controller_root: Path) -> dict[str, str]:
    allowed = {
        "CUDA_VISIBLE_DEVICES",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "TEMP",
        "TMP",
        "TMPDIR",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "NO_PROXY": "*",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": str(seed),
            "PYTHONPATH": str(controller_root.resolve()),
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def _resolve_device_name(requested: str) -> str:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RunnerError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RunnerError("MPS was requested but is unavailable")
    return str(device)


def _environment_sha256(device: str) -> str:
    return _sha256_payload(
        {
            "device": device,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
    )


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RunnerError("required JSONL artifact is missing")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunnerError(f"invalid JSONL at line {line_number}") from error
        if not isinstance(value, dict):
            raise RunnerError("JSONL records must be objects")
        records.append(value)
    return records


def _last_event(records: Sequence[dict[str, Any]], event: str) -> dict[str, Any]:
    matching = [record for record in records if record.get("event") == event]
    if not matching:
        raise RunnerError(f"metrics are missing the {event} event")
    return matching[-1]


def _last_json_object(path: Path) -> dict[str, Any]:
    records = _read_json_lines(path)
    if not records:
        raise RunnerError("protected process did not emit a JSON result")
    return records[-1]


def _log_text(*paths: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in paths if path.is_file()
    ).lower()


def classify_failure(outcome: ProcessOutcome, output: str, *, phase: str) -> tuple[RunStatus, str]:
    if outcome.cancelled:
        return RunStatus.CANCELLED, f"{phase} was cancelled"
    if outcome.timed_out:
        return RunStatus.TIMEOUT, f"{phase} exceeded its timeout"
    if "out of memory" in output or "cannot allocate memory" in output:
        return RunStatus.OOM, f"{phase} exhausted memory"
    if "non-finite" in output or re.search(r"\bnan\b", output):
        return RunStatus.NON_FINITE, f"{phase} produced a non-finite value"
    return RunStatus.CRASHED, f"{phase} exited with code {outcome.returncode}"


def _optional_finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _valid_sha256(value: Any) -> str | None:
    return value if isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) else None


def _record_or_none(
    ledger: ExperimentLedger,
    requested_id: str,
    expected_type: type[Any],
) -> Any | None:
    try:
        event = ledger.get(requested_id)
    except LedgerError as error:
        if str(error).startswith("record does not exist:"):
            return None
        raise
    if not isinstance(event.record, expected_type):
        raise RunnerError(f"record ID {requested_id} has the wrong type")
    return event.record


def _worktree_is_clean(path: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _file_matches(path: Path, expected_sha256: str) -> bool:
    try:
        return path.is_file() and file_sha256(path) == expected_sha256
    except OSError:
        return False


class PairedExperimentRunner:
    def __init__(
        self,
        request: ExperimentRequest,
        *,
        process_runner: ProcessRunner = run_process,
    ) -> None:
        self.request = request
        self.process_runner = process_runner
        self.repository_root = request.repository_root.resolve()
        self.output_root = request.output_root.resolve()
        self.data_root = request.data_root.resolve()
        self.evaluator_path = Path(__file__).resolve().with_name("evaluator.py")
        self.runner_path = Path(__file__).resolve()
        self.device = _resolve_device_name(request.device)
        self.target_config: TargetConfig | None = None
        self.plugin: TargetPluginSpec | None = None
        if request.target_config_path is not None:
            try:
                self.target_config = TargetConfig.from_path(request.target_config_path)
                self.plugin = self.target_config.load_plugin(self.repository_root)
            except (TargetError, TargetPluginError) as error:
                raise RunnerError(str(error)) from error
        self.trainer_path = "train.py" if self.plugin is None else self.plugin.trainer_path
        self.editable_paths = ("train.py",) if self.plugin is None else self.plugin.editable_paths
        self.policy_contract_sha256 = (
            policy_sha256()
            if self.target_config is None
            else _sha256_payload(
                {
                    "plugin": None if self.plugin is None else self.plugin.to_mapping(),
                    "target": self.target_config.to_mapping(),
                }
            )
        )
        self.public_data_root = (
            None
            if self.target_config is None
            else self.target_config.resolved_public_data_root(self.repository_root)
        )
        if self.target_config is not None:
            if self.data_root != self.target_config.resolved_data_root(self.repository_root):
                raise RunnerError("experiment data_root differs from the target contract")
            if request.limits.max_parameter_count != self.target_config.max_parameter_count:
                raise RunnerError("experiment parameter cap differs from the target contract")

    def _worktree_root(self, trainer: Path) -> Path:
        root = trainer
        for _ in PurePosixPath(self.trainer_path).parts:
            root = root.parent
        return root

    def _validate_plugin_evaluators(self, worktrees: dict[RunArm, Path]) -> None:
        if self.plugin is None:
            return
        paths = [
            resolve_repository_path(worktrees[arm], self.plugin.evaluator_path)
            for arm in (RunArm.PARENT, RunArm.CANDIDATE)
        ]
        if any(not path.is_file() or path.is_symlink() for path in paths):
            raise RunnerError("external target protected evaluator is missing or invalid")
        if len({file_sha256(path) for path in paths}) != 1:
            raise RunnerError("external target protected evaluator changed in the candidate")

    def _run_process(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        seed: int,
        stdout_path: Path,
        stderr_path: Path,
    ) -> ProcessOutcome:
        return self.process_runner(
            command,
            cwd=cwd,
            environment=sanitized_environment(seed, self.repository_root),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout_seconds=self.request.timeout_seconds,
        )

    def _inspect_trainer(
        self,
        trainer: Path,
        *,
        arm: RunArm,
        control_root: Path,
    ) -> dict[str, Any]:
        stdout_path = control_root / f"inspect-{arm.value}.jsonl"
        stderr_path = control_root / f"inspect-{arm.value}.stderr.log"
        if self.plugin is None:
            command = [
                sys.executable,
                str(self.evaluator_path),
                "inspect",
                "--trainer",
                str(trainer),
                "--parameter-cap",
                str(self.request.limits.max_parameter_count),
            ]
        else:
            command = self.plugin.render_command(
                "inspect",
                self._command_values(trainer=trainer),
            )
        outcome = self._run_process(
            command,
            cwd=self._worktree_root(trainer),
            seed=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        if outcome.returncode != 0 or outcome.timed_out or outcome.cancelled:
            raise RunnerError(f"protected {arm.value} model inspection failed")
        payload = _last_json_object(stdout_path)
        expected_event = "protected_inspection" if self.plugin is None else "target_inspection"
        if payload.get("event") != expected_event or payload.get("trainer_sha256") != file_sha256(
            trainer
        ):
            raise RunnerError(f"protected {arm.value} inspection contract mismatch")
        count = payload.get("parameter_count")
        if type(count) is not int or count <= 0 or count > self.request.limits.max_parameter_count:
            raise RunnerError(f"protected {arm.value} parameter count is invalid")
        return payload

    def _command_values(self, *, trainer: Path, **values: object) -> dict[str, object]:
        if self.plugin is None:
            raise RunnerError("built-in target does not use plugin command templates")
        worktree = self._worktree_root(trainer)
        evaluator = resolve_repository_path(worktree, self.plugin.evaluator_path)
        return {
            "evaluator": evaluator,
            "parameter_cap": self.request.limits.max_parameter_count,
            "python": sys.executable,
            "repository_root": worktree,
            "trainer": trainer,
            **values,
        }

    def _training_command(
        self,
        trainer: Path,
        trial: TrialSpec,
        paths: dict[str, Path],
        public_data_root: Path,
    ) -> list[str]:
        defaults = STAGE_DEFAULTS[trial.stage]
        if self.plugin is not None:
            return self.plugin.render_command(
                "train",
                self._command_values(
                    trainer=trainer,
                    batch_size=trial.batch_size,
                    checkpoint=paths["checkpoint"],
                    device=trial.device,
                    eval_batch_size=trial.eval_batch_size,
                    metrics=paths["metrics"],
                    public_data_root=public_data_root,
                    seed=trial.seed,
                    stage=trial.stage.value,
                    token_budget=trial.token_budget,
                ),
            )
        log_interval = min(trial.token_budget, max(1, trial.token_budget // 4))
        return [
            sys.executable,
            str(trainer),
            "train",
            "--mode",
            defaults.training_mode,
            "--data-root",
            str(public_data_root),
            "--device",
            trial.device,
            "--seed",
            str(trial.seed),
            "--token-budget",
            str(trial.token_budget),
            "--batch-size",
            str(trial.batch_size),
            "--eval-batch-size",
            str(trial.eval_batch_size),
            "--log-every-tokens",
            str(log_interval),
            "--checkpoint-every-tokens",
            str(trial.token_budget),
            "--checkpoint-out",
            str(paths["checkpoint"]),
            "--metrics-file",
            str(paths["metrics"]),
            "--skip-eval",
            "--no-generate",
            "--deterministic",
        ]

    def _evaluation_command(
        self,
        trainer: Path,
        trial: TrialSpec,
        checkpoint_path: Path,
    ) -> list[str]:
        if self.plugin is not None:
            return self.plugin.render_command(
                "evaluate",
                self._command_values(
                    trainer=trainer,
                    batch_size=trial.eval_batch_size,
                    checkpoint=checkpoint_path,
                    data_root=self.data_root,
                    device=trial.device,
                    eval_tokens=0 if trial.eval_tokens is None else trial.eval_tokens,
                    seed=trial.seed,
                    split=STAGE_DEFAULTS[trial.stage].evaluator_split,
                    stage=trial.stage.value,
                ),
            )
        command = [
            sys.executable,
            str(self.evaluator_path),
            "evaluate",
            "--trainer",
            str(trainer),
            "--checkpoint",
            str(checkpoint_path),
            "--data-root",
            str(self.data_root),
            "--split",
            STAGE_DEFAULTS[trial.stage].evaluator_split,
            "--batch-size",
            str(trial.eval_batch_size),
            "--device",
            trial.device,
            "--parameter-cap",
            str(trial.limits.max_parameter_count),
        ]
        if trial.eval_tokens is not None:
            command.extend(["--maximum-tokens", str(trial.eval_tokens)])
        return command

    def _arm_paths(self, candidate_root: Path, trial: TrialSpec, arm: RunArm) -> dict[str, Path]:
        root = candidate_root / trial.stage.value / f"seed-{trial.seed}" / arm.value
        return {
            "root": root,
            "checkpoint": root / "checkpoint.pt",
            "metrics": root / "metrics.jsonl",
            "stdout": root / "training.stdout.log",
            "stderr": root / "training.stderr.log",
            "evaluation": root / "evaluation.jsonl",
            "evaluation_stderr": root / "evaluation.stderr.log",
        }

    def _artifact_manifest(
        self,
        run: RunResult,
        paths: dict[str, Path],
    ) -> ArtifactManifest:
        kinds = (
            ("checkpoint", "checkpoint", ArtifactRetention.EPHEMERAL),
            ("metrics", "metrics", ArtifactRetention.COMPACT),
            ("stdout", "training_stdout", ArtifactRetention.COMPACT),
            ("stderr", "training_stderr", ArtifactRetention.COMPACT),
            ("evaluation", "protected_evaluation", ArtifactRetention.COMPACT),
            ("evaluation_stderr", "evaluation_stderr", ArtifactRetention.COMPACT),
        )
        artifacts = []
        for key, kind, retention in kinds:
            path = paths[key]
            if not path.is_file():
                continue
            relative = path.resolve().relative_to(self.output_root).as_posix()
            artifacts.append(
                ArtifactRef(
                    artifact_id=_stable_id("artifact", run.run_id, kind),
                    kind=kind,
                    relative_path=relative,
                    sha256=file_sha256(path),
                    size_bytes=path.stat().st_size,
                    retention=retention,
                )
            )
        if not artifacts:
            raise RunnerError("run produced no artifacts to manifest")
        return ArtifactManifest(
            manifest_id=_stable_id("manifest", run.run_id),
            run_id=run.run_id,
            artifacts=tuple(artifacts),
        )

    def _verify_manifest(self, manifest: ArtifactManifest) -> None:
        for artifact in manifest.artifacts:
            relative = PurePosixPath(artifact.relative_path)
            path = self.output_root.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(self.output_root) or not path.is_file():
                raise RunnerError("retained run artifact is missing or outside the output root")
            if path.stat().st_size != artifact.size_bytes or file_sha256(path) != artifact.sha256:
                raise RunnerError("retained run artifact failed hash verification")

    def _failed_run(
        self,
        trial: TrialSpec,
        arm: RunArm,
        status: RunStatus,
        reason: str,
        training: ProcessOutcome,
        records: Sequence[dict[str, Any]],
        *,
        parameter_count: int,
        evaluation: ProcessOutcome | None = None,
    ) -> RunResult:
        summary_event = "summary" if self.plugin is None else "target_training_summary"
        summary = next(
            (record for record in reversed(records) if record.get("event") == summary_event),
            {},
        )
        train_event = "train" if self.plugin is None else "target_training_progress"
        train_events = [record for record in records if record.get("event") == train_event]
        latest_train = train_events[-1] if train_events else {}
        tokens_seen = _optional_nonnegative_int(
            summary.get(
                "tokens_seen",
                summary.get("units_seen", latest_train.get("tokens_seen", 0)),
            )
        )
        tokens_seen = min(tokens_seen or 0, trial.token_budget)
        data_order = _valid_sha256(
            summary.get("data_order_sha256", latest_train.get("data_order_sha256"))
        )
        mean_loss = _optional_finite(summary.get("mean_train_loss", latest_train.get("loss")))
        if mean_loss is not None and mean_loss < 0.0:
            mean_loss = None
        peak_values = [
            value
            for value in (
                training.peak_process_rss_bytes,
                None if evaluation is None else evaluation.peak_process_rss_bytes,
                _optional_nonnegative_int(summary.get("peak_process_rss_bytes")),
            )
            if value is not None
        ]
        evaluation_wall = 0.0 if evaluation is None else evaluation.wall_seconds
        return RunResult(
            run_id=_stable_id("run", trial.trial_id, arm.value),
            trial_id=trial.trial_id,
            arm=arm,
            status=status,
            seed=trial.seed,
            target_tokens=trial.token_budget,
            tokens_seen=tokens_seen,
            evaluation_tokens=0,
            parameter_count=parameter_count,
            validation_bpb=None,
            mean_train_loss=mean_loss,
            training_tokens_per_second=None,
            evaluation_tokens_per_second=None,
            peak_process_rss_bytes=max(peak_values) if peak_values else None,
            peak_device_allocated_bytes=_optional_nonnegative_int(
                summary.get("peak_device_allocated_bytes")
            ),
            peak_device_reserved_bytes=_optional_nonnegative_int(
                summary.get("peak_device_reserved_bytes")
            ),
            training_seconds=training.wall_seconds,
            evaluation_seconds=None,
            wall_seconds=training.wall_seconds + evaluation_wall,
            data_order_sha256=data_order,
            failure_reason=reason,
        )

    def _successful_run(
        self,
        trial: TrialSpec,
        arm: RunArm,
        trainer: Path,
        paths: dict[str, Path],
        training: ProcessOutcome,
        evaluation: ProcessOutcome,
        *,
        parameter_count: int,
    ) -> RunResult:
        records = _read_json_lines(paths["metrics"])
        config_event = "config" if self.plugin is None else "target_training_config"
        summary_event = "summary" if self.plugin is None else "target_training_summary"
        config = _last_event(records, config_event)
        summary = _last_event(records, summary_event)
        protected = _last_json_object(paths["evaluation"])
        if self.plugin is not None:
            return self._successful_plugin_run(
                trial,
                arm,
                trainer,
                paths,
                training,
                evaluation,
                config=config,
                summary=summary,
                protected=protected,
                parameter_count=parameter_count,
            )
        expected = {
            "seed": trial.seed,
            "target_tokens": trial.token_budget,
            "parameter_count": parameter_count,
        }
        for key, value in expected.items():
            if config.get(key) != value or summary.get(key) != value:
                raise RunnerError(f"training metrics changed the protected {key} contract")
        if config.get("data_config_sha256") != trial.data_config_sha256:
            raise RunnerError("training metrics changed the protected data contract")
        if config.get("tokenizer_sha256") != trial.tokenizer_sha256:
            raise RunnerError("training metrics changed the protected tokenizer contract")
        if summary.get("tokens_seen") != trial.token_budget:
            raise RunnerError("successful training did not consume the exact token budget")
        data_order = _valid_sha256(summary.get("data_order_sha256"))
        if data_order is None:
            raise RunnerError("training summary has no valid data-order commitment")
        checkpoint_hash = file_sha256(paths["checkpoint"])
        if protected.get("checkpoint_sha256") != checkpoint_hash:
            raise RunnerError("protected evaluator used a different checkpoint")
        if protected.get("trainer_sha256") != file_sha256(trainer):
            raise RunnerError("protected evaluator used a different trainer")
        if protected.get("parameter_count") != parameter_count:
            raise RunnerError("protected evaluator parameter count changed")
        validation_bpb = _optional_finite(protected.get("validation_bpb"))
        mean_train_loss = _optional_finite(summary.get("mean_train_loss"))
        evaluation_seconds = _optional_finite(protected.get("evaluation_seconds"))
        evaluation_tps = _optional_finite(protected.get("evaluation_tokens_per_second"))
        evaluation_tokens = _optional_nonnegative_int(protected.get("predicted_tokens"))
        if (
            validation_bpb is None
            or mean_train_loss is None
            or evaluation_seconds is None
            or evaluation_tps is None
            or not evaluation_tokens
        ):
            raise RunnerError("successful run is missing protected finite outcomes")
        if trial.eval_tokens is not None and evaluation_tokens > trial.eval_tokens:
            raise RunnerError("protected evaluator exceeded its declared token budget")
        peak_values = [
            value
            for value in (
                training.peak_process_rss_bytes,
                evaluation.peak_process_rss_bytes,
                _optional_nonnegative_int(protected.get("peak_process_rss_bytes")),
            )
            if value is not None
        ]
        if not peak_values:
            raise RunnerError("protected process RSS measurement is unavailable")
        allocated_values = [
            value
            for value in (
                _optional_nonnegative_int(summary.get("peak_device_allocated_bytes")),
                _optional_nonnegative_int(protected.get("peak_device_allocated_bytes")),
            )
            if value is not None
        ]
        reserved_values = [
            value
            for value in (
                _optional_nonnegative_int(summary.get("peak_device_reserved_bytes")),
                _optional_nonnegative_int(protected.get("peak_device_reserved_bytes")),
            )
            if value is not None
        ]
        if training.wall_seconds <= 0.0:
            raise RunnerError("training wall time is invalid")
        return RunResult(
            run_id=_stable_id("run", trial.trial_id, arm.value),
            trial_id=trial.trial_id,
            arm=arm,
            status=RunStatus.SUCCEEDED,
            seed=trial.seed,
            target_tokens=trial.token_budget,
            tokens_seen=trial.token_budget,
            evaluation_tokens=evaluation_tokens,
            parameter_count=parameter_count,
            validation_bpb=validation_bpb,
            mean_train_loss=mean_train_loss,
            training_tokens_per_second=trial.token_budget / training.wall_seconds,
            evaluation_tokens_per_second=evaluation_tps,
            peak_process_rss_bytes=max(peak_values),
            peak_device_allocated_bytes=max(allocated_values) if allocated_values else None,
            peak_device_reserved_bytes=max(reserved_values) if reserved_values else None,
            training_seconds=training.wall_seconds,
            evaluation_seconds=evaluation_seconds,
            wall_seconds=training.wall_seconds + evaluation.wall_seconds,
            data_order_sha256=data_order,
        )

    def _successful_plugin_run(
        self,
        trial: TrialSpec,
        arm: RunArm,
        trainer: Path,
        paths: dict[str, Path],
        training: ProcessOutcome,
        evaluation: ProcessOutcome,
        *,
        config: dict[str, Any],
        summary: dict[str, Any],
        protected: dict[str, Any],
        parameter_count: int,
    ) -> RunResult:
        assert self.plugin is not None
        expected = {
            "seed": trial.seed,
            "target_units": trial.token_budget,
            "parameter_count": parameter_count,
        }
        for key, value in expected.items():
            if config.get(key) != value or summary.get(key) != value:
                raise RunnerError(f"target metrics changed the protected {key} contract")
        if config.get("data_config_sha256") != trial.data_config_sha256:
            raise RunnerError("target metrics changed the protected data contract")
        if config.get("tokenizer_sha256") != trial.tokenizer_sha256:
            raise RunnerError("target metrics changed the protected tokenizer contract")
        if summary.get("units_seen") != trial.token_budget:
            raise RunnerError("successful target training did not consume the exact budget")
        data_order = _valid_sha256(summary.get("data_order_sha256"))
        if data_order is None:
            raise RunnerError("target training summary has no data-order commitment")
        if protected.get("event") != "target_evaluation":
            raise RunnerError("protected target evaluator emitted the wrong event")
        if protected.get("checkpoint_sha256") != file_sha256(paths["checkpoint"]):
            raise RunnerError("protected target evaluator used a different checkpoint")
        if protected.get("trainer_sha256") != file_sha256(trainer):
            raise RunnerError("protected target evaluator used a different trainer")
        if protected.get("parameter_count") != parameter_count:
            raise RunnerError("protected target evaluator parameter count changed")
        if (
            protected.get("metric_name") != self.plugin.metric.name
            or protected.get("metric_direction") != self.plugin.metric.direction.value
        ):
            raise RunnerError("protected target evaluator changed the declared metric")
        try:
            objective = self.plugin.metric.canonical_objective(protected.get("metric_value"))
        except TargetPluginError as error:
            raise RunnerError(str(error)) from error
        mean_train_loss = _optional_finite(summary.get("mean_train_loss"))
        evaluation_seconds = _optional_finite(protected.get("evaluation_seconds"))
        evaluation_rate = _optional_finite(protected.get("evaluation_units_per_second"))
        evaluation_units = _optional_nonnegative_int(protected.get("evaluation_units"))
        if (
            mean_train_loss is None
            or mean_train_loss < 0.0
            or evaluation_seconds is None
            or evaluation_seconds < 0.0
            or evaluation_rate is None
            or evaluation_rate < 0.0
            or not evaluation_units
        ):
            raise RunnerError("successful target run is missing protected finite outcomes")
        if trial.eval_tokens is not None and evaluation_units > trial.eval_tokens:
            raise RunnerError("protected target evaluator exceeded its declared budget")
        peak_values = [
            value
            for value in (
                training.peak_process_rss_bytes,
                evaluation.peak_process_rss_bytes,
                _optional_nonnegative_int(summary.get("peak_process_rss_bytes")),
                _optional_nonnegative_int(protected.get("peak_process_rss_bytes")),
            )
            if value is not None
        ]
        if not peak_values or training.wall_seconds <= 0.0:
            raise RunnerError("protected target resource measurement is unavailable")
        allocated = [
            value
            for value in (
                _optional_nonnegative_int(summary.get("peak_device_allocated_bytes")),
                _optional_nonnegative_int(protected.get("peak_device_allocated_bytes")),
            )
            if value is not None
        ]
        reserved = [
            value
            for value in (
                _optional_nonnegative_int(summary.get("peak_device_reserved_bytes")),
                _optional_nonnegative_int(protected.get("peak_device_reserved_bytes")),
            )
            if value is not None
        ]
        return RunResult(
            run_id=_stable_id("run", trial.trial_id, arm.value),
            trial_id=trial.trial_id,
            arm=arm,
            status=RunStatus.SUCCEEDED,
            seed=trial.seed,
            target_tokens=trial.token_budget,
            tokens_seen=trial.token_budget,
            evaluation_tokens=evaluation_units,
            parameter_count=parameter_count,
            validation_bpb=objective,
            mean_train_loss=mean_train_loss,
            training_tokens_per_second=trial.token_budget / training.wall_seconds,
            evaluation_tokens_per_second=evaluation_rate,
            peak_process_rss_bytes=max(peak_values),
            peak_device_allocated_bytes=max(allocated) if allocated else None,
            peak_device_reserved_bytes=max(reserved) if reserved else None,
            training_seconds=training.wall_seconds,
            evaluation_seconds=evaluation_seconds,
            wall_seconds=training.wall_seconds + evaluation.wall_seconds,
            data_order_sha256=data_order,
        )

    def _compute_record(self, run: RunResult, trial: TrialSpec) -> ComputeRecord:
        accelerator_seconds = run.wall_seconds if trial.device != "cpu" else 0.0
        cost = None
        if self.request.estimated_accelerator_hour_usd is not None:
            cost = accelerator_seconds / 3_600.0 * self.request.estimated_accelerator_hour_usd
        return ComputeRecord(
            compute_id=_stable_id("compute", run.run_id),
            trial_id=trial.trial_id,
            run_id=run.run_id,
            device=trial.device,
            wall_seconds=run.wall_seconds,
            accelerator_seconds=accelerator_seconds,
            training_tokens=run.tokens_seen,
            evaluation_tokens=run.evaluation_tokens,
            attempts=1,
            estimated_cost_usd=cost,
        )

    def _execute_arm(
        self,
        ledger: ExperimentLedger,
        candidate_root: Path,
        public_data_root: Path,
        trial: TrialSpec,
        arm: RunArm,
        trainer: Path,
        *,
        parameter_count: int,
    ) -> RunResult:
        run_id = _stable_id("run", trial.trial_id, arm.value)
        existing = _record_or_none(ledger, run_id, RunResult)
        if existing is not None:
            manifest = _record_or_none(ledger, _stable_id("manifest", run_id), ArtifactManifest)
            if manifest is None:
                raise RunnerError("existing run is missing its atomic artifact manifest")
            self._verify_manifest(manifest)
            return existing

        paths = self._arm_paths(candidate_root, trial, arm)
        paths["root"].mkdir(parents=True, exist_ok=True)
        training = self._run_process(
            self._training_command(trainer, trial, paths, public_data_root),
            cwd=self._worktree_root(trainer),
            seed=trial.seed,
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
        )
        records: list[dict[str, Any]] = []
        with suppress(RunnerError):
            records = _read_json_lines(paths["metrics"])
        evaluation: ProcessOutcome | None = None
        if training.returncode == 0 and not training.timed_out and not training.cancelled:
            expected_trainer_hash = (
                trial.parent_trainer_sha256
                if arm is RunArm.PARENT
                else trial.candidate_trainer_sha256
            )
            if not _file_matches(trainer, expected_trainer_hash) or not _worktree_is_clean(
                self._worktree_root(trainer)
            ):
                status = RunStatus.INTEGRITY_FAILURE
                reason = "training modified its isolated Git worktree"
            elif paths["checkpoint"].is_file() and paths["metrics"].is_file():
                evaluation = self._run_process(
                    self._evaluation_command(trainer, trial, paths["checkpoint"]),
                    cwd=self._worktree_root(trainer),
                    seed=trial.seed,
                    stdout_path=paths["evaluation"],
                    stderr_path=paths["evaluation_stderr"],
                )
            else:
                status = RunStatus.INTEGRITY_FAILURE
                reason = "training completed without required checkpoint and metrics artifacts"
        else:
            status, reason = classify_failure(
                training,
                _log_text(paths["stdout"], paths["stderr"]),
                phase="training",
            )

        if evaluation is not None:
            if evaluation.returncode != 0 or evaluation.timed_out or evaluation.cancelled:
                status, reason = classify_failure(
                    evaluation,
                    _log_text(paths["evaluation"], paths["evaluation_stderr"]),
                    phase="protected evaluation",
                )
            elif not _worktree_is_clean(self._worktree_root(trainer)):
                status = RunStatus.INTEGRITY_FAILURE
                reason = "protected evaluation modified its isolated Git worktree"
            else:
                try:
                    run = self._successful_run(
                        trial,
                        arm,
                        trainer,
                        paths,
                        training,
                        evaluation,
                        parameter_count=parameter_count,
                    )
                except (RunnerError, ValueError) as error:
                    status = RunStatus.INTEGRITY_FAILURE
                    with paths["evaluation_stderr"].open("a", encoding="utf-8") as output:
                        output.write(
                            f"runner artifact validation failed: {type(error).__name__}: {error}\n"
                        )
                    reason = "protected artifact validation failed"
                else:
                    manifest = self._artifact_manifest(run, paths)
                    compute = self._compute_record(run, trial)
                    ledger.append_many(
                        (
                            (run, WriterRole.EVALUATOR),
                            (manifest, WriterRole.EVALUATOR),
                            (compute, WriterRole.EVALUATOR),
                        )
                    )
                    return run

        run = self._failed_run(
            trial,
            arm,
            status,
            reason,
            training,
            records,
            parameter_count=parameter_count,
            evaluation=evaluation,
        )
        manifest = self._artifact_manifest(run, paths)
        compute = self._compute_record(run, trial)
        ledger.append_many(
            (
                (run, WriterRole.EVALUATOR),
                (manifest, WriterRole.EVALUATOR),
                (compute, WriterRole.EVALUATOR),
            )
        )
        if run.status is RunStatus.CANCELLED:
            raise RunnerCancelled(f"{arm.value} run was cancelled after evidence was retained")
        return run

    def _trial_specs(
        self,
        candidate: CandidateRecord,
        *,
        parent_trainer_sha256: str,
        data_manifest: dict[str, Any] | None,
        evaluator_sha256: str | None = None,
    ) -> tuple[TrialSpec, ...]:
        orders = assign_execution_orders(
            self.request.seeds,
            assignment_seed=self.request.assignment_seed,
        )
        assignment = {
            "algorithm": "balanced-random-v1",
            "assignment_seed": self.request.assignment_seed,
            "candidate_id": candidate.candidate_id,
            "orders": [[arm.value for arm in order] for order in orders],
            "seeds": list(self.request.seeds),
            "stage": self.request.stage.value,
        }
        assignment_hash = _sha256_payload(assignment)
        environment_hash = _environment_sha256(self.device)
        if self.plugin is None:
            evaluator_hash = file_sha256(self.evaluator_path)
            assert data_manifest is not None
            data_config_sha256 = data_manifest["pipeline"]["config_sha256"]
            tokenizer_sha256 = data_manifest["tokenizer"]["artifact"]["sha256"]
        else:
            if evaluator_sha256 is None:
                raise RunnerError("external target evaluator hash is missing")
            evaluator_hash = evaluator_sha256
            data_config_sha256 = self.plugin.data_config_sha256
            tokenizer_sha256 = self.plugin.tokenizer_sha256
        runner_hash = file_sha256(self.runner_path)
        return tuple(
            TrialSpec(
                trial_id=_stable_id(
                    "trial", candidate.candidate_id, self.request.stage.value, seed
                ),
                candidate_id=candidate.candidate_id,
                parent_commit=candidate.parent_commit,
                candidate_commit=candidate.candidate_commit,
                stage=self.request.stage,
                seed=seed,
                token_budget=self.request.token_budget,
                eval_tokens=self.request.eval_tokens,
                batch_size=self.request.batch_size,
                eval_batch_size=self.request.eval_batch_size,
                execution_order=order,
                data_config_sha256=data_config_sha256,
                tokenizer_sha256=tokenizer_sha256,
                parent_trainer_sha256=parent_trainer_sha256,
                candidate_trainer_sha256=candidate.trainer_sha256,
                evaluator_sha256=evaluator_hash,
                runner_sha256=runner_hash,
                environment_sha256=environment_hash,
                order_assignment_sha256=assignment_hash,
                device=self.device,
                limits=self.request.limits,
            )
            for seed, order in zip(self.request.seeds, orders, strict=True)
        )

    def register_candidate(self) -> CandidateRecord:
        """Inspect and register a candidate before the controller schedules its trials."""
        if self.plugin is None and not self.evaluator_path.is_file():
            raise RunnerError("protected evaluator is missing")
        ledger = ExperimentLedger.open(self.request.ledger_path, read_only=False)
        proposal_event = ledger.get(self.request.proposal_id)
        if not isinstance(proposal_event.record, PatchProposal):
            raise RunnerError("proposal_id does not identify a patch proposal")
        proposal = proposal_event.record
        if proposal.parent_commit != ledger.current_parent():
            raise RunnerError("proposal parent is no longer the accepted ledger parent")
        validation = validate_candidate_patch(
            self.repository_root,
            parent_commit=proposal.parent_commit,
            candidate_commit=self.request.candidate_commit,
            allowed_paths=self.editable_paths,
            trainer_path=self.trainer_path,
        )
        existing = [
            event.record
            for event in ledger.events()
            if isinstance(event.record, CandidateRecord)
            and event.record.proposal_id == proposal.proposal_id
        ]
        if existing:
            if len(existing) != 1:
                raise RunnerError("proposal has multiple candidate records")
            candidate = existing[0]
            trainer_blob = _git(
                self.repository_root,
                "show",
                f"{validation.candidate_commit}:{self.trainer_path}",
                text=False,
            )
            assert isinstance(trainer_blob, bytes)
            if (
                candidate.candidate_commit != validation.candidate_commit
                or candidate.parent_commit != validation.parent_commit
                or candidate.changed_paths != validation.changed_paths
                or candidate.diff_sha256 != validation.diff_sha256
                or candidate.trainer_sha256 != hashlib.sha256(trainer_blob).hexdigest()
                or candidate.policy_sha256 != self.policy_contract_sha256
            ):
                raise RunnerError("existing candidate record differs from its immutable commit")
            return candidate
        self.output_root.mkdir(parents=True, exist_ok=True)
        control_root = self.output_root / ".control"
        with isolated_worktrees(
            self.repository_root,
            control_root / "registration-worktrees",
            parent_commit=validation.parent_commit,
            candidate_commit=validation.candidate_commit,
            trainer_path=self.trainer_path,
        ) as worktrees:
            self._validate_plugin_evaluators(worktrees)
            inspections = {
                arm: self._inspect_trainer(
                    worktrees[arm] / self.trainer_path,
                    arm=arm,
                    control_root=control_root,
                )
                for arm in (RunArm.PARENT, RunArm.CANDIDATE)
            }
            if any(not _worktree_is_clean(path) for path in worktrees.values()):
                raise RunnerError("protected inspection changed an isolated worktree")
            candidate = CandidateRecord(
                candidate_id=_stable_id(
                    "candidate", proposal.proposal_id, validation.candidate_commit
                ),
                proposal_id=proposal.proposal_id,
                parent_commit=validation.parent_commit,
                candidate_commit=validation.candidate_commit,
                diff_sha256=validation.diff_sha256,
                changed_paths=validation.changed_paths,
                trainer_sha256=file_sha256(worktrees[RunArm.CANDIDATE] / self.trainer_path),
                policy_sha256=self.policy_contract_sha256,
                parameter_count=int(inspections[RunArm.CANDIDATE]["parameter_count"]),
            )
        ledger.ensure(candidate, writer_role=WriterRole.CONTROLLER)
        return candidate

    def run(self) -> dict[str, Any]:
        candidate = self.register_candidate()
        ledger = ExperimentLedger.open(self.request.ledger_path, read_only=False)
        proposal_event = ledger.get(self.request.proposal_id)
        if not isinstance(proposal_event.record, PatchProposal):
            raise RunnerError("proposal_id does not identify a patch proposal")
        proposal = proposal_event.record
        if proposal.parent_commit != ledger.current_parent():
            raise RunnerError("proposal parent is no longer the accepted ledger parent")
        validation = validate_candidate_patch(
            self.repository_root,
            parent_commit=proposal.parent_commit,
            candidate_commit=self.request.candidate_commit,
            allowed_paths=self.editable_paths,
            trainer_path=self.trainer_path,
        )
        if self.plugin is None:
            data_manifest: dict[str, Any] | None = verify_dataset(self.data_root, scope="public")
        else:
            data_manifest = None
            if self.public_data_root is None or not self.public_data_root.is_dir():
                raise RunnerError("external target public_data_root is missing")
            if not self.data_root.is_dir():
                raise RunnerError("external target protected data_root is missing")
            if self.public_data_root == self.data_root:
                raise RunnerError("external target public and protected data roots must differ")
        self.output_root.mkdir(parents=True, exist_ok=True)
        control_root = self.output_root / ".control"
        public_data_root = (
            prepare_public_data_view(self.data_root, control_root / "public-data")
            if self.plugin is None
            else self.public_data_root
        )
        assert public_data_root is not None
        worktree_parent = control_root / "worktrees"
        with isolated_worktrees(
            self.repository_root,
            worktree_parent,
            parent_commit=validation.parent_commit,
            candidate_commit=validation.candidate_commit,
            trainer_path=self.trainer_path,
        ) as worktrees:
            self._validate_plugin_evaluators(worktrees)
            inspections = {
                arm: self._inspect_trainer(
                    worktrees[arm] / self.trainer_path,
                    arm=arm,
                    control_root=control_root,
                )
                for arm in (RunArm.PARENT, RunArm.CANDIDATE)
            }
            if (
                candidate.parent_commit != validation.parent_commit
                or candidate.candidate_commit != validation.candidate_commit
                or candidate.trainer_sha256
                != file_sha256(worktrees[RunArm.CANDIDATE] / self.trainer_path)
                or candidate.parameter_count
                != int(inspections[RunArm.CANDIDATE]["parameter_count"])
                or candidate.policy_sha256 != self.policy_contract_sha256
            ):
                raise RunnerError("registered candidate differs from protected inspection")
            trials = self._trial_specs(
                candidate,
                parent_trainer_sha256=file_sha256(worktrees[RunArm.PARENT] / self.trainer_path),
                data_manifest=data_manifest,
                evaluator_sha256=(
                    None
                    if self.plugin is None
                    else file_sha256(
                        resolve_repository_path(
                            worktrees[RunArm.PARENT], self.plugin.evaluator_path
                        )
                    )
                ),
            )
            candidate_root = self.output_root / candidate.candidate_id
            contract = {
                "assignment_seed": self.request.assignment_seed,
                "candidate_commit": candidate.candidate_commit,
                "candidate_id": candidate.candidate_id,
                "eval_tokens": self.request.eval_tokens,
                "parent_commit": candidate.parent_commit,
                "proposal_id": proposal.proposal_id,
                "schema_version": RUNNER_SCHEMA_VERSION,
                "seeds": list(self.request.seeds),
                "stage": self.request.stage.value,
                "target_contract_sha256": self.policy_contract_sha256,
                "token_budget": self.request.token_budget,
                "trial_contract_sha256": [
                    _sha256_payload(record_to_envelope(trial)) for trial in trials
                ],
                "trial_ids": [trial.trial_id for trial in trials],
            }
            contract_sha256 = _sha256_payload(contract)
            contract_path = (
                candidate_root / self.request.stage.value / f"contract-{contract_sha256}.json"
            )
            if contract_path.is_file():
                if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
                    raise RunnerError("existing candidate output contract differs")
            else:
                _write_json(contract_path, contract)
            ledger.append_many(
                tuple((trial, WriterRole.CONTROLLER) for trial in trials),
                idempotent=True,
            )

            run_summaries = []
            pair_ids = []
            for trial in trials:
                results: dict[RunArm, RunResult] = {}
                for arm in trial.execution_order:
                    results[arm] = self._execute_arm(
                        ledger,
                        candidate_root,
                        public_data_root,
                        trial,
                        arm,
                        worktrees[arm] / self.trainer_path,
                        parameter_count=int(inspections[arm]["parameter_count"]),
                    )
                parent = results[RunArm.PARENT]
                candidate_run = results[RunArm.CANDIDATE]
                paired: PairedResult | None = None
                if (
                    parent.status is RunStatus.SUCCEEDED
                    and candidate_run.status is RunStatus.SUCCEEDED
                ):
                    pair_id = _stable_id("pair", trial.trial_id)
                    paired = _record_or_none(ledger, pair_id, PairedResult)
                    if paired is None:
                        failures = resource_constraint_failures(trial, parent, candidate_run)
                        paired = build_paired_result(
                            pair_id,
                            trial=trial,
                            candidate_id=trial.candidate_id,
                            parent=parent,
                            candidate=candidate_run,
                            constraint_failures=failures,
                        )
                        ledger.append(paired, writer_role=WriterRole.EVALUATOR)
                    pair_ids.append(paired.paired_result_id)
                run_summaries.append(
                    {
                        "candidate_status": candidate_run.status.value,
                        "execution_order": [arm.value for arm in trial.execution_order],
                        "gain_bpb": None if paired is None else paired.gain_bpb,
                        "parent_status": parent.status.value,
                        "seed": trial.seed,
                        "trial_id": trial.trial_id,
                    }
                )
            if (
                file_sha256(worktrees[RunArm.PARENT] / self.trainer_path)
                != trials[0].parent_trainer_sha256
            ):
                raise RunnerError("parent trainer changed during paired execution")
            if (
                file_sha256(worktrees[RunArm.CANDIDATE] / self.trainer_path)
                != candidate.trainer_sha256
            ):
                raise RunnerError("candidate trainer changed during paired execution")
        return {
            "candidate_id": candidate.candidate_id,
            "event": "paired_experiment_completed",
            "pair_ids": pair_ids,
            "runs": run_summaries,
            "schema_version": RUNNER_SCHEMA_VERSION,
            "stage": self.request.stage.value,
        }


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run protected paired patch experiments.")
    parser.add_argument("--repository-root", type=_path, default=Path.cwd())
    parser.add_argument("--target-config", type=_path)
    parser.add_argument("--ledger-path", type=_path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--data-root", type=_path, default=default_output_root())
    parser.add_argument("--output-root", type=_path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--proposal-id", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument(
        "--stage",
        type=ExperimentStage,
        choices=tuple(ExperimentStage),
        default=ExperimentStage.CHEAP,
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--assignment-seed", type=int, required=True)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--eval-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=7_200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-parameter-count", type=int, default=DEFAULT_PARAMETER_CAP)
    parser.add_argument("--max-peak-process-rss-bytes", type=int)
    parser.add_argument("--max-peak-device-bytes", type=int)
    parser.add_argument("--min-training-tokens-per-second", type=float)
    parser.add_argument("--max-throughput-regression", type=float)
    parser.add_argument("--max-process-rss-regression", type=float)
    parser.add_argument("--max-device-memory-regression", type=float)
    parser.add_argument("--estimated-accelerator-hour-usd", type=float)
    return parser


def request_from_args(args: argparse.Namespace) -> ExperimentRequest:
    defaults = STAGE_DEFAULTS[args.stage]
    token_budget = defaults.token_budget if args.token_budget is None else args.token_budget
    eval_tokens = defaults.eval_tokens if args.eval_tokens is None else args.eval_tokens
    target = None if args.target_config is None else TargetConfig.from_path(args.target_config)
    repository_root = args.repository_root.resolve()
    parameter_cap = args.max_parameter_count if target is None else target.max_parameter_count
    limits = ResourceLimits(
        timeout_seconds=args.timeout_seconds,
        max_parameter_count=parameter_cap,
        max_peak_process_rss_bytes=args.max_peak_process_rss_bytes,
        max_peak_device_bytes=args.max_peak_device_bytes,
        min_training_tokens_per_second=args.min_training_tokens_per_second,
        max_training_throughput_regression_fraction=args.max_throughput_regression,
        max_peak_process_rss_regression_fraction=args.max_process_rss_regression,
        max_peak_device_regression_fraction=args.max_device_memory_regression,
    )
    return ExperimentRequest(
        repository_root=args.repository_root,
        ledger_path=args.ledger_path,
        data_root=(
            args.data_root if target is None else target.resolved_data_root(repository_root)
        ),
        output_root=args.output_root,
        proposal_id=args.proposal_id,
        candidate_commit=args.candidate_commit,
        stage=args.stage,
        seeds=tuple(args.seeds),
        assignment_seed=args.assignment_seed,
        token_budget=token_budget,
        eval_tokens=eval_tokens,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        timeout_seconds=args.timeout_seconds,
        device=args.device if target is None else target.device,
        limits=limits,
        estimated_accelerator_hour_usd=(
            args.estimated_accelerator_hour_usd
            if target is None
            else target.estimated_accelerator_hour_usd
        ),
        target_config_path=args.target_config,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = PairedExperimentRunner(request_from_args(args)).run()
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (LedgerError, RunnerError, TargetError, TargetPluginError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
