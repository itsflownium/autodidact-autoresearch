"""Freeze accepted lineages, run sealed evaluation, and publish reproducible reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autodidact.checkpoints import file_sha256
from autodidact.integrity import canonical_json_bytes
from autodidact.ledger import ExperimentLedger, LedgerError
from autodidact.records import (
    CandidateRecord,
    DecisionRecord,
    DecisionVerdict,
    LineageRecord,
    PatchProposal,
)
from autodidact.rl import (
    RLContractError,
    validate_evaluation_diagnostics,
    validate_training_diagnostics,
)
from autodidact.runner import (
    ProcessOutcome,
    run_process,
    sanitized_environment,
)
from autodidact.runstate import RepositoryLock, RunStateError
from autodidact.target import TargetConfig, TargetError
from autodidact.target_plugins import TargetPluginError, TargetPluginSpec, resolve_repository_path

SEALED_SCHEMA_VERSION = 2
DEFAULT_SEALED_ROOT = Path("artifacts/sealed")
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")
_COMMIT_LENGTHS = {40, 64}
_SHA256_LENGTH = 64


class SealedError(RuntimeError):
    """Raised when frozen or sealed evidence is incomplete or inconsistent."""


def _valid_name(value: str) -> bool:
    return (
        isinstance(value, str)
        and 3 <= len(value) <= 64
        and value[0].isalpha()
        and value[0].islower()
        and all(character in _NAME_CHARS for character in value)
    )


def _valid_commit(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) in _COMMIT_LENGTHS
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class FrozenGeneration:
    generation: int
    parent_commit: str | None
    commit: str
    candidate_id: str | None
    decision_id: str | None
    minimum_useful_gain: float | None

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise SealedError("frozen generation must be a nonnegative integer")
        if not _valid_commit(self.commit):
            raise SealedError("frozen generation commit is invalid")
        if self.generation == 0:
            if any(
                value is not None
                for value in (
                    self.parent_commit,
                    self.candidate_id,
                    self.decision_id,
                    self.minimum_useful_gain,
                )
            ):
                raise SealedError("generation zero cannot identify a promotion")
            return
        if not _valid_commit(self.parent_commit or ""):
            raise SealedError("promoted generation parent commit is invalid")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise SealedError("promoted generation requires a candidate ID")
        if not isinstance(self.decision_id, str) or not self.decision_id:
            raise SealedError("promoted generation requires a decision ID")
        gain = self.minimum_useful_gain
        if not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
            raise SealedError("promoted generation requires a positive useful-gain threshold")

    @classmethod
    def from_mapping(cls, value: Any) -> FrozenGeneration:
        if not isinstance(value, dict):
            raise SealedError("frozen generation must be an object")
        try:
            return cls(**value)
        except TypeError as error:
            raise SealedError("frozen generation schema is invalid") from error


@dataclass(frozen=True, slots=True)
class FrozenArm:
    name: str
    ledger_path: str
    ledger_id: str
    ledger_head_sha256: str
    compute: dict[str, float | int]
    generations: tuple[FrozenGeneration, ...]

    def __post_init__(self) -> None:
        if not _valid_name(self.name):
            raise SealedError("sealed arm name is invalid")
        if not isinstance(self.ledger_path, str) or not self.ledger_path:
            raise SealedError("sealed arm ledger path is invalid")
        if not isinstance(self.ledger_id, str) or not self.ledger_id:
            raise SealedError("sealed arm ledger ID is invalid")
        if not _valid_sha256(self.ledger_head_sha256):
            raise SealedError("sealed arm ledger head is invalid")
        if not self.generations or self.generations[0].generation != 0:
            raise SealedError("sealed arm must begin with generation zero")
        for expected, generation in enumerate(self.generations):
            if generation.generation != expected:
                raise SealedError("sealed arm generations must be contiguous")
            if expected and generation.parent_commit != self.generations[expected - 1].commit:
                raise SealedError("sealed arm lineage commits are not contiguous")
        required_compute = {
            "accelerator_seconds",
            "estimated_cost_usd",
            "evaluation_tokens",
            "training_tokens",
            "wall_seconds",
        }
        if set(self.compute) != required_compute or any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in self.compute.values()
        ):
            raise SealedError("sealed arm compute summary is invalid")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "compute": self.compute,
            "generations": [asdict(item) for item in self.generations],
            "ledger_head_sha256": self.ledger_head_sha256,
            "ledger_id": self.ledger_id,
            "ledger_path": self.ledger_path,
            "name": self.name,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> FrozenArm:
        if not isinstance(value, dict):
            raise SealedError("frozen arm must be an object")
        try:
            return cls(
                name=value["name"],
                ledger_path=value["ledger_path"],
                ledger_id=value["ledger_id"],
                ledger_head_sha256=value["ledger_head_sha256"],
                compute=value["compute"],
                generations=tuple(
                    FrozenGeneration.from_mapping(item) for item in value["generations"]
                ),
            )
        except (KeyError, TypeError) as error:
            raise SealedError("frozen arm schema is invalid") from error


@dataclass(frozen=True, slots=True)
class SealedPlan:
    plan_id: str
    initial_parent_commit: str
    arms: tuple[FrozenArm, ...]
    seeds: tuple[int, ...]
    assignment_seed: int
    token_budget: int
    batch_size: int
    eval_batch_size: int
    timeout_seconds: int
    device: str
    parameter_cap: int
    data_root: str
    public_data_root: str
    plugin: dict[str, Any]
    target_contract_sha256: str
    evaluator_sha256: str
    runner_sha256: str

    def __post_init__(self) -> None:
        if not _valid_name(self.plan_id):
            raise SealedError("sealed plan ID is invalid")
        if not _valid_commit(self.initial_parent_commit):
            raise SealedError("sealed initial parent is invalid")
        if not self.arms or len({arm.name for arm in self.arms}) != len(self.arms):
            raise SealedError("sealed plan requires unique arms")
        if any(arm.generations[0].commit != self.initial_parent_commit for arm in self.arms):
            raise SealedError("sealed arms do not share the same initial parent")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise SealedError("sealed evaluation requires at least two unique seeds")
        if any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in self.seeds):
            raise SealedError("sealed plan contains an invalid seed")
        for name in (
            "assignment_seed",
            "token_budget",
            "batch_size",
            "eval_batch_size",
            "timeout_seconds",
            "parameter_cap",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "assignment_seed" else 1
            if type(value) is not int or value < minimum:
                raise SealedError(f"{name} is invalid")
        if self.assignment_seed > 2**32 - 1:
            raise SealedError("assignment_seed exceeds 32 bits")
        for name in ("device", "data_root", "public_data_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise SealedError(f"{name} must be nonempty portable text")
        for name in ("target_contract_sha256", "evaluator_sha256", "runner_sha256"):
            if not _valid_sha256(getattr(self, name)):
                raise SealedError(f"{name} is invalid")
        try:
            plugin = TargetPluginSpec.from_mapping(self.plugin)
        except TargetPluginError as error:
            raise SealedError(str(error)) from error
        expected_contract = hashlib.sha256(canonical_json_bytes(plugin.to_mapping())).hexdigest()
        if expected_contract != self.target_contract_sha256:
            raise SealedError("sealed target plugin differs from its contract hash")
        if Path(self.public_data_root).resolve() == Path(self.data_root).resolve():
            raise SealedError("sealed public and protected data roots must differ")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "arms": [arm.to_mapping() for arm in self.arms],
            "assignment_seed": self.assignment_seed,
            "batch_size": self.batch_size,
            "data_root": self.data_root,
            "device": self.device,
            "eval_batch_size": self.eval_batch_size,
            "evaluator_sha256": self.evaluator_sha256,
            "initial_parent_commit": self.initial_parent_commit,
            "parameter_cap": self.parameter_cap,
            "plan_id": self.plan_id,
            "plugin": self.plugin,
            "public_data_root": self.public_data_root,
            "runner_sha256": self.runner_sha256,
            "schema_version": SEALED_SCHEMA_VERSION,
            "seeds": list(self.seeds),
            "timeout_seconds": self.timeout_seconds,
            "token_budget": self.token_budget,
            "target_contract_sha256": self.target_contract_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> SealedPlan:
        if not isinstance(value, dict) or value.get("schema_version") != SEALED_SCHEMA_VERSION:
            raise SealedError("sealed plan schema version is invalid")
        expected = {
            "arms",
            "assignment_seed",
            "batch_size",
            "data_root",
            "device",
            "eval_batch_size",
            "evaluator_sha256",
            "initial_parent_commit",
            "parameter_cap",
            "plan_id",
            "plugin",
            "public_data_root",
            "runner_sha256",
            "schema_version",
            "seeds",
            "timeout_seconds",
            "token_budget",
            "target_contract_sha256",
        }
        if set(value) != expected:
            raise SealedError("sealed plan keys are invalid")
        return cls(
            plan_id=value["plan_id"],
            initial_parent_commit=value["initial_parent_commit"],
            arms=tuple(FrozenArm.from_mapping(item) for item in value["arms"]),
            seeds=tuple(value["seeds"]),
            assignment_seed=value["assignment_seed"],
            token_budget=value["token_budget"],
            batch_size=value["batch_size"],
            eval_batch_size=value["eval_batch_size"],
            timeout_seconds=value["timeout_seconds"],
            device=value["device"],
            parameter_cap=value["parameter_cap"],
            data_root=value["data_root"],
            public_data_root=value["public_data_root"],
            plugin=value["plugin"],
            target_contract_sha256=value["target_contract_sha256"],
            evaluator_sha256=value["evaluator_sha256"],
            runner_sha256=value["runner_sha256"],
        )


def _resolve(repository_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise SealedError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _freeze_arm(
    repository: Path,
    name: str,
    ledger_path: Path,
    *,
    trainer_path: str,
) -> FrozenArm:
    if not _valid_name(name):
        raise SealedError(f"invalid arm name: {name}")
    ledger = ExperimentLedger.open(ledger_path, read_only=True)
    summary = ledger.summary()
    if summary["running_trial_ids"]:
        raise SealedError(f"{name} has unfinished trials and cannot be frozen")
    records = [event.record for event in ledger.events()]
    terminal_candidates = {
        record.candidate_id
        for record in records
        if isinstance(record, DecisionRecord)
        and record.verdict in {DecisionVerdict.REJECT, DecisionVerdict.PROMOTE}
    }
    pending_candidates = {
        record.candidate_id for record in records if isinstance(record, CandidateRecord)
    } - terminal_candidates
    if pending_candidates:
        raise SealedError(f"{name} has candidates without terminal decisions")
    generations = [
        FrozenGeneration(
            generation=0,
            parent_commit=None,
            commit=summary["initial_parent_commit"],
            candidate_id=None,
            decision_id=None,
            minimum_useful_gain=None,
        )
    ]
    lineages = sorted(
        (record for record in records if isinstance(record, LineageRecord)),
        key=lambda item: item.generation,
    )
    for lineage in lineages:
        candidate_event = ledger.get(lineage.candidate_id).record
        decision_event = ledger.get(lineage.decision_id).record
        if not isinstance(candidate_event, CandidateRecord) or not isinstance(
            decision_event, DecisionRecord
        ):
            raise SealedError(f"{name} lineage references invalid candidate evidence")
        proposal_event = ledger.get(candidate_event.proposal_id).record
        if not isinstance(proposal_event, PatchProposal):
            raise SealedError(f"{name} lineage proposal is missing")
        if decision_event.verdict is not DecisionVerdict.PROMOTE:
            raise SealedError(f"{name} lineage decision is not a promotion")
        generations.append(
            FrozenGeneration(
                generation=lineage.generation,
                parent_commit=lineage.parent_commit,
                commit=lineage.candidate_commit,
                candidate_id=lineage.candidate_id,
                decision_id=lineage.decision_id,
                minimum_useful_gain=proposal_event.minimum_useful_gain,
            )
        )
    if generations[-1].commit != summary["current_parent_commit"]:
        raise SealedError(f"{name} frozen lineage does not end at the ledger parent")
    for generation in generations:
        _git(repository, "cat-file", "-e", f"{generation.commit}^{{commit}}")
        if _git(repository, "cat-file", "-t", f"{generation.commit}:{trainer_path}") != "blob":
            raise SealedError(f"{name} generation {generation.generation} lacks {trainer_path}")
    return FrozenArm(
        name=name,
        ledger_path=str(ledger_path),
        ledger_id=summary["ledger_id"],
        ledger_head_sha256=summary["head_event_sha256"],
        compute=summary["compute"],
        generations=tuple(generations),
    )


def _plan_digest_payload(
    *,
    arms: tuple[FrozenArm, ...],
    seeds: tuple[int, ...],
    assignment_seed: int,
    token_budget: int,
    batch_size: int,
    eval_batch_size: int,
    timeout_seconds: int,
    device: str,
    parameter_cap: int,
    data_root: str,
    public_data_root: str,
    plugin: dict[str, Any],
    target_contract_sha256: str,
    evaluator_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    return {
        "arms": [arm.to_mapping() for arm in arms],
        "assignment_seed": assignment_seed,
        "batch_size": batch_size,
        "data_root": data_root,
        "device": device,
        "eval_batch_size": eval_batch_size,
        "evaluator_sha256": evaluator_sha256,
        "parameter_cap": parameter_cap,
        "plugin": plugin,
        "public_data_root": public_data_root,
        "runner_sha256": runner_sha256,
        "seeds": list(seeds),
        "timeout_seconds": timeout_seconds,
        "token_budget": token_budget,
        "target_contract_sha256": target_contract_sha256,
    }


def create_plan(
    *,
    repository_root: Path,
    sealed_root: Path,
    arm_ledgers: Sequence[tuple[str, Path]],
    seeds: tuple[int, ...],
    assignment_seed: int,
    token_budget: int,
    batch_size: int,
    eval_batch_size: int,
    timeout_seconds: int,
    target_config_path: Path,
) -> SealedPlan:
    repository = repository_root.expanduser().resolve()
    root = sealed_root.expanduser().resolve()
    if root.exists():
        raise SealedError(f"sealed root already exists: {root}")
    if not arm_ledgers:
        raise SealedError("sealed plan requires at least one arm ledger")
    target = TargetConfig.from_path(target_config_path)
    plugin = target.load_plugin(repository)
    with RepositoryLock(repository, campaign_id="sealed-freeze"):
        arms = tuple(
            _freeze_arm(
                repository,
                name,
                path.expanduser().resolve(),
                trainer_path=plugin.trainer_path,
            )
            for name, path in arm_ledgers
        )
    initial = arms[0].generations[0].commit
    if any(arm.generations[0].commit != initial for arm in arms):
        raise SealedError("all sealed arms must share one initial parent")
    data_root = target.resolved_data_root(repository)
    public_data_root = target.resolved_public_data_root(repository)
    if public_data_root is None or not public_data_root.is_dir():
        raise SealedError("target public_data_root is missing")
    if not data_root.is_dir() or data_root == public_data_root:
        raise SealedError("target protected data_root is missing or not isolated")
    evaluator_path = resolve_repository_path(repository, plugin.evaluator_path)
    plugin_mapping = plugin.to_mapping()
    target_contract_sha256 = hashlib.sha256(canonical_json_bytes(plugin_mapping)).hexdigest()
    runner_path = Path(__file__).resolve()
    payload = _plan_digest_payload(
        arms=arms,
        seeds=seeds,
        assignment_seed=assignment_seed,
        token_budget=token_budget,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        timeout_seconds=timeout_seconds,
        device=target.device,
        parameter_cap=target.max_parameter_count,
        data_root=str(data_root.expanduser().resolve()),
        public_data_root=str(public_data_root),
        plugin=plugin_mapping,
        target_contract_sha256=target_contract_sha256,
        evaluator_sha256=file_sha256(evaluator_path),
        runner_sha256=file_sha256(runner_path),
    )
    plan = SealedPlan(
        plan_id=f"sealed-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]}",
        initial_parent_commit=initial,
        arms=arms,
        seeds=seeds,
        assignment_seed=assignment_seed,
        token_budget=token_budget,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        timeout_seconds=timeout_seconds,
        device=target.device,
        parameter_cap=target.max_parameter_count,
        data_root=str(data_root.expanduser().resolve()),
        public_data_root=str(public_data_root),
        plugin=plugin_mapping,
        target_contract_sha256=target_contract_sha256,
        evaluator_sha256=payload["evaluator_sha256"],
        runner_sha256=payload["runner_sha256"],
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        mapping = plan.to_mapping()
        raw = canonical_json_bytes(mapping)
        (staging / "plan.json").write_bytes(raw + b"\n")
        (staging / "plan.sha256").write_text(
            hashlib.sha256(raw).hexdigest() + "\n", encoding="ascii"
        )
        (staging / ".autodidact-sealed").write_text(
            f"schema_version={SEALED_SCHEMA_VERSION}\n", encoding="ascii"
        )
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return plan


def load_plan(sealed_root: Path) -> SealedPlan:
    root = sealed_root.expanduser().resolve()
    if not (root / ".autodidact-sealed").is_file():
        raise SealedError("sealed root is missing its ownership marker")
    try:
        raw = (root / "plan.json").read_bytes()
        expected = (root / "plan.sha256").read_text(encoding="ascii").strip()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SealedError(f"cannot read sealed plan: {error}") from error
    canonical = canonical_json_bytes(value)
    if raw != canonical + b"\n" or hashlib.sha256(canonical).hexdigest() != expected:
        raise SealedError("sealed plan is not canonical or its digest is invalid")
    return SealedPlan.from_mapping(value)


def _verify_frozen_inputs(repository: Path, plan: SealedPlan) -> None:
    if file_sha256(Path(__file__).resolve()) != plan.runner_sha256:
        raise SealedError("sealed runner changed after plan creation")
    plugin = TargetPluginSpec.from_mapping(plan.plugin)
    evaluator_path = resolve_repository_path(repository, plugin.evaluator_path)
    if file_sha256(evaluator_path) != plan.evaluator_sha256:
        raise SealedError("protected evaluator changed after plan creation")
    if hashlib.sha256(canonical_json_bytes(plugin.to_mapping())).hexdigest() != (
        plan.target_contract_sha256
    ):
        raise SealedError("target plugin changed after plan creation")
    if not Path(plan.public_data_root).is_dir() or not Path(plan.data_root).is_dir():
        raise SealedError("sealed target data roots are unavailable")
    for arm in plan.arms:
        ledger = ExperimentLedger.open(Path(arm.ledger_path), read_only=True)
        summary = ledger.summary()
        if summary["ledger_id"] != arm.ledger_id or summary["head_event_sha256"] != (
            arm.ledger_head_sha256
        ):
            raise SealedError(f"{arm.name} ledger changed after sealed plan creation")
        if summary["current_parent_commit"] != arm.generations[-1].commit:
            raise SealedError(f"{arm.name} ledger parent differs from its frozen lineage")
    for commit in {item.commit for arm in plan.arms for item in arm.generations}:
        _git(repository, "cat-file", "-e", f"{commit}^{{commit}}")


def _run_specs(plan: SealedPlan) -> tuple[tuple[str, int], ...]:
    commits = {generation.commit for arm in plan.arms for generation in arm.generations}

    def assignment(item: tuple[str, int]) -> str:
        commit, seed = item
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "assignment_seed": plan.assignment_seed,
                    "commit": commit,
                    "domain": "autodidact-sealed-order-v1",
                    "seed": seed,
                }
            )
        ).hexdigest()

    return tuple(
        item
        for seed in plan.seeds
        for item in sorted(((commit, seed) for commit in commits), key=assignment)
    )


def _run_root(sealed_root: Path, commit: str, seed: int) -> Path:
    return sealed_root / "runs" / commit / f"seed-{seed}"


def _write_canonical(path: Path, value: Any) -> None:
    raw = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(raw + b"\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _read_canonical(path: Path) -> Any:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SealedError(f"cannot read canonical artifact {path}: {error}") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise SealedError(f"artifact is not canonical JSON: {path}")
    return value


def _last_json(path: Path, *, event: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SealedError(f"cannot read process output {path}: {error}") from error
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("event") == event:
            return value
    raise SealedError(f"process output lacks required {event} event: {path}")


def _process_payload(outcome: ProcessOutcome) -> dict[str, Any]:
    return {
        "cancelled": outcome.cancelled,
        "peak_process_rss_bytes": outcome.peak_process_rss_bytes,
        "returncode": outcome.returncode,
        "timed_out": outcome.timed_out,
        "wall_seconds": outcome.wall_seconds,
    }


def _require_success(outcome: ProcessOutcome, *, phase: str, stderr_path: Path) -> None:
    if outcome.returncode == 0 and not outcome.timed_out and not outcome.cancelled:
        return
    try:
        detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2_000:].strip()
    except OSError:
        detail = ""
    suffix = f": {detail}" if detail else ""
    raise SealedError(f"sealed {phase} failed with status {outcome.returncode}{suffix}")


@contextmanager
def _single_worktree(
    repository: Path,
    worktree_parent: Path,
    commit: str,
    *,
    trainer_path: str,
) -> Iterator[Path]:
    worktree_parent.mkdir(parents=True, exist_ok=True)
    holder = Path(tempfile.mkdtemp(prefix=f"sealed-{commit[:12]}-", dir=worktree_parent))
    root = holder / "checkout"
    added = False
    try:
        _git(repository, "worktree", "add", "--detach", str(root), commit)
        added = True
        trainer = resolve_repository_path(root, trainer_path)
        if not trainer.is_file() or trainer.is_symlink():
            raise SealedError(f"frozen commit has an invalid {trainer_path}")
        yield root
    finally:
        if added:
            with suppress(SealedError):
                _git(repository, "worktree", "remove", "--force", str(root))
        shutil.rmtree(holder, ignore_errors=True)
        with suppress(SealedError):
            _git(repository, "worktree", "prune")


def _worktree_clean(worktree: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _run_contract(plan: SealedPlan, commit: str, seed: int, trainer_sha256: str) -> dict[str, Any]:
    return {
        "commit": commit,
        "parameter_cap": plan.parameter_cap,
        "plan_id": plan.plan_id,
        "schema_version": SEALED_SCHEMA_VERSION,
        "seed": seed,
        "split": "sealed_final",
        "target_contract_sha256": plan.target_contract_sha256,
        "token_budget": plan.token_budget,
        "trainer_sha256": trainer_sha256,
    }


def _load_retained_result(
    run_root: Path,
    *,
    expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    result_path = run_root / "result.json"
    digest_path = run_root / "result.sha256"
    if not result_path.exists() and not digest_path.exists():
        return None
    if not result_path.is_file() or not digest_path.is_file():
        raise SealedError(f"partial sealed result exists under {run_root}")
    result = _read_canonical(result_path)
    expected = digest_path.read_text(encoding="ascii").strip()
    if hashlib.sha256(canonical_json_bytes(result)).hexdigest() != expected:
        raise SealedError(f"sealed result digest is invalid under {run_root}")
    if expected_contract is not None and result.get("contract") != expected_contract:
        raise SealedError(f"sealed result contract differs under {run_root}")
    checkpoint = run_root / "checkpoint.pt"
    if not checkpoint.is_file() or file_sha256(checkpoint) != result.get("checkpoint_sha256"):
        raise SealedError(f"sealed checkpoint differs under {run_root}")
    return result


def _validate_result(
    plan: SealedPlan,
    result: dict[str, Any],
    *,
    commit: str,
    seed: int,
) -> None:
    expected_keys = {
        "checkpoint_sha256",
        "contract",
        "evaluation",
        "inspection",
        "processes",
        "training_summary",
    }
    if set(result) != expected_keys:
        raise SealedError("sealed result keys are invalid")
    contract = result["contract"]
    if not isinstance(contract, dict) or not _valid_sha256(contract.get("trainer_sha256")):
        raise SealedError("sealed result trainer contract is invalid")
    expected_contract = _run_contract(plan, commit, seed, contract["trainer_sha256"])
    if contract != expected_contract:
        raise SealedError("sealed result differs from the frozen run contract")
    if not _valid_sha256(result["checkpoint_sha256"]):
        raise SealedError("sealed result checkpoint hash is invalid")
    inspection = result["inspection"]
    if (
        not isinstance(inspection, dict)
        or inspection.get("event") != "target_inspection"
        or inspection.get("trainer_sha256") != contract["trainer_sha256"]
        or type(inspection.get("parameter_count")) is not int
        or not 0 < inspection["parameter_count"] <= plan.parameter_cap
    ):
        raise SealedError("sealed protected inspection is invalid")
    evaluation = result["evaluation"]
    plugin = TargetPluginSpec.from_mapping(plan.plugin)
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("event") != "target_evaluation"
        or evaluation.get("checkpoint_sha256") != result["checkpoint_sha256"]
        or evaluation.get("trainer_sha256") != contract["trainer_sha256"]
        or evaluation.get("parameter_count") != inspection["parameter_count"]
        or evaluation.get("metric_name") != plugin.metric.name
        or evaluation.get("metric_direction") != plugin.metric.direction.value
        or not isinstance(evaluation.get("objective_value"), (int, float))
        or not math.isfinite(evaluation["objective_value"])
    ):
        raise SealedError("sealed protected evaluation is invalid")
    try:
        expected_objective = plugin.metric.canonical_objective(evaluation.get("metric_value"))
    except TargetPluginError as error:
        raise SealedError(str(error)) from error
    if not math.isclose(
        float(evaluation["objective_value"]), expected_objective, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SealedError("sealed evaluation objective transform is invalid")
    training = result["training_summary"]
    if (
        not isinstance(training, dict)
        or training.get("event") != "target_training_summary"
        or training.get("units_seen") != plan.token_budget
        or training.get("parameter_count") != inspection["parameter_count"]
    ):
        raise SealedError("sealed training summary is invalid")
    if plugin.rl is not None:
        expected_training_contract = {
            "budget_unit": plugin.rl.budget_unit,
            "training_paradigm": plugin.rl.paradigm.value,
        }
        if any(training.get(key) != value for key, value in expected_training_contract.items()):
            raise SealedError("sealed RL training changed the frozen RL contract")
        if (
            evaluation.get("training_paradigm") != plugin.rl.paradigm.value
            or evaluation.get("reward_source") != plugin.rl.reward_source.value
        ):
            raise SealedError("sealed RL evaluation changed the frozen reward contract")
        try:
            validate_training_diagnostics(plugin.rl, training)
            validate_evaluation_diagnostics(plugin.rl, evaluation)
        except RLContractError as error:
            raise SealedError(str(error)) from error
    processes = result["processes"]
    if not isinstance(processes, dict) or set(processes) != {
        "evaluation",
        "inspection",
        "training",
    }:
        raise SealedError("sealed process evidence is invalid")
    for process in processes.values():
        if (
            not isinstance(process, dict)
            or process.get("returncode") != 0
            or process.get("timed_out") is not False
            or process.get("cancelled") is not False
        ):
            raise SealedError("sealed process did not complete successfully")


def _execute_run(
    plan: SealedPlan,
    *,
    sealed_root: Path,
    repository: Path,
    public_data_root: Path,
    commit: str,
    seed: int,
) -> dict[str, Any]:
    run_root = _run_root(sealed_root, commit, seed)
    plugin = TargetPluginSpec.from_mapping(plan.plugin)
    with _single_worktree(
        repository,
        sealed_root / ".control" / "worktrees",
        commit,
        trainer_path=plugin.trainer_path,
    ) as worktree:
        trainer = resolve_repository_path(worktree, plugin.trainer_path)
        evaluator_path = resolve_repository_path(worktree, plugin.evaluator_path)
        if file_sha256(evaluator_path) != plan.evaluator_sha256:
            raise SealedError("protected evaluator differs in a frozen worktree")
        trainer_hash = file_sha256(trainer)
        contract = _run_contract(plan, commit, seed, trainer_hash)
        existing_contract = run_root / "contract.json"
        if existing_contract.is_file():
            if _read_canonical(existing_contract) != contract:
                raise SealedError(f"sealed run contract changed under {run_root}")
        else:
            _write_canonical(existing_contract, contract)
        retained = _load_retained_result(run_root, expected_contract=contract)
        if retained is not None:
            _validate_result(plan, retained, commit=commit, seed=seed)
            return retained

        environment = sanitized_environment(seed, repository)
        inspect_stdout = run_root / "inspect.jsonl"
        inspect_stderr = run_root / "inspect.stderr.log"
        common_values: dict[str, object] = {
            "device": plan.device,
            "evaluator": evaluator_path,
            "parameter_cap": plan.parameter_cap,
            "python": sys.executable,
            "repository_root": worktree,
            "trainer": trainer,
        }
        inspection_outcome = run_process(
            plugin.render_command("inspect", common_values),
            cwd=worktree,
            environment=environment,
            stdout_path=inspect_stdout,
            stderr_path=inspect_stderr,
            timeout_seconds=plan.timeout_seconds,
        )
        _require_success(
            inspection_outcome,
            phase="inspection",
            stderr_path=inspect_stderr,
        )
        inspection = _last_json(inspect_stdout, event="target_inspection")
        if inspection.get("trainer_sha256") != trainer_hash:
            raise SealedError("sealed inspection trainer hash mismatch")

        checkpoint = run_root / "checkpoint.pt"
        metrics = run_root / "metrics.jsonl"
        train_stdout = run_root / "training.stdout.log"
        train_stderr = run_root / "training.stderr.log"
        if checkpoint.is_file():
            checkpoint.unlink()
        training_command = plugin.render_command(
            "train",
            {
                **common_values,
                "batch_size": plan.batch_size,
                "checkpoint": checkpoint,
                "eval_batch_size": plan.eval_batch_size,
                "metrics": metrics,
                "public_data_root": public_data_root,
                "seed": seed,
                "stage": "sealed_final",
                "training_budget": plan.token_budget,
                "token_budget": plan.token_budget,
            },
        )
        training_outcome = run_process(
            training_command,
            cwd=worktree,
            environment=environment,
            stdout_path=train_stdout,
            stderr_path=train_stderr,
            timeout_seconds=plan.timeout_seconds,
        )
        _require_success(training_outcome, phase="training", stderr_path=train_stderr)
        training_summary = _last_json(metrics, event="target_training_summary")
        if training_summary.get("units_seen") != plan.token_budget:
            raise SealedError("sealed training did not consume its exact token budget")

        evaluation_stdout = run_root / "evaluation.jsonl"
        evaluation_stderr = run_root / "evaluation.stderr.log"
        evaluation_outcome = run_process(
            plugin.render_command(
                "evaluate",
                {
                    **common_values,
                    "batch_size": plan.eval_batch_size,
                    "checkpoint": checkpoint,
                    "data_root": plan.data_root,
                    "eval_tokens": 0,
                    "seed": seed,
                    "split": "sealed_final",
                    "stage": "sealed_final",
                },
            ),
            cwd=worktree,
            environment=environment,
            stdout_path=evaluation_stdout,
            stderr_path=evaluation_stderr,
            timeout_seconds=plan.timeout_seconds,
        )
        _require_success(
            evaluation_outcome,
            phase="evaluation",
            stderr_path=evaluation_stderr,
        )
        evaluation = _last_json(evaluation_stdout, event="target_evaluation")
        if (
            evaluation.get("checkpoint_sha256") != file_sha256(checkpoint)
            or evaluation.get("metric_name") != plugin.metric.name
            or evaluation.get("metric_direction") != plugin.metric.direction.value
        ):
            raise SealedError("sealed evaluation contract mismatch")
        evaluation = {
            **evaluation,
            "objective_value": plugin.metric.canonical_objective(evaluation.get("metric_value")),
        }
        if file_sha256(trainer) != trainer_hash or not _worktree_clean(worktree):
            raise SealedError("frozen worktree changed during sealed execution")
        result = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "contract": contract,
            "evaluation": evaluation,
            "inspection": inspection,
            "processes": {
                "evaluation": _process_payload(evaluation_outcome),
                "inspection": _process_payload(inspection_outcome),
                "training": _process_payload(training_outcome),
            },
            "training_summary": training_summary,
        }
        _validate_result(plan, result, commit=commit, seed=seed)
        _write_canonical(run_root / "result.json", result)
        (run_root / "result.sha256").write_text(
            hashlib.sha256(canonical_json_bytes(result)).hexdigest() + "\n",
            encoding="ascii",
        )
        return result


def run_sealed_plan(
    sealed_root: Path,
    *,
    repository_root: Path,
) -> list[dict[str, Any]]:
    root = sealed_root.expanduser().resolve()
    repository = repository_root.expanduser().resolve()
    plan = load_plan(root)
    with RepositoryLock(repository, campaign_id=plan.plan_id):
        _verify_frozen_inputs(repository, plan)
        public_data_root = Path(plan.public_data_root)
        results = []
        for commit, seed in _run_specs(plan):
            results.append(
                _execute_run(
                    plan,
                    sealed_root=root,
                    repository=repository,
                    public_data_root=public_data_root,
                    commit=commit,
                    seed=seed,
                )
            )
    return results


def load_results(sealed_root: Path, plan: SealedPlan) -> dict[tuple[str, int], dict[str, Any]]:
    root = sealed_root.expanduser().resolve()
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for commit, seed in _run_specs(plan):
        run_root = _run_root(root, commit, seed)
        result = _load_retained_result(run_root)
        if result is None:
            raise SealedError(f"sealed result is missing for {commit} seed {seed}")
        _validate_result(plan, result, commit=commit, seed=seed)
        results[(commit, seed)] = result
    return results


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise SealedError("cannot summarize missing or non-finite sealed values")
    count = len(values)
    deviation = statistics.stdev(values) if count > 1 else 0.0
    error = deviation / math.sqrt(count)
    mean = statistics.fmean(values)
    critical_values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
    }
    degrees = count - 1
    if degrees <= 20:
        critical = critical_values[degrees]
    elif degrees <= 25:
        critical = 2.086
    elif degrees <= 30:
        critical = 2.060
    else:
        critical = 1.96
    return {
        "count": count,
        "lower_95": mean - critical * error,
        "maximum": max(values),
        "mean": mean,
        "minimum": min(values),
        "sample_standard_deviation": deviation,
        "standard_error": error,
        "upper_95": mean + critical * error,
    }


def build_report(
    plan: SealedPlan,
    results: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    all_false = 0
    all_promotions = 0
    all_useful = 0
    for arm in plan.arms:
        generations = []
        for generation in arm.generations:
            values = [
                float(results[(generation.commit, seed)]["evaluation"]["objective_value"])
                for seed in plan.seeds
            ]
            generations.append(
                {
                    "commit": generation.commit,
                    "generation": generation.generation,
                    "sealed_objective": _summary(values),
                }
            )
        transitions = []
        false_promotions = 0
        useful_confirmed = 0
        for index in range(1, len(arm.generations)):
            current = arm.generations[index]
            previous = arm.generations[index - 1]
            gains = [
                float(results[(previous.commit, seed)]["evaluation"]["objective_value"])
                - float(results[(current.commit, seed)]["evaluation"]["objective_value"])
                for seed in plan.seeds
            ]
            gain = _summary(gains)
            minimum = float(current.minimum_useful_gain or 0.0)
            if float(gain["mean"]) <= 0.0:
                classification = "false_promotion"
                false_promotions += 1
            elif float(gain["lower_95"]) >= minimum:
                classification = "useful_confirmed"
                useful_confirmed += 1
            else:
                classification = "unconfirmed"
            transitions.append(
                {
                    "candidate_id": current.candidate_id,
                    "classification": classification,
                    "decision_id": current.decision_id,
                    "from_commit": previous.commit,
                    "objective_gain": gain,
                    "generation": current.generation,
                    "minimum_useful_gain": minimum,
                    "to_commit": current.commit,
                }
            )
        baseline = arm.generations[0]
        final = arm.generations[-1]
        final_gains = [
            float(results[(baseline.commit, seed)]["evaluation"]["objective_value"])
            - float(results[(final.commit, seed)]["evaluation"]["objective_value"])
            for seed in plan.seeds
        ]
        promotions = len(transitions)
        all_false += false_promotions
        all_promotions += promotions
        all_useful += useful_confirmed
        arms[arm.name] = {
            "campaign_compute": arm.compute,
            "compute_per_useful_confirmed": (
                None
                if useful_confirmed == 0
                else {
                    "accelerator_seconds": float(arm.compute["accelerator_seconds"])
                    / useful_confirmed,
                    "estimated_cost_usd": float(arm.compute["estimated_cost_usd"])
                    / useful_confirmed,
                    "training_tokens": float(arm.compute["training_tokens"]) / useful_confirmed,
                }
            ),
            "false_promotion_count": false_promotions,
            "false_promotion_rate": (None if promotions == 0 else false_promotions / promotions),
            "final_objective_gain": _summary(final_gains),
            "final_parent_commit": final.commit,
            "generations": generations,
            "promotion_count": promotions,
            "transitions": transitions,
            "useful_confirmed_count": useful_confirmed,
        }
    return {
        "arms": arms,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(plan.to_mapping())).hexdigest(),
        "plan_id": plan.plan_id,
        "schema_version": SEALED_SCHEMA_VERSION,
        "sealed_split": "sealed_final",
        "seeds": list(plan.seeds),
        "source_results": [
            {
                "commit": commit,
                "result_sha256": hashlib.sha256(canonical_json_bytes(result)).hexdigest(),
                "seed": seed,
            }
            for (commit, seed), result in sorted(results.items())
        ],
        "summary": {
            "false_promotion_count": all_false,
            "false_promotion_rate": (None if all_promotions == 0 else all_false / all_promotions),
            "promotion_count": all_promotions,
            "useful_confirmed_count": all_useful,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sealed autoresearch report",
        "",
        f"Plan: `{report['plan_id']}`  ",
        f"Seeds: `{', '.join(str(seed) for seed in report['seeds'])}`  ",
        "Split: `sealed_final`",
        "",
        "## Arm summary",
        "",
        (
            "| Arm | Promotions | Useful confirmed | False promotions | False rate "
            "| Final objective gain |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, arm in report["arms"].items():
        false_rate = arm["false_promotion_rate"]
        lines.append(
            f"| {name} | {arm['promotion_count']} | {arm['useful_confirmed_count']} | "
            f"{arm['false_promotion_count']} | "
            f"{'n/a' if false_rate is None else f'{false_rate:.3f}'} | "
            f"{arm['final_objective_gain']['mean']:.6f} |"
        )
    lines.extend(["", "## Promotion confirmation", ""])
    for name, arm in report["arms"].items():
        lines.extend([f"### {name}", ""])
        if not arm["transitions"]:
            lines.extend(["No patches were promoted in this arm.", ""])
            continue
        lines.extend(
            [
                (
                    "| Generation | Classification | Mean objective gain | 95% interval "
                    "| Minimum useful |"
                ),
                "| ---: | --- | ---: | --- | ---: |",
            ]
        )
        for transition in arm["transitions"]:
            gain = transition["objective_gain"]
            lines.append(
                f"| {transition['generation']} | {transition['classification']} | "
                f"{gain['mean']:.6f} | [{gain['lower_95']:.6f}, {gain['upper_95']:.6f}] | "
                f"{transition['minimum_useful_gain']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "A useful confirmation requires the paired sealed Student-t 95% lower bound to meet "
            "the patch's "
            "predeclared minimum useful gain. A false promotion has nonpositive mean sealed gain. "
            "Other positive outcomes are reported as unconfirmed rather than silently counted as "
            "successes.",
            "",
        ]
    )
    return "\n".join(lines)


def render_csv(report: dict[str, Any]) -> str:
    rows: list[list[Any]] = [
        [
            "arm",
            "generation",
            "classification",
            "mean_objective_gain",
            "lower_95",
            "upper_95",
            "minimum_useful_gain",
            "from_commit",
            "to_commit",
        ]
    ]
    for name, arm in report["arms"].items():
        for transition in arm["transitions"]:
            gain = transition["objective_gain"]
            rows.append(
                [
                    name,
                    transition["generation"],
                    transition["classification"],
                    gain["mean"],
                    gain["lower_95"],
                    gain["upper_95"],
                    transition["minimum_useful_gain"],
                    transition["from_commit"],
                    transition["to_commit"],
                ]
            )
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()


def render_svg(report: dict[str, Any]) -> str:
    width, height = 960, 540
    left, right, top, bottom = 88, 32, 48, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = [
        float(generation["sealed_objective"]["mean"])
        for arm in report["arms"].values()
        for generation in arm["generations"]
    ]
    if not points:
        raise SealedError("sealed graph has no points")
    low, high = min(points), max(points)
    padding = max((high - low) * 0.1, 0.001)
    low -= padding
    high += padding
    max_generation = max(
        int(generation["generation"])
        for arm in report["arms"].values()
        for generation in arm["generations"]
    )

    def x(value: int) -> float:
        return left + (
            plot_width / 2 if max_generation == 0 else value / max_generation * plot_width
        )

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="28" font-family="sans-serif" font-size="20" '
        'font-weight="600">Sealed objective by accepted generation</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#222"/>',
    ]
    for tick in range(5):
        value = low + (high - low) * tick / 4
        y_value = y(value)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y_value:.2f}" x2="{left + plot_width}" '
                f'y2="{y_value:.2f}" stroke="#dddddd"/>',
                f'<text x="{left - 10}" y="{y_value + 4:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{value:.4f}</text>',
            ]
        )
    for index, (name, arm) in enumerate(report["arms"].items()):
        color = colors[index % len(colors)]
        coordinates = " ".join(
            f"{x(int(item['generation'])):.2f},{y(float(item['sealed_objective']['mean'])):.2f}"
            for item in arm["generations"]
        )
        elements.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for item in arm["generations"]:
            elements.append(
                f'<circle cx="{x(int(item["generation"])):.2f}" '
                f'cy="{y(float(item["sealed_objective"]["mean"])):.2f}" r="4" fill="{color}"/>'
            )
        legend_y = top + index * 24
        elements.extend(
            [
                f'<line x1="{left + plot_width - 170}" y1="{legend_y}" '
                f'x2="{left + plot_width - 145}" y2="{legend_y}" stroke="{color}" '
                'stroke-width="3"/>',
                f'<text x="{left + plot_width - 137}" y="{legend_y + 4}" '
                f'font-family="sans-serif" font-size="12">{html.escape(name)}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" '
            'font-family="sans-serif" font-size="14">Accepted generation</text>',
            f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" '
            'transform="rotate(-90 20 258)" font-family="sans-serif" '
            'font-size="14">Canonical objective (lower is better)</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def write_report(sealed_root: Path) -> dict[str, Any]:
    root = sealed_root.expanduser().resolve()
    plan = load_plan(root)
    results = load_results(root, plan)
    report = build_report(plan, results)
    report_root = root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    _write_canonical(report_root / "report.json", report)
    (report_root / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (report_root / "promotions.csv").write_text(render_csv(report), encoding="utf-8")
    (report_root / "sealed-results.svg").write_text(render_svg(report), encoding="utf-8")
    artifacts = {}
    for path in sorted(report_root.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "artifacts": artifacts,
        "plan_id": plan.plan_id,
        "plan_sha256": report["plan_sha256"],
        "schema_version": SEALED_SCHEMA_VERSION,
    }
    _write_canonical(report_root / "manifest.json", manifest)
    return {"manifest": manifest, "report": report}


def sealed_status(sealed_root: Path) -> dict[str, Any]:
    root = sealed_root.expanduser().resolve()
    plan = load_plan(root)
    expected = _run_specs(plan)
    complete = 0
    for commit, seed in expected:
        if _load_retained_result(_run_root(root, commit, seed)) is not None:
            complete += 1
    report_manifest = root / "report" / "manifest.json"
    return {
        "complete_runs": complete,
        "expected_runs": len(expected),
        "plan_id": plan.plan_id,
        "report_ready": report_manifest.is_file(),
        "schema_version": SEALED_SCHEMA_VERSION,
    }


def _arm_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not _valid_name(name) or not path:
        raise argparse.ArgumentTypeError("arm must use NAME=LEDGER_PATH")
    return name, Path(path).expanduser()


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze and evaluate accepted sealed lineages.")
    parser.add_argument("--repository-root", type=_path, default=Path.cwd())
    parser.add_argument("--sealed-root", type=_path, default=DEFAULT_SEALED_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="freeze ledgers without opening protected data")
    plan.add_argument("--arm", type=_arm_spec, action="append", required=True)
    plan.add_argument("--seeds", type=int, nargs="+", required=True)
    plan.add_argument("--assignment-seed", type=int, required=True)
    plan.add_argument("--token-budget", type=int, required=True)
    plan.add_argument("--batch-size", type=int, default=64)
    plan.add_argument("--eval-batch-size", type=int, default=64)
    plan.add_argument("--timeout-seconds", type=int, default=7_200)
    plan.add_argument("--target-config", type=_path, required=True)
    commands.add_parser("run", help="execute every missing frozen sealed run")
    commands.add_parser("report", help="build reports from retained sealed results")
    commands.add_parser("status", help="show sealed run and report completion")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            plan = create_plan(
                repository_root=args.repository_root,
                sealed_root=args.sealed_root,
                arm_ledgers=tuple(args.arm),
                seeds=tuple(args.seeds),
                assignment_seed=args.assignment_seed,
                token_budget=args.token_budget,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                timeout_seconds=args.timeout_seconds,
                target_config_path=args.target_config,
            )
            payload = {"plan": plan.to_mapping(), "status": sealed_status(args.sealed_root)}
        elif args.command == "run":
            run_sealed_plan(args.sealed_root, repository_root=args.repository_root)
            payload = write_report(args.sealed_root)
        elif args.command == "report":
            payload = write_report(args.sealed_root)
        else:
            payload = sealed_status(args.sealed_root)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        LedgerError,
        OSError,
        RunStateError,
        SealedError,
        TargetError,
        TargetPluginError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
