"""Matched three-arm autoresearch study coordination."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from autodidact.controller import (
    ControllerError,
    DecisionMode,
    PatchRCTPolicy,
    synchronize_accepted_ref,
)
from autodidact.data.config import default_output_root
from autodidact.data.integrity import canonical_json_bytes
from autodidact.ledger import ExperimentLedger, LedgerError
from autodidact.orchestrator import (
    AutonomousResearchOrchestrator,
    OrchestratorConfig,
    OrchestratorError,
)
from autodidact.records import DecisionRecord, DecisionVerdict, LineageRecord
from autodidact.researcher import ResearcherConfig, ResearcherError
from autodidact.researcher_providers import build_researcher_adapter
from autodidact.reward import RewardError
from autodidact.runner import RunnerError
from autodidact.runstate import CampaignLimits, CampaignStore, RunStateError
from autodidact.target import TargetConfig, TargetError

STUDY_SCHEMA_VERSION = 1
DEFAULT_STUDY_ROOT = Path("artifacts/studies/three-arm")
_STUDY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_KEYS = frozenset(
    {
        "arm_order",
        "assignment_seed",
        "data_root",
        "device",
        "estimated_accelerator_hour_usd",
        "initial_parent_commit",
        "limits",
        "policy_sha256",
        "program_sha256",
        "program_path",
        "researcher_config_path",
        "researcher_config_sha256",
        "reward_calibration_labels",
        "schema_version",
        "study_id",
        "target_config_path",
        "target_config_sha256",
    }
)
_LIMIT_KEYS = frozenset(
    {
        "max_compute_seconds",
        "max_proposals",
        "max_researcher_tokens",
        "max_training_tokens",
        "max_wall_seconds",
    }
)


class StudyError(RuntimeError):
    """Raised when a three-arm study contract is invalid."""


class StudyArm(StrEnum):
    GREEDY = "greedy"
    PATCH_RCT = "patch_rct"
    PATCH_RCT_BAYESIAN = "patch_rct_bayesian"


@dataclass(frozen=True, slots=True)
class StudyLimits:
    max_proposals: int
    max_wall_seconds: float
    max_researcher_tokens: int
    max_training_tokens: int
    max_compute_seconds: float

    def __post_init__(self) -> None:
        for name in ("max_proposals", "max_researcher_tokens", "max_training_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise StudyError(f"{name} must be a positive integer")
        for name in ("max_wall_seconds", "max_compute_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise StudyError(f"{name} must be finite and positive")

    @classmethod
    def from_mapping(cls, value: Any) -> StudyLimits:
        if not isinstance(value, dict) or frozenset(value) != _LIMIT_KEYS:
            raise StudyError("study limits have an invalid schema")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class StudyManifest:
    study_id: str
    initial_parent_commit: str
    arm_order: tuple[StudyArm, ...]
    assignment_seed: int
    limits: StudyLimits
    reward_calibration_labels: int
    researcher_config_path: str
    researcher_config_sha256: str
    target_config_path: str | None
    target_config_sha256: str | None
    program_path: str
    program_sha256: str
    data_root: str
    device: str
    estimated_accelerator_hour_usd: float | None
    policy_sha256: tuple[tuple[StudyArm, str], ...]

    def __post_init__(self) -> None:
        if not _STUDY_ID_PATTERN.fullmatch(self.study_id):
            raise StudyError("study_id must be a portable lowercase identifier")
        if not _COMMIT_PATTERN.fullmatch(self.initial_parent_commit):
            raise StudyError("initial_parent_commit must be a full lowercase Git commit")
        if set(self.arm_order) != set(StudyArm) or len(self.arm_order) != len(StudyArm):
            raise StudyError("arm_order must contain every study arm exactly once")
        if type(self.assignment_seed) is not int or not 0 <= self.assignment_seed <= 2**32 - 1:
            raise StudyError("assignment_seed must be a 32-bit nonnegative integer")
        if (
            type(self.reward_calibration_labels) is not int
            or self.reward_calibration_labels <= 0
            or self.reward_calibration_labels > self.limits.max_proposals
        ):
            raise StudyError("reward calibration labels must fit inside the proposal budget")
        for name in ("researcher_config_path", "program_path", "data_root", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise StudyError(f"{name} must be nonempty portable text")
        if self.target_config_path is not None and (
            not isinstance(self.target_config_path, str)
            or not self.target_config_path.strip()
            or "\x00" in self.target_config_path
        ):
            raise StudyError("target_config_path must be null or nonempty portable text")
        if (self.target_config_path is None) != (self.target_config_sha256 is None):
            raise StudyError("target config path and hash must be set together")
        for name in ("researcher_config_sha256", "program_sha256"):
            if not _SHA256_PATTERN.fullmatch(getattr(self, name)):
                raise StudyError(f"{name} must be a lowercase SHA-256 digest")
        if self.target_config_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            self.target_config_sha256
        ):
            raise StudyError("target_config_sha256 must be a lowercase SHA-256 digest")
        price = self.estimated_accelerator_hour_usd
        if price is not None and (
            not isinstance(price, (int, float)) or not math.isfinite(price) or price < 0
        ):
            raise StudyError("estimated accelerator price must be finite and nonnegative")
        policies = dict(self.policy_sha256)
        if set(policies) != set(StudyArm) or len(policies) != len(self.policy_sha256):
            raise StudyError("policy_sha256 must identify every arm exactly once")
        if any(not _SHA256_PATTERN.fullmatch(value) for value in policies.values()):
            raise StudyError("study policy hashes must be lowercase SHA-256 digests")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "arm_order": [arm.value for arm in self.arm_order],
            "assignment_seed": self.assignment_seed,
            "data_root": self.data_root,
            "device": self.device,
            "estimated_accelerator_hour_usd": self.estimated_accelerator_hour_usd,
            "initial_parent_commit": self.initial_parent_commit,
            "limits": asdict(self.limits),
            "policy_sha256": {arm.value: digest for arm, digest in self.policy_sha256},
            "program_path": self.program_path,
            "program_sha256": self.program_sha256,
            "researcher_config_path": self.researcher_config_path,
            "researcher_config_sha256": self.researcher_config_sha256,
            "reward_calibration_labels": self.reward_calibration_labels,
            "schema_version": STUDY_SCHEMA_VERSION,
            "study_id": self.study_id,
            "target_config_path": self.target_config_path,
            "target_config_sha256": self.target_config_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Any) -> StudyManifest:
        if not isinstance(value, dict) or frozenset(value) != _MANIFEST_KEYS:
            raise StudyError("study manifest has an invalid schema")
        if value["schema_version"] != STUDY_SCHEMA_VERSION:
            raise StudyError("study manifest schema version is unsupported")
        try:
            arm_order = tuple(StudyArm(item) for item in value["arm_order"])
            policy_sha256 = tuple(
                (StudyArm(arm), digest) for arm, digest in sorted(value["policy_sha256"].items())
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise StudyError("study manifest arm values are invalid") from error
        return cls(
            study_id=value["study_id"],
            initial_parent_commit=value["initial_parent_commit"],
            arm_order=arm_order,
            assignment_seed=value["assignment_seed"],
            limits=StudyLimits.from_mapping(value["limits"]),
            reward_calibration_labels=value["reward_calibration_labels"],
            researcher_config_path=value["researcher_config_path"],
            researcher_config_sha256=value["researcher_config_sha256"],
            target_config_path=value["target_config_path"],
            target_config_sha256=value["target_config_sha256"],
            program_path=value["program_path"],
            program_sha256=value["program_sha256"],
            data_root=value["data_root"],
            device=value["device"],
            estimated_accelerator_hour_usd=value["estimated_accelerator_hour_usd"],
            policy_sha256=policy_sha256,
        )


def _arm_order(study_id: str, assignment_seed: int) -> tuple[StudyArm, ...]:
    def key(arm: StudyArm) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "arm": arm.value,
                    "assignment_seed": assignment_seed,
                    "domain": "autodidact-three-arm-order-v1",
                    "study_id": study_id,
                }
            )
        ).hexdigest()

    return tuple(sorted(StudyArm, key=key))


def _accepted_ref(study_id: str, arm: StudyArm) -> str:
    return f"refs/autodidact/studies/{study_id}/{arm.value}/accepted"


def _campaign_id(study_id: str, arm: StudyArm) -> str:
    return f"{study_id}-{arm.value.replace('_', '-')}"


def _arm_root(study_root: Path, arm: StudyArm) -> Path:
    return study_root / "arms" / arm.value


def _resolve(repository_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _git_head(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or not _COMMIT_PATTERN.fullmatch(head):
        raise StudyError(f"cannot resolve repository HEAD: {completed.stderr.strip()}")
    return head


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise StudyError(f"cannot read pinned study input {path}: {error}") from error


def _verify_pinned_inputs(repository: Path, manifest: StudyManifest) -> None:
    researcher_path = _resolve(repository, manifest.researcher_config_path)
    program_path = _resolve(repository, manifest.program_path)
    if _file_sha256(researcher_path) != manifest.researcher_config_sha256:
        raise StudyError("researcher configuration changed after study initialization")
    if _file_sha256(program_path) != manifest.program_sha256:
        raise StudyError("research program changed after study initialization")
    if manifest.target_config_path is not None:
        target_path = _resolve(repository, manifest.target_config_path)
        if _file_sha256(target_path) != manifest.target_config_sha256:
            raise StudyError("target configuration changed after study initialization")


def _arm_policy(
    arm: StudyArm,
    *,
    max_parameter_count: int,
    minimum_reward_labels: int,
) -> PatchRCTPolicy:
    return PatchRCTPolicy(
        decision_mode=(DecisionMode.GREEDY if arm is StudyArm.GREEDY else DecisionMode.PATCH_RCT),
        max_parameter_count=max_parameter_count,
        use_downstream_allocation=arm is StudyArm.PATCH_RCT_BAYESIAN,
        minimum_downstream_labels=minimum_reward_labels,
    )


def _write_manifest(study_root: Path, manifest: StudyManifest) -> None:
    payload = canonical_json_bytes(manifest.to_mapping())
    (study_root / "manifest.json").write_bytes(payload + b"\n")
    (study_root / "manifest.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
    )
    (study_root / ".autodidact-study").write_text(
        f"schema_version={STUDY_SCHEMA_VERSION}\n",
        encoding="ascii",
    )


def load_manifest(study_root: Path) -> StudyManifest:
    root = study_root.expanduser().resolve()
    if not (root / ".autodidact-study").is_file():
        raise StudyError("study root is missing its ownership marker")
    try:
        raw = (root / "manifest.json").read_bytes()
        expected = (root / "manifest.sha256").read_text(encoding="ascii").strip()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise StudyError(f"cannot read study manifest: {error}") from error
    canonical = canonical_json_bytes(value)
    if raw != canonical + b"\n" or hashlib.sha256(canonical).hexdigest() != expected:
        raise StudyError("study manifest is not canonical or its digest is invalid")
    return StudyManifest.from_mapping(value)


def initialize_study(
    *,
    study_root: Path,
    repository_root: Path,
    study_id: str,
    assignment_seed: int,
    limits: StudyLimits,
    reward_calibration_labels: int,
    researcher_config_path: str,
    target_config_path: str | None,
    program_path: str,
    data_root: str,
    device: str,
    estimated_accelerator_hour_usd: float | None,
) -> StudyManifest:
    repository = repository_root.expanduser().resolve()
    root = study_root.expanduser().resolve()
    if root.exists():
        raise StudyError(f"study root already exists: {root}")
    researcher_path = _resolve(repository, researcher_config_path)
    program = _resolve(repository, program_path)
    ResearcherConfig.from_path(researcher_path)
    if not program.is_file():
        raise StudyError(f"research program does not exist: {program}")
    target_path = None if target_config_path is None else _resolve(repository, target_config_path)
    target = None if target_path is None else TargetConfig.from_path(target_path)
    max_parameters = 1_050_000 if target is None else target.max_parameter_count
    policies = tuple(
        (
            arm,
            _arm_policy(
                arm,
                max_parameter_count=max_parameters,
                minimum_reward_labels=reward_calibration_labels,
            ).sha256(),
        )
        for arm in StudyArm
    )
    parent = _git_head(repository)
    manifest = StudyManifest(
        study_id=study_id,
        initial_parent_commit=parent,
        arm_order=_arm_order(study_id, assignment_seed),
        assignment_seed=assignment_seed,
        limits=limits,
        reward_calibration_labels=reward_calibration_labels,
        researcher_config_path=researcher_config_path,
        researcher_config_sha256=_file_sha256(researcher_path),
        target_config_path=target_config_path,
        target_config_sha256=(None if target_path is None else _file_sha256(target_path)),
        program_path=program_path,
        program_sha256=_file_sha256(program),
        data_root=data_root,
        device=device,
        estimated_accelerator_hour_usd=estimated_accelerator_hour_usd,
        policy_sha256=policies,
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name(f".{root.name}.staging")
    if staging.exists():
        raise StudyError(f"stale study staging path requires inspection: {staging}")
    staging.mkdir()
    try:
        for arm in StudyArm:
            arm_root = _arm_root(staging, arm)
            arm_root.mkdir(parents=True)
            ExperimentLedger.create(
                arm_root / "ledger.sqlite3",
                initial_parent_commit=parent,
            )
            CampaignStore.create(
                arm_root / "campaign.sqlite3",
                campaign_id=_campaign_id(study_id, arm),
                initial_parent_commit=parent,
                limits=CampaignLimits(
                    max_proposals=limits.max_proposals,
                    max_wall_seconds=limits.max_wall_seconds,
                    max_researcher_tokens=limits.max_researcher_tokens,
                    max_training_tokens=limits.max_training_tokens,
                    max_compute_seconds=limits.max_compute_seconds,
                    reward_calibration_labels=(
                        reward_calibration_labels if arm is StudyArm.PATCH_RCT_BAYESIAN else 0
                    ),
                    use_downstream_allocation=arm is StudyArm.PATCH_RCT_BAYESIAN,
                ),
            )
        _write_manifest(staging, manifest)
        os.replace(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for arm in StudyArm:
        synchronize_accepted_ref(repository, _accepted_ref(study_id, arm), parent)
    return manifest


def study_status(
    study_root: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    root = study_root.expanduser().resolve()
    manifest = load_manifest(root)
    if repository_root is not None:
        _verify_pinned_inputs(repository_root.expanduser().resolve(), manifest)
    arms: dict[str, Any] = {}
    for arm in StudyArm:
        arm_root = _arm_root(root, arm)
        state = CampaignStore.open(arm_root / "campaign.sqlite3").snapshot()
        ledger = ExperimentLedger.open(arm_root / "ledger.sqlite3", read_only=True)
        records = [event.record for event in ledger.events()]
        decisions = [record for record in records if isinstance(record, DecisionRecord)]
        arms[arm.value] = {
            "accepted_ref": _accepted_ref(manifest.study_id, arm),
            "campaign": asdict(state),
            "decision_count": len(decisions),
            "ledger": ledger.summary(),
            "lineage_count": sum(isinstance(record, LineageRecord) for record in records),
            "promotion_count": sum(
                isinstance(record, DecisionRecord) and record.verdict is DecisionVerdict.PROMOTE
                for record in records
            ),
            "policy_sha256": dict(manifest.policy_sha256)[arm],
        }
    return {
        "arm_order": [arm.value for arm in manifest.arm_order],
        "arms": arms,
        "initial_parent_commit": manifest.initial_parent_commit,
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": manifest.study_id,
    }


def run_study(
    study_root: Path,
    *,
    repository_root: Path,
    max_new_proposals_per_arm: int | None,
) -> dict[str, Any]:
    if max_new_proposals_per_arm is not None and max_new_proposals_per_arm <= 0:
        raise StudyError("max_new_proposals_per_arm must be positive")
    root = study_root.expanduser().resolve()
    repository = repository_root.expanduser().resolve()
    manifest = load_manifest(root)
    _verify_pinned_inputs(repository, manifest)
    target = (
        None
        if manifest.target_config_path is None
        else TargetConfig.from_path(_resolve(repository, manifest.target_config_path))
    )
    max_parameters = 1_050_000 if target is None else target.max_parameter_count
    expected_policies = dict(manifest.policy_sha256)
    outcomes: dict[str, Any] = {}
    for arm in manifest.arm_order:
        arm_root = _arm_root(root, arm)
        policy = _arm_policy(
            arm,
            max_parameter_count=max_parameters,
            minimum_reward_labels=manifest.reward_calibration_labels,
        )
        if policy.sha256() != expected_policies[arm]:
            raise StudyError(f"{arm.value} runtime policy differs from the study manifest")
        state = CampaignStore.open(arm_root / "campaign.sqlite3")
        ledger = ExperimentLedger.open(arm_root / "ledger.sqlite3", read_only=False)
        researcher_config = ResearcherConfig.from_path(
            _resolve(repository, manifest.researcher_config_path)
        )
        data_root = _resolve(repository, manifest.data_root)
        device = manifest.device
        target_name = "configured study target"
        execution_location = "local"
        estimated_cost = manifest.estimated_accelerator_hour_usd
        if target is not None:
            data_root = target.resolved_data_root(repository)
            device = target.device
            target_name = target.name
            execution_location = target.execution_location.value
            estimated_cost = target.estimated_accelerator_hour_usd
        orchestrator = AutonomousResearchOrchestrator(
            OrchestratorConfig(
                repository_root=repository,
                ledger_path=arm_root / "ledger.sqlite3",
                data_root=data_root,
                output_root=arm_root / "experiments",
                workspace_root=arm_root / "workspaces",
                researcher_artifact_root=arm_root / "researcher",
                reward_root=arm_root / "reward",
                program_path=_resolve(repository, manifest.program_path),
                device=device,
                minimum_reward_labels=manifest.reward_calibration_labels,
                estimated_accelerator_hour_usd=estimated_cost,
                max_parameter_count=max_parameters,
                target_name=target_name,
                target_execution_location=execution_location,
                accepted_ref=_accepted_ref(manifest.study_id, arm),
            ),
            state=state,
            ledger=ledger,
            researcher=build_researcher_adapter(researcher_config),
            policy=policy,
        )
        outcomes[arm.value] = orchestrator.run(max_new_proposals=max_new_proposals_per_arm)
    return {
        "arm_order": [arm.value for arm in manifest.arm_order],
        "outcomes": outcomes,
        "status": study_status(root, repository_root=repository),
        "study_id": manifest.study_id,
    }


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a matched three-arm autoresearch study.")
    parser.add_argument("--repository-root", type=_path, default=Path.cwd())
    parser.add_argument("--study-root", type=_path, default=DEFAULT_STUDY_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--study-id", required=True)
    initialize.add_argument("--assignment-seed", type=int, required=True)
    initialize.add_argument("--researcher-config", required=True)
    initialize.add_argument("--target-config")
    initialize.add_argument("--program", default="program.md")
    initialize.add_argument("--data-root", default=str(default_output_root()))
    initialize.add_argument("--device", default="auto")
    initialize.add_argument("--estimated-accelerator-hour-usd", type=float)
    initialize.add_argument("--max-proposals", type=int, required=True)
    initialize.add_argument("--max-wall-seconds", type=float, required=True)
    initialize.add_argument("--max-researcher-tokens", type=int, required=True)
    initialize.add_argument("--max-training-tokens", type=int, required=True)
    initialize.add_argument("--max-compute-seconds", type=float, required=True)
    initialize.add_argument("--reward-calibration-labels", type=int, default=40)
    run = commands.add_parser("run")
    run.add_argument("--max-new-proposals-per-arm", type=int)
    commands.add_parser("status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "initialize":
            manifest = initialize_study(
                study_root=args.study_root,
                repository_root=args.repository_root,
                study_id=args.study_id,
                assignment_seed=args.assignment_seed,
                limits=StudyLimits(
                    max_proposals=args.max_proposals,
                    max_wall_seconds=args.max_wall_seconds,
                    max_researcher_tokens=args.max_researcher_tokens,
                    max_training_tokens=args.max_training_tokens,
                    max_compute_seconds=args.max_compute_seconds,
                ),
                reward_calibration_labels=args.reward_calibration_labels,
                researcher_config_path=args.researcher_config,
                target_config_path=args.target_config,
                program_path=args.program,
                data_root=args.data_root,
                device=args.device,
                estimated_accelerator_hour_usd=args.estimated_accelerator_hour_usd,
            )
            payload = {
                "manifest": manifest.to_mapping(),
                "status": study_status(
                    args.study_root,
                    repository_root=args.repository_root,
                ),
            }
        elif args.command == "run":
            payload = run_study(
                args.study_root,
                repository_root=args.repository_root,
                max_new_proposals_per_arm=args.max_new_proposals_per_arm,
            )
        else:
            payload = study_status(
                args.study_root,
                repository_root=args.repository_root,
            )
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        ControllerError,
        LedgerError,
        OSError,
        OrchestratorError,
        ResearcherError,
        RewardError,
        RunnerError,
        RunStateError,
        StudyError,
        TargetError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
