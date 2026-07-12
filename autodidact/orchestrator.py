"""Autonomous proposal, paired-experiment, decision, and lineage orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from autodidact.controller import (
    DEFAULT_ACCEPTED_REF,
    PatchRCTController,
    PatchRCTPolicy,
    synchronize_accepted_ref,
)
from autodidact.data.config import default_output_root
from autodidact.data.integrity import canonical_json_bytes
from autodidact.ledger import ExperimentLedger, LedgerError, WriterRole
from autodidact.records import (
    CandidateRecord,
    ComputeRecord,
    DecisionRecord,
    DecisionVerdict,
    EffectEstimate,
    ExperimentStage,
    PairedResult,
    PatchProposal,
    RunResult,
    RunStatus,
    TrialSchedule,
    TrialSpec,
    record_to_envelope,
)
from autodidact.researcher import (
    CommandResearcherAdapter,
    ProposalDraft,
    ResearchAttempt,
    ResearcherAdapter,
    ResearcherConfig,
    ResearcherError,
    ResearchRequest,
    load_research_attempt,
)
from autodidact.reward import (
    DEFAULT_MINIMUM_LABELS,
    RewardError,
    build_downstream_prediction,
    build_full_budget_label,
    calibrate_model,
    extract_learning_curve_features,
    load_features,
    load_labels,
    load_model,
    save_model,
    store_full_budget_label,
    store_learning_curve_features,
)
from autodidact.runner import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_OUTPUT_ROOT,
    ExperimentRequest,
    PairedExperimentRunner,
    RunnerError,
    validate_candidate_patch,
)
from autodidact.runstate import (
    DEFAULT_STATE_PATH,
    BudgetAmount,
    BudgetExceeded,
    CampaignLimits,
    CampaignSnapshot,
    CampaignStatus,
    CampaignStore,
    ClaimDisposition,
    OperationStatus,
    RepositoryLock,
    ReservationStatus,
    RunStateError,
)

ORCHESTRATOR_SCHEMA_VERSION = 1
DEFAULT_PROGRAM_PATH = Path("program.md")
DEFAULT_RESEARCHER_CONFIG_PATH = Path("artifacts/control/researcher.json")
DEFAULT_RESEARCHER_ARTIFACT_ROOT = Path("artifacts/researcher")
DEFAULT_REWARD_ROOT = Path("artifacts/reward")
DEFAULT_WORKSPACE_ROOT = Path("artifacts/control/workspaces")
DEFAULT_RESEARCHER_TOKEN_ALLOWANCE = 50_000
DEFAULT_PREVIOUS_RESULT_LIMIT = 20
MAX_CONTROL_TRANSITIONS = 100


class OrchestratorError(RuntimeError):
    """Raised when an autonomous campaign cannot safely continue."""


class _CampaignStopped(OrchestratorError):
    """Internal clean-boundary control-flow signal."""


class ExperimentRunner(Protocol):
    def register_candidate(self) -> CandidateRecord: ...

    def run(self) -> dict[str, Any]: ...


RunnerFactory = Callable[[ExperimentRequest], ExperimentRunner]


def _default_runner_factory(request: ExperimentRequest) -> ExperimentRunner:
    return PairedExperimentRunner(request)


def _stable_id(prefix: str, *parts: object) -> str:
    payload = canonical_json_bytes([str(part) for part in parts])
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise OrchestratorError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _reward_calibration_status(
    target: int,
    *,
    features_path: Path,
    labels_path: Path,
) -> dict[str, Any]:
    if features_path.is_file() and labels_path.is_file():
        feature_candidates = {feature.candidate_id for feature in load_features(features_path)}
        label_candidates = {label.candidate_id for label in load_labels(labels_path)}
        completed = len(feature_candidates.intersection(label_candidates))
    else:
        completed = 0
    return {
        "active": target > 0 and completed < target,
        "completed_labels": completed,
        "remaining_labels": max(0, target - completed),
        "target_labels": target,
    }


def _downstream_allocation_status(
    enabled: bool,
    *,
    calibration: dict[str, Any],
    minimum_labels: int,
    model_path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": enabled,
        "minimum_labels": minimum_labels,
        "ready": False,
    }
    if not enabled:
        result["reason"] = "disabled"
        return result
    if calibration["active"]:
        result.update(
            {
                "label_count": calibration["completed_labels"],
                "reason": "reward calibration is still collecting labels",
            }
        )
        return result
    if not model_path.is_file():
        result.update({"label_count": 0, "reason": "reward model is missing"})
        return result
    try:
        model = load_model(model_path)
    except (OSError, RewardError):
        result.update({"label_count": 0, "reason": "reward model cannot be verified"})
        return result
    ready = model.calibrated and model.label_count >= minimum_labels
    result.update(
        {
            "label_count": model.label_count,
            "model_sha256": model.sha256(),
            "ready": ready,
            "reason": "ready" if ready else "reward model is not sufficiently calibrated",
        }
    )
    return result


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    repository_root: Path
    ledger_path: Path = DEFAULT_LEDGER_PATH
    data_root: Path = default_output_root()
    output_root: Path = DEFAULT_OUTPUT_ROOT
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT
    researcher_artifact_root: Path = DEFAULT_RESEARCHER_ARTIFACT_ROOT
    reward_root: Path = DEFAULT_REWARD_ROOT
    program_path: Path = DEFAULT_PROGRAM_PATH
    device: str = "auto"
    researcher_token_allowance: int = DEFAULT_RESEARCHER_TOKEN_ALLOWANCE
    previous_result_limit: int = DEFAULT_PREVIOUS_RESULT_LIMIT
    minimum_reward_labels: int = DEFAULT_MINIMUM_LABELS
    estimated_accelerator_hour_usd: float | None = None

    def __post_init__(self) -> None:
        if type(self.researcher_token_allowance) is not int or self.researcher_token_allowance <= 0:
            raise OrchestratorError("researcher_token_allowance must be positive")
        if type(self.previous_result_limit) is not int or self.previous_result_limit <= 0:
            raise OrchestratorError("previous_result_limit must be positive")
        if type(self.minimum_reward_labels) is not int or self.minimum_reward_labels <= 0:
            raise OrchestratorError("minimum_reward_labels must be positive")
        if self.estimated_accelerator_hour_usd is not None and (
            not math.isfinite(self.estimated_accelerator_hour_usd)
            or self.estimated_accelerator_hour_usd < 0.0
        ):
            raise OrchestratorError("estimated accelerator price must be finite and nonnegative")

    @property
    def features_path(self) -> Path:
        return self.reward_root / "features.jsonl"

    @property
    def labels_path(self) -> Path:
        return self.reward_root / "full-labels.jsonl"

    @property
    def model_path(self) -> Path:
        return self.reward_root / "model.json"


@dataclass(frozen=True, slots=True)
class ProposalOutcome:
    action: str
    proposal_id: str | None
    candidate_id: str | None
    parent_commit: str
    resulting_parent_commit: str
    reason: str


class AutonomousResearchOrchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        state: CampaignStore,
        ledger: ExperimentLedger,
        researcher: ResearcherAdapter,
        policy: PatchRCTPolicy | None = None,
        runner_factory: RunnerFactory = _default_runner_factory,
    ) -> None:
        self.config = config
        self.state = state
        self.ledger = ledger
        self.researcher = researcher
        self.policy = policy or PatchRCTPolicy()
        self.runner_factory = runner_factory
        self.repository_root = config.repository_root.expanduser().resolve()

    def _calibration_target(self) -> int:
        return self.state.snapshot().limits.reward_calibration_labels

    def calibration_status(self) -> dict[str, Any]:
        target = self._calibration_target()
        return _reward_calibration_status(
            target,
            features_path=self.config.features_path,
            labels_path=self.config.labels_path,
        )

    def downstream_allocation_status(self) -> dict[str, Any]:
        enabled = self.state.snapshot().limits.use_downstream_allocation
        minimum_labels = self._calibration_target() or self.config.minimum_reward_labels
        calibration = self.calibration_status()
        return _downstream_allocation_status(
            enabled,
            calibration=calibration,
            minimum_labels=minimum_labels,
            model_path=self.config.model_path,
        )

    def _controller_for_candidate(self, candidate: CandidateRecord) -> PatchRCTController:
        minimum_labels = self._calibration_target() or self.config.minimum_reward_labels
        standard_policy = replace(
            self.policy,
            force_full_evaluation=False,
            use_downstream_allocation=False,
            minimum_downstream_labels=minimum_labels,
        )
        calibration_policy = replace(
            self.policy,
            force_full_evaluation=True,
            use_downstream_allocation=False,
            minimum_downstream_labels=minimum_labels,
        )
        allocation_policy = replace(
            self.policy,
            force_full_evaluation=False,
            use_downstream_allocation=True,
            minimum_downstream_labels=minimum_labels,
        )
        schedules = self._records(TrialSchedule, candidate.candidate_id)
        if schedules:
            policy_hashes = {schedule.policy_sha256 for schedule in schedules}
            if len(policy_hashes) != 1:
                raise OrchestratorError("candidate schedules use inconsistent controller policies")
            policy_hash = next(iter(policy_hashes))
            if policy_hash == calibration_policy.sha256():
                selected = calibration_policy
            elif policy_hash == allocation_policy.sha256():
                selected = allocation_policy
            elif policy_hash == standard_policy.sha256():
                selected = standard_policy
            else:
                raise OrchestratorError("candidate schedule policy differs from campaign policy")
        else:
            status = self.calibration_status()
            if status["active"]:
                selected = calibration_policy
            elif self.state.snapshot().limits.use_downstream_allocation:
                allocation_status = self.downstream_allocation_status()
                if not allocation_status["ready"]:
                    raise OrchestratorError(str(allocation_status["reason"]))
                selected = allocation_policy
            else:
                selected = standard_policy
        return PatchRCTController(
            self.ledger,
            policy=selected,
            repository_root=self.repository_root,
        )

    def _assert_consistent_parent(self) -> None:
        snapshot = self.state.snapshot()
        ledger_parent = self.ledger.current_parent()
        if snapshot.accepted_parent_commit != ledger_parent:
            active_candidate = self._candidate(snapshot.active_candidate_id)
            promoted_during_crash = (
                active_candidate is not None
                and active_candidate.parent_commit == snapshot.accepted_parent_commit
                and active_candidate.candidate_commit == ledger_parent
            )
            if not promoted_during_crash:
                raise OrchestratorError("campaign state and evidence ledger parents differ")
            self.state.set_progress(
                phase="ready",
                accepted_parent_commit=ledger_parent,
                generation=snapshot.generation + 1,
                active_proposal_id=None,
                active_candidate_id=None,
            )
        synchronize_accepted_ref(
            self.repository_root,
            DEFAULT_ACCEPTED_REF,
            self.ledger.current_parent(),
        )

    def _record_snapshot(self) -> tuple[Any, ...]:
        return tuple(event.record for event in self.ledger.events())

    def _records(
        self,
        expected_type: type[Any],
        candidate_id: str | None = None,
        *,
        source: Sequence[Any] | None = None,
    ) -> list[Any]:
        records = []
        evidence = self._record_snapshot() if source is None else source
        for record in evidence:
            if not isinstance(record, expected_type):
                continue
            if candidate_id is not None and getattr(record, "candidate_id", None) != candidate_id:
                continue
            records.append(record)
        return records

    def _candidate(self, candidate_id: str | None) -> CandidateRecord | None:
        if candidate_id is None:
            return None
        event = self.ledger.get(candidate_id)
        if not isinstance(event.record, CandidateRecord):
            raise OrchestratorError("active candidate ID does not identify a candidate")
        return event.record

    def _candidate_for_proposal(self, proposal_id: str) -> CandidateRecord | None:
        evidence = self._record_snapshot()
        candidates = [
            record
            for record in self._records(CandidateRecord, source=evidence)
            if record.proposal_id == proposal_id
        ]
        if len(candidates) > 1:
            raise OrchestratorError("proposal has multiple candidate records")
        return candidates[0] if candidates else None

    def _previous_results(self) -> tuple[dict[str, Any], ...]:
        evidence = self._record_snapshot()
        proposals = {
            record.proposal_id: record for record in self._records(PatchProposal, source=evidence)
        }
        candidates = {
            record.candidate_id: record
            for record in self._records(CandidateRecord, source=evidence)
        }
        effects = {
            record.estimate_id: record for record in self._records(EffectEstimate, source=evidence)
        }
        summaries = []
        for decision in self._records(DecisionRecord, source=evidence):
            candidate = candidates.get(decision.candidate_id)
            proposal = None if candidate is None else proposals.get(candidate.proposal_id)
            effect = (
                None
                if decision.effect_estimate_id is None
                else effects.get(decision.effect_estimate_id)
            )
            if proposal is None:
                continue
            summaries.append(
                {
                    "candidate_id": decision.candidate_id,
                    "constraints_passed": decision.constraints_passed,
                    "hypothesis": proposal.hypothesis,
                    "mean_gain_bpb": None if effect is None else effect.mean_gain_bpb,
                    "probability_exceeds_minimum": (
                        None if effect is None else effect.probability_exceeds_minimum
                    ),
                    "reasons": list(decision.reasons),
                    "stage": decision.stage.value,
                    "title": proposal.title,
                    "verdict": decision.verdict.value,
                }
            )
        return tuple(summaries[-self.config.previous_result_limit :])

    def _proposal_number(self, snapshot: CampaignSnapshot) -> int:
        if snapshot.active_proposal_id is not None:
            return max(snapshot.used.proposals, 1)
        return snapshot.used.proposals + 1

    def _request_id(self, campaign_id: str, proposal_number: int) -> str:
        return f"research-{campaign_id}-{proposal_number}"

    def _workspace_path(self, request_id: str) -> Path:
        return self.config.workspace_root.expanduser().resolve() / request_id

    def _ensure_workspace(self, path: Path, parent_commit: str) -> None:
        if path.exists():
            head = _git(path, "rev-parse", "HEAD")
            if head != parent_commit:
                try:
                    validate_candidate_patch(
                        self.repository_root,
                        parent_commit=parent_commit,
                        candidate_commit=head,
                    )
                except RunnerError as error:
                    raise OrchestratorError(
                        "recovered research workspace is at an unexpected commit"
                    ) from error
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(
            self.repository_root,
            "worktree",
            "add",
            "--detach",
            str(path),
            parent_commit,
        )

    def _remove_workspace(self, path: Path) -> None:
        if not path.exists():
            return
        _git(self.repository_root, "worktree", "remove", "--force", str(path))

    @staticmethod
    def _attempt_payload(
        attempt: ResearchAttempt,
        proposal_id: str,
        artifact_root: Path,
    ) -> dict[str, Any]:
        try:
            transcript = attempt.transcript_path.resolve().relative_to(artifact_root.resolve())
        except ValueError as error:
            raise OrchestratorError("research transcript is outside its artifact root") from error
        return {
            "changed_paths": list(attempt.changed_paths),
            "diff_sha256": attempt.diff_sha256,
            "failure_reason": attempt.failure_reason,
            "prompt_sha256": attempt.prompt_sha256,
            "proposal": None if attempt.proposal is None else asdict(attempt.proposal),
            "proposal_id": proposal_id,
            "response_sha256": attempt.response_sha256,
            "status": attempt.status.value,
            "timed_out": attempt.timed_out,
            "transcript": transcript.as_posix(),
            "usage": asdict(attempt.usage),
        }

    def _research(
        self,
        snapshot: CampaignSnapshot,
    ) -> tuple[ProposalDraft | None, str, str, Path, str]:
        proposal_number = self._proposal_number(snapshot)
        request_id = self._request_id(snapshot.campaign_id, proposal_number)
        proposal_id = _stable_id("proposal", request_id, snapshot.accepted_parent_commit)
        workspace = self._workspace_path(request_id)
        try:
            program_text = self.config.program_path.read_text(encoding="utf-8")
        except OSError as error:
            raise OrchestratorError(f"cannot read research program: {error}") from error
        request = ResearchRequest(
            request_id=request_id,
            parent_commit=snapshot.accepted_parent_commit,
            proposal_number=proposal_number,
            program_text=program_text,
            previous_results=self._previous_results(),
            maximum_total_tokens=self.config.researcher_token_allowance,
        )
        prompt_hash = hashlib.sha256(request.prompt().encode("utf-8")).hexdigest()
        ledger_before = self.ledger.verify()
        operation_key = f"invoke-{request_id}"
        claim = self.state.begin_operation(
            operation_key,
            "research",
            {
                "parent_commit": request.parent_commit,
                "ledger_head": ledger_before.head_event_sha256,
                "prompt_sha256": prompt_hash,
                "proposal_number": proposal_number,
                "request_id": request_id,
            },
        )
        reservation_id = f"budget-{operation_key}"
        requested_budget = BudgetAmount(
            proposals=1,
            researcher_tokens=self.config.researcher_token_allowance,
        )

        if claim.disposition is ClaimDisposition.REPLAY:
            reservation = self.state.reserve_budget(
                reservation_id,
                requested_budget,
                operation_key=operation_key,
            )
            if claim.status is OperationStatus.FAILED or claim.result is None:
                if reservation.status is ReservationStatus.RESERVED:
                    self.state.settle_budget(reservation_id, requested_budget)
                return None, proposal_id, request_id, workspace, claim.error or "research failed"
            result = claim.result
            if reservation.status is ReservationStatus.RESERVED:
                usage = result.get("usage", {})
                self.state.settle_budget(
                    reservation_id,
                    BudgetAmount(
                        proposals=1,
                        researcher_tokens=int(
                            result.get(
                                "charged_researcher_tokens",
                                usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                            )
                        ),
                    ),
                )
            draft = (
                None
                if result.get("proposal") is None
                else ProposalDraft.from_mapping(result["proposal"])
            )
            return draft, proposal_id, request_id, workspace, str(result.get("status"))

        self._ensure_workspace(workspace, request.parent_commit)
        if claim.disposition is ClaimDisposition.EXECUTE:
            try:
                self.state.reserve_budget(
                    reservation_id,
                    requested_budget,
                    operation_key=operation_key,
                )
            except BudgetExceeded as error:
                self.state.fail_operation(operation_key, str(error))
                raise
            attempt = self.researcher.run(
                request,
                workspace=workspace,
                artifact_root=self.config.researcher_artifact_root,
            )
        else:
            self.state.reserve_budget(
                reservation_id,
                requested_budget,
                operation_key=operation_key,
            )
            transcript_path = self.config.researcher_artifact_root / f"{request_id}.json"
            if transcript_path.is_file():
                attempt = load_research_attempt(
                    transcript_path,
                    request,
                    workspace=workspace,
                )
            else:
                reason = "research invocation was interrupted before retaining a transcript"
                self.state.fail_operation(operation_key, reason)
                self.state.settle_budget(reservation_id, requested_budget)
                return None, proposal_id, request_id, workspace, reason

        ledger_after = self.ledger.verify()
        if (
            ledger_after.event_count != ledger_before.event_count
            or ledger_after.head_event_sha256 != ledger_before.head_event_sha256
        ):
            raise OrchestratorError("researcher invocation changed protected ledger evidence")

        actual_researcher_tokens = (
            attempt.usage.total_tokens
            if attempt.response is not None
            else requested_budget.researcher_tokens
        )
        actual_budget = BudgetAmount(
            proposals=1,
            researcher_tokens=actual_researcher_tokens,
        )
        result = self._attempt_payload(
            attempt,
            proposal_id,
            self.config.researcher_artifact_root,
        )
        result["charged_researcher_tokens"] = actual_researcher_tokens
        self.state.complete_operation(operation_key, result)
        self.state.settle_budget(reservation_id, actual_budget)
        return attempt.proposal, proposal_id, request_id, workspace, attempt.status.value

    def _append_proposal(
        self,
        draft: ProposalDraft,
        proposal_id: str,
        parent_commit: str,
    ) -> PatchProposal:
        proposal = draft.to_record(
            proposal_id=proposal_id,
            parent_commit=parent_commit,
        )
        self.ledger.ensure(proposal, writer_role=WriterRole.RESEARCH_AGENT)
        return proposal

    def _candidate_ref(self, proposal_id: str) -> str:
        return f"refs/autodidact/candidates/{proposal_id}"

    def _commit_candidate(
        self,
        proposal: PatchProposal,
        workspace: Path,
    ) -> str:
        operation_key = f"commit-{proposal.proposal_id}"
        claim = self.state.begin_operation(
            operation_key,
            "candidate-commit",
            {
                "parent_commit": proposal.parent_commit,
                "proposal_id": proposal.proposal_id,
                "title": proposal.title,
            },
        )
        candidate_ref = self._candidate_ref(proposal.proposal_id)
        existing_ref = _git(
            self.repository_root,
            "rev-parse",
            "--verify",
            candidate_ref,
            check=False,
        )
        if claim.disposition is ClaimDisposition.REPLAY:
            if claim.result is None:
                raise OrchestratorError("candidate commit operation has no result")
            candidate_commit = str(claim.result["candidate_commit"])
        elif existing_ref:
            candidate_commit = existing_ref
            validate_candidate_patch(
                self.repository_root,
                parent_commit=proposal.parent_commit,
                candidate_commit=candidate_commit,
            )
            self.state.complete_operation(
                operation_key,
                {"candidate_commit": candidate_commit, "recovered": True},
            )
        else:
            workspace_head = _git(workspace, "rev-parse", "HEAD")
            if claim.disposition is ClaimDisposition.RECOVER and workspace_head != (
                proposal.parent_commit
            ):
                candidate_commit = workspace_head
                validate_candidate_patch(
                    self.repository_root,
                    parent_commit=proposal.parent_commit,
                    candidate_commit=candidate_commit,
                )
            else:
                if claim.disposition is ClaimDisposition.RECOVER:
                    self.state.restart_interrupted(operation_key)
                _git(workspace, "diff", "--check")
                _git(workspace, "add", "--", "train.py")
                if not _git(workspace, "diff", "--cached", "--name-only"):
                    raise OrchestratorError("research proposal has no staged candidate change")
                title = " ".join(proposal.title.split())[:72]
                _git(workspace, "commit", "--no-gpg-sign", "-m", f"Research: {title}")
                candidate_commit = _git(workspace, "rev-parse", "HEAD")
                validate_candidate_patch(
                    self.repository_root,
                    parent_commit=proposal.parent_commit,
                    candidate_commit=candidate_commit,
                )
            _git(
                self.repository_root,
                "update-ref",
                candidate_ref,
                candidate_commit,
            )
            self.state.complete_operation(
                operation_key,
                {"candidate_commit": candidate_commit, "recovered": False},
            )
        if existing_ref and existing_ref != candidate_commit:
            raise OrchestratorError("candidate ref differs from the retained operation result")
        validate_candidate_patch(
            self.repository_root,
            parent_commit=proposal.parent_commit,
            candidate_commit=candidate_commit,
        )
        if not existing_ref:
            _git(
                self.repository_root,
                "update-ref",
                candidate_ref,
                candidate_commit,
            )
        return candidate_commit

    def _registration_request(
        self,
        proposal: PatchProposal,
        candidate_commit: str,
    ) -> ExperimentRequest:
        stage = ExperimentStage.CHEAP
        return ExperimentRequest(
            repository_root=self.repository_root,
            ledger_path=self.config.ledger_path,
            data_root=self.config.data_root,
            output_root=self.config.output_root,
            proposal_id=proposal.proposal_id,
            candidate_commit=candidate_commit,
            stage=stage,
            seeds=self.policy.seed_pool[: self.policy.initial_pairs(stage)],
            assignment_seed=0,
            token_budget=self.policy.token_budget(stage),
            eval_tokens=self.policy.eval_tokens(stage),
            batch_size=self.policy.batch_size,
            eval_batch_size=self.policy.eval_batch_size,
            timeout_seconds=self.policy.timeout_seconds,
            device=self.config.device,
            limits=self.policy.resource_limits(),
            estimated_accelerator_hour_usd=self.config.estimated_accelerator_hour_usd,
        )

    def _register_candidate(
        self,
        proposal: PatchProposal,
        candidate_commit: str,
    ) -> CandidateRecord:
        operation_key = f"register-{proposal.proposal_id}"
        claim = self.state.begin_operation(
            operation_key,
            "candidate-registration",
            {
                "candidate_commit": candidate_commit,
                "parent_commit": proposal.parent_commit,
                "proposal_id": proposal.proposal_id,
            },
        )
        existing = self._candidate_for_proposal(proposal.proposal_id)
        if claim.disposition is ClaimDisposition.REPLAY:
            if existing is None:
                raise OrchestratorError("registered candidate is missing from the ledger")
            return existing
        if existing is not None:
            candidate = existing
            self.state.complete_operation(
                operation_key,
                {"candidate_id": candidate.candidate_id, "recovered": True},
            )
            return candidate
        if claim.disposition is ClaimDisposition.RECOVER:
            self.state.restart_interrupted(operation_key)
        candidate = self.runner_factory(
            self._registration_request(proposal, candidate_commit)
        ).register_candidate()
        self.state.complete_operation(
            operation_key,
            {"candidate_id": candidate.candidate_id, "recovered": False},
        )
        return candidate

    def _schedule_trials(
        self,
        schedule: TrialSchedule,
        *,
        source: Sequence[Any] | None = None,
    ) -> list[TrialSpec]:
        return [
            trial
            for trial in self._records(
                TrialSpec,
                schedule.candidate_id,
                source=source,
            )
            if trial.stage is schedule.stage and trial.seed in schedule.seeds
        ]

    def _schedule_complete(
        self,
        schedule: TrialSchedule,
        *,
        source: Sequence[Any] | None = None,
    ) -> bool:
        evidence = self._record_snapshot() if source is None else source
        trials = self._schedule_trials(schedule, source=evidence)
        if {trial.seed for trial in trials} != set(schedule.seeds):
            return False
        pairs = {
            pair.trial_id
            for pair in self._records(
                PairedResult,
                schedule.candidate_id,
                source=evidence,
            )
        }
        runs = self._records(RunResult, source=evidence)
        for trial in trials:
            trial_runs = [run for run in runs if run.trial_id == trial.trial_id]
            failed_pair = len(trial_runs) == 2 and any(
                run.status is not RunStatus.SUCCEEDED for run in trial_runs
            )
            if trial.trial_id not in pairs and not failed_pair:
                return False
        return True

    def _pending_schedule(self, candidate_id: str) -> TrialSchedule | None:
        evidence = self._record_snapshot()
        schedules = self._records(TrialSchedule, candidate_id, source=evidence)
        return next(
            (
                schedule
                for schedule in schedules
                if not self._schedule_complete(schedule, source=evidence)
            ),
            None,
        )

    def _experiment_request(
        self,
        candidate: CandidateRecord,
        schedule: TrialSchedule,
    ) -> ExperimentRequest:
        return ExperimentRequest(
            repository_root=self.repository_root,
            ledger_path=self.config.ledger_path,
            data_root=self.config.data_root,
            output_root=self.config.output_root,
            proposal_id=candidate.proposal_id,
            candidate_commit=candidate.candidate_commit,
            stage=schedule.stage,
            seeds=schedule.seeds,
            assignment_seed=schedule.assignment_seed,
            token_budget=schedule.token_budget,
            eval_tokens=schedule.eval_tokens,
            batch_size=schedule.batch_size,
            eval_batch_size=schedule.eval_batch_size,
            timeout_seconds=schedule.limits.timeout_seconds,
            device=self.config.device,
            limits=schedule.limits,
            estimated_accelerator_hour_usd=self.config.estimated_accelerator_hour_usd,
        )

    def _schedule_usage(self, schedule: TrialSchedule) -> BudgetAmount:
        evidence = self._record_snapshot()
        trials = self._schedule_trials(schedule, source=evidence)
        trial_ids = {trial.trial_id for trial in trials}
        runs = [
            run for run in self._records(RunResult, source=evidence) if run.trial_id in trial_ids
        ]
        run_ids = {run.run_id for run in runs}
        compute = [
            record
            for record in self._records(ComputeRecord, source=evidence)
            if record.run_id in run_ids
        ]
        return BudgetAmount(
            training_tokens=sum(run.tokens_seen for run in runs),
            compute_seconds=math.fsum(record.accelerator_seconds for record in compute),
        )

    def _run_schedule(
        self,
        candidate: CandidateRecord,
        schedule: TrialSchedule,
    ) -> dict[str, Any]:
        operation_key = f"run-{schedule.schedule_id}"
        claim = self.state.begin_operation(
            operation_key,
            "paired-run",
            record_to_envelope(schedule),
        )
        reservation_id = f"budget-{operation_key}"
        requested = BudgetAmount(
            training_tokens=schedule.token_budget * 2 * len(schedule.seeds),
            compute_seconds=(schedule.limits.timeout_seconds * 4.0 * len(schedule.seeds)),
        )
        if claim.disposition is ClaimDisposition.REPLAY:
            reservation = self.state.reserve_budget(
                reservation_id,
                requested,
                operation_key=operation_key,
            )
            if reservation.status is ReservationStatus.RESERVED:
                self.state.settle_budget(reservation_id, self._schedule_usage(schedule))
            if claim.status is OperationStatus.FAILED or claim.result is None:
                raise OrchestratorError(claim.error or "paired run failed")
            return claim.result
        if claim.disposition is ClaimDisposition.EXECUTE:
            try:
                self.state.reserve_budget(
                    reservation_id,
                    requested,
                    operation_key=operation_key,
                )
            except BudgetExceeded as error:
                self.state.fail_operation(operation_key, str(error))
                raise
        else:
            try:
                self.state.reserve_budget(
                    reservation_id,
                    requested,
                    operation_key=operation_key,
                )
            except BudgetExceeded as error:
                self.state.fail_operation(operation_key, str(error))
                raise
        if claim.disposition is ClaimDisposition.RECOVER and self._schedule_complete(schedule):
            result = {
                "candidate_id": candidate.candidate_id,
                "recovered": True,
                "schedule_id": schedule.schedule_id,
                "stage": schedule.stage.value,
            }
            self.state.complete_operation(operation_key, result)
            self.state.settle_budget(reservation_id, self._schedule_usage(schedule))
            return result
        if claim.disposition is ClaimDisposition.RECOVER:
            self.state.restart_interrupted(operation_key)

        try:
            result = self.runner_factory(self._experiment_request(candidate, schedule)).run()
            result = {**result, "schedule_id": schedule.schedule_id}
            self.state.complete_operation(operation_key, result)
            return result
        except Exception as error:
            message = f"{type(error).__name__}: {error}"[:4_000]
            self.state.fail_operation(operation_key, message)
            raise
        finally:
            reservation = self.state.reserve_budget(
                reservation_id,
                requested,
                operation_key=operation_key,
            )
            if reservation.status is ReservationStatus.RESERVED:
                self.state.settle_budget(reservation_id, self._schedule_usage(schedule))

    def _reward_update(
        self,
        candidate: CandidateRecord,
        schedule: TrialSchedule,
    ) -> dict[str, Any]:
        operation_key = f"reward-{schedule.schedule_id}"
        evidence = self._record_snapshot()
        schedule_trials = self._schedule_trials(schedule, source=evidence)
        schedule_trial_ids = {trial.trial_id for trial in schedule_trials}
        schedule_pairs = [
            pair
            for pair in self._records(
                PairedResult,
                candidate.candidate_id,
                source=evidence,
            )
            if pair.trial_id in schedule_trial_ids
        ]
        claim = self.state.begin_operation(
            operation_key,
            "reward-update",
            {
                "candidate_id": candidate.candidate_id,
                "paired_result_ids": sorted(pair.paired_result_id for pair in schedule_pairs),
                "schedule_id": schedule.schedule_id,
                "stage": schedule.stage.value,
            },
        )
        if claim.disposition is ClaimDisposition.REPLAY:
            if claim.status is OperationStatus.FAILED or claim.result is None:
                raise OrchestratorError(claim.error or "reward update failed")
            return claim.result
        if claim.disposition is ClaimDisposition.RECOVER:
            self.state.restart_interrupted(operation_key)
        try:
            failed_runs = [
                run
                for run in self._records(RunResult, source=evidence)
                if run.trial_id in schedule_trial_ids and run.status is not RunStatus.SUCCEEDED
            ]
            if failed_runs or not schedule_pairs:
                result = {
                    "candidate_id": candidate.candidate_id,
                    "reason": "reward update requires successful paired evidence",
                    "recommendation": "unavailable",
                    "stage": schedule.stage.value,
                }
                self.state.complete_operation(operation_key, result)
                return result
            if schedule.stage in {ExperimentStage.CHEAP, ExperimentStage.INTERMEDIATE}:
                features = extract_learning_curve_features(
                    self.ledger,
                    self.config.output_root,
                    candidate.candidate_id,
                )
                store_learning_curve_features(self.config.features_path, features)
                result: dict[str, Any] = {
                    "candidate_id": candidate.candidate_id,
                    "feature_id": features.feature_id,
                    "recommendation": "run_full_for_calibration",
                    "stage": schedule.stage.value,
                }
                if self.config.model_path.is_file():
                    model = load_model(self.config.model_path)
                    prediction, distribution, recommendation = build_downstream_prediction(
                        self.ledger,
                        candidate.candidate_id,
                        features,
                        model,
                        reject_probability=self.policy.allocation_rejection_probability,
                        full_test_probability=self.policy.allocation_full_test_probability,
                    )
                    self.ledger.ensure(prediction, writer_role=WriterRole.CONTROLLER)
                    result.update(
                        {
                            "expected_full_gain_bpb": distribution.mean,
                            "prediction_id": prediction.prediction_id,
                            "probability_exceeds_minimum": (
                                distribution.probability_exceeds_minimum
                            ),
                            "recommendation": recommendation,
                        }
                    )
            elif schedule.stage is ExperimentStage.FULL:
                label = build_full_budget_label(self.ledger, candidate.candidate_id)
                store_full_budget_label(self.config.labels_path, label)
                model = calibrate_model(
                    load_features(self.config.features_path),
                    load_labels(self.config.labels_path),
                    minimum_label_count=(
                        self._calibration_target() or self.config.minimum_reward_labels
                    ),
                )
                save_model(self.config.model_path, model)
                result = {
                    "calibrated": model.calibrated,
                    "candidate_id": candidate.candidate_id,
                    "full_budget_label_count": model.label_count,
                    "label_id": label.label_id,
                    "model_sha256": model.sha256(),
                    "stage": schedule.stage.value,
                }
            else:
                result = {
                    "candidate_id": candidate.candidate_id,
                    "stage": schedule.stage.value,
                }
            self.state.complete_operation(operation_key, result)
            return result
        except Exception as error:
            message = f"{type(error).__name__}: {error}"[:4_000]
            self.state.fail_operation(operation_key, message)
            raise

    def _terminal_outcome(
        self,
        candidate: CandidateRecord,
        action: dict[str, Any],
    ) -> ProposalOutcome:
        snapshot = self.state.snapshot()
        ledger_parent = self.ledger.current_parent()
        promoted = ledger_parent == candidate.candidate_commit
        if promoted:
            generation = snapshot.generation + 1
            result_action = "promote"
        else:
            generation = snapshot.generation
            result_action = "reject"
        self.state.set_progress(
            phase="ready",
            accepted_parent_commit=ledger_parent,
            generation=generation,
            active_proposal_id=None,
            active_candidate_id=None,
        )
        reasons = action.get("reasons", [])
        reason = "; ".join(str(item) for item in reasons) or result_action
        return ProposalOutcome(
            action=result_action,
            proposal_id=candidate.proposal_id,
            candidate_id=candidate.candidate_id,
            parent_commit=candidate.parent_commit,
            resulting_parent_commit=ledger_parent,
            reason=reason,
        )

    def _drive_candidate(self, candidate: CandidateRecord) -> ProposalOutcome:
        controller = self._controller_for_candidate(candidate)
        for _transition in range(MAX_CONTROL_TRANSITIONS):
            status = self.state.checkpoint_control()
            if status is not CampaignStatus.RUNNING:
                raise _CampaignStopped(f"campaign stopped while {status.value}")
            action = controller.advance(candidate.candidate_id)
            if action.get("action") in {"reject", "promote"}:
                return self._terminal_outcome(candidate, action)
            if action.get("action") == "status" and action.get("latest_verdict") in {
                DecisionVerdict.REJECT.value,
                DecisionVerdict.PROMOTE.value,
            }:
                return self._terminal_outcome(candidate, action)
            schedule = self._pending_schedule(candidate.candidate_id)
            if schedule is None:
                continue
            self.state.set_progress(
                phase="running_experiment",
                accepted_parent_commit=self.state.snapshot().accepted_parent_commit,
                generation=self.state.snapshot().generation,
                active_proposal_id=candidate.proposal_id,
                active_candidate_id=candidate.candidate_id,
            )
            self._run_schedule(candidate, schedule)
            self._reward_update(candidate, schedule)
            self.state.set_progress(
                phase="candidate_ready",
                accepted_parent_commit=self.state.snapshot().accepted_parent_commit,
                generation=self.state.snapshot().generation,
                active_proposal_id=candidate.proposal_id,
                active_candidate_id=candidate.candidate_id,
            )
        raise OrchestratorError("candidate exceeded the maximum controller transitions")

    def run_one_proposal(self) -> ProposalOutcome:
        self._assert_consistent_parent()
        snapshot = self.state.snapshot()
        if snapshot.active_candidate_id is not None:
            candidate = self._candidate(snapshot.active_candidate_id)
            assert candidate is not None
            return self._drive_candidate(candidate)

        if snapshot.active_proposal_id is None:
            draft, proposal_id, _request_id, workspace, research_status = self._research(snapshot)
            if draft is None:
                self._remove_workspace(workspace)
                refreshed = self.state.snapshot()
                self.state.set_progress(
                    phase="ready",
                    accepted_parent_commit=refreshed.accepted_parent_commit,
                    generation=refreshed.generation,
                    active_proposal_id=None,
                    active_candidate_id=None,
                )
                return ProposalOutcome(
                    action=research_status,
                    proposal_id=None,
                    candidate_id=None,
                    parent_commit=refreshed.accepted_parent_commit,
                    resulting_parent_commit=refreshed.accepted_parent_commit,
                    reason=research_status,
                )
            proposal = self._append_proposal(
                draft,
                proposal_id,
                snapshot.accepted_parent_commit,
            )
            self.state.set_progress(
                phase="proposal_ready",
                accepted_parent_commit=snapshot.accepted_parent_commit,
                generation=snapshot.generation,
                active_proposal_id=proposal.proposal_id,
                active_candidate_id=None,
            )
        else:
            event = self.ledger.get(snapshot.active_proposal_id)
            if not isinstance(event.record, PatchProposal):
                raise OrchestratorError("active proposal is missing from the ledger")
            proposal = event.record
            proposal_number = self._proposal_number(snapshot)
            workspace = self._workspace_path(
                self._request_id(snapshot.campaign_id, proposal_number)
            )
            self._ensure_workspace(workspace, proposal.parent_commit)

        candidate_commit = self._commit_candidate(proposal, workspace)
        candidate = self._register_candidate(proposal, candidate_commit)
        self._remove_workspace(workspace)
        self.state.set_progress(
            phase="candidate_ready",
            accepted_parent_commit=proposal.parent_commit,
            generation=self.state.snapshot().generation,
            active_proposal_id=proposal.proposal_id,
            active_candidate_id=candidate.candidate_id,
        )
        self._controller_for_candidate(candidate).initialize(candidate.candidate_id)
        return self._drive_candidate(candidate)

    def run(self, *, max_new_proposals: int | None = None) -> dict[str, Any]:
        if max_new_proposals is not None and max_new_proposals <= 0:
            raise OrchestratorError("max_new_proposals must be positive")
        snapshot = self.state.snapshot()
        with RepositoryLock(self.repository_root, campaign_id=snapshot.campaign_id):
            self.state.mark_running_interrupted()
            outcomes = []
            try:
                while True:
                    current = self.state.snapshot()
                    control = self.state.checkpoint_control()
                    if control is not CampaignStatus.RUNNING:
                        break
                    if current.status in {
                        CampaignStatus.COMPLETED,
                        CampaignStatus.FAILED,
                        CampaignStatus.CANCELLED,
                    }:
                        break
                    if current.active_proposal_id is None and current.used.proposals >= (
                        current.limits.max_proposals
                    ):
                        self.state.mark_completed()
                        break
                    if max_new_proposals is not None and len(outcomes) >= max_new_proposals:
                        break
                    outcomes.append(asdict(self.run_one_proposal()))
            except KeyboardInterrupt:
                self.state.request_pause("interrupt requested after retaining active evidence")
                self.state.checkpoint_control()
            except _CampaignStopped:
                pass
            except BudgetExceeded as error:
                current = self.state.snapshot()
                if current.active_proposal_id is None:
                    self.state.mark_completed()
                else:
                    self.state.mark_failed(str(error))
            final = self.state.snapshot()
            return {
                "campaign_id": final.campaign_id,
                "current_parent_commit": final.accepted_parent_commit,
                "generation": final.generation,
                "outcomes": outcomes,
                "phase": final.phase,
                "reward_calibration": self.calibration_status(),
                "downstream_allocation": self.downstream_allocation_status(),
                "status": final.status.value,
                "usage": asdict(final.used),
            }


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _initial_parent(repository_root: Path, accepted_ref: str) -> str:
    accepted = _git(
        repository_root,
        "rev-parse",
        "--verify",
        accepted_ref,
        check=False,
    )
    return accepted or _git(repository_root, "rev-parse", "HEAD")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the autonomous research campaign loop.")
    parser.add_argument("--repository-root", type=_path, default=Path.cwd())
    parser.add_argument("--ledger-path", type=_path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--state-path", type=_path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--data-root", type=_path, default=default_output_root())
    parser.add_argument("--output-root", type=_path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workspace-root", type=_path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument(
        "--researcher-artifact-root",
        type=_path,
        default=DEFAULT_RESEARCHER_ARTIFACT_ROOT,
    )
    parser.add_argument("--reward-root", type=_path, default=DEFAULT_REWARD_ROOT)
    parser.add_argument("--program", type=_path, default=DEFAULT_PROGRAM_PATH)
    parser.add_argument("--researcher-config", type=_path, default=DEFAULT_RESEARCHER_CONFIG_PATH)
    parser.add_argument("--device", default="auto")
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("initialize")
    initialize.add_argument("--campaign-id", required=True)
    initialize.add_argument("--max-proposals", type=int, required=True)
    initialize.add_argument("--max-wall-seconds", type=float, required=True)
    initialize.add_argument("--max-researcher-tokens", type=int, required=True)
    initialize.add_argument("--max-training-tokens", type=int, required=True)
    initialize.add_argument("--max-compute-seconds", type=float, required=True)
    initialize.add_argument("--reward-calibration-labels", type=int, default=0)
    initialize.add_argument("--use-downstream-allocation", action="store_true")

    run = commands.add_parser("run")
    run.add_argument("--max-new-proposals", type=int)
    run.add_argument(
        "--researcher-token-allowance",
        type=int,
        default=DEFAULT_RESEARCHER_TOKEN_ALLOWANCE,
    )
    commands.add_parser("status")
    for name in ("pause", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("--reason", required=True)
    commands.add_parser("resume")
    return parser


def _config_from_args(args: argparse.Namespace) -> OrchestratorConfig:
    return OrchestratorConfig(
        repository_root=args.repository_root,
        ledger_path=args.ledger_path,
        data_root=args.data_root,
        output_root=args.output_root,
        workspace_root=args.workspace_root,
        researcher_artifact_root=args.researcher_artifact_root,
        reward_root=args.reward_root,
        program_path=args.program,
        device=args.device,
        researcher_token_allowance=getattr(
            args,
            "researcher_token_allowance",
            DEFAULT_RESEARCHER_TOKEN_ALLOWANCE,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repository_root = args.repository_root.resolve()
        if args.command == "initialize":
            with RepositoryLock(repository_root, campaign_id=args.campaign_id):
                if args.ledger_path.exists():
                    ledger = ExperimentLedger.open(args.ledger_path, read_only=False)
                    parent = ledger.current_parent()
                else:
                    parent = _initial_parent(repository_root, DEFAULT_ACCEPTED_REF)
                    ledger = ExperimentLedger.create(
                        args.ledger_path,
                        initial_parent_commit=parent,
                    )
                state = CampaignStore.create(
                    args.state_path,
                    campaign_id=args.campaign_id,
                    initial_parent_commit=parent,
                    limits=CampaignLimits(
                        max_proposals=args.max_proposals,
                        max_wall_seconds=args.max_wall_seconds,
                        max_researcher_tokens=args.max_researcher_tokens,
                        max_training_tokens=args.max_training_tokens,
                        max_compute_seconds=args.max_compute_seconds,
                        reward_calibration_labels=args.reward_calibration_labels,
                        use_downstream_allocation=args.use_downstream_allocation,
                    ),
                )
                synchronize_accepted_ref(repository_root, DEFAULT_ACCEPTED_REF, parent)
            config = _config_from_args(args)
            calibration = _reward_calibration_status(
                args.reward_calibration_labels,
                features_path=config.features_path,
                labels_path=config.labels_path,
            )
            payload: dict[str, Any] = {
                "campaign": asdict(state.snapshot()),
                "downstream_allocation": _downstream_allocation_status(
                    args.use_downstream_allocation,
                    calibration=calibration,
                    minimum_labels=(args.reward_calibration_labels or config.minimum_reward_labels),
                    model_path=config.model_path,
                ),
                "ledger": ledger.summary(),
                "reward_calibration": calibration,
            }
        else:
            state = CampaignStore.open(args.state_path)
            ledger = ExperimentLedger.open(
                args.ledger_path,
                read_only=args.command == "status",
            )
            if args.command == "status":
                config = _config_from_args(args)
                limits = state.snapshot().limits
                target = limits.reward_calibration_labels
                calibration = _reward_calibration_status(
                    target,
                    features_path=config.features_path,
                    labels_path=config.labels_path,
                )
                payload = {
                    "campaign": asdict(state.snapshot()),
                    "downstream_allocation": _downstream_allocation_status(
                        limits.use_downstream_allocation,
                        calibration=calibration,
                        minimum_labels=target or config.minimum_reward_labels,
                        model_path=config.model_path,
                    ),
                    "ledger": ledger.summary(),
                    "reward_calibration": calibration,
                }
            elif args.command == "pause":
                payload = {"campaign": asdict(state.request_pause(args.reason))}
            elif args.command == "cancel":
                payload = {"campaign": asdict(state.request_cancel(args.reason))}
            elif args.command == "resume":
                payload = {"campaign": asdict(state.resume())}
            else:
                researcher = CommandResearcherAdapter(
                    ResearcherConfig.from_path(args.researcher_config)
                )
                orchestrator = AutonomousResearchOrchestrator(
                    _config_from_args(args),
                    state=state,
                    ledger=ledger,
                    researcher=researcher,
                )
                payload = orchestrator.run(max_new_proposals=args.max_new_proposals)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (
        LedgerError,
        OSError,
        OrchestratorError,
        ResearcherError,
        RewardError,
        RunnerError,
        RunStateError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
