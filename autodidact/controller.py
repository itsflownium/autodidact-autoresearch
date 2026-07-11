"""Sequential PatchRCT scheduling, evidence estimation, and promotion decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autodidact.data.integrity import canonical_json_bytes
from autodidact.ledger import ExperimentLedger, LedgerError, WriterRole
from autodidact.records import (
    CandidateRecord,
    DecisionRecord,
    DecisionVerdict,
    EffectEstimate,
    ExperimentStage,
    LineageRecord,
    PairedResult,
    PatchProposal,
    ResourceLimits,
    RunResult,
    RunStatus,
    TrialSchedule,
    TrialSpec,
    build_effect_estimate,
)

CONTROLLER_SCHEMA_VERSION = 1
DEFAULT_LEDGER_PATH = Path("artifacts/ledger/experiments.sqlite3")
DEFAULT_ACCEPTED_REF = "refs/autodidact/accepted"
SCHEDULER_VERSION = "patchrct-sequential-v1"
ESTIMATOR_VERSION = "patchrct-normal-normal-v1"

_STAGES = (
    ExperimentStage.CHEAP,
    ExperimentStage.INTERMEDIATE,
    ExperimentStage.FULL,
)
_NEXT_STAGE = {
    ExperimentStage.CHEAP: ExperimentStage.INTERMEDIATE,
    ExperimentStage.INTERMEDIATE: ExperimentStage.FULL,
}


class ControllerError(RuntimeError):
    """Raised when PatchRCT cannot make a valid state transition."""


@dataclass(frozen=True, slots=True)
class PosteriorEstimate:
    mean_gain_bpb: float
    standard_deviation_bpb: float
    observation_standard_deviation_bpb: float
    probability_exceeds_minimum: float


@dataclass(frozen=True, slots=True)
class PatchRCTPolicy:
    seed_pool: tuple[int, ...] = (11, 23, 37, 53, 71)
    cheap_initial_pairs: int = 1
    intermediate_initial_pairs: int = 2
    full_initial_pairs: int = 3
    cheap_token_budget: int = 2_000_000
    intermediate_token_budget: int = 6_000_000
    full_token_budget: int = 20_000_000
    cheap_eval_tokens: int | None = 250_000
    intermediate_eval_tokens: int | None = 1_000_000
    full_eval_tokens: int | None = None
    batch_size: int = 64
    eval_batch_size: int = 64
    timeout_seconds: int = 7_200
    prior_mean_gain_bpb: float = 0.0
    prior_standard_deviation_bpb: float = 0.01
    seed_noise_standard_deviation_bpb: float = 0.004654
    rejection_probability: float = 0.10
    continuation_probability: float = 0.80
    promotion_probability: float = 0.95
    max_parameter_count: int = 1_050_000
    max_peak_process_rss_bytes: int | None = None
    max_peak_device_bytes: int | None = None
    min_training_tokens_per_second: float | None = None
    max_training_throughput_regression_fraction: float | None = 0.10
    max_peak_process_rss_regression_fraction: float | None = 0.10
    max_peak_device_regression_fraction: float | None = 0.10

    def __post_init__(self) -> None:
        if not self.seed_pool or len(set(self.seed_pool)) != len(self.seed_pool):
            raise ControllerError("seed_pool must be a nonempty unique sequence")
        if any(type(seed) is not int or seed < 0 or seed > 2**32 - 1 for seed in self.seed_pool):
            raise ControllerError("seed_pool contains an invalid seed")
        initial_counts = (
            self.cheap_initial_pairs,
            self.intermediate_initial_pairs,
            self.full_initial_pairs,
        )
        if any(count <= 0 or count > len(self.seed_pool) for count in initial_counts):
            raise ControllerError("initial stage pair counts must fit inside seed_pool")
        if any(
            value <= 0
            for value in (
                self.cheap_token_budget,
                self.intermediate_token_budget,
                self.full_token_budget,
                self.batch_size,
                self.eval_batch_size,
                self.timeout_seconds,
            )
        ):
            raise ControllerError("budgets, batch sizes, and timeout must be positive")
        for value in (
            self.cheap_eval_tokens,
            self.intermediate_eval_tokens,
            self.full_eval_tokens,
        ):
            if value is not None and value <= 0:
                raise ControllerError("evaluation budgets must be positive or None")
        for name in (
            "prior_mean_gain_bpb",
            "prior_standard_deviation_bpb",
            "seed_noise_standard_deviation_bpb",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ControllerError(f"{name} must be finite")
        if self.prior_standard_deviation_bpb <= 0.0:
            raise ControllerError("prior_standard_deviation_bpb must be positive")
        if self.seed_noise_standard_deviation_bpb <= 0.0:
            raise ControllerError("seed_noise_standard_deviation_bpb must be positive")
        probabilities = (
            self.rejection_probability,
            self.continuation_probability,
            self.promotion_probability,
        )
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ControllerError("policy probabilities must lie between zero and one")
        if not (
            self.rejection_probability < self.continuation_probability <= self.promotion_probability
        ):
            raise ControllerError("policy probability thresholds are not ordered")
        self.resource_limits()

    def initial_pairs(self, stage: ExperimentStage) -> int:
        return {
            ExperimentStage.CHEAP: self.cheap_initial_pairs,
            ExperimentStage.INTERMEDIATE: self.intermediate_initial_pairs,
            ExperimentStage.FULL: self.full_initial_pairs,
        }[stage]

    def token_budget(self, stage: ExperimentStage) -> int:
        return {
            ExperimentStage.CHEAP: self.cheap_token_budget,
            ExperimentStage.INTERMEDIATE: self.intermediate_token_budget,
            ExperimentStage.FULL: self.full_token_budget,
        }[stage]

    def eval_tokens(self, stage: ExperimentStage) -> int | None:
        return {
            ExperimentStage.CHEAP: self.cheap_eval_tokens,
            ExperimentStage.INTERMEDIATE: self.intermediate_eval_tokens,
            ExperimentStage.FULL: self.full_eval_tokens,
        }[stage]

    def resource_limits(self) -> ResourceLimits:
        return ResourceLimits(
            timeout_seconds=self.timeout_seconds,
            max_parameter_count=self.max_parameter_count,
            max_peak_process_rss_bytes=self.max_peak_process_rss_bytes,
            max_peak_device_bytes=self.max_peak_device_bytes,
            min_training_tokens_per_second=self.min_training_tokens_per_second,
            max_training_throughput_regression_fraction=(
                self.max_training_throughput_regression_fraction
            ),
            max_peak_process_rss_regression_fraction=(
                self.max_peak_process_rss_regression_fraction
            ),
            max_peak_device_regression_fraction=self.max_peak_device_regression_fraction,
        )

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(asdict(self))).hexdigest()


def useful_gain_posterior(
    gains: tuple[float, ...],
    *,
    minimum_useful_gain_bpb: float,
    policy: PatchRCTPolicy,
) -> PosteriorEstimate:
    if not gains or any(not math.isfinite(gain) for gain in gains):
        raise ControllerError("posterior requires finite paired gains")
    empirical_variance = statistics.variance(gains) if len(gains) > 1 else 0.0
    observation_variance = max(
        policy.seed_noise_standard_deviation_bpb**2,
        empirical_variance,
    )
    prior_variance = policy.prior_standard_deviation_bpb**2
    posterior_variance = 1.0 / (1.0 / prior_variance + len(gains) / observation_variance)
    posterior_mean = posterior_variance * (
        policy.prior_mean_gain_bpb / prior_variance + sum(gains) / observation_variance
    )
    posterior_standard_deviation = math.sqrt(posterior_variance)
    distribution = statistics.NormalDist(
        mu=posterior_mean,
        sigma=posterior_standard_deviation,
    )
    probability = 1.0 - distribution.cdf(minimum_useful_gain_bpb)
    return PosteriorEstimate(
        mean_gain_bpb=posterior_mean,
        standard_deviation_bpb=posterior_standard_deviation,
        observation_standard_deviation_bpb=math.sqrt(observation_variance),
        probability_exceeds_minimum=min(1.0, max(0.0, probability)),
    )


def _stable_id(prefix: str, *parts: object) -> str:
    payload = canonical_json_bytes([str(part) for part in parts])
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _git(repository_root: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise ControllerError(f"Git reference update failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def synchronize_accepted_ref(
    repository_root: Path,
    accepted_ref: str,
    desired_commit: str,
) -> None:
    if not accepted_ref.startswith("refs/") or ".." in accepted_ref:
        raise ControllerError("accepted_ref must be a full safe Git reference")
    repository_root = repository_root.resolve()
    _git(repository_root, "cat-file", "-e", f"{desired_commit}^{{commit}}")
    current = _git(repository_root, "rev-parse", "--verify", accepted_ref, check=False)
    if not current:
        _git(repository_root, "update-ref", accepted_ref, desired_commit)
        return
    if current == desired_commit:
        return
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", current, desired_commit],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ControllerError("accepted Git ref does not precede the ledger parent")
    _git(repository_root, "update-ref", accepted_ref, desired_commit, current)


class PatchRCTController:
    def __init__(
        self,
        ledger: ExperimentLedger,
        *,
        policy: PatchRCTPolicy | None = None,
        repository_root: Path | None = None,
        accepted_ref: str = DEFAULT_ACCEPTED_REF,
    ) -> None:
        self.ledger = ledger
        self.policy = policy or PatchRCTPolicy()
        self.repository_root = None if repository_root is None else repository_root.resolve()
        self.accepted_ref = accepted_ref

    def _records(self, expected_type: type[Any], candidate_id: str) -> list[Any]:
        records = []
        for event in self.ledger.events():
            record = event.record
            if isinstance(record, expected_type) and getattr(record, "candidate_id", None) == (
                candidate_id
            ):
                records.append(record)
        return records

    def _candidate(self, candidate_id: str) -> tuple[CandidateRecord, PatchProposal]:
        event = self.ledger.get(candidate_id)
        if not isinstance(event.record, CandidateRecord):
            raise ControllerError("candidate_id does not identify a candidate record")
        proposal_event = self.ledger.get(event.record.proposal_id)
        if not isinstance(proposal_event.record, PatchProposal):
            raise ControllerError("candidate proposal record is missing")
        return event.record, proposal_event.record

    def _sync_ref(self) -> None:
        if self.repository_root is not None:
            synchronize_accepted_ref(
                self.repository_root,
                self.accepted_ref,
                self.ledger.current_parent(),
            )

    def _assignment_seed(
        self,
        candidate_id: str,
        stage: ExperimentStage,
        schedule_index: int,
    ) -> int:
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "candidate_id": candidate_id,
                    "policy_sha256": self.policy.sha256(),
                    "schedule_index": schedule_index,
                    "stage": stage.value,
                }
            )
        ).digest()
        return int.from_bytes(digest[:4], "big")

    def _schedule(
        self,
        candidate: CandidateRecord,
        *,
        stage: ExperimentStage,
        seeds: tuple[int, ...],
        source_effect: EffectEstimate | None,
        reason: str,
    ) -> TrialSchedule:
        existing = self._records(TrialSchedule, candidate.candidate_id)
        stage_index = sum(schedule.stage is stage for schedule in existing)
        schedule = TrialSchedule(
            schedule_id=_stable_id(
                "schedule",
                candidate.candidate_id,
                stage.value,
                ",".join(str(seed) for seed in seeds),
                self.policy.sha256(),
            ),
            candidate_id=candidate.candidate_id,
            parent_commit=candidate.parent_commit,
            stage=stage,
            seeds=seeds,
            assignment_seed=self._assignment_seed(
                candidate.candidate_id,
                stage,
                stage_index,
            ),
            token_budget=self.policy.token_budget(stage),
            eval_tokens=self.policy.eval_tokens(stage),
            batch_size=self.policy.batch_size,
            eval_batch_size=self.policy.eval_batch_size,
            limits=self.policy.resource_limits(),
            policy_sha256=self.policy.sha256(),
            source_effect_estimate_id=(
                None if source_effect is None else source_effect.estimate_id
            ),
            scheduler_version=SCHEDULER_VERSION,
            reason=reason,
        )
        return schedule

    @staticmethod
    def _schedule_payload(schedule: TrialSchedule) -> dict[str, Any]:
        return {
            "action": "schedule",
            "assignment_seed": schedule.assignment_seed,
            "batch_size": schedule.batch_size,
            "candidate_id": schedule.candidate_id,
            "eval_batch_size": schedule.eval_batch_size,
            "eval_tokens": schedule.eval_tokens,
            "limits": asdict(schedule.limits),
            "reason": schedule.reason,
            "schedule_id": schedule.schedule_id,
            "seeds": list(schedule.seeds),
            "stage": schedule.stage.value,
            "token_budget": schedule.token_budget,
        }

    def initialize(self, candidate_id: str) -> dict[str, Any]:
        candidate, _proposal = self._candidate(candidate_id)
        if candidate.parent_commit != self.ledger.current_parent():
            raise ControllerError("cannot initialize a candidate from a stale parent")
        self._sync_ref()
        schedules = self._records(TrialSchedule, candidate_id)
        if schedules:
            return self.status(candidate_id)
        count = self.policy.initial_pairs(ExperimentStage.CHEAP)
        schedule = self._schedule(
            candidate,
            stage=ExperimentStage.CHEAP,
            seeds=self.policy.seed_pool[:count],
            source_effect=None,
            reason="initial cheap paired evidence",
        )
        self.ledger.append(schedule, writer_role=WriterRole.CONTROLLER)
        return self._schedule_payload(schedule)

    def _stage_trials(
        self,
        candidate_id: str,
        stage: ExperimentStage,
        scheduled_seeds: set[int],
    ) -> dict[int, TrialSpec]:
        trials = {
            trial.seed: trial
            for trial in self._records(TrialSpec, candidate_id)
            if trial.stage is stage and trial.seed in scheduled_seeds
        }
        return trials

    def _stage_pairs(
        self,
        candidate_id: str,
        trials: dict[int, TrialSpec],
    ) -> tuple[PairedResult, ...]:
        trial_ids = {trial.trial_id for trial in trials.values()}
        pairs = [
            pair for pair in self._records(PairedResult, candidate_id) if pair.trial_id in trial_ids
        ]
        seed_order = {seed: index for index, seed in enumerate(self.policy.seed_pool)}
        return tuple(sorted(pairs, key=lambda pair: seed_order[pair.seed]))

    def _stage_state(
        self,
        candidate_id: str,
        stage: ExperimentStage,
        scheduled_seeds: set[int],
    ) -> tuple[bool, list[str], tuple[PairedResult, ...]]:
        trials = self._stage_trials(candidate_id, stage, scheduled_seeds)
        if set(trials) != scheduled_seeds:
            missing = sorted(scheduled_seeds - set(trials))
            return False, [f"trial not yet created for seed {seed}" for seed in missing], ()
        pairs = self._stage_pairs(candidate_id, trials)
        pairs_by_trial = {pair.trial_id for pair in pairs}
        trial_ids = {trial.trial_id for trial in trials.values()}
        run_results = [
            event.record
            for event in self.ledger.events()
            if isinstance(event.record, RunResult) and event.record.trial_id in trial_ids
        ]
        failures = []
        waiting = []
        for seed, trial in trials.items():
            if trial.trial_id in pairs_by_trial:
                continue
            runs = [run for run in run_results if run.trial_id == trial.trial_id]
            if len(runs) < 2:
                waiting.append(f"trial seed {seed} is incomplete")
                continue
            failed = [run for run in runs if run.status is not RunStatus.SUCCEEDED]
            if failed:
                failures.extend(
                    f"{run.arm.value} seed {seed} ended {run.status.value}" for run in failed
                )
            else:
                waiting.append(f"trial seed {seed} has no paired result")
        return not waiting, [*failures, *waiting], pairs

    def _effect(
        self,
        candidate: CandidateRecord,
        proposal: PatchProposal,
        stage: ExperimentStage,
        pairs: tuple[PairedResult, ...],
    ) -> EffectEstimate:
        posterior = useful_gain_posterior(
            tuple(pair.gain_bpb for pair in pairs),
            minimum_useful_gain_bpb=proposal.minimum_useful_gain_bpb,
            policy=self.policy,
        )
        estimate = build_effect_estimate(
            _stable_id(
                "estimate",
                candidate.candidate_id,
                stage.value,
                ",".join(pair.paired_result_id for pair in pairs),
                self.policy.sha256(),
            ),
            candidate_id=candidate.candidate_id,
            stage=stage,
            pairs=pairs,
            minimum_useful_gain_bpb=proposal.minimum_useful_gain_bpb,
            probability_exceeds_minimum=posterior.probability_exceeds_minimum,
            estimator_version=f"{ESTIMATOR_VERSION}:{self.policy.sha256()[:16]}",
        )
        self.ledger.ensure(estimate, writer_role=WriterRole.CONTROLLER)
        return estimate

    def _decision(
        self,
        candidate: CandidateRecord,
        proposal: PatchProposal,
        stage: ExperimentStage,
        verdict: DecisionVerdict,
        *,
        effect: EffectEstimate | None,
        probability_threshold: float,
        constraints_passed: bool,
        reasons: tuple[str, ...],
        next_stage: ExperimentStage | None = None,
    ) -> DecisionRecord:
        return DecisionRecord(
            decision_id=_stable_id(
                "decision",
                candidate.candidate_id,
                stage.value,
                verdict.value,
                "none" if effect is None else effect.estimate_id,
                self.policy.sha256(),
            ),
            candidate_id=candidate.candidate_id,
            stage=stage,
            verdict=verdict,
            effect_estimate_id=None if effect is None else effect.estimate_id,
            downstream_prediction_id=None,
            minimum_useful_gain_bpb=proposal.minimum_useful_gain_bpb,
            probability_threshold=probability_threshold,
            constraints_passed=constraints_passed,
            reasons=reasons,
            next_stage=next_stage,
            resulting_parent_commit=(
                candidate.candidate_commit if verdict is DecisionVerdict.PROMOTE else None
            ),
        )

    def _reject(
        self,
        candidate: CandidateRecord,
        proposal: PatchProposal,
        stage: ExperimentStage,
        *,
        effect: EffectEstimate | None,
        constraints_passed: bool,
        reasons: tuple[str, ...],
    ) -> dict[str, Any]:
        threshold = (
            self.policy.promotion_probability
            if stage is ExperimentStage.FULL
            else self.policy.continuation_probability
        )
        decision = self._decision(
            candidate,
            proposal,
            stage,
            DecisionVerdict.REJECT,
            effect=effect,
            probability_threshold=threshold,
            constraints_passed=constraints_passed,
            reasons=reasons,
        )
        self.ledger.append(decision, writer_role=WriterRole.CONTROLLER)
        return {
            "action": "reject",
            "candidate_id": candidate.candidate_id,
            "decision_id": decision.decision_id,
            "reasons": list(reasons),
            "stage": stage.value,
        }

    def _escalate(
        self,
        candidate: CandidateRecord,
        proposal: PatchProposal,
        stage: ExperimentStage,
        effect: EffectEstimate,
    ) -> dict[str, Any]:
        next_stage = _NEXT_STAGE[stage]
        decision = self._decision(
            candidate,
            proposal,
            stage,
            DecisionVerdict.ESCALATE,
            effect=effect,
            probability_threshold=self.policy.continuation_probability,
            constraints_passed=True,
            reasons=("useful-gain probability satisfies the stage continuation threshold",),
            next_stage=next_stage,
        )
        count = self.policy.initial_pairs(next_stage)
        schedule = self._schedule(
            candidate,
            stage=next_stage,
            seeds=self.policy.seed_pool[:count],
            source_effect=effect,
            reason=f"{stage.value} evidence escalated to {next_stage.value}",
        )
        self.ledger.append_many(
            (
                (decision, WriterRole.CONTROLLER),
                (schedule, WriterRole.CONTROLLER),
            )
        )
        payload = self._schedule_payload(schedule)
        payload.update(
            {
                "decision_id": decision.decision_id,
                "from_stage": stage.value,
                "verdict": "escalate",
            }
        )
        return payload

    def _promote(
        self,
        candidate: CandidateRecord,
        proposal: PatchProposal,
        effect: EffectEstimate,
    ) -> dict[str, Any]:
        decision = self._decision(
            candidate,
            proposal,
            ExperimentStage.FULL,
            DecisionVerdict.PROMOTE,
            effect=effect,
            probability_threshold=self.policy.promotion_probability,
            constraints_passed=True,
            reasons=("full-stage useful-gain probability satisfies the promotion threshold",),
        )
        lineages = [
            event.record
            for event in self.ledger.events()
            if isinstance(event.record, LineageRecord)
        ]
        previous = lineages[-1] if lineages else None
        lineage = LineageRecord(
            lineage_id=_stable_id("lineage", candidate.candidate_id, decision.decision_id),
            generation=1 if previous is None else previous.generation + 1,
            previous_lineage_id=None if previous is None else previous.lineage_id,
            parent_commit=candidate.parent_commit,
            candidate_id=candidate.candidate_id,
            candidate_commit=candidate.candidate_commit,
            decision_id=decision.decision_id,
        )
        self.ledger.append_many(
            (
                (decision, WriterRole.CONTROLLER),
                (lineage, WriterRole.CONTROLLER),
            )
        )
        self._sync_ref()
        return {
            "action": "promote",
            "candidate_id": candidate.candidate_id,
            "decision_id": decision.decision_id,
            "lineage_id": lineage.lineage_id,
            "new_parent_commit": candidate.candidate_commit,
            "stage": ExperimentStage.FULL.value,
        }

    def advance(self, candidate_id: str) -> dict[str, Any]:
        candidate, proposal = self._candidate(candidate_id)
        decisions = self._records(DecisionRecord, candidate_id)
        terminal = next(
            (
                decision
                for decision in reversed(decisions)
                if decision.verdict in {DecisionVerdict.REJECT, DecisionVerdict.PROMOTE}
            ),
            None,
        )
        if terminal is not None:
            self._sync_ref()
            return self.status(candidate_id)
        schedules = self._records(TrialSchedule, candidate_id)
        if not schedules:
            return self.initialize(candidate_id)
        stage = max(schedules, key=lambda item: _STAGES.index(item.stage)).stage
        stage_schedules = [schedule for schedule in schedules if schedule.stage is stage]
        scheduled_seeds = {seed for schedule in stage_schedules for seed in schedule.seeds}
        complete, failures, pairs = self._stage_state(
            candidate_id,
            stage,
            scheduled_seeds,
        )
        if not complete:
            return {
                "action": "waiting",
                "candidate_id": candidate_id,
                "reasons": failures,
                "scheduled_seeds": sorted(scheduled_seeds),
                "stage": stage.value,
            }
        if failures:
            return self._reject(
                candidate,
                proposal,
                stage,
                effect=None,
                constraints_passed=False,
                reasons=tuple(failures),
            )
        if not pairs:
            raise ControllerError("completed stage has no paired evidence")
        effect = self._effect(candidate, proposal, stage, pairs)
        if not effect.constraints_passed:
            return self._reject(
                candidate,
                proposal,
                stage,
                effect=effect,
                constraints_passed=False,
                reasons=("one or more paired resource constraints failed",),
            )
        probability = effect.probability_exceeds_minimum
        if probability <= self.policy.rejection_probability:
            return self._reject(
                candidate,
                proposal,
                stage,
                effect=effect,
                constraints_passed=True,
                reasons=("useful-gain probability is below the rejection threshold",),
            )
        required = self.policy.initial_pairs(stage)
        used = {pair.seed for pair in pairs}
        remaining = tuple(seed for seed in self.policy.seed_pool if seed not in used)
        if len(pairs) < required:
            needed = min(required - len(pairs), len(remaining))
            if needed == 0:
                raise ControllerError("stage cannot reach its required paired evidence count")
            schedule = self._schedule(
                candidate,
                stage=stage,
                seeds=remaining[:needed],
                source_effect=effect,
                reason="stage requires its minimum paired evidence count",
            )
            self.ledger.append(schedule, writer_role=WriterRole.CONTROLLER)
            return self._schedule_payload(schedule)
        if stage in _NEXT_STAGE and probability >= self.policy.continuation_probability:
            return self._escalate(candidate, proposal, stage, effect)
        if stage is ExperimentStage.FULL and probability >= self.policy.promotion_probability:
            return self._promote(candidate, proposal, effect)
        if remaining:
            schedule = self._schedule(
                candidate,
                stage=stage,
                seeds=remaining[:1],
                source_effect=effect,
                reason="posterior remains uncertain; gather the next predetermined seed",
            )
            self.ledger.append(schedule, writer_role=WriterRole.CONTROLLER)
            return self._schedule_payload(schedule)
        threshold = (
            self.policy.promotion_probability
            if stage is ExperimentStage.FULL
            else self.policy.continuation_probability
        )
        return self._reject(
            candidate,
            proposal,
            stage,
            effect=effect,
            constraints_passed=True,
            reasons=(f"fixed seed pool exhausted below probability threshold {threshold:.3f}",),
        )

    def status(self, candidate_id: str) -> dict[str, Any]:
        candidate, _proposal = self._candidate(candidate_id)
        schedules = self._records(TrialSchedule, candidate_id)
        decisions = self._records(DecisionRecord, candidate_id)
        effects = self._records(EffectEstimate, candidate_id)
        latest_decision = decisions[-1] if decisions else None
        return {
            "action": "status",
            "candidate_id": candidate.candidate_id,
            "current_parent_commit": self.ledger.current_parent(),
            "latest_effect": (
                None
                if not effects
                else {
                    "estimate_id": effects[-1].estimate_id,
                    "mean_gain_bpb": effects[-1].mean_gain_bpb,
                    "probability_exceeds_minimum": (effects[-1].probability_exceeds_minimum),
                    "stage": effects[-1].stage.value,
                }
            ),
            "latest_verdict": (None if latest_decision is None else latest_decision.verdict.value),
            "policy_sha256": self.policy.sha256(),
            "schedule_ids": [schedule.schedule_id for schedule in schedules],
            "stages_scheduled": [schedule.stage.value for schedule in schedules],
        }


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schedule and decide PatchRCT evidence.")
    parser.add_argument("--ledger-path", type=_path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--repository-root", type=_path, default=Path.cwd())
    parser.add_argument("--accepted-ref", default=DEFAULT_ACCEPTED_REF)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("initialize", "advance", "status"):
        command = commands.add_parser(name)
        command.add_argument("--candidate-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ledger = ExperimentLedger.open(
            args.ledger_path,
            read_only=args.command == "status",
        )
        controller = PatchRCTController(
            ledger,
            repository_root=args.repository_root,
            accepted_ref=args.accepted_ref,
        )
        if args.command == "initialize":
            payload = controller.initialize(args.candidate_id)
        elif args.command == "advance":
            payload = controller.advance(args.candidate_id)
        else:
            payload = controller.status(args.candidate_id)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ControllerError, LedgerError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
