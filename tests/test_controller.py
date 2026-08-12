from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from autodidact.controller import (
    DEFAULT_ACCEPTED_REF,
    ControllerError,
    DecisionMode,
    PatchRCTController,
    PatchRCTPolicy,
    main,
    useful_gain_posterior,
)
from autodidact.ledger import ExperimentLedger, LedgerStateError, WriterRole
from autodidact.records import (
    AllocationAction,
    ArtifactManifest,
    ArtifactRef,
    ArtifactRetention,
    CandidateRecord,
    DecisionRecord,
    DecisionVerdict,
    DownstreamAllocation,
    DownstreamPrediction,
    EffectEstimate,
    ExperimentStage,
    LineageRecord,
    PairedResult,
    PatchProposal,
    RunResult,
    RunStatus,
    TrialSchedule,
    TrialSpec,
    build_paired_result,
)
from tests.experiment_fixtures import digest, evidence_records


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "train.py").write_text("BASELINE = True\n", encoding="utf-8")
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add parent")
    parent = _git(repository, "rev-parse", "HEAD")
    (repository / "train.py").write_text("BASELINE = True\nTREATMENT = True\n", encoding="utf-8")
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add candidate")
    candidate = _git(repository, "rev-parse", "HEAD")
    return repository, parent, candidate


def _setup(
    tmp_path: Path,
    *,
    repository: bool = False,
    policy: PatchRCTPolicy | None = None,
) -> tuple[ExperimentLedger, PatchRCTController, CandidateRecord, Path | None]:
    if repository:
        repository_root, parent_commit, candidate_commit = _repository(tmp_path)
    else:
        repository_root = None
        parent_commit = "a" * 40
        candidate_commit = "b" * 40
    ledger = ExperimentLedger.create(
        tmp_path / "ledger.sqlite3",
        initial_parent_commit=parent_commit,
    )
    proposal = PatchProposal(
        proposal_id="proposal-controller-001",
        parent_commit=parent_commit,
        title="Controller candidate",
        hypothesis="The treatment should produce a useful paired objective gain.",
        mechanism="Exercise sequential PatchRCT evidence decisions.",
        change="Change train.py only.",
        expected_effect=0.01,
        minimum_useful_gain=0.001,
        resource_risk="No expected resource regression.",
        failure_signal="Useful-gain probability remains low.",
        interaction_risk="No known interaction.",
    )
    candidate = CandidateRecord(
        candidate_id="candidate-controller-001",
        proposal_id=proposal.proposal_id,
        parent_commit=parent_commit,
        candidate_commit=candidate_commit,
        diff_sha256=digest("1"),
        changed_paths=("train.py",),
        trainer_sha256=digest("2"),
        policy_sha256=digest("3"),
        parameter_count=2_000_000,
    )
    ledger.append_many(
        (
            (proposal, WriterRole.RESEARCH_AGENT),
            (candidate, WriterRole.CONTROLLER),
        )
    )
    controller = PatchRCTController(
        ledger,
        policy=policy,
        repository_root=repository_root,
    )
    return ledger, controller, candidate, repository_root


def _manifest(run: RunResult, suffix: str) -> ArtifactManifest:
    return ArtifactManifest(
        manifest_id=f"manifest-{suffix}",
        run_id=run.run_id,
        artifacts=(
            ArtifactRef(
                artifact_id=f"checkpoint-{suffix}",
                kind="checkpoint",
                relative_path=f"runs/{suffix}/checkpoint.pt",
                sha256=hashlib.sha256(f"checkpoint-{suffix}".encode()).hexdigest(),
                size_bytes=100,
                retention=ArtifactRetention.EPHEMERAL,
            ),
            ArtifactRef(
                artifact_id=f"metrics-{suffix}",
                kind="metrics",
                relative_path=f"runs/{suffix}/metrics.jsonl",
                sha256=hashlib.sha256(f"metrics-{suffix}".encode()).hexdigest(),
                size_bytes=50,
                retention=ArtifactRetention.COMPACT,
            ),
        ),
    )


def _complete_schedule(
    ledger: ExperimentLedger,
    candidate: CandidateRecord,
    schedule: TrialSchedule,
    gains: dict[int, float],
    *,
    failed_seed: int | None = None,
) -> None:
    base = evidence_records()
    for seed in schedule.seeds:
        trial_id = f"trial-{schedule.stage.value}-{seed}"
        trial = replace(
            base["trial"],
            trial_id=trial_id,
            candidate_id=candidate.candidate_id,
            parent_commit=candidate.parent_commit,
            candidate_commit=candidate.candidate_commit,
            stage=schedule.stage,
            seed=seed,
            token_budget=schedule.token_budget,
            eval_tokens=schedule.eval_tokens,
            batch_size=schedule.batch_size,
            eval_batch_size=schedule.eval_batch_size,
            candidate_trainer_sha256=candidate.trainer_sha256,
            limits=schedule.limits,
        )
        evaluation_tokens = schedule.eval_tokens or 250_000
        order_hash = hashlib.sha256(f"order-{schedule.stage.value}-{seed}".encode()).hexdigest()
        parent = replace(
            base["parent_run"],
            run_id=f"run-parent-{schedule.stage.value}-{seed}",
            trial_id=trial_id,
            seed=seed,
            target_tokens=schedule.token_budget,
            tokens_seen=schedule.token_budget,
            evaluation_tokens=evaluation_tokens,
            data_order_sha256=order_hash,
        )
        candidate_run = replace(
            base["candidate_run"],
            run_id=f"run-candidate-{schedule.stage.value}-{seed}",
            trial_id=trial_id,
            seed=seed,
            target_tokens=schedule.token_budget,
            tokens_seen=schedule.token_budget,
            evaluation_tokens=evaluation_tokens,
            objective_value=parent.objective_value - gains.get(seed, 0.0),
            data_order_sha256=order_hash,
        )
        entries: list[tuple[object, WriterRole]] = [
            (trial, WriterRole.CONTROLLER),
            (parent, WriterRole.EVALUATOR),
            (_manifest(parent, f"parent-{schedule.stage.value}-{seed}"), WriterRole.EVALUATOR),
        ]
        if seed == failed_seed:
            candidate_run = replace(
                candidate_run,
                status=RunStatus.OOM,
                tokens_seen=schedule.token_budget // 2,
                evaluation_tokens=0,
                objective_value=None,
                training_tokens_per_second=None,
                evaluation_tokens_per_second=None,
                evaluation_seconds=None,
                data_order_sha256=None,
                failure_reason="candidate exhausted memory",
            )
            entries.append((candidate_run, WriterRole.EVALUATOR))
            ledger.append_many(tuple(entries))
            continue
        entries.extend(
            (
                (candidate_run, WriterRole.EVALUATOR),
                (
                    _manifest(candidate_run, f"candidate-{schedule.stage.value}-{seed}"),
                    WriterRole.EVALUATOR,
                ),
            )
        )
        ledger.append_many(tuple(entries))
        pair = build_paired_result(
            f"pair-{schedule.stage.value}-{seed}",
            trial=trial,
            candidate_id=candidate.candidate_id,
            parent=parent,
            candidate=candidate_run,
        )
        ledger.append(pair, writer_role=WriterRole.EVALUATOR)


def _append_prediction(
    ledger: ExperimentLedger,
    candidate: CandidateRecord,
    probability: float,
    *,
    label_count: int = 2,
) -> DownstreamPrediction:
    trials = {
        event.record.trial_id: event.record
        for event in ledger.events()
        if isinstance(event.record, TrialSpec)
        and event.record.candidate_id == candidate.candidate_id
        and event.record.stage in {ExperimentStage.CHEAP, ExperimentStage.INTERMEDIATE}
    }
    source_trial_ids = tuple(
        event.record.trial_id
        for event in ledger.events()
        if isinstance(event.record, PairedResult) and event.record.trial_id in trials
    )
    source_stages = tuple(
        stage
        for stage in (ExperimentStage.CHEAP, ExperimentStage.INTERMEDIATE)
        if any(trials[trial_id].stage is stage for trial_id in source_trial_ids)
    )
    proposal = ledger.get(candidate.proposal_id).record
    assert isinstance(proposal, PatchProposal)
    prediction = DownstreamPrediction(
        prediction_id=(
            f"prediction-allocation-{len(source_trial_ids):02d}-{int(probability * 1000):03d}"
        ),
        candidate_id=candidate.candidate_id,
        source_trial_ids=source_trial_ids,
        source_stages=source_stages,
        target_stage=ExperimentStage.FULL,
        expected_objective_gain=0.01 if probability >= 0.5 else -0.01,
        predictive_standard_deviation=0.005,
        interval_lower=-0.02,
        interval_upper=0.02,
        minimum_useful_gain=proposal.minimum_useful_gain,
        probability_exceeds_minimum=probability,
        model_version="test-bayesian-reward-v1",
        full_budget_label_count=label_count,
    )
    ledger.append(prediction, writer_role=WriterRole.CONTROLLER)
    return prediction


def _complete_allocation_early_stages(
    ledger: ExperimentLedger,
    controller: PatchRCTController,
    candidate: CandidateRecord,
    *,
    objective_gain: float = -0.01,
) -> None:
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: objective_gain})
    action = controller.advance(candidate.candidate_id)
    assert action["stage"] == ExperimentStage.INTERMEDIATE.value
    intermediate = _latest_schedule(ledger, ExperimentStage.INTERMEDIATE)
    _complete_schedule(
        ledger,
        candidate,
        intermediate,
        {seed: objective_gain for seed in intermediate.seeds},
    )


def _latest_schedule(
    ledger: ExperimentLedger,
    stage: ExperimentStage,
) -> TrialSchedule:
    schedules = [
        event.record
        for event in ledger.events()
        if isinstance(event.record, TrialSchedule) and event.record.stage is stage
    ]
    return schedules[-1]


def test_posterior_probability_increases_with_stronger_and_repeated_gains() -> None:
    policy = PatchRCTPolicy()
    low = useful_gain_posterior((-0.01,), minimum_useful_gain=0.001, policy=policy)
    neutral = useful_gain_posterior((0.0,), minimum_useful_gain=0.001, policy=policy)
    strong = useful_gain_posterior((0.01,), minimum_useful_gain=0.001, policy=policy)
    repeated = useful_gain_posterior((0.01, 0.01, 0.01), minimum_useful_gain=0.001, policy=policy)

    assert low.probability_exceeds_minimum < neutral.probability_exceeds_minimum
    assert neutral.probability_exceeds_minimum < strong.probability_exceeds_minimum
    assert repeated.probability_exceeds_minimum > strong.probability_exceeds_minimum
    assert repeated.standard_deviation < strong.standard_deviation


def test_greedy_mode_keeps_one_positive_cheap_result_and_advances_lineage(
    tmp_path: Path,
) -> None:
    policy = PatchRCTPolicy(decision_mode=DecisionMode.GREEDY)
    ledger, controller, candidate, repository = _setup(
        tmp_path,
        repository=True,
        policy=policy,
    )
    assert repository is not None
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: 0.0001})

    action = controller.advance(candidate.candidate_id)

    assert action["action"] == "promote"
    assert action["stage"] == ExperimentStage.CHEAP.value
    assert ledger.current_parent() == candidate.candidate_commit
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert decisions[-1].probability_threshold == 0.0
    assert "greedy keep/discard" in decisions[-1].reasons[0]
    assert _git(repository, "rev-parse", DEFAULT_ACCEPTED_REF) == candidate.candidate_commit


def test_greedy_mode_discards_nonpositive_result_without_escalating(tmp_path: Path) -> None:
    policy = PatchRCTPolicy(decision_mode=DecisionMode.GREEDY)
    ledger, controller, candidate, _repository_root = _setup(tmp_path, policy=policy)
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: 0.0})

    action = controller.advance(candidate.candidate_id)

    assert action["action"] == "reject"
    assert action["stage"] == ExperimentStage.CHEAP.value
    schedules = [
        event.record for event in ledger.events() if isinstance(event.record, TrialSchedule)
    ]
    assert [schedule.stage for schedule in schedules] == [ExperimentStage.CHEAP]
    assert ledger.current_parent() == candidate.parent_commit


def test_greedy_mode_cannot_enable_patchrct_calibration_or_allocation() -> None:
    with pytest.raises(ControllerError, match="greedy mode"):
        PatchRCTPolicy(
            decision_mode=DecisionMode.GREEDY,
            use_downstream_allocation=True,
        )


def test_scout_patchrct_escalates_positive_scout_without_promoting(tmp_path: Path) -> None:
    policy = PatchRCTPolicy(decision_mode=DecisionMode.SCOUT_PATCH_RCT)
    ledger, controller, candidate, _repository_root = _setup(tmp_path, policy=policy)
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: 0.0001})

    action = controller.advance(candidate.candidate_id)

    assert action["verdict"] == "escalate"
    assert action["from_stage"] == ExperimentStage.CHEAP.value
    assert action["stage"] == ExperimentStage.INTERMEDIATE.value
    assert action["seeds"] == [11, 23]
    assert ledger.current_parent() == candidate.parent_commit
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert decisions[-1].probability_threshold == 0.0
    assert "positive scout gain" in decisions[-1].reasons[0]


def test_scout_patchrct_rejects_nonpositive_scout(tmp_path: Path) -> None:
    policy = PatchRCTPolicy(decision_mode=DecisionMode.SCOUT_PATCH_RCT)
    ledger, controller, candidate, _repository_root = _setup(tmp_path, policy=policy)
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: 0.0})

    action = controller.advance(candidate.candidate_id)

    assert action["action"] == "reject"
    assert action["stage"] == ExperimentStage.CHEAP.value
    assert ledger.current_parent() == candidate.parent_commit
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert decisions[-1].probability_threshold == 0.0
    assert "cheap scout" in decisions[-1].reasons[0]


def test_scout_patchrct_requires_exactly_one_initial_cheap_pair() -> None:
    with pytest.raises(ControllerError, match="requires one cheap pair"):
        PatchRCTPolicy(
            decision_mode=DecisionMode.SCOUT_PATCH_RCT,
            cheap_initial_pairs=2,
        )


def test_scout_patchrct_does_not_bias_forced_calibration_labels(tmp_path: Path) -> None:
    policy = PatchRCTPolicy(
        decision_mode=DecisionMode.SCOUT_PATCH_RCT,
        force_full_evaluation=True,
    )
    ledger, controller, candidate, _repository_root = _setup(tmp_path, policy=policy)
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: -0.01})

    action = controller.advance(candidate.candidate_id)

    assert action["verdict"] == "escalate"
    assert action["stage"] == ExperimentStage.INTERMEDIATE.value
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert "calibration policy" in decisions[-1].reasons[0]


def test_controller_escalates_all_stages_and_atomically_promotes_git_parent(
    tmp_path: Path,
) -> None:
    ledger, controller, candidate, repository = _setup(tmp_path, repository=True)
    assert repository is not None

    initial = controller.initialize(candidate.candidate_id)
    assert initial["action"] == "schedule"
    assert initial["stage"] == "cheap"
    assert initial["seeds"] == [11]
    assert _git(repository, "rev-parse", DEFAULT_ACCEPTED_REF) == candidate.parent_commit

    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: 0.01})
    intermediate_action = controller.advance(candidate.candidate_id)
    assert intermediate_action["verdict"] == "escalate"
    assert intermediate_action["stage"] == "intermediate"
    assert intermediate_action["seeds"] == [11, 23]

    intermediate = _latest_schedule(ledger, ExperimentStage.INTERMEDIATE)
    _complete_schedule(ledger, candidate, intermediate, {11: 0.01, 23: 0.01})
    full_action = controller.advance(candidate.candidate_id)
    assert full_action["verdict"] == "escalate"
    assert full_action["stage"] == "full"
    assert full_action["seeds"] == [11, 23, 37]

    full = _latest_schedule(ledger, ExperimentStage.FULL)
    _complete_schedule(
        ledger,
        candidate,
        full,
        {11: 0.01, 23: 0.01, 37: 0.01},
    )
    promoted = controller.advance(candidate.candidate_id)

    assert promoted["action"] == "promote"
    assert promoted["new_parent_commit"] == candidate.candidate_commit
    assert ledger.current_parent() == candidate.candidate_commit
    assert _git(repository, "rev-parse", DEFAULT_ACCEPTED_REF) == candidate.candidate_commit
    events = ledger.events()
    decisions = [event.record for event in events if isinstance(event.record, DecisionRecord)]
    assert [decision.verdict for decision in decisions] == [
        DecisionVerdict.ESCALATE,
        DecisionVerdict.ESCALATE,
        DecisionVerdict.PROMOTE,
    ]
    assert len([event for event in events if isinstance(event.record, EffectEstimate)]) == 3
    assert len([event for event in events if isinstance(event.record, LineageRecord)]) == 1


def test_uncertain_effect_schedules_the_next_fixed_seed(tmp_path: Path) -> None:
    ledger, controller, candidate, _repository_root = _setup(tmp_path)
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: 0.0})

    action = controller.advance(candidate.candidate_id)

    assert action["action"] == "schedule"
    assert action["stage"] == "cheap"
    assert action["seeds"] == [23]
    additional = _latest_schedule(ledger, ExperimentStage.CHEAP)
    assert additional.source_effect_estimate_id is not None
    assert additional.policy_sha256 == controller.policy.sha256()
    waiting = controller.advance(candidate.candidate_id)
    assert waiting["action"] == "waiting"
    assert waiting["scheduled_seeds"] == [11, 23]


def test_low_probability_is_rejected_without_searching_more_seeds(tmp_path: Path) -> None:
    ledger, controller, candidate, _repository_root = _setup(tmp_path)
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: -0.01})

    rejected = controller.advance(candidate.candidate_id)

    assert rejected["action"] == "reject"
    decision = next(
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    )
    assert decision.verdict is DecisionVerdict.REJECT
    assert decision.effect_estimate_id is not None
    assert len([event for event in ledger.events() if isinstance(event.record, TrialSchedule)]) == 1


def test_calibration_policy_forces_safe_low_gain_patch_to_full_label(tmp_path: Path) -> None:
    policy = PatchRCTPolicy(force_full_evaluation=True)
    ledger, controller, candidate, _repository_root = _setup(tmp_path, policy=policy)
    controller.initialize(candidate.candidate_id)

    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {11: -0.01})
    intermediate_action = controller.advance(candidate.candidate_id)
    assert intermediate_action["stage"] == "intermediate"

    intermediate = _latest_schedule(ledger, ExperimentStage.INTERMEDIATE)
    _complete_schedule(ledger, candidate, intermediate, {11: -0.01, 23: -0.01})
    full_action = controller.advance(candidate.candidate_id)
    assert full_action["stage"] == "full"

    full = _latest_schedule(ledger, ExperimentStage.FULL)
    _complete_schedule(
        ledger,
        candidate,
        full,
        {11: -0.01, 23: -0.01, 37: -0.01},
    )
    rejected = controller.advance(candidate.candidate_id)

    assert rejected["action"] == "reject"
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert [decision.verdict for decision in decisions] == [
        DecisionVerdict.ESCALATE,
        DecisionVerdict.ESCALATE,
        DecisionVerdict.REJECT,
    ]
    assert [decision.probability_threshold for decision in decisions[:2]] == [0.0, 0.0]
    assert all("calibration policy" in decision.reasons[0] for decision in decisions[:2])
    assert [
        schedule.stage for schedule in controller._records(TrialSchedule, candidate.candidate_id)
    ] == [
        ExperimentStage.CHEAP,
        ExperimentStage.INTERMEDIATE,
        ExperimentStage.FULL,
    ]


def test_calibration_policy_does_not_override_run_failure_rejection(tmp_path: Path) -> None:
    ledger, controller, candidate, _repository_root = _setup(
        tmp_path,
        policy=PatchRCTPolicy(force_full_evaluation=True),
    )
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {}, failed_seed=11)

    rejected = controller.advance(candidate.candidate_id)

    assert rejected["action"] == "reject"
    assert len([event for event in ledger.events() if isinstance(event.record, TrialSchedule)]) == 1


def test_calibrated_low_prediction_stops_after_intermediate_evidence(tmp_path: Path) -> None:
    policy = PatchRCTPolicy(
        use_downstream_allocation=True,
        allocation_audit_fraction=0.0,
        minimum_downstream_labels=2,
    )
    ledger, controller, candidate, _repository_root = _setup(tmp_path, policy=policy)
    _complete_allocation_early_stages(ledger, controller, candidate)
    prediction = _append_prediction(ledger, candidate, 0.05)
    before_allocation = tmp_path / "before-allocation.sqlite3"
    shutil.copy2(ledger.path, before_allocation)

    rejected = controller.advance(candidate.candidate_id)

    assert rejected["action"] == "reject"
    assert rejected["allocation_action"] == AllocationAction.STOP.value
    allocations = [
        event.record for event in ledger.events() if isinstance(event.record, DownstreamAllocation)
    ]
    assert len(allocations) == 1
    allocation = allocations[0]
    assert allocation.downstream_prediction_id == prediction.prediction_id
    decision = ledger.get(allocation.planned_decision_id or "missing").record
    assert isinstance(decision, DecisionRecord)
    assert decision.verdict is DecisionVerdict.REJECT
    assert decision.downstream_prediction_id == prediction.prediction_id
    assert ledger.verify().event_count == len(ledger.events())

    incomplete = ExperimentLedger.open(before_allocation, read_only=False)
    effect = ledger.get(allocation.effect_estimate_id).record
    assert isinstance(effect, EffectEstimate)
    incomplete.append(effect, writer_role=WriterRole.CONTROLLER)
    with pytest.raises(LedgerStateError, match="decision hash differs"):
        incomplete.append_many(
            (
                (
                    replace(allocation, planned_decision_sha256="f" * 64),
                    WriterRole.CONTROLLER,
                ),
                (decision, WriterRole.CONTROLLER),
            )
        )
    with pytest.raises(LedgerStateError, match="planned decision is missing"):
        incomplete.append(allocation, writer_role=WriterRole.CONTROLLER)
    assert not any(isinstance(event.record, DownstreamAllocation) for event in incomplete.events())


def test_high_prediction_allocates_full_but_cannot_promote_without_full_effect(
    tmp_path: Path,
) -> None:
    policy = PatchRCTPolicy(
        use_downstream_allocation=True,
        allocation_audit_fraction=0.0,
        minimum_downstream_labels=2,
    )
    ledger, controller, candidate, _repository_root = _setup(tmp_path, policy=policy)
    _complete_allocation_early_stages(ledger, controller, candidate)
    prediction = _append_prediction(ledger, candidate, 0.90)

    full_action = controller.advance(candidate.candidate_id)

    assert full_action["allocation_action"] == AllocationAction.RUN_FULL.value
    assert full_action["stage"] == ExperimentStage.FULL.value
    allocation = next(
        event.record for event in ledger.events() if isinstance(event.record, DownstreamAllocation)
    )
    assert allocation.downstream_prediction_id == prediction.prediction_id
    full = _latest_schedule(ledger, ExperimentStage.FULL)
    _complete_schedule(
        ledger,
        candidate,
        full,
        {seed: -0.01 for seed in full.seeds},
    )

    rejected = controller.advance(candidate.candidate_id)

    assert rejected["action"] == "reject"
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert decisions[-1].stage is ExperimentStage.FULL
    assert decisions[-1].verdict is DecisionVerdict.REJECT
    assert decisions[-1].downstream_prediction_id is None


def test_uncertain_prediction_gathers_next_fixed_intermediate_seed(tmp_path: Path) -> None:
    policy = PatchRCTPolicy(
        use_downstream_allocation=True,
        allocation_audit_fraction=0.0,
        minimum_downstream_labels=2,
    )
    ledger, controller, candidate, _repository_root = _setup(tmp_path, policy=policy)
    _complete_allocation_early_stages(ledger, controller, candidate)
    _append_prediction(ledger, candidate, 0.50)

    for expected_seed in (37, 53, 71):
        action = controller.advance(candidate.candidate_id)
        assert action["allocation_action"] == AllocationAction.GATHER_MORE.value
        assert action["stage"] == ExperimentStage.INTERMEDIATE.value
        assert action["seeds"] == [expected_seed]
        schedule = _latest_schedule(ledger, ExperimentStage.INTERMEDIATE)
        _complete_schedule(ledger, candidate, schedule, {expected_seed: -0.01})
        _append_prediction(ledger, candidate, 0.50)

    full_action = controller.advance(candidate.candidate_id)

    assert full_action["allocation_action"] == AllocationAction.UNCERTAIN_FULL.value
    assert full_action["stage"] == ExperimentStage.FULL.value
    allocations = [
        event.record for event in ledger.events() if isinstance(event.record, DownstreamAllocation)
    ]
    assert [allocation.action for allocation in allocations] == [
        AllocationAction.GATHER_MORE,
        AllocationAction.GATHER_MORE,
        AllocationAction.GATHER_MORE,
        AllocationAction.UNCERTAIN_FULL,
    ]
    assert [allocation.next_seed for allocation in allocations[:3]] == [37, 53, 71]
    assert all(allocation.planned_decision_id is None for allocation in allocations[:3])


def test_low_prediction_audit_sample_runs_full_and_uncalibrated_prediction_fails_closed(
    tmp_path: Path,
) -> None:
    policy = PatchRCTPolicy(
        use_downstream_allocation=True,
        allocation_audit_fraction=1.0,
        minimum_downstream_labels=2,
    )
    ledger, controller, candidate, _repository_root = _setup(tmp_path, policy=policy)
    _complete_allocation_early_stages(ledger, controller, candidate)
    _append_prediction(ledger, candidate, 0.05, label_count=1)

    with pytest.raises(ControllerError, match="sufficiently calibrated"):
        controller.advance(candidate.candidate_id)

    prediction = _append_prediction(ledger, candidate, 0.04, label_count=2)
    action = controller.advance(candidate.candidate_id)

    assert action["allocation_action"] == AllocationAction.AUDIT_FULL.value
    assert action["stage"] == ExperimentStage.FULL.value
    allocation = next(
        event.record for event in ledger.events() if isinstance(event.record, DownstreamAllocation)
    )
    assert allocation.downstream_prediction_id == prediction.prediction_id


def test_allocation_policy_does_not_override_run_failure_rejection(tmp_path: Path) -> None:
    ledger, controller, candidate, _repository_root = _setup(
        tmp_path,
        policy=PatchRCTPolicy(
            use_downstream_allocation=True,
            minimum_downstream_labels=2,
        ),
    )
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {}, failed_seed=11)

    rejected = controller.advance(candidate.candidate_id)

    assert rejected["action"] == "reject"
    assert not any(isinstance(event.record, DownstreamAllocation) for event in ledger.events())


def test_terminal_run_failure_is_rejected_without_fabricating_effect(tmp_path: Path) -> None:
    ledger, controller, candidate, _repository_root = _setup(tmp_path)
    controller.initialize(candidate.candidate_id)
    cheap = _latest_schedule(ledger, ExperimentStage.CHEAP)
    _complete_schedule(ledger, candidate, cheap, {}, failed_seed=11)

    rejected = controller.advance(candidate.candidate_id)

    assert rejected["action"] == "reject"
    decision = next(
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    )
    assert decision.effect_estimate_id is None
    assert decision.constraints_passed is False
    assert any("oom" in reason for reason in decision.reasons)


def test_ledger_rejects_trials_outside_the_protected_schedule(tmp_path: Path) -> None:
    ledger, controller, candidate, _repository_root = _setup(tmp_path)
    controller.initialize(candidate.candidate_id)
    schedule = _latest_schedule(ledger, ExperimentStage.CHEAP)
    trial = replace(
        evidence_records()["trial"],
        trial_id="trial-unscheduled-001",
        candidate_id=candidate.candidate_id,
        parent_commit=candidate.parent_commit,
        candidate_commit=candidate.candidate_commit,
        seed=999,
        token_budget=schedule.token_budget,
        eval_tokens=schedule.eval_tokens,
        batch_size=schedule.batch_size,
        eval_batch_size=schedule.eval_batch_size,
        candidate_trainer_sha256=candidate.trainer_sha256,
        limits=schedule.limits,
    )

    with pytest.raises(LedgerStateError, match="protected stage schedule"):
        ledger.append(trial, writer_role=WriterRole.CONTROLLER)


def test_ledger_rejects_effect_free_decision_without_failed_run(tmp_path: Path) -> None:
    ledger, controller, candidate, _repository_root = _setup(tmp_path)
    controller.initialize(candidate.candidate_id)
    proposal = ledger.get(candidate.proposal_id).record
    assert isinstance(proposal, PatchProposal)
    decision = DecisionRecord(
        decision_id="decision-unproved-failure",
        candidate_id=candidate.candidate_id,
        stage=ExperimentStage.CHEAP,
        verdict=DecisionVerdict.REJECT,
        effect_estimate_id=None,
        downstream_prediction_id=None,
        minimum_useful_gain=proposal.minimum_useful_gain,
        probability_threshold=0.8,
        constraints_passed=False,
        reasons=("unproved failure",),
    )

    with pytest.raises(LedgerStateError, match="recorded failed stage run"):
        ledger.append(decision, writer_role=WriterRole.CONTROLLER)


def test_controller_waits_until_every_scheduled_trial_exists(tmp_path: Path) -> None:
    _ledger, controller, candidate, _repository_root = _setup(tmp_path)
    controller.initialize(candidate.candidate_id)

    status = controller.advance(candidate.candidate_id)

    assert status["action"] == "waiting"
    assert status["reasons"] == ["trial not yet created for seed 11"]


def test_controller_cli_initializes_and_reports_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger, _controller, candidate, repository = _setup(tmp_path, repository=True)
    assert repository is not None
    common = [
        "--ledger-path",
        str(ledger.path),
        "--repository-root",
        str(repository),
    ]

    assert main([*common, "initialize", "--candidate-id", candidate.candidate_id]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["action"] == "schedule"

    assert main([*common, "status", "--candidate-id", candidate.candidate_id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["schedule_ids"] == [initialized["schedule_id"]]
