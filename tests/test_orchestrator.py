from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autodidact.checkpoints import file_sha256
from autodidact.controller import PatchRCTPolicy
from autodidact.data.integrity import policy_sha256
from autodidact.ledger import ExperimentLedger, WriterRole
from autodidact.orchestrator import (
    AutonomousResearchOrchestrator,
    OrchestratorConfig,
    main,
)
from autodidact.records import (
    AllocationAction,
    ArtifactManifest,
    ArtifactRef,
    ArtifactRetention,
    CandidateRecord,
    ComputeRecord,
    DecisionRecord,
    DecisionVerdict,
    DownstreamAllocation,
    DownstreamPrediction,
    ExperimentStage,
    PairedResult,
    RunArm,
    RunResult,
    RunStatus,
    TrialSpec,
    build_paired_result,
)
from autodidact.researcher import CommandResearcherAdapter, ResearcherConfig
from autodidact.reward import load_labels, load_model
from autodidact.runner import ExperimentRequest, validate_candidate_patch
from autodidact.runstate import CampaignLimits, CampaignStatus, CampaignStore


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "train.py").write_text("PATCH_COUNT = 0\n", encoding="utf-8")
    (repository / "program.md").write_text(
        "Propose one falsifiable change to train.py.\n",
        encoding="utf-8",
    )
    _git(repository, "add", "train.py", "program.md")
    _git(repository, "commit", "-m", "Add research parent")
    return repository, _git(repository, "rev-parse", "HEAD")


def _researcher(tmp_path: Path) -> CommandResearcherAdapter:
    script = tmp_path / "fake_researcher.py"
    script.write_text(
        """
import json
import pathlib
import sys

request = json.load(sys.stdin)
path = pathlib.Path("train.py")
path.write_text(
    path.read_text() + f"PATCH_{request['proposal_number']} = True\\n",
    encoding="utf-8",
)
proposal_number = request["proposal_number"]
response = {
    "failure_reason": None,
    "proposal": {
        "change": f"Enable synthetic patch {proposal_number}.",
        "expected_effect_bpb": -0.05,
        "failure_signal": "Paired BPB does not improve.",
        "hypothesis": f"Synthetic patch {proposal_number} should improve BPB.",
        "interaction_risk": "May interact with prior accepted patches.",
        "mechanism": "Exercise the complete autonomous campaign path.",
        "minimum_useful_gain_bpb": 0.001,
        "resource_risk": "No expected synthetic resource change.",
        "title": f"Synthetic patch {proposal_number}",
    },
    "status": "proposed",
    "usage": {"input_tokens": 10, "output_tokens": 10},
}
print(json.dumps(response, sort_keys=True))
""",
        encoding="utf-8",
    )
    return CommandResearcherAdapter(
        ResearcherConfig(
            command=(sys.executable, str(script)),
            timeout_seconds=5,
        )
    )


def _digest(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


class SyntheticRunnerFactory:
    def __init__(self, gain_bpb: float) -> None:
        self.gain_bpb = gain_bpb
        self.registration_calls = 0
        self.run_calls: list[tuple[ExperimentStage, tuple[int, ...]]] = []

    def __call__(self, request: ExperimentRequest) -> SyntheticRunner:
        return SyntheticRunner(self, request)


class SyntheticRunner:
    def __init__(self, factory: SyntheticRunnerFactory, request: ExperimentRequest) -> None:
        self.factory = factory
        self.request = request

    def _ledger(self) -> ExperimentLedger:
        return ExperimentLedger.open(self.request.ledger_path, read_only=False)

    def register_candidate(self) -> CandidateRecord:
        self.factory.registration_calls += 1
        ledger = self._ledger()
        proposal_event = ledger.get(self.request.proposal_id)
        proposal = proposal_event.record
        validation = validate_candidate_patch(
            self.request.repository_root,
            parent_commit=proposal.parent_commit,
            candidate_commit=self.request.candidate_commit,
        )
        trainer = subprocess.run(
            ["git", "show", f"{validation.candidate_commit}:train.py"],
            cwd=self.request.repository_root,
            capture_output=True,
            check=True,
        ).stdout
        candidate = CandidateRecord(
            candidate_id=f"candidate-{_digest(self.request.proposal_id)[:24]}",
            proposal_id=self.request.proposal_id,
            parent_commit=validation.parent_commit,
            candidate_commit=validation.candidate_commit,
            diff_sha256=validation.diff_sha256,
            changed_paths=validation.changed_paths,
            trainer_sha256=hashlib.sha256(trainer).hexdigest(),
            policy_sha256=policy_sha256(),
            parameter_count=1_016_960,
        )
        ledger.ensure(candidate, writer_role=WriterRole.CONTROLLER)
        return candidate

    def _candidate(self, ledger: ExperimentLedger) -> CandidateRecord:
        candidates = [
            event.record
            for event in ledger.events()
            if isinstance(event.record, CandidateRecord)
            and event.record.proposal_id == self.request.proposal_id
        ]
        assert len(candidates) == 1
        return candidates[0]

    def _trial(self, candidate: CandidateRecord, seed: int) -> TrialSpec:
        return TrialSpec(
            trial_id=(
                f"trial-{_digest(candidate.candidate_id, self.request.stage.value, seed)[:24]}"
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
            execution_order=(RunArm.PARENT, RunArm.CANDIDATE),
            data_config_sha256=_digest("data"),
            tokenizer_sha256=_digest("tokenizer"),
            parent_trainer_sha256=_digest(candidate.parent_commit, "trainer"),
            candidate_trainer_sha256=candidate.trainer_sha256,
            evaluator_sha256=_digest("evaluator"),
            runner_sha256=_digest("runner"),
            environment_sha256=_digest("environment"),
            order_assignment_sha256=_digest("order", seed),
            device="cpu",
            limits=self.request.limits,
        )

    def _run_result(self, trial: TrialSpec, arm: RunArm) -> RunResult:
        candidate_arm = arm is RunArm.CANDIDATE
        return RunResult(
            run_id=f"run-{_digest(trial.trial_id, arm.value)[:24]}",
            trial_id=trial.trial_id,
            arm=arm,
            status=RunStatus.SUCCEEDED,
            seed=trial.seed,
            target_tokens=trial.token_budget,
            tokens_seen=trial.token_budget,
            evaluation_tokens=trial.eval_tokens or 128,
            parameter_count=1_016_960,
            validation_bpb=1.0 - self.factory.gain_bpb if candidate_arm else 1.0,
            mean_train_loss=0.9 if candidate_arm else 1.0,
            training_tokens_per_second=10_000.0,
            evaluation_tokens_per_second=5_000.0,
            peak_process_rss_bytes=100_000_000,
            peak_device_allocated_bytes=None,
            peak_device_reserved_bytes=None,
            training_seconds=1.0,
            evaluation_seconds=0.1,
            wall_seconds=1.1,
            data_order_sha256=_digest("data-order", trial.seed),
        )

    def _manifest(self, run: RunResult) -> ArtifactManifest:
        root = self.request.output_root / run.trial_id / run.arm.value
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "checkpoint.pt"
        metrics = root / "metrics.jsonl"
        checkpoint.write_text(f"checkpoint for {run.run_id}\n", encoding="utf-8")
        metrics.write_text(
            "\n".join(
                json.dumps(record, sort_keys=True)
                for record in (
                    {"event": "train", "loss": 1.2, "tokens_seen": 1},
                    {
                        "event": "train",
                        "loss": run.mean_train_loss,
                        "tokens_seen": run.tokens_seen,
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )

        def artifact(path: Path, kind: str, retention: ArtifactRetention) -> ArtifactRef:
            relative = path.relative_to(self.request.output_root).as_posix()
            return ArtifactRef(
                artifact_id=f"artifact-{_digest(run.run_id, kind)[:24]}",
                kind=kind,
                relative_path=relative,
                sha256=file_sha256(path),
                size_bytes=path.stat().st_size,
                retention=retention,
            )

        return ArtifactManifest(
            manifest_id=f"manifest-{_digest(run.run_id)[:24]}",
            run_id=run.run_id,
            artifacts=(
                artifact(checkpoint, "checkpoint", ArtifactRetention.EPHEMERAL),
                artifact(metrics, "metrics", ArtifactRetention.COMPACT),
            ),
        )

    def run(self) -> dict[str, object]:
        self.factory.run_calls.append((self.request.stage, self.request.seeds))
        ledger = self._ledger()
        candidate = self._candidate(ledger)
        trials = tuple(self._trial(candidate, seed) for seed in self.request.seeds)
        ledger.append_many(
            tuple((trial, WriterRole.CONTROLLER) for trial in trials),
            idempotent=True,
        )
        pair_ids = []
        for trial in trials:
            existing_pairs = [
                event.record
                for event in ledger.events()
                if isinstance(event.record, PairedResult)
                and event.record.trial_id == trial.trial_id
            ]
            if existing_pairs:
                pair_ids.append(existing_pairs[0].paired_result_id)
                continue
            parent = self._run_result(trial, RunArm.PARENT)
            candidate_run = self._run_result(trial, RunArm.CANDIDATE)
            for run in (parent, candidate_run):
                ledger.ensure(run, writer_role=WriterRole.EVALUATOR)
                ledger.ensure(self._manifest(run), writer_role=WriterRole.EVALUATOR)
                ledger.ensure(
                    ComputeRecord(
                        compute_id=f"compute-{_digest(run.run_id)[:24]}",
                        trial_id=trial.trial_id,
                        run_id=run.run_id,
                        device=trial.device,
                        wall_seconds=run.wall_seconds,
                        accelerator_seconds=run.wall_seconds,
                        training_tokens=run.tokens_seen,
                        evaluation_tokens=run.evaluation_tokens,
                        attempts=1,
                    ),
                    writer_role=WriterRole.EVALUATOR,
                )
            pair = build_paired_result(
                f"pair-{_digest(trial.trial_id)[:24]}",
                trial=trial,
                candidate_id=candidate.candidate_id,
                parent=parent,
                candidate=candidate_run,
            )
            ledger.ensure(pair, writer_role=WriterRole.EVALUATOR)
            pair_ids.append(pair.paired_result_id)
        return {
            "candidate_id": candidate.candidate_id,
            "event": "synthetic_paired_experiment_completed",
            "pair_ids": pair_ids,
            "stage": self.request.stage.value,
        }


def _policy(
    *,
    use_downstream_allocation: bool = False,
    allocation_audit_fraction: float = 0.10,
) -> PatchRCTPolicy:
    return PatchRCTPolicy(
        seed_pool=(11, 23, 37, 53, 71),
        cheap_initial_pairs=1,
        intermediate_initial_pairs=2,
        full_initial_pairs=3,
        cheap_token_budget=100,
        intermediate_token_budget=200,
        full_token_budget=300,
        cheap_eval_tokens=64,
        intermediate_eval_tokens=64,
        full_eval_tokens=64,
        batch_size=2,
        eval_batch_size=2,
        timeout_seconds=10,
        prior_standard_deviation_bpb=0.1,
        seed_noise_standard_deviation_bpb=0.001,
        max_peak_device_regression_fraction=None,
        use_downstream_allocation=use_downstream_allocation,
        allocation_audit_fraction=allocation_audit_fraction,
    )


def _campaign(
    tmp_path: Path,
    *,
    max_proposals: int,
    gain_bpb: float,
    max_training_tokens: int = 100_000,
    reward_calibration_labels: int = 0,
    use_downstream_allocation: bool = False,
) -> tuple[
    AutonomousResearchOrchestrator,
    CampaignStore,
    ExperimentLedger,
    SyntheticRunnerFactory,
    Path,
]:
    repository, parent = _repository(tmp_path)
    ledger_path = tmp_path / "ledger.sqlite3"
    state_path = tmp_path / "campaign.sqlite3"
    ledger = ExperimentLedger.create(ledger_path, initial_parent_commit=parent)
    state = CampaignStore.create(
        state_path,
        campaign_id="campaign-001",
        initial_parent_commit=parent,
        limits=CampaignLimits(
            max_proposals=max_proposals,
            max_wall_seconds=3_600,
            max_researcher_tokens=10_000,
            max_training_tokens=max_training_tokens,
            max_compute_seconds=10_000,
            reward_calibration_labels=reward_calibration_labels,
            use_downstream_allocation=use_downstream_allocation,
        ),
    )
    output_root = tmp_path / "experiment-artifacts"
    factory = SyntheticRunnerFactory(gain_bpb)
    orchestrator = AutonomousResearchOrchestrator(
        OrchestratorConfig(
            repository_root=repository,
            ledger_path=ledger_path,
            data_root=tmp_path / "unused-data",
            output_root=output_root,
            workspace_root=tmp_path / "workspaces",
            researcher_artifact_root=tmp_path / "researcher-artifacts",
            reward_root=tmp_path / "reward",
            program_path=repository / "program.md",
            device="cpu",
            researcher_token_allowance=500,
            minimum_reward_labels=40,
        ),
        state=state,
        ledger=ledger,
        researcher=_researcher(tmp_path),
        policy=_policy(
            use_downstream_allocation=use_downstream_allocation,
            allocation_audit_fraction=0.0 if use_downstream_allocation else 0.10,
        ),
        runner_factory=factory,
    )
    return orchestrator, state, ledger, factory, repository


def test_campaign_resumes_across_calls_promotes_and_starts_from_new_parent(
    tmp_path: Path,
) -> None:
    orchestrator, state, ledger, factory, repository = _campaign(
        tmp_path,
        max_proposals=2,
        gain_bpb=0.05,
    )
    initial_parent = ledger.current_parent()

    first = orchestrator.run(max_new_proposals=1)
    first_parent = ledger.current_parent()

    assert first["status"] == "running"
    assert first_parent != initial_parent
    assert state.snapshot().generation == 1
    assert factory.registration_calls == 1
    assert [stage for stage, _seeds in factory.run_calls] == [
        ExperimentStage.CHEAP,
        ExperimentStage.INTERMEDIATE,
        ExperimentStage.FULL,
    ]

    second = orchestrator.run()

    assert second["status"] == "completed"
    assert second["generation"] == 2
    assert ledger.current_parent() not in {initial_parent, first_parent}
    assert factory.registration_calls == 2
    assert len(factory.run_calls) == 6
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert [decision.verdict for decision in decisions] == [
        DecisionVerdict.ESCALATE,
        DecisionVerdict.ESCALATE,
        DecisionVerdict.PROMOTE,
        DecisionVerdict.ESCALATE,
        DecisionVerdict.ESCALATE,
        DecisionVerdict.PROMOTE,
    ]
    assert (
        len(
            [
                event.record
                for event in ledger.events()
                if isinstance(event.record, DownstreamPrediction)
            ]
        )
        >= 1
    )
    snapshot = state.snapshot()
    assert snapshot.used.proposals == 2
    assert snapshot.used.researcher_tokens == 40
    assert snapshot.used.training_tokens == 5_600
    assert snapshot.reserved.training_tokens == 0
    assert (tmp_path / "reward" / "model.json").is_file()
    model = json.loads((tmp_path / "reward" / "model.json").read_text(encoding="utf-8"))
    assert model["label_count"] == 2
    assert model["minimum_label_count"] == 40

    second_transcript = json.loads(
        (tmp_path / "researcher-artifacts" / "research-campaign-001-2.json").read_text(
            encoding="utf-8"
        )
    )
    second_prompt = json.loads(second_transcript["prompt"])
    assert second_prompt["parent_commit"] == first_parent
    assert second_prompt["previous_results"]
    assert any(item["verdict"] == "promote" for item in second_prompt["previous_results"])
    assert _git(repository, "rev-parse", "refs/autodidact/accepted") == (ledger.current_parent())
    assert list((tmp_path / "workspaces").iterdir()) == []

    calls_before_replay = list(factory.run_calls)
    replay = orchestrator.run()
    assert replay["status"] == "completed"
    assert replay["outcomes"] == []
    assert factory.run_calls == calls_before_replay


def test_bad_patch_is_rejected_without_advancing_parent(tmp_path: Path) -> None:
    orchestrator, state, ledger, factory, _repository_root = _campaign(
        tmp_path,
        max_proposals=1,
        gain_bpb=-0.02,
    )
    parent = ledger.current_parent()

    result = orchestrator.run()

    assert result["status"] == "completed"
    assert ledger.current_parent() == parent
    assert state.snapshot().generation == 0
    assert factory.run_calls == [(ExperimentStage.CHEAP, (11,))]
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert len(decisions) == 1
    assert decisions[0].verdict is DecisionVerdict.REJECT
    assert not (tmp_path / "reward" / "model.json").exists()


def test_calibration_campaign_collects_target_then_restores_early_rejection(
    tmp_path: Path,
) -> None:
    orchestrator, state, ledger, factory, _repository_root = _campaign(
        tmp_path,
        max_proposals=3,
        gain_bpb=-0.02,
        reward_calibration_labels=2,
    )
    parent = ledger.current_parent()

    result = orchestrator.run()

    assert result["status"] == "completed"
    assert ledger.current_parent() == parent
    assert state.snapshot().generation == 0
    assert factory.run_calls == [
        (ExperimentStage.CHEAP, (11,)),
        (ExperimentStage.INTERMEDIATE, (11, 23)),
        (ExperimentStage.FULL, (11, 23, 37)),
        (ExperimentStage.CHEAP, (11,)),
        (ExperimentStage.INTERMEDIATE, (11, 23)),
        (ExperimentStage.FULL, (11, 23, 37)),
        (ExperimentStage.CHEAP, (11,)),
    ]
    assert result["reward_calibration"] == {
        "active": False,
        "completed_labels": 2,
        "remaining_labels": 0,
        "target_labels": 2,
    }
    labels = load_labels(tmp_path / "reward" / "full-labels.jsonl")
    assert len({label.candidate_id for label in labels}) == 2
    model = load_model(tmp_path / "reward" / "model.json")
    assert model.calibrated
    assert model.label_count == 2
    assert model.minimum_label_count == 2
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert [decision.verdict for decision in decisions] == [
        DecisionVerdict.ESCALATE,
        DecisionVerdict.ESCALATE,
        DecisionVerdict.REJECT,
        DecisionVerdict.ESCALATE,
        DecisionVerdict.ESCALATE,
        DecisionVerdict.REJECT,
        DecisionVerdict.REJECT,
    ]
    assert all(
        decision.probability_threshold == 0.0
        for decision in decisions
        if decision.verdict is DecisionVerdict.ESCALATE
    )


def test_calibration_then_prediction_allocation_stops_bad_patch_before_full(
    tmp_path: Path,
) -> None:
    orchestrator, state, ledger, factory, _repository_root = _campaign(
        tmp_path,
        max_proposals=3,
        gain_bpb=-0.02,
        reward_calibration_labels=2,
        use_downstream_allocation=True,
    )
    parent = ledger.current_parent()

    result = orchestrator.run()

    assert result["status"] == "completed"
    assert ledger.current_parent() == parent
    assert state.snapshot().generation == 0
    assert factory.run_calls == [
        (ExperimentStage.CHEAP, (11,)),
        (ExperimentStage.INTERMEDIATE, (11, 23)),
        (ExperimentStage.FULL, (11, 23, 37)),
        (ExperimentStage.CHEAP, (11,)),
        (ExperimentStage.INTERMEDIATE, (11, 23)),
        (ExperimentStage.FULL, (11, 23, 37)),
        (ExperimentStage.CHEAP, (11,)),
        (ExperimentStage.INTERMEDIATE, (11, 23)),
    ]
    assert result["downstream_allocation"]["ready"] is True
    allocations = [
        event.record for event in ledger.events() if isinstance(event.record, DownstreamAllocation)
    ]
    assert len(allocations) == 1
    assert allocations[0].action is AllocationAction.STOP
    decision = ledger.get(allocations[0].planned_decision_id or "missing").record
    assert isinstance(decision, DecisionRecord)
    assert decision.verdict is DecisionVerdict.REJECT
    assert decision.downstream_prediction_id == allocations[0].downstream_prediction_id
    assert len(load_labels(tmp_path / "reward" / "full-labels.jsonl")) == 2


def test_uncertain_patch_runs_each_next_predetermined_seed(tmp_path: Path) -> None:
    orchestrator, state, ledger, factory, _repository_root = _campaign(
        tmp_path,
        max_proposals=1,
        gain_bpb=0.001,
    )
    parent = ledger.current_parent()

    result = orchestrator.run()

    assert result["status"] == "completed"
    assert ledger.current_parent() == parent
    assert state.snapshot().generation == 0
    assert factory.run_calls == [
        (ExperimentStage.CHEAP, (11,)),
        (ExperimentStage.CHEAP, (23,)),
        (ExperimentStage.CHEAP, (37,)),
        (ExperimentStage.CHEAP, (53,)),
        (ExperimentStage.CHEAP, (71,)),
    ]
    decisions = [
        event.record for event in ledger.events() if isinstance(event.record, DecisionRecord)
    ]
    assert len(decisions) == 1
    assert decisions[0].verdict is DecisionVerdict.REJECT
    assert "fixed seed pool exhausted" in decisions[0].reasons[0]


def test_training_budget_is_enforced_before_paired_runner_launch(tmp_path: Path) -> None:
    orchestrator, state, _ledger, factory, _repository_root = _campaign(
        tmp_path,
        max_proposals=1,
        gain_bpb=0.05,
        max_training_tokens=100,
    )

    result = orchestrator.run()

    assert result["status"] == "failed"
    assert state.snapshot().used.training_tokens == 0
    assert factory.run_calls == []
    assert factory.registration_calls == 1


def test_pause_request_prevents_new_research_invocation(tmp_path: Path) -> None:
    orchestrator, state, ledger, factory, _repository_root = _campaign(
        tmp_path,
        max_proposals=1,
        gain_bpb=0.05,
    )
    parent = ledger.current_parent()
    state.request_pause("pause before the next proposal")

    result = orchestrator.run()

    assert result["status"] == CampaignStatus.PAUSED.value
    assert ledger.current_parent() == parent
    assert factory.registration_calls == 0
    assert state.snapshot().used.proposals == 0


def test_cli_initializes_and_reports_campaign_without_researcher_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository, parent = _repository(tmp_path)
    ledger_path = tmp_path / "cli-ledger.sqlite3"
    state_path = tmp_path / "cli-state.sqlite3"
    common = [
        "--repository-root",
        str(repository),
        "--ledger-path",
        str(ledger_path),
        "--state-path",
        str(state_path),
    ]

    exit_code = main(
        [
            *common,
            "initialize",
            "--campaign-id",
            "campaign-cli-001",
            "--max-proposals",
            "2",
            "--max-wall-seconds",
            "3600",
            "--max-researcher-tokens",
            "10000",
            "--max-training-tokens",
            "100000",
            "--max-compute-seconds",
            "1000",
            "--reward-calibration-labels",
            "2",
            "--use-downstream-allocation",
        ]
    )

    assert exit_code == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["campaign"]["accepted_parent_commit"] == parent
    assert initialized["campaign"]["limits"]["reward_calibration_labels"] == 2
    assert initialized["campaign"]["limits"]["use_downstream_allocation"] is True
    assert initialized["downstream_allocation"]["ready"] is False
    assert initialized["reward_calibration"]["remaining_labels"] == 2
    assert _git(repository, "rev-parse", "refs/autodidact/accepted") == parent

    assert main([*common, "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["campaign"]["status"] == "running"
    assert status["ledger"]["event_count"] == 0
    assert status["downstream_allocation"] == {
        "enabled": True,
        "label_count": 0,
        "minimum_labels": 2,
        "ready": False,
        "reason": "reward calibration is still collecting labels",
    }
    assert status["reward_calibration"] == {
        "active": True,
        "completed_labels": 0,
        "remaining_labels": 2,
        "target_labels": 2,
    }
