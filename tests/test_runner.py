from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from autodidact.checkpoints import file_sha256
from autodidact.controller import PatchRCTController
from autodidact.data.integrity import verify_dataset
from autodidact.ledger import ExperimentLedger, WriterRole
from autodidact.records import (
    ArtifactManifest,
    CandidateRecord,
    ComputeRecord,
    ExperimentStage,
    PairedResult,
    PatchProposal,
    ResourceLimits,
    RunArm,
    RunExecutionMode,
    RunPlan,
    RunResult,
    RunStatus,
    TrialSchedule,
    TrialSpec,
)
from autodidact.runner import (
    ExperimentRequest,
    PairedExperimentRunner,
    ProcessOutcome,
    RunnerError,
    assign_execution_orders,
    build_parser,
    classify_failure,
    prepare_public_data_view,
    request_from_args,
    run_process,
    validate_candidate_patch,
)

ROOT = Path(__file__).resolve().parents[1]


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
    (repository / "README.md").write_text("test repository\n", encoding="utf-8")
    _git(repository, "add", "train.py", "README.md")
    _git(repository, "commit", "-m", "Add parent")
    parent = _git(repository, "rev-parse", "HEAD")
    (repository / "train.py").write_text(
        "BASELINE = True\n# candidate treatment\n", encoding="utf-8"
    )
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add candidate")
    candidate = _git(repository, "rev-parse", "HEAD")
    return repository, parent, candidate


def _proposal(parent: str) -> PatchProposal:
    return PatchProposal(
        proposal_id="proposal-runner-001",
        parent_commit=parent,
        title="Candidate treatment",
        hypothesis="The candidate should reduce held-out BPB.",
        mechanism="Exercise the protected runner contract.",
        change="Change train.py only.",
        expected_effect_bpb=0.01,
        minimum_useful_gain_bpb=0.001,
        resource_risk="No expected resource change.",
        failure_signal="Candidate BPB does not improve.",
        interaction_risk="No known interaction.",
    )


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class FakeProcesses:
    def __init__(
        self,
        *,
        ledger_path: Path,
        expected_trial_count: int,
        fail_candidate: bool = False,
    ) -> None:
        self.ledger_path = ledger_path
        self.expected_trial_count = expected_trial_count
        self.fail_candidate = fail_candidate
        self.inspection_calls = 0
        self.training_calls = 0
        self.evaluation_calls = 0

    def __call__(
        self,
        command: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
    ) -> ProcessOutcome:
        del cwd, timeout_seconds
        command = list(command)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert "PYTHONHASHSEED" in environment

        if len(command) > 2 and command[2] == "inspect":
            self.inspection_calls += 1
            trainer = Path(_argument(command, "--trainer"))
            stdout_path.write_text(
                json.dumps(
                    {
                        "context_length": 256,
                        "event": "protected_inspection",
                        "parameter_count": 1_016_960,
                        "trainer_sha256": file_sha256(trainer),
                        "vocab_size": 1_792,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return ProcessOutcome(0, 0.1, 20_000_000)

        if len(command) > 2 and command[2] == "train":
            self.training_calls += 1
            records = ExperimentLedger.open(self.ledger_path, read_only=True).events()
            assert sum(isinstance(event.record, TrialSpec) for event in records) == (
                self.expected_trial_count
            )
            trainer = Path(command[1])
            is_candidate = "candidate treatment" in trainer.read_text(encoding="utf-8")
            public_root = Path(_argument(command, "--data-root"))
            assert not (public_root / "protected").exists()
            manifest = verify_dataset(public_root, scope="public")
            if is_candidate and self.fail_candidate:
                stderr_path.write_text(
                    "RuntimeError: MPS backend out of memory\n", encoding="utf-8"
                )
                return ProcessOutcome(1, 1.5, 650_000_000)
            seed = int(_argument(command, "--seed"))
            token_budget = int(_argument(command, "--token-budget"))
            stop_after_tokens = int(_argument(command, "--stop-after-tokens"))
            trajectory_milestones: list[int] = []
            if "--trajectory-milestones" in command:
                start = command.index("--trajectory-milestones") + 1
                end = next(
                    (
                        index
                        for index in range(start, len(command))
                        if command[index].startswith("--")
                    ),
                    len(command),
                )
                trajectory_milestones = [int(value) for value in command[start:end]]
            resume_sha256 = None
            start_tokens = 0
            if "--resume" in command:
                resume = Path(_argument(command, "--resume"))
                resume_sha256 = file_sha256(resume)
                start_tokens = int(resume.read_text(encoding="utf-8").split(":")[-1])
            checkpoint = Path(_argument(command, "--checkpoint-out"))
            metrics = Path(_argument(command, "--metrics-file"))
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(
                f"{'candidate' if is_candidate else 'parent'}-{seed}:{stop_after_tokens}",
                encoding="utf-8",
            )
            order_hash = hashlib.sha256(f"order-{seed}".encode()).hexdigest()
            events = [
                {
                    "event": "config",
                    "data_config_sha256": manifest["pipeline"]["config_sha256"],
                    "parameter_count": 1_016_960,
                    "resume_checkpoint_sha256": resume_sha256,
                    "seed": seed,
                    "stop_after_tokens": stop_after_tokens,
                    "target_tokens": token_budget,
                    "tokenizer_sha256": manifest["tokenizer"]["artifact"]["sha256"],
                    "trajectory_milestones": trajectory_milestones,
                },
                {
                    "event": "summary",
                    "data_order_sha256": order_hash,
                    "mean_train_loss": 1.2,
                    "parameter_count": 1_016_960,
                    "peak_device_allocated_bytes": None,
                    "peak_device_reserved_bytes": None,
                    "resume_checkpoint_sha256": resume_sha256,
                    "seed": seed,
                    "stop_after_tokens": stop_after_tokens,
                    "target_tokens": token_budget,
                    "tokens_seen": stop_after_tokens,
                    "training_tokens_this_process": stop_after_tokens - start_tokens,
                    "trajectory_milestones": trajectory_milestones,
                    "validation_bpb": 0.0,
                },
            ]
            metrics.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            return ProcessOutcome(0, 2.0, 600_000_000)

        if len(command) > 2 and command[2] == "evaluate":
            self.evaluation_calls += 1
            trainer = Path(_argument(command, "--trainer"))
            checkpoint = Path(_argument(command, "--checkpoint"))
            is_candidate = "candidate treatment" in trainer.read_text(encoding="utf-8")
            maximum_tokens = int(_argument(command, "--maximum-tokens"))
            stdout_path.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "evaluation_seconds": 0.5,
                        "evaluation_tokens_per_second": maximum_tokens / 0.5,
                        "event": "protected_evaluation",
                        "parameter_count": 1_016_960,
                        "peak_device_allocated_bytes": None,
                        "peak_device_reserved_bytes": None,
                        "peak_process_rss_bytes": 300_000_000,
                        "predicted_tokens": maximum_tokens,
                        "trainer_sha256": file_sha256(trainer),
                        "validation_bpb": 1.09 if is_candidate else 1.10,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return ProcessOutcome(0, 0.6, 300_000_000)
        raise AssertionError(f"unexpected process command: {command}")


def _request(
    tmp_path: Path,
    prepared_dataset: Path,
    repository: Path,
    ledger_path: Path,
    candidate: str,
    *,
    seeds: tuple[int, ...] = (11, 23),
) -> ExperimentRequest:
    return ExperimentRequest(
        repository_root=repository,
        ledger_path=ledger_path,
        data_root=prepared_dataset,
        output_root=tmp_path / "experiments",
        proposal_id="proposal-runner-001",
        candidate_commit=candidate,
        stage=ExperimentStage.CHEAP,
        seeds=seeds,
        assignment_seed=101,
        token_budget=128,
        eval_tokens=128,
        batch_size=2,
        eval_batch_size=2,
        timeout_seconds=30,
        device="cpu",
        limits=ResourceLimits(
            timeout_seconds=30,
            max_peak_process_rss_bytes=900_000_000,
            max_training_throughput_regression_fraction=0.2,
            max_peak_process_rss_regression_fraction=0.2,
        ),
    )


def test_candidate_validation_requires_one_train_only_commit(tmp_path: Path) -> None:
    repository, parent, candidate = _repository(tmp_path)
    validation = validate_candidate_patch(
        repository,
        parent_commit=parent,
        candidate_commit=candidate,
    )
    assert validation.changed_paths == ("train.py",)
    assert len(validation.diff_sha256) == 64

    (repository / "README.md").write_text("changed protected file\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "-m", "Change protected file")
    protected_candidate = _git(repository, "rev-parse", "HEAD")
    with pytest.raises(RunnerError, match="protected path"):
        validate_candidate_patch(
            repository,
            parent_commit=candidate,
            candidate_commit=protected_candidate,
        )


def test_seeded_execution_order_is_reproducible_and_balanced() -> None:
    first = assign_execution_orders((11, 23, 37, 41), assignment_seed=99)
    second = assign_execution_orders((11, 23, 37, 41), assignment_seed=99)

    assert first == second
    assert sum(order[0] is RunArm.PARENT for order in first) == 2
    assert sum(order[0] is RunArm.CANDIDATE for order in first) == 2
    assert all(set(order) == {RunArm.PARENT, RunArm.CANDIDATE} for order in first)


def test_public_data_view_uses_hardlinks_without_exposing_protected_data(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    view = prepare_public_data_view(prepared_dataset, tmp_path / "public-view")
    manifest = verify_dataset(view, scope="public")
    artifact = manifest["tokenizer"]["artifact"]["path"]

    assert not (view / "protected").exists()
    assert view.stat().st_mode & 0o222 == 0
    assert (view / "public").stat().st_mode & 0o222 == 0
    assert (
        os.stat(view / "public" / artifact).st_ino
        == os.stat(prepared_dataset / "public" / artifact).st_ino
    )


def test_process_runner_records_timeout_and_peak_rss(tmp_path: Path) -> None:
    success = run_process(
        [sys.executable, "-c", "x = bytearray(1000000)"],
        cwd=tmp_path,
        environment=dict(os.environ),
        stdout_path=tmp_path / "success.out",
        stderr_path=tmp_path / "success.err",
        timeout_seconds=5,
    )
    timeout = run_process(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        cwd=tmp_path,
        environment=dict(os.environ),
        stdout_path=tmp_path / "timeout.out",
        stderr_path=tmp_path / "timeout.err",
        timeout_seconds=1,
    )

    assert success.returncode == 0
    assert success.peak_process_rss_bytes is not None
    assert timeout.timed_out is True
    assert timeout.wall_seconds < 2

    launch_failure = run_process(
        [str(tmp_path / "missing-executable")],
        cwd=tmp_path,
        environment=dict(os.environ),
        stdout_path=tmp_path / "missing.out",
        stderr_path=tmp_path / "missing.err",
        timeout_seconds=1,
    )
    assert launch_failure.returncode == 126


def test_failure_classification_is_structured() -> None:
    assert (
        classify_failure(ProcessOutcome(1, 1.0, None), "CUDA out of memory", phase="training")[0]
        is RunStatus.OOM
    )
    assert (
        classify_failure(ProcessOutcome(None, 1.0, None, timed_out=True), "", phase="training")[0]
        is RunStatus.TIMEOUT
    )
    assert (
        classify_failure(ProcessOutcome(2, 1.0, None), "non-finite loss", phase="training")[0]
        is RunStatus.NON_FINITE
    )


def test_end_to_end_runner_records_matched_evidence_and_resumes_idempotently(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    repository, parent, candidate = _repository(tmp_path)
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = ExperimentLedger.create(ledger_path, initial_parent_commit=parent)
    ledger.append(_proposal(parent), writer_role=WriterRole.RESEARCH_AGENT)
    request = _request(tmp_path, prepared_dataset, repository, ledger_path, candidate)
    processes = FakeProcesses(ledger_path=ledger_path, expected_trial_count=2)
    runner = PairedExperimentRunner(request, process_runner=processes)

    result = runner.run()

    assert result["event"] == "paired_experiment_completed"
    assert len(result["pair_ids"]) == 2
    assert all(run["gain_bpb"] == pytest.approx(0.01) for run in result["runs"])
    events = ledger.events()
    assert sum(isinstance(event.record, CandidateRecord) for event in events) == 1
    assert sum(isinstance(event.record, TrialSpec) for event in events) == 2
    assert sum(isinstance(event.record, RunResult) for event in events) == 4
    assert sum(isinstance(event.record, ArtifactManifest) for event in events) == 4
    assert sum(isinstance(event.record, ComputeRecord) for event in events) == 4
    assert sum(isinstance(event.record, PairedResult) for event in events) == 2
    runs = [event.record for event in events if isinstance(event.record, RunResult)]
    assert all(run.validation_bpb in {1.09, 1.10} for run in runs)
    assert all(run.validation_bpb != 0.0 for run in runs)
    for trial in [event.record for event in events if isinstance(event.record, TrialSpec)]:
        trial_runs = [run for run in runs if run.trial_id == trial.trial_id]
        assert len({run.data_order_sha256 for run in trial_runs}) == 1
    assert list((request.output_root / ".control" / "worktrees").iterdir()) == []

    training_calls = processes.training_calls
    evaluation_calls = processes.evaluation_calls
    repeated = runner.run()
    assert repeated == result
    assert processes.training_calls == training_calls
    assert processes.evaluation_calls == evaluation_calls


def test_runner_keeps_output_contracts_separate_across_stages_and_seed_batches(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    repository, parent, candidate_commit = _repository(tmp_path)
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = ExperimentLedger.create(ledger_path, initial_parent_commit=parent)
    ledger.append(_proposal(parent), writer_role=WriterRole.RESEARCH_AGENT)

    cheap_request = _request(
        tmp_path,
        prepared_dataset,
        repository,
        ledger_path,
        candidate_commit,
        seeds=(11,),
    )
    PairedExperimentRunner(
        cheap_request,
        process_runner=FakeProcesses(ledger_path=ledger_path, expected_trial_count=1),
    ).run()
    candidate = next(
        event.record for event in ledger.events() if isinstance(event.record, CandidateRecord)
    )
    candidate_root = cheap_request.output_root / candidate.candidate_id
    [cheap_contract_path] = list((candidate_root / "cheap").glob("contract-*.json"))
    assert json.loads(cheap_contract_path.read_text(encoding="utf-8"))["stage"] == "cheap"

    # Legacy candidates retained one contract at their root. It must not block another stage.
    legacy_contract_path = candidate_root / "contract.json"
    cheap_contract_path.replace(legacy_contract_path)
    intermediate_request = replace(
        cheap_request,
        stage=ExperimentStage.INTERMEDIATE,
        seeds=(11, 23),
        token_budget=256,
        eval_tokens=256,
    )

    result = PairedExperimentRunner(
        intermediate_request,
        process_runner=FakeProcesses(ledger_path=ledger_path, expected_trial_count=3),
    ).run()

    assert len(result["pair_ids"]) == 2
    assert json.loads(legacy_contract_path.read_text(encoding="utf-8"))["stage"] == "cheap"
    [intermediate_contract_path] = list((candidate_root / "intermediate").glob("contract-*.json"))
    intermediate_contract = json.loads(intermediate_contract_path.read_text(encoding="utf-8"))
    assert intermediate_contract["stage"] == "intermediate"
    assert intermediate_contract["seeds"] == [11, 23]

    next_seed_request = replace(intermediate_request, seeds=(37,))
    next_seed_result = PairedExperimentRunner(
        next_seed_request,
        process_runner=FakeProcesses(ledger_path=ledger_path, expected_trial_count=4),
    ).run()

    assert len(next_seed_result["pair_ids"]) == 1
    intermediate_contracts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((candidate_root / "intermediate").glob("contract-*.json"))
    ]
    assert sorted(contract["seeds"] for contract in intermediate_contracts) == [
        [11, 23],
        [37],
    ]


def test_runner_continues_milestones_and_reuses_only_compatible_parent(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    repository, parent, first_candidate = _repository(tmp_path)
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = ExperimentLedger.create(ledger_path, initial_parent_commit=parent)
    ledger.append(_proposal(parent), writer_role=WriterRole.RESEARCH_AGENT)
    cheap = replace(
        _request(
            tmp_path,
            prepared_dataset,
            repository,
            ledger_path,
            first_candidate,
            seeds=(11,),
        ),
        trajectory_token_budget=384,
        trajectory_milestones=(128, 256),
    )
    PairedExperimentRunner(
        cheap,
        process_runner=FakeProcesses(ledger_path=ledger_path, expected_trial_count=1),
    ).run()

    intermediate = replace(
        cheap,
        stage=ExperimentStage.INTERMEDIATE,
        token_budget=256,
        eval_tokens=256,
    )
    intermediate_processes = FakeProcesses(
        ledger_path=ledger_path,
        expected_trial_count=2,
    )
    result = PairedExperimentRunner(
        intermediate,
        process_runner=intermediate_processes,
    ).run()

    assert len(result["pair_ids"]) == 1
    assert intermediate_processes.training_calls == 2
    intermediate_trial_id = result["runs"][0]["trial_id"]
    intermediate_plans = [
        event.record
        for event in ledger.events()
        if isinstance(event.record, RunPlan) and event.record.trial_id == intermediate_trial_id
    ]
    assert {plan.execution_mode for plan in intermediate_plans} == {RunExecutionMode.CONTINUE}
    continued_compute = [
        event.record
        for event in ledger.events()
        if isinstance(event.record, ComputeRecord)
        and event.record.trial_id == intermediate_trial_id
    ]
    assert [record.training_tokens for record in continued_compute] == [128, 128]

    _git(repository, "switch", "-c", "candidate-two", parent)
    (repository / "train.py").write_text(
        "BASELINE = True\n# second candidate treatment\n",
        encoding="utf-8",
    )
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add second candidate")
    second_candidate = _git(repository, "rev-parse", "HEAD")
    second_proposal = replace(
        _proposal(parent),
        proposal_id="proposal-runner-002",
        title="Second candidate treatment",
    )
    ledger.append(second_proposal, writer_role=WriterRole.RESEARCH_AGENT)
    second_request = replace(
        cheap,
        proposal_id=second_proposal.proposal_id,
        candidate_commit=second_candidate,
    )
    second_processes = FakeProcesses(
        ledger_path=ledger_path,
        expected_trial_count=3,
    )

    second_result = PairedExperimentRunner(
        second_request,
        process_runner=second_processes,
    ).run()

    assert len(second_result["pair_ids"]) == 1
    assert second_processes.training_calls == 1
    assert second_processes.evaluation_calls == 1
    assert second_result["runs"][0]["parent_execution_mode"] == "reuse"
    assert second_result["runs"][0]["candidate_execution_mode"] == "fresh"
    second_trial_id = second_result["runs"][0]["trial_id"]
    second_compute = [
        event.record
        for event in ledger.events()
        if isinstance(event.record, ComputeRecord) and event.record.trial_id == second_trial_id
    ]
    assert sorted(record.training_tokens for record in second_compute) == [0, 128]
    assert ledger.summary()["compute"]["training_tokens"] == 640

    _git(repository, "switch", "-c", "candidate-three", parent)
    (repository / "train.py").write_text(
        "BASELINE = True\n# third candidate treatment\n",
        encoding="utf-8",
    )
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add third candidate")
    third_candidate = _git(repository, "rev-parse", "HEAD")
    third_proposal = replace(
        _proposal(parent),
        proposal_id="proposal-runner-003",
        title="Third candidate treatment",
    )
    ledger.append(third_proposal, writer_role=WriterRole.RESEARCH_AGENT)
    incompatible_request = replace(
        cheap,
        proposal_id=third_proposal.proposal_id,
        candidate_commit=third_candidate,
        batch_size=4,
    )
    incompatible_processes = FakeProcesses(
        ledger_path=ledger_path,
        expected_trial_count=4,
    )

    incompatible_result = PairedExperimentRunner(
        incompatible_request,
        process_runner=incompatible_processes,
    ).run()

    assert incompatible_processes.training_calls == 2
    assert incompatible_result["runs"][0]["parent_execution_mode"] == "fresh"


def test_parent_reuse_fails_closed_after_checkpoint_tampering(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    repository, parent, first_candidate = _repository(tmp_path)
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = ExperimentLedger.create(ledger_path, initial_parent_commit=parent)
    ledger.append(_proposal(parent), writer_role=WriterRole.RESEARCH_AGENT)
    request = _request(
        tmp_path,
        prepared_dataset,
        repository,
        ledger_path,
        first_candidate,
        seeds=(11,),
    )
    PairedExperimentRunner(
        request,
        process_runner=FakeProcesses(ledger_path=ledger_path, expected_trial_count=1),
    ).run()
    parent_run = next(
        event.record
        for event in ledger.events()
        if isinstance(event.record, RunResult) and event.record.arm is RunArm.PARENT
    )
    parent_manifest = next(
        event.record
        for event in ledger.events()
        if isinstance(event.record, ArtifactManifest) and event.record.run_id == parent_run.run_id
    )
    checkpoint = next(
        artifact for artifact in parent_manifest.artifacts if artifact.kind == "checkpoint"
    )
    checkpoint_path = request.output_root / checkpoint.relative_path
    checkpoint_path.write_text("tampered", encoding="utf-8")

    _git(repository, "switch", "-c", "candidate-two", parent)
    (repository / "train.py").write_text(
        "BASELINE = True\n# second candidate treatment\n",
        encoding="utf-8",
    )
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add second candidate")
    second_candidate = _git(repository, "rev-parse", "HEAD")
    second_proposal = replace(
        _proposal(parent),
        proposal_id="proposal-runner-002",
        title="Second candidate treatment",
    )
    ledger.append(second_proposal, writer_role=WriterRole.RESEARCH_AGENT)
    second_request = replace(
        request,
        proposal_id=second_proposal.proposal_id,
        candidate_commit=second_candidate,
    )

    with pytest.raises(RunnerError, match="hash verification"):
        PairedExperimentRunner(
            second_request,
            process_runner=FakeProcesses(
                ledger_path=ledger_path,
                expected_trial_count=2,
            ),
        ).run()


def test_candidate_preflight_is_recorded_before_controller_schedule(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    repository, parent, candidate_commit = _repository(tmp_path)
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = ExperimentLedger.create(ledger_path, initial_parent_commit=parent)
    ledger.append(_proposal(parent), writer_role=WriterRole.RESEARCH_AGENT)
    request = _request(
        tmp_path,
        prepared_dataset,
        repository,
        ledger_path,
        candidate_commit,
        seeds=(11,),
    )
    processes = FakeProcesses(ledger_path=ledger_path, expected_trial_count=0)
    runner = PairedExperimentRunner(request, process_runner=processes)

    candidate = runner.register_candidate()
    inspection_calls = processes.inspection_calls
    assert runner.register_candidate() == candidate
    assert processes.inspection_calls == inspection_calls
    schedule = PatchRCTController(ledger).initialize(candidate.candidate_id)

    events = ledger.events()
    candidate_sequence = next(
        event.sequence for event in events if isinstance(event.record, CandidateRecord)
    )
    schedule_sequence = next(
        event.sequence for event in events if isinstance(event.record, TrialSchedule)
    )
    assert candidate_sequence < schedule_sequence
    assert schedule["action"] == "schedule"
    assert not any(isinstance(event.record, TrialSpec) for event in events)


def test_runner_records_candidate_oom_and_still_completes_parent_arm(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    repository, parent, candidate = _repository(tmp_path)
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = ExperimentLedger.create(ledger_path, initial_parent_commit=parent)
    ledger.append(_proposal(parent), writer_role=WriterRole.RESEARCH_AGENT)
    request = _request(
        tmp_path,
        prepared_dataset,
        repository,
        ledger_path,
        candidate,
        seeds=(11,),
    )
    processes = FakeProcesses(
        ledger_path=ledger_path,
        expected_trial_count=1,
        fail_candidate=True,
    )

    result = PairedExperimentRunner(request, process_runner=processes).run()

    assert result["pair_ids"] == []
    runs = [event.record for event in ledger.events() if isinstance(event.record, RunResult)]
    assert {run.arm: run.status for run in runs} == {
        RunArm.PARENT: RunStatus.SUCCEEDED,
        RunArm.CANDIDATE: RunStatus.OOM,
    }
    assert ledger.running_trials() == ()
    assert processes.training_calls == 2


def test_real_runner_launches_both_training_processes_and_protected_evaluation(
    baseline_dataset: Path,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "real-repository"
    subprocess.run(
        ["git", "clone", "--shared", "--quiet", str(ROOT), str(repository)],
        check=True,
    )
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    parent = _git(repository, "rev-parse", "HEAD")
    trainer = repository / "train.py"
    trainer.write_text(
        trainer.read_text(encoding="utf-8") + "\n# paired runner smoke candidate\n",
        encoding="utf-8",
    )
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add smoke candidate")
    candidate = _git(repository, "rev-parse", "HEAD")
    ledger_path = tmp_path / "real-ledger.sqlite3"
    ledger = ExperimentLedger.create(ledger_path, initial_parent_commit=parent)
    ledger.append(_proposal(parent), writer_role=WriterRole.RESEARCH_AGENT)
    request = _request(
        tmp_path,
        baseline_dataset,
        repository,
        ledger_path,
        candidate,
        seeds=(11,),
    )

    result = PairedExperimentRunner(request).run()

    assert len(result["pair_ids"]) == 1
    pair = next(event.record for event in ledger.events() if isinstance(event.record, PairedResult))
    assert pair.gain_bpb == pytest.approx(0.0, abs=1e-12)
    assert pair.constraints_passed is True
    runs = [event.record for event in ledger.events() if isinstance(event.record, RunResult)]
    assert len(runs) == 2
    assert all(run.status is RunStatus.SUCCEEDED for run in runs)
    assert all(run.evaluation_tokens > 0 for run in runs)


def test_request_defaults_follow_stage_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--proposal-id",
            "proposal-runner-001",
            "--candidate-commit",
            "a" * 40,
            "--stage",
            "intermediate",
            "--seeds",
            "11",
            "--assignment-seed",
            "7",
        ]
    )
    request = request_from_args(args)
    assert request.token_budget == 6_000_000
    assert request.eval_tokens == 1_000_000
