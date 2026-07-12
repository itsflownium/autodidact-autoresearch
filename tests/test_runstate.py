from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from autodidact.runstate import (
    BudgetAmount,
    BudgetExceeded,
    CampaignLimits,
    CampaignStatus,
    CampaignStore,
    ClaimDisposition,
    OperationStatus,
    RepositoryLock,
    RepositoryLocked,
    ReservationStatus,
    RunStateConflict,
    RunStateError,
    main,
)

PARENT = "a" * 40
CANDIDATE = "b" * 40


class FakeClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _limits() -> CampaignLimits:
    return CampaignLimits(
        max_proposals=5,
        max_wall_seconds=3_600.0,
        max_researcher_tokens=50_000,
        max_training_tokens=100_000_000,
        max_compute_seconds=20_000.0,
    )


def _store(tmp_path: Path, *, clock: FakeClock | None = None) -> CampaignStore:
    return CampaignStore.create(
        tmp_path / "campaign.sqlite3",
        campaign_id="campaign-001",
        initial_parent_commit=PARENT,
        limits=_limits(),
        clock=clock or FakeClock(),
    )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    return repository


def test_campaign_creation_progress_and_persistent_snapshot(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock=clock)

    initial = store.snapshot()
    assert initial.status is CampaignStatus.RUNNING
    assert initial.phase == "ready"
    assert initial.accepted_parent_commit == PARENT
    assert initial.remaining.proposals == 5

    clock.value += 12.5
    updated = store.set_progress(
        phase="candidate_ready",
        accepted_parent_commit=CANDIDATE,
        generation=1,
        active_proposal_id="proposal-001",
        active_candidate_id="candidate-001",
    )
    reopened = CampaignStore.open(store.path, clock=clock).snapshot()

    assert updated.generation == 1
    assert reopened == updated
    assert reopened.elapsed_wall_seconds == pytest.approx(12.5)
    with pytest.raises(RunStateError, match="exactly one generation"):
        store.set_progress(
            phase="candidate_ready",
            accepted_parent_commit="c" * 40,
            generation=3,
            active_proposal_id=None,
            active_candidate_id=None,
        )


def test_repository_lock_excludes_second_controller(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first = RepositoryLock(repository, campaign_id="campaign-001")
    second = RepositoryLock(repository, campaign_id="campaign-002")

    with first:
        metadata = json.loads(first.lock_path.read_text(encoding="utf-8"))
        assert metadata["campaign_id"] == "campaign-001"
        with pytest.raises(RepositoryLocked, match="another campaign process"):
            second.acquire()

    with second:
        assert second.lock_path == first.lock_path


def test_budget_reservation_settlement_and_replay_are_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_operation("operation-001", "research", {"proposal": 1})
    requested = BudgetAmount(
        proposals=1,
        researcher_tokens=5_000,
        training_tokens=20_000,
        compute_seconds=100.0,
    )

    first = store.reserve_budget(
        "reservation-001",
        requested,
        operation_key="operation-001",
    )
    replay = store.reserve_budget(
        "reservation-001",
        requested,
        operation_key="operation-001",
    )
    assert first == replay
    assert store.snapshot().reserved == requested

    actual = BudgetAmount(
        proposals=1,
        researcher_tokens=1_200,
        training_tokens=18_000,
        compute_seconds=80.0,
    )
    settled = store.settle_budget("reservation-001", actual)
    assert settled.status is ReservationStatus.SETTLED
    assert store.settle_budget("reservation-001", actual) == settled
    snapshot = store.snapshot()
    assert snapshot.used == actual
    assert snapshot.reserved == BudgetAmount()


def test_budget_is_rejected_before_work_and_actual_cannot_exceed_reservation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    with pytest.raises(BudgetExceeded, match="exceed campaign limits"):
        store.reserve_budget(
            "reservation-too-large",
            BudgetAmount(proposals=6),
        )

    store.reserve_budget(
        "reservation-001",
        BudgetAmount(researcher_tokens=100),
    )
    with pytest.raises(BudgetExceeded, match="exceeds the prior reservation"):
        store.settle_budget(
            "reservation-001",
            BudgetAmount(researcher_tokens=101),
        )
    assert store.snapshot().reserved.researcher_tokens == 100
    released = store.release_budget("reservation-001")
    assert released.status is ReservationStatus.RELEASED
    assert store.release_budget("reservation-001") == released
    assert store.snapshot().reserved == BudgetAmount()


def test_wall_time_limit_blocks_new_operations_and_reservations(tmp_path: Path) -> None:
    clock = FakeClock()
    store = _store(tmp_path, clock=clock)
    clock.value += _limits().max_wall_seconds

    with pytest.raises(BudgetExceeded, match="wall-time"):
        store.ensure_can_work()
    with pytest.raises(BudgetExceeded, match="wall-time"):
        store.begin_operation("operation-late", "research", {})
    with pytest.raises(BudgetExceeded, match="wall-time"):
        store.reserve_budget("reservation-late", BudgetAmount(proposals=1))


def test_completed_operation_replays_without_executing_again(tmp_path: Path) -> None:
    store = _store(tmp_path)
    claim = store.begin_operation(
        "operation-001",
        "paired-run",
        {"candidate_id": "candidate-001", "seed": 11},
    )
    assert claim.disposition is ClaimDisposition.EXECUTE
    assert claim.attempts == 1

    completed = store.complete_operation("operation-001", {"pair_id": "pair-001"})
    assert completed.status is OperationStatus.SUCCEEDED
    replay = store.begin_operation(
        "operation-001",
        "paired-run",
        {"candidate_id": "candidate-001", "seed": 11},
    )
    assert replay.disposition is ClaimDisposition.REPLAY
    assert replay.result == {"pair_id": "pair-001"}

    with pytest.raises(RunStateConflict, match="different inputs"):
        store.begin_operation(
            "operation-001",
            "paired-run",
            {"candidate_id": "candidate-001", "seed": 23},
        )
    with pytest.raises(RunStateConflict, match="different result"):
        store.complete_operation("operation-001", {"pair_id": "pair-other"})


def test_interrupted_operation_requires_reconciliation_before_restart(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_operation("operation-001", "research", {"proposal": 1})

    assert store.mark_running_interrupted() == 1
    recovery = store.begin_operation("operation-001", "research", {"proposal": 1})
    assert recovery.disposition is ClaimDisposition.RECOVER
    assert recovery.status is OperationStatus.INTERRUPTED

    restarted = store.restart_interrupted("operation-001")
    assert restarted.disposition is ClaimDisposition.EXECUTE
    assert restarted.attempts == 2
    with pytest.raises(RunStateError, match="only interrupted"):
        store.restart_interrupted("operation-001")


def test_interrupted_operation_can_be_reconciled_from_authoritative_evidence(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.begin_operation("operation-001", "paired-run", {"trial": "trial-001"})
    store.mark_running_interrupted("host restarted")

    recovered = store.complete_operation(
        "operation-001",
        {"paired_result_id": "pair-001", "recovered": True},
    )

    assert recovered.status is OperationStatus.SUCCEEDED
    assert recovered.attempts == 1
    assert store.operation("operation-001").result["recovered"] is True


def test_failed_operation_is_terminal_and_replayed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.begin_operation("operation-001", "research", {"proposal": 1})
    failed = store.fail_operation("operation-001", "researcher returned no patch")

    assert failed.status is OperationStatus.FAILED
    replay = store.begin_operation("operation-001", "research", {"proposal": 1})
    assert replay.disposition is ClaimDisposition.REPLAY
    assert replay.error == "researcher returned no patch"
    assert store.fail_operation("operation-001", replay.error) == replay
    with pytest.raises(RunStateError, match="failed operations cannot be completed"):
        store.complete_operation("operation-001", {})


def test_pause_resume_and_cancel_apply_at_clean_checkpoints(tmp_path: Path) -> None:
    store = _store(tmp_path)

    requested = store.request_pause("finish the active operation first")
    assert requested.status is CampaignStatus.PAUSE_REQUESTED
    with pytest.raises(RunStateError, match="cannot start work"):
        store.begin_operation("operation-blocked", "research", {})
    assert store.checkpoint_control() is CampaignStatus.PAUSED
    assert store.resume().status is CampaignStatus.RUNNING

    assert store.request_cancel("stop after retained evidence").status is (
        CampaignStatus.CANCEL_REQUESTED
    )
    assert store.checkpoint_control() is CampaignStatus.CANCELLED
    with pytest.raises(RunStateError, match="cannot start work"):
        store.reserve_budget("reservation-blocked", BudgetAmount(proposals=1))

    paused = CampaignStore.create(
        tmp_path / "paused.sqlite3",
        campaign_id="campaign-002",
        initial_parent_commit=PARENT,
        limits=_limits(),
        clock=FakeClock(),
    )
    paused.request_pause("pause first")
    assert paused.checkpoint_control() is CampaignStatus.PAUSED
    assert paused.request_cancel("cancel while paused").status is (CampaignStatus.CANCEL_REQUESTED)


def test_campaign_terminal_states_are_idempotent(tmp_path: Path) -> None:
    completed = _store(tmp_path)
    assert completed.mark_completed().status is CampaignStatus.COMPLETED
    assert completed.mark_completed().status is CampaignStatus.COMPLETED
    with pytest.raises(RunStateError, match="different terminal state"):
        completed.mark_failed("unexpected failure")

    failed = CampaignStore.create(
        tmp_path / "failed.sqlite3",
        campaign_id="campaign-002",
        initial_parent_commit=PARENT,
        limits=_limits(),
        clock=FakeClock(),
    )
    assert failed.mark_failed("experiment contract failed").status is CampaignStatus.FAILED
    assert failed.mark_failed("experiment contract failed").status is CampaignStatus.FAILED


def test_integrity_verification_rejects_counter_tampering(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.reserve_budget("reservation-001", BudgetAmount(proposals=1))
    connection = sqlite3.connect(store.path)
    try:
        connection.execute("UPDATE campaign SET reserved_proposals = 2")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RunStateError, match="counters do not match"):
        CampaignStore.open(store.path)


def test_cli_creates_controls_and_recovers_campaign(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path = tmp_path / "cli.sqlite3"
    common = ["--state-path", str(state_path)]
    assert (
        main(
            [
                *common,
                "create",
                "--campaign-id",
                "campaign-cli-001",
                "--initial-parent",
                PARENT,
                "--max-proposals",
                "3",
                "--max-wall-seconds",
                "3600",
                "--max-researcher-tokens",
                "10000",
                "--max-training-tokens",
                "1000000",
                "--max-compute-seconds",
                "1000",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "running"

    store = CampaignStore.open(state_path)
    store.begin_operation("operation-cli-001", "research", {})
    assert main([*common, "recover"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "running"
    assert store.operation("operation-cli-001").status is OperationStatus.INTERRUPTED

    assert main([*common, "pause", "--reason", "manual pause"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pause_requested"
    assert main([*common, "resume"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "running"
