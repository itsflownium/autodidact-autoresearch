from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from autodidact.ledger import (
    ExperimentLedger,
    LedgerConflictError,
    LedgerIntegrityError,
    LedgerPermissionError,
    LedgerStateError,
    WriterRole,
    main,
)
from autodidact.records import (
    DecisionVerdict,
    ExperimentStage,
    ResourceLimits,
    build_paired_result,
)
from tests.experiment_fixtures import (
    CANDIDATE_COMMIT,
    PARENT_COMMIT,
    digest,
    evidence_records,
    lifecycle_entries,
)


def create_ledger(tmp_path: Path) -> ExperimentLedger:
    return ExperimentLedger.create(
        tmp_path / "experiments.sqlite3",
        initial_parent_commit=PARENT_COMMIT,
        ledger_id="ledger-test-001",
    )


def test_complete_lifecycle_is_reconstructable_from_immutable_events(tmp_path: Path) -> None:
    ledger = create_ledger(tmp_path)
    appended = ledger.append_many(lifecycle_entries())

    verification = ledger.verify()
    summary = ledger.summary()
    points = ledger.progress_points()

    assert [event.sequence for event in appended] == list(range(1, 15))
    assert verification.event_count == 14
    assert verification.head_event_sha256 == appended[-1].event_sha256
    assert ledger.current_parent() == CANDIDATE_COMMIT
    assert ledger.running_trials() == ()
    assert ledger.get("decision-001").record.verdict is DecisionVerdict.PROMOTE
    assert summary["generation"] == 1
    assert summary["decision_counts"] == {"promote": 1}
    assert summary["compute"] == {
        "accelerator_seconds": 214.0,
        "estimated_cost_usd": 0.0,
        "evaluation_tokens": 500_000,
        "training_tokens": 4_000_000,
        "wall_seconds": 214.0,
    }
    assert points == [
        {
            "candidate_bpb_mean": 1.095,
            "candidate_id": "candidate-001",
            "event_sequence": 2,
            "experiment_index": 1,
            "mean_gain_bpb": pytest.approx(0.005),
            "paired_seed_count": 1,
            "parent_bpb_mean": 1.1,
            "promoted_parent_commit": CANDIDATE_COMMIT,
            "stage": "cheap",
            "status": "promote",
        }
    ]


def test_writer_roles_and_read_only_mode_are_enforced(tmp_path: Path) -> None:
    ledger = create_ledger(tmp_path)
    records = evidence_records()
    ledger.append(records["proposal"], writer_role=WriterRole.RESEARCH_AGENT)

    with pytest.raises(LedgerPermissionError, match="cannot append candidate"):
        ledger.append(records["candidate"], writer_role=WriterRole.RESEARCH_AGENT)

    read_only = ExperimentLedger.open(ledger.path, read_only=True)
    assert read_only.verify().event_count == 1
    with pytest.raises(LedgerPermissionError, match="read-only"):
        read_only.append(records["candidate"], writer_role=WriterRole.CONTROLLER)


def test_append_many_rolls_back_the_entire_batch_on_failure(tmp_path: Path) -> None:
    ledger = create_ledger(tmp_path)
    records = evidence_records()

    with pytest.raises(LedgerPermissionError):
        ledger.append_many(
            (
                (records["proposal"], WriterRole.RESEARCH_AGENT),
                (records["candidate"], WriterRole.RESEARCH_AGENT),
            )
        )

    assert ledger.verify().event_count == 0


def test_ensure_is_idempotent_but_rejects_changed_content(tmp_path: Path) -> None:
    ledger = create_ledger(tmp_path)
    proposal = evidence_records()["proposal"]

    first = ledger.ensure(proposal, writer_role=WriterRole.RESEARCH_AGENT)
    second = ledger.ensure(proposal, writer_role=WriterRole.RESEARCH_AGENT)
    assert first == second
    assert ledger.verify().event_count == 1

    changed = replace(proposal, title="A different immutable claim")
    with pytest.raises(LedgerConflictError, match="record ID already exists"):
        ledger.ensure(changed, writer_role=WriterRole.RESEARCH_AGENT)


def test_trial_seed_and_evaluation_contracts_cannot_drift(tmp_path: Path) -> None:
    ledger = create_ledger(tmp_path)
    records = evidence_records()
    ledger.append_many(lifecycle_entries()[:3])

    wrong_eval_budget = replace(records["parent_run"], evaluation_tokens=249_999)
    with pytest.raises(LedgerStateError, match="evaluation budget"):
        ledger.append(wrong_eval_budget, writer_role=WriterRole.EVALUATOR)

    ledger.append(records["parent_run"], writer_role=WriterRole.EVALUATOR)
    duplicate_trial = replace(records["trial"], trial_id="trial-duplicate-001")
    with pytest.raises(LedgerStateError, match="stage and seed"):
        ledger.append(duplicate_trial, writer_role=WriterRole.CONTROLLER)


def test_pair_requires_matched_data_order_and_both_manifests(tmp_path: Path) -> None:
    records = evidence_records()
    ledger = create_ledger(tmp_path)
    entries = lifecycle_entries()
    ledger.append_many((entries[0], entries[1], entries[2], entries[3], entries[6]))

    with pytest.raises(LedgerStateError, match="artifact manifests"):
        ledger.append(records["paired"], writer_role=WriterRole.EVALUATOR)

    mismatch_ledger = ExperimentLedger.create(
        tmp_path / "mismatch.sqlite3",
        initial_parent_commit=PARENT_COMMIT,
    )
    mismatched_candidate = replace(records["candidate_run"], data_order_sha256=digest("0"))
    mismatch_ledger.append_many(
        (
            entries[0],
            entries[1],
            entries[2],
            entries[3],
            entries[4],
            (mismatched_candidate, WriterRole.EVALUATOR),
            entries[7],
        )
    )
    with pytest.raises(LedgerStateError, match="different seeded data orders"):
        mismatch_ledger.append(records["paired"], writer_role=WriterRole.EVALUATOR)


def test_pair_resource_constraints_are_recomputed_by_the_ledger(tmp_path: Path) -> None:
    records = evidence_records()
    trial = replace(
        records["trial"],
        limits=replace(
            records["trial"].limits,
            max_peak_process_rss_bytes=610_000_000,
        ),
    )
    ledger = create_ledger(tmp_path)
    ledger.append_many(
        (
            (records["proposal"], WriterRole.RESEARCH_AGENT),
            (records["candidate"], WriterRole.CONTROLLER),
            (trial, WriterRole.CONTROLLER),
            (records["parent_run"], WriterRole.EVALUATOR),
            (records["parent_manifest"], WriterRole.EVALUATOR),
            (records["candidate_run"], WriterRole.EVALUATOR),
            (records["candidate_manifest"], WriterRole.EVALUATOR),
        )
    )

    with pytest.raises(LedgerStateError, match="protected run outcomes"):
        ledger.append(records["paired"], writer_role=WriterRole.EVALUATOR)

    correct_pair = build_paired_result(
        "pair-constrained-001",
        trial=trial,
        candidate_id=records["candidate"].candidate_id,
        parent=records["parent_run"],
        candidate=records["candidate_run"],
        constraint_failures=("peak_process_rss",),
    )
    ledger.append(correct_pair, writer_role=WriterRole.EVALUATOR)
    assert ledger.get(correct_pair.paired_result_id).record.constraints_passed is False


def test_estimates_and_predictions_must_follow_predeclared_evidence(tmp_path: Path) -> None:
    records = evidence_records()
    ledger = create_ledger(tmp_path)
    ledger.append_many(lifecycle_entries()[:10])

    forged_effect = replace(records["effect"], mean_gain_bpb=0.5)
    with pytest.raises(LedgerStateError, match="statistics"):
        ledger.append(forged_effect, writer_role=WriterRole.EVALUATOR)

    wrong_minimum = replace(records["effect"], minimum_useful_gain_bpb=0.002)
    with pytest.raises(LedgerStateError, match="proposal contract"):
        ledger.append(wrong_minimum, writer_role=WriterRole.EVALUATOR)

    incomplete = ExperimentLedger.create(
        tmp_path / "incomplete.sqlite3",
        initial_parent_commit=PARENT_COMMIT,
    )
    incomplete.append_many(lifecycle_entries()[:3])
    with pytest.raises(LedgerStateError, match="completed paired results"):
        incomplete.append(records["prediction"], writer_role=WriterRole.EVALUATOR)


def test_stale_parents_cannot_receive_new_proposals_after_promotion(tmp_path: Path) -> None:
    ledger = create_ledger(tmp_path)
    ledger.append_many(lifecycle_entries())
    stale = replace(
        evidence_records()["proposal"],
        proposal_id="proposal-stale-001",
        title="Stale proposal",
    )

    with pytest.raises(LedgerStateError, match="stale parent"):
        ledger.append(stale, writer_role=WriterRole.RESEARCH_AGENT)


def test_sqlite_triggers_block_mutation_and_hash_verification_detects_tampering(
    tmp_path: Path,
) -> None:
    ledger = create_ledger(tmp_path)
    ledger.append(evidence_records()["proposal"], writer_role=WriterRole.RESEARCH_AGENT)
    connection = sqlite3.connect(ledger.path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE events SET writer_role = 'controller'")
        connection.rollback()

        connection.execute("DROP TRIGGER events_no_update")
        payload = json.loads(
            connection.execute("SELECT payload_json FROM events WHERE sequence = 1").fetchone()[0]
        )
        payload["payload"]["title"] = "Tampered title"
        connection.execute(
            "UPDATE events SET payload_json = ? WHERE sequence = 1",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
        )
        connection.execute(
            """
            CREATE TRIGGER events_no_update
            BEFORE UPDATE ON events
            BEGIN
                SELECT RAISE(ABORT, 'ledger events are append-only');
            END
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(LedgerIntegrityError, match="payload hash mismatch"):
        ledger.verify()


def test_explicit_migration_preserves_the_event_head(tmp_path: Path) -> None:
    ledger = create_ledger(tmp_path)
    ledger.append(evidence_records()["proposal"], writer_role=WriterRole.RESEARCH_AGENT)
    before = ledger.verify()
    connection = sqlite3.connect(ledger.path)
    try:
        connection.execute("PRAGMA user_version = 0")
    finally:
        connection.close()

    ExperimentLedger.migrate(ledger.path)
    after = ledger.verify()
    assert after.event_count == before.event_count
    assert after.head_event_sha256 == before.head_event_sha256


def test_exports_are_sanitized_and_do_not_disclose_database_paths(tmp_path: Path) -> None:
    ledger = create_ledger(tmp_path)
    proposal = replace(evidence_records()["proposal"], title="Tune project-secret warmup")
    ledger.append(proposal, writer_role=WriterRole.RESEARCH_AGENT)
    snapshot_path = tmp_path / "snapshot.json"
    jsonl_path = tmp_path / "events.jsonl"

    ledger.export(snapshot_path, redactions=("project-secret",))
    ledger.export(jsonl_path, output_format="jsonl", redactions=("project-secret",))

    snapshot_text = snapshot_path.read_text(encoding="utf-8")
    snapshot = json.loads(snapshot_text)
    jsonl = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert "project-secret" not in snapshot_text
    assert str(ledger.path) not in snapshot_text
    assert snapshot["events"][0]["record"]["payload"]["title"] == ("Tune <redacted> warmup")
    assert snapshot["ledger"]["event_count"] == 1
    assert len(jsonl) == 2


def test_machine_local_paths_are_rejected_before_append(tmp_path: Path) -> None:
    ledger = create_ledger(tmp_path)
    proposal = replace(
        evidence_records()["proposal"],
        title=f"Read local file {Path.home()}/private.txt",
    )
    with pytest.raises(LedgerStateError, match="machine-local"):
        ledger.append(proposal, writer_role=WriterRole.RESEARCH_AGENT)
    assert ledger.verify().event_count == 0


def test_cli_initializes_verifies_summarizes_and_exports(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "cli.sqlite3"
    export_path = tmp_path / "cli-export.json"

    assert main(["--path", str(path), "init", "--initial-parent", PARENT_COMMIT]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["event_count"] == 0

    assert main(["--path", str(path), "verify"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["verified"] is True

    assert main(["--path", str(path), "summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["current_parent_commit"] == PARENT_COMMIT

    assert (
        main(
            [
                "--path",
                str(path),
                "export",
                "--output",
                str(export_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert json.loads(export_path.read_text(encoding="utf-8"))["ledger"]["event_count"] == 0


def test_invalid_creation_contract_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(LedgerStateError, match="structured ID"):
        ExperimentLedger.create(
            tmp_path / "invalid.sqlite3",
            initial_parent_commit=PARENT_COMMIT,
            ledger_id="Not Portable",
        )
    with pytest.raises(LedgerStateError, match="full Git commit"):
        ExperimentLedger.create(
            tmp_path / "invalid-commit.sqlite3",
            initial_parent_commit="short",
        )


def test_prediction_target_must_be_later_than_source_stage(tmp_path: Path) -> None:
    records = evidence_records()
    ledger = create_ledger(tmp_path)
    ledger.append_many(lifecycle_entries()[:10])
    prediction = replace(
        records["prediction"],
        prediction_id="prediction-invalid-001",
        target_stage=ExperimentStage.CHEAP,
    )
    with pytest.raises(LedgerStateError, match="target must be later"):
        ledger.append(prediction, writer_role=WriterRole.EVALUATOR)


def test_resource_limit_values_remain_versioned_in_trial_records() -> None:
    limits = ResourceLimits(timeout_seconds=30)
    assert limits.max_parameter_count == 1_050_000
    assert limits.max_peak_process_rss_bytes is None
