from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from autodidact.records import (
    ArtifactRef,
    ArtifactRetention,
    CandidateRecord,
    DecisionRecord,
    ExperimentStage,
    RecordValidationError,
    RunArm,
    RunStatus,
    build_effect_estimate,
    build_paired_result,
    record_from_envelope,
    record_to_envelope,
)
from tests.experiment_fixtures import digest, evidence_records


def test_every_record_round_trips_through_a_json_envelope() -> None:
    for record in evidence_records().values():
        envelope = record_to_envelope(record)
        decoded = json.loads(json.dumps(envelope, allow_nan=False))

        assert record_from_envelope(decoded) == record
        assert decoded["record_id"] in decoded["payload"].values()
        assert decoded["schema_version"] == 2


def test_paired_and_effect_builders_recompute_protected_statistics() -> None:
    records = evidence_records()
    trial = records["trial"]
    parent = records["parent_run"]
    candidate = records["candidate_run"]
    pair = records["paired"]
    assert pair.objective_gain == pytest.approx(0.005)
    assert pair.training_throughput_delta == 1_000.0
    assert pair.peak_process_rss_delta_bytes == 20_000_000

    second_trial = replace(trial, trial_id="trial-002", seed=23)
    second_parent = replace(
        parent,
        run_id="run-parent-002",
        trial_id=second_trial.trial_id,
        seed=second_trial.seed,
        data_order_sha256=digest("0"),
    )
    second_candidate = replace(
        candidate,
        run_id="run-candidate-002",
        trial_id=second_trial.trial_id,
        seed=second_trial.seed,
        objective_value=1.097,
        data_order_sha256=second_parent.data_order_sha256,
    )
    second_pair = build_paired_result(
        "pair-002",
        trial=second_trial,
        candidate_id=records["candidate"].candidate_id,
        parent=second_parent,
        candidate=second_candidate,
    )
    effect = build_effect_estimate(
        "estimate-002",
        candidate_id=records["candidate"].candidate_id,
        stage=ExperimentStage.CHEAP,
        pairs=(pair, second_pair),
        minimum_useful_gain=0.001,
        probability_exceeds_minimum=0.95,
        estimator_version="paired-normal-v1",
    )

    assert effect.mean_objective_gain == pytest.approx(0.004)
    assert effect.sample_variance == pytest.approx(0.000002)
    assert effect.standard_error == pytest.approx(0.001)
    assert effect.seeds == (11, 23)


def test_envelope_parser_rejects_schema_drift_and_invalid_enums() -> None:
    envelope = record_to_envelope(evidence_records()["trial"])
    envelope["payload"]["stage"] = "unknown"
    with pytest.raises(RecordValidationError, match="invalid trial_spec record"):
        record_from_envelope(envelope)

    envelope = record_to_envelope(evidence_records()["trial"])
    envelope["payload"]["unexpected"] = True
    with pytest.raises(RecordValidationError, match="unexpected unexpected"):
        record_from_envelope(envelope)

    envelope = record_to_envelope(evidence_records()["trial"])
    envelope["schema_version"] = True
    with pytest.raises(RecordValidationError, match="must be an integer"):
        record_from_envelope(envelope)

    with pytest.raises(RecordValidationError, match="must be an object"):
        record_from_envelope([])  # type: ignore[arg-type]


def test_records_reject_mutable_or_wrongly_typed_fields() -> None:
    records = evidence_records()
    with pytest.raises(RecordValidationError, match="immutable tuple"):
        replace(records["candidate"], changed_paths=["train.py"])
    with pytest.raises(RecordValidationError, match="must be an integer"):
        replace(records["trial"], seed=True)
    with pytest.raises(RecordValidationError, match="ExperimentStage"):
        replace(records["effect"], stage="cheap")
    with pytest.raises(RecordValidationError, match="must be a boolean"):
        replace(records["decision"], constraints_passed=1)


def test_candidate_records_keep_portable_paths_and_positive_parameter_counts() -> None:
    records = evidence_records()
    candidate = records["candidate"]
    assert isinstance(candidate, CandidateRecord)
    with pytest.raises(RecordValidationError, match="safe repository-relative"):
        replace(candidate, changed_paths=("../autodidact/ledger.py",))
    with pytest.raises(RecordValidationError, match="at least 1"):
        replace(candidate, parameter_count=0)
    assert replace(candidate, parameter_count=1_050_001).parameter_count == 1_050_001


def test_artifacts_must_use_portable_relative_paths() -> None:
    with pytest.raises(RecordValidationError, match="safe relative POSIX"):
        ArtifactRef(
            artifact_id="artifact-001",
            kind="metrics",
            relative_path="/tmp/private/metrics.jsonl",
            sha256=digest("1"),
            size_bytes=10,
            retention=ArtifactRetention.COMPACT,
        )


def test_success_and_failure_results_cannot_claim_incompatible_outcomes() -> None:
    run = evidence_records()["parent_run"]
    with pytest.raises(RecordValidationError, match="positive training throughput"):
        replace(run, training_tokens_per_second=0.0)
    with pytest.raises(RecordValidationError, match="exact token budget"):
        replace(run, tokens_seen=run.target_tokens - 1)
    with pytest.raises(RecordValidationError, match="failed runs cannot claim"):
        replace(run, status=RunStatus.CRASHED, failure_reason="worker exited")
    with pytest.raises(RecordValidationError, match="failure reason"):
        replace(
            run,
            status=RunStatus.CRASHED,
            objective_value=None,
            failure_reason=None,
        )


def test_non_finite_and_forged_derived_values_are_rejected() -> None:
    records = evidence_records()
    with pytest.raises(RecordValidationError, match="must be finite"):
        replace(records["proposal"], expected_effect=math.nan)
    with pytest.raises(RecordValidationError, match="must be finite"):
        replace(records["decision"], minimum_useful_gain=math.inf)
    with pytest.raises(RecordValidationError, match="must equal"):
        replace(records["paired"], objective_gain=1.0)


def test_direct_enum_values_are_required() -> None:
    run = evidence_records()["parent_run"]
    with pytest.raises(RecordValidationError, match="RunArm"):
        replace(run, arm="parent")
    with pytest.raises(RecordValidationError, match="RunStatus"):
        replace(run, status="succeeded")


def test_decision_shape_depends_on_verdict() -> None:
    decision = evidence_records()["decision"]
    assert isinstance(decision, DecisionRecord)
    with pytest.raises(RecordValidationError, match="promotion requires"):
        replace(decision, resulting_parent_commit=None)
    with pytest.raises(RecordValidationError, match="constraint failure"):
        replace(decision, constraints_passed=False)


def test_builder_requires_correct_run_arms() -> None:
    records = evidence_records()
    parent = replace(records["parent_run"], arm=RunArm.CANDIDATE)
    with pytest.raises(RecordValidationError, match="arms are reversed"):
        build_paired_result(
            "pair-003",
            trial=records["trial"],
            candidate_id=records["candidate"].candidate_id,
            parent=parent,
            candidate=records["candidate_run"],
        )
