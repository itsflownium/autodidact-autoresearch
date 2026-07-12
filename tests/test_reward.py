from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from autodidact.checkpoints import file_sha256
from autodidact.ledger import ExperimentLedger, WriterRole
from autodidact.records import (
    ArtifactManifest,
    ArtifactRef,
    ArtifactRetention,
    DownstreamPrediction,
    ExperimentStage,
    RunResult,
    build_paired_result,
)
from autodidact.reward import (
    FEATURE_NAMES,
    BayesianRewardModel,
    FullBudgetLabel,
    LearningCurveFeatures,
    PredictiveDistribution,
    RewardError,
    build_downstream_prediction,
    build_full_budget_label,
    calibrate_model,
    extract_learning_curve_features,
    load_model,
    main,
    recommendation,
    save_model,
    student_t_cdf,
    student_t_quantile,
)
from tests.experiment_fixtures import PARENT_COMMIT, digest, evidence_records


def _feature(index: int, signal: float) -> LearningCurveFeatures:
    values = (
        1.0,
        1.0,
        0.3,
        signal,
        signal,
        signal * 0.1,
        0.001,
        signal * 2.0,
        signal,
        signal,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
    )
    return LearningCurveFeatures(
        feature_id=f"feature-synthetic-{index:03d}",
        candidate_id=f"candidate-synthetic-{index:03d}",
        source_trial_ids=(f"trial-synthetic-{index:03d}",),
        source_stages=(ExperimentStage.INTERMEDIATE,),
        artifact_sha256s=(digest(hex(index % 10)[2:]),),
        feature_names=FEATURE_NAMES,
        feature_values=values,
        captured_event_sequence=index + 1,
    )


def _label(index: int, signal: float) -> FullBudgetLabel:
    return FullBudgetLabel(
        label_id=f"label-synthetic-{index:03d}",
        candidate_id=f"candidate-synthetic-{index:03d}",
        full_trial_ids=(f"trial-full-{index:03d}",),
        mean_full_gain_bpb=0.001 + 0.8 * signal,
        sample_variance_bpb=0.000001,
        constraints_passed=True,
        captured_event_sequence=1_000 + index,
    )


def _metrics(path: Path, losses: tuple[float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"event": "train", "loss": losses[0], "tokens_seen": 1_000_000})
        + "\n"
        + json.dumps({"event": "train", "loss": losses[1], "tokens_seen": 2_000_000})
        + "\n",
        encoding="utf-8",
    )


def _manifest(
    artifact_root: Path,
    run: RunResult,
    suffix: str,
    losses: tuple[float, float],
) -> ArtifactManifest:
    metrics = artifact_root / "runs" / suffix / "metrics.jsonl"
    checkpoint = artifact_root / "runs" / suffix / "checkpoint.pt"
    _metrics(metrics, losses)
    checkpoint.write_bytes(f"checkpoint-{suffix}".encode())
    return ArtifactManifest(
        manifest_id=f"manifest-{suffix}",
        run_id=run.run_id,
        artifacts=(
            ArtifactRef(
                artifact_id=f"checkpoint-{suffix}",
                kind="checkpoint",
                relative_path=checkpoint.relative_to(artifact_root).as_posix(),
                sha256=file_sha256(checkpoint),
                size_bytes=checkpoint.stat().st_size,
                retention=ArtifactRetention.EPHEMERAL,
            ),
            ArtifactRef(
                artifact_id=f"metrics-{suffix}",
                kind="metrics",
                relative_path=metrics.relative_to(artifact_root).as_posix(),
                sha256=file_sha256(metrics),
                size_bytes=metrics.stat().st_size,
                retention=ArtifactRetention.COMPACT,
            ),
        ),
    )


def _ledger_with_candidate(tmp_path: Path) -> tuple[ExperimentLedger, Path, str]:
    records = evidence_records()
    ledger = ExperimentLedger.create(
        tmp_path / "ledger.sqlite3",
        initial_parent_commit=PARENT_COMMIT,
    )
    ledger.append_many(
        (
            (records["proposal"], WriterRole.RESEARCH_AGENT),
            (records["candidate"], WriterRole.CONTROLLER),
        )
    )
    return ledger, tmp_path / "artifacts", records["candidate"].candidate_id


def _append_pair(
    ledger: ExperimentLedger,
    artifact_root: Path,
    *,
    stage: ExperimentStage,
    seed: int,
    gain: float,
) -> None:
    records = evidence_records()
    candidate = records["candidate"]
    suffix = f"{stage.value}-{seed}"
    budget = {
        ExperimentStage.CHEAP: 2_000_000,
        ExperimentStage.INTERMEDIATE: 6_000_000,
        ExperimentStage.FULL: 20_000_000,
    }[stage]
    trial = replace(
        records["trial"],
        trial_id=f"trial-{suffix}",
        stage=stage,
        seed=seed,
        token_budget=budget,
    )
    order_hash = digest(str(seed % 10))
    parent = replace(
        records["parent_run"],
        run_id=f"run-parent-{suffix}",
        trial_id=trial.trial_id,
        seed=seed,
        target_tokens=budget,
        tokens_seen=budget,
        data_order_sha256=order_hash,
    )
    candidate_run = replace(
        records["candidate_run"],
        run_id=f"run-candidate-{suffix}",
        trial_id=trial.trial_id,
        seed=seed,
        target_tokens=budget,
        tokens_seen=budget,
        validation_bpb=parent.validation_bpb - gain,
        data_order_sha256=order_hash,
    )
    parent_manifest = _manifest(
        artifact_root,
        parent,
        f"parent-{suffix}",
        (1.4, 1.2),
    )
    candidate_manifest = _manifest(
        artifact_root,
        candidate_run,
        f"candidate-{suffix}",
        (1.3, 1.1),
    )
    ledger.append_many(
        (
            (trial, WriterRole.CONTROLLER),
            (parent, WriterRole.EVALUATOR),
            (parent_manifest, WriterRole.EVALUATOR),
            (candidate_run, WriterRole.EVALUATOR),
            (candidate_manifest, WriterRole.EVALUATOR),
        )
    )
    pair = build_paired_result(
        f"pair-{suffix}",
        trial=trial,
        candidate_id=candidate.candidate_id,
        parent=parent,
        candidate=candidate_run,
    )
    ledger.append(pair, writer_role=WriterRole.EVALUATOR)


def test_student_t_math_matches_cauchy_special_case() -> None:
    assert student_t_cdf(0.0, 1.0) == 0.5
    assert student_t_cdf(1.0, 1.0) == pytest.approx(0.75, abs=1e-10)
    assert student_t_cdf(-1.0, 1.0) == pytest.approx(0.25, abs=1e-10)
    assert student_t_quantile(0.975, 1.0) == pytest.approx(math.tan(math.pi * 0.475), rel=1e-8)


def test_bayesian_calibration_learns_signal_uncertainty_and_diagnostics(
    tmp_path: Path,
) -> None:
    signals = [(-0.01 + index * 0.0005) for index in range(45)]
    features = [_feature(index, signal) for index, signal in enumerate(signals)]
    labels = [_label(index, signal) for index, signal in enumerate(signals)]

    model = calibrate_model(features, labels, minimum_label_count=40)
    predicted = model.predict(
        _feature(999, 0.005),
        minimum_useful_gain_bpb=0.001,
    )

    assert model.calibrated is True
    assert model.label_count == 45
    assert model.calibration_rmse_bpb is not None
    assert model.calibration_rmse_bpb < 0.002
    assert model.calibration_mean_absolute_error_bpb is not None
    assert model.calibration_interval_coverage_90 is not None
    assert 0.0 <= model.calibration_interval_coverage_90 <= 1.0
    assert predicted.mean == pytest.approx(0.005, abs=0.002)
    assert predicted.standard_deviation > 0.0
    assert predicted.interval_lower < predicted.mean < predicted.interval_upper
    assert predicted.probability_exceeds_minimum > 0.5

    model_path = tmp_path / "model.json"
    save_model(model_path, model)
    restored = load_model(model_path)
    assert restored == model
    assert restored.sha256() == model.sha256()


def test_underfilled_model_requires_full_labels_for_calibration() -> None:
    features = [_feature(index, index * 0.001) for index in range(5)]
    labels = [_label(index, index * 0.001) for index in range(5)]
    model = calibrate_model(features, labels)
    distribution = model.predict(features[-1], minimum_useful_gain_bpb=0.001)

    assert model.calibrated is False
    assert recommendation(model, distribution) == "run_full_for_calibration"


def test_calibration_rejects_feature_snapshots_captured_after_full_label() -> None:
    feature = replace(_feature(1, 0.002), captured_event_sequence=2_000)
    label = _label(1, 0.002)

    with pytest.raises(RewardError, match="must precede"):
        calibrate_model([feature], [label])


def test_calibrated_recommendations_allocate_longer_tests_by_probability() -> None:
    features = [_feature(index, index * 0.001) for index in range(4)]
    labels = [_label(index, index * 0.001) for index in range(4)]
    model = calibrate_model(features, labels, minimum_label_count=4)

    def distribution(probability: float) -> PredictiveDistribution:
        return PredictiveDistribution(0.0, 0.01, -0.02, 0.02, 8.0, probability)

    assert recommendation(model, distribution(0.05)) == "stop"
    assert recommendation(model, distribution(0.50)) == "gather_more_early_evidence"
    assert recommendation(model, distribution(0.90)) == "run_full"
    assert (
        recommendation(
            model,
            distribution(0.50),
            reject_probability=0.60,
            full_test_probability=0.95,
        )
        == "stop"
    )
    with pytest.raises(RewardError, match="thresholds are not ordered"):
        recommendation(
            model,
            distribution(0.50),
            reject_probability=0.90,
            full_test_probability=0.80,
        )


def test_feature_extraction_verifies_artifacts_and_captures_learning_curves(
    tmp_path: Path,
) -> None:
    ledger, artifact_root, candidate_id = _ledger_with_candidate(tmp_path)
    _append_pair(
        ledger,
        artifact_root,
        stage=ExperimentStage.CHEAP,
        seed=11,
        gain=0.005,
    )

    features = extract_learning_curve_features(ledger, artifact_root, candidate_id)
    values = features.as_mapping()

    assert values["cheap_pair_count"] == 1.0
    assert values["intermediate_pair_count"] == 0.0
    assert values["latest_budget_fraction"] == 0.1
    assert values["mean_gain_bpb"] == pytest.approx(0.005)
    assert values["mean_train_loss_delta"] == pytest.approx(0.1)
    assert values["mean_loss_slope_delta"] == pytest.approx(0.0)
    assert values["mean_loss_area_delta"] == pytest.approx(0.1)
    assert values["candidate_failure_rate"] == 0.0
    assert features.source_stages == (ExperimentStage.CHEAP,)
    assert len(features.artifact_sha256s) == 2

    metrics = artifact_root / "runs/candidate-cheap-11/metrics.jsonl"
    metrics.write_text(metrics.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RewardError, match="hash or size"):
        extract_learning_curve_features(ledger, artifact_root, candidate_id)


def test_full_budget_label_aggregates_completed_full_pairs(tmp_path: Path) -> None:
    ledger, artifact_root, candidate_id = _ledger_with_candidate(tmp_path)
    _append_pair(
        ledger,
        artifact_root,
        stage=ExperimentStage.FULL,
        seed=11,
        gain=0.004,
    )
    _append_pair(
        ledger,
        artifact_root,
        stage=ExperimentStage.FULL,
        seed=23,
        gain=0.002,
    )

    label = build_full_budget_label(ledger, candidate_id)

    assert label.mean_full_gain_bpb == pytest.approx(0.003)
    assert label.sample_variance_bpb == pytest.approx(0.000002)
    assert len(label.full_trial_ids) == 2
    assert label.constraints_passed is True


def test_prediction_is_appended_to_ledger_with_model_evidence(tmp_path: Path) -> None:
    ledger, artifact_root, candidate_id = _ledger_with_candidate(tmp_path)
    _append_pair(
        ledger,
        artifact_root,
        stage=ExperimentStage.CHEAP,
        seed=11,
        gain=0.005,
    )
    extracted = extract_learning_curve_features(ledger, artifact_root, candidate_id)
    signals = [(-0.01 + index * 0.0005) for index in range(40)]
    model = calibrate_model(
        [_feature(index, signal) for index, signal in enumerate(signals)],
        [_label(index, signal) for index, signal in enumerate(signals)],
    )

    prediction, distribution, allocation = build_downstream_prediction(
        ledger,
        candidate_id,
        extracted,
        model,
    )
    ledger.append(prediction, writer_role=WriterRole.CONTROLLER)

    assert isinstance(ledger.get(prediction.prediction_id).record, DownstreamPrediction)
    assert prediction.target_stage is ExperimentStage.FULL
    assert prediction.full_budget_label_count == 40
    assert prediction.model_version.endswith(model.sha256()[:16])
    assert distribution.standard_deviation > 0.0
    assert allocation in {"stop", "gather_more_early_evidence", "run_full"}


def test_calibration_cli_writes_a_versioned_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    features_path = tmp_path / "features.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    model_path = tmp_path / "model.json"
    features = [_feature(index, index * 0.001) for index in range(5)]
    labels = [_label(index, index * 0.001) for index in range(5)]
    features_path.write_text(
        "".join(json.dumps(feature.to_dict()) + "\n" for feature in features),
        encoding="utf-8",
    )
    labels_path.write_text(
        "".join(json.dumps(label.to_dict()) + "\n" for label in labels),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "calibrate",
                "--features",
                str(features_path),
                "--labels",
                str(labels_path),
                "--output",
                str(model_path),
                "--minimum-labels",
                "5",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["calibrated"] is True
    assert payload["label_count"] == 5
    assert payload["calibration_rmse_bpb"] is not None
    assert isinstance(load_model(model_path), BayesianRewardModel)


def test_extract_and_predict_cli_append_downstream_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger, artifact_root, candidate_id = _ledger_with_candidate(tmp_path)
    _append_pair(
        ledger,
        artifact_root,
        stage=ExperimentStage.CHEAP,
        seed=11,
        gain=0.005,
    )
    features_path = tmp_path / "features.jsonl"
    model_path = tmp_path / "model.json"
    signals = [(-0.01 + index * 0.0005) for index in range(40)]
    model = calibrate_model(
        [_feature(index, signal) for index, signal in enumerate(signals)],
        [_label(index, signal) for index, signal in enumerate(signals)],
    )
    save_model(model_path, model)
    ledger_args = ["--ledger-path", str(ledger.path)]

    assert (
        main(
            [
                *ledger_args,
                "extract",
                "--candidate-id",
                candidate_id,
                "--artifact-root",
                str(artifact_root),
                "--output",
                str(features_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert len(features_path.read_text(encoding="utf-8").splitlines()) == 1

    assert (
        main(
            [
                *ledger_args,
                "predict",
                "--candidate-id",
                candidate_id,
                "--features",
                str(features_path),
                "--model",
                str(model_path),
            ]
        )
        == 0
    )
    predicted = json.loads(capsys.readouterr().out)
    assert predicted["full_budget_label_count"] == 40
    assert predicted["recommendation"] in {
        "stop",
        "gather_more_early_evidence",
        "run_full",
    }
    assert isinstance(ledger.get(predicted["prediction_id"]).record, DownstreamPrediction)
