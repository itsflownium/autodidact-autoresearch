"""Bayesian learning-curve prediction for full-budget patch reward."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from autodidact.checkpoints import file_sha256
from autodidact.data.integrity import canonical_json_bytes
from autodidact.ledger import ExperimentLedger, LedgerError, WriterRole
from autodidact.records import (
    ArtifactManifest,
    CandidateRecord,
    DownstreamPrediction,
    ExperimentStage,
    PairedResult,
    PatchProposal,
    RunArm,
    RunResult,
    RunStatus,
    TrialSpec,
)

REWARD_SCHEMA_VERSION = 1
EXTRACTOR_VERSION = "learning-curve-v1"
MODEL_VERSION = "bayesian-linear-nig-v1"
DEFAULT_LEDGER_PATH = Path("artifacts/ledger/experiments.sqlite3")
DEFAULT_ARTIFACT_ROOT = Path("artifacts/experiments")
DEFAULT_FEATURES_PATH = Path("artifacts/reward/features.jsonl")
DEFAULT_LABELS_PATH = Path("artifacts/reward/full-labels.jsonl")
DEFAULT_MODEL_PATH = Path("artifacts/reward/model.json")
DEFAULT_MINIMUM_LABELS = 40
FULL_TOKEN_BUDGET = 20_000_000

FEATURE_NAMES = (
    "cheap_pair_count",
    "intermediate_pair_count",
    "latest_budget_fraction",
    "mean_gain_bpb",
    "latest_gain_bpb",
    "gain_slope_per_million_tokens",
    "gain_sample_standard_deviation",
    "mean_train_loss_delta",
    "mean_loss_slope_delta",
    "mean_loss_area_delta",
    "mean_training_throughput_ratio",
    "mean_peak_process_rss_ratio",
    "candidate_failure_rate",
    "constraint_pass_rate",
    "mean_parameter_ratio",
)

_STAGE_ORDER = {
    ExperimentStage.CHEAP: 0,
    ExperimentStage.INTERMEDIATE: 1,
    ExperimentStage.FULL: 2,
    ExperimentStage.PROMOTION: 3,
    ExperimentStage.SEALED_FINAL: 4,
}


class RewardError(RuntimeError):
    """Raised when reward evidence or a calibrated model cannot be trusted."""


@dataclass(frozen=True, slots=True)
class LearningCurveFeatures:
    feature_id: str
    candidate_id: str
    source_trial_ids: tuple[str, ...]
    source_stages: tuple[ExperimentStage, ...]
    artifact_sha256s: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    captured_event_sequence: int
    extractor_version: str = EXTRACTOR_VERSION
    schema_version: int = REWARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.feature_id or not self.candidate_id or not self.source_trial_ids:
            raise RewardError("feature identity and source trials cannot be empty")
        if len(set(self.source_trial_ids)) != len(self.source_trial_ids):
            raise RewardError("feature source trials must be unique")
        if not self.source_stages or len(set(self.source_stages)) != len(self.source_stages):
            raise RewardError("feature source stages must be nonempty and unique")
        if any(
            stage not in {ExperimentStage.CHEAP, ExperimentStage.INTERMEDIATE}
            for stage in self.source_stages
        ):
            raise RewardError("learning-curve features cannot contain full-stage evidence")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.artifact_sha256s
        ):
            raise RewardError("feature artifact hashes must be lowercase SHA-256 values")
        if len(self.feature_names) != len(self.feature_values):
            raise RewardError("feature names and values differ in length")
        if self.feature_names != FEATURE_NAMES:
            raise RewardError("feature vector does not match the extractor schema")
        if any(not math.isfinite(value) for value in self.feature_values):
            raise RewardError("feature values must be finite")
        if self.captured_event_sequence <= 0:
            raise RewardError("captured_event_sequence must be positive")
        if self.schema_version != REWARD_SCHEMA_VERSION:
            raise RewardError("unsupported feature schema version")

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.feature_values, strict=True))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_stages"] = [stage.value for stage in self.source_stages]
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LearningCurveFeatures:
        return cls(
            feature_id=str(value["feature_id"]),
            candidate_id=str(value["candidate_id"]),
            source_trial_ids=tuple(str(item) for item in value["source_trial_ids"]),
            source_stages=tuple(ExperimentStage(item) for item in value["source_stages"]),
            artifact_sha256s=tuple(str(item) for item in value["artifact_sha256s"]),
            feature_names=tuple(str(item) for item in value["feature_names"]),
            feature_values=tuple(float(item) for item in value["feature_values"]),
            captured_event_sequence=int(value["captured_event_sequence"]),
            extractor_version=str(value["extractor_version"]),
            schema_version=int(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class FullBudgetLabel:
    label_id: str
    candidate_id: str
    full_trial_ids: tuple[str, ...]
    mean_full_gain_bpb: float
    sample_variance_bpb: float
    constraints_passed: bool
    captured_event_sequence: int
    schema_version: int = REWARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.label_id or not self.candidate_id or not self.full_trial_ids:
            raise RewardError("label identity and full trials cannot be empty")
        if len(set(self.full_trial_ids)) != len(self.full_trial_ids):
            raise RewardError("full label trial IDs must be unique")
        if not math.isfinite(self.mean_full_gain_bpb):
            raise RewardError("full label gain must be finite")
        if not math.isfinite(self.sample_variance_bpb) or self.sample_variance_bpb < 0.0:
            raise RewardError("full label variance must be finite and nonnegative")
        if type(self.constraints_passed) is not bool:
            raise RewardError("constraints_passed must be boolean")
        if self.captured_event_sequence <= 0:
            raise RewardError("captured_event_sequence must be positive")
        if self.schema_version != REWARD_SCHEMA_VERSION:
            raise RewardError("unsupported label schema version")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FullBudgetLabel:
        if type(value.get("constraints_passed")) is not bool:
            raise RewardError("label constraints_passed must be boolean")
        return cls(
            label_id=str(value["label_id"]),
            candidate_id=str(value["candidate_id"]),
            full_trial_ids=tuple(str(item) for item in value["full_trial_ids"]),
            mean_full_gain_bpb=float(value["mean_full_gain_bpb"]),
            sample_variance_bpb=float(value["sample_variance_bpb"]),
            constraints_passed=value["constraints_passed"],
            captured_event_sequence=int(value["captured_event_sequence"]),
            schema_version=int(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class PredictiveDistribution:
    mean: float
    standard_deviation: float
    interval_lower: float
    interval_upper: float
    degrees_of_freedom: float
    probability_exceeds_minimum: float


@dataclass(frozen=True, slots=True)
class BayesianRewardModel:
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    posterior_mean: tuple[float, ...]
    posterior_precision: tuple[tuple[float, ...], ...]
    posterior_shape: float
    posterior_scale: float
    label_count: int
    minimum_label_count: int
    training_feature_ids: tuple[str, ...]
    training_label_ids: tuple[str, ...]
    calibration_rmse_bpb: float | None = None
    calibration_mean_absolute_error_bpb: float | None = None
    calibration_interval_coverage_90: float | None = None
    model_version: str = MODEL_VERSION
    schema_version: int = REWARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        dimension = len(self.feature_names)
        if self.feature_names != FEATURE_NAMES:
            raise RewardError("model feature schema differs from the extractor")
        if len(self.feature_means) != dimension or len(self.feature_scales) != dimension:
            raise RewardError("model standardization vectors have the wrong size")
        if len(self.posterior_mean) != dimension + 1:
            raise RewardError("model posterior mean has the wrong size")
        if len(self.posterior_precision) != dimension + 1 or any(
            len(row) != dimension + 1 for row in self.posterior_precision
        ):
            raise RewardError("model precision matrix has the wrong shape")
        numeric = (
            *self.feature_means,
            *self.feature_scales,
            *self.posterior_mean,
            *(value for row in self.posterior_precision for value in row),
            self.posterior_shape,
            self.posterior_scale,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise RewardError("model parameters must be finite")
        if any(scale <= 0.0 for scale in self.feature_scales):
            raise RewardError("feature scales must be positive")
        if self.posterior_shape <= 1.0 or self.posterior_scale <= 0.0:
            raise RewardError("posterior noise parameters are invalid")
        if self.label_count <= 0 or self.minimum_label_count <= 0:
            raise RewardError("model label counts must be positive")
        if self.label_count != len(self.training_feature_ids) or self.label_count != len(
            self.training_label_ids
        ):
            raise RewardError("model training evidence counts differ")
        for name in (
            "calibration_rmse_bpb",
            "calibration_mean_absolute_error_bpb",
            "calibration_interval_coverage_90",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise RewardError(f"{name} must be finite when provided")
        for name in ("calibration_rmse_bpb", "calibration_mean_absolute_error_bpb"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise RewardError(f"{name} cannot be negative")
        if self.calibration_interval_coverage_90 is not None and not (
            0.0 <= self.calibration_interval_coverage_90 <= 1.0
        ):
            raise RewardError("calibration interval coverage must lie between zero and one")
        if self.schema_version != REWARD_SCHEMA_VERSION:
            raise RewardError("unsupported model schema version")

    @property
    def calibrated(self) -> bool:
        return self.label_count >= self.minimum_label_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BayesianRewardModel:
        return cls(
            feature_names=tuple(str(item) for item in value["feature_names"]),
            feature_means=tuple(float(item) for item in value["feature_means"]),
            feature_scales=tuple(float(item) for item in value["feature_scales"]),
            posterior_mean=tuple(float(item) for item in value["posterior_mean"]),
            posterior_precision=tuple(
                tuple(float(item) for item in row) for row in value["posterior_precision"]
            ),
            posterior_shape=float(value["posterior_shape"]),
            posterior_scale=float(value["posterior_scale"]),
            label_count=int(value["label_count"]),
            minimum_label_count=int(value["minimum_label_count"]),
            training_feature_ids=tuple(str(item) for item in value["training_feature_ids"]),
            training_label_ids=tuple(str(item) for item in value["training_label_ids"]),
            calibration_rmse_bpb=(
                None
                if value.get("calibration_rmse_bpb") is None
                else float(value["calibration_rmse_bpb"])
            ),
            calibration_mean_absolute_error_bpb=(
                None
                if value.get("calibration_mean_absolute_error_bpb") is None
                else float(value["calibration_mean_absolute_error_bpb"])
            ),
            calibration_interval_coverage_90=(
                None
                if value.get("calibration_interval_coverage_90") is None
                else float(value["calibration_interval_coverage_90"])
            ),
            model_version=str(value["model_version"]),
            schema_version=int(value["schema_version"]),
        )

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def predict(
        self,
        features: LearningCurveFeatures,
        *,
        minimum_useful_gain_bpb: float,
        interval_mass: float = 0.90,
    ) -> PredictiveDistribution:
        if features.feature_names != self.feature_names:
            raise RewardError("prediction feature schema differs from the model")
        if interval_mass <= 0.0 or interval_mass >= 1.0:
            raise RewardError("interval_mass must lie between zero and one")
        values = np.asarray(features.feature_values, dtype=np.float64)
        means = np.asarray(self.feature_means, dtype=np.float64)
        scales = np.asarray(self.feature_scales, dtype=np.float64)
        design = np.concatenate(([1.0], (values - means) / scales))
        posterior_mean = np.asarray(self.posterior_mean, dtype=np.float64)
        precision = np.asarray(self.posterior_precision, dtype=np.float64)
        covariance = np.linalg.inv(precision)
        location = float(design @ posterior_mean)
        degrees_of_freedom = 2.0 * self.posterior_shape
        predictive_scale_squared = (
            self.posterior_scale
            / self.posterior_shape
            * (1.0 + float(design @ covariance @ design))
        )
        predictive_scale = math.sqrt(max(predictive_scale_squared, 1e-18))
        if degrees_of_freedom <= 2.0:
            raise RewardError("predictive variance is undefined")
        standard_deviation = predictive_scale * math.sqrt(
            degrees_of_freedom / (degrees_of_freedom - 2.0)
        )
        standardized_minimum = (minimum_useful_gain_bpb - location) / predictive_scale
        probability = 1.0 - student_t_cdf(standardized_minimum, degrees_of_freedom)
        tail = (1.0 - interval_mass) / 2.0
        lower_quantile = student_t_quantile(tail, degrees_of_freedom)
        upper_quantile = student_t_quantile(1.0 - tail, degrees_of_freedom)
        return PredictiveDistribution(
            mean=location,
            standard_deviation=standard_deviation,
            interval_lower=location + predictive_scale * lower_quantile,
            interval_upper=location + predictive_scale * upper_quantile,
            degrees_of_freedom=degrees_of_freedom,
            probability_exceeds_minimum=min(1.0, max(0.0, probability)),
        )


def _continued_beta_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 200
    epsilon = 3e-14
    minimum = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + even) * (a + even))
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / ((a + even) * (qap + even))
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise RewardError("incomplete beta calculation did not converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if a <= 0.0 or b <= 0.0 or x < 0.0 or x > 1.0:
        raise RewardError("invalid incomplete beta arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _continued_beta_fraction(a, b, x) / a
    return 1.0 - factor * _continued_beta_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: float) -> float:
    if not math.isfinite(value) or degrees_of_freedom <= 0.0:
        raise RewardError("invalid Student-t CDF arguments")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    beta = regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)
    return 1.0 - 0.5 * beta if value > 0.0 else 0.5 * beta


def student_t_quantile(probability: float, degrees_of_freedom: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        raise RewardError("Student-t quantile probability must be internal")
    lower = -64.0
    upper = 64.0
    for _iteration in range(120):
        midpoint = (lower + upper) / 2.0
        if student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _stable_id(prefix: str, *parts: object) -> str:
    payload = canonical_json_bytes([str(part) for part in parts])
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RewardError(f"required JSONL evidence is missing: {path.name}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RewardError(f"invalid JSONL evidence at line {line_number}") from error
        if not isinstance(value, dict):
            raise RewardError("JSONL evidence records must be objects")
        records.append(value)
    return records


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _append_unique_jsonl(path: Path, payload: dict[str, Any], *, key: str) -> None:
    existing = _read_jsonl(path) if path.is_file() else []
    matching = [item for item in existing if item.get(key) == payload[key]]
    if matching:
        if matching[-1] != payload:
            raise RewardError(f"existing {key} has different immutable content")
        return
    existing.append(payload)
    _atomic_write(
        path,
        "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in existing),
    )


def load_features(path: Path) -> list[LearningCurveFeatures]:
    return [LearningCurveFeatures.from_dict(item) for item in _read_jsonl(path)]


def load_labels(path: Path) -> list[FullBudgetLabel]:
    return [FullBudgetLabel.from_dict(item) for item in _read_jsonl(path)]


def load_model(path: Path) -> BayesianRewardModel:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RewardError("reward model artifact is missing or invalid") from error
    if not isinstance(value, dict):
        raise RewardError("reward model artifact must be an object")
    return BayesianRewardModel.from_dict(value)


def store_learning_curve_features(path: Path, features: LearningCurveFeatures) -> None:
    _append_unique_jsonl(path, features.to_dict(), key="feature_id")


def store_full_budget_label(path: Path, label: FullBudgetLabel) -> None:
    _append_unique_jsonl(path, label.to_dict(), key="label_id")


def save_model(path: Path, model: BayesianRewardModel) -> None:
    _atomic_write(
        path,
        json.dumps(model.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _artifact_path(root: Path, manifest: ArtifactManifest, *, kind: str) -> tuple[Path, str]:
    matching = [artifact for artifact in manifest.artifacts if artifact.kind == kind]
    if len(matching) != 1:
        raise RewardError(f"run manifest must contain exactly one {kind} artifact")
    artifact = matching[0]
    relative = PurePosixPath(artifact.relative_path)
    path = root.joinpath(*relative.parts).resolve()
    root = root.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise RewardError("metrics artifact is outside the declared artifact root or missing")
    if path.stat().st_size != artifact.size_bytes or file_sha256(path) != artifact.sha256:
        raise RewardError("metrics artifact hash or size differs from its manifest")
    return path, artifact.sha256


def _curve_summary(metrics_path: Path, fallback_loss: float) -> tuple[float, float, float]:
    events = [event for event in _read_jsonl(metrics_path) if event.get("event") == "train"]
    points = []
    for event in events:
        tokens = event.get("tokens_seen")
        loss = event.get("loss")
        if (
            type(tokens) is int
            and tokens > 0
            and isinstance(loss, (int, float))
            and not isinstance(loss, bool)
            and math.isfinite(float(loss))
        ):
            points.append((tokens, float(loss)))
    if not points:
        if not math.isfinite(fallback_loss):
            raise RewardError("training curve and fallback loss are unavailable")
        return fallback_loss, 0.0, fallback_loss
    points.sort()
    maximum_tokens = points[-1][0]
    normalized = [(tokens / maximum_tokens, loss) for tokens, loss in points]
    if len(normalized) == 1:
        return normalized[-1][1], 0.0, normalized[-1][1]
    slope = (normalized[-1][1] - normalized[0][1]) / (normalized[-1][0] - normalized[0][0])
    area = 0.0
    for (left_x, left_y), (right_x, right_y) in zip(
        normalized,
        normalized[1:],
        strict=False,
    ):
        area += (right_x - left_x) * (left_y + right_y) / 2.0
    covered = normalized[-1][0] - normalized[0][0]
    normalized_area = area / covered if covered > 0.0 else normalized[-1][1]
    return normalized[-1][1], slope, normalized_area


def _linear_slope(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(set(x_values)) < 2:
        return 0.0
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    return (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )


def extract_learning_curve_features(
    ledger: ExperimentLedger,
    artifact_root: Path,
    candidate_id: str,
) -> LearningCurveFeatures:
    events = ledger.events()
    candidate = next(
        (
            event.record
            for event in events
            if isinstance(event.record, CandidateRecord)
            and event.record.candidate_id == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise RewardError("candidate record does not exist")
    trials = {
        event.record.trial_id: event.record
        for event in events
        if isinstance(event.record, TrialSpec)
        and event.record.candidate_id == candidate_id
        and event.record.stage in {ExperimentStage.CHEAP, ExperimentStage.INTERMEDIATE}
    }
    pairs = [
        event.record
        for event in events
        if isinstance(event.record, PairedResult)
        and event.record.candidate_id == candidate_id
        and event.record.trial_id in trials
    ]
    if not pairs:
        raise RewardError("candidate has no completed cheap or intermediate paired evidence")
    sequence_by_pair = {
        event.record.paired_result_id: event.sequence
        for event in events
        if isinstance(event.record, PairedResult)
    }
    pairs.sort(
        key=lambda pair: (
            trials[pair.trial_id].token_budget,
            sequence_by_pair[pair.paired_result_id],
        )
    )
    runs = {
        event.record.run_id: event.record for event in events if isinstance(event.record, RunResult)
    }
    manifests = {
        event.record.run_id: event.record
        for event in events
        if isinstance(event.record, ArtifactManifest)
    }
    artifact_hashes = []
    loss_deltas = []
    slope_deltas = []
    area_deltas = []
    throughput_ratios = []
    rss_ratios = []
    parameter_ratios = []
    gains = []
    budgets_millions = []
    for pair in pairs:
        trial = trials[pair.trial_id]
        parent = runs.get(pair.parent_run_id)
        candidate_run = runs.get(pair.candidate_run_id)
        if parent is None or candidate_run is None:
            raise RewardError("paired result is missing linked run results")
        if parent.mean_train_loss is None or candidate_run.mean_train_loss is None:
            raise RewardError("paired run is missing training-loss diagnostics")
        parent_manifest = manifests.get(parent.run_id)
        candidate_manifest = manifests.get(candidate_run.run_id)
        if parent_manifest is None or candidate_manifest is None:
            raise RewardError("paired run is missing its artifact manifest")
        parent_metrics, parent_hash = _artifact_path(artifact_root, parent_manifest, kind="metrics")
        candidate_metrics, candidate_hash = _artifact_path(
            artifact_root, candidate_manifest, kind="metrics"
        )
        artifact_hashes.extend((parent_hash, candidate_hash))
        parent_final, parent_slope, parent_area = _curve_summary(
            parent_metrics, parent.mean_train_loss
        )
        candidate_final, candidate_slope, candidate_area = _curve_summary(
            candidate_metrics, candidate_run.mean_train_loss
        )
        loss_deltas.append(parent_final - candidate_final)
        slope_deltas.append(parent_slope - candidate_slope)
        area_deltas.append(parent_area - candidate_area)
        assert parent.training_tokens_per_second is not None
        assert candidate_run.training_tokens_per_second is not None
        assert parent.peak_process_rss_bytes is not None
        assert candidate_run.peak_process_rss_bytes is not None
        throughput_ratios.append(
            candidate_run.training_tokens_per_second / parent.training_tokens_per_second
        )
        rss_ratios.append(
            candidate_run.peak_process_rss_bytes / max(parent.peak_process_rss_bytes, 1)
        )
        parameter_ratios.append(candidate_run.parameter_count / parent.parameter_count)
        gains.append(pair.gain_bpb)
        budgets_millions.append(trial.token_budget / 1_000_000.0)
    candidate_runs = [
        event.record
        for event in events
        if isinstance(event.record, RunResult)
        and event.record.arm is RunArm.CANDIDATE
        and event.record.trial_id in trials
    ]
    failures = sum(run.status is not RunStatus.SUCCEEDED for run in candidate_runs)
    gain_variance = statistics.variance(gains) if len(gains) > 1 else 0.0
    latest_trial = trials[pairs[-1].trial_id]
    source_stages = tuple(
        sorted(
            {
                trial.stage
                for trial in trials.values()
                if trial.trial_id in {p.trial_id for p in pairs}
            },
            key=lambda stage: _STAGE_ORDER[stage],
        )
    )
    values = (
        float(sum(trials[pair.trial_id].stage is ExperimentStage.CHEAP for pair in pairs)),
        float(sum(trials[pair.trial_id].stage is ExperimentStage.INTERMEDIATE for pair in pairs)),
        min(1.0, latest_trial.token_budget / FULL_TOKEN_BUDGET),
        statistics.fmean(gains),
        gains[-1],
        _linear_slope(budgets_millions, gains),
        math.sqrt(gain_variance),
        statistics.fmean(loss_deltas),
        statistics.fmean(slope_deltas),
        statistics.fmean(area_deltas),
        statistics.fmean(throughput_ratios),
        statistics.fmean(rss_ratios),
        failures / max(len(candidate_runs), 1),
        sum(pair.constraints_passed for pair in pairs) / len(pairs),
        statistics.fmean(parameter_ratios),
    )
    source_trial_ids = tuple(pair.trial_id for pair in pairs)
    candidate_run_sequences = [
        event.sequence
        for event in events
        if isinstance(event.record, RunResult)
        and event.record.arm is RunArm.CANDIDATE
        and event.record.trial_id in trials
    ]
    captured_sequence = max(
        *(sequence_by_pair[pair.paired_result_id] for pair in pairs),
        *candidate_run_sequences,
    )
    feature_hash = hashlib.sha256(
        canonical_json_bytes(
            {
                "artifacts": artifact_hashes,
                "candidate_id": candidate_id,
                "extractor_version": EXTRACTOR_VERSION,
                "source_trial_ids": source_trial_ids,
                "values": values,
            }
        )
    ).hexdigest()
    return LearningCurveFeatures(
        feature_id=f"feature-{feature_hash[:24]}",
        candidate_id=candidate_id,
        source_trial_ids=source_trial_ids,
        source_stages=source_stages,
        artifact_sha256s=tuple(artifact_hashes),
        feature_names=FEATURE_NAMES,
        feature_values=values,
        captured_event_sequence=captured_sequence,
    )


def build_full_budget_label(
    ledger: ExperimentLedger,
    candidate_id: str,
) -> FullBudgetLabel:
    events = ledger.events()
    trials = {
        event.record.trial_id: event.record
        for event in events
        if isinstance(event.record, TrialSpec)
        and event.record.candidate_id == candidate_id
        and event.record.stage is ExperimentStage.FULL
    }
    pair_events = [
        event
        for event in events
        if isinstance(event.record, PairedResult)
        and event.record.candidate_id == candidate_id
        and event.record.trial_id in trials
    ]
    if not pair_events:
        raise RewardError("candidate has no completed full-budget paired labels")
    pairs = [event.record for event in pair_events]
    gains = [pair.gain_bpb for pair in pairs]
    variance = statistics.variance(gains) if len(gains) > 1 else 0.0
    label_hash = hashlib.sha256(
        canonical_json_bytes(
            {
                "candidate_id": candidate_id,
                "pair_ids": [pair.paired_result_id for pair in pairs],
                "gains": gains,
            }
        )
    ).hexdigest()
    return FullBudgetLabel(
        label_id=f"label-{label_hash[:24]}",
        candidate_id=candidate_id,
        full_trial_ids=tuple(pair.trial_id for pair in pairs),
        mean_full_gain_bpb=statistics.fmean(gains),
        sample_variance_bpb=variance,
        constraints_passed=all(pair.constraints_passed for pair in pairs),
        captured_event_sequence=max(event.sequence for event in pair_events),
    )


def calibrate_model(
    features: list[LearningCurveFeatures],
    labels: list[FullBudgetLabel],
    *,
    minimum_label_count: int = DEFAULT_MINIMUM_LABELS,
    prior_precision: float = 1.0,
    prior_shape: float = 2.0,
    prior_noise_standard_deviation_bpb: float = 0.005,
    _compute_diagnostics: bool = True,
) -> BayesianRewardModel:
    if minimum_label_count <= 0:
        raise RewardError("minimum_label_count must be positive")
    if prior_precision <= 0.0 or prior_shape <= 1.0:
        raise RewardError("Bayesian prior parameters are invalid")
    if prior_noise_standard_deviation_bpb <= 0.0:
        raise RewardError("prior noise standard deviation must be positive")
    latest_features: dict[str, LearningCurveFeatures] = {}
    for feature in features:
        current = latest_features.get(feature.candidate_id)
        if current is None or feature.captured_event_sequence > current.captured_event_sequence:
            latest_features[feature.candidate_id] = feature
    latest_labels: dict[str, FullBudgetLabel] = {}
    for label in labels:
        current = latest_labels.get(label.candidate_id)
        if current is None or label.captured_event_sequence > current.captured_event_sequence:
            latest_labels[label.candidate_id] = label
    candidate_ids = sorted(set(latest_features).intersection(latest_labels))
    if not candidate_ids:
        raise RewardError("calibration has no candidates with both early features and full labels")
    selected_features = [latest_features[candidate_id] for candidate_id in candidate_ids]
    selected_labels = [latest_labels[candidate_id] for candidate_id in candidate_ids]
    for feature, label in zip(selected_features, selected_labels, strict=True):
        if feature.captured_event_sequence >= label.captured_event_sequence:
            raise RewardError("calibration feature snapshot must precede its full-budget label")
    raw = np.asarray([feature.feature_values for feature in selected_features], dtype=np.float64)
    targets = np.asarray([label.mean_full_gain_bpb for label in selected_labels], dtype=np.float64)
    means = raw.mean(axis=0)
    scales = raw.std(axis=0, ddof=1) if len(raw) > 1 else np.ones(raw.shape[1])
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (raw - means) / scales
    design = np.column_stack((np.ones(len(raw)), standardized))
    dimension = design.shape[1]
    prior_matrix = np.eye(dimension, dtype=np.float64) * prior_precision
    prior_matrix[0, 0] = prior_precision * 0.01
    posterior_precision = prior_matrix + design.T @ design
    posterior_mean = np.linalg.solve(posterior_precision, design.T @ targets)
    posterior_shape = prior_shape + len(raw) / 2.0
    prior_scale = (prior_shape - 1.0) * prior_noise_standard_deviation_bpb**2
    posterior_scale = prior_scale + 0.5 * float(
        targets @ targets - posterior_mean @ posterior_precision @ posterior_mean
    )
    posterior_scale = max(posterior_scale, 1e-18)
    model = BayesianRewardModel(
        feature_names=FEATURE_NAMES,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        posterior_mean=tuple(float(value) for value in posterior_mean),
        posterior_precision=tuple(
            tuple(float(value) for value in row) for row in posterior_precision
        ),
        posterior_shape=posterior_shape,
        posterior_scale=posterior_scale,
        label_count=len(candidate_ids),
        minimum_label_count=minimum_label_count,
        training_feature_ids=tuple(feature.feature_id for feature in selected_features),
        training_label_ids=tuple(label.label_id for label in selected_labels),
    )
    if not _compute_diagnostics or len(candidate_ids) < 3:
        return model
    predictions = []
    actual = []
    covered = []
    for index, (feature, label) in enumerate(zip(selected_features, selected_labels, strict=True)):
        fold_features = [
            item for item_index, item in enumerate(selected_features) if item_index != index
        ]
        fold_labels = [
            item for item_index, item in enumerate(selected_labels) if item_index != index
        ]
        fold_model = calibrate_model(
            fold_features,
            fold_labels,
            minimum_label_count=minimum_label_count,
            prior_precision=prior_precision,
            prior_shape=prior_shape,
            prior_noise_standard_deviation_bpb=prior_noise_standard_deviation_bpb,
            _compute_diagnostics=False,
        )
        distribution = fold_model.predict(feature, minimum_useful_gain_bpb=0.0)
        predictions.append(distribution.mean)
        actual.append(label.mean_full_gain_bpb)
        covered.append(
            distribution.interval_lower <= label.mean_full_gain_bpb <= distribution.interval_upper
        )
    errors = [prediction - target for prediction, target in zip(predictions, actual, strict=True)]
    return replace(
        model,
        calibration_rmse_bpb=math.sqrt(statistics.fmean(error * error for error in errors)),
        calibration_mean_absolute_error_bpb=statistics.fmean(abs(error) for error in errors),
        calibration_interval_coverage_90=(sum(covered) / len(covered)),
    )


def recommendation(
    model: BayesianRewardModel,
    prediction: PredictiveDistribution,
    *,
    reject_probability: float = 0.10,
    full_test_probability: float = 0.80,
) -> str:
    if not model.calibrated:
        return "run_full_for_calibration"
    if prediction.probability_exceeds_minimum <= reject_probability:
        return "stop"
    if prediction.probability_exceeds_minimum >= full_test_probability:
        return "run_full"
    return "gather_more_early_evidence"


def build_downstream_prediction(
    ledger: ExperimentLedger,
    candidate_id: str,
    features: LearningCurveFeatures,
    model: BayesianRewardModel,
) -> tuple[DownstreamPrediction, PredictiveDistribution, str]:
    candidate_event = ledger.get(candidate_id)
    if not isinstance(candidate_event.record, CandidateRecord):
        raise RewardError("candidate_id does not identify a candidate record")
    if features.candidate_id != candidate_id:
        raise RewardError("prediction features belong to another candidate")
    proposal_event = ledger.get(candidate_event.record.proposal_id)
    if not isinstance(proposal_event.record, PatchProposal):
        raise RewardError("candidate proposal record is missing")
    proposal = proposal_event.record
    distribution = model.predict(
        features,
        minimum_useful_gain_bpb=proposal.minimum_useful_gain_bpb,
    )
    model_hash = model.sha256()
    prediction = DownstreamPrediction(
        prediction_id=_stable_id(
            "prediction",
            candidate_id,
            features.feature_id,
            model_hash,
        ),
        candidate_id=candidate_id,
        source_trial_ids=features.source_trial_ids,
        source_stages=features.source_stages,
        target_stage=ExperimentStage.FULL,
        expected_gain_bpb=distribution.mean,
        predictive_standard_deviation=distribution.standard_deviation,
        interval_lower_bpb=distribution.interval_lower,
        interval_upper_bpb=distribution.interval_upper,
        minimum_useful_gain_bpb=proposal.minimum_useful_gain_bpb,
        probability_exceeds_minimum=distribution.probability_exceeds_minimum,
        model_version=f"{MODEL_VERSION}:{model_hash[:16]}",
        full_budget_label_count=model.label_count,
    )
    return prediction, distribution, recommendation(model, distribution)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate full-budget patch reward.")
    parser.add_argument("--ledger-path", type=_path, default=DEFAULT_LEDGER_PATH)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="capture early learning-curve features")
    extract.add_argument("--candidate-id", required=True)
    extract.add_argument("--artifact-root", type=_path, default=DEFAULT_ARTIFACT_ROOT)
    extract.add_argument("--output", type=_path, default=DEFAULT_FEATURES_PATH)

    label = commands.add_parser("label", help="capture a completed full-budget target")
    label.add_argument("--candidate-id", required=True)
    label.add_argument("--output", type=_path, default=DEFAULT_LABELS_PATH)

    calibrate = commands.add_parser("calibrate", help="fit Bayesian full-budget prediction")
    calibrate.add_argument("--features", type=_path, default=DEFAULT_FEATURES_PATH)
    calibrate.add_argument("--labels", type=_path, default=DEFAULT_LABELS_PATH)
    calibrate.add_argument("--output", type=_path, default=DEFAULT_MODEL_PATH)
    calibrate.add_argument("--minimum-labels", type=int, default=DEFAULT_MINIMUM_LABELS)

    predict = commands.add_parser("predict", help="predict and allocate the next experiment")
    predict.add_argument("--candidate-id", required=True)
    predict.add_argument("--features", type=_path, default=DEFAULT_FEATURES_PATH)
    predict.add_argument("--model", type=_path, default=DEFAULT_MODEL_PATH)
    predict.add_argument(
        "--append-ledger",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _latest_candidate_features(
    features: list[LearningCurveFeatures],
    candidate_id: str,
) -> LearningCurveFeatures:
    matching = [feature for feature in features if feature.candidate_id == candidate_id]
    if not matching:
        raise RewardError("feature store has no row for this candidate")
    return max(matching, key=lambda feature: feature.captured_event_sequence)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            model = calibrate_model(
                load_features(args.features),
                load_labels(args.labels),
                minimum_label_count=args.minimum_labels,
            )
            save_model(args.output, model)
            payload: dict[str, Any] = {
                "calibration_interval_coverage_90": (model.calibration_interval_coverage_90),
                "calibration_mean_absolute_error_bpb": (model.calibration_mean_absolute_error_bpb),
                "calibration_rmse_bpb": model.calibration_rmse_bpb,
                "calibrated": model.calibrated,
                "label_count": model.label_count,
                "minimum_label_count": model.minimum_label_count,
                "model_path": str(args.output),
                "model_sha256": model.sha256(),
            }
        else:
            ledger = ExperimentLedger.open(
                args.ledger_path,
                read_only=args.command != "predict" or not args.append_ledger,
            )
            if args.command == "extract":
                features = extract_learning_curve_features(
                    ledger,
                    args.artifact_root,
                    args.candidate_id,
                )
                store_learning_curve_features(args.output, features)
                payload = features.to_dict()
            elif args.command == "label":
                label = build_full_budget_label(ledger, args.candidate_id)
                store_full_budget_label(args.output, label)
                payload = label.to_dict()
            else:
                features = _latest_candidate_features(
                    load_features(args.features),
                    args.candidate_id,
                )
                model = load_model(args.model)
                prediction, distribution, allocation = build_downstream_prediction(
                    ledger,
                    args.candidate_id,
                    features,
                    model,
                )
                if args.append_ledger:
                    ledger.ensure(prediction, writer_role=WriterRole.CONTROLLER)
                payload = {
                    "calibrated": model.calibrated,
                    "candidate_id": args.candidate_id,
                    "expected_full_gain_bpb": distribution.mean,
                    "full_budget_label_count": model.label_count,
                    "interval_lower_bpb": distribution.interval_lower,
                    "interval_upper_bpb": distribution.interval_upper,
                    "prediction_id": prediction.prediction_id,
                    "probability_exceeds_minimum": (distribution.probability_exceeds_minimum),
                    "recommendation": allocation,
                }
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (LedgerError, RewardError, ValueError, np.linalg.LinAlgError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
