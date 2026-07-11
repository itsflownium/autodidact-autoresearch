"""Versioned, portable records for protected autoresearch evidence."""

from __future__ import annotations

import math
import re
import statistics
import uuid
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, ClassVar, TypeAlias

from autodidact.data.integrity import ProtectedPathError, assert_research_paths_allowed

RECORD_SCHEMA_VERSION = 1
MAX_PYTHON_SEED = 2**32 - 1
DEFAULT_PARAMETER_CAP = 1_050_000

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,95}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class RecordValidationError(ValueError):
    """Raised when an evidence record violates its schema."""


class ExperimentStage(StrEnum):
    CHEAP = "cheap"
    INTERMEDIATE = "intermediate"
    FULL = "full"
    PROMOTION = "promotion"
    SEALED_FINAL = "sealed_final"


class RunArm(StrEnum):
    PARENT = "parent"
    CANDIDATE = "candidate"


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    CRASHED = "crashed"
    TIMEOUT = "timeout"
    OOM = "oom"
    NON_FINITE = "non_finite"
    INTEGRITY_FAILURE = "integrity_failure"
    CANCELLED = "cancelled"


class DecisionVerdict(StrEnum):
    REJECT = "reject"
    ESCALATE = "escalate"
    PROMOTE = "promote"


class ArtifactRetention(StrEnum):
    EPHEMERAL = "ephemeral"
    RETAINED = "retained"
    COMPACT = "compact"


def _validate_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise RecordValidationError(
            f"{name} must start with a letter and contain only lowercase letters, "
            "digits, underscores, or hyphens"
        )


def _validate_text(name: str, value: str, *, maximum: int = 4_000) -> None:
    if not isinstance(value, str):
        raise RecordValidationError(f"{name} must be text")
    if not value.strip():
        raise RecordValidationError(f"{name} cannot be empty")
    if len(value) > maximum:
        raise RecordValidationError(f"{name} exceeds {maximum} characters")
    if "\x00" in value:
        raise RecordValidationError(f"{name} contains a null byte")


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise RecordValidationError(f"{name} must be a lowercase SHA-256 digest")


def _validate_git_commit(name: str, value: str) -> None:
    if not isinstance(value, str) or not _GIT_COMMIT_PATTERN.fullmatch(value):
        raise RecordValidationError(f"{name} must be a full SHA-1 or SHA-256 Git commit")


def validate_git_commit(value: str) -> None:
    """Validate a full SHA-1 or SHA-256 Git object ID."""

    _validate_git_commit("git commit", value)


def _validate_seed(seed: int) -> None:
    _validate_integer("seed", seed, minimum=0, maximum=MAX_PYTHON_SEED)


def _validate_integer(
    name: str,
    value: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise RecordValidationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise RecordValidationError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise RecordValidationError(f"{name} must be at most {maximum}")


def _validate_enum(name: str, value: Any, enum_type: type[StrEnum]) -> None:
    if not isinstance(value, enum_type):
        raise RecordValidationError(f"{name} must be a {enum_type.__name__} value")


def _validate_boolean(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise RecordValidationError(f"{name} must be a boolean")


def _validate_finite(name: str, value: float, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordValidationError(f"{name} must be a number")
    if not math.isfinite(value):
        raise RecordValidationError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise RecordValidationError(f"{name} must be at least {minimum}")


def _validate_probability(name: str, value: float) -> None:
    _validate_finite(name, value)
    if value < 0.0 or value > 1.0:
        raise RecordValidationError(f"{name} must be between zero and one")


def _validate_unique(name: str, values: tuple[Any, ...]) -> None:
    if not isinstance(values, tuple):
        raise RecordValidationError(f"{name} must be an immutable tuple")
    try:
        unique_count = len(set(values))
    except TypeError as error:
        raise RecordValidationError(f"{name} must contain immutable values") from error
    if unique_count != len(values):
        raise RecordValidationError(f"{name} must not contain duplicates")


def new_record_id(prefix: str) -> str:
    _validate_id("prefix", prefix)
    return f"{prefix}-{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    timeout_seconds: int
    max_parameter_count: int = DEFAULT_PARAMETER_CAP
    max_peak_process_rss_bytes: int | None = None
    max_peak_device_bytes: int | None = None
    min_training_tokens_per_second: float | None = None
    max_training_throughput_regression_fraction: float | None = None
    max_peak_process_rss_regression_fraction: float | None = None
    max_peak_device_regression_fraction: float | None = None

    def __post_init__(self) -> None:
        _validate_integer("timeout_seconds", self.timeout_seconds, minimum=1)
        _validate_integer("max_parameter_count", self.max_parameter_count, minimum=1)
        for name in ("max_peak_process_rss_bytes", "max_peak_device_bytes"):
            value = getattr(self, name)
            if value is not None:
                _validate_integer(name, value, minimum=1)
        if self.min_training_tokens_per_second is not None:
            _validate_finite(
                "min_training_tokens_per_second",
                self.min_training_tokens_per_second,
                minimum=0.0,
            )
            if self.min_training_tokens_per_second == 0.0:
                raise RecordValidationError("min_training_tokens_per_second must be positive")
        for name in (
            "max_training_throughput_regression_fraction",
            "max_peak_process_rss_regression_fraction",
            "max_peak_device_regression_fraction",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_finite(name, value, minimum=0.0)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    relative_path: str
    sha256: str
    size_bytes: int
    retention: ArtifactRetention

    def __post_init__(self) -> None:
        _validate_id("artifact_id", self.artifact_id)
        _validate_text("kind", self.kind, maximum=80)
        _validate_text("relative_path", self.relative_path, maximum=1_000)
        _validate_sha256("sha256", self.sha256)
        _validate_integer("size_bytes", self.size_bytes, minimum=0)
        _validate_enum("retention", self.retention, ArtifactRetention)
        candidate = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or "\\" in self.relative_path
            or candidate.as_posix() in {"", "."}
        ):
            raise RecordValidationError("artifact paths must be safe relative POSIX paths")


@dataclass(frozen=True, slots=True)
class PatchProposal:
    RECORD_TYPE: ClassVar[str] = "patch_proposal"

    proposal_id: str
    parent_commit: str
    title: str
    hypothesis: str
    mechanism: str
    change: str
    expected_effect_bpb: float
    minimum_useful_gain_bpb: float
    resource_risk: str
    failure_signal: str
    interaction_risk: str
    primary_metric: str = "validation_bpb"
    expected_direction: str = "lower"

    def __post_init__(self) -> None:
        _validate_id("proposal_id", self.proposal_id)
        _validate_git_commit("parent_commit", self.parent_commit)
        for name in (
            "title",
            "hypothesis",
            "mechanism",
            "change",
            "resource_risk",
            "failure_signal",
            "interaction_risk",
        ):
            _validate_text(name, getattr(self, name))
        _validate_finite("expected_effect_bpb", self.expected_effect_bpb)
        _validate_finite("minimum_useful_gain_bpb", self.minimum_useful_gain_bpb, minimum=0.0)
        if self.minimum_useful_gain_bpb == 0.0:
            raise RecordValidationError("minimum_useful_gain_bpb must be positive")
        if self.primary_metric != "validation_bpb" or self.expected_direction != "lower":
            raise RecordValidationError(
                "the initial research contract requires lower validation_bpb"
            )


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    RECORD_TYPE: ClassVar[str] = "candidate"

    candidate_id: str
    proposal_id: str
    parent_commit: str
    candidate_commit: str
    diff_sha256: str
    changed_paths: tuple[str, ...]
    trainer_sha256: str
    policy_sha256: str
    parameter_count: int

    def __post_init__(self) -> None:
        _validate_id("candidate_id", self.candidate_id)
        _validate_id("proposal_id", self.proposal_id)
        _validate_git_commit("parent_commit", self.parent_commit)
        _validate_git_commit("candidate_commit", self.candidate_commit)
        if self.parent_commit == self.candidate_commit:
            raise RecordValidationError("candidate_commit must differ from parent_commit")
        for name in ("diff_sha256", "trainer_sha256", "policy_sha256"):
            _validate_sha256(name, getattr(self, name))
        if not self.changed_paths:
            raise RecordValidationError("changed_paths cannot be empty")
        _validate_unique("changed_paths", self.changed_paths)
        for path in self.changed_paths:
            _validate_text("changed path", path, maximum=1_000)
        try:
            assert_research_paths_allowed(list(self.changed_paths))
        except (ProtectedPathError, ValueError) as error:
            raise RecordValidationError(str(error)) from error
        _validate_integer(
            "parameter_count",
            self.parameter_count,
            minimum=1,
            maximum=DEFAULT_PARAMETER_CAP,
        )


@dataclass(frozen=True, slots=True)
class TrialSpec:
    RECORD_TYPE: ClassVar[str] = "trial_spec"

    trial_id: str
    candidate_id: str
    parent_commit: str
    candidate_commit: str
    stage: ExperimentStage
    seed: int
    token_budget: int
    eval_tokens: int | None
    batch_size: int
    eval_batch_size: int
    execution_order: tuple[RunArm, RunArm]
    data_config_sha256: str
    tokenizer_sha256: str
    parent_trainer_sha256: str
    candidate_trainer_sha256: str
    evaluator_sha256: str
    runner_sha256: str
    environment_sha256: str
    order_assignment_sha256: str
    device: str
    limits: ResourceLimits

    def __post_init__(self) -> None:
        _validate_id("trial_id", self.trial_id)
        _validate_id("candidate_id", self.candidate_id)
        _validate_git_commit("parent_commit", self.parent_commit)
        _validate_git_commit("candidate_commit", self.candidate_commit)
        if self.parent_commit == self.candidate_commit:
            raise RecordValidationError("trial parent and candidate commits must differ")
        _validate_enum("stage", self.stage, ExperimentStage)
        _validate_seed(self.seed)
        _validate_integer("token_budget", self.token_budget, minimum=1)
        if self.eval_tokens is not None:
            _validate_integer("eval_tokens", self.eval_tokens, minimum=1)
        _validate_integer("batch_size", self.batch_size, minimum=1)
        _validate_integer("eval_batch_size", self.eval_batch_size, minimum=1)
        _validate_unique("execution_order", self.execution_order)
        for arm in self.execution_order:
            _validate_enum("execution order arm", arm, RunArm)
        if self.execution_order not in (
            (RunArm.PARENT, RunArm.CANDIDATE),
            (RunArm.CANDIDATE, RunArm.PARENT),
        ):
            raise RecordValidationError("execution_order must contain parent and candidate once")
        for name in (
            "data_config_sha256",
            "tokenizer_sha256",
            "parent_trainer_sha256",
            "candidate_trainer_sha256",
            "evaluator_sha256",
            "runner_sha256",
            "environment_sha256",
            "order_assignment_sha256",
        ):
            _validate_sha256(name, getattr(self, name))
        _validate_text("device", self.device, maximum=80)
        if not isinstance(self.limits, ResourceLimits):
            raise RecordValidationError("limits must be a ResourceLimits value")


@dataclass(frozen=True, slots=True)
class TrialSchedule:
    RECORD_TYPE: ClassVar[str] = "trial_schedule"

    schedule_id: str
    candidate_id: str
    parent_commit: str
    stage: ExperimentStage
    seeds: tuple[int, ...]
    assignment_seed: int
    token_budget: int
    eval_tokens: int | None
    batch_size: int
    eval_batch_size: int
    limits: ResourceLimits
    policy_sha256: str
    source_effect_estimate_id: str | None
    scheduler_version: str
    reason: str

    def __post_init__(self) -> None:
        _validate_id("schedule_id", self.schedule_id)
        _validate_id("candidate_id", self.candidate_id)
        _validate_git_commit("parent_commit", self.parent_commit)
        _validate_enum("stage", self.stage, ExperimentStage)
        if not self.seeds:
            raise RecordValidationError("scheduled seeds cannot be empty")
        _validate_unique("scheduled seeds", self.seeds)
        for seed in self.seeds:
            _validate_seed(seed)
        _validate_seed(self.assignment_seed)
        _validate_integer("token_budget", self.token_budget, minimum=1)
        if self.eval_tokens is not None:
            _validate_integer("eval_tokens", self.eval_tokens, minimum=1)
        _validate_integer("batch_size", self.batch_size, minimum=1)
        _validate_integer("eval_batch_size", self.eval_batch_size, minimum=1)
        if not isinstance(self.limits, ResourceLimits):
            raise RecordValidationError("limits must be a ResourceLimits value")
        _validate_sha256("policy_sha256", self.policy_sha256)
        if self.source_effect_estimate_id is not None:
            _validate_id("source_effect_estimate_id", self.source_effect_estimate_id)
        _validate_text("scheduler_version", self.scheduler_version, maximum=120)
        _validate_text("reason", self.reason)


@dataclass(frozen=True, slots=True)
class RunResult:
    RECORD_TYPE: ClassVar[str] = "run_result"

    run_id: str
    trial_id: str
    arm: RunArm
    status: RunStatus
    seed: int
    target_tokens: int
    tokens_seen: int
    evaluation_tokens: int
    parameter_count: int
    validation_bpb: float | None
    mean_train_loss: float | None
    training_tokens_per_second: float | None
    evaluation_tokens_per_second: float | None
    peak_process_rss_bytes: int | None
    peak_device_allocated_bytes: int | None
    peak_device_reserved_bytes: int | None
    training_seconds: float
    evaluation_seconds: float | None
    wall_seconds: float
    data_order_sha256: str | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_id("run_id", self.run_id)
        _validate_id("trial_id", self.trial_id)
        _validate_enum("arm", self.arm, RunArm)
        _validate_enum("status", self.status, RunStatus)
        _validate_seed(self.seed)
        _validate_integer("target_tokens", self.target_tokens, minimum=1)
        _validate_integer("tokens_seen", self.tokens_seen, minimum=0)
        if self.tokens_seen > self.target_tokens:
            raise RecordValidationError("tokens_seen must be within the declared target")
        _validate_integer("evaluation_tokens", self.evaluation_tokens, minimum=0)
        _validate_integer("parameter_count", self.parameter_count, minimum=1)
        for name in ("training_seconds", "wall_seconds"):
            _validate_finite(name, getattr(self, name), minimum=0.0)
        if self.evaluation_seconds is not None:
            _validate_finite("evaluation_seconds", self.evaluation_seconds, minimum=0.0)
        for name in (
            "peak_process_rss_bytes",
            "peak_device_allocated_bytes",
            "peak_device_reserved_bytes",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_integer(name, value, minimum=0)

        metric_names = (
            "validation_bpb",
            "mean_train_loss",
            "training_tokens_per_second",
            "evaluation_tokens_per_second",
        )
        for name in metric_names:
            value = getattr(self, name)
            if value is not None:
                _validate_finite(name, value, minimum=0.0)

        if self.status is RunStatus.SUCCEEDED:
            if self.tokens_seen != self.target_tokens:
                raise RecordValidationError("successful runs must consume their exact token budget")
            required = (
                "validation_bpb",
                "mean_train_loss",
                "training_tokens_per_second",
                "evaluation_tokens_per_second",
                "peak_process_rss_bytes",
                "data_order_sha256",
            )
            if any(getattr(self, name) is None for name in required):
                raise RecordValidationError("successful runs are missing required outcomes")
            if self.evaluation_tokens <= 0:
                raise RecordValidationError("successful runs must evaluate held-out tokens")
            assert self.data_order_sha256 is not None
            _validate_sha256("data_order_sha256", self.data_order_sha256)
            assert self.training_tokens_per_second is not None
            assert self.evaluation_tokens_per_second is not None
            if self.training_tokens_per_second == 0.0:
                raise RecordValidationError("successful runs require positive training throughput")
            if self.evaluation_tokens_per_second == 0.0:
                raise RecordValidationError(
                    "successful runs require positive evaluation throughput"
                )
            if self.failure_reason is not None:
                raise RecordValidationError("successful runs cannot contain a failure reason")
        else:
            if self.validation_bpb is not None:
                raise RecordValidationError("failed runs cannot claim validation BPB")
            if self.failure_reason is None:
                raise RecordValidationError("failed runs require a failure reason")
            _validate_text("failure_reason", self.failure_reason)
            if self.data_order_sha256 is not None:
                _validate_sha256("data_order_sha256", self.data_order_sha256)


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    RECORD_TYPE: ClassVar[str] = "artifact_manifest"

    manifest_id: str
    run_id: str
    artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        _validate_id("manifest_id", self.manifest_id)
        _validate_id("run_id", self.run_id)
        if not self.artifacts:
            raise RecordValidationError("artifact manifests cannot be empty")
        _validate_unique("artifacts", self.artifacts)
        if any(not isinstance(item, ArtifactRef) for item in self.artifacts):
            raise RecordValidationError("artifact manifests must contain ArtifactRef values")
        _validate_unique("artifact IDs", tuple(item.artifact_id for item in self.artifacts))
        _validate_unique("artifact paths", tuple(item.relative_path for item in self.artifacts))


@dataclass(frozen=True, slots=True)
class PairedResult:
    RECORD_TYPE: ClassVar[str] = "paired_result"

    paired_result_id: str
    trial_id: str
    candidate_id: str
    seed: int
    parent_run_id: str
    candidate_run_id: str
    parent_bpb: float
    candidate_bpb: float
    gain_bpb: float
    training_throughput_delta: float
    peak_process_rss_delta_bytes: int
    constraints_passed: bool
    constraint_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("paired_result_id", "trial_id", "candidate_id"):
            _validate_id(name, getattr(self, name))
        for name in ("parent_run_id", "candidate_run_id"):
            _validate_id(name, getattr(self, name))
        if self.parent_run_id == self.candidate_run_id:
            raise RecordValidationError("paired runs must be distinct")
        _validate_seed(self.seed)
        for name in (
            "parent_bpb",
            "candidate_bpb",
            "gain_bpb",
            "training_throughput_delta",
        ):
            _validate_finite(name, getattr(self, name))
        expected_gain = self.parent_bpb - self.candidate_bpb
        if not math.isclose(self.gain_bpb, expected_gain, rel_tol=0.0, abs_tol=1e-12):
            raise RecordValidationError("gain_bpb must equal parent_bpb - candidate_bpb")
        _validate_integer("peak_process_rss_delta_bytes", self.peak_process_rss_delta_bytes)
        _validate_unique("constraint_failures", self.constraint_failures)
        _validate_boolean("constraints_passed", self.constraints_passed)
        if self.constraints_passed == bool(self.constraint_failures):
            raise RecordValidationError(
                "constraints_passed must be true exactly when constraint_failures is empty"
            )
        for failure in self.constraint_failures:
            _validate_text("constraint failure", failure)


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    RECORD_TYPE: ClassVar[str] = "effect_estimate"

    estimate_id: str
    candidate_id: str
    stage: ExperimentStage
    paired_result_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    mean_gain_bpb: float
    sample_variance: float
    standard_error: float
    minimum_useful_gain_bpb: float
    probability_exceeds_minimum: float
    constraints_passed: bool
    estimator_version: str

    def __post_init__(self) -> None:
        _validate_id("estimate_id", self.estimate_id)
        _validate_id("candidate_id", self.candidate_id)
        _validate_enum("stage", self.stage, ExperimentStage)
        if not self.paired_result_ids or len(self.paired_result_ids) != len(self.seeds):
            raise RecordValidationError(
                "paired_result_ids and seeds must have equal nonzero length"
            )
        for item in self.paired_result_ids:
            _validate_id("paired result ID", item)
        _validate_unique("paired_result_ids", self.paired_result_ids)
        _validate_unique("seeds", self.seeds)
        for seed in self.seeds:
            _validate_seed(seed)
        for name in (
            "mean_gain_bpb",
            "sample_variance",
            "standard_error",
            "minimum_useful_gain_bpb",
        ):
            _validate_finite(name, getattr(self, name))
        if self.sample_variance < 0.0 or self.standard_error < 0.0:
            raise RecordValidationError("effect uncertainty cannot be negative")
        if self.minimum_useful_gain_bpb <= 0.0:
            raise RecordValidationError("minimum_useful_gain_bpb must be positive")
        _validate_probability("probability_exceeds_minimum", self.probability_exceeds_minimum)
        _validate_boolean("constraints_passed", self.constraints_passed)
        _validate_text("estimator_version", self.estimator_version, maximum=120)


@dataclass(frozen=True, slots=True)
class DownstreamPrediction:
    RECORD_TYPE: ClassVar[str] = "downstream_prediction"

    prediction_id: str
    candidate_id: str
    source_trial_ids: tuple[str, ...]
    source_stages: tuple[ExperimentStage, ...]
    target_stage: ExperimentStage
    expected_gain_bpb: float
    predictive_standard_deviation: float
    interval_lower_bpb: float
    interval_upper_bpb: float
    minimum_useful_gain_bpb: float
    probability_exceeds_minimum: float
    model_version: str
    full_budget_label_count: int

    def __post_init__(self) -> None:
        _validate_id("prediction_id", self.prediction_id)
        _validate_id("candidate_id", self.candidate_id)
        if not self.source_trial_ids:
            raise RecordValidationError("source_trial_ids cannot be empty")
        for item in self.source_trial_ids:
            _validate_id("source trial ID", item)
        _validate_unique("source_trial_ids", self.source_trial_ids)
        if not self.source_stages:
            raise RecordValidationError("source_stages cannot be empty")
        _validate_unique("source_stages", self.source_stages)
        for stage in self.source_stages:
            _validate_enum("source stage", stage, ExperimentStage)
        _validate_enum("target_stage", self.target_stage, ExperimentStage)
        for name in (
            "expected_gain_bpb",
            "predictive_standard_deviation",
            "interval_lower_bpb",
            "interval_upper_bpb",
            "minimum_useful_gain_bpb",
        ):
            _validate_finite(name, getattr(self, name))
        if self.predictive_standard_deviation < 0.0:
            raise RecordValidationError("predictive_standard_deviation cannot be negative")
        if self.interval_lower_bpb > self.interval_upper_bpb:
            raise RecordValidationError("predictive interval bounds are reversed")
        if self.minimum_useful_gain_bpb <= 0.0:
            raise RecordValidationError("minimum_useful_gain_bpb must be positive")
        _validate_probability("probability_exceeds_minimum", self.probability_exceeds_minimum)
        _validate_text("model_version", self.model_version, maximum=120)
        _validate_integer("full_budget_label_count", self.full_budget_label_count, minimum=0)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    RECORD_TYPE: ClassVar[str] = "decision"

    decision_id: str
    candidate_id: str
    stage: ExperimentStage
    verdict: DecisionVerdict
    effect_estimate_id: str | None
    downstream_prediction_id: str | None
    minimum_useful_gain_bpb: float
    probability_threshold: float
    constraints_passed: bool
    reasons: tuple[str, ...]
    next_stage: ExperimentStage | None = None
    resulting_parent_commit: str | None = None

    def __post_init__(self) -> None:
        for name in ("decision_id", "candidate_id"):
            _validate_id(name, getattr(self, name))
        if self.effect_estimate_id is not None:
            _validate_id("effect_estimate_id", self.effect_estimate_id)
        _validate_enum("stage", self.stage, ExperimentStage)
        _validate_enum("verdict", self.verdict, DecisionVerdict)
        if self.downstream_prediction_id is not None:
            _validate_id("downstream_prediction_id", self.downstream_prediction_id)
        _validate_finite("minimum_useful_gain_bpb", self.minimum_useful_gain_bpb)
        if self.minimum_useful_gain_bpb <= 0.0:
            raise RecordValidationError("minimum_useful_gain_bpb must be positive")
        _validate_probability("probability_threshold", self.probability_threshold)
        if not self.reasons:
            raise RecordValidationError("decisions require at least one reason")
        _validate_unique("reasons", self.reasons)
        for reason in self.reasons:
            _validate_text("decision reason", reason)
        _validate_boolean("constraints_passed", self.constraints_passed)
        if self.next_stage is not None:
            _validate_enum("next_stage", self.next_stage, ExperimentStage)
        if self.verdict is DecisionVerdict.ESCALATE:
            if self.effect_estimate_id is None:
                raise RecordValidationError("escalation requires an effect estimate")
            if self.next_stage is None or self.resulting_parent_commit is not None:
                raise RecordValidationError("escalation requires next_stage and no new parent")
        elif self.verdict is DecisionVerdict.PROMOTE:
            if self.effect_estimate_id is None:
                raise RecordValidationError("promotion requires an effect estimate")
            if self.next_stage is not None or self.resulting_parent_commit is None:
                raise RecordValidationError("promotion requires resulting_parent_commit")
            _validate_git_commit("resulting_parent_commit", self.resulting_parent_commit)
            if not self.constraints_passed:
                raise RecordValidationError("a constraint failure cannot be promoted")
        elif self.next_stage is not None or self.resulting_parent_commit is not None:
            raise RecordValidationError("rejection cannot set next_stage or a new parent")


@dataclass(frozen=True, slots=True)
class LineageRecord:
    RECORD_TYPE: ClassVar[str] = "lineage"

    lineage_id: str
    generation: int
    previous_lineage_id: str | None
    parent_commit: str
    candidate_id: str
    candidate_commit: str
    decision_id: str

    def __post_init__(self) -> None:
        for name in ("lineage_id", "candidate_id", "decision_id"):
            _validate_id(name, getattr(self, name))
        if self.previous_lineage_id is not None:
            _validate_id("previous_lineage_id", self.previous_lineage_id)
        _validate_integer("generation", self.generation, minimum=1)
        _validate_git_commit("parent_commit", self.parent_commit)
        _validate_git_commit("candidate_commit", self.candidate_commit)
        if self.parent_commit == self.candidate_commit:
            raise RecordValidationError("lineage parent and candidate commits must differ")


@dataclass(frozen=True, slots=True)
class ComputeRecord:
    RECORD_TYPE: ClassVar[str] = "compute"

    compute_id: str
    trial_id: str
    run_id: str
    device: str
    wall_seconds: float
    accelerator_seconds: float
    training_tokens: int
    evaluation_tokens: int
    attempts: int
    estimated_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("compute_id", "trial_id", "run_id"):
            _validate_id(name, getattr(self, name))
        _validate_text("device", self.device, maximum=80)
        for name in ("wall_seconds", "accelerator_seconds"):
            _validate_finite(name, getattr(self, name), minimum=0.0)
        _validate_integer("training_tokens", self.training_tokens, minimum=0)
        _validate_integer("evaluation_tokens", self.evaluation_tokens, minimum=0)
        _validate_integer("attempts", self.attempts, minimum=1)
        if self.estimated_cost_usd is not None:
            _validate_finite("estimated_cost_usd", self.estimated_cost_usd, minimum=0.0)


ExperimentRecord: TypeAlias = (
    PatchProposal
    | CandidateRecord
    | TrialSpec
    | TrialSchedule
    | RunResult
    | ArtifactManifest
    | PairedResult
    | EffectEstimate
    | DownstreamPrediction
    | DecisionRecord
    | LineageRecord
    | ComputeRecord
)

_RECORD_CLASSES: dict[str, type[ExperimentRecord]] = {
    cls.RECORD_TYPE: cls
    for cls in (
        PatchProposal,
        CandidateRecord,
        TrialSpec,
        TrialSchedule,
        RunResult,
        ArtifactManifest,
        PairedResult,
        EffectEstimate,
        DownstreamPrediction,
        DecisionRecord,
        LineageRecord,
        ComputeRecord,
    )
}

_RECORD_ID_FIELDS: dict[type[ExperimentRecord], str] = {
    PatchProposal: "proposal_id",
    CandidateRecord: "candidate_id",
    TrialSpec: "trial_id",
    TrialSchedule: "schedule_id",
    RunResult: "run_id",
    ArtifactManifest: "manifest_id",
    PairedResult: "paired_result_id",
    EffectEstimate: "estimate_id",
    DownstreamPrediction: "prediction_id",
    DecisionRecord: "decision_id",
    LineageRecord: "lineage_id",
    ComputeRecord: "compute_id",
}


def record_id(record: ExperimentRecord) -> str:
    return str(getattr(record, _RECORD_ID_FIELDS[type(record)]))


def _json_value(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def record_to_envelope(record: ExperimentRecord) -> dict[str, Any]:
    return {
        "payload": _json_value(record),
        "record_id": record_id(record),
        "record_type": record.RECORD_TYPE,
        "schema_version": RECORD_SCHEMA_VERSION,
    }


def _strict_payload(record_class: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    expected = {field.name for field in fields(record_class)}
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise RecordValidationError("record payload fields differ: " + "; ".join(details))
    return dict(payload)


def _parse_record(record_type: str, payload: dict[str, Any]) -> ExperimentRecord:
    record_class = _RECORD_CLASSES.get(record_type)
    if record_class is None:
        raise RecordValidationError(f"unsupported record type: {record_type}")
    values = _strict_payload(record_class, payload)

    tuple_fields: dict[str, tuple[str, ...]] = {
        CandidateRecord.RECORD_TYPE: ("changed_paths",),
        TrialSpec.RECORD_TYPE: ("execution_order",),
        TrialSchedule.RECORD_TYPE: ("seeds",),
        ArtifactManifest.RECORD_TYPE: ("artifacts",),
        PairedResult.RECORD_TYPE: ("constraint_failures",),
        EffectEstimate.RECORD_TYPE: ("paired_result_ids", "seeds"),
        DownstreamPrediction.RECORD_TYPE: (
            "source_trial_ids",
            "source_stages",
        ),
        DecisionRecord.RECORD_TYPE: ("reasons",),
    }
    for name in tuple_fields.get(record_type, ()):
        if not isinstance(values[name], list):
            raise RecordValidationError(f"{name} must be a JSON array")
        values[name] = tuple(values[name])

    try:
        if record_type == TrialSpec.RECORD_TYPE:
            values["stage"] = ExperimentStage(values["stage"])
            values["execution_order"] = tuple(RunArm(item) for item in values["execution_order"])
            if not isinstance(values["limits"], dict):
                raise RecordValidationError("limits must be an object")
            values["limits"] = ResourceLimits(**_strict_payload(ResourceLimits, values["limits"]))
        elif record_type == TrialSchedule.RECORD_TYPE:
            values["stage"] = ExperimentStage(values["stage"])
            if not isinstance(values["limits"], dict):
                raise RecordValidationError("limits must be an object")
            values["limits"] = ResourceLimits(**_strict_payload(ResourceLimits, values["limits"]))
        elif record_type == RunResult.RECORD_TYPE:
            values["arm"] = RunArm(values["arm"])
            values["status"] = RunStatus(values["status"])
        elif record_type == ArtifactManifest.RECORD_TYPE:
            artifacts = []
            for artifact in values["artifacts"]:
                if not isinstance(artifact, dict):
                    raise RecordValidationError("artifact entries must be objects")
                artifact_values = _strict_payload(ArtifactRef, artifact)
                artifact_values["retention"] = ArtifactRetention(artifact_values["retention"])
                artifacts.append(ArtifactRef(**artifact_values))
            values["artifacts"] = tuple(artifacts)
        elif record_type == EffectEstimate.RECORD_TYPE:
            values["stage"] = ExperimentStage(values["stage"])
        elif record_type == DownstreamPrediction.RECORD_TYPE:
            values["source_stages"] = tuple(
                ExperimentStage(item) for item in values["source_stages"]
            )
            values["target_stage"] = ExperimentStage(values["target_stage"])
        elif record_type == DecisionRecord.RECORD_TYPE:
            values["stage"] = ExperimentStage(values["stage"])
            values["verdict"] = DecisionVerdict(values["verdict"])
            if values["next_stage"] is not None:
                values["next_stage"] = ExperimentStage(values["next_stage"])

        return record_class(**values)
    except (TypeError, ValueError) as error:
        if isinstance(error, RecordValidationError):
            raise
        raise RecordValidationError(f"invalid {record_type} record: {error}") from error


def record_from_envelope(value: dict[str, Any]) -> ExperimentRecord:
    if not isinstance(value, dict):
        raise RecordValidationError("record envelope must be an object")
    expected = {"payload", "record_id", "record_type", "schema_version"}
    if set(value) != expected:
        raise RecordValidationError("record envelope fields differ from schema")
    if type(value["schema_version"]) is not int:
        raise RecordValidationError("record schema_version must be an integer")
    if value["schema_version"] != RECORD_SCHEMA_VERSION:
        raise RecordValidationError(f"unsupported record schema version: {value['schema_version']}")
    if not isinstance(value["record_id"], str):
        raise RecordValidationError("record envelope record_id must be text")
    if not isinstance(value["record_type"], str):
        raise RecordValidationError("record envelope record_type must be text")
    if not isinstance(value["payload"], dict):
        raise RecordValidationError("record payload must be an object")
    record = _parse_record(value["record_type"], value["payload"])
    if value["record_id"] != record_id(record):
        raise RecordValidationError("envelope record_id does not match its payload")
    return record


def build_paired_result(
    paired_result_id: str,
    *,
    trial: TrialSpec,
    candidate_id: str,
    parent: RunResult,
    candidate: RunResult,
    constraint_failures: tuple[str, ...] = (),
) -> PairedResult:
    if parent.arm is not RunArm.PARENT or candidate.arm is not RunArm.CANDIDATE:
        raise RecordValidationError("paired result arms are reversed")
    if parent.status is not RunStatus.SUCCEEDED or candidate.status is not RunStatus.SUCCEEDED:
        raise RecordValidationError("paired results require two successful runs")
    if parent.trial_id != trial.trial_id or candidate.trial_id != trial.trial_id:
        raise RecordValidationError("paired runs do not belong to the trial")
    assert parent.validation_bpb is not None
    assert candidate.validation_bpb is not None
    assert parent.training_tokens_per_second is not None
    assert candidate.training_tokens_per_second is not None
    assert parent.peak_process_rss_bytes is not None
    assert candidate.peak_process_rss_bytes is not None
    return PairedResult(
        paired_result_id=paired_result_id,
        trial_id=trial.trial_id,
        candidate_id=candidate_id,
        seed=trial.seed,
        parent_run_id=parent.run_id,
        candidate_run_id=candidate.run_id,
        parent_bpb=parent.validation_bpb,
        candidate_bpb=candidate.validation_bpb,
        gain_bpb=parent.validation_bpb - candidate.validation_bpb,
        training_throughput_delta=(
            candidate.training_tokens_per_second - parent.training_tokens_per_second
        ),
        peak_process_rss_delta_bytes=(
            candidate.peak_process_rss_bytes - parent.peak_process_rss_bytes
        ),
        constraints_passed=not constraint_failures,
        constraint_failures=constraint_failures,
    )


def build_effect_estimate(
    estimate_id: str,
    *,
    candidate_id: str,
    stage: ExperimentStage,
    pairs: tuple[PairedResult, ...],
    minimum_useful_gain_bpb: float,
    probability_exceeds_minimum: float,
    estimator_version: str,
) -> EffectEstimate:
    if not pairs:
        raise RecordValidationError("at least one paired result is required")
    gains = [pair.gain_bpb for pair in pairs]
    variance = statistics.variance(gains) if len(gains) > 1 else 0.0
    standard_error = math.sqrt(variance / len(gains))
    return EffectEstimate(
        estimate_id=estimate_id,
        candidate_id=candidate_id,
        stage=stage,
        paired_result_ids=tuple(pair.paired_result_id for pair in pairs),
        seeds=tuple(pair.seed for pair in pairs),
        mean_gain_bpb=statistics.fmean(gains),
        sample_variance=variance,
        standard_error=standard_error,
        minimum_useful_gain_bpb=minimum_useful_gain_bpb,
        probability_exceeds_minimum=probability_exceeds_minimum,
        constraints_passed=all(pair.constraints_passed for pair in pairs),
        estimator_version=estimator_version,
    )
