"""Versioned control-plane contracts for customizable RL and RLVR targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from autodidact.integrity import ProtectedPathError, normalize_repository_path

RL_CONTRACT_SCHEMA_VERSION = 1
_RL_KEYS = frozenset(
    {
        "algorithm_paths",
        "budget_unit",
        "paradigm",
        "reward_maximum",
        "reward_minimum",
        "reward_source",
        "schema_version",
    }
)


class RLContractError(ValueError):
    """Raised when an RL target or its emitted diagnostics violate the contract."""


class TrainingParadigm(StrEnum):
    RL = "rl"
    RLVR = "rlvr"


class RewardSource(StrEnum):
    ENVIRONMENT = "environment"
    REWARD_MODEL = "reward_model"
    VERIFIER = "verifier"
    HYBRID = "hybrid"


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RLContractError(f"{field} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise RLContractError(f"{field} must be finite numeric data")
    return result


def _bounded(value: Any, *, field: str, minimum: float, maximum: float) -> float:
    result = _finite(value, field=field)
    if result < minimum or result > maximum:
        raise RLContractError(f"{field} must be between {minimum} and {maximum}")
    return result


def _optional_finite(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    result = _finite(value, field=field)
    if minimum is not None and result < minimum:
        raise RLContractError(f"{field} must be at least {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class RLTargetContract:
    """Protected facts about an RL target, intentionally excluding an algorithm choice."""

    paradigm: TrainingParadigm
    reward_source: RewardSource
    budget_unit: str
    reward_minimum: float
    reward_maximum: float
    algorithm_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            paradigm = TrainingParadigm(self.paradigm)
            reward_source = RewardSource(self.reward_source)
        except (TypeError, ValueError) as error:
            raise RLContractError("RL paradigm or reward source is invalid") from error
        object.__setattr__(self, "paradigm", paradigm)
        object.__setattr__(self, "reward_source", reward_source)
        if paradigm is TrainingParadigm.RLVR and reward_source not in {
            RewardSource.VERIFIER,
            RewardSource.HYBRID,
        }:
            raise RLContractError("RLVR requires a verifier or hybrid reward source")
        if (
            not isinstance(self.budget_unit, str)
            or not self.budget_unit.strip()
            or len(self.budget_unit) > 64
        ):
            raise RLContractError("RL budget_unit must be nonempty text of at most 64 characters")
        minimum = _finite(self.reward_minimum, field="rl.reward_minimum")
        maximum = _finite(self.reward_maximum, field="rl.reward_maximum")
        if minimum >= maximum:
            raise RLContractError("rl.reward_minimum must be below rl.reward_maximum")
        object.__setattr__(self, "reward_minimum", minimum)
        object.__setattr__(self, "reward_maximum", maximum)
        try:
            paths = tuple(
                normalize_repository_path(path, field="rl.algorithm_paths")
                for path in self.algorithm_paths
            )
        except ProtectedPathError as error:
            raise RLContractError(str(error)) from error
        if not paths or len(set(paths)) != len(paths):
            raise RLContractError("rl.algorithm_paths must be a nonempty unique sequence")
        object.__setattr__(self, "algorithm_paths", paths)

    @classmethod
    def from_mapping(cls, value: Any) -> RLTargetContract:
        if not isinstance(value, dict):
            raise RLContractError("rl must be an object")
        unknown = frozenset(value) - _RL_KEYS
        missing = _RL_KEYS - value.keys()
        if unknown:
            raise RLContractError(f"unknown rl keys: {sorted(unknown)}")
        if missing:
            raise RLContractError(f"rl is missing keys: {sorted(missing)}")
        if value["schema_version"] != RL_CONTRACT_SCHEMA_VERSION:
            raise RLContractError("RL contract schema version is unsupported")
        paths = value["algorithm_paths"]
        if not isinstance(paths, list):
            raise RLContractError("rl.algorithm_paths must be an array")
        return cls(
            paradigm=value["paradigm"],
            reward_source=value["reward_source"],
            budget_unit=value["budget_unit"],
            reward_minimum=value["reward_minimum"],
            reward_maximum=value["reward_maximum"],
            algorithm_paths=tuple(paths),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "algorithm_paths": list(self.algorithm_paths),
            "budget_unit": self.budget_unit,
            "paradigm": self.paradigm.value,
            "reward_maximum": self.reward_maximum,
            "reward_minimum": self.reward_minimum,
            "reward_source": self.reward_source.value,
            "schema_version": RL_CONTRACT_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class RLTrainingDiagnostics:
    algorithm_id: str
    mean_reward: float
    reward_standard_deviation: float
    rollout_valid_fraction: float
    policy_loss: float | None
    kl_divergence: float | None
    policy_entropy: float | None


@dataclass(frozen=True, slots=True)
class RLEvaluationDiagnostics:
    reward_standard_deviation: float
    verifier_coverage: float | None


def validate_training_diagnostics(
    contract: RLTargetContract,
    summary: dict[str, Any],
) -> RLTrainingDiagnostics:
    algorithm_id = summary.get("algorithm_id")
    if not isinstance(algorithm_id, str) or not algorithm_id.strip() or len(algorithm_id) > 128:
        raise RLContractError("RL training summary requires a portable algorithm_id")
    mean_reward = _bounded(
        summary.get("mean_train_reward"),
        field="mean_train_reward",
        minimum=contract.reward_minimum,
        maximum=contract.reward_maximum,
    )
    reward_standard_deviation = _optional_finite(
        summary.get("train_reward_standard_deviation"),
        field="train_reward_standard_deviation",
        minimum=0.0,
    )
    if reward_standard_deviation is None:
        raise RLContractError("RL training summary requires train_reward_standard_deviation")
    rollout_valid_fraction = _bounded(
        summary.get("rollout_valid_fraction"),
        field="rollout_valid_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    return RLTrainingDiagnostics(
        algorithm_id=algorithm_id,
        mean_reward=mean_reward,
        reward_standard_deviation=reward_standard_deviation,
        rollout_valid_fraction=rollout_valid_fraction,
        policy_loss=_optional_finite(summary.get("policy_loss"), field="policy_loss"),
        kl_divergence=_optional_finite(
            summary.get("kl_divergence"), field="kl_divergence", minimum=0.0
        ),
        policy_entropy=_optional_finite(
            summary.get("policy_entropy"), field="policy_entropy", minimum=0.0
        ),
    )


def validate_evaluation_diagnostics(
    contract: RLTargetContract,
    result: dict[str, Any],
) -> RLEvaluationDiagnostics:
    _bounded(
        result.get("metric_value"),
        field="metric_value",
        minimum=contract.reward_minimum,
        maximum=contract.reward_maximum,
    )
    reward_standard_deviation = _optional_finite(
        result.get("reward_standard_deviation"),
        field="reward_standard_deviation",
        minimum=0.0,
    )
    if reward_standard_deviation is None:
        raise RLContractError("RL evaluation requires reward_standard_deviation")
    verifier_coverage = None
    if contract.paradigm is TrainingParadigm.RLVR:
        verifier_coverage = _bounded(
            result.get("verifier_coverage"),
            field="verifier_coverage",
            minimum=0.0,
            maximum=1.0,
        )
    return RLEvaluationDiagnostics(
        reward_standard_deviation=reward_standard_deviation,
        verifier_coverage=verifier_coverage,
    )
