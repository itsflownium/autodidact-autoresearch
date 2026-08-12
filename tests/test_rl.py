from __future__ import annotations

import pytest

from autodidact.rl import (
    RewardSource,
    RLContractError,
    RLTargetContract,
    TrainingParadigm,
    validate_evaluation_diagnostics,
    validate_training_diagnostics,
)


def _contract() -> RLTargetContract:
    return RLTargetContract(
        paradigm=TrainingParadigm.RLVR,
        reward_source=RewardSource.VERIFIER,
        budget_unit="rollouts",
        reward_minimum=0.0,
        reward_maximum=1.0,
        algorithm_paths=("target/algorithm.py",),
    )


def test_rlvr_contract_protects_reward_semantics_without_selecting_an_algorithm() -> None:
    contract = _contract()
    mapping = contract.to_mapping()

    assert mapping["paradigm"] == "rlvr"
    assert mapping["reward_source"] == "verifier"
    assert mapping["algorithm_paths"] == ["target/algorithm.py"]
    assert "algorithm" not in mapping

    with pytest.raises(RLContractError, match="requires a verifier"):
        RLTargetContract(
            paradigm="rlvr",
            reward_source="reward_model",
            budget_unit="episodes",
            reward_minimum=-1.0,
            reward_maximum=1.0,
            algorithm_paths=("target/algorithm.py",),
        )


def test_custom_algorithm_diagnostics_are_accepted_and_bounded() -> None:
    contract = _contract()
    training = validate_training_diagnostics(
        contract,
        {
            "algorithm_id": "researcher-designed-credit-assignment-v7",
            "kl_divergence": 0.03,
            "mean_train_reward": 0.62,
            "policy_entropy": 1.4,
            "policy_loss": -0.12,
            "rollout_valid_fraction": 0.97,
            "train_reward_standard_deviation": 0.18,
        },
    )
    evaluation = validate_evaluation_diagnostics(
        contract,
        {
            "metric_value": 0.68,
            "reward_standard_deviation": 0.12,
            "verifier_coverage": 1.0,
        },
    )

    assert training.algorithm_id == "researcher-designed-credit-assignment-v7"
    assert training.policy_loss == pytest.approx(-0.12)
    assert evaluation.verifier_coverage == 1.0

    with pytest.raises(RLContractError, match="metric_value"):
        validate_evaluation_diagnostics(
            contract,
            {
                "metric_value": 1.1,
                "reward_standard_deviation": 0.1,
                "verifier_coverage": 1.0,
            },
        )

    with pytest.raises(RLContractError, match="verifier_coverage"):
        validate_evaluation_diagnostics(
            contract,
            {
                "metric_value": 0.7,
                "reward_standard_deviation": 0.1,
            },
        )


def test_rl_contract_rejects_unsafe_or_duplicate_algorithm_paths() -> None:
    with pytest.raises(RLContractError, match="safe repository-relative"):
        RLTargetContract(
            paradigm="rl",
            reward_source="environment",
            budget_unit="steps",
            reward_minimum=-10.0,
            reward_maximum=10.0,
            algorithm_paths=("../algorithm.py",),
        )

    with pytest.raises(RLContractError, match="unique"):
        RLTargetContract(
            paradigm="rl",
            reward_source="environment",
            budget_unit="steps",
            reward_minimum=-10.0,
            reward_maximum=10.0,
            algorithm_paths=("target/algorithm.py", "target/algorithm.py"),
        )
