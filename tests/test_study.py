from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from autodidact.controller import DecisionMode
from autodidact.ledger import ExperimentLedger
from autodidact.researcher import ResearcherConfig
from autodidact.runstate import CampaignStore
from autodidact.study import (
    StudyArm,
    StudyError,
    StudyLimits,
    _accepted_ref,
    _arm_policy,
    initialize_study,
    load_manifest,
    study_status,
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


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "train.py").write_text("MODEL = 'test'\n", encoding="utf-8")
    (repository / "program.md").write_text("Change train.py only.\n", encoding="utf-8")
    _git(repository, "add", "train.py", "program.md")
    _git(repository, "commit", "-m", "Add study parent")
    return repository, _git(repository, "rev-parse", "HEAD")


def _researcher_config(repository: Path) -> Path:
    path = repository / "researcher.json"
    path.write_text(
        json.dumps(ResearcherConfig(command=(sys.executable, "fake-researcher.py")).to_mapping()),
        encoding="utf-8",
    )
    return path


def _initialize(tmp_path: Path, *, assignment_seed: int = 17):
    repository, parent = _repository(tmp_path)
    researcher = _researcher_config(repository)
    root = tmp_path / "study"
    limits = StudyLimits(
        max_proposals=50,
        max_wall_seconds=10_000,
        max_researcher_tokens=500_000,
        max_training_tokens=7_400_000_000,
        max_compute_seconds=100_000,
    )
    manifest = initialize_study(
        study_root=root,
        repository_root=repository,
        study_id="pilot-study",
        assignment_seed=assignment_seed,
        limits=limits,
        reward_calibration_labels=40,
        researcher_config_path=str(researcher),
        target_config_path=None,
        program_path="program.md",
        data_root="artifacts/data",
        device="cpu",
        estimated_accelerator_hour_usd=None,
    )
    return repository, parent, root, limits, manifest


def test_study_initialization_isolates_three_arms_with_matched_limits(tmp_path: Path) -> None:
    repository, parent, root, limits, manifest = _initialize(tmp_path)

    assert set(manifest.arm_order) == set(StudyArm)
    assert manifest.initial_parent_commit == parent
    assert load_manifest(root) == manifest
    policy_hashes = dict(manifest.policy_sha256)
    assert len(set(policy_hashes.values())) == 3
    for arm in StudyArm:
        arm_root = root / "arms" / arm.value
        state = CampaignStore.open(arm_root / "campaign.sqlite3").snapshot()
        ledger = ExperimentLedger.open(arm_root / "ledger.sqlite3", read_only=True)
        assert state.initial_parent_commit == parent
        assert state.limits.max_proposals == limits.max_proposals
        assert state.limits.max_researcher_tokens == limits.max_researcher_tokens
        assert state.limits.max_training_tokens == limits.max_training_tokens
        assert state.limits.max_compute_seconds == limits.max_compute_seconds
        assert ledger.current_parent() == parent
        assert _git(repository, "rev-parse", _accepted_ref("pilot-study", arm)) == parent
        bayesian = arm is StudyArm.PATCH_RCT_BAYESIAN
        assert state.limits.use_downstream_allocation is bayesian
        assert state.limits.reward_calibration_labels == (40 if bayesian else 0)
        assert state.limits.decision_mode == ("greedy" if arm is StudyArm.GREEDY else "patch_rct")


def test_study_arm_order_and_policy_hashes_are_deterministic(tmp_path: Path) -> None:
    _repository_one, _parent_one, _root_one, _limits_one, first = _initialize(
        tmp_path / "first",
        assignment_seed=29,
    )
    _repository_two, _parent_two, _root_two, _limits_two, second = _initialize(
        tmp_path / "second",
        assignment_seed=29,
    )

    assert first.arm_order == second.arm_order
    assert first.policy_sha256 == second.policy_sha256
    assert (
        _arm_policy(
            StudyArm.GREEDY,
            max_parameter_count=1_050_000,
            minimum_reward_labels=40,
        ).decision_mode
        is DecisionMode.GREEDY
    )
    assert _arm_policy(
        StudyArm.PATCH_RCT_BAYESIAN,
        max_parameter_count=1_050_000,
        minimum_reward_labels=40,
    ).use_downstream_allocation


def test_study_rejects_budget_below_forced_calibration_minimum(tmp_path: Path) -> None:
    repository, _parent = _repository(tmp_path)
    researcher = _researcher_config(repository)
    root = tmp_path / "underfunded-study"

    with pytest.raises(StudyError, match="requires at least 4800000000"):
        initialize_study(
            study_root=root,
            repository_root=repository,
            study_id="underfunded-study",
            assignment_seed=17,
            limits=StudyLimits(
                max_proposals=50,
                max_wall_seconds=10_000,
                max_researcher_tokens=500_000,
                max_training_tokens=4_799_999_999,
                max_compute_seconds=100_000,
            ),
            reward_calibration_labels=40,
            researcher_config_path=str(researcher),
            target_config_path=None,
            program_path="program.md",
            data_root="artifacts/data",
            device="cpu",
            estimated_accelerator_hour_usd=None,
        )

    assert not root.exists()


def test_study_status_reports_each_isolated_arm(tmp_path: Path) -> None:
    _repository_root, parent, root, _limits, manifest = _initialize(tmp_path)

    status = study_status(root)

    assert status["study_id"] == manifest.study_id
    assert status["initial_parent_commit"] == parent
    assert list(status["arms"]) == [arm.value for arm in StudyArm]
    for arm in StudyArm:
        assert status["arms"][arm.value]["ledger"]["event_count"] == 0
        assert status["arms"][arm.value]["lineage_count"] == 0
        assert status["arms"][arm.value]["promotion_count"] == 0


def test_study_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    _repository_root, _parent, root, _limits, _manifest = _initialize(tmp_path)
    manifest_path = root / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["assignment_seed"] += 1
    manifest_path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(StudyError, match="digest is invalid"):
        load_manifest(root)


def test_study_refuses_changed_researcher_configuration(tmp_path: Path) -> None:
    repository, _parent, root, _limits, manifest = _initialize(tmp_path)
    researcher_path = Path(manifest.researcher_config_path)
    researcher_path.write_text(
        json.dumps(ResearcherConfig(command=(sys.executable, "changed.py")).to_mapping()),
        encoding="utf-8",
    )

    with pytest.raises(StudyError, match="researcher configuration changed"):
        study_status(root, repository_root=repository)
