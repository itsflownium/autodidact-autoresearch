from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from autodidact.records import ExperimentStage, ResourceLimits, RunArm
from autodidact.runner import (
    ExperimentRequest,
    RunnerError,
    assign_execution_orders,
    isolated_worktrees,
    validate_candidate_patch,
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


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    repository = tmp_path / "repository"
    (repository / "target").mkdir(parents=True)
    (repository / "control").mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "target" / "trainer.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "target" / "algorithm.py").write_text("ALGORITHM = 'base'\n", encoding="utf-8")
    (repository / "control" / "evaluate.py").write_text("# protected\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "Add target")
    parent = _git(repository, "rev-parse", "HEAD")
    (repository / "target" / "algorithm.py").write_text(
        "ALGORITHM = 'custom-policy-gradient'\n", encoding="utf-8"
    )
    _git(repository, "add", "target/algorithm.py")
    _git(repository, "commit", "-m", "Change algorithm")
    return repository, parent, _git(repository, "rev-parse", "HEAD")


def test_execution_order_is_seeded_reproducible_and_balanced() -> None:
    first = assign_execution_orders((3, 5, 7, 11, 13), assignment_seed=19)
    second = assign_execution_orders((3, 5, 7, 11, 13), assignment_seed=19)

    assert first == second
    assert all(set(order) == {RunArm.PARENT, RunArm.CANDIDATE} for order in first)
    parent_first = sum(order[0] is RunArm.PARENT for order in first)
    assert parent_first in {2, 3}


def test_patch_validation_allows_custom_algorithm_and_protects_evaluator(
    tmp_path: Path,
) -> None:
    repository, parent, candidate = _repository(tmp_path)

    validated = validate_candidate_patch(
        repository,
        parent_commit=parent,
        candidate_commit=candidate,
        allowed_paths=("target/trainer.py", "target/algorithm.py"),
        trainer_path="target/trainer.py",
    )

    assert validated.changed_paths == ("target/algorithm.py",)
    (repository / "control" / "evaluate.py").write_text("# tampered\n", encoding="utf-8")
    _git(repository, "add", "control/evaluate.py")
    _git(repository, "commit", "-m", "Tamper with evaluator")
    with pytest.raises(RunnerError, match="protected path"):
        validate_candidate_patch(
            repository,
            parent_commit=candidate,
            candidate_commit=_git(repository, "rev-parse", "HEAD"),
            allowed_paths=("target/trainer.py", "target/algorithm.py"),
            trainer_path="target/trainer.py",
        )


def test_isolated_worktrees_materialize_both_commits_and_clean_up(tmp_path: Path) -> None:
    repository, parent, candidate = _repository(tmp_path)
    root = tmp_path / "worktrees"

    with isolated_worktrees(
        repository,
        root,
        parent_commit=parent,
        candidate_commit=candidate,
        trainer_path="target/trainer.py",
    ) as worktrees:
        parent_algorithm = (worktrees[RunArm.PARENT] / "target" / "algorithm.py").read_text()
        candidate_algorithm = (worktrees[RunArm.CANDIDATE] / "target" / "algorithm.py").read_text()
        assert "base" in parent_algorithm
        assert "custom-policy-gradient" in candidate_algorithm

    assert not any(root.iterdir())


def test_experiment_request_requires_a_target_contract(tmp_path: Path) -> None:
    with pytest.raises(RunnerError, match="target configuration"):
        ExperimentRequest(
            repository_root=tmp_path,
            ledger_path=tmp_path / "ledger.sqlite3",
            data_root=tmp_path / "protected",
            output_root=tmp_path / "output",
            proposal_id="proposal-test-001",
            candidate_commit="a" * 40,
            stage=ExperimentStage.CHEAP,
            seeds=(7,),
            assignment_seed=11,
            token_budget=32,
            eval_tokens=16,
            batch_size=2,
            eval_batch_size=2,
            timeout_seconds=30,
            device="cpu",
            limits=ResourceLimits(timeout_seconds=30),
            target_config_path=None,
        )
