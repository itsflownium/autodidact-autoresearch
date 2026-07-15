from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from autodidact.interactions import (
    build_interaction_audit_plan,
    reverse_patch_applicability,
)
from autodidact.ledger import ExperimentLedger
from tests.experiment_fixtures import PARENT_COMMIT, evidence_records, lifecycle_entries


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def test_plan_marks_latest_promotion_as_directly_tested(tmp_path: Path) -> None:
    ledger = ExperimentLedger.create(
        tmp_path / "ledger.sqlite3",
        initial_parent_commit=PARENT_COMMIT,
    )
    ledger.append_many(lifecycle_entries())

    plan = build_interaction_audit_plan(ledger)

    assert plan["plan_id"].startswith("interaction-plan-")
    assert len(plan["audits"]) == 1
    audit = plan["audits"][0]
    assert audit["candidate_id"] == "candidate-001"
    assert audit["requires_leave_one_out"] is False
    assert audit["status"] == "directly_tested_current_stack"
    assert audit["reverse_patch_applicable"] is None


def test_plan_requires_leave_one_out_after_a_later_promotion() -> None:
    first = evidence_records()
    second_proposal = replace(
        first["proposal"],
        proposal_id="proposal-002",
        parent_commit=first["candidate"].candidate_commit,
        title="Tune decay",
    )
    second_candidate = replace(
        first["candidate"],
        candidate_id="candidate-002",
        proposal_id=second_proposal.proposal_id,
        parent_commit=first["candidate"].candidate_commit,
        candidate_commit="c" * 40,
    )
    second_effect = replace(
        first["effect"],
        estimate_id="estimate-002",
        candidate_id=second_candidate.candidate_id,
    )
    second_decision = replace(
        first["decision"],
        decision_id="decision-002",
        candidate_id=second_candidate.candidate_id,
        effect_estimate_id=second_effect.estimate_id,
        resulting_parent_commit=second_candidate.candidate_commit,
    )
    second_lineage = replace(
        first["lineage"],
        lineage_id="lineage-002",
        generation=2,
        previous_lineage_id=first["lineage"].lineage_id,
        parent_commit=second_candidate.parent_commit,
        candidate_id=second_candidate.candidate_id,
        candidate_commit=second_candidate.candidate_commit,
        decision_id=second_decision.decision_id,
    )
    records = (
        first["proposal"],
        first["candidate"],
        first["effect"],
        first["decision"],
        first["lineage"],
        second_proposal,
        second_candidate,
        second_effect,
        second_decision,
        second_lineage,
    )
    ledger = SimpleNamespace(
        verify=lambda: SimpleNamespace(head_event_sha256="d" * 64),
        events=lambda: tuple(SimpleNamespace(record=record) for record in records),
        current_parent=lambda: second_candidate.candidate_commit,
    )

    plan = build_interaction_audit_plan(ledger)  # type: ignore[arg-type]

    assert [audit["status"] for audit in plan["audits"]] == [
        "leave_one_out_required",
        "directly_tested_current_stack",
    ]
    assert plan["audits"][0]["later_promotion_count"] == 1


def test_reverse_patch_applicability_uses_an_isolated_worktree(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    trainer = repository / "train.py"
    trainer.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "parent")
    parent = _git(repository, "rev-parse", "HEAD")
    trainer.write_text("VALUE = 2\n", encoding="utf-8")
    _git(repository, "commit", "-am", "candidate")
    candidate = _git(repository, "rev-parse", "HEAD")

    applicable, error = reverse_patch_applicability(
        repository,
        patch_parent_commit=parent,
        patch_candidate_commit=candidate,
        current_stack_commit=candidate,
    )

    assert applicable is True
    assert error is None
    assert trainer.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert _git(repository, "worktree", "list", "--porcelain").count("worktree ") == 1
