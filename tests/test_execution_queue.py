from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from autodidact.execution_queue import ExecutionQueue, ExecutionQueueError


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _queue(tmp_path: Path) -> tuple[ExecutionQueue, Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    target = repository / "algorithm.py"
    target.write_text("METHOD = 'base'\n", encoding="utf-8")
    _git(repository, "add", "algorithm.py")
    _git(repository, "commit", "-m", "Add target")
    parent = _git(repository, "rev-parse", "HEAD")
    bank_root = repository / "research"
    patch_root = bank_root / "patches"
    patch_root.mkdir(parents=True)
    proposals = []
    metadata = []
    for number, method in ((1, "custom-a"), (2, "custom-b")):
        target.write_text(f"METHOD = '{method}'\n", encoding="utf-8")
        patch = _git(repository, "diff", "--binary", "HEAD", "--", "algorithm.py") + "\n"
        target.write_text("METHOD = 'base'\n", encoding="utf-8")
        relative_patch = f"research/patches/{number:03d}.patch"
        patch_path = repository / relative_patch
        patch_path.write_text(patch, encoding="utf-8")
        digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        proposals.append(
            {
                "change": f"Use {method}.",
                "diff_sha256": digest,
                "expected_effect": 0.02,
                "experimental_status": "queued_unmeasured",
                "failure_signal": "The protected objective does not improve.",
                "hypothesis": f"{method} should improve the protected objective.",
                "interaction_risk": "The alternatives are mutually exclusive.",
                "mechanism": "Change the policy optimization rule.",
                "minimum_useful_gain": 0.005,
                "parent_commit": parent,
                "patch_path": relative_patch,
                "proposal_number": number,
                "resource_risk": "May change runtime slightly.",
                "title": f"Try {method}",
                "training_performed": False,
            }
        )
        metadata.append((relative_patch, digest))
    manifest_relative = "research/proposals.json"
    (repository / manifest_relative).write_text(
        json.dumps(
            {
                "experimental_status": "queued_unmeasured",
                "frozen_parent_commit": parent,
                "proposal_count": 2,
                "proposals": proposals,
                "training_performed": False,
            }
        ),
        encoding="utf-8",
    )
    queue_path = bank_root / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "conflict_groups": [
                    {
                        "group_id": "optimizer-choice",
                        "label": "Optimizer choice",
                        "members": [1, 2],
                        "rationale": "Only one replacement can be active at a time.",
                        "severity": "exclusive",
                    }
                ],
                "evidence_status": "queued_unmeasured",
                "frozen_parent_commit": parent,
                "items": [
                    {
                        "adaptation_risk": "low",
                        "conflict_groups": ["optimizer-choice"],
                        "patch_path": metadata[number - 1][0],
                        "patch_sha256": metadata[number - 1][1],
                        "priority_reason": f"Deterministic rank {number}.",
                        "proposal_number": number,
                        "rank": number,
                        "resource_risk": "low",
                        "source_manifest": manifest_relative,
                        "tier": "screen",
                    }
                    for number in (1, 2)
                ],
                "objective": "Evaluate two unmeasured algorithm proposals.",
                "queue_id": "test-algorithm-queue",
                "ranking_policy": {
                    "principles": ["Use deterministic order."],
                    "tie_breaker": "Lower proposal number first.",
                },
                "schema_version": 2,
                "source_banks": [manifest_relative],
            }
        ),
        encoding="utf-8",
    )
    return ExecutionQueue.from_path(queue_path, repository_root=repository), queue_path, repository


def test_queue_loads_every_unmeasured_proposal_and_builds_assignment(tmp_path: Path) -> None:
    queue, _path, _repository = _queue(tmp_path)

    assert len(queue.items) == 2
    assert queue.summary()["tier_counts"] == {"defer": 0, "explore": 0, "screen": 2}
    source = queue.assignment(1, current_parent_commit=queue.frozen_parent_commit)
    rebased = queue.assignment(1, current_parent_commit="f" * 40)
    assert source["adaptation_required"] is False
    assert rebased["adaptation_required"] is True
    assert rebased["proposal_number"] == 1
    assert rebased["patch"]


def test_queue_rejects_patch_and_conflict_tampering(tmp_path: Path) -> None:
    queue, queue_path, repository = _queue(tmp_path)
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    payload["items"][0]["patch_sha256"] = "0" * 64
    queue_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExecutionQueueError, match="patch hash differs"):
        ExecutionQueue.from_path(queue_path, repository_root=repository)

    payload["items"][0]["patch_sha256"] = queue.items[0].patch_sha256
    payload["items"][0]["conflict_groups"] = []
    queue_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExecutionQueueError, match="conflict membership differs"):
        ExecutionQueue.from_path(queue_path, repository_root=repository)
