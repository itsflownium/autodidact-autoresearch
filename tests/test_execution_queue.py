from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autodidact.execution_queue import ExecutionQueue, ExecutionQueueError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = REPOSITORY_ROOT / "docs" / "proposals" / "execution-queue.json"


def _head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def test_committed_queue_covers_all_unmeasured_proposals() -> None:
    queue = ExecutionQueue.from_path(QUEUE_PATH, repository_root=REPOSITORY_ROOT)

    assert len(queue.items) == 60
    assert len(queue.conflict_groups) == 16
    assert {item.proposal_number for item in queue.items} == set(range(1, 61))
    assert queue.summary()["tier_counts"] == {
        "defer": 15,
        "explore": 25,
        "screen": 20,
    }
    assert all(item.conflict_groups for item in queue.items)
    assert all(item.proposal["title"] for item in queue.items)


def test_assignment_preserves_source_patch_and_marks_parent_adaptation() -> None:
    queue = ExecutionQueue.from_path(QUEUE_PATH, repository_root=REPOSITORY_ROOT)

    source_assignment = queue.assignment(
        1,
        current_parent_commit=queue.frozen_parent_commit,
    )
    current_assignment = queue.assignment(1, current_parent_commit=_head())

    assert source_assignment["adaptation_required"] is False
    assert current_assignment["adaptation_required"] is True
    assert current_assignment["proposal_number"] == 18
    assert current_assignment["patch"]
    assert current_assignment["patch_sha256"] == queue.items[0].patch_sha256
    assert current_assignment["evidence_status"] == "queued_unmeasured"


def test_tampered_queue_patch_hash_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    payload["items"][0]["patch_sha256"] = "0" * 64
    tampered = tmp_path / "execution-queue.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExecutionQueueError, match="patch hash differs"):
        ExecutionQueue.from_path(tampered, repository_root=REPOSITORY_ROOT)


def test_tampered_conflict_membership_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    payload["items"][0]["conflict_groups"] = []
    tampered = tmp_path / "execution-queue.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExecutionQueueError, match="conflict membership differs"):
        ExecutionQueue.from_path(tampered, repository_root=REPOSITORY_ROOT)
