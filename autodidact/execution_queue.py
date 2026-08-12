"""Validated, fixed-priority execution queues for unmeasured proposal banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

EXECUTION_QUEUE_SCHEMA_VERSION = 2
DEFAULT_EXECUTION_QUEUE_PATH = Path("docs/proposals/execution-queue.json")
QUEUE_ASSIGNMENT_MARKER = "AUTODIDACT_EXECUTION_QUEUE_ASSIGNMENT_V1"
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TOP_LEVEL_KEYS = frozenset(
    {
        "conflict_groups",
        "evidence_status",
        "frozen_parent_commit",
        "items",
        "objective",
        "queue_id",
        "ranking_policy",
        "schema_version",
        "source_banks",
    }
)
_GROUP_KEYS = frozenset({"group_id", "label", "members", "rationale", "severity"})
_ITEM_KEYS = frozenset(
    {
        "adaptation_risk",
        "conflict_groups",
        "patch_path",
        "patch_sha256",
        "priority_reason",
        "proposal_number",
        "rank",
        "resource_risk",
        "source_manifest",
        "tier",
    }
)
_RANKING_POLICY_KEYS = frozenset({"principles", "tie_breaker"})
_PROPOSAL_FIELDS = (
    "title",
    "hypothesis",
    "mechanism",
    "change",
    "expected_effect",
    "minimum_useful_gain",
    "resource_risk",
    "failure_signal",
    "interaction_risk",
)


class ExecutionQueueError(RuntimeError):
    """Raised when an execution queue or its proposal evidence is invalid."""


class QueueTier(StrEnum):
    SCREEN = "screen"
    EXPLORE = "explore"
    DEFER = "defer"


class QueueRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConflictSeverity(StrEnum):
    EXCLUSIVE = "exclusive"
    REBASE_REQUIRED = "rebase_required"
    BUDGET_RECHECK = "budget_recheck"
    INTERACTION = "interaction"


def _strict_keys(value: dict[str, Any], expected: frozenset[str], *, name: str) -> None:
    keys = frozenset(value)
    if keys != expected:
        raise ExecutionQueueError(
            f"{name} keys differ; missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected)}"
        )


def _required_text(name: str, value: Any, *, maximum: int = 8_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ExecutionQueueError(f"{name} must be nonempty text of at most {maximum} characters")
    return value


def _positive_integer(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ExecutionQueueError(f"{name} must be a positive integer")
    return value


def _portable_relative_path(name: str, value: Any) -> str:
    text = _required_text(name, value, maximum=1_000).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or path.as_posix() in {"", "."} or ".." in path.parts:
        raise ExecutionQueueError(f"{name} must be a safe repository-relative path")
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionQueueError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutionQueueError(f"{name} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class ConflictGroup:
    group_id: str
    label: str
    severity: ConflictSeverity
    rationale: str
    members: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExecutionQueueItem:
    rank: int
    proposal_number: int
    tier: QueueTier
    adaptation_risk: QueueRisk
    resource_risk: QueueRisk
    conflict_groups: tuple[str, ...]
    priority_reason: str
    source_manifest: str
    patch_path: str
    patch_sha256: str
    proposal: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionQueue:
    queue_id: str
    objective: str
    evidence_status: str
    frozen_parent_commit: str
    source_banks: tuple[str, ...]
    ranking_policy: dict[str, Any]
    conflict_groups: tuple[ConflictGroup, ...]
    items: tuple[ExecutionQueueItem, ...]
    path: Path
    repository_root: Path

    @classmethod
    def from_path(cls, path: Path, *, repository_root: Path | None = None) -> ExecutionQueue:
        path = path.expanduser().resolve()
        root = (path.parent if repository_root is None else repository_root).expanduser().resolve()
        value = _load_object(path, name="execution queue")
        _strict_keys(value, _TOP_LEVEL_KEYS, name="execution queue")
        if value["schema_version"] != EXECUTION_QUEUE_SCHEMA_VERSION:
            raise ExecutionQueueError("execution queue schema version is unsupported")
        queue_id = _required_text("queue_id", value["queue_id"], maximum=128)
        if not _ID_PATTERN.fullmatch(queue_id):
            raise ExecutionQueueError("queue_id must be a portable structured identifier")
        objective = _required_text("objective", value["objective"])
        if value["evidence_status"] != "queued_unmeasured":
            raise ExecutionQueueError("execution queue must contain only queued, unmeasured ideas")
        parent = value["frozen_parent_commit"]
        if not isinstance(parent, str) or not _COMMIT_PATTERN.fullmatch(parent):
            raise ExecutionQueueError("frozen_parent_commit must be a full lowercase Git commit")

        source_banks_value = value["source_banks"]
        if not isinstance(source_banks_value, list) or not source_banks_value:
            raise ExecutionQueueError("source_banks must be a nonempty array")
        source_banks = tuple(
            _portable_relative_path("source bank", item) for item in source_banks_value
        )
        if len(set(source_banks)) != len(source_banks):
            raise ExecutionQueueError("source_banks contains duplicates")

        ranking_policy = value["ranking_policy"]
        if not isinstance(ranking_policy, dict):
            raise ExecutionQueueError("ranking_policy must be an object")
        _strict_keys(ranking_policy, _RANKING_POLICY_KEYS, name="ranking_policy")
        principles = ranking_policy["principles"]
        if (
            not isinstance(principles, list)
            or not principles
            or any(not isinstance(item, str) or not item.strip() for item in principles)
        ):
            raise ExecutionQueueError("ranking_policy principles must be nonempty text")
        _required_text("ranking_policy tie_breaker", ranking_policy["tie_breaker"])

        proposal_sources: dict[int, tuple[dict[str, Any], str]] = {}
        for relative in source_banks:
            manifest_path = root / relative
            manifest = _load_object(manifest_path, name="proposal-bank manifest")
            if manifest.get("training_performed") is not False:
                raise ExecutionQueueError(f"source bank {relative} is not unmeasured")
            if manifest.get("experimental_status") != "queued_unmeasured":
                raise ExecutionQueueError(f"source bank {relative} has an invalid status")
            if manifest.get("frozen_parent_commit") != parent:
                raise ExecutionQueueError(f"source bank {relative} uses another frozen parent")
            proposals = manifest.get("proposals")
            if not isinstance(proposals, list) or manifest.get("proposal_count") != len(proposals):
                raise ExecutionQueueError(f"source bank {relative} has an invalid proposal count")
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    raise ExecutionQueueError(f"source bank {relative} contains a non-object")
                number = _positive_integer("proposal_number", proposal.get("proposal_number"))
                if number in proposal_sources:
                    raise ExecutionQueueError(f"proposal {number} appears in multiple source banks")
                if proposal.get("parent_commit") != parent:
                    raise ExecutionQueueError(f"proposal {number} uses another frozen parent")
                if proposal.get("training_performed") is not False:
                    raise ExecutionQueueError(f"proposal {number} is not unmeasured")
                if proposal.get("experimental_status") != "queued_unmeasured":
                    raise ExecutionQueueError(f"proposal {number} has an invalid status")
                for field in _PROPOSAL_FIELDS:
                    if field not in proposal:
                        raise ExecutionQueueError(f"proposal {number} is missing {field}")
                proposal_sources[number] = (proposal, relative)

        groups_value = value["conflict_groups"]
        if not isinstance(groups_value, list) or not groups_value:
            raise ExecutionQueueError("conflict_groups must be a nonempty array")
        groups = []
        group_ids: set[str] = set()
        expected_memberships: dict[int, set[str]] = {number: set() for number in proposal_sources}
        for index, group_value in enumerate(groups_value):
            if not isinstance(group_value, dict):
                raise ExecutionQueueError("conflict group must be an object")
            _strict_keys(group_value, _GROUP_KEYS, name=f"conflict_groups[{index}]")
            group_id = _required_text("group_id", group_value["group_id"], maximum=128)
            if not _ID_PATTERN.fullmatch(group_id) or group_id in group_ids:
                raise ExecutionQueueError(f"invalid or duplicate conflict group: {group_id}")
            group_ids.add(group_id)
            members_value = group_value["members"]
            if not isinstance(members_value, list) or len(members_value) < 2:
                raise ExecutionQueueError(f"conflict group {group_id} needs at least two members")
            members = tuple(_positive_integer("conflict member", item) for item in members_value)
            if len(set(members)) != len(members):
                raise ExecutionQueueError(f"conflict group {group_id} contains duplicate members")
            unknown = sorted(set(members) - set(proposal_sources))
            if unknown:
                raise ExecutionQueueError(
                    f"conflict group {group_id} has unknown members {unknown}"
                )
            try:
                severity = ConflictSeverity(group_value["severity"])
            except (TypeError, ValueError) as error:
                raise ExecutionQueueError(
                    f"conflict group {group_id} severity is invalid"
                ) from error
            group = ConflictGroup(
                group_id=group_id,
                label=_required_text("conflict label", group_value["label"]),
                severity=severity,
                rationale=_required_text("conflict rationale", group_value["rationale"]),
                members=members,
            )
            groups.append(group)
            for member in members:
                expected_memberships[member].add(group_id)

        items_value = value["items"]
        if not isinstance(items_value, list) or not items_value:
            raise ExecutionQueueError("items must be a nonempty array")
        items = []
        proposal_numbers: set[int] = set()
        for index, item_value in enumerate(items_value):
            if not isinstance(item_value, dict):
                raise ExecutionQueueError("queue item must be an object")
            _strict_keys(item_value, _ITEM_KEYS, name=f"items[{index}]")
            rank = _positive_integer("rank", item_value["rank"])
            if rank != index + 1:
                raise ExecutionQueueError("queue ranks must be contiguous and stored in order")
            number = _positive_integer("proposal_number", item_value["proposal_number"])
            if number in proposal_numbers:
                raise ExecutionQueueError(f"proposal {number} appears more than once in the queue")
            proposal_numbers.add(number)
            try:
                tier = QueueTier(item_value["tier"])
                adaptation_risk = QueueRisk(item_value["adaptation_risk"])
                resource_risk = QueueRisk(item_value["resource_risk"])
            except (TypeError, ValueError) as error:
                raise ExecutionQueueError(
                    f"queue item {rank} has an invalid tier or risk"
                ) from error
            memberships_value = item_value["conflict_groups"]
            if not isinstance(memberships_value, list) or any(
                not isinstance(item, str) for item in memberships_value
            ):
                raise ExecutionQueueError(f"queue item {rank} conflict_groups must be text")
            memberships = tuple(memberships_value)
            if len(set(memberships)) != len(memberships):
                raise ExecutionQueueError(f"queue item {rank} repeats a conflict group")
            if set(memberships) != expected_memberships.get(number, set()):
                raise ExecutionQueueError(
                    f"queue item {rank} conflict membership differs from conflict_groups"
                )
            try:
                proposal, expected_source = proposal_sources[number]
            except KeyError as error:
                raise ExecutionQueueError(
                    f"queue item {rank} references unknown proposal {number}"
                ) from error
            source_manifest = _portable_relative_path(
                "source_manifest", item_value["source_manifest"]
            )
            if source_manifest != expected_source:
                raise ExecutionQueueError(f"queue item {rank} references the wrong source manifest")
            patch_path = _portable_relative_path("patch_path", item_value["patch_path"])
            if patch_path != proposal.get("patch_path"):
                raise ExecutionQueueError(f"queue item {rank} patch path differs from its proposal")
            patch_sha256 = item_value["patch_sha256"]
            if patch_sha256 != proposal.get("diff_sha256"):
                raise ExecutionQueueError(f"queue item {rank} patch hash differs from its proposal")
            patch_file = root / patch_path
            if _sha256_file(patch_file) != patch_sha256:
                raise ExecutionQueueError(f"queue item {rank} patch content hash is invalid")
            items.append(
                ExecutionQueueItem(
                    rank=rank,
                    proposal_number=number,
                    tier=tier,
                    adaptation_risk=adaptation_risk,
                    resource_risk=resource_risk,
                    conflict_groups=memberships,
                    priority_reason=_required_text(
                        "priority_reason", item_value["priority_reason"]
                    ),
                    source_manifest=source_manifest,
                    patch_path=patch_path,
                    patch_sha256=patch_sha256,
                    proposal={field: proposal[field] for field in _PROPOSAL_FIELDS},
                )
            )

        missing = sorted(set(proposal_sources) - proposal_numbers)
        extra = sorted(proposal_numbers - set(proposal_sources))
        if missing or extra:
            raise ExecutionQueueError(
                f"queue coverage differs from source banks; missing={missing}, extra={extra}"
            )
        return cls(
            queue_id=queue_id,
            objective=objective,
            evidence_status="queued_unmeasured",
            frozen_parent_commit=parent,
            source_banks=source_banks,
            ranking_policy=ranking_policy,
            conflict_groups=tuple(groups),
            items=tuple(items),
            path=path,
            repository_root=root,
        )

    def item_for_rank(self, rank: int) -> ExecutionQueueItem:
        if type(rank) is not int or not 1 <= rank <= len(self.items):
            raise ExecutionQueueError(
                f"execution queue rank {rank!r} is outside 1..{len(self.items)}"
            )
        return self.items[rank - 1]

    def assignment(self, rank: int, *, current_parent_commit: str) -> dict[str, Any]:
        if not _COMMIT_PATTERN.fullmatch(current_parent_commit):
            raise ExecutionQueueError("current parent must be a full lowercase Git commit")
        item = self.item_for_rank(rank)
        groups = {group.group_id: group for group in self.conflict_groups}
        patch = (self.repository_root / item.patch_path).read_text(encoding="utf-8")
        return {
            "adaptation_required": current_parent_commit != self.frozen_parent_commit,
            "adaptation_risk": item.adaptation_risk.value,
            "conflict_groups": [
                {
                    "group_id": groups[group_id].group_id,
                    "label": groups[group_id].label,
                    "severity": groups[group_id].severity.value,
                }
                for group_id in item.conflict_groups
            ],
            "current_parent_commit": current_parent_commit,
            "evidence_status": self.evidence_status,
            "patch": patch,
            "patch_sha256": item.patch_sha256,
            "priority_reason": item.priority_reason,
            "proposal": item.proposal,
            "proposal_number": item.proposal_number,
            "queue_id": self.queue_id,
            "queue_rank": item.rank,
            "resource_risk": item.resource_risk.value,
            "source_parent_commit": self.frozen_parent_commit,
            "tier": item.tier.value,
        }

    def assignment_text(self, rank: int, *, current_parent_commit: str) -> str:
        assignment = self.assignment(rank, current_parent_commit=current_parent_commit)
        return (
            "\n\n## Protected execution queue assignment\n\n"
            "This is an unmeasured queued hypothesis, not a known improvement. Work only on this "
            "assigned hypothesis; do not substitute another idea. Apply the stored patch when it "
            "fits the current parent. When the current parent differs or its code has evolved, "
            "adapt only the same atomic mechanism and preserve the declared causal claim. Return "
            "`no_change` with a precise reason if the hypothesis is obsolete or cannot be adapted "
            "without bundling another change. Never treat queue rank or expected effect as "
            "evidence.\n\n"
            f"<!-- {QUEUE_ASSIGNMENT_MARKER}:START -->\n"
            "```json\n"
            + json.dumps(assignment, indent=2, sort_keys=True, allow_nan=False)
            + "\n```\n"
            f"<!-- {QUEUE_ASSIGNMENT_MARKER}:END -->\n"
        )

    def summary(self) -> dict[str, Any]:
        tier_counts = {tier.value: 0 for tier in QueueTier}
        for item in self.items:
            tier_counts[item.tier.value] += 1
        return {
            "conflict_group_count": len(self.conflict_groups),
            "evidence_status": self.evidence_status,
            "frozen_parent_commit": self.frozen_parent_commit,
            "item_count": len(self.items),
            "queue_id": self.queue_id,
            "queue_sha256": _sha256_file(self.path),
            "source_banks": list(self.source_banks),
            "tier_counts": tier_counts,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and inspect a proposal execution queue.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--queue", type=Path, default=DEFAULT_EXECUTION_QUEUE_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    show = commands.add_parser("show")
    show.add_argument("--rank", type=int, required=True)
    show.add_argument("--current-parent")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.repository_root.expanduser().resolve()
        queue_path = args.queue if args.queue.is_absolute() else root / args.queue
        queue = ExecutionQueue.from_path(queue_path, repository_root=root)
        if args.command == "verify":
            payload = queue.summary()
        else:
            parent = args.current_parent or queue.frozen_parent_commit
            payload = queue.assignment(args.rank, current_parent_commit=parent)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ExecutionQueueError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
