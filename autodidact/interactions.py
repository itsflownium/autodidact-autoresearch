"""Plan promoted-stack interaction audits without mutating accepted lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from autodidact.integrity import canonical_json_bytes
from autodidact.ledger import DEFAULT_LEDGER_PATH, ExperimentLedger, LedgerError
from autodidact.records import (
    CandidateRecord,
    DecisionRecord,
    EffectEstimate,
    LineageRecord,
    PatchProposal,
)


class InteractionAuditError(RuntimeError):
    """Raised when an interaction audit plan cannot be trusted."""


def _git(repository_root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise InteractionAuditError(message or f"git {arguments[0]} failed")
    return completed.stdout


def reverse_patch_applicability(
    repository_root: Path,
    *,
    patch_parent_commit: str,
    patch_candidate_commit: str,
    current_stack_commit: str,
) -> tuple[bool, str | None]:
    """Check whether a promoted patch can be removed from the current stack as one diff."""

    repository_root = repository_root.expanduser().resolve()
    patch = _git(
        repository_root,
        "diff",
        "--binary",
        patch_parent_commit,
        patch_candidate_commit,
    )
    temporary_root = Path(tempfile.mkdtemp(prefix="autodidact-interaction-"))
    worktree = temporary_root / "stack"
    try:
        _git(
            repository_root,
            "worktree",
            "add",
            "--detach",
            str(worktree),
            current_stack_commit,
        )
        completed = subprocess.run(
            ["git", "apply", "--reverse", "--check", "--whitespace=error-all", "-"],
            cwd=worktree,
            input=patch,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return True, None
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        return False, (message[:1_000] or "reverse patch does not apply cleanly")
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=repository_root,
            capture_output=True,
            check=False,
        )
        shutil.rmtree(temporary_root, ignore_errors=True)


def build_interaction_audit_plan(
    ledger: ExperimentLedger,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic leave-one-out plan from verified promotion evidence."""

    verification = ledger.verify()
    events = ledger.events()
    proposals = {
        record.proposal_id: record
        for event in events
        if isinstance((record := event.record), PatchProposal)
    }
    candidates = {
        record.candidate_id: record
        for event in events
        if isinstance((record := event.record), CandidateRecord)
    }
    decisions = {
        record.decision_id: record
        for event in events
        if isinstance((record := event.record), DecisionRecord)
    }
    effects = {
        record.estimate_id: record
        for event in events
        if isinstance((record := event.record), EffectEstimate)
    }
    lineages = [event.record for event in events if isinstance(event.record, LineageRecord)]
    current_stack = ledger.current_parent()
    audits = []
    for index, lineage in enumerate(lineages):
        candidate = candidates[lineage.candidate_id]
        proposal = proposals[candidate.proposal_id]
        decision = decisions[lineage.decision_id]
        effect = (
            None if decision.effect_estimate_id is None else effects[decision.effect_estimate_id]
        )
        later_promotions = len(lineages) - index - 1
        requires_leave_one_out = later_promotions > 0
        applicable = None
        applicability_error = None
        if repository_root is not None:
            applicable, applicability_error = reverse_patch_applicability(
                repository_root,
                patch_parent_commit=candidate.parent_commit,
                patch_candidate_commit=candidate.candidate_commit,
                current_stack_commit=current_stack,
            )
        audits.append(
            {
                "ablation_base_commit": current_stack,
                "candidate_id": candidate.candidate_id,
                "changed_paths": list(candidate.changed_paths),
                "generation": lineage.generation,
                "interaction_risk": proposal.interaction_risk,
                "later_promotion_count": later_promotions,
                "promoted_commit": candidate.candidate_commit,
                "promotion_effect_estimate_id": (None if effect is None else effect.estimate_id),
                "promotion_mean_objective_gain": (
                    None if effect is None else effect.mean_objective_gain
                ),
                "requires_leave_one_out": requires_leave_one_out,
                "reverse_patch_applicable": applicable,
                "reverse_patch_error": applicability_error,
                "reverse_patch_range": (f"{candidate.parent_commit}..{candidate.candidate_commit}"),
                "status": (
                    "leave_one_out_required"
                    if requires_leave_one_out
                    else "directly_tested_current_stack"
                ),
                "title": proposal.title,
            }
        )
    plan_payload = {
        "audits": audits,
        "current_stack_commit": current_stack,
        "ledger_head_sha256": verification.head_event_sha256,
        "schema_version": 1,
    }
    return {
        **plan_payload,
        "plan_id": "interaction-plan-"
        + hashlib.sha256(canonical_json_bytes(plan_payload)).hexdigest()[:24],
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan protected promoted-stack interaction audits."
    )
    parser.add_argument("--ledger-path", type=_path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--repository-root", type=_path)
    parser.add_argument("--output", type=_path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ledger = ExperimentLedger.open(args.ledger_path, read_only=True)
        plan = build_interaction_audit_plan(
            ledger,
            repository_root=args.repository_root,
        )
        content = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            _atomic_write(args.output.expanduser().resolve(), content)
        print(content, end="")
        return 0
    except (InteractionAuditError, LedgerError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
