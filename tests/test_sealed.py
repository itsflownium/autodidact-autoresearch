from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from autodidact.checkpoints import file_sha256
from autodidact.data.integrity import canonical_json_bytes
from autodidact.ledger import ExperimentLedger, LedgerError, WriterRole
from autodidact.runner import ProcessOutcome
from autodidact.sealed import (
    SealedError,
    _execute_run,
    _load_retained_result,
    _run_contract,
    _run_root,
    _run_specs,
    _single_worktree,
    _verify_frozen_inputs,
    _write_canonical,
    build_report,
    create_plan,
    load_plan,
    load_results,
    render_csv,
    render_markdown,
    render_svg,
    sealed_status,
    write_report,
)
from tests.experiment_fixtures import evidence_records


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
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "train.py").write_text("MODEL = 'parent'\n", encoding="utf-8")
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add parent")
    parent = _git(repository, "rev-parse", "HEAD")
    (repository / "train.py").write_text("MODEL = 'candidate'\n", encoding="utf-8")
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add candidate")
    return repository, parent, _git(repository, "rev-parse", "HEAD")


def _ledger(path: Path, parent: str, candidate_commit: str) -> ExperimentLedger:
    records = evidence_records()
    proposal = replace(records["proposal"], parent_commit=parent)
    candidate = replace(
        records["candidate"],
        parent_commit=parent,
        candidate_commit=candidate_commit,
    )
    trial = replace(
        records["trial"],
        parent_commit=parent,
        candidate_commit=candidate_commit,
    )
    decision = replace(records["decision"], resulting_parent_commit=candidate_commit)
    lineage = replace(
        records["lineage"],
        parent_commit=parent,
        candidate_commit=candidate_commit,
    )
    ledger = ExperimentLedger.create(path, initial_parent_commit=parent)
    ledger.append_many(
        (
            (proposal, WriterRole.RESEARCH_AGENT),
            (candidate, WriterRole.CONTROLLER),
            (trial, WriterRole.CONTROLLER),
            (records["parent_run"], WriterRole.EVALUATOR),
            (records["parent_manifest"], WriterRole.EVALUATOR),
            (records["parent_compute"], WriterRole.EVALUATOR),
            (records["candidate_run"], WriterRole.EVALUATOR),
            (records["candidate_manifest"], WriterRole.EVALUATOR),
            (records["candidate_compute"], WriterRole.EVALUATOR),
            (records["paired"], WriterRole.EVALUATOR),
            (records["effect"], WriterRole.EVALUATOR),
            (records["prediction"], WriterRole.EVALUATOR),
            (decision, WriterRole.CONTROLLER),
            (lineage, WriterRole.CONTROLLER),
        )
    )
    return ledger


def _plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, arm_count: int = 1):
    repository, parent, candidate = _repository(tmp_path)
    data_root = tmp_path / "data"
    (data_root / "public").mkdir(parents=True)
    (data_root / "public" / "manifest.json").write_text("{}\n", encoding="utf-8")
    scopes: list[str] = []

    def verify(_root: Path, *, scope: str):
        scopes.append(scope)
        return {}

    monkeypatch.setattr("autodidact.sealed.verify_dataset", verify)
    ledgers = []
    for index in range(arm_count):
        path = tmp_path / f"ledger-{index}.sqlite3"
        _ledger(path, parent, candidate)
        ledgers.append((f"arm-{index}", path))
    root = tmp_path / "sealed"
    plan = create_plan(
        repository_root=repository,
        sealed_root=root,
        arm_ledgers=tuple(ledgers),
        seeds=(11, 23, 37),
        assignment_seed=101,
        token_budget=20_000_000,
        batch_size=64,
        eval_batch_size=64,
        timeout_seconds=7_200,
        data_root=data_root,
        device="cpu",
        parameter_cap=1_050_000,
    )
    return repository, root, plan, scopes


def _retain_results(root: Path, plan, *, candidate_bpb: float) -> None:
    for commit, seed in _run_specs(plan):
        run_root = _run_root(root, commit, seed)
        run_root.mkdir(parents=True, exist_ok=True)
        checkpoint = run_root / "checkpoint.pt"
        checkpoint.write_bytes(f"{commit}:{seed}".encode())
        bpb = 1.0 if commit == plan.initial_parent_commit else candidate_bpb
        trainer_sha256 = "a" * 64
        result = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "contract": _run_contract(plan, commit, seed, trainer_sha256),
            "evaluation": {
                "checkpoint_sha256": file_sha256(checkpoint),
                "event": "protected_evaluation",
                "parameter_count": 1_016_960,
                "split": "sealed_final",
                "trainer_sha256": trainer_sha256,
                "validation_bpb": bpb,
            },
            "inspection": {
                "event": "protected_inspection",
                "parameter_count": 1_016_960,
                "trainer_sha256": trainer_sha256,
            },
            "processes": {
                phase: {
                    "cancelled": False,
                    "peak_process_rss_bytes": 100,
                    "returncode": 0,
                    "timed_out": False,
                    "wall_seconds": 1.0,
                }
                for phase in ("evaluation", "inspection", "training")
            },
            "training_summary": {
                "event": "summary",
                "parameter_count": 1_016_960,
                "tokens_seen": plan.token_budget,
            },
        }
        _write_canonical(run_root / "result.json", result)
        (run_root / "result.sha256").write_text(
            hashlib.sha256(canonical_json_bytes(result)).hexdigest() + "\n",
            encoding="ascii",
        )


def test_plan_freezes_lineages_before_opening_protected_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, root, plan, scopes = _plan(tmp_path, monkeypatch, arm_count=2)

    assert scopes == ["public"]
    assert load_plan(root) == plan
    assert all(len(arm.generations) == 2 for arm in plan.arms)
    assert plan.arms[0].generations[0].commit == plan.initial_parent_commit
    assert len(_run_specs(plan)) == 6
    _verify_frozen_inputs(repository, plan)

    with pytest.raises(SealedError, match="at least two unique seeds"):
        replace(plan, seeds=(11,))


def test_frozen_commit_uses_a_detached_clean_worktree(tmp_path: Path) -> None:
    repository, parent, _candidate = _repository(tmp_path)

    with _single_worktree(repository, tmp_path / "worktrees", parent) as worktree:
        assert (worktree / "train.py").read_text(encoding="utf-8") == "MODEL = 'parent'\n"
        assert _git(worktree, "rev-parse", "HEAD") == parent

    assert not list((tmp_path / "worktrees").iterdir())


def test_sealed_executor_retains_verified_result_without_training_real_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, root, plan, _scopes = _plan(tmp_path, monkeypatch)
    commit = plan.arms[0].generations[-1].commit

    def fake_process(
        command,
        *,
        cwd,
        environment,
        stdout_path,
        stderr_path,
        timeout_seconds,
    ):
        del cwd, environment, timeout_seconds
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text("", encoding="utf-8")
        if "inspect" in command:
            trainer = Path(command[command.index("--trainer") + 1])
            stdout_path.write_text(
                json.dumps(
                    {
                        "event": "protected_inspection",
                        "parameter_count": 1_016_960,
                        "trainer_sha256": file_sha256(trainer),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        elif "train" in command:
            checkpoint = Path(command[command.index("--checkpoint-out") + 1])
            checkpoint.write_bytes(b"checkpoint")
            metrics = Path(command[command.index("--metrics-file") + 1])
            metrics.write_text(
                json.dumps(
                    {
                        "event": "summary",
                        "parameter_count": 1_016_960,
                        "tokens_seen": plan.token_budget,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stdout_path.write_text("", encoding="utf-8")
        else:
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            trainer = Path(command[command.index("--trainer") + 1])
            stdout_path.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "event": "protected_evaluation",
                        "parameter_count": 1_016_960,
                        "split": "sealed_final",
                        "trainer_sha256": file_sha256(trainer),
                        "validation_bpb": 0.99,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return ProcessOutcome(
            returncode=0,
            wall_seconds=1.0,
            peak_process_rss_bytes=100,
        )

    monkeypatch.setattr("autodidact.sealed.run_process", fake_process)

    result = _execute_run(
        plan,
        sealed_root=root,
        repository=repository,
        public_data_root=tmp_path / "public-data",
        commit=commit,
        seed=11,
    )

    assert result["evaluation"]["validation_bpb"] == 0.99
    assert _load_retained_result(_run_root(root, commit, 11)) == result


def test_plan_and_frozen_ledger_tampering_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, root, plan, _scopes = _plan(tmp_path, monkeypatch)
    plan_path = root / "plan.json"
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    value["assignment_seed"] += 1
    plan_path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SealedError, match="digest is invalid"):
        load_plan(root)

    _write_canonical(plan_path, plan.to_mapping())
    (root / "plan.sha256").write_text(
        hashlib.sha256(plan_path.read_bytes().rstrip(b"\n")).hexdigest() + "\n",
        encoding="ascii",
    )
    ledger_path = Path(plan.arms[0].ledger_path)
    with sqlite3.connect(ledger_path) as connection:
        triggers = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
        for (trigger,) in triggers:
            connection.execute(f'DROP TRIGGER "{trigger}"')
        connection.execute("UPDATE events SET event_sha256 = ? WHERE sequence = 1", ("0" * 64,))
        connection.commit()
    with pytest.raises(LedgerError):
        _verify_frozen_inputs(repository, plan)


def test_report_confirms_useful_promotions_and_writes_hashed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository_root, root, plan, _scopes = _plan(tmp_path, monkeypatch)
    _retain_results(root, plan, candidate_bpb=0.995)

    results = load_results(root, plan)
    report = build_report(plan, results)

    transition = report["arms"]["arm-0"]["transitions"][0]
    assert transition["classification"] == "useful_confirmed"
    assert transition["gain_bpb"]["mean"] == pytest.approx(0.005)
    assert report["summary"]["false_promotion_count"] == 0
    assert "Useful confirmed" in render_markdown(report)
    assert "useful_confirmed" in render_csv(report)
    assert "Sealed BPB" in render_svg(report)
    payload = write_report(root)
    assert set(payload["manifest"]["artifacts"]) == {
        "promotions.csv",
        "report.json",
        "report.md",
        "sealed-results.svg",
    }
    assert sealed_status(root)["report_ready"] is True


def test_report_marks_nonpositive_sealed_gain_as_false_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository_root, root, plan, _scopes = _plan(tmp_path, monkeypatch)
    _retain_results(root, plan, candidate_bpb=1.002)

    report = build_report(plan, load_results(root, plan))

    transition = report["arms"]["arm-0"]["transitions"][0]
    assert transition["classification"] == "false_promotion"
    assert report["summary"]["false_promotion_rate"] == 1.0
    assert _load_retained_result(_run_root(root, plan.initial_parent_commit, 11)) is not None
