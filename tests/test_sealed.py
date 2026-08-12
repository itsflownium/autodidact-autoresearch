from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from autodidact.checkpoints import file_sha256
from autodidact.integrity import canonical_json_bytes
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
from autodidact.target import TargetConfig
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


def _plugin_mapping() -> dict[str, object]:
    return {
        "commands": {
            "evaluate": [
                "{python}",
                "{evaluator}",
                "evaluate",
                "--trainer",
                "{trainer}",
                "--checkpoint",
                "{checkpoint}",
                "--data-root",
                "{data_root}",
            ],
            "inspect": [
                "{python}",
                "{evaluator}",
                "inspect",
                "--trainer",
                "{trainer}",
                "--parameter-cap",
                "{parameter_cap}",
            ],
            "train": [
                "{python}",
                "{trainer}",
                "train",
                "--data-root",
                "{public_data_root}",
                "--seed",
                "{seed}",
                "--rollouts",
                "{training_budget}",
                "--checkpoint",
                "{checkpoint}",
                "--metrics",
                "{metrics}",
            ],
        },
        "data_config_sha256": "1" * 64,
        "editable_paths": ["target/train.py", "target/algorithm.py"],
        "evaluator_path": "control/evaluate.py",
        "metric": {
            "direction": "higher",
            "name": "verified_reward",
            "objective_offset": 1.0,
            "objective_scale": 1.0,
        },
        "plugin_id": "test.sealed-rlvr",
        "plugin_version": "1",
        "rl": {
            "algorithm_paths": ["target/algorithm.py"],
            "budget_unit": "rollouts",
            "paradigm": "rlvr",
            "reward_maximum": 1.0,
            "reward_minimum": 0.0,
            "reward_source": "verifier",
            "schema_version": 1,
        },
        "schema_version": 2,
        "tokenizer_sha256": "2" * 64,
        "trainer_path": "target/train.py",
    }


def _repository(tmp_path: Path) -> tuple[Path, str, str, Path]:
    repository = tmp_path / "repository"
    (repository / "target").mkdir(parents=True)
    (repository / "control").mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "target" / "train.py").write_text("MODEL = 'external'\n", encoding="utf-8")
    (repository / "target" / "algorithm.py").write_text("ALGORITHM = 'initial'\n", encoding="utf-8")
    (repository / "control" / "evaluate.py").write_text(
        "# protected verifier and evaluator\n", encoding="utf-8"
    )
    plugin_path = repository / "control" / "target-plugin.json"
    plugin_path.write_text(json.dumps(_plugin_mapping()), encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "Add external RLVR target")
    parent = _git(repository, "rev-parse", "HEAD")
    (repository / "target" / "algorithm.py").write_text(
        "ALGORITHM = 'custom-advantage-v2'\n", encoding="utf-8"
    )
    _git(repository, "add", "target/algorithm.py")
    _git(repository, "commit", "-m", "Change the RL algorithm")
    return repository, parent, _git(repository, "rev-parse", "HEAD"), plugin_path


def _ledger(path: Path, parent: str, candidate_commit: str) -> ExperimentLedger:
    records = evidence_records()
    proposal = replace(records["proposal"], parent_commit=parent)
    candidate = replace(
        records["candidate"],
        parent_commit=parent,
        candidate_commit=candidate_commit,
        changed_paths=("target/algorithm.py",),
        parameter_count=2_000_000,
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


def _plan(tmp_path: Path, *, arm_count: int = 1):
    repository, parent, candidate, plugin_path = _repository(tmp_path)
    public_root = tmp_path / "public"
    protected_root = tmp_path / "protected"
    public_root.mkdir()
    protected_root.mkdir()
    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            TargetConfig(
                name="sealed RLVR target",
                data_root=protected_root,
                public_data_root=public_root,
                plugin_spec_path=plugin_path,
                trainer_path="target/train.py",
                max_parameter_count=5_000_000,
                device="cpu",
            ).to_mapping()
        ),
        encoding="utf-8",
    )
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
        token_budget=256,
        batch_size=8,
        eval_batch_size=8,
        timeout_seconds=30,
        target_config_path=target_path,
    )
    return repository, root, plan


def _retain_results(root: Path, plan, *, candidate_objective: float) -> None:
    for commit, seed in _run_specs(plan):
        run_root = _run_root(root, commit, seed)
        run_root.mkdir(parents=True, exist_ok=True)
        checkpoint = run_root / "checkpoint.pt"
        checkpoint.write_bytes(f"{commit}:{seed}".encode())
        objective = 0.5 if commit == plan.initial_parent_commit else candidate_objective
        trainer_sha256 = "a" * 64
        result = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "contract": _run_contract(plan, commit, seed, trainer_sha256),
            "evaluation": {
                "checkpoint_sha256": file_sha256(checkpoint),
                "event": "target_evaluation",
                "metric_direction": "higher",
                "metric_name": "verified_reward",
                "metric_value": 1.0 - objective,
                "objective_value": objective,
                "parameter_count": 2_000_000,
                "reward_source": "verifier",
                "reward_standard_deviation": 0.1,
                "trainer_sha256": trainer_sha256,
                "training_paradigm": "rlvr",
                "verifier_coverage": 1.0,
            },
            "inspection": {
                "event": "target_inspection",
                "parameter_count": 2_000_000,
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
                "algorithm_id": "custom-advantage-v2",
                "budget_unit": "rollouts",
                "event": "target_training_summary",
                "mean_train_reward": 0.5,
                "parameter_count": 2_000_000,
                "rollout_valid_fraction": 1.0,
                "train_reward_standard_deviation": 0.1,
                "training_paradigm": "rlvr",
                "units_seen": plan.token_budget,
            },
        }
        _write_canonical(run_root / "result.json", result)
        (run_root / "result.sha256").write_text(
            hashlib.sha256(canonical_json_bytes(result)).hexdigest() + "\n",
            encoding="ascii",
        )


def test_plan_freezes_plugin_lineages_before_sealed_evaluation(tmp_path: Path) -> None:
    repository, root, plan = _plan(tmp_path, arm_count=2)

    assert load_plan(root) == plan
    assert all(len(arm.generations) == 2 for arm in plan.arms)
    assert plan.plugin["rl"]["paradigm"] == "rlvr"
    assert plan.plugin["rl"]["algorithm_paths"] == ["target/algorithm.py"]
    assert len(_run_specs(plan)) == 6
    _verify_frozen_inputs(repository, plan)

    with pytest.raises(SealedError, match="at least two unique seeds"):
        replace(plan, seeds=(11,))


def test_frozen_commit_uses_a_detached_clean_worktree(tmp_path: Path) -> None:
    repository, parent, _candidate, _plugin = _repository(tmp_path)

    with _single_worktree(
        repository,
        tmp_path / "worktrees",
        parent,
        trainer_path="target/train.py",
    ) as worktree:
        algorithm = (worktree / "target" / "algorithm.py").read_text(encoding="utf-8")
        assert "initial" in algorithm
        assert _git(worktree, "rev-parse", "HEAD") == parent

    assert not list((tmp_path / "worktrees").iterdir())


def test_sealed_executor_accepts_custom_rl_algorithm_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, root, plan = _plan(tmp_path)
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
        operation = command[2]
        trainer = (
            Path(command[1])
            if operation == "train"
            else Path(command[command.index("--trainer") + 1])
        )
        if operation == "inspect":
            payload = {
                "event": "target_inspection",
                "parameter_count": 2_000_000,
                "trainer_sha256": file_sha256(trainer),
            }
            stdout_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        elif operation == "train":
            assert command[command.index("--rollouts") + 1] == str(plan.token_budget)
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            checkpoint.write_bytes(b"checkpoint")
            metrics = Path(command[command.index("--metrics") + 1])
            summary = {
                "algorithm_id": "custom-advantage-v2",
                "budget_unit": "rollouts",
                "event": "target_training_summary",
                "mean_train_reward": 0.55,
                "parameter_count": 2_000_000,
                "rollout_valid_fraction": 0.98,
                "train_reward_standard_deviation": 0.2,
                "training_paradigm": "rlvr",
                "units_seen": plan.token_budget,
            }
            metrics.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            stdout_path.write_text("", encoding="utf-8")
        else:
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            payload = {
                "checkpoint_sha256": file_sha256(checkpoint),
                "event": "target_evaluation",
                "metric_direction": "higher",
                "metric_name": "verified_reward",
                "metric_value": 0.61,
                "parameter_count": 2_000_000,
                "reward_source": "verifier",
                "reward_standard_deviation": 0.1,
                "trainer_sha256": file_sha256(trainer),
                "training_paradigm": "rlvr",
                "verifier_coverage": 1.0,
            }
            stdout_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return ProcessOutcome(returncode=0, wall_seconds=1.0, peak_process_rss_bytes=100)

    monkeypatch.setattr("autodidact.sealed.run_process", fake_process)
    result = _execute_run(
        plan,
        sealed_root=root,
        repository=repository,
        public_data_root=Path(plan.public_data_root),
        commit=commit,
        seed=11,
    )

    assert result["training_summary"]["algorithm_id"] == "custom-advantage-v2"
    assert result["evaluation"]["objective_value"] == pytest.approx(0.39)
    assert _load_retained_result(_run_root(root, commit, 11)) == result


def test_plan_and_frozen_ledger_tampering_fail_closed(tmp_path: Path) -> None:
    repository, root, plan = _plan(tmp_path)
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


def test_report_confirms_or_rejects_promotions_and_hashes_artifacts(tmp_path: Path) -> None:
    _repository_root, root, plan = _plan(tmp_path)
    _retain_results(root, plan, candidate_objective=0.495)

    report = build_report(plan, load_results(root, plan))
    transition = report["arms"]["arm-0"]["transitions"][0]
    assert transition["classification"] == "useful_confirmed"
    assert transition["objective_gain"]["mean"] == pytest.approx(0.005)
    assert "Useful confirmed" in render_markdown(report)
    assert "useful_confirmed" in render_csv(report)
    assert "Sealed objective" in render_svg(report)
    payload = write_report(root)
    assert set(payload["manifest"]["artifacts"]) == {
        "promotions.csv",
        "report.json",
        "report.md",
        "sealed-results.svg",
    }
    assert sealed_status(root)["report_ready"] is True

    false_root = tmp_path / "sealed-false"
    false_plan = replace(plan, plan_id="sealed-false-promotion")
    false_root.mkdir()
    _retain_results(false_root, false_plan, candidate_objective=0.502)
    false_report = build_report(false_plan, load_results(false_root, false_plan))
    assert false_report["arms"]["arm-0"]["transitions"][0]["classification"] == ("false_promotion")
