from __future__ import annotations

import json
from pathlib import Path

import pytest

from autodidact.orchestrator import _config_from_args
from autodidact.orchestrator import build_parser as build_orchestrator_parser
from autodidact.target import ExecutionLocation, TargetConfig, TargetError, main


def _plugin_mapping(*, trainer_path: str = "target/train.py") -> dict[str, object]:
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
        "editable_paths": [trainer_path, "target/algorithm.py"],
        "evaluator_path": "control/evaluate.py",
        "metric": {
            "direction": "higher",
            "name": "verified_reward",
            "objective_offset": 1.0,
            "objective_scale": 1.0,
        },
        "plugin_id": "test.configurable-rlvr",
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
        "trainer_path": trainer_path,
    }


def _target_repository(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    (repository / "target").mkdir(parents=True)
    (repository / "control").mkdir()
    trainer = repository / "target" / "train.py"
    trainer.write_text("MODEL = 'user-supplied'\n", encoding="utf-8")
    (repository / "target" / "algorithm.py").write_text("ALGORITHM = 'custom'\n", encoding="utf-8")
    evaluator = repository / "control" / "evaluate.py"
    evaluator.write_text(
        """import argparse, hashlib, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("operation")
parser.add_argument("--trainer")
parser.add_argument("--parameter-cap")
args = parser.parse_args()
data = pathlib.Path(args.trainer).read_bytes()
print(json.dumps({"event": "target_inspection", "parameter_count": 2000000,
                  "trainer_sha256": hashlib.sha256(data).hexdigest()}))
""",
        encoding="utf-8",
    )
    plugin = repository / "control" / "target-plugin.json"
    plugin.write_text(json.dumps(_plugin_mapping()), encoding="utf-8")
    public = tmp_path / "public"
    protected = tmp_path / "protected"
    public.mkdir()
    protected.mkdir()
    return repository, plugin, public, protected


def test_target_config_round_trip_and_resolves_plugin_roots(tmp_path: Path) -> None:
    repository, plugin, public, protected = _target_repository(tmp_path)
    config = TargetConfig(
        name="custom policy target",
        data_root=protected,
        public_data_root=public,
        plugin_spec_path=plugin,
        trainer_path="target/train.py",
        device="cuda:0",
        execution_location=ExecutionLocation.GPU_HOST,
        max_parameter_count=70_000_000_000,
        estimated_accelerator_hour_usd=2.5,
    )
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config.to_mapping()), encoding="utf-8")

    restored = TargetConfig.from_path(path)

    assert restored == config
    assert restored.load_plugin(repository).rl is not None
    assert restored.resolved_data_root(repository) == protected
    assert restored.resolved_public_data_root(repository) == public


def test_target_requires_a_plugin_and_separate_public_data(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="plugin_spec_path"):
        TargetConfig(name="missing plugin", data_root=tmp_path)

    with pytest.raises(TargetError, match="public_data_root"):
        TargetConfig(
            name="missing public data",
            data_root=tmp_path,
            plugin_spec_path=tmp_path / "plugin.json",
        )

    repository, plugin, public, _protected = _target_repository(tmp_path / "separate")
    config = TargetConfig(
        name="same roots",
        data_root=public,
        public_data_root=public,
        plugin_spec_path=plugin,
        trainer_path="target/train.py",
    )
    with pytest.raises(TargetError, match="must be distinct"):
        config.load_plugin(repository)


def test_target_init_show_and_doctor_are_training_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository, plugin, public, protected = _target_repository(tmp_path)
    path = tmp_path / "target.json"
    arguments = [
        "init",
        "--name",
        "local-rlvr",
        "--data-root",
        str(protected),
        "--public-data-root",
        str(public),
        "--plugin-spec",
        str(plugin),
        "--trainer-path",
        "target/train.py",
        "--max-parameter-count",
        "5000000",
        "--device",
        "cpu",
        "--config",
        str(path),
    ]

    assert main(arguments) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["schema_version"] == 3
    assert main(["show", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["trainer_path"] == "target/train.py"
    assert (
        main(
            [
                "doctor",
                "--config",
                str(path),
                "--repository-root",
                str(repository),
            ]
        )
        == 0
    )
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["ready"] is True
    assert doctor["inspection"]["parameter_count"] == 2_000_000
    assert doctor["plugin"]["rl"]["algorithm_paths"] == ["target/algorithm.py"]

    assert main(arguments) == 2
    assert "already exists" in capsys.readouterr().err


def test_orchestrator_receives_agent_neutral_rl_contract(tmp_path: Path) -> None:
    repository, plugin, public, protected = _target_repository(tmp_path)
    target_path = repository / "target.json"
    target_path.write_text(
        json.dumps(
            TargetConfig(
                name="research target",
                data_root=protected,
                public_data_root=public,
                plugin_spec_path=plugin,
                trainer_path="target/train.py",
                max_parameter_count=5_000_000,
            ).to_mapping()
        ),
        encoding="utf-8",
    )
    args = build_orchestrator_parser().parse_args(
        [
            "--repository-root",
            str(repository),
            "--target-config",
            str(target_path),
            "status",
        ]
    )

    config = _config_from_args(args)
    summary = config.target_summary()

    assert config.allowed_paths == ("target/train.py", "target/algorithm.py")
    assert summary["rl"]["paradigm"] == "rlvr"
    assert summary["rl"]["algorithm_paths"] == ["target/algorithm.py"]
    assert "algorithm" not in summary["rl"]
    researcher_summary = config.researcher_target_summary()
    assert "data_root" not in researcher_summary
    assert researcher_summary["protected_evaluation"] == "configured"
