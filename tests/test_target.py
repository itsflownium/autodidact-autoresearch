from __future__ import annotations

import json
from pathlib import Path

import pytest

from autodidact.orchestrator import _config_from_args
from autodidact.orchestrator import build_parser as build_orchestrator_parser
from autodidact.target import ExecutionLocation, TargetConfig, TargetError, main


def test_target_config_round_trip_and_relative_data_resolution(tmp_path: Path) -> None:
    config = TargetConfig(
        name="test-transformer",
        data_root=Path("artifacts/data/test"),
        device="cuda",
        execution_location=ExecutionLocation.GPU_HOST,
        max_parameter_count=1_040_000,
        estimated_accelerator_hour_usd=3.25,
    )
    path = tmp_path / "target.json"
    path.write_text(json.dumps(config.to_mapping()), encoding="utf-8")

    restored = TargetConfig.from_path(path)

    assert restored == config
    assert restored.resolved_data_root(tmp_path) == tmp_path / "artifacts/data/test"


def test_target_schema_rejects_arbitrary_trainer_until_plugin_contract_exists() -> None:
    with pytest.raises(TargetError, match="requires trainer_path to be train.py"):
        TargetConfig(
            name="unsafe-shell-target",
            data_root=Path("data"),
            trainer_path="scripts/train.py",
        )

    with pytest.raises(TargetError, match="caps models at 1050000 parameters"):
        TargetConfig(
            name="oversized-target",
            data_root=Path("data"),
            max_parameter_count=1_050_001,
        )


def test_target_init_and_show_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "target.json"

    assert (
        main(
            [
                "init",
                "--name",
                "local-transformer",
                "--data-root",
                "prepared-data",
                "--device",
                "mps",
                "--config",
                str(path),
            ]
        )
        == 0
    )
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["name"] == "local-transformer"
    assert initialized["device"] == "mps"

    assert main(["show", "--config", str(path)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["schema_version"] == 1
    assert shown["trainer_path"] == "train.py"


def test_target_init_refuses_to_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "target.json"
    arguments = [
        "init",
        "--name",
        "target",
        "--data-root",
        "data",
        "--config",
        str(path),
    ]
    assert main(arguments) == 0
    capsys.readouterr()

    assert main(arguments) == 2
    assert "already exists" in capsys.readouterr().err


def test_orchestrator_loads_target_device_data_and_parameter_cap(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            TargetConfig(
                name="configured-target",
                data_root=Path("prepared-data"),
                device="cuda",
                max_parameter_count=1_040_000,
            ).to_mapping()
        ),
        encoding="utf-8",
    )
    repository = tmp_path / "repository"
    repository.mkdir()
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

    assert config.target_name == "configured-target"
    assert config.data_root == repository / "prepared-data"
    assert config.device == "cuda"
    assert config.max_parameter_count == 1_040_000


def test_target_doctor_inspects_model_without_training(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            TargetConfig(
                name="included-baseline",
                data_root=Path("unused-for-inspection"),
            ).to_mapping()
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "doctor",
                "--config",
                str(target_path),
                "--repository-root",
                str(Path.cwd()),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["inspection"]["parameter_count"] == 1_016_960
    assert payload["resolved_device"] in {"cpu", "mps", "cuda"}
