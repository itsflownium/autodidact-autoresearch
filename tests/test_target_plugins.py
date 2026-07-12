from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from autodidact.checkpoints import file_sha256
from autodidact.ledger import ExperimentLedger, WriterRole
from autodidact.records import (
    ExperimentStage,
    PatchProposal,
    ResourceLimits,
    RunResult,
)
from autodidact.runner import (
    ExperimentRequest,
    PairedExperimentRunner,
    ProcessOutcome,
    RunnerError,
    validate_candidate_patch,
)
from autodidact.target import TargetConfig
from autodidact.target import main as target_main
from autodidact.target_plugins import TargetPluginError, TargetPluginSpec

DATA_HASH = "1" * 64
TOKENIZER_HASH = "2" * 64


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
                "--maximum-units",
                "{eval_tokens}",
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
                "--target-units",
                "{token_budget}",
                "--checkpoint",
                "{checkpoint}",
                "--metrics",
                "{metrics}",
            ],
        },
        "data_config_sha256": DATA_HASH,
        "editable_paths": ["model/train.py", "model/layers.py"],
        "evaluator_path": "control/evaluate.py",
        "metric": {
            "direction": "higher",
            "name": "validation_accuracy",
            "objective_offset": 1.0,
            "objective_scale": 1.0,
        },
        "plugin_id": "example.accuracy-target",
        "plugin_version": "1.0.0",
        "schema_version": 1,
        "tokenizer_sha256": TOKENIZER_HASH,
        "trainer_path": "model/train.py",
    }


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_plugin_contract_is_strict_shell_free_and_normalizes_higher_metrics() -> None:
    plugin = TargetPluginSpec.from_mapping(_plugin_mapping())

    assert plugin.metric.canonical_objective(0.8) == pytest.approx(0.2)
    assert plugin.editable_paths == ("model/train.py", "model/layers.py")
    assert len(plugin.contract_sha256()) == 64

    unsafe = _plugin_mapping()
    unsafe["commands"] = dict(unsafe["commands"], train=["sh", "-c", "cat {data_root}"])
    with pytest.raises(TargetPluginError, match="unsupported placeholder: data_root"):
        TargetPluginSpec.from_mapping(unsafe)

    editable_evaluator = _plugin_mapping()
    editable_evaluator["editable_paths"] = ["model/train.py", "control/evaluate.py"]
    with pytest.raises(TargetPluginError, match="protected evaluator"):
        TargetPluginSpec.from_mapping(editable_evaluator)


def test_external_plugins_can_raise_the_built_in_parameter_cap(tmp_path: Path) -> None:
    config = TargetConfig(
        name="larger-user-target",
        data_root=tmp_path / "protected",
        public_data_root=tmp_path / "public",
        plugin_spec_path=tmp_path / "plugin.json",
        trainer_path="model/train.py",
        max_parameter_count=50_000_000,
    )

    assert config.max_parameter_count == 50_000_000


def test_target_doctor_invokes_external_inspection_without_training(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "doctor-repository"
    (repository / "model").mkdir(parents=True)
    (repository / "control").mkdir()
    trainer = repository / "model" / "train.py"
    trainer.write_text("MODEL = 'doctor'\n", encoding="utf-8")
    evaluator = repository / "control" / "evaluate.py"
    evaluator.write_text(
        """import argparse, hashlib, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("operation")
parser.add_argument("--trainer")
parser.add_argument("--parameter-cap")
args = parser.parse_args()
data = pathlib.Path(args.trainer).read_bytes()
print(json.dumps({"event": "target_inspection", "parameter_count": 1500000,
                  "trainer_sha256": hashlib.sha256(data).hexdigest()}))
""",
        encoding="utf-8",
    )
    plugin_path = repository / "control" / "target-plugin.json"
    plugin_path.write_text(json.dumps(_plugin_mapping()), encoding="utf-8")
    public_root = tmp_path / "doctor-public"
    protected_root = tmp_path / "doctor-protected"
    public_root.mkdir()
    protected_root.mkdir()
    config_path = tmp_path / "doctor-target.json"
    config_path.write_text(
        json.dumps(
            TargetConfig(
                name="doctor-target",
                data_root=protected_root,
                public_data_root=public_root,
                plugin_spec_path=plugin_path,
                trainer_path="model/train.py",
                max_parameter_count=2_000_000,
                device="cpu",
            ).to_mapping()
        ),
        encoding="utf-8",
    )

    assert (
        target_main(
            [
                "doctor",
                "--config",
                str(config_path),
                "--repository-root",
                str(repository),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["inspection"]["parameter_count"] == 1_500_000
    assert payload["plugin"]["plugin_id"] == "example.accuracy-target"


def _repository(tmp_path: Path) -> tuple[Path, str, str, Path]:
    repository = tmp_path / "repository"
    (repository / "model").mkdir(parents=True)
    (repository / "control").mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "model" / "train.py").write_text("MODEL = 'example'\n", encoding="utf-8")
    (repository / "model" / "layers.py").write_text("WIDTH = 8\n", encoding="utf-8")
    (repository / "control" / "evaluate.py").write_text(
        "# protected evaluator adapter\n", encoding="utf-8"
    )
    plugin_path = repository / "control" / "target-plugin.json"
    plugin_path.write_text(json.dumps(_plugin_mapping()), encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "Add target")
    parent = _git(repository, "rev-parse", "HEAD")
    (repository / "model" / "layers.py").write_text(
        "WIDTH = 8\nIMPROVED = True\n", encoding="utf-8"
    )
    _git(repository, "add", "model/layers.py")
    _git(repository, "commit", "-m", "Improve target")
    candidate = _git(repository, "rev-parse", "HEAD")
    return repository, parent, candidate, plugin_path


class ExternalTargetProcesses:
    def __init__(self, *, public_root: Path, protected_root: Path) -> None:
        self.public_root = public_root.resolve()
        self.protected_root = protected_root.resolve()
        self.training_commands: list[list[str]] = []

    def __call__(
        self,
        command: list[str] | tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
        timeout_seconds: int,
    ) -> ProcessOutcome:
        del environment, timeout_seconds
        command = list(command)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        operation = command[2]
        trainer = (
            Path(_argument(command, "--trainer")) if "--trainer" in command else Path(command[1])
        )
        if operation == "inspect":
            stdout_path.write_text(
                json.dumps(
                    {
                        "event": "target_inspection",
                        "parameter_count": 1_500_000,
                        "trainer_sha256": file_sha256(trainer),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return ProcessOutcome(0, 0.1, 10_000_000)
        if operation == "train":
            self.training_commands.append(command)
            assert Path(_argument(command, "--data-root")).resolve() == self.public_root
            assert str(self.protected_root) not in command
            seed = int(_argument(command, "--seed"))
            budget = int(_argument(command, "--target-units"))
            checkpoint = Path(_argument(command, "--checkpoint"))
            metrics = Path(_argument(command, "--metrics"))
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"checkpoint-{seed}".encode())
            common = {
                "data_config_sha256": DATA_HASH,
                "parameter_count": 1_500_000,
                "seed": seed,
                "target_units": budget,
                "tokenizer_sha256": TOKENIZER_HASH,
            }
            events = [
                {"event": "target_training_config", **common},
                {
                    "event": "target_training_summary",
                    **common,
                    "data_order_sha256": hashlib.sha256(f"order-{seed}".encode()).hexdigest(),
                    "mean_train_loss": 0.4,
                    "units_seen": budget,
                },
            ]
            metrics.write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            return ProcessOutcome(0, 1.0, 20_000_000)
        if operation == "evaluate":
            assert Path(_argument(command, "--data-root")).resolve() == self.protected_root
            checkpoint = Path(_argument(command, "--checkpoint"))
            improved = "IMPROVED" in (cwd / "model" / "layers.py").read_text(encoding="utf-8")
            units = int(_argument(command, "--maximum-units"))
            stdout_path.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "evaluation_seconds": 0.25,
                        "evaluation_units": units,
                        "evaluation_units_per_second": units / 0.25,
                        "event": "target_evaluation",
                        "metric_direction": "higher",
                        "metric_name": "validation_accuracy",
                        "metric_value": 0.8 if improved else 0.7,
                        "parameter_count": 1_500_000,
                        "peak_process_rss_bytes": 12_000_000,
                        "trainer_sha256": file_sha256(trainer),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return ProcessOutcome(0, 0.25, 12_000_000)
        raise AssertionError(f"unexpected target command: {command}")


def test_external_target_runs_paired_evidence_without_exposing_protected_data(
    tmp_path: Path,
) -> None:
    repository, parent, candidate, plugin_path = _repository(tmp_path)
    public_root = tmp_path / "public-data"
    protected_root = tmp_path / "protected-data"
    public_root.mkdir()
    protected_root.mkdir()
    target_path = tmp_path / "target.json"
    target = TargetConfig(
        name="accuracy-target",
        data_root=protected_root,
        public_data_root=public_root,
        plugin_spec_path=plugin_path,
        trainer_path="model/train.py",
        max_parameter_count=2_000_000,
        device="cpu",
    )
    target_path.write_text(json.dumps(target.to_mapping()), encoding="utf-8")
    ledger_path = tmp_path / "ledger.sqlite3"
    ledger = ExperimentLedger.create(ledger_path, initial_parent_commit=parent)
    ledger.append(
        PatchProposal(
            proposal_id="proposal-plugin-001",
            parent_commit=parent,
            title="Improve layers",
            hypothesis="The change improves protected accuracy.",
            mechanism="Exercise the external target contract.",
            change="Change an allowed model component.",
            expected_effect_bpb=0.1,
            minimum_useful_gain_bpb=0.01,
            resource_risk="No expected resource change.",
            failure_signal="Accuracy does not improve.",
            interaction_risk="No known interaction.",
        ),
        writer_role=WriterRole.RESEARCH_AGENT,
    )
    request = ExperimentRequest(
        repository_root=repository,
        ledger_path=ledger_path,
        data_root=protected_root,
        output_root=tmp_path / "experiments",
        proposal_id="proposal-plugin-001",
        candidate_commit=candidate,
        stage=ExperimentStage.CHEAP,
        seeds=(7,),
        assignment_seed=11,
        token_budget=64,
        eval_tokens=32,
        batch_size=2,
        eval_batch_size=2,
        timeout_seconds=30,
        device="cpu",
        limits=ResourceLimits(timeout_seconds=30, max_parameter_count=2_000_000),
        target_config_path=target_path,
    )
    processes = ExternalTargetProcesses(
        public_root=public_root,
        protected_root=protected_root,
    )

    result = PairedExperimentRunner(request, process_runner=processes).run()

    assert result["runs"][0]["gain_bpb"] == pytest.approx(0.1)
    runs = [event.record for event in ledger.events() if isinstance(event.record, RunResult)]
    assert sorted(run.validation_bpb for run in runs) == pytest.approx([0.2, 0.3])
    assert len(processes.training_commands) == 2
    validation = validate_candidate_patch(
        repository,
        parent_commit=parent,
        candidate_commit=candidate,
        allowed_paths=("model/train.py", "model/layers.py"),
        trainer_path="model/train.py",
    )
    assert validation.changed_paths == ("model/layers.py",)

    drifted = _plugin_mapping()
    drifted["plugin_version"] = "1.0.1"
    plugin_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(RunnerError, match="existing candidate record differs"):
        PairedExperimentRunner(request, process_runner=processes).register_candidate()
    plugin_path.write_text(json.dumps(_plugin_mapping()), encoding="utf-8")

    (repository / "control" / "evaluate.py").write_text("changed\n", encoding="utf-8")
    _git(repository, "add", "control/evaluate.py")
    _git(repository, "commit", "-m", "Try to change evaluator")
    protected_candidate = _git(repository, "rev-parse", "HEAD")
    with pytest.raises(RunnerError, match="protected path"):
        validate_candidate_patch(
            repository,
            parent_commit=candidate,
            candidate_commit=protected_candidate,
            allowed_paths=("model/train.py", "model/layers.py"),
            trainer_path="model/train.py",
        )
