from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from autodidact.data.reader import open_public_split
from autodidact.evaluator import EvaluationError, evaluate_checkpoint, inspect_trainer, main
from train import TransformerLM, count_parameters, evaluate_bpb

ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(path: Path) -> TransformerLM:
    model = TransformerLM()
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "parameter_count": count_parameters(model),
        },
        path,
    )
    return model


def test_protected_inspection_counts_parameters_without_trusting_train_cli() -> None:
    inspection = inspect_trainer(ROOT / "train.py")

    assert inspection["event"] == "protected_inspection"
    assert inspection["parameter_count"] == 1_016_960
    assert inspection["context_length"] == 256
    assert inspection["vocab_size"] == 1_792
    assert len(inspection["trainer_sha256"]) == 64


def test_protected_evaluation_matches_reference_bpb(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    model = _checkpoint(checkpoint)
    expected = evaluate_bpb(
        model,
        open_public_split(prepared_dataset, "dev"),
        device=torch.device("cpu"),
        maximum_tokens=400,
        batch_size=4,
    )

    protected = evaluate_checkpoint(
        ROOT / "train.py",
        checkpoint,
        prepared_dataset,
        split_name="dev",
        maximum_tokens=400,
        batch_size=4,
        device_name="cpu",
    )

    assert protected["event"] == "protected_evaluation"
    assert protected["validation_bpb"] == pytest.approx(expected["bpb"])
    assert protected["predicted_tokens"] == expected["predicted_tokens"]
    assert protected["parameter_count"] == 1_016_960
    assert protected["evaluation_seconds"] > 0


def test_candidate_evaluation_function_cannot_forge_protected_bpb(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "train.py"
    trainer.write_text(
        (ROOT / "train.py").read_text(encoding="utf-8")
        + "\n\ndef evaluate_bpb(*args, **kwargs):\n"
        + "    return {'bpb': 0.0, 'predicted_tokens': 1, 'utf8_bytes': 1}\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)

    protected = evaluate_checkpoint(
        trainer,
        checkpoint,
        prepared_dataset,
        split_name="dev",
        maximum_tokens=200,
        batch_size=4,
        device_name="cpu",
    )

    assert protected["validation_bpb"] > 0.0
    assert protected["predicted_tokens"] > 1


def test_inspection_rejects_models_above_the_controller_cap(tmp_path: Path) -> None:
    trainer = tmp_path / "train.py"
    trainer.write_text(
        """
from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    context_length: int = 8
    vocab_size: int = 16

class TransformerLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(20))
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="cap is 10"):
        inspect_trainer(trainer, parameter_cap=10)


def test_evaluator_cli_emits_machine_readable_inspection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["inspect", "--trainer", str(ROOT / "train.py")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event"] == "protected_inspection"
