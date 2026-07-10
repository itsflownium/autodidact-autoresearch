from __future__ import annotations

import json
from pathlib import Path

import torch

from train import EXPECTED_PARAMETER_COUNT, main


def _json_lines(output: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def test_inspect_command_reports_fixed_contract(capsys) -> None:
    assert main(["inspect"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert payload["tied_embeddings"] is True
    assert payload["model"]["context_length"] == 256
    assert payload["model"]["num_layers"] == 4
    assert payload["model"]["model_width"] == 128
    assert payload["model"]["num_heads"] == 4
    assert payload["model"]["mlp_width"] == 512
    assert payload["training_modes"]["cheap"]["target_tokens"] == 2_000_000
    assert payload["training_modes"]["intermediate"]["target_tokens"] == 6_000_000
    assert payload["training_modes"]["full"]["target_tokens"] == 20_000_000


def test_train_and_generate_commands_are_reproducible(
    baseline_dataset: Path,
    tmp_path: Path,
    capsys,
) -> None:
    first_checkpoint = tmp_path / "first.pt"
    first_metrics = tmp_path / "first.jsonl"
    arguments = [
        "train",
        "--mode",
        "cheap",
        "--data-root",
        str(baseline_dataset),
        "--device",
        "cpu",
        "--seed",
        "71",
        "--token-budget",
        "128",
        "--eval-tokens",
        "100",
        "--batch-size",
        "2",
        "--eval-batch-size",
        "2",
        "--log-every-tokens",
        "64",
        "--checkpoint-every-tokens",
        "64",
        "--checkpoint-out",
        str(first_checkpoint),
        "--metrics-file",
        str(first_metrics),
        "--generate-tokens",
        "3",
        "--temperature",
        "0",
    ]
    assert main(arguments) == 0
    records = _json_lines(capsys.readouterr().out)
    events = [record["event"] for record in records]

    assert events == [
        "config",
        "train",
        "checkpoint",
        "evaluation",
        "generation",
        "summary",
    ]
    assert records[-1]["tokens_seen"] == records[-1]["target_tokens"] == 128
    assert records[-1]["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert records[-1]["validation_bpb"] > 0
    assert records[-1]["training_seconds"] > 0
    assert records[-1]["training_tokens_per_second"] > 0
    assert records[-1]["training_tokens_this_process"] == 128
    assert records[-1]["evaluation_seconds"] > 0
    assert records[-1]["evaluation_tokens_per_second"] > 0
    assert records[-1]["peak_process_rss_bytes"] > 0
    assert len(records[-1]["data_order_sha256"]) == 64
    assert len(records[-1]["checkpoint_state_sha256"]) == 64
    assert first_checkpoint.is_file()
    assert _json_lines(first_metrics.read_text(encoding="utf-8")) == records

    assert (
        main(
            [
                "generate",
                "--checkpoint",
                str(first_checkpoint),
                "--data-root",
                str(baseline_dataset),
                "--device",
                "cpu",
                "--seed",
                "71",
                "--prompt",
                "Once upon a time",
                "--generate-tokens",
                "3",
                "--temperature",
                "0",
            ]
        )
        == 0
    )
    generation = json.loads(capsys.readouterr().out)
    assert generation["event"] == "generation"
    assert generation["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert generation["text"]

    second_checkpoint = tmp_path / "second.pt"
    second_metrics = tmp_path / "second.jsonl"
    repeated_arguments = list(arguments)
    repeated_arguments[repeated_arguments.index(str(first_checkpoint))] = str(second_checkpoint)
    repeated_arguments[repeated_arguments.index(str(first_metrics))] = str(second_metrics)
    assert main(repeated_arguments) == 0
    repeated_records = _json_lines(capsys.readouterr().out)

    first_state = torch.load(first_checkpoint, map_location="cpu", weights_only=False)
    second_state = torch.load(second_checkpoint, map_location="cpu", weights_only=False)
    assert first_state["training"] == second_state["training"]
    for name, tensor in first_state["model_state"].items():
        torch.testing.assert_close(tensor, second_state["model_state"][name], rtol=0, atol=0)
    assert records[-1]["data_order_sha256"] == repeated_records[-1]["data_order_sha256"]
    assert records[-1]["checkpoint_state_sha256"] == repeated_records[-1]["checkpoint_state_sha256"]
    assert records[-1]["validation_bpb"] == repeated_records[-1]["validation_bpb"]

    resume_metrics = tmp_path / "resume.jsonl"
    resume_arguments = list(arguments)
    resume_arguments[resume_arguments.index(str(first_metrics))] = str(resume_metrics)
    resume_arguments.extend(["--resume", str(first_checkpoint)])
    assert main(resume_arguments) == 0
    resumed_records = _json_lines(capsys.readouterr().out)
    resumed_summary = resumed_records[-1]
    assert resumed_summary["tokens_seen"] == 128
    assert resumed_summary["training_tokens_this_process"] == 0
    assert resumed_summary["training_tokens_per_second"] == 0
