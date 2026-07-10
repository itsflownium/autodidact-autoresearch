from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch

import train
from autodidact.data.reader import open_public_split
from train import (
    JsonlMetrics,
    TokenBatcher,
    TransformerLM,
    build_optimizer,
    evaluate_bpb,
    resolve_device,
    restore_training_checkpoint,
    save_checkpoint,
    seed_everything,
)


def test_token_batcher_is_seeded_and_restorable(prepared_dataset: Path) -> None:
    split = open_public_split(prepared_dataset, "train")
    first = TokenBatcher(split, seed=44)
    second = TokenBatcher(split, seed=44)

    first_inputs, first_targets = first.next_batch(
        4,
        32,
        device=torch.device("cpu"),
    )
    second_inputs, second_targets = second.next_batch(
        4,
        32,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(first_inputs, second_inputs, rtol=0, atol=0)
    torch.testing.assert_close(first_targets, second_targets, rtol=0, atol=0)

    saved_state = first.state_dict()
    expected_inputs, expected_targets = first.next_batch(
        3,
        17,
        device=torch.device("cpu"),
    )
    restored = TokenBatcher(split, seed=999)
    restored.load_state_dict(saved_state)
    actual_inputs, actual_targets = restored.next_batch(
        3,
        17,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(actual_inputs, expected_inputs, rtol=0, atol=0)
    torch.testing.assert_close(actual_targets, expected_targets, rtol=0, atol=0)


def test_checkpoint_restores_model_optimizer_rng_and_sampler(
    prepared_dataset: Path,
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")
    split = open_public_split(prepared_dataset, "train")
    seed_everything(17)
    model = TransformerLM()
    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
    assert isinstance(optimizer, torch.optim.AdamW)
    batcher = TokenBatcher(split, seed=17)
    inputs, targets = batcher.next_batch(2, 16, device=device)
    _logits, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()

    checkpoint = tmp_path / "checkpoint.pt"
    cumulative_loss = float(loss.item()) * inputs.numel()
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        batcher=batcher,
        device=device,
        mode="cheap",
        seed=17,
        step=1,
        tokens_seen=inputs.numel(),
        target_tokens=128,
        cumulative_loss=cumulative_loss,
        cumulative_loss_tokens=inputs.numel(),
    )
    expected_random = torch.rand(5)
    expected_inputs, expected_targets = batcher.next_batch(2, 16, device=device)

    seed_everything(999)
    restored_model = TransformerLM()
    restored_optimizer = build_optimizer(
        restored_model,
        learning_rate=1e-3,
        weight_decay=0.1,
    )
    restored_batcher = TokenBatcher(split, seed=999)
    restored_state = restore_training_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        batcher=restored_batcher,
        device=device,
        target_tokens=128,
    )

    assert restored_state == (1, inputs.numel(), 17, cumulative_loss, inputs.numel())
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(5), expected_random, rtol=0, atol=0)
    actual_inputs, actual_targets = restored_batcher.next_batch(2, 16, device=device)
    torch.testing.assert_close(actual_inputs, expected_inputs, rtol=0, atol=0)
    torch.testing.assert_close(actual_targets, expected_targets, rtol=0, atol=0)


def test_public_dev_evaluation_produces_finite_bpb(prepared_dataset: Path) -> None:
    seed_everything(5)
    model = TransformerLM()
    dev = open_public_split(prepared_dataset, "dev")
    metrics = evaluate_bpb(
        model,
        dev,
        device=torch.device("cpu"),
        maximum_tokens=400,
        batch_size=4,
    )

    assert metrics["bpb"] > 0
    assert metrics["negative_log_likelihood"] > 0
    assert 0 < metrics["predicted_tokens"] <= 400
    assert metrics["stories"] > 0
    assert metrics["utf8_bytes"] > 0


def test_jsonl_metrics_writes_identical_machine_readable_records(tmp_path: Path) -> None:
    stream = io.StringIO()
    path = tmp_path / "metrics.jsonl"
    with JsonlMetrics(path, stream=stream) as metrics:
        metrics.emit("train", loss=1.25, step=3, tokens_seen=512)

    stdout_record = json.loads(stream.getvalue())
    file_record = json.loads(path.read_text(encoding="utf-8"))
    assert (
        stdout_record
        == file_record
        == {
            "event": "train",
            "loss": 1.25,
            "schema_version": 1,
            "step": 3,
            "tokens_seen": 512,
        }
    )


def test_device_selection_supports_cpu_cuda_and_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_device("cpu") == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(train.TrainingError, match="CUDA"):
        resolve_device("cuda")

    monkeypatch.setattr(train, "_mps_available", lambda: True)
    assert resolve_device("mps") == torch.device("mps")
    monkeypatch.setattr(train, "_mps_available", lambda: False)
    with pytest.raises(train.TrainingError, match="MPS"):
        resolve_device("mps")
