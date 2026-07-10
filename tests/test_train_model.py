from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from train import (
    EXPECTED_PARAMETER_COUNT,
    MAX_PARAMETER_COUNT,
    TRAINING_MODES,
    ModelConfig,
    RotaryEmbedding,
    TrainingError,
    TransformerLM,
    WeightOnlyLayerNorm,
    batch_shape_for_remaining,
    cosine_learning_rate,
    count_parameters,
    enforce_parameter_count,
    generate_token_ids,
    seed_everything,
)


def test_baseline_architecture_and_exact_parameter_count() -> None:
    model = TransformerLM()
    config = model.config

    assert config == ModelConfig(
        vocab_size=1_792,
        context_length=256,
        num_layers=4,
        model_width=128,
        num_heads=4,
        mlp_width=512,
        rope_base=10_000.0,
        layer_norm_epsilon=1e-5,
        dropout=0.0,
    )
    assert config.head_dim == 32
    assert len(model.blocks) == 4
    assert all(block.attention.num_heads == 4 for block in model.blocks)
    assert all(isinstance(block.attention.rope, RotaryEmbedding) for block in model.blocks)
    assert all(block.mlp[0].out_features == 512 for block in model.blocks)
    assert all(isinstance(block.mlp[1], nn.GELU) for block in model.blocks)
    assert model.output.weight is model.token_embedding.weight
    assert all(module.bias is None for module in model.modules() if isinstance(module, nn.Linear))
    assert all(
        not hasattr(module, "bias")
        for module in model.modules()
        if isinstance(module, WeightOnlyLayerNorm)
    )
    assert not any("position" in name for name, _parameter in model.named_parameters())
    assert count_parameters(model) == EXPECTED_PARAMETER_COUNT == 1_016_960
    assert enforce_parameter_count(model) == EXPECTED_PARAMETER_COUNT
    assert EXPECTED_PARAMETER_COUNT < MAX_PARAMETER_COUNT == 1_050_000


def test_parameter_enforcement_rejects_wrong_size_and_cap_overflow() -> None:
    wrong_size = TransformerLM(ModelConfig(vocab_size=1_793))
    with pytest.raises(TrainingError, match="expected"):
        enforce_parameter_count(wrong_size)
    with pytest.raises(TrainingError, match="cap"):
        enforce_parameter_count(TransformerLM(), maximum=1_000_000)


def test_dense_causal_attention_cannot_see_future_tokens() -> None:
    seed_everything(7)
    model = TransformerLM().eval()
    original = torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16]])
    changed_future = original.clone()
    changed_future[:, 4:] = torch.tensor([101, 103, 105, 107])

    with torch.inference_mode():
        original_logits, _ = model(original)
        changed_logits, _ = model(changed_future)

    torch.testing.assert_close(original_logits[:, :4], changed_logits[:, :4], rtol=0, atol=0)
    assert not torch.equal(original_logits[:, 4:], changed_logits[:, 4:])


def test_forward_loss_and_context_limit() -> None:
    model = TransformerLM()
    inputs = torch.randint(0, model.config.vocab_size, (2, 32))
    logits, loss = model(inputs, inputs)

    assert logits.shape == (2, 32, model.config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    with pytest.raises(ValueError, match="exceeds context length"):
        model(torch.zeros((1, 257), dtype=torch.long))


def test_seed_reproduces_weights_and_sampling() -> None:
    seed_everything(91)
    first = TransformerLM().eval()
    seed_everything(91)
    second = TransformerLM().eval()
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)

    prompt = [3, 5, 7]
    first_sample = generate_token_ids(
        first,
        prompt,
        device=torch.device("cpu"),
        maximum_new_tokens=8,
        temperature=0.8,
        top_k=25,
        seed=123,
    )
    second_sample = generate_token_ids(
        first,
        prompt,
        device=torch.device("cpu"),
        maximum_new_tokens=8,
        temperature=0.8,
        top_k=25,
        seed=123,
    )
    assert first_sample == second_sample


@pytest.mark.parametrize(
    ("mode", "target"),
    [("cheap", 2_000_000), ("intermediate", 6_000_000), ("full", 20_000_000)],
)
def test_training_modes_consume_exact_token_budgets(mode: str, target: int) -> None:
    assert TRAINING_MODES[mode].target_tokens == target
    remaining = target
    consumed = 0
    while remaining:
        batch_size, sequence_length = batch_shape_for_remaining(
            remaining,
            maximum_batch_size=64,
            maximum_sequence_length=256,
        )
        step_tokens = batch_size * sequence_length
        assert 0 < step_tokens <= remaining
        assert batch_size <= 64
        assert sequence_length <= 256
        consumed += step_tokens
        remaining -= step_tokens
    assert consumed == target


def test_cosine_schedule_warms_up_and_reaches_minimum() -> None:
    values = [
        cosine_learning_rate(
            tokens,
            target_tokens=1_000,
            maximum_learning_rate=1e-3,
            minimum_learning_rate=1e-4,
            warmup_ratio=0.1,
        )
        for tokens in (0, 50, 100, 500, 1_000)
    ]

    assert values[0] == 0.0
    assert values[1] == pytest.approx(5e-4)
    assert values[2] == pytest.approx(1e-3)
    assert values[2] > values[3] > values[4]
    assert values[4] == pytest.approx(1e-4)
    assert math.isfinite(values[3])
