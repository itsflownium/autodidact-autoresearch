"""Train, inspect, checkpoint, and sample the 1M-parameter TinyStories baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

try:
    import resource as resource_module
except ImportError:  # pragma: no cover - unavailable on Windows
    resource_module = None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from autodidact.checkpoints import checkpoint_state_sha256, file_sha256
from autodidact.data.config import END_OF_TEXT_TOKEN, default_output_root
from autodidact.data.integrity import verify_dataset
from autodidact.data.reader import PreparedSplit

EXPECTED_PARAMETER_COUNT = 1_016_960
MAX_PARAMETER_COUNT = 1_050_000
CHECKPOINT_SCHEMA_VERSION = 2
METRICS_SCHEMA_VERSION = 1
DEFAULT_SEED = 1_337
DATA_ORDER_DIGEST_DOMAIN = b"autodidact-data-order-v1"


class TrainingError(RuntimeError):
    """Raised when the fixed training contract cannot be satisfied."""


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 1_792
    context_length: int = 256
    num_layers: int = 4
    model_width: int = 128
    num_heads: int = 4
    mlp_width: int = 512
    rope_base: float = 10_000.0
    layer_norm_epsilon: float = 1e-5
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.model_width // self.num_heads

    def validate(self) -> None:
        if self.model_width % self.num_heads:
            raise ValueError("model_width must be divisible by num_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingMode:
    target_tokens: int
    eval_tokens: int | None
    log_every_tokens: int
    checkpoint_every_tokens: int


TRAINING_MODES = {
    "cheap": TrainingMode(
        target_tokens=2_000_000,
        eval_tokens=250_000,
        log_every_tokens=100_000,
        checkpoint_every_tokens=1_000_000,
    ),
    "intermediate": TrainingMode(
        target_tokens=6_000_000,
        eval_tokens=1_000_000,
        log_every_tokens=250_000,
        checkpoint_every_tokens=2_000_000,
    ),
    "full": TrainingMode(
        target_tokens=20_000_000,
        eval_tokens=None,
        log_every_tokens=500_000,
        checkpoint_every_tokens=5_000_000,
    ),
}


class WeightOnlyLayerNorm(nn.Module):
    """LayerNorm with a learned scale and no learned bias."""

    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            inputs,
            (inputs.size(-1),),
            weight=self.weight,
            bias=None,
            eps=self.epsilon,
        )


def _apply_rope(
    values: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    even = values[..., ::2]
    odd = values[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inverse_frequencies", frequencies, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(query.size(-2), device=query.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inverse_frequencies.to(device=query.device))
        cosine = angles.cos().to(dtype=query.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=query.dtype)[None, None, :, :]
        return _apply_rope(query, cosine, sine), _apply_rope(key, cosine, sine)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.query_key_value = nn.Linear(
            config.model_width,
            3 * config.model_width,
            bias=False,
        )
        self.output = nn.Linear(config.model_width, config.model_width, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_base)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, width = inputs.shape
        query, key, value = self.query_key_value(inputs).chunk(3, dim=-1)

        def split_heads(values: torch.Tensor) -> torch.Tensor:
            return values.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        query, key = self.rope(query, key)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                width,
            )
        )
        return self.output(attention)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = WeightOnlyLayerNorm(
            config.model_width,
            config.layer_norm_epsilon,
        )
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = WeightOnlyLayerNorm(
            config.model_width,
            config.layer_norm_epsilon,
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.model_width, config.mlp_width, bias=False),
            nn.GELU(),
            nn.Linear(config.mlp_width, config.model_width, bias=False),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.dropout(self.attention(self.attention_norm(inputs)))
        return inputs + self.dropout(self.mlp(self.mlp_norm(inputs)))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_width)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = WeightOnlyLayerNorm(
            config.model_width,
            config.layer_norm_epsilon,
        )
        self.output = nn.Linear(config.model_width, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        self.output.weight = self.token_embedding.weight
        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp[-1].weight, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.size(1) > self.config.context_length:
            raise ValueError(
                f"sequence length {token_ids.size(1)} exceeds "
                f"context length {self.config.context_length}"
            )
        hidden = self.token_embedding(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def enforce_parameter_count(
    model: nn.Module,
    *,
    expected: int = EXPECTED_PARAMETER_COUNT,
    maximum: int = MAX_PARAMETER_COUNT,
) -> int:
    actual = count_parameters(model)
    if actual > maximum:
        raise TrainingError(f"model has {actual:,} parameters; cap is {maximum:,}")
    if actual != expected:
        raise TrainingError(f"model has {actual:,} parameters; expected {expected:,}")
    return actual


def _mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "mps":
        if not _mps_available():
            raise TrainingError("MPS was requested but is not available")
        return torch.device("mps")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise TrainingError("CUDA was requested but is not available")
        device = torch.device(normalized)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise TrainingError(f"CUDA device index is unavailable: {device.index}")
        return device
    raise TrainingError("device must be auto, cpu, mps, cuda, or cuda:<index>")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _mps_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


class TokenBatcher:
    """Deterministically sample contiguous next-token windows from immutable shards."""

    def __init__(self, split: PreparedSplit, *, seed: int) -> None:
        self.split = split
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.shards = [split.token_shard(index) for index in range(len(split.manifest["shards"]))]
        self.token_counts = np.asarray([len(shard) for shard in self.shards], dtype=np.int64)
        self._window_cache: dict[int, tuple[np.ndarray, int]] = {}
        self._order_digest = hashlib.sha256(DATA_ORDER_DIGEST_DOMAIN).digest()
        self._order_batches = 0
        self._order_tokens = 0

    def _windows(self, sequence_length: int) -> tuple[np.ndarray, int]:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        cached = self._window_cache.get(sequence_length)
        if cached is not None:
            return cached
        valid_starts = np.maximum(self.token_counts - sequence_length, 0)
        cumulative = np.cumsum(valid_starts, dtype=np.int64)
        total = int(cumulative[-1]) if len(cumulative) else 0
        if total <= 0:
            raise TrainingError(
                f"split {self.split.name} has no windows of length {sequence_length + 1}"
            )
        result = cumulative, total
        self._window_cache[sequence_length] = result
        return result

    def next_batch(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cumulative, total = self._windows(sequence_length)
        draws = self.rng.integers(0, total, size=batch_size, dtype=np.int64)
        shard_indexes = np.searchsorted(cumulative, draws, side="right")
        order_record = (
            struct.pack(">QQ", batch_size, sequence_length)
            + draws.astype("<u8", copy=False).tobytes()
        )
        self._order_digest = hashlib.sha256(self._order_digest + order_record).digest()
        self._order_batches += 1
        self._order_tokens += batch_size * sequence_length
        inputs = np.empty((batch_size, sequence_length), dtype=np.int64)
        targets = np.empty((batch_size, sequence_length), dtype=np.int64)
        for row, (draw, shard_index) in enumerate(
            zip(draws.tolist(), shard_indexes.tolist(), strict=True)
        ):
            previous = int(cumulative[shard_index - 1]) if shard_index else 0
            start = int(draw) - previous
            window = self.shards[shard_index][start : start + sequence_length + 1]
            inputs[row] = window[:-1]
            targets[row] = window[1:]
        return (
            torch.from_numpy(inputs).to(device=device),
            torch.from_numpy(targets).to(device=device),
        )

    @property
    def data_order_sha256(self) -> str:
        return self._order_digest.hex()

    def state_dict(self) -> dict[str, Any]:
        return {
            "data_order": {
                "batches": self._order_batches,
                "sha256": self.data_order_sha256,
                "tokens": self._order_tokens,
            },
            "numpy_bit_generator": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = copy.deepcopy(state["numpy_bit_generator"])
        order = state.get("data_order")
        if order is None:
            raise TrainingError("checkpoint does not contain a data-order fingerprint")
        self._order_digest = bytes.fromhex(order["sha256"])
        self._order_batches = int(order["batches"])
        self._order_tokens = int(order["tokens"])


def batch_shape_for_remaining(
    remaining_tokens: int,
    *,
    maximum_batch_size: int,
    maximum_sequence_length: int,
) -> tuple[int, int]:
    """Choose a rectangular batch that never exceeds the exact remaining budget."""

    if remaining_tokens <= 0:
        raise ValueError("remaining_tokens must be positive")
    if maximum_batch_size <= 0 or maximum_sequence_length <= 0:
        raise ValueError("batch and sequence limits must be positive")
    if remaining_tokens >= maximum_batch_size * maximum_sequence_length:
        return maximum_batch_size, maximum_sequence_length
    batch_size = min(maximum_batch_size, remaining_tokens)
    sequence_length = min(maximum_sequence_length, remaining_tokens // batch_size)
    if sequence_length <= 0:
        return 1, 1
    return batch_size, sequence_length


def cosine_learning_rate(
    completed_tokens: int,
    *,
    target_tokens: int,
    maximum_learning_rate: float,
    minimum_learning_rate: float,
    warmup_ratio: float,
) -> float:
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if not 0.0 <= minimum_learning_rate <= maximum_learning_rate:
        raise ValueError("learning-rate bounds are invalid")
    completed = min(max(completed_tokens, 0), target_tokens)
    warmup_tokens = int(target_tokens * warmup_ratio)
    if warmup_tokens and completed <= warmup_tokens:
        return maximum_learning_rate * completed / warmup_tokens
    decay_span = max(target_tokens - warmup_tokens, 1)
    progress = (completed - warmup_tokens) / decay_span
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate + cosine * (maximum_learning_rate - minimum_learning_rate)


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


class JsonlMetrics:
    def __init__(
        self,
        path: Path | None,
        *,
        append: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.path = path
        self.append = append
        self.stream = stream if stream is not None else sys.stdout
        self._file: TextIO | None = None

    def __enter__(self) -> JsonlMetrics:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a" if self.append else "w", encoding="utf-8")
        return self

    def emit(self, event: str, **values: Any) -> dict[str, Any]:
        payload = {"event": event, "schema_version": METRICS_SCHEMA_VERSION, **values}
        line = json.dumps(payload, sort_keys=True, allow_nan=False)
        print(line, file=self.stream, flush=True)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()
        return payload

    def __exit__(self, *_args: object) -> None:
        if self._file is not None:
            self._file.close()


def _process_peak_rss_bytes() -> int | None:
    if resource_module is None:
        return None
    peak = int(resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


class PeakMemoryTracker:
    """Collect an OS high-water mark and accelerator allocator peaks."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._mps_peak_allocated = 0
        self._mps_peak_driver = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self.sample()

    def sample(self) -> None:
        if self.device.type != "mps":
            return
        if hasattr(torch.mps, "current_allocated_memory"):
            self._mps_peak_allocated = max(
                self._mps_peak_allocated,
                int(torch.mps.current_allocated_memory()),
            )
        if hasattr(torch.mps, "driver_allocated_memory"):
            self._mps_peak_driver = max(
                self._mps_peak_driver,
                int(torch.mps.driver_allocated_memory()),
            )

    def snapshot(self) -> dict[str, int | str | None]:
        self.sample()
        if self.device.type == "cuda":
            allocated = int(torch.cuda.max_memory_allocated(self.device))
            reserved = int(torch.cuda.max_memory_reserved(self.device))
            measurement = "allocator_exact"
        elif self.device.type == "mps":
            allocated = self._mps_peak_allocated
            reserved = self._mps_peak_driver
            measurement = "allocator_sampled"
        else:
            allocated = None
            reserved = None
            measurement = "process_only"
        return {
            "device_memory_peak_kind": measurement,
            "peak_device_allocated_bytes": allocated,
            "peak_device_reserved_bytes": reserved,
            "peak_process_rss_bytes": _process_peak_rss_bytes(),
        }


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    def cpu_byte_tensor(value: Any, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
            raise TrainingError(f"checkpoint {name} RNG state is not a byte tensor")
        return value.detach().to(device="cpu").contiguous()

    torch_cpu = cpu_byte_tensor(state["torch_cpu"], "CPU")
    torch_cuda = None
    if device.type == "cuda" and "torch_cuda" in state:
        raw_cuda = state["torch_cuda"]
        if not isinstance(raw_cuda, (list, tuple)):
            raise TrainingError("checkpoint CUDA RNG state is not a sequence")
        torch_cuda = [
            cpu_byte_tensor(value, f"CUDA device {index}") for index, value in enumerate(raw_cuda)
        ]
    torch_mps = None
    if device.type == "mps" and "torch_mps" in state:
        torch_mps = cpu_byte_tensor(state["torch_mps"], "MPS")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch_cpu)
    if torch_cuda is not None:
        torch.cuda.set_rng_state_all(torch_cuda)
    elif torch_mps is not None and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(torch_mps)


def save_checkpoint(
    path: Path,
    *,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    batcher: TokenBatcher,
    device: torch.device,
    mode: str,
    seed: int,
    step: int,
    tokens_seen: int,
    target_tokens: int,
    cumulative_loss: float,
    cumulative_loss_tokens: int,
) -> dict[str, str]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config": asdict(model.config),
        "parameter_count": count_parameters(model),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "batcher_state": batcher.state_dict(),
        "rng_state": _capture_rng_state(device),
        "training": {
            "mode": mode,
            "seed": seed,
            "step": step,
            "target_tokens": target_tokens,
            "tokens_seen": tokens_seen,
            "cumulative_loss": cumulative_loss,
            "cumulative_loss_tokens": cumulative_loss_tokens,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_state_sha256": checkpoint_state_sha256(payload),
    }


def load_checkpoint_payload(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingError(f"checkpoint does not exist: {path}")
    # RNG APIs require CPU byte tensors. Model and optimizer restoration copy
    # their CPU-loaded state onto the destination parameters as needed.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError("unsupported checkpoint schema")
    return payload


def restore_training_checkpoint(
    path: Path,
    *,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    batcher: TokenBatcher,
    device: torch.device,
    target_tokens: int,
) -> tuple[int, int, int, float, int]:
    payload = load_checkpoint_payload(path, device)
    if payload["model_config"] != asdict(model.config):
        raise TrainingError("checkpoint model config does not match the baseline")
    if payload["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise TrainingError("checkpoint parameter count does not match the baseline")
    training = payload["training"]
    if training["target_tokens"] != target_tokens:
        raise TrainingError("checkpoint target token budget does not match this run")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    batcher.load_state_dict(payload["batcher_state"])
    _restore_rng_state(payload["rng_state"], device)
    return (
        int(training["step"]),
        int(training["tokens_seen"]),
        int(training["seed"]),
        float(training["cumulative_loss"]),
        int(training["cumulative_loss_tokens"]),
    )


def _flush_eval_chunks(
    model: TransformerLM,
    chunks: list[tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
) -> tuple[float, int]:
    maximum_length = max(len(inputs) for inputs, _targets in chunks)
    batch_inputs = np.zeros((len(chunks), maximum_length), dtype=np.int64)
    batch_targets = np.full((len(chunks), maximum_length), -100, dtype=np.int64)
    predictions = 0
    for row, (inputs, targets) in enumerate(chunks):
        batch_inputs[row, : len(inputs)] = inputs
        batch_targets[row, : len(targets)] = targets
        predictions += len(targets)
    inputs_tensor = torch.from_numpy(batch_inputs).to(device=device)
    targets_tensor = torch.from_numpy(batch_targets).to(device=device)
    logits, _loss = model(inputs_tensor)
    negative_log_likelihood = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets_tensor.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(negative_log_likelihood.item()), predictions


@torch.inference_mode()
def evaluate_bpb(
    model: TransformerLM,
    split: PreparedSplit,
    *,
    device: torch.device,
    maximum_tokens: int | None,
    batch_size: int = 64,
) -> dict[str, float | int]:
    if maximum_tokens is not None and maximum_tokens <= 0:
        raise ValueError("maximum_tokens must be positive or None")
    was_training = model.training
    model.eval()
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    total_nll = 0.0
    predicted_tokens = 0
    selected_tokens = 0
    utf8_bytes = 0
    stories = 0
    stop = False
    for shard_index in range(len(split.manifest["shards"])):
        tokens = split.token_shard(shard_index)
        index = split.document_index(shard_index)
        for row in index:
            token_count = int(row["token_count"])
            document_predictions = max(token_count - 1, 0)
            if (
                maximum_tokens is not None
                and selected_tokens > 0
                and selected_tokens + document_predictions > maximum_tokens
            ):
                stop = True
                break
            offset = int(row["offset"])
            document = np.asarray(
                tokens[offset : offset + token_count],
                dtype=np.int64,
            )
            for start in range(0, document_predictions, model.config.context_length):
                end = min(start + model.config.context_length, document_predictions)
                chunks.append((document[start:end], document[start + 1 : end + 1]))
                if len(chunks) >= batch_size:
                    nll, count = _flush_eval_chunks(model, chunks, device=device)
                    total_nll += nll
                    predicted_tokens += count
                    chunks.clear()
            utf8_bytes += int(row["utf8_bytes"])
            stories += 1
            selected_tokens += document_predictions
        if stop:
            break
    if chunks:
        nll, count = _flush_eval_chunks(model, chunks, device=device)
        total_nll += nll
        predicted_tokens += count
    if not utf8_bytes or not predicted_tokens:
        raise TrainingError(f"split {split.name} did not yield evaluation data")
    if predicted_tokens != selected_tokens:
        raise TrainingError("evaluation token accounting is inconsistent")
    if was_training:
        model.train()
    bits = total_nll / math.log(2.0)
    return {
        "bpb": bits / utf8_bytes,
        "negative_log_likelihood": total_nll,
        "predicted_tokens": predicted_tokens,
        "stories": stories,
        "utf8_bytes": utf8_bytes,
    }


@torch.inference_mode()
def generate_token_ids(
    model: TransformerLM,
    prompt_ids: list[int],
    *,
    device: torch.device,
    maximum_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    end_of_text_id: int | None = None,
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    if maximum_new_tokens < 0:
        raise ValueError("maximum_new_tokens must be non-negative")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    was_training = model.training
    model.eval()
    generated = list(prompt_ids)
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(seed)
    for _ in range(maximum_new_tokens):
        context = generated[-model.config.context_length :]
        inputs = torch.tensor(context, dtype=torch.long, device=device)[None, :]
        logits, _loss = model(inputs)
        next_logits = logits[0, -1].float()
        if temperature == 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_logits = next_logits / temperature
            retained = min(top_k, next_logits.numel())
            threshold = torch.topk(next_logits, retained).values[-1]
            next_logits = next_logits.masked_fill(next_logits < threshold, -torch.inf)
            probabilities = torch.softmax(next_logits, dim=-1).cpu()
            next_id = int(
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=sampling_generator,
                ).item()
            )
        generated.append(next_id)
        if end_of_text_id is not None and next_id == end_of_text_id:
            break
    if was_training:
        model.train()
    return generated


def _load_verified_public_data(
    dataset_root: Path,
    model_config: ModelConfig,
) -> tuple[dict[str, Any], PreparedSplit, PreparedSplit, Tokenizer]:
    manifest = verify_dataset(dataset_root, scope="public")
    tokenizer_record = manifest["tokenizer"]
    if tokenizer_record["vocab_size"] != model_config.vocab_size:
        raise TrainingError(
            f"dataset vocabulary is {tokenizer_record['vocab_size']}; "
            f"model requires {model_config.vocab_size}"
        )
    tokenizer_path = dataset_root / "public" / tokenizer_record["artifact"]["path"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return (
        manifest,
        PreparedSplit(dataset_root, manifest["splits"]["train"]),
        PreparedSplit(dataset_root, manifest["splits"]["dev"]),
        tokenizer,
    )


def _default_checkpoint_path(mode: str) -> Path:
    return Path("artifacts/checkpoints") / f"baseline-{mode}.pt"


def _default_metrics_path(mode: str) -> Path:
    return Path("artifacts/metrics") / f"baseline-{mode}.jsonl"


def run_training(args: argparse.Namespace) -> int:
    mode = TRAINING_MODES[args.mode]
    target_tokens = args.token_budget if args.token_budget is not None else mode.target_tokens
    if target_tokens <= 0:
        raise TrainingError("token budget must be positive")
    eval_tokens = mode.eval_tokens if args.eval_tokens is None else args.eval_tokens
    if args.skip_eval:
        eval_tokens = 0
    log_every = args.log_every_tokens or mode.log_every_tokens
    checkpoint_every = args.checkpoint_every_tokens or mode.checkpoint_every_tokens
    checkpoint_path = args.checkpoint_out or _default_checkpoint_path(args.mode)
    metrics_path = args.metrics_file or _default_metrics_path(args.mode)
    if eval_tokens is not None and eval_tokens < 0:
        raise TrainingError("evaluation token budget cannot be negative")
    if log_every <= 0 or checkpoint_every <= 0:
        raise TrainingError("logging and checkpoint intervals must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise TrainingError("batch sizes must be positive")
    if args.grad_clip <= 0.0:
        raise TrainingError("gradient clipping threshold must be positive")
    if args.weight_decay < 0.0:
        raise TrainingError("weight decay cannot be negative")
    cosine_learning_rate(
        0,
        target_tokens=target_tokens,
        maximum_learning_rate=args.maximum_learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        warmup_ratio=args.warmup_ratio,
    )

    device = resolve_device(args.device)
    seed_everything(args.seed, deterministic=args.deterministic)
    model_config = ModelConfig()
    manifest, train_split, dev_split, tokenizer = _load_verified_public_data(
        args.data_root,
        model_config,
    )
    model = TransformerLM(model_config).to(device)
    parameter_count = enforce_parameter_count(model)
    optimizer = build_optimizer(
        model,
        learning_rate=args.maximum_learning_rate,
        weight_decay=args.weight_decay,
    )
    batcher = TokenBatcher(train_split, seed=args.seed)

    step = 0
    tokens_seen = 0
    active_seed = args.seed
    cumulative_loss = 0.0
    cumulative_loss_tokens = 0
    if args.resume is not None:
        (
            step,
            tokens_seen,
            active_seed,
            cumulative_loss,
            cumulative_loss_tokens,
        ) = restore_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            batcher=batcher,
            device=device,
            target_tokens=target_tokens,
        )
        if active_seed != args.seed:
            raise TrainingError(
                f"checkpoint seed is {active_seed}; command requested seed {args.seed}"
            )
        if tokens_seen > target_tokens:
            raise TrainingError("checkpoint has already exceeded the target token budget")

    process_start_tokens = tokens_seen
    next_log = (tokens_seen // log_every + 1) * log_every
    next_checkpoint = (tokens_seen // checkpoint_every + 1) * checkpoint_every
    memory = PeakMemoryTracker(device)
    started = time.perf_counter()
    training_seconds = 0.0
    interval_loss = 0.0
    interval_tokens = 0
    interval_seconds = 0.0
    final_grad_norm = 0.0
    final_learning_rate = args.maximum_learning_rate
    evaluation: dict[str, float | int] | None = None
    evaluation_seconds: float | None = None
    evaluation_tokens_per_second: float | None = None
    generated_text: str | None = None

    with JsonlMetrics(metrics_path, append=args.resume is not None) as metrics:
        metrics.emit(
            "config",
            checkpoint_path=str(checkpoint_path),
            data_config_sha256=manifest["pipeline"]["config_sha256"],
            deterministic=args.deterministic,
            device=str(device),
            eval_tokens=eval_tokens,
            mode=args.mode,
            model=asdict(model_config),
            parameter_cap=MAX_PARAMETER_COUNT,
            parameter_count=parameter_count,
            seed=args.seed,
            target_tokens=target_tokens,
            tokenizer_sha256=manifest["tokenizer"]["artifact"]["sha256"],
        )
        model.train()
        while tokens_seen < target_tokens:
            remaining = target_tokens - tokens_seen
            batch_size, sequence_length = batch_shape_for_remaining(
                remaining,
                maximum_batch_size=args.batch_size,
                maximum_sequence_length=model_config.context_length,
            )
            inputs, targets = batcher.next_batch(
                batch_size,
                sequence_length,
                device=device,
            )
            step_tokens = inputs.numel()
            learning_rate = cosine_learning_rate(
                tokens_seen + step_tokens,
                target_tokens=target_tokens,
                maximum_learning_rate=args.maximum_learning_rate,
                minimum_learning_rate=args.minimum_learning_rate,
                warmup_ratio=args.warmup_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            synchronize_device(device)
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            _logits, loss = model(inputs, targets)
            assert loss is not None
            if not torch.isfinite(loss):
                raise TrainingError(f"non-finite loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
            if not torch.isfinite(grad_norm):
                raise TrainingError(f"non-finite gradient norm at step {step + 1}")
            optimizer.step()
            synchronize_device(device)
            step_seconds = time.perf_counter() - step_started
            memory.sample()

            loss_value = float(loss.item())
            step += 1
            tokens_seen += step_tokens
            cumulative_loss += loss_value * step_tokens
            cumulative_loss_tokens += step_tokens
            interval_loss += loss_value * step_tokens
            interval_tokens += step_tokens
            interval_seconds += step_seconds
            training_seconds += step_seconds
            final_grad_norm = float(grad_norm.item())
            final_learning_rate = learning_rate

            if tokens_seen >= next_log or tokens_seen == target_tokens:
                mean_loss = interval_loss / interval_tokens
                metrics.emit(
                    "train",
                    bits_per_token=mean_loss / math.log(2.0),
                    data_order_sha256=batcher.data_order_sha256,
                    grad_norm=final_grad_norm,
                    learning_rate=learning_rate,
                    loss=mean_loss,
                    step=step,
                    tokens_per_second=interval_tokens / max(interval_seconds, 1e-12),
                    tokens_seen=tokens_seen,
                    **memory.snapshot(),
                )
                interval_loss = 0.0
                interval_tokens = 0
                interval_seconds = 0.0
                while next_log <= tokens_seen:
                    next_log += log_every

            if tokens_seen >= next_checkpoint and tokens_seen < target_tokens:
                checkpoint_fingerprints = save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    batcher=batcher,
                    device=device,
                    mode=args.mode,
                    seed=args.seed,
                    step=step,
                    tokens_seen=tokens_seen,
                    target_tokens=target_tokens,
                    cumulative_loss=cumulative_loss,
                    cumulative_loss_tokens=cumulative_loss_tokens,
                )
                metrics.emit(
                    "checkpoint",
                    data_order_sha256=batcher.data_order_sha256,
                    final=False,
                    path=str(checkpoint_path),
                    step=step,
                    tokens_seen=tokens_seen,
                    **checkpoint_fingerprints,
                )
                while next_checkpoint <= tokens_seen:
                    next_checkpoint += checkpoint_every

        final_checkpoint_fingerprints = save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            batcher=batcher,
            device=device,
            mode=args.mode,
            seed=args.seed,
            step=step,
            tokens_seen=tokens_seen,
            target_tokens=target_tokens,
            cumulative_loss=cumulative_loss,
            cumulative_loss_tokens=cumulative_loss_tokens,
        )
        metrics.emit(
            "checkpoint",
            data_order_sha256=batcher.data_order_sha256,
            final=True,
            path=str(checkpoint_path),
            step=step,
            tokens_seen=tokens_seen,
            **final_checkpoint_fingerprints,
        )

        if eval_tokens != 0:
            synchronize_device(device)
            evaluation_started = time.perf_counter()
            evaluation = evaluate_bpb(
                model,
                dev_split,
                device=device,
                maximum_tokens=eval_tokens,
                batch_size=args.eval_batch_size,
            )
            synchronize_device(device)
            evaluation_seconds = time.perf_counter() - evaluation_started
            evaluation_tokens_per_second = int(evaluation["predicted_tokens"]) / max(
                evaluation_seconds, 1e-12
            )
            memory.sample()
            metrics.emit(
                "evaluation",
                evaluation_seconds=evaluation_seconds,
                evaluation_tokens_per_second=evaluation_tokens_per_second,
                split="dev",
                **evaluation,
                **memory.snapshot(),
            )

        if not args.no_generate:
            prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
            end_of_text_id = tokenizer.token_to_id(END_OF_TEXT_TOKEN)
            generated_ids = generate_token_ids(
                model,
                prompt_ids,
                device=device,
                maximum_new_tokens=args.generate_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
                end_of_text_id=end_of_text_id,
            )
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            memory.sample()
            metrics.emit(
                "generation",
                prompt=args.prompt,
                text=generated_text,
                token_count=len(generated_ids),
            )

        metrics.emit(
            "summary",
            checkpoint_path=str(checkpoint_path),
            data_order_sha256=batcher.data_order_sha256,
            elapsed_seconds=time.perf_counter() - started,
            evaluation_seconds=evaluation_seconds,
            evaluation_tokens_per_second=evaluation_tokens_per_second,
            final_grad_norm=final_grad_norm,
            final_learning_rate=final_learning_rate,
            generated_text=generated_text,
            mean_train_loss=(
                cumulative_loss / cumulative_loss_tokens if cumulative_loss_tokens else None
            ),
            mode=args.mode,
            parameter_count=parameter_count,
            seed=args.seed,
            steps=step,
            target_tokens=target_tokens,
            tokens_seen=tokens_seen,
            training_seconds=training_seconds,
            training_tokens_per_second=(tokens_seen - process_start_tokens)
            / max(training_seconds, 1e-12),
            training_tokens_this_process=tokens_seen - process_start_tokens,
            validation_bpb=evaluation["bpb"] if evaluation is not None else None,
            **final_checkpoint_fingerprints,
            **memory.snapshot(),
        )
    return 0


def run_generation(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    seed_everything(args.seed, deterministic=args.deterministic)
    payload = load_checkpoint_payload(args.checkpoint, device)
    config = ModelConfig(**payload["model_config"])
    model = TransformerLM(config).to(device)
    enforce_parameter_count(model)
    model.load_state_dict(payload["model_state"])
    manifest, _train, _dev, tokenizer = _load_verified_public_data(args.data_root, config)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    generated_ids = generate_token_ids(
        model,
        prompt_ids,
        device=device,
        maximum_new_tokens=args.generate_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        end_of_text_id=tokenizer.token_to_id(END_OF_TEXT_TOKEN),
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "event": "generation",
        "parameter_count": count_parameters(model),
        "prompt": args.prompt,
        "schema_version": METRICS_SCHEMA_VERSION,
        "seed": args.seed,
        "text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "token_count": len(generated_ids),
        "tokenizer_sha256": manifest["tokenizer"]["artifact"]["sha256"],
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


def run_inspection(_args: argparse.Namespace) -> int:
    model = TransformerLM()
    parameter_count = enforce_parameter_count(model)
    payload = {
        "device_availability": {
            "cpu": True,
            "cuda": torch.cuda.is_available(),
            "mps": _mps_available(),
        },
        "event": "inspection",
        "model": asdict(model.config),
        "parameter_cap": MAX_PARAMETER_COUNT,
        "parameter_count": parameter_count,
        "schema_version": METRICS_SCHEMA_VERSION,
        "training_modes": {name: asdict(mode) for name, mode in TRAINING_MODES.items()},
        "tied_embeddings": model.output.weight is model.token_embedding.weight,
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _add_device_and_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--generate-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed 1M-parameter TinyStories transformer baseline."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="print the model contract as JSON")
    inspect.set_defaults(handler=run_inspection)

    train = commands.add_parser("train", help="train on immutable TinyStories shards")
    train.add_argument("--mode", choices=tuple(TRAINING_MODES), default="cheap")
    train.add_argument("--data-root", type=_path, default=default_output_root())
    train.add_argument("--token-budget", type=int)
    train.add_argument("--eval-tokens", type=int)
    train.add_argument("--skip-eval", action="store_true")
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--eval-batch-size", type=int, default=64)
    train.add_argument("--maximum-learning-rate", type=float, default=1e-3)
    train.add_argument("--minimum-learning-rate", type=float, default=1e-4)
    train.add_argument("--warmup-ratio", type=float, default=0.05)
    train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--log-every-tokens", type=int)
    train.add_argument("--checkpoint-every-tokens", type=int)
    train.add_argument("--checkpoint-out", type=_path)
    train.add_argument("--resume", type=_path)
    train.add_argument("--metrics-file", type=_path)
    train.add_argument("--no-generate", action="store_true")
    _add_device_and_seed(train)
    _add_generation_options(train)
    train.set_defaults(handler=run_training)

    generate = commands.add_parser("generate", help="sample text from a saved checkpoint")
    generate.add_argument("--checkpoint", type=_path, required=True)
    generate.add_argument("--data-root", type=_path, default=default_output_root())
    _add_device_and_seed(generate)
    _add_generation_options(generate)
    generate.set_defaults(handler=run_generation)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0].startswith("-"):
        arguments.insert(0, "train")
    args = build_parser().parse_args(arguments)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, TrainingError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
