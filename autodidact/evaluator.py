"""Protected checkpoint inspection and held-out BPB evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from autodidact.checkpoints import file_sha256
from autodidact.data.reader import PreparedSplit, open_evaluator_split, open_public_split
from autodidact.records import DEFAULT_PARAMETER_CAP

EVALUATOR_SCHEMA_VERSION = 1


class EvaluationError(RuntimeError):
    """Raised when candidate code or a checkpoint violates the evaluation contract."""


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _load_trainer(path: Path) -> ModuleType:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.name != "train.py":
        raise EvaluationError("trainer must be an existing train.py file")
    module_name = f"_autodidact_trainer_{file_sha256(resolved)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise EvaluationError("cannot load the candidate trainer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    trainer_directory = str(resolved.parent)
    sys.path.insert(0, trainer_directory)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise EvaluationError(f"candidate trainer import failed: {type(error).__name__}") from error
    finally:
        sys.path.remove(trainer_directory)
    return module


def _model_contract(module: ModuleType) -> tuple[Any, torch.nn.Module]:
    config_class = getattr(module, "ModelConfig", None)
    model_class = getattr(module, "TransformerLM", None)
    if config_class is None or model_class is None:
        raise EvaluationError("trainer must expose ModelConfig and TransformerLM")
    try:
        config = config_class()
        model = model_class(config)
    except Exception as error:
        raise EvaluationError(
            f"candidate model construction failed: {type(error).__name__}"
        ) from error
    if not isinstance(model, torch.nn.Module):
        raise EvaluationError("TransformerLM must construct a torch.nn.Module")
    return config, model


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def inspect_trainer(
    trainer_path: Path,
    *,
    parameter_cap: int = DEFAULT_PARAMETER_CAP,
) -> dict[str, Any]:
    module = _load_trainer(trainer_path)
    config, model = _model_contract(module)
    parameter_count = _parameter_count(model)
    if parameter_count <= 0 or parameter_count > parameter_cap:
        raise EvaluationError(f"candidate has {parameter_count} parameters; cap is {parameter_cap}")
    context_length = getattr(config, "context_length", None)
    vocab_size = getattr(config, "vocab_size", None)
    if type(context_length) is not int or context_length <= 0:
        raise EvaluationError("ModelConfig.context_length must be a positive integer")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise EvaluationError("ModelConfig.vocab_size must be a positive integer")
    if is_dataclass(config):
        config_payload: Any = asdict(config)
    else:
        config_payload = {
            key: value
            for key, value in vars(config).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
    return {
        "context_length": context_length,
        "event": "protected_inspection",
        "model_config": config_payload,
        "parameter_cap": parameter_cap,
        "parameter_count": parameter_count,
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "trainer_sha256": file_sha256(trainer_path),
        "vocab_size": vocab_size,
    }


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise EvaluationError("MPS was requested but is unavailable")
    return device


def _load_checkpoint_model(
    trainer_path: Path,
    checkpoint_path: Path,
    *,
    device: torch.device,
    parameter_cap: int,
) -> tuple[torch.nn.Module, Any, dict[str, Any], int]:
    module = _load_trainer(trainer_path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise EvaluationError(f"checkpoint load failed: {type(error).__name__}") from error
    if not isinstance(payload, dict):
        raise EvaluationError("checkpoint payload must be a mapping")
    model_config = payload.get("model_config")
    model_state = payload.get("model_state")
    if not isinstance(model_config, dict) or not isinstance(model_state, dict):
        raise EvaluationError("checkpoint is missing model_config or model_state")
    config_class = getattr(module, "ModelConfig", None)
    model_class = getattr(module, "TransformerLM", None)
    if config_class is None or model_class is None:
        raise EvaluationError("trainer must expose ModelConfig and TransformerLM")
    try:
        config = config_class(**model_config)
        model = model_class(config)
        model.load_state_dict(model_state, strict=True)
    except Exception as error:
        raise EvaluationError(
            f"checkpoint model restoration failed: {type(error).__name__}"
        ) from error
    if not isinstance(model, torch.nn.Module):
        raise EvaluationError("checkpoint did not restore a torch.nn.Module")
    parameter_count = _parameter_count(model)
    if parameter_count <= 0 or parameter_count > parameter_cap:
        raise EvaluationError(
            f"checkpoint has {parameter_count} parameters; cap is {parameter_cap}"
        )
    if payload.get("parameter_count") != parameter_count:
        raise EvaluationError("checkpoint parameter count does not match protected inspection")
    return model.to(device), config, payload, parameter_count


def _process_peak_rss_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if sys.platform == "darwin" else maximum * 1_024


def _device_memory(device: torch.device) -> tuple[int | None, int | None]:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device)), int(
            torch.cuda.max_memory_reserved(device)
        )
    if device.type == "mps" and hasattr(torch, "mps"):
        allocated = (
            int(torch.mps.current_allocated_memory())
            if hasattr(torch.mps, "current_allocated_memory")
            else None
        )
        reserved = (
            int(torch.mps.driver_allocated_memory())
            if hasattr(torch.mps, "driver_allocated_memory")
            else None
        )
        return allocated, reserved
    return None, None


def _model_logits(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output = model(inputs)
    logits = output[0] if isinstance(output, tuple) else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise EvaluationError("TransformerLM forward must return [batch, time, vocab] logits")
    return logits


def _flush_chunks(
    model: torch.nn.Module,
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
    logits = _model_logits(model, inputs_tensor)
    if logits.shape[:2] != inputs_tensor.shape:
        raise EvaluationError("model logits do not match the evaluation input shape")
    negative_log_likelihood = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets_tensor.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    if not torch.isfinite(negative_log_likelihood):
        raise EvaluationError("protected evaluation produced non-finite loss")
    return float(negative_log_likelihood.item()), predictions


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    split: PreparedSplit,
    *,
    context_length: int,
    device: torch.device,
    maximum_tokens: int | None,
    batch_size: int,
) -> dict[str, float | int]:
    if maximum_tokens is not None and maximum_tokens <= 0:
        raise EvaluationError("maximum_tokens must be positive or None")
    if batch_size <= 0:
        raise EvaluationError("evaluation batch size must be positive")
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
            document = np.asarray(tokens[offset : offset + token_count], dtype=np.int64)
            for start in range(0, document_predictions, context_length):
                end = min(start + context_length, document_predictions)
                chunks.append((document[start:end], document[start + 1 : end + 1]))
                if len(chunks) >= batch_size:
                    nll, count = _flush_chunks(model, chunks, device=device)
                    total_nll += nll
                    predicted_tokens += count
                    chunks.clear()
            utf8_bytes += int(row["utf8_bytes"])
            stories += 1
            selected_tokens += document_predictions
        if stop:
            break
    if chunks:
        nll, count = _flush_chunks(model, chunks, device=device)
        total_nll += nll
        predicted_tokens += count
    if not utf8_bytes or not predicted_tokens or predicted_tokens != selected_tokens:
        raise EvaluationError("evaluation token accounting is invalid")
    return {
        "negative_log_likelihood": total_nll,
        "predicted_tokens": predicted_tokens,
        "stories": stories,
        "utf8_bytes": utf8_bytes,
        "validation_bpb": total_nll / math.log(2.0) / utf8_bytes,
    }


def evaluate_checkpoint(
    trainer_path: Path,
    checkpoint_path: Path,
    data_root: Path,
    *,
    split_name: str,
    maximum_tokens: int | None,
    batch_size: int,
    device_name: str,
    parameter_cap: int = DEFAULT_PARAMETER_CAP,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_hash = file_sha256(checkpoint_path)
    device = _resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, config, _payload, parameter_count = _load_checkpoint_model(
        trainer_path,
        checkpoint_path,
        device=device,
        parameter_cap=parameter_cap,
    )
    context_length = getattr(config, "context_length", None)
    if type(context_length) is not int or context_length <= 0:
        raise EvaluationError("checkpoint context length is invalid")
    if split_name == "dev":
        split = open_public_split(data_root, "dev")
    else:
        split = open_evaluator_split(data_root, split_name)
    started = time.perf_counter()
    metrics = evaluate_model(
        model,
        split,
        context_length=context_length,
        device=device,
        maximum_tokens=maximum_tokens,
        batch_size=batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    allocated, reserved = _device_memory(device)
    if file_sha256(checkpoint_path) != checkpoint_hash:
        raise EvaluationError("checkpoint changed during protected evaluation")
    return {
        "batch_size": batch_size,
        "checkpoint_sha256": checkpoint_hash,
        "device": str(device),
        "evaluation_seconds": elapsed,
        "evaluation_tokens_per_second": metrics["predicted_tokens"] / max(elapsed, 1e-12),
        "event": "protected_evaluation",
        "maximum_tokens": maximum_tokens,
        "parameter_count": parameter_count,
        "peak_device_allocated_bytes": allocated,
        "peak_device_reserved_bytes": reserved,
        "peak_process_rss_bytes": _process_peak_rss_bytes(),
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "split": split_name,
        "trainer_sha256": file_sha256(trainer_path),
        **metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run protected model inspection and evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="inspect a trainer without trusting its CLI")
    inspect.add_argument("--trainer", type=_path, required=True)
    inspect.add_argument("--parameter-cap", type=int, default=DEFAULT_PARAMETER_CAP)

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint with protected BPB")
    evaluate.add_argument("--trainer", type=_path, required=True)
    evaluate.add_argument("--checkpoint", type=_path, required=True)
    evaluate.add_argument("--data-root", type=_path, required=True)
    evaluate.add_argument("--split", choices=("dev", "promotion", "sealed_final"), required=True)
    evaluate.add_argument("--maximum-tokens", type=int)
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--parameter-cap", type=int, default=DEFAULT_PARAMETER_CAP)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            payload = inspect_trainer(args.trainer, parameter_cap=args.parameter_cap)
        else:
            payload = evaluate_checkpoint(
                args.trainer,
                args.checkpoint,
                args.data_root,
                split_name=args.split,
                maximum_tokens=args.maximum_tokens,
                batch_size=args.batch_size,
                device_name=args.device,
                parameter_cap=args.parameter_cap,
            )
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0
    except (EvaluationError, FileNotFoundError, PermissionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
