"""Run and summarize repeated baseline experiments for seed-noise calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from autodidact.checkpoints import checkpoint_state_sha256
from autodidact.data.config import default_output_root

CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_SEED = 1_337
DEFAULT_ADDITIONAL_SEEDS = (2_027, 4_099, 7_919, 104_729, 130_363, 155_921, 196_613)
DEFAULT_RUN_ORDER_SEED = 20_260_710
CHECKPOINT_ABSOLUTE_TOLERANCE = 1e-6
CHECKPOINT_METRIC_ABSOLUTE_TOLERANCE = 1e-7
BPB_ABSOLUTE_TOLERANCE = 1e-7
OUTPUT_MARKER = ".autodidact-calibration"
MAX_PYTHON_HASH_SEED = 2**32 - 1
CALIBRATION_MODES = {
    "cheap": {"target_tokens": 2_000_000, "eval_tokens": 250_000},
    "intermediate": {"target_tokens": 6_000_000, "eval_tokens": 1_000_000},
    "full": {"target_tokens": 20_000_000, "eval_tokens": None},
}


class CalibrationError(RuntimeError):
    """Raised when a calibration matrix cannot produce trustworthy evidence."""


@dataclass(frozen=True)
class CalibrationRunSpec:
    run_id: str
    seed: int
    group: str
    replicate: int


def build_run_specs(
    *,
    repeat_seed: int,
    repeat_count: int,
    additional_seeds: list[int],
    run_order_seed: int,
) -> list[CalibrationRunSpec]:
    if repeat_seed < 0 or run_order_seed < 0:
        raise ValueError("seeds must be non-negative")
    if repeat_seed > MAX_PYTHON_HASH_SEED or any(
        seed > MAX_PYTHON_HASH_SEED for seed in additional_seeds
    ):
        raise ValueError(f"training seeds must be at most {MAX_PYTHON_HASH_SEED}")
    if repeat_count < 2:
        raise ValueError("repeat_count must be at least two")
    if not additional_seeds:
        raise ValueError("at least one additional seed is required")
    if any(seed < 0 for seed in additional_seeds):
        raise ValueError("seeds must be non-negative")
    if repeat_seed in additional_seeds:
        raise ValueError("additional seeds must not include the repeated seed")
    if len(set(additional_seeds)) != len(additional_seeds):
        raise ValueError("additional seeds must be unique")

    specs = [
        CalibrationRunSpec(
            run_id=f"repeat-{index:02d}",
            seed=repeat_seed,
            group="same_seed",
            replicate=index,
        )
        for index in range(1, repeat_count + 1)
    ]
    specs.extend(
        CalibrationRunSpec(
            run_id=f"seed-{seed}",
            seed=seed,
            group="additional_seed",
            replicate=1,
        )
        for seed in additional_seeds
    )
    random.Random(run_order_seed).shuffle(specs)
    return specs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output_root(path: Path, *, overwrite: bool) -> Path:
    output_root = path.resolve()
    marker = output_root / OUTPUT_MARKER
    if output_root.exists():
        if not overwrite:
            raise CalibrationError(f"output root already exists: {output_root}")
        if not marker.is_file():
            raise CalibrationError(
                f"refusing to overwrite an unmarked calibration directory: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    (output_root / OUTPUT_MARKER).write_text(
        f"schema_version={CALIBRATION_SCHEMA_VERSION}\n",
        encoding="utf-8",
    )
    return output_root


def _load_metric_events(path: Path) -> list[dict[str, Any]]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read metrics from {path}: {error}") from error
    if not records:
        raise CalibrationError(f"training produced no metrics: {path}")
    return records


def _single_event(records: list[dict[str, Any]], event: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("event") == event]
    if len(matches) != 1:
        raise CalibrationError(f"expected one {event} event, found {len(matches)}")
    return matches[0]


def _tensor_map(value: Any, prefix: str = "root") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu()}
    tensors: dict[str, torch.Tensor] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            tensors.update(_tensor_map(item, f"{prefix}/{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            tensors.update(_tensor_map(item, f"{prefix}/{index}"))
    return tensors


def _replace_tensors_with_contract(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_dtype": str(value.dtype),
            "tensor_shape": list(value.shape),
        }
    if isinstance(value, dict):
        return {key: _replace_tensors_with_contract(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_replace_tensors_with_contract(item) for item in value)
    if isinstance(value, list):
        return [_replace_tensors_with_contract(item) for item in value]
    return value


def _compare_tensor_component(
    payloads: list[dict[str, Any]],
    component: str,
    *,
    absolute_tolerance: float,
) -> dict[str, Any]:
    contracts = [
        checkpoint_state_sha256({"value": _replace_tensors_with_contract(payload[component])})
        for payload in payloads
    ]
    maps = [_tensor_map(payload[component], component) for payload in payloads]
    reference = maps[0]
    contract_equal = len(set(contracts)) == 1
    exact = contract_equal
    within_tolerance = contract_equal
    maximum_absolute_difference = 0.0
    changed_tensors = 0
    pair_count = 0
    for left_index, left in enumerate(maps):
        for right in maps[left_index + 1 :]:
            pair_count += 1
            if left.keys() != right.keys():
                exact = False
                within_tolerance = False
                continue
            for name, expected in left.items():
                actual = right[name]
                if expected.dtype != actual.dtype or expected.shape != actual.shape:
                    exact = False
                    within_tolerance = False
                    continue
                if torch.equal(expected, actual):
                    continue
                exact = False
                changed_tensors += 1
                if expected.is_floating_point() or expected.is_complex():
                    difference = float((expected - actual).abs().max().item())
                    maximum_absolute_difference = max(maximum_absolute_difference, difference)
                    if not torch.allclose(
                        expected,
                        actual,
                        rtol=0.0,
                        atol=absolute_tolerance,
                        equal_nan=False,
                    ):
                        within_tolerance = False
                else:
                    within_tolerance = False
    return {
        "contract_equal": contract_equal,
        "exact": exact,
        "maximum_absolute_difference": maximum_absolute_difference,
        "tensor_comparisons": pair_count * len(reference),
        "tensors_with_exact_differences": changed_tensors,
        "within_absolute_tolerance": within_tolerance,
    }


def compare_checkpoints(
    paths: list[Path],
    *,
    absolute_tolerance: float = CHECKPOINT_ABSOLUTE_TOLERANCE,
    training_loss_absolute_tolerance: float = CHECKPOINT_METRIC_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two checkpoints are required")
    if absolute_tolerance < 0.0 or training_loss_absolute_tolerance < 0.0:
        raise ValueError("checkpoint tolerances must be non-negative")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    state_hashes = [checkpoint_state_sha256(payload) for payload in payloads]
    mean_training_losses = [
        float(payload["training"]["cumulative_loss"])
        / int(payload["training"]["cumulative_loss_tokens"])
        for payload in payloads
    ]
    training_loss_range = max(mean_training_losses) - min(mean_training_losses)
    metadata_hashes = [
        checkpoint_state_sha256(
            {
                key: (
                    {
                        training_key: training_value
                        for training_key, training_value in value.items()
                        if training_key != "cumulative_loss"
                    }
                    if key == "training"
                    else value
                )
                for key, value in payload.items()
                if key not in {"model_state", "optimizer_state"}
            }
        )
        for payload in payloads
    ]
    model = _compare_tensor_component(
        payloads,
        "model_state",
        absolute_tolerance=absolute_tolerance,
    )
    optimizer = _compare_tensor_component(
        payloads,
        "optimizer_state",
        absolute_tolerance=absolute_tolerance,
    )
    metadata_exact = len(set(metadata_hashes)) == 1
    return {
        "absolute_tolerance": absolute_tolerance,
        "checkpoint_count": len(paths),
        "exact_state_hashes_equal": len(set(state_hashes)) == 1,
        "metadata_exact": metadata_exact,
        "training_loss_absolute_tolerance": training_loss_absolute_tolerance,
        "model": model,
        "optimizer": optimizer,
        "training_loss_max_absolute_difference": training_loss_range,
        "training_loss_within_absolute_tolerance": (
            training_loss_range <= training_loss_absolute_tolerance
        ),
        "within_absolute_tolerance": (
            metadata_exact
            and bool(model["within_absolute_tolerance"])
            and bool(optimizer["within_absolute_tolerance"])
            and training_loss_range <= training_loss_absolute_tolerance
        ),
    }


def _run_training_process(
    spec: CalibrationRunSpec,
    *,
    train_script: Path,
    data_root: Path,
    device: str,
    mode: str,
    token_budget: int,
    eval_tokens: int | None,
    batch_size: int,
    eval_batch_size: int,
    output_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    run_root = output_root / "runs" / spec.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_root / "checkpoint.pt"
    metrics_path = run_root / "metrics.jsonl"
    command = [
        sys.executable,
        str(train_script),
        "train",
        "--mode",
        mode,
        "--data-root",
        str(data_root),
        "--device",
        device,
        "--seed",
        str(spec.seed),
        "--token-budget",
        str(token_budget),
    ]
    if eval_tokens is not None:
        command.extend(["--eval-tokens", str(eval_tokens)])
    command.extend(
        [
            "--batch-size",
            str(batch_size),
            "--eval-batch-size",
            str(eval_batch_size),
            "--log-every-tokens",
            str(token_budget),
            "--checkpoint-every-tokens",
            str(token_budget),
            "--checkpoint-out",
            str(checkpoint_path),
            "--metrics-file",
            str(metrics_path),
            "--no-generate",
            "--deterministic",
        ]
    )
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(spec.seed)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=train_script.parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise CalibrationError(f"{spec.run_id} exceeded {timeout_seconds} seconds") from error
    process_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        failure_path = run_root / "failure.log"
        failure_path.write_text(completed.stderr + completed.stdout, encoding="utf-8")
        raise CalibrationError(
            f"{spec.run_id} failed with exit code {completed.returncode}; see {failure_path}"
        )

    records = _load_metric_events(metrics_path)
    config = _single_event(records, "config")
    summary = _single_event(records, "summary")
    if summary.get("validation_bpb") is None:
        raise CalibrationError(f"{spec.run_id} did not report validation BPB")
    if int(summary["tokens_seen"]) != token_budget:
        raise CalibrationError(f"{spec.run_id} did not consume the declared token budget")
    result = {
        **asdict(spec),
        "_checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "checkpoint_state_sha256": summary["checkpoint_state_sha256"],
        "data_config_sha256": config["data_config_sha256"],
        "data_order_sha256": summary["data_order_sha256"],
        "deterministic": config["deterministic"],
        "device": config["device"],
        "device_memory_peak_kind": summary["device_memory_peak_kind"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "eval_tokens": config["eval_tokens"],
        "evaluation_seconds": summary["evaluation_seconds"],
        "evaluation_tokens_per_second": summary["evaluation_tokens_per_second"],
        "mean_train_loss": summary["mean_train_loss"],
        "metrics_path": str(Path("runs") / spec.run_id / "metrics.jsonl"),
        "parameter_count": config["parameter_count"],
        "peak_device_allocated_bytes": summary["peak_device_allocated_bytes"],
        "peak_device_reserved_bytes": summary["peak_device_reserved_bytes"],
        "peak_process_rss_bytes": summary["peak_process_rss_bytes"],
        "process_seconds": process_seconds,
        "tokenizer_sha256": config["tokenizer_sha256"],
        "target_tokens": config["target_tokens"],
        "tokens_seen": summary["tokens_seen"],
        "training_seconds": summary["training_seconds"],
        "training_tokens_per_second": summary["training_tokens_per_second"],
        "training_tokens_this_process": summary["training_tokens_this_process"],
        "validation_bpb": summary["validation_bpb"],
    }
    return result


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("calibration samples must be finite")
    mean = statistics.fmean(values)
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    standard_deviation = math.sqrt(variance)
    return {
        "count": len(values),
        "maximum": max(values),
        "mean": mean,
        "minimum": min(values),
        "sample_standard_deviation": standard_deviation,
        "sample_variance": variance,
        "coefficient_of_variation": standard_deviation / abs(mean) if mean else 0.0,
    }


def _metric_summary(runs: list[dict[str, Any]], key: str) -> dict[str, float | int] | None:
    values = [float(run[key]) for run in runs if run.get(key) is not None]
    return summarize_values(values) if values else None


def build_report(
    runs: list[dict[str, Any]],
    *,
    repeat_seed: int,
    run_order_seed: int,
    mode: str,
    token_budget: int,
    eval_tokens: int | None,
    batch_size: int,
    eval_batch_size: int,
    trainer_sha256: str,
    checkpoint_reproducibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    same_seed = sorted(
        (run for run in runs if run["group"] == "same_seed"),
        key=lambda run: int(run["replicate"]),
    )
    additional = [run for run in runs if run["group"] == "additional_seed"]
    if len(same_seed) < 2 or not additional:
        raise CalibrationError("report requires repeated and additional-seed runs")
    distinct_seed = [same_seed[0], *additional]

    same_order = len({run["data_order_sha256"] for run in same_seed}) == 1
    same_checkpoint_exact = len({run["checkpoint_state_sha256"] for run in same_seed}) == 1
    if checkpoint_reproducibility is None:
        checkpoint_reproducibility = {
            "absolute_tolerance": 0.0,
            "checkpoint_count": len(same_seed),
            "exact_state_hashes_equal": same_checkpoint_exact,
            "metadata_exact": same_checkpoint_exact,
            "training_loss_absolute_tolerance": 0.0,
            "training_loss_max_absolute_difference": 0.0,
            "training_loss_within_absolute_tolerance": same_checkpoint_exact,
            "within_absolute_tolerance": same_checkpoint_exact,
        }
    same_seed_bpb_values = [float(run["validation_bpb"]) for run in same_seed]
    same_bpb_exact = len(set(same_seed_bpb_values)) == 1
    same_bpb_range = max(same_seed_bpb_values) - min(same_seed_bpb_values)
    same_bpb_within_tolerance = same_bpb_range <= BPB_ABSOLUTE_TOLERANCE
    distinct_orders = len({run["data_order_sha256"] for run in distinct_seed}) == len(distinct_seed)
    exact_budgets = all(
        int(run["target_tokens"]) == token_budget
        and int(run["tokens_seen"]) == token_budget
        and int(run["training_tokens_this_process"]) == token_budget
        for run in runs
    )
    same_data_contract = (
        len({run["data_config_sha256"] for run in runs}) == 1
        and len({run["tokenizer_sha256"] for run in runs}) == 1
    )
    same_device = len({run["device"] for run in runs}) == 1
    same_parameter_count = len({int(run["parameter_count"]) for run in runs}) == 1
    deterministic = all(bool(run["deterministic"]) for run in runs)
    same_evaluation_contract = all(run["eval_tokens"] == eval_tokens for run in runs)
    finite_outcomes = all(
        math.isfinite(float(run[key]))
        for run in runs
        for key in (
            "elapsed_seconds",
            "evaluation_tokens_per_second",
            "mean_train_loss",
            "peak_process_rss_bytes",
            "training_tokens_per_second",
            "validation_bpb",
        )
    )
    checks = {
        "different_seeds_change_data_order": distinct_orders,
        "every_run_consumed_exact_budget": exact_budgets,
        "every_run_used_deterministic_algorithms": deterministic,
        "every_run_used_same_data_contract": same_data_contract,
        "every_run_used_same_device": same_device,
        "every_run_used_same_evaluation_contract": same_evaluation_contract,
        "every_run_used_same_parameter_count": same_parameter_count,
        "outcomes_are_finite": finite_outcomes,
        "same_seed_reproduces_checkpoint_within_tolerance": bool(
            checkpoint_reproducibility["within_absolute_tolerance"]
        ),
        "same_seed_reproduces_data_order": same_order,
        "same_seed_reproduces_validation_bpb_within_tolerance": same_bpb_within_tolerance,
    }

    metric_keys = (
        "validation_bpb",
        "mean_train_loss",
        "training_tokens_per_second",
        "evaluation_tokens_per_second",
        "peak_process_rss_bytes",
        "peak_device_allocated_bytes",
        "peak_device_reserved_bytes",
    )
    same_statistics = {key: _metric_summary(same_seed, key) for key in metric_keys}
    distinct_statistics = {key: _metric_summary(distinct_seed, key) for key in metric_keys}
    execution_variance = float(same_statistics["validation_bpb"]["sample_variance"])
    observed_seed_variance = float(distinct_statistics["validation_bpb"]["sample_variance"])
    seed_variance = max(observed_seed_variance - execution_variance, 0.0)

    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "contract": {
            "batch_size": batch_size,
            "data_config_sha256": runs[0]["data_config_sha256"],
            "deterministic": True,
            "device": runs[0]["device"],
            "eval_batch_size": eval_batch_size,
            "eval_tokens": eval_tokens,
            "mode": mode,
            "parameter_count": runs[0]["parameter_count"],
            "repeat_seed": repeat_seed,
            "run_order_seed": run_order_seed,
            "token_budget": token_budget,
            "tokenizer_sha256": runs[0]["tokenizer_sha256"],
            "trainer_sha256": trainer_sha256,
            "validation_bpb_absolute_tolerance": BPB_ABSOLUTE_TOLERANCE,
            "validation_bpb_tolerance_basis": "calibrated_same_seed_execution_noise",
        },
        "environment": {
            "machine": platform.machine(),
            "platform": platform.system(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "checkpoint_reproducibility": checkpoint_reproducibility,
        "noise": {
            "execution_noise_bpb_sample_variance": execution_variance,
            "observed_distinct_seed_bpb_sample_variance": observed_seed_variance,
            "estimated_seed_bpb_variance": seed_variance,
            "estimated_seed_bpb_standard_deviation": math.sqrt(seed_variance),
        },
        "same_seed": {
            "exact_validation_bpb_equal": same_bpb_exact,
            "run_count": len(same_seed),
            "seed": repeat_seed,
            "statistics": same_statistics,
            "validation_bpb_max_absolute_difference": same_bpb_range,
        },
        "distinct_seed": {
            "run_count": len(distinct_seed),
            "seeds": [int(run["seed"]) for run in distinct_seed],
            "statistics": distinct_statistics,
        },
        "runs": runs,
    }


def _format_number(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if value and abs(value) < 10 ** (-digits):
        return f"{value:.6e}"
    return f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    contract = report["contract"]
    noise = report["noise"]
    evaluation_budget = (
        f"{contract['eval_tokens']:,}"
        if contract["eval_tokens"] is not None
        else "complete dev split"
    )
    lines = [
        "# Local baseline calibration",
        "",
        f"Mode: `{contract['mode']}`; device: `{contract['device']}`; "
        f"training tokens per run: {contract['token_budget']:,}; "
        f"evaluation tokens per run: {evaluation_budget}.",
        "",
        "Absolute tolerances: "
        f"BPB `{contract['validation_bpb_absolute_tolerance']:.0e}`; "
        f"checkpoint tensors `{report['checkpoint_reproducibility']['absolute_tolerance']:.0e}`; "
        "checkpoint training loss "
        f"`{report['checkpoint_reproducibility']['training_loss_absolute_tolerance']:.0e}`.",
        "",
        "| Run | Seed | Group | Dev BPB | Train tok/s | Peak process MiB | State hash |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for run in report["runs"]:
        peak_mib = float(run["peak_process_rss_bytes"]) / (1024 * 1024)
        lines.append(
            f"| {run['run_id']} | {run['seed']} | {run['group']} | "
            f"{float(run['validation_bpb']):.9f} | "
            f"{float(run['training_tokens_per_second']):.1f} | {peak_mib:.1f} | "
            f"`{str(run['checkpoint_state_sha256'])[:12]}` |"
        )
    lines.extend(["", "## Determinism checks", ""])
    for name, passed in report["checks"].items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")
    lines.append(
        "- `same_seed_validation_bpb_exact`: "
        f"{'pass' if report['same_seed']['exact_validation_bpb_equal'] else 'diagnostic drift'}"
    )
    lines.append(
        "- `same_seed_validation_bpb_max_absolute_difference`: "
        f"{float(report['same_seed']['validation_bpb_max_absolute_difference']):.12g}"
    )
    checkpoint = report["checkpoint_reproducibility"]
    lines.append(
        "- `same_seed_checkpoint_exact_hashes`: "
        f"{'pass' if checkpoint['exact_state_hashes_equal'] else 'diagnostic drift'}"
    )
    if "model" in checkpoint:
        lines.append(
            "- `same_seed_model_max_absolute_difference`: "
            f"{float(checkpoint['model']['maximum_absolute_difference']):.12g}"
        )
        lines.append(
            "- `same_seed_optimizer_max_absolute_difference`: "
            f"{float(checkpoint['optimizer']['maximum_absolute_difference']):.12g}"
        )
        lines.append(
            "- `same_seed_training_loss_max_absolute_difference`: "
            f"{float(checkpoint['training_loss_max_absolute_difference']):.12g}"
        )
    lines.extend(
        [
            "",
            "## Noise estimates",
            "",
            "| Quantity | Estimate |",
            "| --- | ---: |",
            "| Same-seed execution BPB variance | "
            f"{_format_number(noise['execution_noise_bpb_sample_variance'], 12)} |",
            "| Observed distinct-seed BPB variance | "
            f"{_format_number(noise['observed_distinct_seed_bpb_sample_variance'], 12)} |",
            "| Estimated seed BPB variance | "
            f"{_format_number(noise['estimated_seed_bpb_variance'], 12)} |",
            "| Estimated seed BPB standard deviation | "
            f"{_format_number(noise['estimated_seed_bpb_standard_deviation'], 9)} |",
            "",
            "The seed component is the non-negative difference between distinct-seed and "
            "same-seed sample variances. It is a baseline planning estimate, not a model-quality "
            "claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate baseline execution and seed noise.")
    parser.add_argument("--data-root", type=_path, default=default_output_root())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=tuple(CALIBRATION_MODES), default="cheap")
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--eval-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--repeat-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument(
        "--additional-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ADDITIONAL_SEEDS),
    )
    parser.add_argument("--run-order-seed", type=int, default=DEFAULT_RUN_ORDER_SEED)
    parser.add_argument("--output-root", type=_path, default=Path("artifacts/calibration"))
    parser.add_argument("--json-report", type=_path)
    parser.add_argument("--markdown-report", type=_path)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    return parser


def run_calibration(args: argparse.Namespace) -> int:
    mode = CALIBRATION_MODES[args.mode]
    token_budget = args.token_budget if args.token_budget is not None else mode["target_tokens"]
    eval_tokens = args.eval_tokens if args.eval_tokens is not None else mode["eval_tokens"]
    if token_budget <= 0:
        raise CalibrationError("calibration requires a positive training budget")
    if eval_tokens is not None and eval_tokens <= 0:
        raise CalibrationError("calibration requires a positive evaluation budget")
    if args.batch_size <= 0 or args.eval_batch_size <= 0 or args.timeout_seconds <= 0:
        raise CalibrationError("batch sizes and timeout must be positive")

    train_script = Path(__file__).resolve().parents[1] / "train.py"
    if not train_script.is_file():
        raise CalibrationError(f"training entry point does not exist: {train_script}")
    trainer_sha256 = _sha256(train_script)
    output_root = prepare_output_root(args.output_root, overwrite=args.overwrite)
    json_path = (args.json_report or output_root / "report.json").resolve()
    markdown_path = (args.markdown_report or output_root / "report.md").resolve()
    if json_path == markdown_path:
        raise CalibrationError("JSON and Markdown reports must use different paths")

    specs = build_run_specs(
        repeat_seed=args.repeat_seed,
        repeat_count=args.repeat_count,
        additional_seeds=list(args.additional_seeds),
        run_order_seed=args.run_order_seed,
    )
    runs: list[dict[str, Any]] = []
    repeated_checkpoint_paths: list[Path] = []
    for execution_index, spec in enumerate(specs, start=1):
        print(
            json.dumps(
                {
                    "event": "calibration_run_started",
                    "execution_index": execution_index,
                    "run_count": len(specs),
                    **asdict(spec),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result = _run_training_process(
            spec,
            train_script=train_script,
            data_root=args.data_root.resolve(),
            device=args.device,
            mode=args.mode,
            token_budget=token_budget,
            eval_tokens=eval_tokens,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            output_root=output_root,
            timeout_seconds=args.timeout_seconds,
        )
        checkpoint_path = Path(result.pop("_checkpoint_path"))
        if spec.group == "same_seed":
            repeated_checkpoint_paths.append(checkpoint_path)
        elif not args.keep_checkpoints:
            checkpoint_path.unlink()
        result["execution_index"] = execution_index
        runs.append(result)
        print(
            json.dumps(
                {
                    "event": "calibration_run_completed",
                    "execution_index": execution_index,
                    "run_id": spec.run_id,
                    "seed": spec.seed,
                    "validation_bpb": result["validation_bpb"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if _sha256(train_script) != trainer_sha256:
        raise CalibrationError("trainer changed while calibration was running")
    checkpoint_reproducibility = compare_checkpoints(repeated_checkpoint_paths)
    report = build_report(
        runs,
        repeat_seed=args.repeat_seed,
        run_order_seed=args.run_order_seed,
        mode=args.mode,
        token_budget=token_budget,
        eval_tokens=eval_tokens,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        trainer_sha256=trainer_sha256,
        checkpoint_reproducibility=checkpoint_reproducibility,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    if not args.keep_checkpoints:
        for checkpoint_path in repeated_checkpoint_paths:
            checkpoint_path.unlink()
    print(
        json.dumps(
            {
                "all_checks_passed": report["all_checks_passed"],
                "event": "calibration_completed",
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
                "run_count": len(runs),
            },
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_calibration(args)
    except (CalibrationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
