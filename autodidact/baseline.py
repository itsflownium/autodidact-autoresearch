"""Run, resume, verify, and summarize the retained full-budget baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from autodidact.checkpoints import file_sha256
from autodidact.data.config import default_output_root

BASELINE_SCHEMA_VERSION = 1
BASELINE_OUTPUT_MARKER = ".autodidact-full-baseline"
BASELINE_CONTRACT_FILE = "contract.json"
DEFAULT_OUTPUT_ROOT = Path("artifacts/baseline/full-v1")
DEFAULT_SEEDS = (1_337, 2_027, 4_099)
EXPECTED_PARAMETER_COUNT = 1_016_960
FULL_TOKEN_BUDGET = 20_000_000
FULL_CHECKPOINT_INTERVAL = 5_000_000
FULL_LOG_INTERVAL = 500_000
MAX_PYTHON_HASH_SEED = 2**32 - 1


class BaselineError(RuntimeError):
    """Raised when a full-baseline artifact cannot be trusted or resumed."""


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _validate_seeds(seeds: list[int]) -> None:
    if len(seeds) < 2:
        raise BaselineError("the full baseline requires at least two seeds")
    if len(set(seeds)) != len(seeds):
        raise BaselineError("baseline seeds must be unique")
    if any(seed < 0 or seed > MAX_PYTHON_HASH_SEED for seed in seeds):
        raise BaselineError(f"seeds must be between 0 and {MAX_PYTHON_HASH_SEED}")


def prepare_output_root(
    path: Path,
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[Path, bool]:
    output_root = path.resolve()
    marker = output_root / BASELINE_OUTPUT_MARKER
    if output_root.exists():
        if not marker.is_file():
            raise BaselineError(
                f"refusing to use an unmarked full-baseline directory: {output_root}"
            )
        if overwrite:
            shutil.rmtree(output_root)
        elif resume:
            return output_root, False
        else:
            raise BaselineError(
                f"output root already exists: {output_root}; use --resume or --overwrite"
            )
    elif resume:
        raise BaselineError(f"cannot resume missing output root: {output_root}")

    output_root.mkdir(parents=True)
    _atomic_write_text(
        output_root / BASELINE_OUTPUT_MARKER,
        f"schema_version={BASELINE_SCHEMA_VERSION}\n",
    )
    return output_root, True


def _local_contract(
    *,
    seeds: list[int],
    token_budget: int,
    eval_tokens: int | None,
    data_root: Path,
    device: str,
    batch_size: int,
    eval_batch_size: int,
    prompt: str,
    generate_tokens: int,
    trainer_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "data_root": str(data_root.resolve()),
        "deterministic": True,
        "device_requested": device,
        "eval_batch_size": eval_batch_size,
        "eval_tokens": eval_tokens,
        "generate_tokens": generate_tokens,
        "mode": "full",
        "prompt": prompt,
        "runner_sha256": runner_sha256,
        "schema_version": BASELINE_SCHEMA_VERSION,
        "seeds": seeds,
        "temperature": 0.0,
        "token_budget": token_budget,
        "trainer_sha256": trainer_sha256,
    }


def establish_contract(
    output_root: Path,
    contract: dict[str, Any],
    *,
    created: bool,
) -> None:
    path = output_root / BASELINE_CONTRACT_FILE
    if created:
        _write_json(path, contract)
        return
    if not path.is_file():
        raise BaselineError("resumable baseline is missing its contract")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != contract:
        changed = sorted(
            key for key in set(existing) | set(contract) if existing.get(key) != contract.get(key)
        )
        raise BaselineError(f"resume contract mismatch: {', '.join(changed)}")


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise BaselineError(f"invalid metrics JSON at {path}:{line_number}") from error
    return records


def _last_event(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("event") == name]
    if not matches:
        raise BaselineError(f"metrics do not contain a {name} event")
    return matches[-1]


def _run_paths(output_root: Path, seed: int) -> tuple[Path, Path, Path, Path]:
    run_root = output_root / "runs" / f"seed-{seed}"
    return (
        run_root,
        run_root / "checkpoint.pt",
        run_root / "metrics.jsonl",
        run_root / "processes.jsonl",
    )


def _is_complete(metrics_path: Path, checkpoint_path: Path, token_budget: int) -> bool:
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        return False
    records = _load_events(metrics_path)
    summaries = [record for record in records if record.get("event") == "summary"]
    if not summaries:
        return False
    summary = summaries[-1]
    return (
        int(summary.get("tokens_seen", -1)) == token_budget
        and int(summary.get("target_tokens", -1)) == token_budget
        and summary.get("checkpoint_sha256") == file_sha256(checkpoint_path)
    )


def _training_command(
    *,
    train_script: Path,
    data_root: Path,
    device: str,
    seed: int,
    token_budget: int,
    eval_tokens: int | None,
    batch_size: int,
    eval_batch_size: int,
    prompt: str,
    generate_tokens: int,
    checkpoint_path: Path,
    metrics_path: Path,
    resume_checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(train_script),
        "train",
        "--mode",
        "full",
        "--data-root",
        str(data_root),
        "--device",
        device,
        "--seed",
        str(seed),
        "--token-budget",
        str(token_budget),
        "--batch-size",
        str(batch_size),
        "--eval-batch-size",
        str(eval_batch_size),
        "--log-every-tokens",
        str(min(FULL_LOG_INTERVAL, token_budget)),
        "--checkpoint-every-tokens",
        str(min(FULL_CHECKPOINT_INTERVAL, token_budget)),
        "--checkpoint-out",
        str(checkpoint_path),
        "--metrics-file",
        str(metrics_path),
        "--prompt",
        prompt,
        "--generate-tokens",
        str(generate_tokens),
        "--temperature",
        "0",
        "--deterministic",
    ]
    if eval_tokens is not None:
        command.extend(["--eval-tokens", str(eval_tokens)])
    if resume_checkpoint is not None:
        command.extend(["--resume", str(resume_checkpoint)])
    return command


def execute_seed(
    *,
    output_root: Path,
    train_script: Path,
    data_root: Path,
    device: str,
    seed: int,
    token_budget: int,
    eval_tokens: int | None,
    batch_size: int,
    eval_batch_size: int,
    prompt: str,
    generate_tokens: int,
    timeout_seconds: int,
    resume: bool,
) -> bool:
    run_root, checkpoint_path, metrics_path, processes_path = _run_paths(output_root, seed)
    if _is_complete(metrics_path, checkpoint_path, token_budget):
        return True
    run_root.mkdir(parents=True, exist_ok=True)

    resume_checkpoint: Path | None = None
    if metrics_path.exists() or checkpoint_path.exists():
        if not resume:
            raise BaselineError(f"seed {seed} has incomplete artifacts; rerun with --resume")
        if not metrics_path.is_file() or not checkpoint_path.is_file():
            raise BaselineError(
                f"seed {seed} cannot resume without both metrics and checkpoint artifacts"
            )
        resume_checkpoint = checkpoint_path

    command = _training_command(
        train_script=train_script,
        data_root=data_root,
        device=device,
        seed=seed,
        token_budget=token_budget,
        eval_tokens=eval_tokens,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        prompt=prompt,
        generate_tokens=generate_tokens,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        resume_checkpoint=resume_checkpoint,
    )
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(seed)
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
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
        elapsed = time.perf_counter() - started
        _append_jsonl(
            processes_path,
            {
                "elapsed_seconds": elapsed,
                "exit": "timeout",
                "resume": resume_checkpoint is not None,
                "started_at": started_at,
            },
        )
        raise BaselineError(f"seed {seed} exceeded {timeout_seconds} seconds") from error

    elapsed = time.perf_counter() - started
    _append_jsonl(
        processes_path,
        {
            "elapsed_seconds": elapsed,
            "exit": completed.returncode,
            "resume": resume_checkpoint is not None,
            "started_at": started_at,
        },
    )
    if completed.returncode != 0:
        _atomic_write_text(run_root / "failure.log", completed.stderr + completed.stdout)
        raise BaselineError(
            f"seed {seed} failed with exit code {completed.returncode}; "
            f"see {run_root / 'failure.log'}"
        )
    if not _is_complete(metrics_path, checkpoint_path, token_budget):
        raise BaselineError(f"seed {seed} completed without a valid final artifact")
    return False


def collect_run(output_root: Path, seed: int, token_budget: int) -> dict[str, Any]:
    _run_root, checkpoint_path, metrics_path, processes_path = _run_paths(output_root, seed)
    if not _is_complete(metrics_path, checkpoint_path, token_budget):
        raise BaselineError(f"seed {seed} is not complete")
    records = _load_events(metrics_path)
    configs = [record for record in records if record.get("event") == "config"]
    if not configs:
        raise BaselineError(f"seed {seed} has no config event")
    config = configs[-1]
    for earlier in configs[:-1]:
        for key in (
            "data_config_sha256",
            "deterministic",
            "device",
            "eval_tokens",
            "model",
            "parameter_count",
            "seed",
            "target_tokens",
            "tokenizer_sha256",
        ):
            if earlier.get(key) != config.get(key):
                raise BaselineError(f"seed {seed} changed {key} across resume segments")

    evaluation = _last_event(records, "evaluation")
    generation = _last_event(records, "generation")
    summary = _last_event(records, "summary")
    process_records = _load_events(processes_path)
    checkpoint_hash = file_sha256(checkpoint_path)
    if summary["checkpoint_sha256"] != checkpoint_hash:
        raise BaselineError(f"seed {seed} checkpoint hash does not match metrics")

    relative_checkpoint = checkpoint_path.relative_to(output_root).as_posix()
    relative_metrics = metrics_path.relative_to(output_root).as_posix()
    return {
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_path": relative_checkpoint,
        "checkpoint_sha256": checkpoint_hash,
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
        "generated_text": generation["text"],
        "mean_train_loss": summary["mean_train_loss"],
        "metrics_path": relative_metrics,
        "parameter_count": config["parameter_count"],
        "peak_device_allocated_bytes": summary["peak_device_allocated_bytes"],
        "peak_device_reserved_bytes": summary["peak_device_reserved_bytes"],
        "peak_process_rss_bytes": summary["peak_process_rss_bytes"],
        "predicted_tokens": evaluation["predicted_tokens"],
        "process_attempts": len(process_records),
        "process_seconds": sum(float(record["elapsed_seconds"]) for record in process_records),
        "resume_segments": len(configs) - 1,
        "seed": seed,
        "steps": summary["steps"],
        "stories": evaluation["stories"],
        "target_tokens": config["target_tokens"],
        "tokenizer_sha256": config["tokenizer_sha256"],
        "tokens_seen": summary["tokens_seen"],
        "training_seconds": summary["training_seconds"],
        "training_tokens_per_second": summary["training_tokens_per_second"],
        "training_tokens_this_process": summary["training_tokens_this_process"],
        "utf8_bytes": evaluation["utf8_bytes"],
        "validation_bpb": summary["validation_bpb"],
    }


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise BaselineError("cannot summarize missing or non-finite values")
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "sample_standard_deviation": math.sqrt(variance),
        "sample_variance": variance,
    }


def build_report(
    runs: list[dict[str, Any]],
    *,
    seeds: list[int],
    token_budget: int,
    eval_tokens: int | None,
    batch_size: int,
    eval_batch_size: int,
    prompt: str,
    generate_tokens: int,
    trainer_sha256: str,
    runner_sha256: str,
) -> dict[str, Any]:
    if len(runs) != len(seeds):
        raise BaselineError("report does not contain every declared seed")
    same_data = len({run["data_config_sha256"] for run in runs}) == 1
    same_tokenizer = len({run["tokenizer_sha256"] for run in runs}) == 1
    same_device = len({run["device"] for run in runs}) == 1
    same_parameter_count = len({int(run["parameter_count"]) for run in runs}) == 1
    same_evaluation_set = (
        len(
            {
                (int(run["predicted_tokens"]), int(run["stories"]), int(run["utf8_bytes"]))
                for run in runs
            }
        )
        == 1
    )
    checks = {
        "all_checkpoints_retained_and_hashed": all(
            len(str(run["checkpoint_sha256"])) == 64 and int(run["checkpoint_bytes"]) > 0
            for run in runs
        ),
        "all_generated_samples_present": all(
            bool(str(run["generated_text"]).strip()) for run in runs
        ),
        "all_outcomes_finite": all(
            math.isfinite(float(run[key]))
            for run in runs
            for key in (
                "mean_train_loss",
                "peak_process_rss_bytes",
                "training_tokens_per_second",
                "validation_bpb",
            )
        ),
        "all_runs_consumed_exact_budget": all(
            int(run["target_tokens"]) == token_budget and int(run["tokens_seen"]) == token_budget
            for run in runs
        ),
        "all_runs_used_deterministic_algorithms": all(bool(run["deterministic"]) for run in runs),
        "all_runs_used_same_data": same_data and same_tokenizer,
        "all_runs_used_same_device": same_device,
        "all_runs_used_same_evaluation_set": same_evaluation_set,
        "all_runs_used_same_parameter_count": same_parameter_count,
        "all_runs_used_expected_parameter_count": all(
            int(run["parameter_count"]) == EXPECTED_PARAMETER_COUNT for run in runs
        ),
        "data_orders_are_seed_specific": (
            len({run["data_order_sha256"] for run in runs}) == len(runs)
        ),
        "declared_seeds_are_complete": [int(run["seed"]) for run in runs] == seeds,
        "every_run_has_a_process_record": all(int(run["process_attempts"]) >= 1 for run in runs),
    }
    metric_keys = (
        "validation_bpb",
        "mean_train_loss",
        "training_tokens_per_second",
        "evaluation_tokens_per_second",
        "peak_process_rss_bytes",
        "peak_device_allocated_bytes",
        "peak_device_reserved_bytes",
        "checkpoint_bytes",
        "process_seconds",
    )
    statistics_by_metric: dict[str, dict[str, float | int] | None] = {}
    for key in metric_keys:
        values = [float(run[key]) for run in runs if run.get(key) is not None]
        statistics_by_metric[key] = summarize_values(values) if values else None
    diagnostic_override = token_budget != FULL_TOKEN_BUDGET or eval_tokens is not None
    interpretation = {
        "checkpoint_retention": "retained_for_integrity_and_resume",
        "generated_samples": (
            "model_quality_evidence" if not diagnostic_override else "generation_path_check_only"
        ),
        "model_quality": "full_budget_reference" if not diagnostic_override else "not_estimated",
        "performance_metrics": (
            "full_budget_reference" if not diagnostic_override else "provisional_order_sensitive"
        ),
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "complete_full_baseline": all(checks.values()) and not diagnostic_override,
        "contract": {
            "batch_size": batch_size,
            "data_config_sha256": runs[0]["data_config_sha256"],
            "deterministic": True,
            "device": runs[0]["device"],
            "eval_batch_size": eval_batch_size,
            "eval_tokens": eval_tokens,
            "generate_tokens": generate_tokens,
            "mode": "full",
            "parameter_count": runs[0]["parameter_count"],
            "prompt": prompt,
            "runner_sha256": runner_sha256,
            "seeds": seeds,
            "token_budget": token_budget,
            "tokenizer_sha256": runs[0]["tokenizer_sha256"],
            "trainer_sha256": trainer_sha256,
        },
        "created_at": datetime.now(UTC).isoformat(),
        "diagnostic_override": diagnostic_override,
        "environment": {
            "machine": platform.machine(),
            "platform": platform.system(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "interpretation": interpretation,
        "runs": runs,
        "schema_version": BASELINE_SCHEMA_VERSION,
        "statistics": statistics_by_metric,
    }


def _format_number(value: float | int, digits: int = 6) -> str:
    if isinstance(value, int):
        return str(value)
    if value and abs(value) < 10 ** (-digits):
        return f"{value:.6e}"
    return f"{value:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    contract = report["contract"]
    title = "Full Baseline" if report["complete_full_baseline"] else "Baseline Diagnostic"
    evaluation_budget = (
        f"{contract['eval_tokens']:,} tokens"
        if contract["eval_tokens"] is not None
        else "complete public-dev split"
    )
    lines = [
        f"# {title}",
        "",
        (
            "Classification: **complete full baseline**."
            if report["complete_full_baseline"]
            else "Classification: **diagnostic only; not a complete full baseline**."
        ),
        "",
        f"Mode: `full`; device: `{contract['device']}`; seeds: "
        + ", ".join(f"`{seed}`" for seed in contract["seeds"])
        + ".",
        "",
        f"Training budget per seed: {contract['token_budget']:,} tokens. "
        f"Evaluation: {evaluation_budget}.",
        "",
        "| Seed | Steps | Dev BPB | Train tok/s | Peak RSS MiB | "
        "Checkpoint MiB | Resume segments |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report["runs"]:
        steps = run.get("steps", "not recorded")
        lines.append(
            f"| {run['seed']} | {steps} | {float(run['validation_bpb']):.9f} | "
            f"{float(run['training_tokens_per_second']):.1f} | "
            f"{float(run['peak_process_rss_bytes']) / (1024 * 1024):.1f} | "
            f"{float(run['checkpoint_bytes']) / (1024 * 1024):.1f} | "
            f"{run['resume_segments']} |"
        )
    lines.extend(["", "## Verification", ""])
    for name, passed in report["checks"].items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")
    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Metric | Mean | Sample standard deviation | Minimum | Maximum |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in (
        "validation_bpb",
        "training_tokens_per_second",
        "evaluation_tokens_per_second",
        "peak_process_rss_bytes",
    ):
        summary = report["statistics"][key]
        if summary is None:
            continue
        scale = 1 / (1024 * 1024) if key == "peak_process_rss_bytes" else 1
        label = "peak_process_rss_mib" if key == "peak_process_rss_bytes" else key
        lines.append(
            f"| {label} | {_format_number(float(summary['mean']) * scale)} | "
            f"{_format_number(float(summary['sample_standard_deviation']) * scale)} | "
            f"{_format_number(float(summary['minimum']) * scale)} | "
            f"{_format_number(float(summary['maximum']) * scale)} |"
        )
    lines.extend(["", "## Deterministic Samples", ""])
    for run in report["runs"]:
        sample = str(run["generated_text"]).replace("\n", " ").strip()
        lines.extend([f"Seed `{run['seed']}`:", "", f"> {sample}", ""])
    if report["diagnostic_override"]:
        step_counts = sorted(
            {int(run["steps"]) for run in report["runs"] if run.get("steps") is not None}
        )
        step_description = (
            ", ".join(str(value) for value in step_counts) if step_counts else "not recorded"
        )
        checkpoint_mib = sum(int(run["checkpoint_bytes"]) for run in report["runs"]) / (1024 * 1024)
        lines.extend(
            [
                "## Diagnostic Limitations",
                "",
                f"Optimizer steps per seed: {step_description}.",
                "",
                (
                    "Short diagnostic timings and memory peaks can be dominated by device/compiler "
                    "warm-up, caching, and run order. They are provisional plumbing measurements, "
                    "not stable performance comparisons."
                ),
                "",
                (
                    "Generated samples confirm that checkpoint loading and generation execute. "
                    "They are not model-quality evidence at a diagnostic training budget."
                ),
                "",
                (
                    f"Verified checkpoints are retained for integrity and resume testing "
                    f"({checkpoint_mib:.1f} MiB total). The marked diagnostic output directory is "
                    "self-contained and may be removed after its evidence is no longer needed."
                ),
                "",
                (
                    "This is an unmodified-parent diagnostic. It does not establish a full-budget "
                    "reference or claim a patch improvement."
                ),
            ]
        )
    else:
        lines.append(
            "This is the unmodified parent baseline. It establishes a full-budget reference and "
            "does not claim a patch improvement."
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and verify retained full-budget baseline checkpoints."
    )
    parser.add_argument("--data-root", type=_path, default=default_output_root())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--token-budget", type=int, default=FULL_TOKEN_BUDGET)
    parser.add_argument("--eval-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--generate-tokens", type=int, default=128)
    parser.add_argument("--output-root", type=_path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--json-report", type=_path)
    parser.add_argument("--markdown-report", type=_path)
    parser.add_argument("--timeout-seconds", type=int, default=7_200)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser


def run_baseline(args: argparse.Namespace) -> int:
    seeds = list(args.seeds)
    _validate_seeds(seeds)
    if args.token_budget <= 0:
        raise BaselineError("token budget must be positive")
    if args.eval_tokens is not None and args.eval_tokens <= 0:
        raise BaselineError("evaluation budget must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise BaselineError("batch sizes must be positive")
    if args.generate_tokens <= 0 or args.timeout_seconds <= 0:
        raise BaselineError("generation and timeout values must be positive")

    train_script = Path(__file__).resolve().parents[1] / "train.py"
    if not train_script.is_file():
        raise BaselineError(f"training entry point does not exist: {train_script}")
    trainer_sha256 = file_sha256(train_script)
    runner_path = Path(__file__).resolve()
    runner_sha256 = file_sha256(runner_path)
    output_root, created = prepare_output_root(
        args.output_root,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    contract = _local_contract(
        seeds=seeds,
        token_budget=args.token_budget,
        eval_tokens=args.eval_tokens,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        prompt=args.prompt,
        generate_tokens=args.generate_tokens,
        trainer_sha256=trainer_sha256,
        runner_sha256=runner_sha256,
    )
    establish_contract(output_root, contract, created=created)
    json_path = (args.json_report or output_root / "report.json").resolve()
    markdown_path = (args.markdown_report or output_root / "report.md").resolve()
    if json_path == markdown_path:
        raise BaselineError("JSON and Markdown reports must use different paths")

    runs: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        print(
            json.dumps(
                {
                    "event": "full_baseline_seed_started",
                    "run_count": len(seeds),
                    "run_index": index,
                    "seed": seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        reused = execute_seed(
            output_root=output_root,
            train_script=train_script,
            data_root=args.data_root.resolve(),
            device=args.device,
            seed=seed,
            token_budget=args.token_budget,
            eval_tokens=args.eval_tokens,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            prompt=args.prompt,
            generate_tokens=args.generate_tokens,
            timeout_seconds=args.timeout_seconds,
            resume=args.resume,
        )
        run = collect_run(output_root, seed, args.token_budget)
        runs.append(run)
        print(
            json.dumps(
                {
                    "event": "full_baseline_seed_completed",
                    "reused": reused,
                    "seed": seed,
                    "validation_bpb": run["validation_bpb"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if file_sha256(train_script) != trainer_sha256:
        raise BaselineError("trainer changed while the full baseline was running")
    if file_sha256(runner_path) != runner_sha256:
        raise BaselineError("baseline runner changed while the full baseline was running")
    report = build_report(
        runs,
        seeds=seeds,
        token_budget=args.token_budget,
        eval_tokens=args.eval_tokens,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        prompt=args.prompt,
        generate_tokens=args.generate_tokens,
        trainer_sha256=trainer_sha256,
        runner_sha256=runner_sha256,
    )
    _write_json(json_path, report)
    _atomic_write_text(markdown_path, render_markdown(report))
    print(
        json.dumps(
            {
                "all_checks_passed": report["all_checks_passed"],
                "complete_full_baseline": report["complete_full_baseline"],
                "event": "full_baseline_completed",
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
        return run_baseline(args)
    except (BaselineError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
