from __future__ import annotations

import json

import pytest

from autodidact.baseline import (
    EXPECTED_PARAMETER_COUNT,
    FULL_TOKEN_BUDGET,
    BaselineError,
    build_report,
    establish_contract,
    prepare_output_root,
    render_markdown,
    summarize_values,
)


def _run(seed: int, *, parameter_count: int = EXPECTED_PARAMETER_COUNT) -> dict[str, object]:
    return {
        "checkpoint_bytes": 10_000_000 + seed,
        "checkpoint_path": f"runs/seed-{seed}/checkpoint.pt",
        "checkpoint_sha256": f"{seed:064x}",
        "checkpoint_state_sha256": f"{seed + 1:064x}",
        "data_config_sha256": "data",
        "data_order_sha256": f"order-{seed}",
        "deterministic": True,
        "device": "mps",
        "device_memory_peak_kind": "allocator_sampled",
        "elapsed_seconds": 10.0,
        "eval_tokens": None,
        "evaluation_seconds": 2.0,
        "evaluation_tokens_per_second": 250_000.0,
        "generated_text": f"Once upon a time, seed {seed} told a story.",
        "mean_train_loss": 1.0,
        "metrics_path": f"runs/seed-{seed}/metrics.jsonl",
        "parameter_count": parameter_count,
        "peak_device_allocated_bytes": 100_000_000,
        "peak_device_reserved_bytes": 150_000_000,
        "peak_process_rss_bytes": 700_000_000,
        "predicted_tokens": 500_000,
        "process_attempts": 1,
        "process_seconds": 10.5,
        "resume_segments": 0,
        "seed": seed,
        "steps": 2_500,
        "stories": 1_000,
        "target_tokens": FULL_TOKEN_BUDGET,
        "tokenizer_sha256": "tokenizer",
        "tokens_seen": FULL_TOKEN_BUDGET,
        "training_seconds": 8.0,
        "training_tokens_per_second": 2_500_000.0,
        "training_tokens_this_process": FULL_TOKEN_BUDGET,
        "utf8_bytes": 750_000,
        "validation_bpb": 1.0 + seed / 1_000_000,
    }


def _report(runs: list[dict[str, object]]) -> dict[str, object]:
    return build_report(
        runs,
        seeds=[int(run["seed"]) for run in runs],
        token_budget=FULL_TOKEN_BUDGET,
        eval_tokens=None,
        batch_size=64,
        eval_batch_size=64,
        prompt="Once upon a time",
        generate_tokens=128,
        trainer_sha256="trainer",
        runner_sha256="runner",
    )


def test_output_root_requires_marker_for_resume_and_overwrite(tmp_path) -> None:
    output = tmp_path / "baseline"
    prepared, created = prepare_output_root(output, overwrite=False, resume=False)

    assert created is True
    assert (prepared / ".autodidact-full-baseline").is_file()
    (prepared / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(BaselineError, match="--resume or --overwrite"):
        prepare_output_root(output, overwrite=False, resume=False)

    resumed, created = prepare_output_root(output, overwrite=False, resume=True)
    assert resumed == prepared
    assert created is False

    replaced, created = prepare_output_root(output, overwrite=True, resume=False)
    assert created is True
    assert not (replaced / "stale.txt").exists()

    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    with pytest.raises(BaselineError, match="unmarked"):
        prepare_output_root(unmarked, overwrite=True, resume=False)


def test_resume_contract_must_match_exactly(tmp_path) -> None:
    contract = {"schema_version": 1, "seeds": [11, 23], "token_budget": 100}
    establish_contract(tmp_path, contract, created=True)
    establish_contract(tmp_path, contract, created=False)

    changed = dict(contract, token_budget=200)
    with pytest.raises(BaselineError, match="token_budget"):
        establish_contract(tmp_path, changed, created=False)

    stored = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))
    assert stored == contract


def test_full_report_requires_complete_matched_parent_runs() -> None:
    runs = [_run(11), _run(23), _run(37)]
    report = _report(runs)

    assert report["all_checks_passed"] is True
    assert report["complete_full_baseline"] is True
    assert report["statistics"]["validation_bpb"]["count"] == 3
    assert report["checks"]["data_orders_are_seed_specific"] is True
    markdown = render_markdown(report)
    assert "This is the unmodified parent baseline" in markdown
    assert "establishes a full-budget reference" in markdown
    assert "Diagnostic Limitations" not in markdown
    assert report["interpretation"]["model_quality"] == "full_budget_reference"
    assert "`all_runs_used_expected_parameter_count`: pass" in markdown


def test_report_rejects_wrong_model_size_and_reused_data_order() -> None:
    runs = [_run(11), _run(23, parameter_count=EXPECTED_PARAMETER_COUNT - 1)]
    runs[1]["data_order_sha256"] = runs[0]["data_order_sha256"]

    report = _report(runs)

    assert report["all_checks_passed"] is False
    assert report["complete_full_baseline"] is False
    assert report["checks"]["all_runs_used_expected_parameter_count"] is False
    assert report["checks"]["data_orders_are_seed_specific"] is False


def test_diagnostic_budget_cannot_be_labeled_full_baseline() -> None:
    runs = [_run(11), _run(23)]
    for run in runs:
        run["target_tokens"] = 1_000
        run["tokens_seen"] = 1_000

    report = build_report(
        runs,
        seeds=[11, 23],
        token_budget=1_000,
        eval_tokens=500,
        batch_size=2,
        eval_batch_size=2,
        prompt="Once",
        generate_tokens=4,
        trainer_sha256="trainer",
        runner_sha256="runner",
    )

    assert report["all_checks_passed"] is True
    assert report["diagnostic_override"] is True
    assert report["complete_full_baseline"] is False
    assert report["interpretation"] == {
        "checkpoint_retention": "retained_for_integrity_and_resume",
        "generated_samples": "generation_path_check_only",
        "model_quality": "not_estimated",
        "performance_metrics": "provisional_order_sensitive",
    }
    markdown = render_markdown(report)
    assert "Diagnostic Limitations" in markdown
    assert "not stable performance comparisons" in markdown
    assert "not model-quality evidence" in markdown
    assert "does not establish a full-budget reference" in markdown
    assert "It establishes a full-budget reference" not in markdown


def test_nullable_accelerator_memory_is_supported() -> None:
    runs = [_run(11), _run(23)]
    for run in runs:
        run["device"] = "cpu"
        run["device_memory_peak_kind"] = "process_only"
        run["peak_device_allocated_bytes"] = None
        run["peak_device_reserved_bytes"] = None

    report = _report(runs)

    assert report["all_checks_passed"] is True
    assert report["statistics"]["peak_device_allocated_bytes"] is None
    assert report["statistics"]["peak_device_reserved_bytes"] is None


def test_diagnostic_markdown_tolerates_legacy_runs_without_step_counts() -> None:
    runs = [_run(11), _run(23)]
    for run in runs:
        run["target_tokens"] = 1_000
        run["tokens_seen"] = 1_000
        del run["steps"]
    report = build_report(
        runs,
        seeds=[11, 23],
        token_budget=1_000,
        eval_tokens=500,
        batch_size=2,
        eval_batch_size=2,
        prompt="Once",
        generate_tokens=4,
        trainer_sha256="trainer",
        runner_sha256="runner",
    )

    markdown = render_markdown(report)

    assert "Optimizer steps per seed: not recorded." in markdown
    assert "| 11 | not recorded |" in markdown


def test_statistics_reject_non_finite_input() -> None:
    with pytest.raises(BaselineError, match="non-finite"):
        summarize_values([1.0, float("nan")])
