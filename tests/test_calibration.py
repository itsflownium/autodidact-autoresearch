from __future__ import annotations

import pytest
import torch

from autodidact.calibration import (
    CalibrationError,
    build_report,
    build_run_specs,
    compare_checkpoints,
    prepare_output_root,
    render_markdown,
    summarize_values,
)


def _run(
    run_id: str,
    seed: int,
    group: str,
    replicate: int,
    bpb: float,
    order_hash: str,
    state_hash: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "seed": seed,
        "group": group,
        "replicate": replicate,
        "checkpoint_sha256": f"file-{run_id}",
        "checkpoint_state_sha256": state_hash,
        "data_config_sha256": "data",
        "data_order_sha256": order_hash,
        "deterministic": True,
        "device": "mps",
        "device_memory_peak_kind": "allocator_sampled",
        "elapsed_seconds": 2.0,
        "eval_tokens": 250_000,
        "evaluation_seconds": 0.5,
        "evaluation_tokens_per_second": 200.0,
        "execution_index": replicate,
        "mean_train_loss": 1.5,
        "metrics_path": f"runs/{run_id}/metrics.jsonl",
        "parameter_count": 1_016_960,
        "peak_device_allocated_bytes": 10_000,
        "peak_device_reserved_bytes": 20_000,
        "peak_process_rss_bytes": 30_000,
        "process_seconds": 2.1,
        "tokenizer_sha256": "tokenizer",
        "target_tokens": 2_000_000,
        "tokens_seen": 2_000_000,
        "training_seconds": 1.0,
        "training_tokens_per_second": 2_000_000.0,
        "training_tokens_this_process": 2_000_000,
        "validation_bpb": bpb,
    }


def test_run_specs_are_randomized_reproducibly() -> None:
    first = build_run_specs(
        repeat_seed=11,
        repeat_count=3,
        additional_seeds=[23, 37],
        run_order_seed=101,
    )
    second = build_run_specs(
        repeat_seed=11,
        repeat_count=3,
        additional_seeds=[23, 37],
        run_order_seed=101,
    )

    assert first == second
    assert len(first) == 5
    assert {spec.seed for spec in first} == {11, 23, 37}
    assert sum(spec.group == "same_seed" for spec in first) == 3

    with pytest.raises(ValueError, match="must not include"):
        build_run_specs(
            repeat_seed=11,
            repeat_count=3,
            additional_seeds=[11, 23],
            run_order_seed=101,
        )
    with pytest.raises(ValueError, match="at most"):
        build_run_specs(
            repeat_seed=2**32,
            repeat_count=3,
            additional_seeds=[23],
            run_order_seed=101,
        )


def test_statistics_use_sample_variance() -> None:
    summary = summarize_values([1.0, 2.0, 3.0])

    assert summary["mean"] == 2.0
    assert summary["sample_variance"] == 1.0
    assert summary["sample_standard_deviation"] == 1.0


def test_report_separates_execution_noise_from_seed_noise() -> None:
    runs = [
        _run("repeat-01", 11, "same_seed", 1, 1.0, "order-11", "state-11"),
        _run("seed-23", 23, "additional_seed", 1, 0.9, "order-23", "state-23"),
        _run("repeat-03", 11, "same_seed", 3, 1.0, "order-11", "state-11"),
        _run("seed-37", 37, "additional_seed", 1, 1.1, "order-37", "state-37"),
        _run("repeat-02", 11, "same_seed", 2, 1.0, "order-11", "state-11"),
    ]
    report = build_report(
        runs,
        repeat_seed=11,
        run_order_seed=101,
        mode="cheap",
        token_budget=2_000_000,
        eval_tokens=250_000,
        batch_size=64,
        eval_batch_size=64,
        trainer_sha256="trainer",
    )

    assert report["all_checks_passed"] is True
    assert report["noise"]["execution_noise_bpb_sample_variance"] == 0.0
    assert report["noise"]["observed_distinct_seed_bpb_sample_variance"] == pytest.approx(0.01)
    assert report["noise"]["estimated_seed_bpb_variance"] == pytest.approx(0.01)
    assert report["same_seed"]["run_count"] == 3
    assert report["distinct_seed"]["run_count"] == 3
    markdown = render_markdown(report)
    assert "same_seed_reproduces_checkpoint_within_tolerance`: pass" in markdown
    assert "Estimated seed BPB variance" in markdown


def test_report_exposes_reproducibility_failure() -> None:
    runs = [
        _run("repeat-01", 11, "same_seed", 1, 1.0, "order-11", "state-a"),
        _run("repeat-02", 11, "same_seed", 2, 1.0, "order-11", "state-b"),
        _run("seed-23", 23, "additional_seed", 1, 0.9, "order-23", "state-23"),
    ]
    report = build_report(
        runs,
        repeat_seed=11,
        run_order_seed=101,
        mode="cheap",
        token_budget=2_000_000,
        eval_tokens=250_000,
        batch_size=64,
        eval_batch_size=64,
        trainer_sha256="trainer",
    )

    assert report["all_checks_passed"] is False
    assert report["checks"]["same_seed_reproduces_checkpoint_within_tolerance"] is False


def test_report_accepts_tiny_bpb_drift_and_preserves_its_variance() -> None:
    runs = [
        _run("repeat-01", 11, "same_seed", 1, 1.0, "order-11", "state-11"),
        _run("repeat-02", 11, "same_seed", 2, 1.0 + 5e-9, "order-11", "state-11"),
        _run("seed-23", 23, "additional_seed", 1, 0.9, "order-23", "state-23"),
    ]
    report = build_report(
        runs,
        repeat_seed=11,
        run_order_seed=101,
        mode="cheap",
        token_budget=2_000_000,
        eval_tokens=250_000,
        batch_size=64,
        eval_batch_size=64,
        trainer_sha256="trainer",
    )

    assert report["all_checks_passed"] is True
    assert report["same_seed"]["exact_validation_bpb_equal"] is False
    assert report["same_seed"]["validation_bpb_max_absolute_difference"] == pytest.approx(5e-9)
    assert report["noise"]["execution_noise_bpb_sample_variance"] > 0


def test_report_rejects_bpb_drift_above_calibrated_bound() -> None:
    runs = [
        _run("repeat-01", 11, "same_seed", 1, 1.0, "order-11", "state-11"),
        _run("repeat-02", 11, "same_seed", 2, 1.0 + 2e-7, "order-11", "state-11"),
        _run("seed-23", 23, "additional_seed", 1, 0.9, "order-23", "state-23"),
    ]
    report = build_report(
        runs,
        repeat_seed=11,
        run_order_seed=101,
        mode="cheap",
        token_budget=2_000_000,
        eval_tokens=250_000,
        batch_size=64,
        eval_batch_size=64,
        trainer_sha256="trainer",
    )

    assert report["all_checks_passed"] is False
    assert report["checks"]["same_seed_reproduces_validation_bpb_within_tolerance"] is False


def test_checkpoint_comparison_distinguishes_exact_and_numerical_reproducibility(
    tmp_path,
) -> None:
    first = {
        "schema_version": 2,
        "model_state": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer_state": {"state": {0: {"moment": torch.tensor([0.25])}}},
        "training": {
            "cumulative_loss": 64.0,
            "cumulative_loss_tokens": 128,
            "seed": 11,
            "tokens_seen": 128,
        },
    }
    second = {
        "schema_version": 2,
        "model_state": {"weight": torch.tensor([1.0 + 1e-7, 2.0])},
        "optimizer_state": {"state": {0: {"moment": torch.tensor([0.25])}}},
        "training": {
            "cumulative_loss": 64.0 + 5e-7,
            "cumulative_loss_tokens": 128,
            "seed": 11,
            "tokens_seen": 128,
        },
    }
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    torch.save(first, first_path)
    torch.save(second, second_path)

    comparison = compare_checkpoints([first_path, second_path], absolute_tolerance=1e-6)

    assert comparison["exact_state_hashes_equal"] is False
    assert comparison["metadata_exact"] is True
    assert comparison["model"]["exact"] is False
    assert comparison["model"]["maximum_absolute_difference"] == pytest.approx(
        torch.finfo(torch.float32).eps
    )
    assert comparison["training_loss_within_absolute_tolerance"] is True
    assert comparison["within_absolute_tolerance"] is True


def test_checkpoint_comparison_checks_every_pair(tmp_path) -> None:
    paths = []
    for index, value in enumerate((0.0, 0.75e-6, -0.75e-6)):
        payload = {
            "schema_version": 2,
            "model_state": {"weight": torch.tensor([value])},
            "optimizer_state": {"state": {0: {"moment": torch.tensor([0.25])}}},
            "training": {
                "cumulative_loss": 64.0,
                "cumulative_loss_tokens": 128,
                "seed": 11,
                "tokens_seen": 128,
            },
        }
        path = tmp_path / f"checkpoint-{index}.pt"
        torch.save(payload, path)
        paths.append(path)

    comparison = compare_checkpoints(paths, absolute_tolerance=1e-6)

    assert comparison["model"]["maximum_absolute_difference"] == pytest.approx(1.5e-6)
    assert comparison["model"]["within_absolute_tolerance"] is False
    assert comparison["within_absolute_tolerance"] is False


def test_output_overwrite_requires_calibration_marker(tmp_path) -> None:
    output = tmp_path / "calibration"
    prepared = prepare_output_root(output, overwrite=False)
    assert (prepared / ".autodidact-calibration").is_file()
    (prepared / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(CalibrationError, match="already exists"):
        prepare_output_root(output, overwrite=False)
    replaced = prepare_output_root(output, overwrite=True)
    assert not (replaced / "stale.txt").exists()

    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    with pytest.raises(CalibrationError, match="unmarked"):
        prepare_output_root(unmarked, overwrite=True)
