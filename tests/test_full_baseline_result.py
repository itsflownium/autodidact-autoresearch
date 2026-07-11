from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs/baseline/m4-full.json"


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_committed_full_baseline_has_complete_contract() -> None:
    report = _report()
    contract = report["contract"]
    runs = report["runs"]

    assert report["all_checks_passed"] is True
    assert report["complete_full_baseline"] is True
    assert report["diagnostic_override"] is False
    assert all(report["checks"].values())
    assert contract["mode"] == "full"
    assert contract["device"] == "mps"
    assert contract["seeds"] == [1337, 2027, 4099]
    assert contract["token_budget"] == 20_000_000
    assert contract["eval_tokens"] is None
    assert contract["parameter_count"] == 1_016_960
    assert len(contract["trainer_sha256"]) == 64
    assert len(contract["runner_sha256"]) == 64
    assert [run["seed"] for run in runs] == contract["seeds"]


def test_committed_full_baseline_runs_and_statistics_are_consistent() -> None:
    report = _report()
    runs = report["runs"]
    validation_bpb = [float(run["validation_bpb"]) for run in runs]

    for run in runs:
        assert run["tokens_seen"] == 20_000_000
        assert run["target_tokens"] == 20_000_000
        assert run["parameter_count"] == 1_016_960
        assert run["predicted_tokens"] == 2_680_300
        assert run["stories"] == 10_998
        assert run["utf8_bytes"] == 9_549_555
        assert run["process_attempts"] == 1
        assert run["resume_segments"] == 0
        assert len(run["checkpoint_sha256"]) == 64
        assert len(run["checkpoint_state_sha256"]) == 64
        assert not Path(run["checkpoint_path"]).is_absolute()
        assert not Path(run["metrics_path"]).is_absolute()

    summary = report["statistics"]["validation_bpb"]
    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(statistics.fmean(validation_bpb))
    assert summary["sample_standard_deviation"] == pytest.approx(statistics.stdev(validation_bpb))
    assert summary["minimum"] == min(validation_bpb)
    assert summary["maximum"] == max(validation_bpb)


def test_committed_full_baseline_report_is_portable_and_documented() -> None:
    serialized = REPORT_PATH.read_text(encoding="utf-8")
    markdown = (ROOT / "docs/baseline/m4-full.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert str(Path.home()) not in serialized
    for sensitive_label in ("serial" + " number", "hardware " + "uuid"):
        assert sensitive_label not in serialized.lower()
    assert "1.031838031" in markdown
    assert "docs/baseline/m4-full.md" in readme
    assert "m4-full.json" in readme
