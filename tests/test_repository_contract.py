from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_contains_control_plane_without_a_bundled_target() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    program = (ROOT / "program.md").read_text(encoding="utf-8")

    assert not (ROOT / "train.py").exists()
    assert not (ROOT / "prepare.py").exists()
    data_package = ROOT / "autodidact" / "data"
    assert not any(data_package.glob("*.py"))
    assert not any(data_package.glob("*.json"))
    for dependency in ("torch", "tokenizers", "huggingface-hub", "zstandard"):
        assert dependency not in project
    assert "configured target" in program.lower()
    assert "editable_paths" in program
    assert "protected evaluator" in program.lower()
    assert "GRPO" not in program


def test_ci_runs_locked_control_plane_checks_with_read_only_permissions() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "pull_request_target" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "uv sync --locked --all-groups" in workflow
    for command in (
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run pytest -q",
        "uv run python -m compileall -q autodidact tests",
        "uv build",
    ):
        assert command in workflow
    assert "train.py inspect" not in workflow
    assert "secrets." not in workflow
