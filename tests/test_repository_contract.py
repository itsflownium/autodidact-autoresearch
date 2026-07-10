from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_research_program_matches_protected_policy() -> None:
    policy = json.loads((ROOT / "autodidact/data/policy.json").read_text(encoding="utf-8"))
    program = (ROOT / "program.md").read_text(encoding="utf-8")

    assert policy["research_agent"]["allowed_repository_paths"] == ["train.py"]
    assert "You may edit exactly one repository file" in program
    assert "1,016,960" in program
    assert "1,050,000" in program
    assert "parent_bpb - candidate_bpb" in program
    assert "Never choose, search, retry, discard, or report seeds" in program
    for protected in ("prepare.py", "program.md", "pyproject.toml", "uv.lock"):
        assert f"`{protected}`" in program


def test_ci_runs_locked_repository_checks_with_read_only_permissions() -> None:
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
        "uv run python -m compileall",
        "uv run train.py inspect",
        "uv build",
    ):
        assert command in workflow
    assert "secrets." not in workflow
