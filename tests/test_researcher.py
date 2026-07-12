from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from autodidact.researcher import (
    CommandResearcherAdapter,
    ResearcherConfig,
    ResearcherError,
    ResearchRequest,
    ResearchStatus,
    StructuredResearchResponse,
    load_research_attempt,
    main,
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "train.py").write_text("LEARNING_RATE = 0.01\n", encoding="utf-8")
    (repository / "README.md").write_text("protected\n", encoding="utf-8")
    _git(repository, "add", "train.py", "README.md")
    _git(repository, "commit", "-m", "Add parent")
    return repository, _git(repository, "rev-parse", "HEAD")


def _response(*, status: str = "proposed") -> dict[str, object]:
    proposal = {
        "change": "Reduce the learning rate.",
        "expected_effect_bpb": -0.004,
        "failure_signal": "Held-out BPB does not improve.",
        "hypothesis": "A smaller step should reduce optimization noise.",
        "interaction_risk": "The warmup may need retuning.",
        "mechanism": "Take smaller updates after warmup.",
        "minimum_useful_gain_bpb": 0.001,
        "resource_risk": "No expected resource change.",
        "title": "Reduce learning rate",
    }
    return {
        "failure_reason": None,
        "proposal": proposal if status == "proposed" else None,
        "status": status,
        "usage": {"input_tokens": 120, "output_tokens": 45},
    }


def _request(parent: str, *, request_id: str = "request-001") -> ResearchRequest:
    return ResearchRequest(
        request_id=request_id,
        parent_commit=parent,
        proposal_number=1,
        program_text="# Research contract\nChange only train.py.",
        previous_results=(
            {
                "gain_bpb": -0.002,
                "proposal": "Previous trial",
                "verdict": "rejected",
            },
        ),
    )


def _script(tmp_path: Path, source: str, *, name: str = "fake_researcher.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _adapter(script: Path, *, timeout: float = 5.0) -> CommandResearcherAdapter:
    return CommandResearcherAdapter(
        ResearcherConfig(
            command=(sys.executable, str(script)),
            timeout_seconds=timeout,
        )
    )


def test_command_adapter_captures_structured_proposal_and_evidence(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path)
    response = json.dumps(_response(), sort_keys=True)
    script = _script(
        tmp_path,
        """
import json
import pathlib
import sys

request = json.load(sys.stdin)
assert request["allowed_paths"] == ["train.py"]
path = pathlib.Path("train.py")
path.write_text(path.read_text() + "LEARNING_RATE = 0.008\\n")
print(RESPONSE)
""".replace("RESPONSE", repr(response)),
    )
    artifact_root = tmp_path / "artifacts"

    attempt = _adapter(script).run(
        _request(parent),
        workspace=repository,
        artifact_root=artifact_root,
    )

    assert attempt.status is ResearchStatus.PROPOSED
    assert attempt.proposal is not None
    assert attempt.proposal.minimum_useful_gain_bpb == pytest.approx(0.001)
    assert attempt.changed_paths == ("train.py",)
    assert attempt.diff_sha256 is not None
    assert attempt.usage.total_tokens == 165
    transcript = json.loads(attempt.transcript_path.read_text(encoding="utf-8"))
    prompt = json.loads(transcript["prompt"])
    assert prompt["program_md"].startswith("# Research contract")
    assert prompt["previous_results"][0]["verdict"] == "rejected"
    assert prompt["maximum_total_tokens"] == 50_000
    assert prompt["contract"]["evaluation"].startswith("Do not grade")
    assert transcript["response"]["proposal"]["hypothesis"] == (
        "A smaller step should reduce optimization noise."
    )
    assert "LEARNING_RATE = 0.008" in transcript["diff"]
    assert transcript["failure_reason"] is None

    recovered = load_research_attempt(
        attempt.transcript_path,
        _request(parent),
        workspace=repository,
    )
    assert recovered == attempt


def test_adapter_rejects_protected_file_changes_and_records_failure(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path)
    response = json.dumps(_response(), sort_keys=True)
    script = _script(
        tmp_path,
        """
import pathlib

pathlib.Path("README.md").write_text("changed protected file\\n")
print(RESPONSE)
""".replace("RESPONSE", repr(response)),
    )

    attempt = _adapter(script).run(
        _request(parent),
        workspace=repository,
        artifact_root=tmp_path / "artifacts",
    )

    assert attempt.status is ResearchStatus.FAILED
    assert attempt.proposal is None
    assert attempt.changed_paths == ("README.md",)
    assert attempt.failure_reason == "researcher changed protected paths: ['README.md']"
    transcript = json.loads(attempt.transcript_path.read_text(encoding="utf-8"))
    assert transcript["failure_reason"] == attempt.failure_reason
    assert transcript["stdout"]


def test_response_schema_cannot_return_protected_decisions() -> None:
    response = _response()
    response["verdict"] = "promote"

    with pytest.raises(ResearcherError, match="response keys differ"):
        StructuredResearchResponse.from_json(json.dumps(response))


def test_adapter_records_nonzero_exit_and_does_not_fabricate_usage(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path)
    script = _script(
        tmp_path,
        """
import sys

print("process failed", file=sys.stderr)
raise SystemExit(7)
""",
    )

    attempt = _adapter(script).run(
        _request(parent),
        workspace=repository,
        artifact_root=tmp_path / "artifacts",
    )

    assert attempt.status is ResearchStatus.FAILED
    assert attempt.failure_reason == "researcher command exited with status 7"
    assert attempt.returncode == 7
    assert attempt.usage.total_tokens == 0
    transcript = json.loads(attempt.transcript_path.read_text(encoding="utf-8"))
    assert "process failed" in transcript["stderr"]


def test_adapter_times_out_process_group_and_records_attempt(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path)
    script = _script(
        tmp_path,
        """
import time

time.sleep(30)
""",
    )

    attempt = _adapter(script, timeout=0.05).run(
        _request(parent),
        workspace=repository,
        artifact_root=tmp_path / "artifacts",
    )

    assert attempt.status is ResearchStatus.FAILED
    assert attempt.timed_out
    assert attempt.failure_reason == "researcher command timed out"


def test_adapter_rejects_reported_usage_above_assigned_budget(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path)
    response = _response()
    response["usage"] = {"input_tokens": 80, "output_tokens": 30}
    script = _script(
        tmp_path,
        """
import pathlib

path = pathlib.Path("train.py")
path.write_text(path.read_text() + "LEARNING_RATE = 0.008\\n")
print(RESPONSE)
""".replace("RESPONSE", repr(json.dumps(response))),
    )

    attempt = _adapter(script).run(
        replace(_request(parent), maximum_total_tokens=100),
        workspace=repository,
        artifact_root=tmp_path / "artifacts",
    )

    assert attempt.status is ResearchStatus.FAILED
    assert attempt.proposal is None
    assert attempt.failure_reason == ("researcher reported usage above its assigned token budget")


def test_no_change_response_requires_clean_workspace(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path)
    response = _response(status="no_change")
    script = _script(tmp_path, f"print({json.dumps(json.dumps(response))})\n")

    attempt = _adapter(script).run(
        _request(parent),
        workspace=repository,
        artifact_root=tmp_path / "artifacts",
    )

    assert attempt.status is ResearchStatus.NO_CHANGE
    assert attempt.failure_reason is None
    assert attempt.changed_paths == ()


def test_existing_transcript_prevents_duplicate_invocation(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path)
    marker = tmp_path / "invocations.txt"
    response = json.dumps(_response(status="no_change"), sort_keys=True)
    script = _script(
        tmp_path,
        """
import pathlib

marker = pathlib.Path(MARKER)
marker.write_text(marker.read_text() + "run\\n" if marker.exists() else "run\\n")
print(RESPONSE)
""".replace("MARKER", repr(str(marker))).replace("RESPONSE", repr(response)),
    )
    adapter = _adapter(script)
    request = _request(parent)
    artifact_root = tmp_path / "artifacts"

    adapter.run(request, workspace=repository, artifact_root=artifact_root)
    with pytest.raises(ResearcherError, match="recover it instead of rerunning"):
        adapter.run(request, workspace=repository, artifact_root=artifact_root)

    assert marker.read_text(encoding="utf-8") == "run\n"


def test_configuration_is_strict_and_cli_uses_fake_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository, parent = _repository(tmp_path)
    response = json.dumps(_response(status="no_change"), sort_keys=True)
    script = _script(tmp_path, f"print({json.dumps(response)})\n")
    config_path = tmp_path / "researcher.json"
    config_path.write_text(
        json.dumps(
            {
                "command": [sys.executable, str(script)],
                "environment": {},
                "inherit_environment": ["PATH"],
                "max_diff_bytes": 100_000,
                "max_output_bytes": 100_000,
                "timeout_seconds": 5,
            }
        ),
        encoding="utf-8",
    )
    program_path = tmp_path / "program.md"
    program_path.write_text("Research contract.\n", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "allowed_paths": ["train.py"],
                "parent_commit": parent,
                "maximum_total_tokens": 500,
                "previous_results": [],
                "program_path": str(program_path),
                "proposal_number": 1,
                "request_id": "request-cli-001",
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--workspace",
            str(repository),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--request",
            str(request_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no_change"
    assert payload["usage"] == {"input_tokens": 120, "output_tokens": 45}


def test_configuration_rejects_unknown_capabilities(tmp_path: Path) -> None:
    path = tmp_path / "researcher.json"
    path.write_text(json.dumps({"command": ["agent"], "ledger_path": "state.sqlite3"}))

    with pytest.raises(ResearcherError, match="unknown researcher configuration keys"):
        ResearcherConfig.from_path(path)
