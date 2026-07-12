from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from autodidact.agent_cli import main as agent_main
from autodidact.researcher import (
    ResearcherConfig,
    ResearcherError,
    ResearcherProvider,
    ResearchRequest,
    ResearchStatus,
)
from autodidact.researcher import (
    main as researcher_main,
)
from autodidact.researcher_providers import (
    ClaudeCodeResearcherAdapter,
    CodexResearcherAdapter,
    HermesAgentResearcherAdapter,
    build_researcher_adapter,
    response_json_schema,
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
    _git(repository, "add", "train.py")
    _git(repository, "commit", "-m", "Add parent")
    return repository, _git(repository, "rev-parse", "HEAD")


def _request(parent: str, *, maximum_total_tokens: int = 10_000) -> ResearchRequest:
    return ResearchRequest(
        request_id="request-native-001",
        parent_commit=parent,
        proposal_number=1,
        program_text="Change only train.py.",
        previous_results=(),
        maximum_total_tokens=maximum_total_tokens,
    )


def _response() -> dict[str, object]:
    return {
        "failure_reason": None,
        "proposal": {
            "change": "Reduce the learning rate.",
            "expected_effect_bpb": 0.004,
            "failure_signal": "Held-out BPB does not improve.",
            "hypothesis": "A smaller step should reduce optimization noise.",
            "interaction_risk": "Warmup may need retuning.",
            "mechanism": "Take smaller updates after warmup.",
            "minimum_useful_gain_bpb": 0.001,
            "resource_risk": "No expected resource change.",
            "title": "Reduce learning rate",
        },
        "status": "proposed",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _executable(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/usr/bin/env python3\n" + source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _codex_executable(tmp_path: Path, *, usage: bool = True) -> Path:
    return _executable(
        tmp_path,
        "fake-codex",
        f"""
import json
import os
import pathlib
import sys

if "--version" in sys.argv:
    print("codex-cli test-1")
    raise SystemExit(0)
assert sys.argv[1] == "exec"
assert "--ephemeral" in sys.argv
sandbox_index = sys.argv.index("--sandbox")
assert ["--sandbox", "workspace-write"] == sys.argv[sandbox_index:sandbox_index + 2]
schema = pathlib.Path(sys.argv[sys.argv.index("--output-schema") + 1])
assert json.loads(schema.read_text())["additionalProperties"] is False
request = json.load(sys.stdin)
assert request["maximum_total_tokens"] == 10000
assert os.environ["AUTODIDACT_RESEARCHER_TOKEN_BUDGET"] == "10000"
path = pathlib.Path("train.py")
path.write_text(path.read_text() + "LEARNING_RATE = 0.008\\n")
response = pathlib.Path(sys.argv[sys.argv.index("--output-last-message") + 1])
response.write_text(json.dumps({repr(_response())}))
event = {{"type": "turn.completed", "model": "test-codex-model"}}
if {usage!r}:
    event["usage"] = {{"input_tokens": 410, "output_tokens": 90}}
print(json.dumps(event))
""",
    )


def _claude_executable(tmp_path: Path) -> Path:
    return _executable(
        tmp_path,
        "fake-claude",
        f"""
import json
import pathlib
import sys

if "--version" in sys.argv:
    print("Claude Code test-1")
    raise SystemExit(0)
for required in ("--print", "--json-schema", "--no-session-persistence", "--strict-mcp-config"):
    assert required in sys.argv
schema = json.loads(sys.argv[sys.argv.index("--json-schema") + 1])
assert schema["additionalProperties"] is False
request = json.load(sys.stdin)
assert request["allowed_paths"] == ["train.py"]
path = pathlib.Path("train.py")
path.write_text(path.read_text() + "LEARNING_RATE = 0.008\\n")
print(json.dumps({{
    "type": "result",
    "is_error": False,
    "model": "test-claude-model",
    "structured_output": {repr(_response())},
    "usage": {{"input_tokens": 520, "output_tokens": 80}}
}}))
""",
    )


def _hermes_executable(tmp_path: Path) -> Path:
    return _executable(
        tmp_path,
        "fake-hermes",
        f"""
import json
import pathlib
import sys

if "--version" in sys.argv:
    print("Hermes Agent test-1")
    raise SystemExit(0)
assert "--oneshot" in sys.argv
toolsets_index = sys.argv.index("--toolsets")
assert ["--toolsets", "terminal"] == sys.argv[toolsets_index:toolsets_index + 2]
request_paths = list(pathlib.Path.cwd().glob(".autodidact-request-*.json"))
assert len(request_paths) == 1
request = json.loads(request_paths[0].read_text())
assert request["proposal_number"] == 1
path = pathlib.Path("train.py")
path.write_text(path.read_text() + "LEARNING_RATE = 0.008\\n")
usage_path = pathlib.Path(sys.argv[sys.argv.index("--usage-file") + 1])
usage_path.write_text(json.dumps({{
    "completed": True,
    "failed": False,
    "input_tokens": 630,
    "output_tokens": 70,
    "model": "test-hermes-model",
    "provider": "test-backend"
}}))
print(json.dumps({repr(_response())}))
""",
    )


@pytest.mark.parametrize(
    ("provider", "adapter_type", "builder", "expected_usage", "expected_model"),
    [
        (
            ResearcherProvider.CODEX,
            CodexResearcherAdapter,
            _codex_executable,
            500,
            "test-codex-model",
        ),
        (
            ResearcherProvider.CLAUDE_CODE,
            ClaudeCodeResearcherAdapter,
            _claude_executable,
            600,
            "test-claude-model",
        ),
        (
            ResearcherProvider.HERMES_AGENT,
            HermesAgentResearcherAdapter,
            _hermes_executable,
            700,
            "test-hermes-model",
        ),
    ],
)
def test_native_provider_normalizes_patch_response_and_trusted_usage(
    tmp_path: Path,
    provider: ResearcherProvider,
    adapter_type: type,
    builder: object,
    expected_usage: int,
    expected_model: str,
) -> None:
    repository, parent = _repository(tmp_path)
    executable = builder(tmp_path)  # type: ignore[operator]
    config = ResearcherConfig(provider=provider, executable=str(executable))
    adapter = build_researcher_adapter(config)

    attempt = adapter.run(
        _request(parent),
        workspace=repository,
        artifact_root=tmp_path / "artifacts",
    )

    assert isinstance(adapter, adapter_type)
    assert attempt.status is ResearchStatus.PROPOSED
    assert attempt.changed_paths == ("train.py",)
    assert attempt.usage.total_tokens == expected_usage
    assert attempt.usage_verified
    assert attempt.provider is provider
    assert attempt.resolved_model == expected_model
    assert (
        attempt.inference_provider
        == {
            ResearcherProvider.CODEX: "openai",
            ResearcherProvider.CLAUDE_CODE: "anthropic",
            ResearcherProvider.HERMES_AGENT: "test-backend",
        }[provider]
    )
    assert attempt.cli_version is not None
    assert not list(repository.glob(".autodidact-request-*.json"))
    transcript = json.loads(attempt.transcript_path.read_text(encoding="utf-8"))
    assert transcript["provider"] == provider.value
    assert transcript["inference_provider"] == attempt.inference_provider
    assert set(transcript["provider_configuration"]) == {
        "backend_provider",
        "max_budget_usd",
        "max_turns",
        "model",
        "profile",
        "reasoning_effort",
    }
    assert transcript["usage_verified"] is True
    assert transcript["response"]["usage"] == {
        "input_tokens": expected_usage - {500: 90, 600: 80, 700: 70}[expected_usage],
        "output_tokens": {500: 90, 600: 80, 700: 70}[expected_usage],
    }


def test_native_provider_fails_closed_without_trusted_usage(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path)
    adapter = CodexResearcherAdapter(
        ResearcherConfig(
            provider=ResearcherProvider.CODEX,
            executable=str(_codex_executable(tmp_path, usage=False)),
        )
    )

    attempt = adapter.run(
        _request(parent),
        workspace=repository,
        artifact_root=tmp_path / "artifacts",
    )

    assert attempt.status is ResearchStatus.FAILED
    assert attempt.failure_reason == "researcher command exited with status 65"
    assert not attempt.usage_verified
    transcript = json.loads(attempt.transcript_path.read_text(encoding="utf-8"))
    assert "trusted token-usage record" in transcript["stderr"]


def test_trusted_native_usage_enforces_request_budget(tmp_path: Path) -> None:
    repository, parent = _repository(tmp_path)
    adapter = ClaudeCodeResearcherAdapter(
        ResearcherConfig(
            provider=ResearcherProvider.CLAUDE_CODE,
            executable=str(_claude_executable(tmp_path)),
        )
    )

    attempt = adapter.run(
        _request(parent, maximum_total_tokens=599),
        workspace=repository,
        artifact_root=tmp_path / "artifacts",
    )

    assert attempt.status is ResearchStatus.FAILED
    assert attempt.failure_reason == "researcher usage exceeded its assigned token budget"


def test_response_json_schema_is_closed_and_requires_every_field() -> None:
    schema = response_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"status", "proposal", "failure_reason", "usage"}
    proposal = schema["properties"]["proposal"]["anyOf"][0]
    assert proposal["additionalProperties"] is False
    assert proposal["properties"]["minimum_useful_gain_bpb"] == {"type": "number"}


def test_setup_and_doctor_write_private_native_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    executable = _claude_executable(tmp_path)
    config_path = tmp_path / "control" / "researcher.json"

    exit_code = agent_main(
        [
            "setup",
            "--provider",
            "claude-code",
            "--executable",
            str(executable),
            "--model",
            "test-model",
            "--reasoning-effort",
            "high",
            "--max-turns",
            "12",
            "--max-budget-usd",
            "3.5",
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    setup_payload = json.loads(capsys.readouterr().out)
    assert setup_payload["provider"] == "claude_code"
    assert stat_mode(config_path) == 0o600
    config = ResearcherConfig.from_path(config_path)
    assert config.provider is ResearcherProvider.CLAUDE_CODE
    assert config.model == "test-model"
    assert config.max_turns == 12

    assert agent_main(["doctor", "--config", str(config_path), "--workspace", str(tmp_path)]) == 0
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["ready"] is True
    assert doctor_payload["version"] == "Claude Code test-1"


def test_setup_config_drives_researcher_cli_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository, parent = _repository(tmp_path)
    config_path = tmp_path / "researcher.json"
    assert (
        agent_main(
            [
                "setup",
                "--provider",
                "codex",
                "--executable",
                str(_codex_executable(tmp_path)),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    program_path = tmp_path / "program.md"
    program_path.write_text("Change only train.py.\n", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "allowed_paths": ["train.py"],
                "maximum_total_tokens": 10_000,
                "parent_commit": parent,
                "previous_results": [],
                "program_path": str(program_path),
                "proposal_number": 1,
                "request_id": "request-native-001",
            }
        ),
        encoding="utf-8",
    )

    exit_code = researcher_main(
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
    assert payload["provider"] == "codex"
    assert payload["usage"] == {"input_tokens": 410, "output_tokens": 90}
    assert payload["usage_verified"] is True


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


def test_setup_refuses_overwrite_and_broken_executable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    executable = _claude_executable(tmp_path)
    config_path = tmp_path / "researcher.json"
    arguments = [
        "setup",
        "--provider",
        "claude-code",
        "--executable",
        str(executable),
        "--config",
        str(config_path),
    ]
    assert agent_main(arguments) == 0
    capsys.readouterr()
    assert agent_main(arguments) == 2
    assert "already exists" in capsys.readouterr().err

    broken_path = tmp_path / "broken.json"
    assert (
        agent_main(
            [
                "setup",
                "--provider",
                "codex",
                "--executable",
                str(tmp_path / "missing"),
                "--config",
                str(broken_path),
            ]
        )
        == 2
    )
    assert not broken_path.exists()


def test_setup_requires_hermes_backend_and_model_together(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        agent_main(
            [
                "setup",
                "--provider",
                "hermes-agent",
                "--executable",
                str(_hermes_executable(tmp_path)),
                "--backend-provider",
                "test-backend",
                "--config",
                str(tmp_path / "researcher.json"),
            ]
        )
        == 2
    )
    assert "requires --backend-provider and --model together" in capsys.readouterr().err


def test_native_configuration_rejects_cross_provider_options() -> None:
    with pytest.raises(ResearcherError, match="only by the codex provider"):
        ResearcherConfig(
            provider=ResearcherProvider.CLAUDE_CODE,
            executable="claude",
            profile="research",
        )
    with pytest.raises(ResearcherError, match="only by claude_code"):
        ResearcherConfig(
            provider=ResearcherProvider.HERMES_AGENT,
            executable="hermes",
            max_budget_usd=1.0,
        )
