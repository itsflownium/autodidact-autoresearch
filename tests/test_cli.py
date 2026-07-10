from __future__ import annotations

from pathlib import Path

import autodidact.data.cli as cli
from autodidact.data.cli import main


def test_check_paths_command(capsys) -> None:
    assert main(["check-paths", "train.py"]) == 0
    assert "allowed" in capsys.readouterr().out


def test_check_paths_command_rejects_protected_file(capsys) -> None:
    assert main(["check-paths", "prepare.py"]) == 2
    assert "protected" in capsys.readouterr().err


def test_prepare_verifies_existing_output_without_downloading(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output_root = tmp_path / "prepared"
    output_root.mkdir()
    manifest = {"dataset": {}, "pipeline": {}, "splits": {}, "tokenizer": {}}

    monkeypatch.setattr(cli, "verify_dataset", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        cli,
        "ensure_sources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing output must not trigger a download")
        ),
    )

    assert main(["prepare", "--output-root", str(output_root)]) == 0
    assert "already exists" in capsys.readouterr().err
