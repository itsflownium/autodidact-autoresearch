"""Command-line interface for immutable data preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autodidact.data.config import default_output_root, default_raw_dir
from autodidact.data.download import (
    SourceIntegrityError,
    ensure_sources,
    verify_pinned_sources,
)
from autodidact.data.integrity import (
    DatasetIntegrityError,
    ProtectedPathError,
    assert_research_paths_allowed,
    verify_dataset,
)
from autodidact.data.pipeline import DatasetBuildError, build_pinned_dataset


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _add_raw_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--raw-dir",
        type=_path,
        default=default_raw_dir(),
        help="source download cache (default: %(default)s)",
    )


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        type=_path,
        default=default_output_root(),
        help="versioned prepared dataset root (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare.py",
        description="Build and verify immutable TinyStories token shards.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download and verify pinned source files")
    _add_raw_dir(download)

    build = commands.add_parser("build", help="build from already downloaded pinned sources")
    _add_raw_dir(build)
    _add_output_root(build)

    prepare = commands.add_parser("prepare", help="download, build, seal, and verify")
    _add_raw_dir(prepare)
    _add_output_root(prepare)

    verify = commands.add_parser("verify", help="verify manifests, hashes, and permissions")
    _add_output_root(verify)
    verify.add_argument("--scope", choices=("public", "all"), default="all")

    summary = commands.add_parser("summary", help="print the public data manifest summary")
    _add_output_root(summary)

    paths = commands.add_parser(
        "check-paths", help="verify that proposed research changes stay inside the allowlist"
    )
    paths.add_argument("paths", nargs="+")
    return parser


def _print_summary(manifest: dict[str, object]) -> None:
    splits = manifest["splits"]
    assert isinstance(splits, dict)
    summary = {
        "dataset": manifest["dataset"],
        "pipeline": manifest["pipeline"],
        "splits": {
            name: {
                "stories": split["stories"],
                "token_count": split["token_count"],
                "utf8_bytes": split["utf8_bytes"],
            }
            for name, split in splits.items()
        },
        "tokenizer": manifest["tokenizer"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_or_verify(raw_dir: Path, output_root: Path) -> dict[str, object]:
    if output_root.exists():
        print(f"prepared dataset already exists; verifying {output_root}", file=sys.stderr)
        return verify_dataset(output_root, scope="all")
    verify_pinned_sources(raw_dir)
    return build_pinned_dataset(raw_dir, output_root)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            sources = ensure_sources(args.raw_dir)
            for role, path in sources.items():
                print(f"{role}: {path}")
            return 0
        if args.command == "build":
            manifest = _build_or_verify(args.raw_dir, args.output_root)
            _print_summary(manifest)
            return 0
        if args.command == "prepare":
            if args.output_root.exists():
                print(
                    f"prepared dataset already exists; verifying {args.output_root}",
                    file=sys.stderr,
                )
                manifest = verify_dataset(args.output_root, scope="all")
            else:
                ensure_sources(args.raw_dir)
                manifest = build_pinned_dataset(args.raw_dir, args.output_root)
            _print_summary(manifest)
            return 0
        if args.command == "verify":
            manifest = verify_dataset(args.output_root, scope=args.scope)
            print(f"verified {args.scope} data under {args.output_root}")
            _print_summary(manifest)
            return 0
        if args.command == "summary":
            manifest = verify_dataset(args.output_root, scope="public")
            _print_summary(manifest)
            return 0
        if args.command == "check-paths":
            assert_research_paths_allowed(args.paths)
            print("all proposed paths are allowed")
            return 0
    except (
        DatasetBuildError,
        DatasetIntegrityError,
        FileExistsError,
        ProtectedPathError,
        SourceIntegrityError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
