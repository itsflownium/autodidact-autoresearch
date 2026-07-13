"""Append-only SQLite evidence ledger for protected autoresearch experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from autodidact.data.integrity import canonical_json_bytes
from autodidact.records import (
    RECORD_SCHEMA_VERSION,
    AllocationAction,
    ArtifactManifest,
    CandidateRecord,
    ComputeRecord,
    DecisionRecord,
    DecisionVerdict,
    DownstreamAllocation,
    DownstreamPrediction,
    EffectEstimate,
    ExperimentRecord,
    ExperimentStage,
    LineageRecord,
    PairedResult,
    PatchProposal,
    RunArm,
    RunResult,
    RunStatus,
    TrialSchedule,
    TrialSpec,
    build_effect_estimate,
    build_paired_result,
    downstream_audit_assignment,
    new_record_id,
    record_from_envelope,
    record_id,
    record_to_envelope,
)

LEDGER_SCHEMA_VERSION = 1
LEDGER_EXPORT_SCHEMA_VERSION = 1
DEFAULT_LEDGER_PATH = Path("artifacts/ledger/experiments.sqlite3")
APPLICATION_ID = 0x41554444

_GENESIS_DOMAIN = b"autodidact-ledger-genesis-v1\0"
_EVENT_DOMAIN = b"autodidact-ledger-event-v1\0"
_RECORD_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,95}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_HOME_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/[^/\s\"']+")
_WINDOWS_HOME_PATTERN = re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+")

_STAGE_ORDER = {
    ExperimentStage.CHEAP: 0,
    ExperimentStage.INTERMEDIATE: 1,
    ExperimentStage.FULL: 2,
    ExperimentStage.PROMOTION: 3,
    ExperimentStage.SEALED_FINAL: 4,
}

_SCHEMA_SQL = f"""
PRAGMA application_id = {APPLICATION_ID};

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY,
    record_id TEXT NOT NULL UNIQUE,
    record_type TEXT NOT NULL,
    writer_role TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    previous_event_sha256 TEXT NOT NULL,
    event_sha256 TEXT NOT NULL UNIQUE,
    CHECK (length(payload_sha256) = 64),
    CHECK (length(previous_event_sha256) = 64),
    CHECK (length(event_sha256) = 64)
);

CREATE INDEX IF NOT EXISTS events_record_type_idx ON events(record_type, sequence);

CREATE TRIGGER IF NOT EXISTS events_no_update
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'ledger events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS events_no_delete
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'ledger events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS metadata_no_update
BEFORE UPDATE ON metadata
BEGIN
    SELECT RAISE(ABORT, 'ledger metadata is immutable');
END;

CREATE TRIGGER IF NOT EXISTS metadata_no_delete
BEFORE DELETE ON metadata
BEGIN
    SELECT RAISE(ABORT, 'ledger metadata is immutable');
END;
"""

_REQUIRED_TABLES = {"metadata", "events"}
_REQUIRED_TRIGGERS = {
    "events_no_update",
    "events_no_delete",
    "metadata_no_update",
    "metadata_no_delete",
}
_METADATA_KEYS = {
    "created_at",
    "genesis_sha256",
    "initial_parent_commit",
    "ledger_id",
    "record_schema_version",
}


class LedgerError(RuntimeError):
    """Base class for protected ledger failures."""


class LedgerIntegrityError(LedgerError):
    """Raised when ledger structure, records, or hashes cannot be trusted."""


class LedgerConflictError(LedgerError):
    """Raised when an append conflicts with existing immutable evidence."""


class LedgerStateError(LedgerError):
    """Raised when a record violates the experiment lifecycle."""


class LedgerPermissionError(LedgerError):
    """Raised when a writer role attempts an unauthorized record type."""


class WriterRole(StrEnum):
    RESEARCH_AGENT = "research_agent"
    CONTROLLER = "controller"
    EVALUATOR = "evaluator"


def resource_constraint_failures(
    trial: TrialSpec,
    parent: RunResult,
    candidate: RunResult,
) -> tuple[str, ...]:
    """Recompute candidate resource failures from a protected paired trial."""

    limits = trial.limits
    failures: list[str] = []
    if candidate.parameter_count > limits.max_parameter_count:
        failures.append("parameter_count")
    if (
        limits.max_peak_process_rss_bytes is not None
        and candidate.peak_process_rss_bytes is not None
        and candidate.peak_process_rss_bytes > limits.max_peak_process_rss_bytes
    ):
        failures.append("peak_process_rss")
    if limits.max_peak_device_bytes is not None and (
        candidate.peak_device_allocated_bytes is None
        or candidate.peak_device_allocated_bytes > limits.max_peak_device_bytes
    ):
        failures.append("peak_device_memory")
    if (
        limits.min_training_tokens_per_second is not None
        and candidate.training_tokens_per_second is not None
        and candidate.training_tokens_per_second < limits.min_training_tokens_per_second
    ):
        failures.append("training_throughput_minimum")

    if parent.training_tokens_per_second is None or candidate.training_tokens_per_second is None:
        raise LedgerStateError("successful paired runs require training throughput")
    if limits.max_training_throughput_regression_fraction is not None:
        regression = (
            parent.training_tokens_per_second - candidate.training_tokens_per_second
        ) / parent.training_tokens_per_second
        if regression > limits.max_training_throughput_regression_fraction:
            failures.append("training_throughput_regression")
    if parent.peak_process_rss_bytes is None or candidate.peak_process_rss_bytes is None:
        raise LedgerStateError("successful paired runs require process memory")
    if limits.max_peak_process_rss_regression_fraction is not None:
        regression = (candidate.peak_process_rss_bytes - parent.peak_process_rss_bytes) / max(
            parent.peak_process_rss_bytes, 1
        )
        if regression > limits.max_peak_process_rss_regression_fraction:
            failures.append("peak_process_rss_regression")
    if limits.max_peak_device_regression_fraction is not None:
        if (
            parent.peak_device_allocated_bytes is None
            or candidate.peak_device_allocated_bytes is None
        ):
            failures.append("peak_device_regression_unavailable")
        else:
            regression = (
                candidate.peak_device_allocated_bytes - parent.peak_device_allocated_bytes
            ) / max(parent.peak_device_allocated_bytes, 1)
            if regression > limits.max_peak_device_regression_fraction:
                failures.append("peak_device_regression")
    return tuple(sorted(failures))


_ALLOWED_WRITERS: dict[str, frozenset[WriterRole]] = {
    PatchProposal.RECORD_TYPE: frozenset({WriterRole.RESEARCH_AGENT, WriterRole.CONTROLLER}),
    CandidateRecord.RECORD_TYPE: frozenset({WriterRole.CONTROLLER}),
    TrialSchedule.RECORD_TYPE: frozenset({WriterRole.CONTROLLER}),
    TrialSpec.RECORD_TYPE: frozenset({WriterRole.CONTROLLER}),
    RunResult.RECORD_TYPE: frozenset({WriterRole.CONTROLLER, WriterRole.EVALUATOR}),
    ArtifactManifest.RECORD_TYPE: frozenset({WriterRole.CONTROLLER, WriterRole.EVALUATOR}),
    PairedResult.RECORD_TYPE: frozenset({WriterRole.CONTROLLER, WriterRole.EVALUATOR}),
    EffectEstimate.RECORD_TYPE: frozenset({WriterRole.CONTROLLER, WriterRole.EVALUATOR}),
    DownstreamPrediction.RECORD_TYPE: frozenset({WriterRole.CONTROLLER, WriterRole.EVALUATOR}),
    DownstreamAllocation.RECORD_TYPE: frozenset({WriterRole.CONTROLLER}),
    DecisionRecord.RECORD_TYPE: frozenset({WriterRole.CONTROLLER}),
    LineageRecord.RECORD_TYPE: frozenset({WriterRole.CONTROLLER}),
    ComputeRecord.RECORD_TYPE: frozenset({WriterRole.CONTROLLER, WriterRole.EVALUATOR}),
}


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    sequence: int
    record: ExperimentRecord
    writer_role: WriterRole
    recorded_at: str
    payload_sha256: str
    previous_event_sha256: str
    event_sha256: str


@dataclass(frozen=True, slots=True)
class LedgerVerification:
    ledger_id: str
    initial_parent_commit: str
    event_count: int
    head_event_sha256: str
    schema_version: int
    record_schema_version: int


@dataclass(frozen=True, slots=True)
class _VerifiedSnapshot:
    revision: tuple[int, str | None]
    storage_fingerprint: tuple[tuple[str, int, int, int, int, int] | tuple[str], ...]
    verification: LedgerVerification
    events: tuple[LedgerEvent, ...]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_timestamp(value: str) -> None:
    if not isinstance(value, str):
        raise LedgerIntegrityError("ledger timestamps must be text")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise LedgerIntegrityError(f"invalid ledger timestamp: {value}") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise LedgerIntegrityError("ledger timestamps must include a UTC offset")


def _metadata_genesis(metadata: dict[str, str]) -> str:
    committed = {key: value for key, value in metadata.items() if key != "genesis_sha256"}
    return hashlib.sha256(_GENESIS_DOMAIN + canonical_json_bytes(committed)).hexdigest()


def _event_hash(
    *,
    sequence: int,
    record: dict[str, Any],
    writer_role: str,
    recorded_at: str,
    payload_sha256: str,
    previous_event_sha256: str,
) -> str:
    committed = {
        "payload_sha256": payload_sha256,
        "previous_event_sha256": previous_event_sha256,
        "record": record,
        "recorded_at": recorded_at,
        "sequence": sequence,
        "writer_role": writer_role,
    }
    return hashlib.sha256(_EVENT_DOMAIN + canonical_json_bytes(committed)).hexdigest()


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _assert_portable_record(envelope: dict[str, Any]) -> None:
    home = str(Path.home())
    temporary = tempfile.gettempdir()
    for value in _iter_strings(envelope):
        if "\x00" in value:
            raise LedgerStateError("records cannot contain null bytes")
        if (home and home in value) or (temporary and temporary in value):
            raise LedgerStateError("records cannot contain machine-local home or temp paths")
        if _HOME_PATH_PATTERN.search(value) or _WINDOWS_HOME_PATTERN.search(value):
            raise LedgerStateError("records cannot contain machine-local home paths")


def _redact_value(value: Any, redactions: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        result = value
        for secret in redactions:
            if secret:
                result = result.replace(secret, "<redacted>")
        result = _HOME_PATH_PATTERN.sub("<home>", result)
        return _WINDOWS_HOME_PATTERN.sub("<home>", result)
    if isinstance(value, dict):
        return {key: _redact_value(item, redactions) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, redactions) for item in value]
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


class ExperimentLedger:
    """Transactional, append-only experiment evidence store."""

    def __init__(self, path: Path, *, read_only: bool) -> None:
        self.path = path.resolve()
        self.read_only = read_only
        self._verified_snapshot: _VerifiedSnapshot | None = None

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        initial_parent_commit: str,
        ledger_id: str | None = None,
    ) -> ExperimentLedger:
        resolved = path.expanduser().resolve()
        if resolved.exists():
            raise LedgerConflictError(f"ledger already exists: {resolved}")
        if not isinstance(initial_parent_commit, str) or not _GIT_COMMIT_PATTERN.fullmatch(
            initial_parent_commit
        ):
            raise LedgerStateError("initial_parent_commit must be a full Git commit")
        ledger_id = ledger_id or new_record_id("ledger")
        if not isinstance(ledger_id, str) or not _RECORD_ID_PATTERN.fullmatch(ledger_id):
            raise LedgerStateError("ledger_id must be a portable structured ID")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(resolved, timeout=30.0, isolation_level=None)
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA foreign_keys = ON")
            cls._migrate_connection(connection)
            metadata = {
                "created_at": _utc_now(),
                "initial_parent_commit": initial_parent_commit,
                "ledger_id": ledger_id,
                "record_schema_version": str(RECORD_SCHEMA_VERSION),
            }
            metadata["genesis_sha256"] = _metadata_genesis(metadata)
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            connection.close()
            for candidate in (resolved, Path(str(resolved) + "-shm"), Path(str(resolved) + "-wal")):
                candidate.unlink(missing_ok=True)
            raise
        finally:
            with suppress(sqlite3.Error):
                connection.close()
        ledger = cls(resolved, read_only=False)
        ledger.verify()
        return ledger

    @classmethod
    def open(cls, path: Path, *, read_only: bool = False) -> ExperimentLedger:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise LedgerError(f"ledger does not exist: {resolved}")
        ledger = cls(resolved, read_only=read_only)
        ledger.verify()
        return ledger

    @classmethod
    def migrate(cls, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise LedgerError(f"ledger does not exist: {resolved}")
        connection = sqlite3.connect(resolved, timeout=30.0, isolation_level=None)
        try:
            before = cls._raw_head_hash(connection)
            cls._migrate_connection(connection)
            after = cls._raw_head_hash(connection)
            if before != after:
                raise LedgerIntegrityError("schema migration changed immutable event evidence")
        finally:
            connection.close()
        cls.open(resolved, read_only=True)

    @staticmethod
    def _migrate_connection(connection: sqlite3.Connection) -> None:
        current = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current > LEDGER_SCHEMA_VERSION:
            raise LedgerIntegrityError(f"ledger schema {current} is newer than supported")
        if current < 1:
            connection.executescript(_SCHEMA_SQL)
            connection.execute(f"PRAGMA user_version = {LEDGER_SCHEMA_VERSION}")
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        if application_id != APPLICATION_ID:
            raise LedgerIntegrityError("file is not an Autodidact experiment ledger")

    @staticmethod
    def _raw_head_hash(connection: sqlite3.Connection) -> str | None:
        try:
            row = connection.execute(
                "SELECT event_sha256 FROM events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return None if row is None else str(row[0])

    @staticmethod
    def _raw_revision(connection: sqlite3.Connection) -> tuple[int, str | None]:
        row = connection.execute(
            """
            SELECT COUNT(*),
                (SELECT event_sha256 FROM events ORDER BY sequence DESC LIMIT 1)
            FROM events
            """
        ).fetchone()
        return int(row[0]), None if row[1] is None else str(row[1])

    def _storage_fingerprint(
        self,
    ) -> tuple[tuple[str, int, int, int, int, int] | tuple[str], ...]:
        # SQLite may create and remove an empty WAL around read-only connections.
        result: list[tuple[str, int, int, int, int, int] | tuple[str]] = []
        for suffix in ("", "-wal"):
            candidate = Path(str(self.path) + suffix)
            try:
                stat = candidate.stat()
            except FileNotFoundError:
                result.append((suffix,))
                continue
            if suffix == "-wal" and stat.st_size == 0:
                result.append((suffix,))
                continue
            result.append(
                (
                    suffix,
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            )
        return tuple(result)

    def _remember_verified_snapshot(
        self,
        connection: sqlite3.Connection,
        verification: LedgerVerification,
        events: Sequence[LedgerEvent],
    ) -> None:
        before = self._storage_fingerprint()
        revision = self._raw_revision(connection)
        after = self._storage_fingerprint()
        expected_revision = (
            verification.event_count,
            None if verification.event_count == 0 else verification.head_event_sha256,
        )
        if before != after or revision != expected_revision:
            self._verified_snapshot = None
            return
        self._verified_snapshot = _VerifiedSnapshot(
            revision=revision,
            storage_fingerprint=after,
            verification=verification,
            events=tuple(events),
        )

    def _refresh_snapshot_fingerprint(self) -> None:
        cached = self._verified_snapshot
        if cached is None:
            return
        self._verified_snapshot = _VerifiedSnapshot(
            revision=cached.revision,
            storage_fingerprint=self._storage_fingerprint(),
            verification=cached.verification,
            events=cached.events,
        )

    def _verified_connection(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[LedgerVerification, list[LedgerEvent]]:
        # Reuse only an unchanged, fully verified immutable ledger prefix.
        cached = self._verified_snapshot
        if cached is not None:
            before = self._storage_fingerprint()
            revision = self._raw_revision(connection)
            after = self._storage_fingerprint()
            if before == after == cached.storage_fingerprint and revision == cached.revision:
                return cached.verification, list(cached.events)
        verification, events = self._verify_connection(connection)
        self._remember_verified_snapshot(connection, verification, events)
        return verification, events

    def _connect(self, *, write: bool = False) -> sqlite3.Connection:
        if write and self.read_only:
            raise LedgerPermissionError("ledger was opened read-only")
        if self.read_only:
            connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro",
                uri=True,
                timeout=30.0,
                isolation_level=None,
            )
        else:
            connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        if self.read_only:
            connection.execute("PRAGMA query_only = ON")
        return connection

    @staticmethod
    def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
        rows = connection.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
        metadata = {str(row["key"]): str(row["value"]) for row in rows}
        if set(metadata) != _METADATA_KEYS:
            raise LedgerIntegrityError("ledger metadata keys differ from schema")
        if not _GIT_COMMIT_PATTERN.fullmatch(metadata["initial_parent_commit"]):
            raise LedgerIntegrityError("ledger has an invalid initial parent commit")
        if not _RECORD_ID_PATTERN.fullmatch(metadata["ledger_id"]):
            raise LedgerIntegrityError("ledger has an invalid ledger ID")
        if metadata["record_schema_version"] != str(RECORD_SCHEMA_VERSION):
            raise LedgerIntegrityError("ledger record schema is unsupported")
        if not _SHA256_PATTERN.fullmatch(metadata["genesis_sha256"]):
            raise LedgerIntegrityError("ledger has an invalid genesis hash")
        if metadata["genesis_sha256"] != _metadata_genesis(metadata):
            raise LedgerIntegrityError("ledger genesis hash does not match metadata")
        _validate_timestamp(metadata["created_at"])
        return metadata

    @staticmethod
    def _assert_schema(connection: sqlite3.Connection) -> None:
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if user_version != LEDGER_SCHEMA_VERSION:
            raise LedgerIntegrityError(
                f"ledger schema is {user_version}; expected {LEDGER_SCHEMA_VERSION}"
            )
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        if application_id != APPLICATION_ID:
            raise LedgerIntegrityError("file is not an Autodidact experiment ledger")
        objects = connection.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'trigger')"
        ).fetchall()
        tables = {str(row["name"]) for row in objects if row["type"] == "table"}
        triggers = {str(row["name"]) for row in objects if row["type"] == "trigger"}
        if not _REQUIRED_TABLES.issubset(tables):
            raise LedgerIntegrityError("ledger is missing required tables")
        if not _REQUIRED_TRIGGERS.issubset(triggers):
            raise LedgerIntegrityError("ledger is missing append-only triggers")

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> LedgerEvent:
        try:
            envelope = json.loads(str(row["payload_json"]))
            record = record_from_envelope(envelope)
            role = WriterRole(str(row["writer_role"]))
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise LedgerIntegrityError(
                f"invalid event payload at sequence {row['sequence']}"
            ) from error
        return LedgerEvent(
            sequence=int(row["sequence"]),
            record=record,
            writer_role=role,
            recorded_at=str(row["recorded_at"]),
            payload_sha256=str(row["payload_sha256"]),
            previous_event_sha256=str(row["previous_event_sha256"]),
            event_sha256=str(row["event_sha256"]),
        )

    @classmethod
    def _verify_connection(
        cls,
        connection: sqlite3.Connection,
        *,
        verify_semantics: bool = True,
    ) -> tuple[LedgerVerification, list[LedgerEvent]]:
        cls._assert_schema(connection)
        metadata = cls._metadata(connection)
        rows = connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        events: list[LedgerEvent] = []
        previous = metadata["genesis_sha256"]
        for expected_sequence, row in enumerate(rows, start=1):
            if int(row["sequence"]) != expected_sequence:
                raise LedgerIntegrityError("ledger event sequence is not contiguous")
            payload_json = str(row["payload_json"])
            payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            if payload_hash != row["payload_sha256"]:
                raise LedgerIntegrityError(f"payload hash mismatch at sequence {expected_sequence}")
            try:
                envelope = json.loads(payload_json)
            except json.JSONDecodeError as error:
                raise LedgerIntegrityError(
                    f"payload JSON is invalid at sequence {expected_sequence}"
                ) from error
            try:
                canonical_payload = canonical_json_bytes(envelope).decode("ascii")
                if payload_json != canonical_payload:
                    raise LedgerIntegrityError(
                        f"payload JSON is not canonical at sequence {expected_sequence}"
                    )
                record = record_from_envelope(envelope)
                _assert_portable_record(envelope)
            except (TypeError, ValueError, LedgerStateError) as error:
                raise LedgerIntegrityError(
                    f"invalid evidence record at sequence {expected_sequence}: {error}"
                ) from error
            if row["record_id"] != record_id(record) or row["record_type"] != record.RECORD_TYPE:
                raise LedgerIntegrityError(
                    f"record columns mismatch payload at sequence {expected_sequence}"
                )
            if row["previous_event_sha256"] != previous:
                raise LedgerIntegrityError(f"hash chain breaks at sequence {expected_sequence}")
            try:
                role = WriterRole(str(row["writer_role"]))
                cls._assert_writer(role, record)
            except (ValueError, LedgerPermissionError) as error:
                raise LedgerIntegrityError(
                    f"invalid writer role at sequence {expected_sequence}: {error}"
                ) from error
            recorded_at = str(row["recorded_at"])
            _validate_timestamp(recorded_at)
            expected_hash = _event_hash(
                sequence=expected_sequence,
                record=envelope,
                writer_role=role.value,
                recorded_at=recorded_at,
                payload_sha256=payload_hash,
                previous_event_sha256=previous,
            )
            if row["event_sha256"] != expected_hash:
                raise LedgerIntegrityError(f"event hash mismatch at sequence {expected_sequence}")
            event = cls._row_to_event(row)
            events.append(event)
            previous = expected_hash

        if verify_semantics and events:
            cls._verify_semantic_history(metadata, events)
        verification = LedgerVerification(
            ledger_id=metadata["ledger_id"],
            initial_parent_commit=metadata["initial_parent_commit"],
            event_count=len(events),
            head_event_sha256=previous,
            schema_version=LEDGER_SCHEMA_VERSION,
            record_schema_version=RECORD_SCHEMA_VERSION,
        )
        return verification, events

    @classmethod
    def _verify_semantic_history(
        cls,
        metadata: dict[str, str],
        events: list[LedgerEvent],
    ) -> None:
        replay = sqlite3.connect(":memory:", isolation_level=None)
        replay.row_factory = sqlite3.Row
        try:
            cls._migrate_connection(replay)
            replay.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )
            for event in events:
                cls._validate_transition(replay, event.record)
                envelope = record_to_envelope(event.record)
                payload_json = canonical_json_bytes(envelope).decode("ascii")
                replay.execute(
                    """
                    INSERT INTO events(
                        sequence, record_id, record_type, writer_role, recorded_at,
                        payload_json, payload_sha256, previous_event_sha256, event_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.sequence,
                        record_id(event.record),
                        event.record.RECORD_TYPE,
                        event.writer_role.value,
                        event.recorded_at,
                        payload_json,
                        event.payload_sha256,
                        event.previous_event_sha256,
                        event.event_sha256,
                    ),
                )
            cls._validate_allocation_links(replay)
        except LedgerError as error:
            raise LedgerIntegrityError(f"ledger semantic history is invalid: {error}") from error
        finally:
            replay.close()

    @staticmethod
    def _assert_writer(role: WriterRole, record: ExperimentRecord) -> None:
        if role not in _ALLOWED_WRITERS[record.RECORD_TYPE]:
            raise LedgerPermissionError(f"{role.value} cannot append {record.RECORD_TYPE} records")

    def verify(self) -> LedgerVerification:
        connection = self._connect()
        try:
            verification, events = self._verify_connection(connection)
            self._remember_verified_snapshot(connection, verification, events)
            return verification
        finally:
            connection.close()
            self._refresh_snapshot_fingerprint()

    def events(self) -> tuple[LedgerEvent, ...]:
        connection = self._connect()
        try:
            _verification, events = self._verified_connection(connection)
            return tuple(events)
        finally:
            connection.close()
            self._refresh_snapshot_fingerprint()

    def append(
        self,
        record: ExperimentRecord,
        *,
        writer_role: WriterRole,
        recorded_at: str | None = None,
    ) -> LedgerEvent:
        return self.append_many(
            ((record, writer_role),),
            recorded_at=recorded_at,
        )[0]

    def ensure(
        self,
        record: ExperimentRecord,
        *,
        writer_role: WriterRole,
    ) -> LedgerEvent:
        return self.append_many(((record, writer_role),), idempotent=True)[0]

    def append_many(
        self,
        entries: Sequence[tuple[ExperimentRecord, WriterRole]],
        *,
        recorded_at: str | None = None,
        idempotent: bool = False,
    ) -> tuple[LedgerEvent, ...]:
        if not entries:
            raise ValueError("at least one ledger entry is required")
        if recorded_at is not None:
            _validate_timestamp(recorded_at)
        connection = self._connect(write=True)
        try:
            connection.execute("BEGIN IMMEDIATE")
            verification, existing_events = self._verified_connection(connection)
            previous = verification.head_event_sha256
            next_sequence = verification.event_count + 1
            appended: list[LedgerEvent] = []
            newly_appended: list[LedgerEvent] = []
            for record, role in entries:
                self._assert_writer(role, record)
                envelope = record_to_envelope(record)
                _assert_portable_record(envelope)
                existing = self._event_by_record_id(connection, record_id(record))
                if existing is not None:
                    if idempotent and existing.record == record and existing.writer_role is role:
                        appended.append(existing)
                        continue
                    raise LedgerConflictError(f"record ID already exists: {record_id(record)}")
                self._validate_transition(connection, record)
                event_time = recorded_at or _utc_now()
                payload_json = canonical_json_bytes(envelope).decode("ascii")
                payload_hash = hashlib.sha256(payload_json.encode("ascii")).hexdigest()
                event_hash = _event_hash(
                    sequence=next_sequence,
                    record=envelope,
                    writer_role=role.value,
                    recorded_at=event_time,
                    payload_sha256=payload_hash,
                    previous_event_sha256=previous,
                )
                try:
                    connection.execute(
                        """
                        INSERT INTO events(
                            sequence, record_id, record_type, writer_role, recorded_at,
                            payload_json, payload_sha256, previous_event_sha256, event_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            next_sequence,
                            record_id(record),
                            record.RECORD_TYPE,
                            role.value,
                            event_time,
                            payload_json,
                            payload_hash,
                            previous,
                            event_hash,
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise LedgerConflictError(f"cannot append record: {error}") from error
                event = LedgerEvent(
                    sequence=next_sequence,
                    record=record,
                    writer_role=role,
                    recorded_at=event_time,
                    payload_sha256=payload_hash,
                    previous_event_sha256=previous,
                    event_sha256=event_hash,
                )
                appended.append(event)
                newly_appended.append(event)
                previous = event_hash
                next_sequence += 1
            self._validate_allocation_links(connection)
            connection.commit()
            updated_verification = LedgerVerification(
                ledger_id=verification.ledger_id,
                initial_parent_commit=verification.initial_parent_commit,
                event_count=verification.event_count + len(newly_appended),
                head_event_sha256=previous,
                schema_version=verification.schema_version,
                record_schema_version=verification.record_schema_version,
            )
            self._remember_verified_snapshot(
                connection,
                updated_verification,
                (*existing_events, *newly_appended),
            )
            return tuple(appended)
        except Exception:
            connection.rollback()
            self._verified_snapshot = None
            raise
        finally:
            connection.close()
            self._refresh_snapshot_fingerprint()

    @classmethod
    def _event_by_record_id(
        cls,
        connection: sqlite3.Connection,
        requested_id: str,
    ) -> LedgerEvent | None:
        row = connection.execute(
            "SELECT * FROM events WHERE record_id = ?", (requested_id,)
        ).fetchone()
        return None if row is None else cls._row_to_event(row)

    @classmethod
    def _require_record(
        cls,
        connection: sqlite3.Connection,
        requested_id: str,
        expected_type: type[Any],
    ) -> Any:
        event = cls._event_by_record_id(connection, requested_id)
        if event is None or not isinstance(event.record, expected_type):
            raise LedgerStateError(
                f"required {expected_type.RECORD_TYPE} record does not exist: {requested_id}"
            )
        return event.record

    @classmethod
    def _records_of_type(
        cls,
        connection: sqlite3.Connection,
        expected_type: type[Any],
    ) -> list[Any]:
        rows = connection.execute(
            "SELECT * FROM events WHERE record_type = ? ORDER BY sequence",
            (expected_type.RECORD_TYPE,),
        ).fetchall()
        records = []
        for row in rows:
            event = cls._row_to_event(row)
            if not isinstance(event.record, expected_type):
                raise LedgerIntegrityError("record_type column does not match its payload")
            records.append(event.record)
        return records

    @classmethod
    def _current_lineage(
        cls,
        connection: sqlite3.Connection,
    ) -> tuple[str, LineageRecord | None]:
        metadata = cls._metadata(connection)
        current = metadata["initial_parent_commit"]
        latest: LineageRecord | None = None
        for lineage in cls._records_of_type(connection, LineageRecord):
            current = lineage.candidate_commit
            latest = lineage
        return current, latest

    @classmethod
    def _validate_transition(
        cls,
        connection: sqlite3.Connection,
        record: ExperimentRecord,
    ) -> None:
        current_parent, latest_lineage = cls._current_lineage(connection)

        if isinstance(record, PatchProposal):
            if record.parent_commit != current_parent:
                raise LedgerStateError("proposal is based on a stale parent")
            return

        if isinstance(record, CandidateRecord):
            proposal = cls._require_record(connection, record.proposal_id, PatchProposal)
            if (
                proposal.parent_commit != record.parent_commit
                or record.parent_commit != current_parent
            ):
                raise LedgerStateError("candidate parent does not match its proposal and lineage")
            existing = [
                item
                for item in cls._records_of_type(connection, CandidateRecord)
                if item.proposal_id == record.proposal_id
            ]
            if existing:
                raise LedgerStateError("a proposal can produce only one immutable candidate")
            return

        if isinstance(record, TrialSchedule):
            candidate = cls._require_record(connection, record.candidate_id, CandidateRecord)
            if (
                record.parent_commit != candidate.parent_commit
                or record.parent_commit != current_parent
            ):
                raise LedgerStateError("trial schedule is based on a stale candidate parent")
            decisions = [
                item
                for item in cls._records_of_type(connection, DecisionRecord)
                if item.candidate_id == record.candidate_id
            ]
            if any(
                item.verdict in {DecisionVerdict.REJECT, DecisionVerdict.PROMOTE}
                for item in decisions
            ):
                raise LedgerStateError("terminal candidates cannot receive new schedules")
            if any(item.stage is record.stage for item in decisions):
                raise LedgerStateError("a decided stage cannot receive another schedule")
            schedules = [
                item
                for item in cls._records_of_type(connection, TrialSchedule)
                if item.candidate_id == record.candidate_id
            ]
            used_seeds = {
                seed for item in schedules if item.stage is record.stage for seed in item.seeds
            }
            used_seeds.update(
                item.seed
                for item in cls._records_of_type(connection, TrialSpec)
                if item.candidate_id == record.candidate_id and item.stage is record.stage
            )
            if used_seeds.intersection(record.seeds):
                raise LedgerStateError("trial schedule repeats an assigned stage seed")
            source_effect = None
            if record.source_effect_estimate_id is not None:
                source_effect = cls._require_record(
                    connection,
                    record.source_effect_estimate_id,
                    EffectEstimate,
                )
                if source_effect.candidate_id != record.candidate_id:
                    raise LedgerStateError(
                        "trial schedule source effect belongs to another candidate"
                    )
                if _STAGE_ORDER[source_effect.stage] > _STAGE_ORDER[record.stage]:
                    raise LedgerStateError("trial schedule source effect is from a later stage")
            if not schedules:
                if record.stage is not ExperimentStage.CHEAP:
                    raise LedgerStateError("a candidate's first schedule must be cheap")
                if source_effect is not None:
                    raise LedgerStateError("the initial cheap schedule cannot cite an effect")
                return
            latest_stage = max(schedules, key=lambda item: _STAGE_ORDER[item.stage]).stage
            if record.stage is latest_stage:
                if source_effect is None or source_effect.stage is not record.stage:
                    raise LedgerStateError(
                        "an additional same-stage schedule requires its latest effect"
                    )
                return
            expected_order = _STAGE_ORDER[latest_stage] + 1
            if _STAGE_ORDER[record.stage] != expected_order:
                raise LedgerStateError("trial schedules must advance one stage at a time")
            escalation = next(
                (
                    item
                    for item in reversed(decisions)
                    if item.stage is latest_stage
                    and item.verdict is DecisionVerdict.ESCALATE
                    and item.next_stage is record.stage
                ),
                None,
            )
            if (
                escalation is None
                or record.source_effect_estimate_id != escalation.effect_estimate_id
            ):
                raise LedgerStateError("later-stage schedules require their escalation decision")
            return

        if isinstance(record, TrialSpec):
            candidate = cls._require_record(connection, record.candidate_id, CandidateRecord)
            if (
                record.parent_commit != candidate.parent_commit
                or record.candidate_commit != candidate.candidate_commit
                or record.candidate_trainer_sha256 != candidate.trainer_sha256
            ):
                raise LedgerStateError("trial does not match its candidate contract")
            if record.parent_commit != current_parent:
                raise LedgerStateError("cannot schedule a new trial against a stale parent")
            schedules = [
                item
                for item in cls._records_of_type(connection, TrialSchedule)
                if item.candidate_id == record.candidate_id and item.stage is record.stage
            ]
            if schedules and not any(
                record.seed in schedule.seeds
                and record.token_budget == schedule.token_budget
                and record.eval_tokens == schedule.eval_tokens
                and record.batch_size == schedule.batch_size
                and record.eval_batch_size == schedule.eval_batch_size
                and record.limits == schedule.limits
                for schedule in schedules
            ):
                raise LedgerStateError("trial does not match its protected stage schedule")
            duplicates = [
                item
                for item in cls._records_of_type(connection, TrialSpec)
                if (
                    item.candidate_id,
                    item.stage,
                    item.seed,
                )
                == (record.candidate_id, record.stage, record.seed)
            ]
            if duplicates:
                raise LedgerStateError("candidate already has this stage and seed trial")
            return

        if isinstance(record, RunResult):
            trial = cls._require_record(connection, record.trial_id, TrialSpec)
            candidate = cls._require_record(connection, trial.candidate_id, CandidateRecord)
            if record.seed != trial.seed or record.target_tokens != trial.token_budget:
                raise LedgerStateError("run seed or budget does not match its trial")
            if trial.eval_tokens is not None and record.evaluation_tokens > trial.eval_tokens:
                raise LedgerStateError("run exceeds its trial evaluation budget")
            if record.parameter_count > trial.limits.max_parameter_count:
                raise LedgerStateError("run exceeds the trial parameter limit")
            if (
                record.arm is RunArm.CANDIDATE
                and record.parameter_count != candidate.parameter_count
            ):
                raise LedgerStateError(
                    "candidate run parameter count differs from candidate record"
                )
            duplicates = [
                item
                for item in cls._records_of_type(connection, RunResult)
                if item.trial_id == record.trial_id and item.arm is record.arm
            ]
            if duplicates:
                raise LedgerStateError("trial already has a result for this arm")
            if any(
                item.trial_id == record.trial_id
                for item in cls._records_of_type(connection, PairedResult)
            ):
                raise LedgerStateError("cannot append a run after a paired result")
            return

        if isinstance(record, ArtifactManifest):
            run = cls._require_record(connection, record.run_id, RunResult)
            if any(
                item.run_id == record.run_id
                for item in cls._records_of_type(connection, ArtifactManifest)
            ):
                raise LedgerStateError("run already has an artifact manifest")
            if run.status is RunStatus.SUCCEEDED:
                kinds = {artifact.kind for artifact in record.artifacts}
                if not {"checkpoint", "metrics"}.issubset(kinds):
                    raise LedgerStateError(
                        "successful runs require checkpoint and metrics artifacts"
                    )
            return

        if isinstance(record, PairedResult):
            cls._validate_pair(connection, record)
            return

        if isinstance(record, EffectEstimate):
            candidate = cls._require_record(connection, record.candidate_id, CandidateRecord)
            proposal = cls._require_record(connection, candidate.proposal_id, PatchProposal)
            if record.minimum_useful_gain_bpb != proposal.minimum_useful_gain_bpb:
                raise LedgerStateError("effect estimate minimum differs from proposal contract")
            pairs = tuple(
                cls._require_record(connection, pair_id, PairedResult)
                for pair_id in record.paired_result_ids
            )
            for pair in pairs:
                trial = cls._require_record(connection, pair.trial_id, TrialSpec)
                if pair.candidate_id != record.candidate_id or trial.stage is not record.stage:
                    raise LedgerStateError(
                        "effect estimate combines mismatched candidates or stages"
                    )
            expected = build_effect_estimate(
                record.estimate_id,
                candidate_id=record.candidate_id,
                stage=record.stage,
                pairs=pairs,
                minimum_useful_gain_bpb=record.minimum_useful_gain_bpb,
                probability_exceeds_minimum=record.probability_exceeds_minimum,
                estimator_version=record.estimator_version,
            )
            if expected != record:
                raise LedgerStateError("effect estimate statistics do not match paired evidence")
            return

        if isinstance(record, DownstreamPrediction):
            candidate = cls._require_record(connection, record.candidate_id, CandidateRecord)
            proposal = cls._require_record(connection, candidate.proposal_id, PatchProposal)
            if record.minimum_useful_gain_bpb != proposal.minimum_useful_gain_bpb:
                raise LedgerStateError("prediction minimum differs from proposal contract")
            trials = tuple(
                cls._require_record(connection, trial_id, TrialSpec)
                for trial_id in record.source_trial_ids
            )
            if any(trial.candidate_id != record.candidate_id for trial in trials):
                raise LedgerStateError("downstream prediction combines candidates")
            paired_trial_ids = {
                pair.trial_id
                for pair in cls._records_of_type(connection, PairedResult)
                if pair.candidate_id == record.candidate_id
            }
            if any(trial.trial_id not in paired_trial_ids for trial in trials):
                raise LedgerStateError(
                    "downstream prediction sources require completed paired results"
                )
            actual_stages = {trial.stage for trial in trials}
            if actual_stages != set(record.source_stages):
                raise LedgerStateError("downstream prediction source stages do not match trials")
            if _STAGE_ORDER[record.target_stage] <= max(
                _STAGE_ORDER[stage] for stage in actual_stages
            ):
                raise LedgerStateError(
                    "downstream prediction target must be later than its sources"
                )
            return

        if isinstance(record, DownstreamAllocation):
            cls._validate_downstream_allocation(connection, record, current_parent)
            return

        if isinstance(record, DecisionRecord):
            cls._validate_decision(connection, record, current_parent)
            return

        if isinstance(record, LineageRecord):
            decision = cls._require_record(connection, record.decision_id, DecisionRecord)
            candidate = cls._require_record(connection, record.candidate_id, CandidateRecord)
            if decision.verdict is not DecisionVerdict.PROMOTE:
                raise LedgerStateError("lineage records require a promotion decision")
            if decision.candidate_id != record.candidate_id:
                raise LedgerStateError("lineage candidate differs from promotion decision")
            if (
                record.parent_commit != current_parent
                or candidate.parent_commit != current_parent
                or record.candidate_commit != candidate.candidate_commit
                or decision.resulting_parent_commit != record.candidate_commit
            ):
                raise LedgerStateError("lineage commits do not match the promoted candidate")
            expected_generation = 1 if latest_lineage is None else latest_lineage.generation + 1
            expected_previous = None if latest_lineage is None else latest_lineage.lineage_id
            if (
                record.generation != expected_generation
                or record.previous_lineage_id != expected_previous
            ):
                raise LedgerStateError("lineage generation or previous link is invalid")
            if any(
                item.decision_id == record.decision_id
                for item in cls._records_of_type(connection, LineageRecord)
            ):
                raise LedgerStateError("promotion decision already has a lineage record")
            return

        if isinstance(record, ComputeRecord):
            run = cls._require_record(connection, record.run_id, RunResult)
            trial = cls._require_record(connection, record.trial_id, TrialSpec)
            if run.trial_id != trial.trial_id:
                raise LedgerStateError("compute record run and trial differ")
            if record.device != trial.device:
                raise LedgerStateError("compute device differs from trial")
            if (
                record.training_tokens != run.tokens_seen
                or record.evaluation_tokens != run.evaluation_tokens
            ):
                raise LedgerStateError("compute token accounting differs from run result")
            if not math.isclose(record.wall_seconds, run.wall_seconds, rel_tol=0.0, abs_tol=1e-9):
                raise LedgerStateError("compute wall time differs from run result")
            if any(
                item.run_id == record.run_id
                for item in cls._records_of_type(connection, ComputeRecord)
            ):
                raise LedgerStateError("run already has a compute record")
            return

        raise LedgerStateError(f"unsupported record transition: {type(record).__name__}")

    @classmethod
    def _resource_failures(
        cls,
        trial: TrialSpec,
        parent: RunResult,
        candidate: RunResult,
    ) -> tuple[str, ...]:
        return resource_constraint_failures(trial, parent, candidate)

    @classmethod
    def _validate_pair(
        cls,
        connection: sqlite3.Connection,
        record: PairedResult,
    ) -> None:
        trial = cls._require_record(connection, record.trial_id, TrialSpec)
        if record.candidate_id != trial.candidate_id or record.seed != trial.seed:
            raise LedgerStateError("paired result differs from trial candidate or seed")
        if any(
            item.trial_id == record.trial_id
            for item in cls._records_of_type(connection, PairedResult)
        ):
            raise LedgerStateError("trial already has a paired result")
        parent = cls._require_record(connection, record.parent_run_id, RunResult)
        candidate = cls._require_record(connection, record.candidate_run_id, RunResult)
        if parent.data_order_sha256 != candidate.data_order_sha256:
            raise LedgerStateError("paired runs used different seeded data orders")
        for run in (parent, candidate):
            manifests = [
                item
                for item in cls._records_of_type(connection, ArtifactManifest)
                if item.run_id == run.run_id
            ]
            if len(manifests) != 1:
                raise LedgerStateError("paired runs require verified artifact manifests")
        failures = cls._resource_failures(trial, parent, candidate)
        expected = build_paired_result(
            record.paired_result_id,
            trial=trial,
            candidate_id=record.candidate_id,
            parent=parent,
            candidate=candidate,
            constraint_failures=failures,
        )
        if expected != record:
            raise LedgerStateError("paired result does not match protected run outcomes")

    @classmethod
    def _allocation_for_decision(
        cls,
        connection: sqlite3.Connection,
        decision_id: str,
    ) -> DownstreamAllocation | None:
        matches = [
            item
            for item in cls._records_of_type(connection, DownstreamAllocation)
            if item.planned_decision_id == decision_id
        ]
        if len(matches) > 1:
            raise LedgerStateError("multiple allocations authorize the same decision")
        return matches[0] if matches else None

    @classmethod
    def _validate_downstream_allocation(
        cls,
        connection: sqlite3.Connection,
        record: DownstreamAllocation,
        current_parent: str,
    ) -> None:
        candidate = cls._require_record(connection, record.candidate_id, CandidateRecord)
        proposal = cls._require_record(connection, candidate.proposal_id, PatchProposal)
        if candidate.parent_commit != current_parent:
            raise LedgerStateError("cannot allocate compute for a stale candidate")
        decisions = [
            item
            for item in cls._records_of_type(connection, DecisionRecord)
            if item.candidate_id == record.candidate_id
        ]
        if any(
            item.verdict in {DecisionVerdict.REJECT, DecisionVerdict.PROMOTE} for item in decisions
        ):
            raise LedgerStateError("terminal candidates cannot receive allocations")
        if any(item.stage is record.stage for item in decisions):
            raise LedgerStateError("a decided stage cannot receive an allocation")

        prior_allocations = [
            item
            for item in cls._records_of_type(connection, DownstreamAllocation)
            if item.candidate_id == record.candidate_id
        ]
        if any(
            item.effect_estimate_id == record.effect_estimate_id
            or item.downstream_prediction_id == record.downstream_prediction_id
            for item in prior_allocations
        ):
            raise LedgerStateError("early evidence already has a downstream allocation")
        planned_ids = {
            item
            for allocation in prior_allocations
            for item in (allocation.planned_decision_id, allocation.planned_schedule_id)
            if item is not None
        }
        if any(
            item in planned_ids
            for item in (record.planned_decision_id, record.planned_schedule_id)
            if item is not None
        ):
            raise LedgerStateError("allocation plan reuses an authorized record ID")

        schedules = [
            item
            for item in cls._records_of_type(connection, TrialSchedule)
            if item.candidate_id == record.candidate_id
        ]
        if (
            not schedules
            or max(
                schedules,
                key=lambda item: _STAGE_ORDER[item.stage],
            ).stage
            is not ExperimentStage.INTERMEDIATE
        ):
            raise LedgerStateError("downstream allocation requires intermediate schedules")
        if any(item.policy_sha256 != record.policy_sha256 for item in schedules):
            raise LedgerStateError("allocation policy differs from protected schedules")

        effect = cls._require_record(connection, record.effect_estimate_id, EffectEstimate)
        if (
            effect.candidate_id != record.candidate_id
            or effect.stage is not record.stage
            or not effect.constraints_passed
        ):
            raise LedgerStateError("allocation requires safe matching intermediate evidence")
        if effect.minimum_useful_gain_bpb != proposal.minimum_useful_gain_bpb:
            raise LedgerStateError("allocation effect differs from proposal contract")

        pairs = [
            item
            for item in cls._records_of_type(connection, PairedResult)
            if item.candidate_id == record.candidate_id
        ]
        pair_trials = {
            pair.paired_result_id: cls._require_record(
                connection,
                pair.trial_id,
                TrialSpec,
            )
            for pair in pairs
        }
        intermediate_pair_ids = {
            pair_id
            for pair_id, trial in pair_trials.items()
            if trial.stage is ExperimentStage.INTERMEDIATE
        }
        if set(effect.paired_result_ids) != intermediate_pair_ids:
            raise LedgerStateError("allocation effect omits current intermediate evidence")

        prediction = cls._require_record(
            connection,
            record.downstream_prediction_id,
            DownstreamPrediction,
        )
        if (
            prediction.candidate_id != record.candidate_id
            or prediction.target_stage is not ExperimentStage.FULL
            or ExperimentStage.INTERMEDIATE not in prediction.source_stages
        ):
            raise LedgerStateError("allocation prediction does not target this full experiment")
        early_trial_ids = {
            pair_trials[pair.paired_result_id].trial_id
            for pair in pairs
            if pair_trials[pair.paired_result_id].stage
            in {ExperimentStage.CHEAP, ExperimentStage.INTERMEDIATE}
        }
        if set(prediction.source_trial_ids) != early_trial_ids:
            raise LedgerStateError("allocation prediction omits current early evidence")
        if prediction.minimum_useful_gain_bpb != proposal.minimum_useful_gain_bpb:
            raise LedgerStateError("allocation prediction differs from proposal contract")
        if prediction.full_budget_label_count < record.minimum_label_count:
            raise LedgerStateError("allocation prediction is not sufficiently calibrated")

        assignment_sha256, audit_score = downstream_audit_assignment(
            record.candidate_id,
            record.policy_sha256,
        )
        if record.audit_assignment_sha256 != assignment_sha256 or record.audit_score != audit_score:
            raise LedgerStateError("allocation audit assignment is not deterministic")

        probability = prediction.probability_exceeds_minimum
        if probability <= record.rejection_probability:
            expected = (
                AllocationAction.AUDIT_FULL
                if audit_score < record.audit_fraction
                else AllocationAction.STOP
            )
            if record.action is not expected:
                raise LedgerStateError("low-probability allocation violates protected auditing")
        elif probability >= record.full_test_probability:
            if record.action is not AllocationAction.RUN_FULL:
                raise LedgerStateError("high-probability allocation must run the full stage")
        elif record.action not in {
            AllocationAction.GATHER_MORE,
            AllocationAction.UNCERTAIN_FULL,
        }:
            raise LedgerStateError("uncertain allocation must gather evidence or run full")

        if record.action is AllocationAction.GATHER_MORE:
            used_seeds = {
                seed
                for schedule in schedules
                if schedule.stage is ExperimentStage.INTERMEDIATE
                for seed in schedule.seeds
            }
            if record.next_seed in used_seeds:
                raise LedgerStateError("allocation repeats an intermediate seed")

    @classmethod
    def _validate_allocation_links(cls, connection: sqlite3.Connection) -> None:
        for allocation in cls._records_of_type(connection, DownstreamAllocation):
            allocation_event = cls._event_by_record_id(connection, allocation.allocation_id)
            assert allocation_event is not None
            if allocation.planned_decision_id is not None:
                decision_event = cls._event_by_record_id(
                    connection,
                    allocation.planned_decision_id,
                )
                if decision_event is None or not isinstance(decision_event.record, DecisionRecord):
                    raise LedgerStateError("allocation planned decision is missing")
                if decision_event.sequence <= allocation_event.sequence:
                    raise LedgerStateError("allocation must precede its planned decision")
                if decision_event.payload_sha256 != allocation.planned_decision_sha256:
                    raise LedgerStateError("allocation planned decision hash differs")
                decision = decision_event.record
                expected_verdict = (
                    DecisionVerdict.REJECT
                    if allocation.action is AllocationAction.STOP
                    else DecisionVerdict.ESCALATE
                )
                expected_threshold = {
                    AllocationAction.STOP: allocation.rejection_probability,
                    AllocationAction.RUN_FULL: allocation.full_test_probability,
                    AllocationAction.AUDIT_FULL: 0.0,
                    AllocationAction.UNCERTAIN_FULL: 0.0,
                }[allocation.action]
                if (
                    decision.candidate_id != allocation.candidate_id
                    or decision.stage is not allocation.stage
                    or decision.verdict is not expected_verdict
                    or decision.effect_estimate_id != allocation.effect_estimate_id
                    or decision.downstream_prediction_id != allocation.downstream_prediction_id
                    or decision.probability_threshold != expected_threshold
                    or not decision.constraints_passed
                    or decision.next_stage is not allocation.next_stage
                ):
                    raise LedgerStateError(
                        "allocation planned decision does not match its evidence"
                    )

            if allocation.planned_schedule_id is not None:
                schedule_event = cls._event_by_record_id(
                    connection,
                    allocation.planned_schedule_id,
                )
                if schedule_event is None or not isinstance(schedule_event.record, TrialSchedule):
                    raise LedgerStateError("allocation planned schedule is missing")
                if schedule_event.sequence <= allocation_event.sequence:
                    raise LedgerStateError("allocation must precede its planned schedule")
                if schedule_event.payload_sha256 != allocation.planned_schedule_sha256:
                    raise LedgerStateError("allocation planned schedule hash differs")
                schedule = schedule_event.record
                expected_seeds = (
                    (allocation.next_seed,)
                    if allocation.action is AllocationAction.GATHER_MORE
                    else schedule.seeds
                )
                if (
                    schedule.candidate_id != allocation.candidate_id
                    or schedule.stage is not allocation.next_stage
                    or schedule.seeds != expected_seeds
                    or schedule.source_effect_estimate_id != allocation.effect_estimate_id
                    or schedule.policy_sha256 != allocation.policy_sha256
                ):
                    raise LedgerStateError(
                        "allocation planned schedule does not match its evidence"
                    )

    @classmethod
    def _validate_decision(
        cls,
        connection: sqlite3.Connection,
        record: DecisionRecord,
        current_parent: str,
    ) -> None:
        candidate = cls._require_record(connection, record.candidate_id, CandidateRecord)
        proposal = cls._require_record(connection, candidate.proposal_id, PatchProposal)
        if record.effect_estimate_id is None:
            effect = None
            if record.verdict is not DecisionVerdict.REJECT or record.constraints_passed:
                raise LedgerStateError(
                    "only a constraint-failure rejection may omit an effect estimate"
                )
            stage_trial_ids = {
                trial.trial_id
                for trial in cls._records_of_type(connection, TrialSpec)
                if trial.candidate_id == record.candidate_id and trial.stage is record.stage
            }
            if not any(
                run.trial_id in stage_trial_ids and run.status is not RunStatus.SUCCEEDED
                for run in cls._records_of_type(connection, RunResult)
            ):
                raise LedgerStateError("effect-free rejection requires a recorded failed stage run")
        else:
            effect = cls._require_record(connection, record.effect_estimate_id, EffectEstimate)
            if effect.candidate_id != record.candidate_id or effect.stage is not record.stage:
                raise LedgerStateError("decision effect estimate differs from candidate or stage")
            if effect.minimum_useful_gain_bpb != proposal.minimum_useful_gain_bpb:
                raise LedgerStateError("decision effect differs from proposal contract")
            if record.constraints_passed != effect.constraints_passed:
                raise LedgerStateError("decision constraints differ from effect estimate")
        if record.minimum_useful_gain_bpb != proposal.minimum_useful_gain_bpb:
            raise LedgerStateError("decision minimum effect differs from proposal contract")
        if record.downstream_prediction_id is not None:
            prediction = cls._require_record(
                connection,
                record.downstream_prediction_id,
                DownstreamPrediction,
            )
            if prediction.candidate_id != record.candidate_id:
                raise LedgerStateError(
                    "decision downstream prediction belongs to another candidate"
                )
            if prediction.minimum_useful_gain_bpb != proposal.minimum_useful_gain_bpb:
                raise LedgerStateError(
                    "decision downstream prediction differs from proposal contract"
                )
        else:
            prediction = None
        allocation = cls._allocation_for_decision(connection, record.decision_id)
        if (
            prediction is not None
            and record.stage is ExperimentStage.INTERMEDIATE
            and record.verdict in {DecisionVerdict.REJECT, DecisionVerdict.ESCALATE}
            and allocation is None
        ):
            raise LedgerStateError(
                "prediction-linked decisions require a protected allocation record"
            )
        if allocation is not None and (
            prediction is None
            or allocation.downstream_prediction_id != prediction.prediction_id
            or allocation.effect_estimate_id != record.effect_estimate_id
        ):
            raise LedgerStateError("decision differs from its protected allocation")

        prior = [
            item
            for item in cls._records_of_type(connection, DecisionRecord)
            if item.candidate_id == record.candidate_id
        ]
        if any(item.verdict in {DecisionVerdict.REJECT, DecisionVerdict.PROMOTE} for item in prior):
            raise LedgerStateError("candidate already has a terminal decision")
        if any(item.stage is record.stage for item in prior):
            raise LedgerStateError("candidate already has a decision for this stage")
        if prior and _STAGE_ORDER[record.stage] <= max(_STAGE_ORDER[item.stage] for item in prior):
            raise LedgerStateError("candidate decisions must advance through stages")

        if record.verdict is DecisionVerdict.ESCALATE:
            assert effect is not None
            assert record.next_stage is not None
            if _STAGE_ORDER[record.next_stage] <= _STAGE_ORDER[record.stage]:
                raise LedgerStateError("escalation must advance to a later stage")
            if candidate.parent_commit != current_parent:
                raise LedgerStateError("cannot escalate a candidate from a stale parent")
            if (
                allocation is None
                and effect.probability_exceeds_minimum < record.probability_threshold
            ):
                raise LedgerStateError("effect probability does not satisfy escalation threshold")
        elif record.verdict is DecisionVerdict.PROMOTE:
            assert effect is not None
            if candidate.parent_commit != current_parent:
                raise LedgerStateError("cannot promote a candidate from a stale parent")
            if record.resulting_parent_commit != candidate.candidate_commit:
                raise LedgerStateError("promotion parent must be the candidate commit")
            if effect.probability_exceeds_minimum < record.probability_threshold:
                raise LedgerStateError("effect probability does not satisfy promotion threshold")
            if (
                prediction is not None
                and prediction.probability_exceeds_minimum < record.probability_threshold
            ):
                raise LedgerStateError(
                    "downstream probability does not satisfy promotion threshold"
                )

    def get(self, requested_id: str) -> LedgerEvent:
        connection = self._connect()
        try:
            self._verified_connection(connection)
            event = self._event_by_record_id(connection, requested_id)
            if event is None:
                raise LedgerError(f"record does not exist: {requested_id}")
            return event
        finally:
            connection.close()
            self._refresh_snapshot_fingerprint()

    def current_parent(self) -> str:
        connection = self._connect()
        try:
            self._verified_connection(connection)
            parent, _lineage = self._current_lineage(connection)
            return parent
        finally:
            connection.close()
            self._refresh_snapshot_fingerprint()

    def running_trials(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            self._verified_connection(connection)
            return self._running_trial_ids(connection)
        finally:
            connection.close()
            self._refresh_snapshot_fingerprint()

    @classmethod
    def _running_trial_ids(cls, connection: sqlite3.Connection) -> tuple[str, ...]:
        trials = cls._records_of_type(connection, TrialSpec)
        results = cls._records_of_type(connection, RunResult)
        paired_ids = {item.trial_id for item in cls._records_of_type(connection, PairedResult)}
        running = []
        for trial in trials:
            trial_results = [item for item in results if item.trial_id == trial.trial_id]
            terminal_failure = len(trial_results) == 2 and any(
                item.status is not RunStatus.SUCCEEDED for item in trial_results
            )
            if trial.trial_id not in paired_ids and not terminal_failure:
                running.append(trial.trial_id)
        return tuple(running)

    def summary(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            verification, events = self._verified_connection(connection)
            counts = Counter(event.record.RECORD_TYPE for event in events)
            decisions = self._records_of_type(connection, DecisionRecord)
            compute = self._records_of_type(connection, ComputeRecord)
            current_parent, latest_lineage = self._current_lineage(connection)
            return {
                "compute": {
                    "accelerator_seconds": statistics.fsum(
                        item.accelerator_seconds for item in compute
                    ),
                    "estimated_cost_usd": statistics.fsum(
                        item.estimated_cost_usd or 0.0 for item in compute
                    ),
                    "evaluation_tokens": sum(item.evaluation_tokens for item in compute),
                    "training_tokens": sum(item.training_tokens for item in compute),
                    "wall_seconds": statistics.fsum(item.wall_seconds for item in compute),
                },
                "current_parent_commit": current_parent,
                "decision_counts": dict(
                    sorted(Counter(item.verdict.value for item in decisions).items())
                ),
                "event_count": verification.event_count,
                "generation": 0 if latest_lineage is None else latest_lineage.generation,
                "head_event_sha256": verification.head_event_sha256,
                "initial_parent_commit": verification.initial_parent_commit,
                "ledger_id": verification.ledger_id,
                "record_counts": dict(sorted(counts.items())),
                "running_trial_ids": list(self._running_trial_ids(connection)),
                "schema_version": LEDGER_SCHEMA_VERSION,
            }
        finally:
            connection.close()
            self._refresh_snapshot_fingerprint()

    def progress_points(self) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            self._verified_connection(connection)
            candidate_events = [
                event
                for event in self._events_without_verification(connection)
                if isinstance(event.record, CandidateRecord)
            ]
            pairs = self._records_of_type(connection, PairedResult)
            estimates = self._records_of_type(connection, EffectEstimate)
            decisions = self._records_of_type(connection, DecisionRecord)
            lineages = self._records_of_type(connection, LineageRecord)
            points = []
            for index, event in enumerate(candidate_events, start=1):
                candidate = event.record
                assert isinstance(candidate, CandidateRecord)
                candidate_pairs = [
                    item for item in pairs if item.candidate_id == candidate.candidate_id
                ]
                candidate_estimates = [
                    item for item in estimates if item.candidate_id == candidate.candidate_id
                ]
                candidate_decisions = [
                    item for item in decisions if item.candidate_id == candidate.candidate_id
                ]
                latest_decision = candidate_decisions[-1] if candidate_decisions else None
                latest_estimate = candidate_estimates[-1] if candidate_estimates else None
                promoted = next(
                    (item for item in lineages if item.candidate_id == candidate.candidate_id),
                    None,
                )
                points.append(
                    {
                        "candidate_bpb_mean": (
                            statistics.fmean(item.candidate_bpb for item in candidate_pairs)
                            if candidate_pairs
                            else None
                        ),
                        "candidate_id": candidate.candidate_id,
                        "event_sequence": event.sequence,
                        "experiment_index": index,
                        "mean_gain_bpb": (
                            None if latest_estimate is None else latest_estimate.mean_gain_bpb
                        ),
                        "paired_seed_count": len(candidate_pairs),
                        "parent_bpb_mean": (
                            statistics.fmean(item.parent_bpb for item in candidate_pairs)
                            if candidate_pairs
                            else None
                        ),
                        "promoted_parent_commit": (
                            None if promoted is None else promoted.candidate_commit
                        ),
                        "stage": None if latest_estimate is None else latest_estimate.stage.value,
                        "status": (
                            "running" if latest_decision is None else latest_decision.verdict.value
                        ),
                    }
                )
            return points
        finally:
            connection.close()
            self._refresh_snapshot_fingerprint()

    @classmethod
    def _events_without_verification(
        cls,
        connection: sqlite3.Connection,
    ) -> list[LedgerEvent]:
        return [
            cls._row_to_event(row)
            for row in connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        ]

    def export(
        self,
        output_path: Path,
        *,
        output_format: str = "snapshot",
        redactions: Sequence[str] = (),
    ) -> None:
        if output_format not in {"snapshot", "jsonl"}:
            raise ValueError("output_format must be snapshot or jsonl")
        connection = self._connect()
        try:
            verification, events = self._verified_connection(connection)
            metadata = self._metadata(connection)
        finally:
            connection.close()
            self._refresh_snapshot_fingerprint()
        default_redactions = (str(Path.home()), tempfile.gettempdir())
        active_redactions = tuple(
            sorted(
                {item for item in (*default_redactions, *redactions) if item},
                key=len,
                reverse=True,
            )
        )
        exported_events = []
        for event in events:
            envelope = _redact_value(record_to_envelope(event.record), active_redactions)
            exported_events.append(
                {
                    "event_sequence": event.sequence,
                    "export_record_sha256": hashlib.sha256(
                        canonical_json_bytes(envelope)
                    ).hexdigest(),
                    "previous_source_event_sha256": event.previous_event_sha256,
                    "record": envelope,
                    "recorded_at": event.recorded_at,
                    "source_event_sha256": event.event_sha256,
                    "writer_role": event.writer_role.value,
                }
            )
        header = {
            "event_count": verification.event_count,
            "export_schema_version": LEDGER_EXPORT_SCHEMA_VERSION,
            "head_source_event_sha256": verification.head_event_sha256,
            "initial_parent_commit": verification.initial_parent_commit,
            "ledger_id": metadata["ledger_id"],
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "redacted": True,
        }
        if output_format == "snapshot":
            content = (
                json.dumps(
                    {"events": exported_events, "ledger": header},
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        else:
            lines = [json.dumps({"ledger": header}, sort_keys=True, allow_nan=False)]
            lines.extend(
                json.dumps(item, sort_keys=True, allow_nan=False) for item in exported_events
            )
            content = "\n".join(lines) + "\n"
        _atomic_write(output_path.expanduser().resolve(), content)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the protected experiment evidence ledger.")
    parser.add_argument("--path", type=_path, default=DEFAULT_LEDGER_PATH)
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="create a new empty ledger")
    initialize.add_argument("--initial-parent", required=True)
    initialize.add_argument("--ledger-id")

    commands.add_parser("verify", help="verify schema, hash chain, and lifecycle")
    commands.add_parser("summary", help="print a compact ledger summary")

    show = commands.add_parser("show", help="print one evidence record")
    show.add_argument("record_id")

    export = commands.add_parser("export", help="write a sanitized evidence export")
    export.add_argument("--output", type=_path, required=True)
    export.add_argument("--format", choices=("snapshot", "jsonl"), default="snapshot")
    export.add_argument("--redact", action="append", default=[])

    commands.add_parser("migrate", help="apply explicit non-evidence schema migrations")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            ledger = ExperimentLedger.create(
                args.path,
                initial_parent_commit=args.initial_parent,
                ledger_id=args.ledger_id,
            )
            payload: Any = ledger.summary()
        elif args.command == "migrate":
            ExperimentLedger.migrate(args.path)
            payload = asdict(ExperimentLedger.open(args.path, read_only=True).verify())
        else:
            ledger = ExperimentLedger.open(args.path, read_only=True)
            if args.command == "verify":
                verification = ledger.verify()
                payload = {
                    "event_count": verification.event_count,
                    "head_event_sha256": verification.head_event_sha256,
                    "initial_parent_commit": verification.initial_parent_commit,
                    "ledger_id": verification.ledger_id,
                    "record_schema_version": verification.record_schema_version,
                    "schema_version": verification.schema_version,
                    "verified": True,
                }
            elif args.command == "summary":
                payload = ledger.summary()
            elif args.command == "show":
                event = ledger.get(args.record_id)
                payload = {
                    "event_sha256": event.event_sha256,
                    "record": record_to_envelope(event.record),
                    "recorded_at": event.recorded_at,
                    "sequence": event.sequence,
                    "writer_role": event.writer_role.value,
                }
            elif args.command == "export":
                ledger.export(
                    args.output,
                    output_format=args.format,
                    redactions=args.redact,
                )
                payload = {"exported": str(args.output), "format": args.format}
            else:
                raise AssertionError(f"unhandled command: {args.command}")
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (LedgerError, ValueError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
