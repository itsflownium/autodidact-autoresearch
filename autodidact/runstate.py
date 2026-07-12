"""Durable campaign state, replay-safe operation tracking, and repository locking."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from autodidact.data.integrity import canonical_json_bytes

RUN_STATE_SCHEMA_VERSION = 1
APPLICATION_ID = 0x41555352
DEFAULT_STATE_PATH = Path("artifacts/control/campaign.sqlite3")
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PHASE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_LOCK_FILENAME = "autodidact-campaign.lock"

_SCHEMA_SQL = f"""
PRAGMA application_id = {APPLICATION_ID};
PRAGMA user_version = {RUN_STATE_SCHEMA_VERSION};

CREATE TABLE campaign (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    campaign_id TEXT NOT NULL UNIQUE,
    initial_parent_commit TEXT NOT NULL,
    accepted_parent_commit TEXT NOT NULL,
    status TEXT NOT NULL,
    phase TEXT NOT NULL,
    generation INTEGER NOT NULL,
    active_proposal_id TEXT,
    active_candidate_id TEXT,
    started_unix REAL NOT NULL,
    updated_unix REAL NOT NULL,
    control_reason TEXT,
    limits_json TEXT NOT NULL,
    used_proposals INTEGER NOT NULL DEFAULT 0,
    used_researcher_tokens INTEGER NOT NULL DEFAULT 0,
    used_training_tokens INTEGER NOT NULL DEFAULT 0,
    used_compute_seconds REAL NOT NULL DEFAULT 0,
    reserved_proposals INTEGER NOT NULL DEFAULT 0,
    reserved_researcher_tokens INTEGER NOT NULL DEFAULT 0,
    reserved_training_tokens INTEGER NOT NULL DEFAULT 0,
    reserved_compute_seconds REAL NOT NULL DEFAULT 0
);

CREATE TABLE operations (
    operation_key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    result_json TEXT,
    error TEXT,
    started_unix REAL NOT NULL,
    updated_unix REAL NOT NULL
);

CREATE TABLE reservations (
    reservation_id TEXT PRIMARY KEY,
    operation_key TEXT,
    requested_json TEXT NOT NULL,
    actual_json TEXT,
    status TEXT NOT NULL,
    created_unix REAL NOT NULL,
    updated_unix REAL NOT NULL,
    FOREIGN KEY(operation_key) REFERENCES operations(operation_key)
);
"""


class RunStateError(RuntimeError):
    """Raised when durable campaign state cannot make a valid transition."""


class RunStateConflict(RunStateError):
    """Raised when an idempotency key is reused with different inputs."""


class BudgetExceeded(RunStateError):
    """Raised before work that would exceed a campaign limit."""


class RepositoryLocked(RunStateError):
    """Raised when another campaign process owns the repository lock."""


class CampaignStatus(StrEnum):
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class OperationStatus(StrEnum):
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ClaimDisposition(StrEnum):
    EXECUTE = "execute"
    REPLAY = "replay"
    RECOVER = "recover"


class ReservationStatus(StrEnum):
    RESERVED = "reserved"
    SETTLED = "settled"
    RELEASED = "released"


def _validate_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise RunStateError(f"{name} must be a portable structured ID")


def _validate_commit(name: str, value: str) -> None:
    if not isinstance(value, str) or not _COMMIT_PATTERN.fullmatch(value):
        raise RunStateError(f"{name} must be a full lowercase Git commit")


def _validate_reason(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4_000:
        raise RunStateError("reason must be nonempty text of at most 4000 characters")
    return value


def _canonical_object(value: Any, *, name: str) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise RunStateError(f"{name} must be a JSON object")
    try:
        payload = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise RunStateError(f"{name} must contain canonical JSON values") from error
    return payload.decode("ascii"), hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CampaignLimits:
    max_proposals: int
    max_wall_seconds: float
    max_researcher_tokens: int
    max_training_tokens: int
    max_compute_seconds: float
    reward_calibration_labels: int = 0

    def __post_init__(self) -> None:
        for name in ("max_proposals", "max_researcher_tokens", "max_training_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise RunStateError(f"{name} must be a positive integer")
        for name in ("max_wall_seconds", "max_compute_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise RunStateError(f"{name} must be finite and positive")
        if (
            type(self.reward_calibration_labels) is not int
            or self.reward_calibration_labels < 0
            or self.reward_calibration_labels > self.max_proposals
        ):
            raise RunStateError("reward_calibration_labels must be between zero and max_proposals")


@dataclass(frozen=True, slots=True)
class BudgetAmount:
    proposals: int = 0
    researcher_tokens: int = 0
    training_tokens: int = 0
    compute_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("proposals", "researcher_tokens", "training_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise RunStateError(f"{name} must be a nonnegative integer")
        if (
            not isinstance(self.compute_seconds, (int, float))
            or not math.isfinite(self.compute_seconds)
            or self.compute_seconds < 0.0
        ):
            raise RunStateError("compute_seconds must be finite and nonnegative")

    def __add__(self, other: BudgetAmount) -> BudgetAmount:
        return BudgetAmount(
            proposals=self.proposals + other.proposals,
            researcher_tokens=self.researcher_tokens + other.researcher_tokens,
            training_tokens=self.training_tokens + other.training_tokens,
            compute_seconds=self.compute_seconds + other.compute_seconds,
        )

    def __sub__(self, other: BudgetAmount) -> BudgetAmount:
        return BudgetAmount(
            proposals=self.proposals - other.proposals,
            researcher_tokens=self.researcher_tokens - other.researcher_tokens,
            training_tokens=self.training_tokens - other.training_tokens,
            compute_seconds=self.compute_seconds - other.compute_seconds,
        )

    def fits_within(self, other: BudgetAmount) -> bool:
        return (
            self.proposals <= other.proposals
            and self.researcher_tokens <= other.researcher_tokens
            and self.training_tokens <= other.training_tokens
            and self.compute_seconds <= other.compute_seconds
        )

    @classmethod
    def from_json(cls, value: str) -> BudgetAmount:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as error:
            raise RunStateError("stored budget JSON is invalid") from error
        if not isinstance(payload, dict) or set(payload) != {
            "proposals",
            "researcher_tokens",
            "training_tokens",
            "compute_seconds",
        }:
            raise RunStateError("stored budget fields are invalid")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class CampaignSnapshot:
    campaign_id: str
    initial_parent_commit: str
    accepted_parent_commit: str
    status: CampaignStatus
    phase: str
    generation: int
    active_proposal_id: str | None
    active_candidate_id: str | None
    started_unix: float
    updated_unix: float
    elapsed_wall_seconds: float
    control_reason: str | None
    limits: CampaignLimits
    used: BudgetAmount
    reserved: BudgetAmount
    remaining: BudgetAmount


@dataclass(frozen=True, slots=True)
class OperationClaim:
    operation_key: str
    kind: str
    input_sha256: str
    status: OperationStatus
    disposition: ClaimDisposition
    attempts: int
    result: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    operation_key: str | None
    requested: BudgetAmount
    actual: BudgetAmount | None
    status: ReservationStatus


class CampaignStore:
    """Transactional campaign state with durable idempotency and budget reservations."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path.expanduser().resolve()
        self._clock = clock

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        campaign_id: str,
        initial_parent_commit: str,
        limits: CampaignLimits,
        clock: Callable[[], float] = time.time,
    ) -> CampaignStore:
        _validate_id("campaign_id", campaign_id)
        _validate_commit("initial_parent_commit", initial_parent_commit)
        resolved = path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(resolved, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise RunStateConflict(f"campaign state already exists: {resolved}") from error
        os.close(descriptor)
        now = float(clock())
        connection = sqlite3.connect(resolved, timeout=30.0, isolation_level=None)
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(_SCHEMA_SQL)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO campaign(
                    singleton, campaign_id, initial_parent_commit, accepted_parent_commit,
                    status, phase, generation, started_unix, updated_unix, limits_json
                ) VALUES (1, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    campaign_id,
                    initial_parent_commit,
                    initial_parent_commit,
                    CampaignStatus.RUNNING.value,
                    "ready",
                    now,
                    now,
                    canonical_json_bytes(asdict(limits)).decode("ascii"),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            connection.close()
            for candidate in (
                resolved,
                Path(str(resolved) + "-shm"),
                Path(str(resolved) + "-wal"),
            ):
                candidate.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
        store = cls(resolved, clock=clock)
        store.verify()
        return store

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> CampaignStore:
        store = cls(path, clock=clock)
        if not store.path.is_file():
            raise RunStateError(f"campaign state does not exist: {store.path}")
        store.verify()
        return store

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _campaign_row(connection: sqlite3.Connection) -> sqlite3.Row:
        rows = connection.execute("SELECT * FROM campaign").fetchall()
        if len(rows) != 1:
            raise RunStateError("campaign state must contain exactly one campaign")
        return rows[0]

    def verify(self) -> None:
        connection = self._connect()
        try:
            if int(connection.execute("PRAGMA application_id").fetchone()[0]) != APPLICATION_ID:
                raise RunStateError("file is not an Autodidact campaign state store")
            if (
                int(connection.execute("PRAGMA user_version").fetchone()[0])
                != RUN_STATE_SCHEMA_VERSION
            ):
                raise RunStateError("campaign state schema is unsupported")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            if not {"campaign", "operations", "reservations"}.issubset(tables):
                raise RunStateError("campaign state is missing required tables")
            row = self._campaign_row(connection)
            _validate_id("campaign_id", str(row["campaign_id"]))
            _validate_commit("initial_parent_commit", str(row["initial_parent_commit"]))
            _validate_commit("accepted_parent_commit", str(row["accepted_parent_commit"]))
            CampaignStatus(str(row["status"]))
            phase = str(row["phase"])
            if not _PHASE_PATTERN.fullmatch(phase):
                raise RunStateError("campaign phase is invalid")
            if int(row["generation"]) < 0:
                raise RunStateError("campaign generation is invalid")
            for name in ("active_proposal_id", "active_candidate_id"):
                if row[name] is not None:
                    _validate_id(name, str(row[name]))
            limits = CampaignLimits(**json.loads(str(row["limits_json"])))
            used = self._amount_from_row(row, "used")
            reserved = self._amount_from_row(row, "reserved")
            if not (used + reserved).fits_within(self._limit_amount(limits)):
                raise RunStateError("campaign usage exceeds its limits")
            for timestamp in (row["started_unix"], row["updated_unix"]):
                if not math.isfinite(float(timestamp)):
                    raise RunStateError("campaign timestamp is invalid")

            for operation in connection.execute("SELECT * FROM operations"):
                _validate_id("operation_key", str(operation["operation_key"]))
                _validate_id("kind", str(operation["kind"]))
                if not _SHA256_PATTERN.fullmatch(str(operation["input_sha256"])):
                    raise RunStateError("operation input hash is invalid")
                operation_status = OperationStatus(str(operation["status"]))
                if int(operation["attempts"]) <= 0:
                    raise RunStateError("operation attempts must be positive")
                result_json = operation["result_json"]
                error = operation["error"]
                if operation_status is OperationStatus.SUCCEEDED:
                    if result_json is None or error is not None:
                        raise RunStateError("successful operation evidence is invalid")
                    canonical, _digest = _canonical_object(
                        json.loads(str(result_json)),
                        name="operation result",
                    )
                    if canonical != result_json:
                        raise RunStateError("operation result is not canonical")
                elif operation_status is OperationStatus.FAILED:
                    if result_json is not None or error is None:
                        raise RunStateError("failed operation evidence is invalid")
                    _validate_reason(str(error))
                elif result_json is not None:
                    raise RunStateError("unfinished operation cannot have a result")

            expected_used = BudgetAmount()
            expected_reserved = BudgetAmount()
            for reservation_row in connection.execute("SELECT * FROM reservations"):
                reservation = self._reservation_from_row(reservation_row)
                _validate_id("reservation_id", reservation.reservation_id)
                if reservation.operation_key is not None:
                    _validate_id("operation_key", reservation.operation_key)
                requested_json = canonical_json_bytes(asdict(reservation.requested)).decode("ascii")
                if requested_json != reservation_row["requested_json"]:
                    raise RunStateError("reservation request is not canonical")
                if reservation.status is ReservationStatus.RESERVED:
                    if reservation.actual is not None:
                        raise RunStateError("open reservation cannot contain actual usage")
                    expected_reserved += reservation.requested
                elif reservation.status is ReservationStatus.SETTLED:
                    if reservation.actual is None or not reservation.actual.fits_within(
                        reservation.requested
                    ):
                        raise RunStateError("settled reservation usage is invalid")
                    expected_used += reservation.actual
                    actual_json = canonical_json_bytes(asdict(reservation.actual)).decode("ascii")
                    if actual_json != reservation_row["actual_json"]:
                        raise RunStateError("reservation actual usage is not canonical")
                elif reservation.actual is not None:
                    raise RunStateError("released reservation cannot contain actual usage")
            if used != expected_used or reserved != expected_reserved:
                raise RunStateError("campaign counters do not match budget reservations")
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            raise RunStateError("campaign state contains invalid values") from error
        finally:
            connection.close()

    @staticmethod
    def _amount_from_row(row: sqlite3.Row, prefix: str) -> BudgetAmount:
        return BudgetAmount(
            proposals=int(row[f"{prefix}_proposals"]),
            researcher_tokens=int(row[f"{prefix}_researcher_tokens"]),
            training_tokens=int(row[f"{prefix}_training_tokens"]),
            compute_seconds=float(row[f"{prefix}_compute_seconds"]),
        )

    @staticmethod
    def _limits_from_row(row: sqlite3.Row) -> CampaignLimits:
        return CampaignLimits(**json.loads(str(row["limits_json"])))

    def _assert_time_and_running(self, row: sqlite3.Row, now: float) -> None:
        status = CampaignStatus(str(row["status"]))
        if status is not CampaignStatus.RUNNING:
            raise RunStateError(f"campaign cannot start work while {status.value}")
        limits = self._limits_from_row(row)
        if now - float(row["started_unix"]) >= limits.max_wall_seconds:
            raise BudgetExceeded("campaign wall-time limit is exhausted")

    @staticmethod
    def _limit_amount(limits: CampaignLimits) -> BudgetAmount:
        return BudgetAmount(
            proposals=limits.max_proposals,
            researcher_tokens=limits.max_researcher_tokens,
            training_tokens=limits.max_training_tokens,
            compute_seconds=limits.max_compute_seconds,
        )

    def snapshot(self) -> CampaignSnapshot:
        connection = self._connect()
        try:
            row = self._campaign_row(connection)
            limits = self._limits_from_row(row)
            used = self._amount_from_row(row, "used")
            reserved = self._amount_from_row(row, "reserved")
            limit_amount = self._limit_amount(limits)
            return CampaignSnapshot(
                campaign_id=str(row["campaign_id"]),
                initial_parent_commit=str(row["initial_parent_commit"]),
                accepted_parent_commit=str(row["accepted_parent_commit"]),
                status=CampaignStatus(str(row["status"])),
                phase=str(row["phase"]),
                generation=int(row["generation"]),
                active_proposal_id=row["active_proposal_id"],
                active_candidate_id=row["active_candidate_id"],
                started_unix=float(row["started_unix"]),
                updated_unix=float(row["updated_unix"]),
                elapsed_wall_seconds=max(0.0, float(self._clock()) - float(row["started_unix"])),
                control_reason=row["control_reason"],
                limits=limits,
                used=used,
                reserved=reserved,
                remaining=limit_amount - (used + reserved),
            )
        finally:
            connection.close()

    def ensure_can_work(self) -> None:
        connection = self._connect()
        try:
            self._assert_time_and_running(self._campaign_row(connection), float(self._clock()))
        finally:
            connection.close()

    def set_progress(
        self,
        *,
        phase: str,
        accepted_parent_commit: str,
        generation: int,
        active_proposal_id: str | None,
        active_candidate_id: str | None,
    ) -> CampaignSnapshot:
        if not isinstance(phase, str) or not _PHASE_PATTERN.fullmatch(phase):
            raise RunStateError("phase must be a portable snake-case value")
        _validate_commit("accepted_parent_commit", accepted_parent_commit)
        if type(generation) is not int or generation < 0:
            raise RunStateError("generation must be a nonnegative integer")
        for name, value in (
            ("active_proposal_id", active_proposal_id),
            ("active_candidate_id", active_candidate_id),
        ):
            if value is not None:
                _validate_id(name, value)
        now = float(self._clock())
        with self._transaction() as connection:
            row = self._campaign_row(connection)
            self._assert_time_and_running(row, now)
            current_generation = int(row["generation"])
            current_parent = str(row["accepted_parent_commit"])
            if accepted_parent_commit == current_parent:
                if generation != current_generation:
                    raise RunStateError("generation can change only with the accepted parent")
            elif generation != current_generation + 1:
                raise RunStateError("an accepted-parent transition advances exactly one generation")
            connection.execute(
                """
                UPDATE campaign SET phase = ?, accepted_parent_commit = ?, generation = ?,
                    active_proposal_id = ?, active_candidate_id = ?, updated_unix = ?
                WHERE singleton = 1
                """,
                (
                    phase,
                    accepted_parent_commit,
                    generation,
                    active_proposal_id,
                    active_candidate_id,
                    now,
                ),
            )
        return self.snapshot()

    @staticmethod
    def _reservation_from_row(row: sqlite3.Row) -> BudgetReservation:
        return BudgetReservation(
            reservation_id=str(row["reservation_id"]),
            operation_key=row["operation_key"],
            requested=BudgetAmount.from_json(str(row["requested_json"])),
            actual=(
                None if row["actual_json"] is None else BudgetAmount.from_json(row["actual_json"])
            ),
            status=ReservationStatus(str(row["status"])),
        )

    def reserve_budget(
        self,
        reservation_id: str,
        requested: BudgetAmount,
        *,
        operation_key: str | None = None,
    ) -> BudgetReservation:
        _validate_id("reservation_id", reservation_id)
        if operation_key is not None:
            _validate_id("operation_key", operation_key)
        requested_json = canonical_json_bytes(asdict(requested)).decode("ascii")
        now = float(self._clock())
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if existing is not None:
                reservation = self._reservation_from_row(existing)
                if reservation.requested != requested or reservation.operation_key != operation_key:
                    raise RunStateConflict("reservation ID was reused with different inputs")
                return reservation
            row = self._campaign_row(connection)
            self._assert_time_and_running(row, now)
            used = self._amount_from_row(row, "used")
            reserved = self._amount_from_row(row, "reserved")
            limit = self._limit_amount(self._limits_from_row(row))
            prospective = used + reserved + requested
            if not prospective.fits_within(limit):
                raise BudgetExceeded("budget reservation would exceed campaign limits")
            if operation_key is not None:
                operation = connection.execute(
                    "SELECT 1 FROM operations WHERE operation_key = ?",
                    (operation_key,),
                ).fetchone()
                if operation is None:
                    raise RunStateError("budget reservation references an unknown operation")
            connection.execute(
                """
                INSERT INTO reservations(
                    reservation_id, operation_key, requested_json, status,
                    created_unix, updated_unix
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    reservation_id,
                    operation_key,
                    requested_json,
                    ReservationStatus.RESERVED.value,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE campaign SET
                    reserved_proposals = reserved_proposals + ?,
                    reserved_researcher_tokens = reserved_researcher_tokens + ?,
                    reserved_training_tokens = reserved_training_tokens + ?,
                    reserved_compute_seconds = reserved_compute_seconds + ?,
                    updated_unix = ?
                WHERE singleton = 1
                """,
                (
                    requested.proposals,
                    requested.researcher_tokens,
                    requested.training_tokens,
                    requested.compute_seconds,
                    now,
                ),
            )
            created = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            assert created is not None
            return self._reservation_from_row(created)

    def settle_budget(self, reservation_id: str, actual: BudgetAmount) -> BudgetReservation:
        _validate_id("reservation_id", reservation_id)
        actual_json = canonical_json_bytes(asdict(actual)).decode("ascii")
        now = float(self._clock())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise RunStateError("budget reservation does not exist")
            reservation = self._reservation_from_row(row)
            if reservation.status is ReservationStatus.SETTLED:
                if reservation.actual != actual:
                    raise RunStateConflict("settled reservation has different actual usage")
                return reservation
            if reservation.status is not ReservationStatus.RESERVED:
                raise RunStateError("released reservations cannot be settled")
            if not actual.fits_within(reservation.requested):
                raise BudgetExceeded("actual usage exceeds the prior reservation")
            requested = reservation.requested
            connection.execute(
                """
                UPDATE reservations SET status = ?, actual_json = ?, updated_unix = ?
                WHERE reservation_id = ?
                """,
                (ReservationStatus.SETTLED.value, actual_json, now, reservation_id),
            )
            connection.execute(
                """
                UPDATE campaign SET
                    reserved_proposals = reserved_proposals - ?,
                    reserved_researcher_tokens = reserved_researcher_tokens - ?,
                    reserved_training_tokens = reserved_training_tokens - ?,
                    reserved_compute_seconds = reserved_compute_seconds - ?,
                    used_proposals = used_proposals + ?,
                    used_researcher_tokens = used_researcher_tokens + ?,
                    used_training_tokens = used_training_tokens + ?,
                    used_compute_seconds = used_compute_seconds + ?,
                    updated_unix = ?
                WHERE singleton = 1
                """,
                (
                    requested.proposals,
                    requested.researcher_tokens,
                    requested.training_tokens,
                    requested.compute_seconds,
                    actual.proposals,
                    actual.researcher_tokens,
                    actual.training_tokens,
                    actual.compute_seconds,
                    now,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            assert updated is not None
            return self._reservation_from_row(updated)

    def release_budget(self, reservation_id: str) -> BudgetReservation:
        _validate_id("reservation_id", reservation_id)
        now = float(self._clock())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise RunStateError("budget reservation does not exist")
            reservation = self._reservation_from_row(row)
            if reservation.status is ReservationStatus.RELEASED:
                return reservation
            if reservation.status is ReservationStatus.SETTLED:
                raise RunStateError("settled reservations cannot be released")
            requested = reservation.requested
            connection.execute(
                "UPDATE reservations SET status = ?, updated_unix = ? WHERE reservation_id = ?",
                (ReservationStatus.RELEASED.value, now, reservation_id),
            )
            connection.execute(
                """
                UPDATE campaign SET
                    reserved_proposals = reserved_proposals - ?,
                    reserved_researcher_tokens = reserved_researcher_tokens - ?,
                    reserved_training_tokens = reserved_training_tokens - ?,
                    reserved_compute_seconds = reserved_compute_seconds - ?,
                    updated_unix = ?
                WHERE singleton = 1
                """,
                (
                    requested.proposals,
                    requested.researcher_tokens,
                    requested.training_tokens,
                    requested.compute_seconds,
                    now,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            assert updated is not None
            return self._reservation_from_row(updated)

    @staticmethod
    def _operation_from_row(row: sqlite3.Row, disposition: ClaimDisposition) -> OperationClaim:
        result = None if row["result_json"] is None else json.loads(str(row["result_json"]))
        return OperationClaim(
            operation_key=str(row["operation_key"]),
            kind=str(row["kind"]),
            input_sha256=str(row["input_sha256"]),
            status=OperationStatus(str(row["status"])),
            disposition=disposition,
            attempts=int(row["attempts"]),
            result=result,
            error=row["error"],
        )

    def begin_operation(
        self,
        operation_key: str,
        kind: str,
        input_payload: dict[str, Any],
    ) -> OperationClaim:
        _validate_id("operation_key", operation_key)
        _validate_id("kind", kind)
        _input_json, input_hash = _canonical_object(input_payload, name="operation input")
        now = float(self._clock())
        with self._transaction() as connection:
            row = self._campaign_row(connection)
            self._assert_time_and_running(row, now)
            existing = connection.execute(
                "SELECT * FROM operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if existing is not None:
                if str(existing["kind"]) != kind or str(existing["input_sha256"]) != input_hash:
                    raise RunStateConflict("operation key was reused with different inputs")
                status = OperationStatus(str(existing["status"]))
                disposition = (
                    ClaimDisposition.REPLAY
                    if status in {OperationStatus.SUCCEEDED, OperationStatus.FAILED}
                    else ClaimDisposition.RECOVER
                )
                return self._operation_from_row(existing, disposition)
            connection.execute(
                """
                INSERT INTO operations(
                    operation_key, kind, input_sha256, status, attempts,
                    started_unix, updated_unix
                ) VALUES (?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    operation_key,
                    kind,
                    input_hash,
                    OperationStatus.RUNNING.value,
                    now,
                    now,
                ),
            )
            created = connection.execute(
                "SELECT * FROM operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            assert created is not None
            return self._operation_from_row(created, ClaimDisposition.EXECUTE)

    def operation(self, operation_key: str) -> OperationClaim:
        _validate_id("operation_key", operation_key)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise RunStateError("operation does not exist")
            status = OperationStatus(str(row["status"]))
            disposition = (
                ClaimDisposition.REPLAY
                if status in {OperationStatus.SUCCEEDED, OperationStatus.FAILED}
                else ClaimDisposition.RECOVER
            )
            return self._operation_from_row(row, disposition)
        finally:
            connection.close()

    def mark_running_interrupted(self, reason: str = "controller process restarted") -> int:
        _validate_reason(reason)
        now = float(self._clock())
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE operations SET status = ?, error = ?, updated_unix = ?
                WHERE status = ?
                """,
                (
                    OperationStatus.INTERRUPTED.value,
                    reason,
                    now,
                    OperationStatus.RUNNING.value,
                ),
            )
            return int(cursor.rowcount)

    def restart_interrupted(self, operation_key: str) -> OperationClaim:
        _validate_id("operation_key", operation_key)
        now = float(self._clock())
        with self._transaction() as connection:
            campaign = self._campaign_row(connection)
            self._assert_time_and_running(campaign, now)
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise RunStateError("operation does not exist")
            if OperationStatus(str(row["status"])) is not OperationStatus.INTERRUPTED:
                raise RunStateError("only interrupted operations can be restarted")
            connection.execute(
                """
                UPDATE operations SET status = ?, attempts = attempts + 1,
                    result_json = NULL, error = NULL, updated_unix = ?
                WHERE operation_key = ?
                """,
                (OperationStatus.RUNNING.value, now, operation_key),
            )
            updated = connection.execute(
                "SELECT * FROM operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            assert updated is not None
            return self._operation_from_row(updated, ClaimDisposition.EXECUTE)

    def complete_operation(
        self,
        operation_key: str,
        result: dict[str, Any],
    ) -> OperationClaim:
        _validate_id("operation_key", operation_key)
        result_json, _result_hash = _canonical_object(result, name="operation result")
        now = float(self._clock())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise RunStateError("operation does not exist")
            status = OperationStatus(str(row["status"]))
            if status is OperationStatus.SUCCEEDED:
                if str(row["result_json"]) != result_json:
                    raise RunStateConflict("completed operation has a different result")
                return self._operation_from_row(row, ClaimDisposition.REPLAY)
            if status not in {OperationStatus.RUNNING, OperationStatus.INTERRUPTED}:
                raise RunStateError("failed operations cannot be completed")
            connection.execute(
                """
                UPDATE operations SET status = ?, result_json = ?, error = NULL,
                    updated_unix = ? WHERE operation_key = ?
                """,
                (OperationStatus.SUCCEEDED.value, result_json, now, operation_key),
            )
            updated = connection.execute(
                "SELECT * FROM operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            assert updated is not None
            return self._operation_from_row(updated, ClaimDisposition.REPLAY)

    def fail_operation(self, operation_key: str, error: str) -> OperationClaim:
        _validate_id("operation_key", operation_key)
        _validate_reason(error)
        now = float(self._clock())
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise RunStateError("operation does not exist")
            status = OperationStatus(str(row["status"]))
            if status is OperationStatus.FAILED:
                if str(row["error"]) != error:
                    raise RunStateConflict("failed operation has a different error")
                return self._operation_from_row(row, ClaimDisposition.REPLAY)
            if status not in {OperationStatus.RUNNING, OperationStatus.INTERRUPTED}:
                raise RunStateError("successful operations cannot be failed")
            connection.execute(
                """
                UPDATE operations SET status = ?, result_json = NULL, error = ?,
                    updated_unix = ? WHERE operation_key = ?
                """,
                (OperationStatus.FAILED.value, error, now, operation_key),
            )
            updated = connection.execute(
                "SELECT * FROM operations WHERE operation_key = ?",
                (operation_key,),
            ).fetchone()
            assert updated is not None
            return self._operation_from_row(updated, ClaimDisposition.REPLAY)

    def _set_control_request(
        self,
        requested: CampaignStatus,
        reason: str,
    ) -> CampaignSnapshot:
        _validate_reason(reason)
        now = float(self._clock())
        with self._transaction() as connection:
            row = self._campaign_row(connection)
            current = CampaignStatus(str(row["status"]))
            if requested is CampaignStatus.PAUSE_REQUESTED:
                allowed = {
                    CampaignStatus.RUNNING,
                    CampaignStatus.PAUSE_REQUESTED,
                    CampaignStatus.PAUSED,
                }
            else:
                allowed = {
                    CampaignStatus.RUNNING,
                    CampaignStatus.PAUSE_REQUESTED,
                    CampaignStatus.PAUSED,
                    CampaignStatus.CANCEL_REQUESTED,
                    CampaignStatus.CANCELLED,
                }
            if current not in allowed:
                raise RunStateError(f"cannot request {requested.value} while {current.value}")
            settled = (
                CampaignStatus.PAUSED
                if requested is CampaignStatus.PAUSE_REQUESTED
                else CampaignStatus.CANCELLED
            )
            if current is not settled:
                connection.execute(
                    """
                    UPDATE campaign SET status = ?, control_reason = ?, updated_unix = ?
                    WHERE singleton = 1
                    """,
                    (requested.value, reason, now),
                )
        return self.snapshot()

    def request_pause(self, reason: str) -> CampaignSnapshot:
        return self._set_control_request(CampaignStatus.PAUSE_REQUESTED, reason)

    def request_cancel(self, reason: str) -> CampaignSnapshot:
        return self._set_control_request(CampaignStatus.CANCEL_REQUESTED, reason)

    def checkpoint_control(self) -> CampaignStatus:
        now = float(self._clock())
        with self._transaction() as connection:
            row = self._campaign_row(connection)
            status = CampaignStatus(str(row["status"]))
            target = {
                CampaignStatus.PAUSE_REQUESTED: CampaignStatus.PAUSED,
                CampaignStatus.CANCEL_REQUESTED: CampaignStatus.CANCELLED,
            }.get(status)
            if target is not None:
                connection.execute(
                    "UPDATE campaign SET status = ?, updated_unix = ? WHERE singleton = 1",
                    (target.value, now),
                )
                return target
            return status

    def resume(self) -> CampaignSnapshot:
        now = float(self._clock())
        with self._transaction() as connection:
            row = self._campaign_row(connection)
            status = CampaignStatus(str(row["status"]))
            if status not in {CampaignStatus.PAUSED, CampaignStatus.PAUSE_REQUESTED}:
                raise RunStateError("only a paused campaign can resume")
            limits = self._limits_from_row(row)
            if now - float(row["started_unix"]) >= limits.max_wall_seconds:
                raise BudgetExceeded("campaign wall-time limit is exhausted")
            connection.execute(
                """
                UPDATE campaign SET status = ?, control_reason = NULL, updated_unix = ?
                WHERE singleton = 1
                """,
                (CampaignStatus.RUNNING.value, now),
            )
        return self.snapshot()

    def mark_completed(self) -> CampaignSnapshot:
        return self._mark_terminal(CampaignStatus.COMPLETED, None)

    def mark_failed(self, reason: str) -> CampaignSnapshot:
        return self._mark_terminal(CampaignStatus.FAILED, _validate_reason(reason))

    def _mark_terminal(
        self,
        status: CampaignStatus,
        reason: str | None,
    ) -> CampaignSnapshot:
        now = float(self._clock())
        with self._transaction() as connection:
            row = self._campaign_row(connection)
            current = CampaignStatus(str(row["status"]))
            if current in {CampaignStatus.COMPLETED, CampaignStatus.FAILED}:
                if current is not status or row["control_reason"] != reason:
                    raise RunStateError("campaign already has a different terminal state")
            elif current is CampaignStatus.CANCELLED:
                raise RunStateError("cancelled campaigns cannot change terminal state")
            else:
                connection.execute(
                    """
                    UPDATE campaign SET status = ?, control_reason = ?, updated_unix = ?
                    WHERE singleton = 1
                    """,
                    (status.value, reason, now),
                )
        return self.snapshot()


class RepositoryLock:
    """Nonblocking process lock shared by every worktree of one Git repository."""

    def __init__(self, repository_root: Path, *, campaign_id: str) -> None:
        _validate_id("campaign_id", campaign_id)
        self.repository_root = repository_root.expanduser().resolve()
        self.campaign_id = campaign_id
        self.lock_path = self._lock_path()
        self._file: Any = None

    def _lock_path(self) -> Path:
        completed = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=self.repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RunStateError("repository_root is not a Git repository")
        common = Path(completed.stdout.strip())
        if not common.is_absolute():
            common = self.repository_root / common
        return common.resolve() / _LOCK_FILENAME

    def acquire(self) -> RepositoryLock:
        if self._file is not None:
            raise RunStateError("repository lock is already held by this object")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise RepositoryLocked("another campaign process owns the repository lock") from error
        metadata = {
            "campaign_id": self.campaign_id,
            "owner_pid": os.getpid(),
            "schema_version": RUN_STATE_SCHEMA_VERSION,
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._file = handle
        return self

    def release(self) -> None:
        if self._file is None:
            return
        handle = self._file
        self._file = None
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def __enter__(self) -> RepositoryLock:
        return self.acquire()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.release()


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage durable autonomous campaign state.")
    parser.add_argument("--state-path", type=_path, default=DEFAULT_STATE_PATH)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create")
    create.add_argument("--campaign-id", required=True)
    create.add_argument("--initial-parent", required=True)
    create.add_argument("--max-proposals", type=int, required=True)
    create.add_argument("--max-wall-seconds", type=float, required=True)
    create.add_argument("--max-researcher-tokens", type=int, required=True)
    create.add_argument("--max-training-tokens", type=int, required=True)
    create.add_argument("--max-compute-seconds", type=float, required=True)
    create.add_argument("--reward-calibration-labels", type=int, default=0)

    for name in ("pause", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("--reason", required=True)
    commands.add_parser("resume")
    commands.add_parser("recover")
    commands.add_parser("status")
    return parser


def _snapshot_payload(snapshot: CampaignSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            store = CampaignStore.create(
                args.state_path,
                campaign_id=args.campaign_id,
                initial_parent_commit=args.initial_parent,
                limits=CampaignLimits(
                    max_proposals=args.max_proposals,
                    max_wall_seconds=args.max_wall_seconds,
                    max_researcher_tokens=args.max_researcher_tokens,
                    max_training_tokens=args.max_training_tokens,
                    max_compute_seconds=args.max_compute_seconds,
                    reward_calibration_labels=args.reward_calibration_labels,
                ),
            )
        else:
            store = CampaignStore.open(args.state_path)
            if args.command == "pause":
                store.request_pause(args.reason)
            elif args.command == "cancel":
                store.request_cancel(args.reason)
            elif args.command == "resume":
                store.resume()
            elif args.command == "recover":
                store.mark_running_interrupted()
        print(json.dumps(_snapshot_payload(store.snapshot()), indent=2, sort_keys=True))
        return 0
    except (OSError, RunStateError, sqlite3.Error, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
