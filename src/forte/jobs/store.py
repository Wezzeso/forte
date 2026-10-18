"""One durable queue and state machine for API processes and independent workers.

PostgreSQL claims use FOR UPDATE SKIP LOCKED. SQLite is an explicit local
development fallback: BEGIN IMMEDIATE serializes its small write transactions.
No transaction spans model work or result serialization/file I/O. Publication
writes an immutable, attempt-specific file first, then commits its pointer only
after checking the current owner, lease and fencing token under a row lock.
A crash between file creation and commit leaves an unreferenced file, never a
partially visible result. Completed results are retained until an operator
chooses a retention policy; this module never removes accepted inputs/results.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Callable, Iterator, Mapping

from sqlalchemy import (
    JSON, BigInteger, CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, MetaData,
    String, Table, Text, and_, create_engine, event, func, or_, select, update,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import StaticPool


class LeaseLost(RuntimeError):
    """The attempt expired, was superseded/cancelled, or no longer owns work."""


class JobNotFound(LookupError):
    """No accepted job exists for this identifier."""


class JobConflict(ValueError):
    """An existing identifier belongs to a different accepted upload."""


class ReviewNotAllowed(ValueError):
    """Only intact completed positive results and their actual events are reviewable."""


class ReviewConflict(ValueError):
    """The submitted source or expected review revision is stale."""

    def __init__(self, current_revision: int, current_result_sha256: str) -> None:
        super().__init__("review source or revision changed")
        self.current_revision = current_revision
        self.current_result_sha256 = current_result_sha256


def utcnow() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    # SQLite discards offsets from SQLAlchemy DateTime; all stored values are UTC.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    status: str
    stage: str
    accepted_at: datetime
    deadline_at: datetime
    input_path: str
    input_sha256: str
    input_size_bytes: int
    original_filename: str
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    attempt: int = 0
    max_attempts: int = 3
    retry_at: datetime | None = None
    result_path: str | None = None
    result_sha256: str | None = None
    error_code: str | None = None
    error_detail: str | None = field(default=None, repr=False)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    stage_times: dict[str, str] = field(default_factory=dict)

    @property
    def owned_path(self) -> str:
        return self.input_path

    @property
    def content_hash(self) -> str:
        return self.input_sha256

    @property
    def fencing_version(self) -> int:
        return self.attempt

    def remaining_seconds(self, now: datetime | None = None) -> float:
        return max(0.0, (self.deadline_at - _utc(now or utcnow())).total_seconds())

    def public_response(self) -> dict[str, Any]:
        if self.status == "done":
            if not self.result_path:
                raise RuntimeError("completed job has no durable result")
            return json.loads(Path(self.result_path).read_bytes())
        return {"job_id": self.job_id, "status": self.status}


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """A claimed human assessment; actor names are never identity verification."""

    job_id: str
    result_sha256: str
    violation_index: int
    evidence_digest: str
    state: str = "pending"
    revision: int = 0
    actor_claimed: str | None = None
    comment: str = ""
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "result_sha256": self.result_sha256,
            "violation_index": self.violation_index, "evidence_digest": self.evidence_digest,
            "state": self.state, "revision": self.revision, "actor_claimed": self.actor_claimed,
            "comment": self.comment, "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


metadata = MetaData()
jobs = Table(
    "jobs", metadata,
    Column("job_id", String(128), primary_key=True),
    Column("status", String(16), nullable=False),
    Column("stage", String(64), nullable=False),
    Column("accepted_at", DateTime(timezone=True), nullable=False),
    Column("deadline_at", DateTime(timezone=True), nullable=False),
    Column("input_path", Text, nullable=False),
    Column("input_sha256", String(64), nullable=False),
    Column("input_size_bytes", BigInteger, nullable=False),
    Column("original_filename", Text, nullable=False),
    Column("lease_owner", String(256)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("attempt", Integer, nullable=False, default=0),
    Column("max_attempts", Integer, nullable=False),
    Column("retry_at", DateTime(timezone=True)),
    Column("result_path", Text),
    Column("result_sha256", String(64)),
    Column("error_code", String(128)),
    Column("error_detail", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
    Column("stage_times", JSON, nullable=False),
    CheckConstraint("status IN ('processing', 'done', 'failed')", name="jobs_status"),
    CheckConstraint("attempt >= 0 AND max_attempts > 0", name="jobs_attempts"),
    CheckConstraint("input_size_bytes > 0", name="jobs_nonempty_upload"),
    CheckConstraint("status != 'done' OR result_path IS NOT NULL", name="jobs_done_has_result"),
)
Index("ix_jobs_claim", jobs.c.status, jobs.c.accepted_at, jobs.c.lease_expires_at)
Index("ix_jobs_deadline", jobs.c.status, jobs.c.deadline_at)

# History is the sole review source of truth. Revision zero/pending is an
# implicit initial state; later revisions are appended, never updated in place.
review_history = Table(
    "human_review_history", metadata,
    Column("job_id", String(128), ForeignKey("jobs.job_id", ondelete="RESTRICT"), primary_key=True),
    Column("result_sha256", String(64), primary_key=True),
    Column("violation_index", Integer, primary_key=True),
    Column("revision", Integer, primary_key=True),
    Column("evidence_digest", String(64), nullable=False),
    Column("state", String(16), nullable=False),
    Column("actor_claimed", String(120), nullable=False),
    Column("comment", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("state IN ('pending', 'confirmed', 'rejected')", name="human_review_state"),
    CheckConstraint("violation_index >= 0 AND revision > 0", name="human_review_revision"),
    CheckConstraint("length(actor_claimed) BETWEEN 1 AND 120", name="human_review_actor_length"),
    CheckConstraint("length(comment) <= 4000", name="human_review_comment_length"),
)


def _record(row: Mapping[str, Any]) -> JobRecord:
    data = dict(row)
    for key in ("accepted_at", "deadline_at", "lease_expires_at", "retry_at",
                "created_at", "updated_at", "completed_at"):
        if data.get(key) is not None:
            data[key] = _utc(data[key])
    return JobRecord(**data)


def _review_record(row: Mapping[str, Any]) -> ReviewRecord:
    data = dict(row)
    data["updated_at"] = _utc(data["updated_at"])
    return ReviewRecord(**data)


class JobStore:
    """Synchronous repository safe to call from a worker or an API thread pool.

    Use a shared persistent volume for inputs and results across API/workers.
    ``clock`` and ``on_transition`` support deterministic QA and timing export.
    Transition callbacks run after commit and cannot invalidate accepted work.
    ``database_url`` must explicitly select PostgreSQL or local-dev SQLite.
    """

    def __init__(
        self, database_url: str, result_root: str | Path, *,
        lease_seconds: float = 30, job_timeout_seconds: float = 285,
        max_attempts: int = 3, create_schema: bool = True,
        clock: Callable[[], datetime] = utcnow,
        on_transition: Callable[[JobRecord], None] | None = None,
    ) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (lease_seconds, job_timeout_seconds)):
            raise ValueError("lease and timeout must be finite positive seconds")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if database_url.startswith("postgres://"):
            database_url = "postgresql+psycopg://" + database_url[len("postgres://"):]
        elif database_url.startswith("postgresql://"):
            database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]
        sqlite = database_url.startswith("sqlite")
        if not sqlite and not database_url.startswith("postgresql+"):
            raise ValueError("PostgreSQL is required; SQLite is supported only for local development")
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if sqlite:
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if database_url.endswith(":memory:") or database_url in ("sqlite://", "sqlite+pysqlite://"):
                kwargs["poolclass"] = StaticPool
        else:
            kwargs["connect_args"] = {"connect_timeout": 5}
        self.engine: Engine = create_engine(database_url, **kwargs)
        self.is_sqlite = sqlite
        # A test-only in-memory StaticPool has one DBAPI connection. Serialize
        # its access so independent TestClient threads cannot rollback each
        # other's transactions. File-backed SQLite uses actual DB locking.
        self._memory_lock = threading.RLock() if kwargs.get("poolclass") is StaticPool else None
        if sqlite:
            @event.listens_for(self.engine, "connect")
            def _sqlite_options(dbapi_connection: Any, _: Any) -> None:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=FULL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        self.result_root = Path(result_root).resolve()
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = float(lease_seconds)
        self.job_timeout_seconds = float(job_timeout_seconds)
        self.max_attempts = max_attempts
        self.clock = clock
        self.on_transition = on_transition
        if create_schema:
            # API and workers may start together. Serialize initial DDL so two
            # checkfirst/create_all calls cannot race to create the same table.
            with self._transaction() as connection:
                if not sqlite:
                    connection.execute(select(func.pg_advisory_xact_lock(4600014)))
                metadata.create_all(connection)

    def _now(self, connection: Connection | None = None) -> datetime:
        # Production leases use the database clock, so independent API/worker
        # host clock skew cannot resurrect an expired attempt. Tests can inject
        # a deterministic aware clock; SQLite local development uses host UTC.
        if not self.is_sqlite and self.clock is utcnow:
            if connection is not None:
                return _utc(connection.execute(select(func.clock_timestamp())).scalar_one())
            with self._connection() as current_connection:
                return _utc(current_connection.execute(select(func.clock_timestamp())).scalar_one())
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return _utc(value)

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        with self._memory_lock if self._memory_lock is not None else nullcontext():
            with self.engine.connect() as connection:
                yield connection

    @contextmanager
    def _transaction(self) -> Iterator[Connection]:
        with self._connection() as connection:
            if self.is_sqlite:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _emit(self, record: JobRecord) -> JobRecord:
        if self.on_transition is not None:
            try:
                self.on_transition(record)
            except Exception:
                # Monitoring is never allowed to lose an accepted job or lease.
                pass
        return record

    @staticmethod
    def _job_id(job_id: str) -> None:
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            raise ValueError("job_id must contain 1-128 ASCII letters, digits, underscores or hyphens")

    @staticmethod
    def _fetch(connection: Connection, job_id: str, *, lock: bool = False) -> JobRecord | None:
        query = select(jobs).where(jobs.c.job_id == job_id)
        if lock:
            query = query.with_for_update()
        row = connection.execute(query).mappings().first()
        return _record(row) if row else None

    def ping(self) -> bool:
        try:
            with self._connection() as connection:
                return connection.execute(select(1)).scalar_one() == 1
        except SQLAlchemyError:
            return False

    def close(self) -> None:
        self.engine.dispose()

    def create_job(
        self, job_id: str, input_path: str | Path, input_sha256: str,
        input_size_bytes: int, original_filename: str, *,
        deadline_at: datetime | None = None,
    ) -> JobRecord:
        self._job_id(job_id)
        if not re.fullmatch(r"[a-fA-F0-9]{64}", input_sha256):
            raise ValueError("input_sha256 must be a SHA-256 hex digest")
        if (isinstance(input_size_bytes, bool) or not isinstance(input_size_bytes, int)
                or input_size_bytes <= 0):
            raise ValueError("upload must be nonempty")
        path = Path(input_path).resolve(strict=True)
        if not path.is_file() or path.stat().st_size != input_size_bytes:
            raise ValueError("accepted upload must exist in full before enqueue")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != input_sha256.lower():
            raise ValueError("accepted upload does not match its recorded SHA-256")
        now = self._now()
        if deadline_at is not None and deadline_at.tzinfo is None:
            raise ValueError("deadline_at must be timezone aware")
        deadline = _utc(deadline_at) if deadline_at else now + timedelta(seconds=self.job_timeout_seconds)
        if deadline <= now:
            raise ValueError("deadline must be after acceptance")
        values = dict(
            job_id=job_id, status="processing", stage="queued", accepted_at=now,
            deadline_at=deadline, input_path=str(path), input_sha256=digest,
            input_size_bytes=input_size_bytes, original_filename=str(original_filename),
            attempt=0, max_attempts=self.max_attempts, created_at=now, updated_at=now,
            stage_times={"queued": now.isoformat()},
        )
        try:
            with self._transaction() as connection:
                connection.execute(jobs.insert().values(**values))
                record = self._fetch(connection, job_id)
        except IntegrityError as exc:
            existing = self.get_job(job_id)
            if (existing is not None and existing.input_path == str(path)
                    and existing.input_sha256 == digest
                    and existing.input_size_bytes == input_size_bytes):
                return existing
            raise JobConflict("job identifier already exists") from exc
        assert record is not None
        return self._emit(record)

    create = create_job

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connection() as connection:
            record = self._fetch(connection, job_id)
            now = self._now(connection)
        if record and record.status == "processing" and (
            record.deadline_at <= now or (
                record.lease_expires_at is not None and record.lease_expires_at <= now
                and record.attempt >= record.max_attempts
            )
        ):
            with self._transaction() as connection:
                current = self._fetch(connection, job_id, lock=True)
                if current and current.status == "processing":
                    if current.deadline_at <= now:
                        current = self._terminal(connection, current, now, "deadline_exceeded")
                    elif (current.lease_expires_at is not None and current.lease_expires_at <= now
                          and current.attempt >= current.max_attempts):
                        current = self._terminal(connection, current, now, "attempts_exhausted")
                record = current
            if record:
                self._emit(record)
        return record

    get = get_job

    def _terminal(
        self, connection: Connection, record: JobRecord, now: datetime,
        error_code: str, error_detail: str | None = None,
    ) -> JobRecord:
        connection.execute(update(jobs).where(jobs.c.job_id == record.job_id).values(
            status="failed", stage="failed", lease_owner=None, lease_expires_at=None,
            retry_at=None, updated_at=now, completed_at=now, error_code=error_code[:128],
            error_detail=error_detail[:16384] if error_detail else record.error_detail,
            stage_times={**record.stage_times, "failed": now.isoformat()},
        ))
        result = self._fetch(connection, record.job_id)
        assert result is not None
        return result

    def claim(self, worker_id: str) -> JobRecord | None:
        if not worker_id or len(worker_id) > 256:
            raise ValueError("worker_id must contain 1-256 characters")
        changed: list[JobRecord] = []
        claimed: JobRecord | None = None
        with self._transaction() as connection:
            now = self._now(connection)
            # Each selected row is locked only while deciding/claiming it. A
            # deadline may expire even while a worker still has a live lease.
            eligible = and_(
                jobs.c.status == "processing",
                or_(jobs.c.deadline_at <= now, and_(
                    or_(jobs.c.retry_at.is_(None), jobs.c.retry_at <= now),
                    or_(jobs.c.lease_owner.is_(None), jobs.c.lease_expires_at <= now),
                )),
            )
            # Bound housekeeping per claim so a backlog of expired work never
            # turns one queue claim into a long transaction.
            for _ in range(64):
                row = connection.execute(
                    select(jobs).where(eligible)
                    .order_by(jobs.c.accepted_at, jobs.c.job_id)
                    .limit(1).with_for_update(skip_locked=True)
                ).mappings().first()
                if row is None:
                    break
                record = _record(row)
                if record.deadline_at <= now:
                    changed.append(self._terminal(connection, record, now, "deadline_exceeded"))
                    continue
                if record.attempt >= record.max_attempts:
                    changed.append(self._terminal(connection, record, now, "attempts_exhausted"))
                    continue
                connection.execute(update(jobs).where(jobs.c.job_id == record.job_id).values(
                    stage="claimed", lease_owner=worker_id,
                    lease_expires_at=min(record.deadline_at, now + timedelta(seconds=self.lease_seconds)),
                    attempt=record.attempt + 1, retry_at=None, updated_at=now,
                    stage_times={**record.stage_times, "claimed": now.isoformat()},
                ))
                claimed = self._fetch(connection, record.job_id)
                break
        for record in changed:
            self._emit(record)
        return self._emit(claimed) if claimed else None

    def _owned(
        self, connection: Connection, job_id: str, worker_id: str, attempt: int,
        now: datetime,
    ) -> JobRecord:
        record = self._fetch(connection, job_id, lock=True)
        # Re-read time after acquiring the row lock: waiting on another short
        # transaction must not let a lease which expired during that wait pass.
        now = self._now(connection)
        if record is None:
            raise JobNotFound(job_id)
        if (record.status != "processing" or record.lease_owner != worker_id
                or record.attempt != attempt or record.lease_expires_at is None
                or record.lease_expires_at <= now or record.deadline_at <= now):
            raise LeaseLost(f"job {job_id} attempt {attempt} no longer owns a live lease")
        return record

    def heartbeat(self, job_id: str, worker_id: str, attempt: int) -> JobRecord:
        with self._transaction() as connection:
            now = self._now(connection)
            record = self._owned(connection, job_id, worker_id, attempt, now)
            connection.execute(update(jobs).where(jobs.c.job_id == job_id).values(
                lease_expires_at=min(record.deadline_at, now + timedelta(seconds=self.lease_seconds)),
                updated_at=now,
            ))
            result = self._fetch(connection, job_id)
        assert result is not None
        return result

    def set_stage(self, job_id: str, worker_id: str, attempt: int, stage: str) -> JobRecord:
        if stage not in {"claimed", "decoding", "screening", "refining", "assembling"}:
            raise ValueError("processing stage must be claimed, decoding, screening, refining or assembling")
        with self._transaction() as connection:
            now = self._now(connection)
            record = self._owned(connection, job_id, worker_id, attempt, now)
            connection.execute(update(jobs).where(jobs.c.job_id == job_id).values(
                stage=stage, updated_at=now,
                stage_times={**record.stage_times, stage: now.isoformat()},
            ))
            result = self._fetch(connection, job_id)
        assert result is not None
        return self._emit(result)

    def publish(
        self, job_id: str, worker_id: str, attempt: int,
        result: bytes | Mapping[str, Any],
    ) -> JobRecord:
        from forte.shared.schemas import CompletedResponse, JOB_RESPONSE_ADAPTER

        if isinstance(result, bytes):
            # The discriminator requires an explicit status field. Calling only
            # CompletedResponse.model_validate_json would fill status's default
            # even though the original bytes served to the jury omitted it.
            validated = JOB_RESPONSE_ADAPTER.validate_json(result)
            if not isinstance(validated, CompletedResponse):
                raise ValueError("only completed responses can be published")
            # A12 assembles and encodes JPEG evidence once. Validate the final
            # payload while preserving its exact bytes for immutable serving.
            payload = result
        else:
            validated = CompletedResponse.model_validate(result)
            payload = json.dumps(validated.model_dump(mode="json"), ensure_ascii=False,
                                 allow_nan=False, separators=(",", ":")).encode("utf-8")
        if validated.job_id != job_id:
            raise ValueError("result belongs to another job")
        digest = hashlib.sha256(payload).hexdigest()
        self._job_id(job_id)
        destination = self.result_root / f"{job_id}.{attempt}.{digest}.json"
        # File I/O is deliberately outside the short database transaction.
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=self.result_root,
                    prefix=f".{job_id}.{attempt}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # Content-addressed attempt names make any previous bytes identical;
            # os.replace is atomic for files on the shared local filesystem.
            os.replace(temporary, destination)
            temporary = None
            self._sync_directory()
            with self._transaction() as connection:
                now = self._now(connection)
                current = self._fetch(connection, job_id, lock=True)
                if (current and current.status == "done" and current.attempt == attempt
                        and current.result_sha256 == digest):
                    return current  # Same publication retried after uncertain commit.
                record = self._owned(connection, job_id, worker_id, attempt, now)
                connection.execute(update(jobs).where(jobs.c.job_id == job_id).values(
                    status="done", stage="done", result_path=str(destination), result_sha256=digest,
                    lease_owner=None, lease_expires_at=None, retry_at=None,
                    error_code=None, error_detail=None, updated_at=now, completed_at=now,
                    stage_times={**record.stage_times, "done": now.isoformat()},
                ))
                completed = self._fetch(connection, job_id)
            assert completed is not None
            return self._emit(completed)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _sync_directory(self) -> None:
        if os.name == "posix":
            descriptor = os.open(self.result_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def fail(
        self, job_id: str, worker_id: str, attempt: int, error: str | Exception, *,
        retryable: bool = False, error_code: str = "processing_failed",
        retry_delay_seconds: float | None = None,
    ) -> JobRecord:
        delay = min(2 ** min(attempt, 10), 30) if retry_delay_seconds is None else retry_delay_seconds
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("retry delay must be finite and nonnegative")
        with self._transaction() as connection:
            now = self._now(connection)
            record = self._owned(connection, job_id, worker_id, attempt, now)
            retry_at = now + timedelta(seconds=delay)
            if retryable and attempt < record.max_attempts and retry_at < record.deadline_at:
                connection.execute(update(jobs).where(jobs.c.job_id == job_id).values(
                    stage="queued", lease_owner=None, lease_expires_at=None, retry_at=retry_at,
                    updated_at=now, error_code=error_code[:128], error_detail=str(error)[:16384],
                    stage_times={**record.stage_times, "queued": now.isoformat()},
                ))
                result = self._fetch(connection, job_id)
            else:
                result = self._terminal(connection, record, now, error_code, str(error))
        assert result is not None
        return self._emit(result)

    def cancel(self, job_id: str) -> JobRecord:
        with self._transaction() as connection:
            record = self._fetch(connection, job_id, lock=True)
            if record is None:
                raise JobNotFound(job_id)
            if record.status == "processing":
                record = self._terminal(connection, record, self._now(connection), "cancelled")
        return self._emit(record)

    def recover(self, *, limit: int = 1000) -> int:
        """Requeue expired attempts and fail expired/exhausted jobs in batches."""
        if limit < 1:
            raise ValueError("limit must be positive")
        changed: list[JobRecord] = []
        with self._transaction() as connection:
            now = self._now(connection)
            rows = connection.execute(select(jobs).where(
                jobs.c.status == "processing",
                or_(jobs.c.deadline_at <= now,
                    and_(jobs.c.lease_owner.is_not(None), jobs.c.lease_expires_at <= now)),
            ).order_by(jobs.c.accepted_at).limit(limit).with_for_update(skip_locked=True)).mappings().all()
            for row in rows:
                record = _record(row)
                if record.deadline_at <= now:
                    changed.append(self._terminal(connection, record, now, "deadline_exceeded"))
                elif record.attempt >= record.max_attempts:
                    changed.append(self._terminal(connection, record, now, "attempts_exhausted"))
                else:
                    connection.execute(update(jobs).where(jobs.c.job_id == record.job_id).values(
                        stage="queued", lease_owner=None, lease_expires_at=None,
                        retry_at=None, updated_at=now,
                        stage_times={**record.stage_times, "queued": now.isoformat()},
                    ))
                    refreshed = self._fetch(connection, record.job_id)
                    assert refreshed is not None
                    changed.append(refreshed)
        for record in changed:
            self._emit(record)
        return len(changed)

    def get_result_bytes(self, job_id: str) -> bytes | None:
        record = self.get_job(job_id)
        if record is None or record.status != "done" or record.result_path is None:
            return None
        return Path(record.result_path).read_bytes()

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        payload = self.get_result_bytes(job_id)
        return json.loads(payload) if payload is not None else None

    def counts(self) -> dict[str, int]:
        with self._connection() as connection:
            rows = connection.execute(select(jobs.c.status, func.count()).group_by(jobs.c.status))
            return {str(status): int(count) for status, count in rows}

    def _review_source(
        self, job_id: str, result_sha256: str | None,
    ) -> tuple[JobRecord, list[dict[str, Any]]]:
        record = self.get_job(job_id)
        if record is None:
            raise JobNotFound(job_id)
        if record.status != "done" or not record.result_path or not record.result_sha256:
            raise ReviewNotAllowed("review requires a completed positive result")
        if result_sha256 is not None and result_sha256 != record.result_sha256:
            raise ReviewConflict(0, record.result_sha256)
        try:
            path = Path(record.result_path)
            result_limit = 96 * 1024 * 1024
            if path.stat().st_size > result_limit:
                raise ReviewNotAllowed("completed result exceeds the review payload limit")
            with path.open("rb") as stream:
                payload = stream.read(result_limit + 1)
            if len(payload) > result_limit:
                raise ReviewNotAllowed("completed result exceeds the review payload limit")
        except OSError as exc:
            raise ReviewNotAllowed("completed result is unavailable") from exc
        if hashlib.sha256(payload).hexdigest() != record.result_sha256:
            raise ReviewNotAllowed("completed result integrity check failed")
        try:
            response = json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ReviewNotAllowed("completed result is invalid") from exc
        if (not isinstance(response, dict) or response.get("job_id") != job_id
                or response.get("status") != "done" or response.get("verdict") != "violation"
                or not isinstance(response.get("violations"), list) or not response["violations"]):
            raise ReviewNotAllowed("review requires a completed positive result")
        return record, response["violations"]

    @staticmethod
    def _pending_review(record: JobRecord, index: int, violation: dict[str, Any]) -> ReviewRecord:
        try:
            evidence = violation["evidence_frames"]
            digest = hashlib.sha256(json.dumps(
                evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest()
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewNotAllowed("completed event evidence is invalid") from exc
        assert record.result_sha256 is not None
        return ReviewRecord(record.job_id, record.result_sha256, index, digest)

    @staticmethod
    def _review_index(index: int, count: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
            raise ReviewNotAllowed("violation index does not identify a completed event")

    def get_reviews(
        self, job_id: str, result_sha256: str | None = None,
    ) -> list[ReviewRecord]:
        """Current per-event state, with revision-zero pending for unreviewed events."""
        record, violations = self._review_source(job_id, result_sha256)
        source = and_(review_history.c.job_id == job_id,
                      review_history.c.result_sha256 == record.result_sha256)
        latest = select(
            review_history.c.violation_index,
            func.max(review_history.c.revision).label("revision"),
        ).where(source).group_by(review_history.c.violation_index).subquery()
        with self._connection() as connection:
            rows = connection.execute(select(review_history).join(latest, and_(
                review_history.c.violation_index == latest.c.violation_index,
                review_history.c.revision == latest.c.revision,
            )).where(source)).mappings().all()
        current = {row["violation_index"]: _review_record(row) for row in rows}
        result = []
        for index, violation in enumerate(violations):
            pending = self._pending_review(record, index, violation)
            review = current.get(index, pending)
            if review.evidence_digest != pending.evidence_digest:
                raise ReviewNotAllowed("review evidence binding does not match the completed event")
            result.append(review)
        return result

    def append_review(
        self, job_id: str, result_sha256: str, violation_index: int, *,
        state: str, actor_claimed: str, expected_revision: int,
        comment: str = "", evidence_digest: str | None = None,
    ) -> ReviewRecord:
        """Append an assessment after an optimistic revision and source check.

        Actor text is self-asserted metadata. This repository makes no claim
        that an authenticated API key verifies the identity of that person.
        The immutable jury response and the worker state are never modified.
        """
        if not isinstance(state, str) or state not in {"pending", "confirmed", "rejected"}:
            raise ValueError("review state must be pending, confirmed or rejected")
        if not isinstance(actor_claimed, str):
            raise ValueError("a claimed reviewer name is required")
        actor = actor_claimed.strip()
        if not 1 <= len(actor) <= 120 or any(ord(char) < 32 or ord(char) == 127 for char in actor):
            raise ValueError("claimed reviewer name must contain 1-120 visible characters")
        if (not isinstance(comment, str) or len(comment) > 4000
                or any(ord(char) < 32 and char not in "\n\r\t" for char in comment)):
            raise ValueError("review comment must contain at most 4000 characters")
        if (isinstance(expected_revision, bool) or not isinstance(expected_revision, int)
                or expected_revision < 0):
            raise ValueError("expected_revision must be a nonnegative integer")
        record, violations = self._review_source(job_id, result_sha256)
        self._review_index(violation_index, len(violations))
        pending = self._pending_review(record, violation_index, violations[violation_index])
        if evidence_digest is not None and evidence_digest != pending.evidence_digest:
            raise ReviewNotAllowed("submitted evidence digest does not match the completed event")
        with self._transaction() as connection:
            # A single short row lock serializes reviewers of this immutable job.
            # The history itself is append-only, with a unique revision key.
            current_job = self._fetch(connection, job_id, lock=True)
            if (current_job is None or current_job.status != "done"
                    or current_job.result_sha256 != record.result_sha256):
                raise ReviewConflict(0, current_job.result_sha256 if current_job else "")
            current_revision = connection.execute(select(func.max(review_history.c.revision)).where(
                review_history.c.job_id == job_id,
                review_history.c.result_sha256 == result_sha256,
                review_history.c.violation_index == violation_index,
            )).scalar_one() or 0
            if current_revision != expected_revision:
                raise ReviewConflict(current_revision, result_sha256)
            assessment = ReviewRecord(
                job_id=job_id, result_sha256=result_sha256, violation_index=violation_index,
                evidence_digest=pending.evidence_digest, state=state,
                revision=current_revision + 1, actor_claimed=actor, comment=comment,
                updated_at=self._now(connection),
            )
            connection.execute(review_history.insert().values(
                job_id=assessment.job_id, result_sha256=assessment.result_sha256,
                violation_index=assessment.violation_index, evidence_digest=assessment.evidence_digest,
                state=assessment.state, revision=assessment.revision,
                actor_claimed=assessment.actor_claimed, comment=assessment.comment,
                updated_at=assessment.updated_at,
            ))
        return assessment

    def get_review_history(
        self, job_id: str, result_sha256: str, violation_index: int, *,
        after_revision: int = 0, limit: int = 100,
    ) -> list[ReviewRecord]:
        """Read immutable revisions in ascending pages; pending revision zero is implicit."""
        if (isinstance(after_revision, bool) or not isinstance(after_revision, int) or after_revision < 0
                or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500):
            raise ValueError("history requires after_revision >= 0 and limit between 1 and 500")
        record, violations = self._review_source(job_id, result_sha256)
        self._review_index(violation_index, len(violations))
        expected_digest = self._pending_review(record, violation_index, violations[violation_index]).evidence_digest
        with self._connection() as connection:
            rows = connection.execute(select(review_history).where(
                review_history.c.job_id == job_id,
                review_history.c.result_sha256 == result_sha256,
                review_history.c.violation_index == violation_index,
                review_history.c.revision > after_revision,
            ).order_by(review_history.c.revision).limit(limit)).mappings().all()
        history = [_review_record(row) for row in rows]
        if any(item.evidence_digest != expected_digest for item in history):
            raise ReviewNotAllowed("review evidence binding does not match the completed event")
        return history
