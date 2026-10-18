"""Deterministic durable-store failure injection, exclusively synthetic bytes.

These tests prove store fencing/recovery semantics. They do not claim that an
actual GPU worker was killed, that PostgreSQL was tested, or that media worked.
"""

from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from forte.jobs import JobStore, LeaseLost


class UtcClock:
    def __init__(self):
        self.now = datetime(2026, 10, 17, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def state(tmp_path):
    clock = UtcClock()
    database = "sqlite:///" + (tmp_path / "jobs.db").as_posix()
    store = JobStore(database, tmp_path / "results", lease_seconds=5,
                     job_timeout_seconds=40, max_attempts=3, clock=clock)
    source = tmp_path / "synthetic.mov"
    source.write_bytes(b"FORTE_SYNTHETIC_CORRUPT_MEDIA\x00")

    def create(identifier):
        return store.create_job(identifier, source, hashlib.sha256(source.read_bytes()).hexdigest(),
                                source.stat().st_size, "synthetic.mov")

    yield store, clock, create, database, tmp_path
    store.close()


def clean_synthetic_result(identifier):
    # A contract fixture only. Never the fallback for a real failed analysis.
    return {"job_id": identifier, "status": "done", "verdict": "no_violation", "violations": []}


def test_abandoned_lease_recovered_and_stale_publish_fenced(state):
    store, clock, create, _, _ = state
    create("synthetic-recovery")
    first = store.claim("worker-one")
    clock.advance(6)
    assert store.recover() == 1
    second = store.claim("worker-two")
    assert second.attempt > first.attempt
    with pytest.raises(LeaseLost):
        store.publish(first.job_id, "worker-one", first.attempt, clean_synthetic_result(first.job_id))
    assert store.get_job(first.job_id).status == "processing"
    assert store.get_result(first.job_id) is None
    store.publish(second.job_id, "worker-two", second.attempt, clean_synthetic_result(second.job_id))
    assert store.get_job(second.job_id).status == "done"


def test_absolute_deadline_wins_even_with_lease_and_never_clean_verdict(state):
    store, clock, create, _, _ = state
    create("synthetic-deadline")
    claimed = store.claim("worker")
    clock.advance(41)
    with pytest.raises(LeaseLost):
        store.publish(claimed.job_id, "worker", claimed.attempt, clean_synthetic_result(claimed.job_id))
    record = store.get_job(claimed.job_id)
    assert record.status == "failed"
    assert record.public_response() == {"job_id": claimed.job_id, "status": "failed"}
    assert store.get_result(claimed.job_id) is None


def test_cancelled_attempt_cannot_publish(state):
    store, _, create, _, _ = state
    create("synthetic-cancel")
    claimed = store.claim("worker")
    store.cancel(claimed.job_id)
    with pytest.raises(LeaseLost):
        store.publish(claimed.job_id, "worker", claimed.attempt, clean_synthetic_result(claimed.job_id))
    assert store.get_job(claimed.job_id).status == "failed"
    assert store.get_result(claimed.job_id) is None


def test_database_reopen_retains_accepted_job_and_failure_state(state):
    store, clock, create, database, directory = state
    create("synthetic-reopen")
    store.close()
    reopened = JobStore(database, directory / "results", lease_seconds=5,
                        job_timeout_seconds=40, clock=clock)
    try:
        claimed = reopened.claim("reopened-worker")
        assert claimed.job_id == "synthetic-reopen"
        reopened.fail(claimed.job_id, "reopened-worker", claimed.attempt, "corrupt_media")
        assert reopened.get_job(claimed.job_id).status == "failed"
        assert reopened.get_result(claimed.job_id) is None
    finally:
        reopened.close()
