"""Synthetic queue/recovery tests; no supplied media or predictions are read."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy import select

from forte.jobs import JobConflict, JobNotFound, JobStore, LeaseLost, ReviewConflict, ReviewNotAllowed
from forte.jobs.store import jobs, review_history


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 10, 17, 10, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def queue(tmp_path: Path):
    clock = Clock()
    database_url = os.environ.get("FORTE_TEST_POSTGRES_URL", f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    store = JobStore(database_url, tmp_path / "results",
                     clock=clock, lease_seconds=10, job_timeout_seconds=60)
    yield store, clock
    if not store.is_sqlite:
        # The opt-in PostgreSQL run must target an isolated temporary schema.
        with store.engine.begin() as connection:
            connection.execute(review_history.delete())
            connection.execute(jobs.delete())
    store.close()


def enqueue(store: JobStore, tmp_path: Path, job_id: str = "synthetic-job"):
    payload = f"synthetic upload {job_id}".encode()
    source = tmp_path / f"{job_id}.bin"
    source.write_bytes(payload)
    return store.create_job(job_id, source, hashlib.sha256(payload).hexdigest(),
                            len(payload), "synthetic.bin")


def result(job_id: str) -> dict:
    return {"job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []}


def test_only_complete_verified_uploads_are_accepted(queue, tmp_path):
    store, _ = queue
    source = tmp_path / "partial.bin"
    source.write_bytes(b"partial")
    digest = hashlib.sha256(b"partial").hexdigest()
    with pytest.raises(ValueError, match="exist in full"):
        store.create_job("size-mismatch", source, digest, 1000, "test.bin")
    with pytest.raises(ValueError, match="SHA-256"):
        store.create_job("hash-mismatch", source, "0" * 64, 7, "test.bin")
    assert store.get_job("size-mismatch") is None
    assert store.get_job("hash-mismatch") is None
    assert store.claim("worker") is None


def test_acceptance_survives_repository_restart(queue, tmp_path):
    store, clock = queue
    accepted = enqueue(store, tmp_path)
    store.close()
    restarted = JobStore(store.engine.url.render_as_string(hide_password=False), store.result_root, clock=clock,
                         lease_seconds=10, job_timeout_seconds=60)
    try:
        assert restarted.get_job(accepted.job_id).input_sha256 == accepted.input_sha256
        claimed = restarted.claim("after-restart")
        assert claimed.job_id == accepted.job_id
        assert claimed.attempt == 1
        assert claimed.public_response() == {"job_id": accepted.job_id, "status": "processing"}
    finally:
        restarted.close()


def test_concurrent_workers_never_duplicate_claim(queue, tmp_path):
    store, _ = queue
    for number in range(25):
        enqueue(store, tmp_path, f"job-{number:03}")
    barrier = threading.Barrier(12)

    def claim_many(number):
        barrier.wait()
        claimed = []
        while (record := store.claim(f"worker-{number}")) is not None:
            claimed.append(record.job_id)
        return claimed

    with ThreadPoolExecutor(max_workers=12) as executor:
        claimed = [job_id for batch in executor.map(claim_many, range(12)) for job_id in batch]
    assert len(claimed) == len(set(claimed)) == 25
    assert store.counts() == {"processing": 25}


def test_crashed_worker_recovery_fences_all_stale_operations(queue, tmp_path):
    store, clock = queue
    enqueue(store, tmp_path)
    first = store.claim("dead-worker")
    clock.advance(10.01)
    second = store.claim("replacement-worker")
    assert second.job_id == first.job_id
    assert second.attempt == first.attempt + 1
    for stale_operation in (
        lambda: store.heartbeat(first.job_id, "dead-worker", first.attempt),
        lambda: store.set_stage(first.job_id, "dead-worker", first.attempt, "assembling"),
        lambda: store.publish(first.job_id, "dead-worker", first.attempt, result(first.job_id)),
        lambda: store.fail(first.job_id, "dead-worker", first.attempt, "late failure"),
    ):
        with pytest.raises(LeaseLost):
            stale_operation()
    assert store.get_job(first.job_id).lease_owner == "replacement-worker"
    completed = store.publish(second.job_id, "replacement-worker", second.attempt, result(second.job_id))
    assert completed.status == "done"


def test_expired_worker_cannot_revive_own_lease_before_reclaim(queue, tmp_path):
    store, clock = queue
    enqueue(store, tmp_path)
    claimed = store.claim("worker")
    clock.advance(10)
    with pytest.raises(LeaseLost):
        store.heartbeat(claimed.job_id, "worker", claimed.attempt)
    with pytest.raises(LeaseLost):
        store.publish(claimed.job_id, "worker", claimed.attempt, result(claimed.job_id))
    assert store.get_job(claimed.job_id).status == "processing"


def test_heartbeat_extends_lease_but_never_absolute_deadline(queue, tmp_path):
    store, clock = queue
    enqueue(store, tmp_path)
    claimed = store.claim("worker")
    for _ in range(7):
        clock.advance(8)
        claimed = store.heartbeat(claimed.job_id, "worker", claimed.attempt)
    assert claimed.lease_expires_at == claimed.deadline_at
    clock.advance(4)
    with pytest.raises(LeaseLost):
        store.heartbeat(claimed.job_id, "worker", claimed.attempt)
    failed = store.get_job(claimed.job_id)
    assert failed.error_code == "deadline_exceeded"
    assert failed.public_response() == {"job_id": claimed.job_id, "status": "failed"}


def test_retry_backoff_attempt_cap_and_deadline_are_enforced(queue, tmp_path):
    store, clock = queue
    enqueue(store, tmp_path)
    first = store.claim("worker")
    queued = store.fail(first.job_id, "worker", first.attempt, "transient synthetic error", retryable=True)
    assert queued.stage == "queued" and queued.retry_at == clock() + timedelta(seconds=2)
    assert store.claim("early-worker") is None
    clock.advance(2)
    second = store.claim("worker")
    store.fail(second.job_id, "worker", second.attempt, "again", retryable=True, retry_delay_seconds=0)
    third = store.claim("worker")
    failed = store.fail(third.job_id, "worker", third.attempt, "last", retryable=True)
    assert failed.status == "failed" and failed.attempt == 3
    assert store.claim("worker") is None
    assert "error_detail" not in failed.public_response()


def test_retry_that_cannot_fit_deadline_fails_immediately(queue, tmp_path):
    store, clock = queue
    enqueue(store, tmp_path)
    claimed = store.claim("worker")
    failed = store.fail(claimed.job_id, "worker", claimed.attempt, "retry too late",
                        retryable=True, retry_delay_seconds=60)
    assert failed.status == "failed"


def test_deadline_expires_even_when_no_worker_is_running(queue, tmp_path):
    store, clock = queue
    accepted = enqueue(store, tmp_path)
    clock.advance(61)
    failed = store.get_job(accepted.job_id)
    assert failed.status == "failed" and failed.error_code == "deadline_exceeded"
    assert failed.attempt == 0


def test_explicit_recovery_and_attempt_exhaustion(queue, tmp_path):
    store, clock = queue
    enqueue(store, tmp_path)
    for attempt in range(1, 4):
        claimed = store.claim(f"worker-{attempt}")
        assert claimed.attempt == attempt
        clock.advance(11)
        assert store.recover() == 1
    failed = store.get_job(claimed.job_id)
    assert failed.status == "failed" and failed.error_code == "attempts_exhausted"


def test_atomic_publication_persists_exact_bytes_and_idempotent_retry(queue, tmp_path):
    store, _ = queue
    accepted = enqueue(store, tmp_path)
    claimed = store.claim("worker")
    completed = store.publish(claimed.job_id, "worker", claimed.attempt, result(claimed.job_id))
    original_bytes = store.get_result_bytes(claimed.job_id)
    assert Path(completed.result_path).read_bytes() == original_bytes
    assert json.loads(original_bytes) == result(claimed.job_id)
    assert store.publish(claimed.job_id, "worker", claimed.attempt, result(claimed.job_id)) == completed
    assert not list(store.result_root.glob("*.tmp"))
    store.close()
    restarted = JobStore(store.engine.url.render_as_string(hide_password=False), store.result_root)
    try:
        assert restarted.get_result_bytes(accepted.job_id) == original_bytes
    finally:
        restarted.close()


def test_filesystem_failure_cannot_mark_job_done(queue, tmp_path, monkeypatch):
    store, _ = queue
    enqueue(store, tmp_path)
    claimed = store.claim("worker")

    def disk_failure(*args):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr("forte.jobs.store.os.replace", disk_failure)
    with pytest.raises(OSError, match="disk failure"):
        store.publish(claimed.job_id, "worker", claimed.attempt, result(claimed.job_id))
    assert store.get_job(claimed.job_id).status == "processing"
    assert store.get_result_bytes(claimed.job_id) is None
    assert not list(store.result_root.iterdir())


def test_crash_between_file_write_and_commit_leaves_only_unreferenced_file(queue, tmp_path, monkeypatch):
    store, clock = queue
    enqueue(store, tmp_path)
    first = store.claim("dead-worker")
    original_owned = store._owned

    def crash(*args, **kwargs):
        raise RuntimeError("synthetic crash before DB update")

    monkeypatch.setattr(store, "_owned", crash)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        store.publish(first.job_id, "dead-worker", first.attempt, result(first.job_id))
    assert store.get_job(first.job_id).result_path is None
    assert len(list(store.result_root.glob("*.json"))) == 1
    monkeypatch.setattr(store, "_owned", original_owned)
    clock.advance(11)
    second = store.claim("replacement-worker")
    done = store.publish(second.job_id, "replacement-worker", second.attempt, result(second.job_id))
    assert done.result_path != str(next(store.result_root.glob(f"*.{first.attempt}.*.json")))
    assert store.get_result(first.job_id) == result(first.job_id)


def test_lease_expiration_during_filesystem_write_cannot_publish(queue, tmp_path, monkeypatch):
    store, clock = queue
    enqueue(store, tmp_path)
    claimed = store.claim("worker")
    monkeypatch.setattr(store, "_sync_directory", lambda: clock.advance(11))
    with pytest.raises(LeaseLost):
        store.publish(claimed.job_id, "worker", claimed.attempt, result(claimed.job_id))
    assert store.get_result(claimed.job_id) is None


def test_cancel_fences_inflight_work_and_is_idempotent(queue, tmp_path):
    store, _ = queue
    enqueue(store, tmp_path)
    claimed = store.claim("worker")
    cancelled = store.cancel(claimed.job_id)
    assert cancelled.status == "failed" and cancelled.error_code == "cancelled"
    assert store.cancel(claimed.job_id) == cancelled
    with pytest.raises(LeaseLost):
        store.publish(claimed.job_id, "worker", claimed.attempt, result(claimed.job_id))
    with pytest.raises(JobNotFound):
        store.cancel("unknown")


def test_contract_validation_rejects_wrong_job_or_false_clean_verdict(queue, tmp_path):
    store, _ = queue
    enqueue(store, tmp_path)
    claimed = store.claim("worker")
    with pytest.raises(ValueError):
        store.publish(claimed.job_id, "worker", claimed.attempt, result("other-job"))
    with pytest.raises(ValueError):
        store.publish(claimed.job_id, "worker", claimed.attempt,
                       {**result(claimed.job_id), "verdict": "violation"})
    assert store.get_job(claimed.job_id).status == "processing"


def test_duplicate_acceptance_is_idempotent_but_different_upload_conflicts(queue, tmp_path):
    store, _ = queue
    accepted = enqueue(store, tmp_path)
    assert store.create_job(accepted.job_id, accepted.input_path, accepted.input_sha256,
                            accepted.input_size_bytes, accepted.original_filename) == accepted
    other = tmp_path / "other.bin"
    other.write_bytes(b"another upload")
    with pytest.raises(JobConflict):
        store.create_job(accepted.job_id, other, hashlib.sha256(other.read_bytes()).hexdigest(),
                         other.stat().st_size, "another.bin")


def test_telemetry_cannot_break_job_acceptance_or_work(queue, tmp_path):
    store, _ = queue

    def broken_monitor(record):
        raise RuntimeError("monitor unavailable")

    store.on_transition = broken_monitor
    accepted = enqueue(store, tmp_path)
    claimed = store.claim("worker")
    assert claimed.job_id == accepted.job_id
    changed = store.set_stage(claimed.job_id, "worker", claimed.attempt, "decoding")
    assert "decoding" in changed.stage_times
    assert store.ping()


def test_postgresql_claim_statement_uses_skip_locked():
    query = select(jobs).limit(1).with_for_update(skip_locked=True)
    assert "FOR UPDATE SKIP LOCKED" in str(query.compile(dialect=postgresql.dialect()))


def test_bytes_publication_preserves_assembled_json_and_valid_jpeg(queue, tmp_path):
    from PIL import Image

    store, _ = queue
    enqueue(store, tmp_path)
    claimed = store.claim("worker")
    jpeg = BytesIO()
    Image.new("RGB", (8, 8), (100, 30, 80)).save(jpeg, format="JPEG")
    evidence = base64.b64encode(jpeg.getvalue()).decode("ascii")
    payload = json.dumps({
        "job_id": claimed.job_id, "status": "done", "verdict": "violation",
        "violations": [{"type": "synthetic", "timecode_sec": 1.0,
                        "description": "Synthetic QA fixture",
                        "evidence_frames": [{"timestamp_sec": 1.0, "image_base64": evidence}]}],
    }, indent=3).encode("utf-8") + b"\n"
    completed = store.publish(claimed.job_id, "worker", claimed.attempt, payload)
    assert store.get_result_bytes(claimed.job_id) == payload
    assert Path(completed.result_path).read_bytes() == payload
    assert completed.result_sha256 == hashlib.sha256(payload).hexdigest()
    assert store.publish(claimed.job_id, "worker", claimed.attempt, payload) == completed


@pytest.mark.parametrize("payload", [
    b"{invalid-json}",
    b'{"job_id":"synthetic-job","verdict":"no_violation","violations":[]}',
    b'{"job_id":"synthetic-job","status":"done","verdict":"violation","violations":'
    b'[{"type":"synthetic","timecode_sec":1.0,"description":"synthetic",'
    b'"evidence_frames":[{"timestamp_sec":1.0,"image_base64":"/9j/2Q=="}]}]}',
])
def test_bytes_publication_rejects_invalid_json_or_nondecoding_jpeg(queue, tmp_path, payload):
    store, _ = queue
    enqueue(store, tmp_path)
    claimed = store.claim("worker")
    with pytest.raises(ValueError):
        store.publish(claimed.job_id, "worker", claimed.attempt, payload)
    assert store.get_job(claimed.job_id).status == "processing"
    assert store.get_result_bytes(claimed.job_id) is None
    assert not list(store.result_root.iterdir())


def test_bytes_publication_fences_stale_lease(queue, tmp_path):
    store, clock = queue
    enqueue(store, tmp_path)
    first = store.claim("dead-worker")
    payload = json.dumps(result(first.job_id), indent=2).encode()
    clock.advance(11)
    replacement = store.claim("replacement")
    with pytest.raises(LeaseLost):
        store.publish(first.job_id, "dead-worker", first.attempt, payload)
    assert store.get_job(first.job_id).lease_owner == "replacement"
    assert store.get_result_bytes(first.job_id) is None
    store.publish(replacement.job_id, "replacement", replacement.attempt, payload)
    assert store.get_result_bytes(first.job_id) == payload


def test_live_postgresql_skips_a_row_locked_by_another_transaction(queue, tmp_path):
    store, _ = queue
    if store.is_sqlite:
        pytest.skip("live PostgreSQL opt-in required for row-level SKIP LOCKED verification")
    first = enqueue(store, tmp_path, "a-locked-job")
    second = enqueue(store, tmp_path, "b-available-job")
    with store.engine.begin() as other_transaction:
        other_transaction.execute(select(jobs).where(jobs.c.job_id == first.job_id).with_for_update())
        claimed = store.claim("worker")
        assert claimed.job_id == second.job_id
    assert store.claim("another-worker").job_id == first.job_id


def test_live_postgresql_database_clock_enforces_expiry(queue, tmp_path):
    store, _ = queue
    if store.is_sqlite:
        pytest.skip("live PostgreSQL opt-in required for database-clock verification")
    production = JobStore(store.engine.url.render_as_string(hide_password=False),
                          store.result_root, lease_seconds=0.2, job_timeout_seconds=10)
    try:
        accepted = enqueue(production, tmp_path, "database-clock-job")
        claimed = production.claim("worker")
        assert claimed.accepted_at.tzinfo is not None
        assert claimed.deadline_at > accepted.accepted_at
        time.sleep(0.3)
        with pytest.raises(LeaseLost):
            production.heartbeat(claimed.job_id, "worker", claimed.attempt)
        replacement = production.claim("replacement")
        assert replacement.attempt == claimed.attempt + 1
    finally:
        production.close()


def completed_positive(store, tmp_path, *, job_id="review-job", count=2):
    from PIL import Image

    enqueue(store, tmp_path, job_id)
    claimed = store.claim("synthetic-worker")
    jpeg = BytesIO()
    Image.new("RGB", (8, 8), (40, 70, 100)).save(jpeg, format="JPEG")
    evidence = base64.b64encode(jpeg.getvalue()).decode("ascii")
    payload = json.dumps({
        "job_id": job_id, "status": "done", "verdict": "violation",
        "violations": [{"type": "synthetic", "timecode_sec": float(index),
                        "description": f"Synthetic QA event {index}",
                        "evidence_frames": [{"timestamp_sec": float(index), "image_base64": evidence}]}
                       for index in range(count)],
    }, indent=2).encode("utf-8")
    return store.publish(job_id, "synthetic-worker", claimed.attempt, payload)


def test_reviews_bind_pending_events_to_exact_result_and_evidence(queue, tmp_path):
    store, _ = queue
    completed = completed_positive(store, tmp_path)
    payload = store.get_result_bytes(completed.job_id)
    pending = store.get_reviews(completed.job_id, completed.result_sha256)
    assert len(pending) == 2
    for index, review in enumerate(pending):
        frames = json.loads(payload)["violations"][index]["evidence_frames"]
        expected = hashlib.sha256(json.dumps(frames, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assert review.result_sha256 == completed.result_sha256
        assert review.violation_index == index and review.evidence_digest == expected
        assert review.state == "pending" and review.revision == 0
        assert review.actor_claimed is None and review.updated_at is None
        assert store.get_review_history(completed.job_id, completed.result_sha256, index) == []
    confirmed = store.append_review(completed.job_id, completed.result_sha256, 0,
                                    state="confirmed", actor_claimed="Claimed analyst", comment="Synthetic QA",
                                    expected_revision=0, evidence_digest=pending[0].evidence_digest)
    assert confirmed.revision == 1 and confirmed.updated_at.tzinfo is not None
    assert confirmed.actor_claimed == "Claimed analyst"
    assert store.get_reviews(completed.job_id)[1] == pending[1]
    assert store.get_result_bytes(completed.job_id) == payload
    assert store.get_job(completed.job_id) == completed


def test_review_history_retains_each_revision_and_pages_without_mutating_earlier_decisions(queue, tmp_path):
    store, _ = queue
    completed = completed_positive(store, tmp_path)
    for expected, state in enumerate(("confirmed", "rejected", "pending", "confirmed")):
        store.append_review(completed.job_id, completed.result_sha256, 0,
                            state=state, actor_claimed=f"Claimed analyst {expected}",
                            comment=f"Synthetic comment {expected}", expected_revision=expected)
    first_page = store.get_review_history(completed.job_id, completed.result_sha256, 0, limit=2)
    second_page = store.get_review_history(completed.job_id, completed.result_sha256, 0,
                                           after_revision=first_page[-1].revision, limit=2)
    assert [item.revision for item in first_page + second_page] == [1, 2, 3, 4]
    assert [item.state for item in first_page + second_page] == ["confirmed", "rejected", "pending", "confirmed"]
    assert first_page[0].comment == "Synthetic comment 0"
    assert store.get_reviews(completed.job_id)[0].revision == 4
    assert store.get_review_history(completed.job_id, completed.result_sha256, 0, after_revision=4) == []


def test_simultaneous_reviewers_with_same_revision_have_one_winner(queue, tmp_path):
    store, _ = queue
    completed = completed_positive(store, tmp_path)
    barrier = threading.Barrier(8)

    def assess(number):
        barrier.wait()
        try:
            return store.append_review(completed.job_id, completed.result_sha256, 0,
                                       state="confirmed" if number % 2 else "rejected",
                                       actor_claimed=f"Claimed analyst {number}", expected_revision=0)
        except ReviewConflict as conflict:
            return conflict

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(assess, range(8)))
    winners = [item for item in outcomes if not isinstance(item, ReviewConflict)]
    conflicts = [item for item in outcomes if isinstance(item, ReviewConflict)]
    assert len(winners) == 1 and len(conflicts) == 7
    assert all(item.current_revision == 1 for item in conflicts)
    assert all(item.current_result_sha256 == completed.result_sha256 for item in conflicts)
    assert store.get_review_history(completed.job_id, completed.result_sha256, 0) == winners


def test_reviews_of_different_events_have_independent_revisions(queue, tmp_path):
    store, _ = queue
    completed = completed_positive(store, tmp_path)
    barrier = threading.Barrier(2)

    def assess(index):
        barrier.wait()
        return store.append_review(completed.job_id, completed.result_sha256, index,
                                   state="confirmed", actor_claimed="Claimed analyst", expected_revision=0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        reviews = list(executor.map(assess, range(2)))
    assert [item.revision for item in reviews] == [1, 1]
    assert [item.violation_index for item in reviews] == [0, 1]


def test_reviews_reject_stale_source_revision_wrong_evidence_and_unknown_event(queue, tmp_path):
    store, _ = queue
    completed = completed_positive(store, tmp_path)
    with pytest.raises(ReviewConflict) as conflict:
        store.get_reviews(completed.job_id, "0" * 64)
    assert conflict.value.current_result_sha256 == completed.result_sha256
    with pytest.raises(ReviewNotAllowed, match="evidence digest"):
        store.append_review(completed.job_id, completed.result_sha256, 0,
                            state="confirmed", actor_claimed="Claimed analyst", expected_revision=0,
                            evidence_digest="0" * 64)
    for index in (-1, 2, True):
        with pytest.raises(ReviewNotAllowed, match="violation index"):
            store.append_review(completed.job_id, completed.result_sha256, index,
                                state="confirmed", actor_claimed="Claimed analyst", expected_revision=0)
    store.append_review(completed.job_id, completed.result_sha256, 0,
                        state="confirmed", actor_claimed="Claimed analyst", expected_revision=0)
    with pytest.raises(ReviewConflict) as stale:
        store.append_review(completed.job_id, completed.result_sha256, 0,
                            state="rejected", actor_claimed="Another claimed analyst", expected_revision=0)
    assert stale.value.current_revision == 1
    assert store.get_reviews(completed.job_id)[0].state == "confirmed"


def test_review_requires_completed_positive_result(queue, tmp_path):
    store, _ = queue
    queued = enqueue(store, tmp_path, "queued-review")
    with pytest.raises(ReviewNotAllowed):
        store.get_reviews(queued.job_id)
    store.cancel(queued.job_id)
    with pytest.raises(ReviewNotAllowed):
        store.get_reviews(queued.job_id)
    clean = enqueue(store, tmp_path, "clean-review")
    claimed = store.claim("worker")
    clean = store.publish(clean.job_id, "worker", claimed.attempt, result(clean.job_id))
    with pytest.raises(ReviewNotAllowed):
        store.get_reviews(clean.job_id)
    with pytest.raises(JobNotFound):
        store.get_reviews("unknown")


def test_review_checks_actual_result_bytes_not_only_database_digest(queue, tmp_path):
    store, _ = queue
    completed = completed_positive(store, tmp_path)
    original = Path(completed.result_path).read_bytes()
    Path(completed.result_path).write_bytes(original + b" ")
    for operation in (
        lambda: store.get_reviews(completed.job_id),
        lambda: store.append_review(completed.job_id, completed.result_sha256, 0,
                                    state="confirmed", actor_claimed="Claimed analyst", expected_revision=0),
        lambda: store.get_review_history(completed.job_id, completed.result_sha256, 0),
    ):
        with pytest.raises(ReviewNotAllowed, match="integrity"):
            operation()
    Path(completed.result_path).write_bytes(original)
    assert store.get_review_history(completed.job_id, completed.result_sha256, 0) == []


def test_review_source_is_rechecked_after_acquiring_write_lock(queue, tmp_path, monkeypatch):
    store, _ = queue
    completed = completed_positive(store, tmp_path)
    original_source = store._review_source

    def changed_source(*args, **kwargs):
        snapshot = original_source(*args, **kwargs)
        with store.engine.begin() as connection:
            connection.execute(jobs.update().where(jobs.c.job_id == completed.job_id)
                               .values(result_sha256="1" * 64))
        return snapshot

    monkeypatch.setattr(store, "_review_source", changed_source)
    with pytest.raises(ReviewConflict) as conflict:
        store.append_review(completed.job_id, completed.result_sha256, 0,
                            state="confirmed", actor_claimed="Claimed analyst", expected_revision=0)
    assert conflict.value.current_result_sha256 == "1" * 64
    with store.engine.begin() as connection:
        connection.execute(jobs.update().where(jobs.c.job_id == completed.job_id)
                           .values(result_sha256=completed.result_sha256))
    monkeypatch.setattr(store, "_review_source", original_source)
    assert store.get_review_history(completed.job_id, completed.result_sha256, 0) == []


@pytest.mark.parametrize("kwargs", [
    {"actor_claimed": " "}, {"actor_claimed": "x" * 121}, {"actor_claimed": "name\nother"},
    {"comment": "x" * 4001}, {"comment": "invalid\x00text"},
    {"expected_revision": -1}, {"expected_revision": True}, {"state": "approved"},
])
def test_review_text_and_revision_bounds(queue, tmp_path, kwargs):
    store, _ = queue
    completed = completed_positive(store, tmp_path)
    arguments = {"state": "confirmed", "actor_claimed": "Claimed analyst", "expected_revision": 0, **kwargs}
    with pytest.raises(ValueError):
        store.append_review(completed.job_id, completed.result_sha256, 0, **arguments)
    assert store.get_review_history(completed.job_id, completed.result_sha256, 0) == []


def test_review_history_survives_restart(queue, tmp_path):
    store, clock = queue
    completed = completed_positive(store, tmp_path)
    assessment = store.append_review(completed.job_id, completed.result_sha256, 0,
                                      state="rejected", actor_claimed="Self-claimed analyst",
                                      expected_revision=0, comment="Synthetic review")
    store.close()
    restarted = JobStore(store.engine.url.render_as_string(hide_password=False), store.result_root, clock=clock)
    try:
        assert restarted.get_reviews(completed.job_id)[0] == assessment
        assert restarted.get_review_history(completed.job_id, completed.result_sha256, 0) == [assessment]
        assert restarted.get_job(completed.job_id) == completed
    finally:
        restarted.close()


def test_review_payload_limit_rejects_oversize_before_reading_file(queue, tmp_path):
    store, _ = queue
    completed = completed_positive(store, tmp_path)
    with Path(completed.result_path).open("ab") as stream:
        stream.truncate(96 * 1024 * 1024 + 1)
    with pytest.raises(ReviewNotAllowed, match="payload limit"):
        store.get_reviews(completed.job_id)
