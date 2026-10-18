# Durable job service

Production uses PostgreSQL through SQLAlchemy and psycopg. The API and workers
must share persistent input and result paths at the same absolute mount points.
The explicit `sqlite:///...` URL exists for local development on Windows; it is
not a substitute for PostgreSQL concurrency/capacity validation.

Bootstrap the initial schema before API/workers start:

```powershell
$env:FORTE_DATABASE_URL = 'postgresql+psycopg://forte:password@db:5432/forte'
python -m forte.jobs init --result-root /var/lib/forte/results
python -m forte.jobs recover --result-root /var/lib/forte/results
python -m forte.jobs counts --result-root /var/lib/forte/results
```

`init` is idempotent. PostgreSQL advisory locking serializes simultaneous schema
bootstrap; SQLite initialization uses its immediate write transaction. This is
the initial schema, not a general migration runner. Review a migration before
changing a populated installation's schema. The queue does not use a broker.

`JobStore(database_url, result_root, lease_seconds=30,
job_timeout_seconds=285, max_attempts=3)` exposes synchronous methods suitable
for an API thread pool. All returned timestamps are aware UTC datetimes.
PostgreSQL uses its own `clock_timestamp()` for lease/deadline checks so host
clock skew cannot grant stale ownership. Tests may inject an aware clock.

```python
from forte.jobs import JobStore, LeaseLost

store = JobStore(database_url, result_root)
# The uploader first closes/fsyncs the complete owned file and its directory.
accepted = store.create_job(job_id, owned_path, sha256, size_bytes, filename)
# Only ready, prewarmed workers claim. Claim closes its transaction immediately.
job = store.claim(worker_id)
if job is not None:
    store.set_stage(job.job_id, worker_id, job.attempt, "decoding")
    # Heartbeat independently during decode/inference, before lease expiry.
    store.heartbeat(job.job_id, worker_id, job.attempt)
    # Run local processing with no database transaction held.
    store.publish(job.job_id, worker_id, job.attempt, completed_response)
```

The integer `attempt` is the fencing token. Heartbeats, stage changes, failures,
and publications require the matching worker and live attempt. `LeaseLost`
means stop work; an expired worker cannot renew itself. `claim` uses a short
`FOR UPDATE SKIP LOCKED` transaction, preserves acceptance order among available
rows, and increments the attempt when reclaiming a crashed worker's lease.
At most 64 expired rows are finalized by one claim to bound housekeeping time.
`recover(limit=1000)` explicitly requeues expired leases and fails exhausted or
expired jobs. Both `get_job` and `claim` enforce the absolute deadline.

`fail(job_id, worker_id, attempt, error, retryable=True)` requeues only when an
attempt remains and its backoff fits before the deadline. Default backoff is
`min(2**attempt, 30)` seconds. Set `retry_delay_seconds` explicitly when needed.
Terminal errors and `cancel(job_id)` expose exactly `job_id` and `status=failed`;
private details remain in the database. Queued and active work expose
`status=processing`. `set_stage` accepts `claimed`, `decoding`, `screening`,
`refining`, and `assembling`.

`publish` accepts an assembled `bytes` payload or a mapping and validates A01's
`CompletedResponse` before publication. Validated bytes are retained exactly,
including the pre-encoded evidence and JSON formatting. It writes and
fsyncs a temporary JSON file, atomically replaces an attempt/content-addressed
destination, fsyncs its directory where supported, then verifies current
ownership under a row lock before marking the row done. If disk I/O, a lease
check, or the database commit fails, no partial result is exposed. A crash before
commit can leave an unreferenced file. A repeated identical publication by the
same attempt is idempotent. A different or stale attempt cannot overwrite the
committed pointer. FileResponse can serve `record.result_path` directly;
`get_result_bytes` also returns the pre-encoded bytes without re-encoding.

Accepted inputs and completed JSON files are never deleted automatically.
Orphan files require an operator retention policy that preserves active
attempts and every referenced result. Do not indiscriminately clean this shared
directory during judging. Back up the database and shared volume together.

`on_transition(record)` receives committed transitions for metrics; callback
failure cannot invalidate acceptance. `record.stage_times` maps stages to their
latest UTC timestamp, and `record.remaining_seconds()` helps bound work.

Synthetic verification is in `tests/test_jobs.py`. It exercises concurrent
claims, restart, crash recovery, stale owners, deadlines, retries, cancellation,
publication failures/races, and public schema validation using local SQLite.
An isolated local PostgreSQL 18.6 run passed 44 tests, including concurrent
claims, restart/recovery, bypassing an actually locked row with `SKIP LOCKED`,
default database-clock lease expiry, review history persistence, source/evidence
binding, and eight simultaneous reviewers with one winner and seven revision
conflicts. The temporary test container and volume were removed afterward.
Set `FORTE_TEST_POSTGRES_URL` to a
dedicated empty test schema to repeat the opt-in run; its fixture clears the
test jobs table between cases. Never point this variable at accepted work.

## Human review ledger

Operational reviews are separate from the model response. The additive
`human_review_history` table stores append-only revisions for each
`(job_id, result_sha256, violation_index)`. `get_reviews` returns an implicit
`pending` revision zero until somebody submits an assessment; subsequent states
are `confirmed`, `rejected`, or explicitly reset to `pending`. A claimed reviewer
name is self-asserted text, not a verified identity or authorization mechanism.
API authentication and who may submit decisions are the API's responsibility.

```python
reviews = store.get_reviews(job_id, immutable_result_sha256)
assessment = store.append_review(
    job_id, immutable_result_sha256, violation_index,
    state="confirmed", actor_claimed="Claimed reviewer name",
    comment="Human assessment", expected_revision=reviews[violation_index].revision,
    evidence_digest=reviews[violation_index].evidence_digest,
)
history = store.get_review_history(
    job_id, immutable_result_sha256, violation_index,
    after_revision=0, limit=100,
)
```

Only completed positive jobs and actual event indices are reviewable. Each read
checks the stored response's actual SHA-256 before parsing it, with a 96 MiB
limit and no repeated JPEG decoding. Each record also binds the canonical
evidence-frames digest (frame timestamps and exact JPEG base64). An append
rechecks the completed job's source hash while holding its short row lock, then
compares the expected revision and inserts the next history row. Concurrent
decisions based on the same revision have exactly one winner. `ReviewConflict`
exposes `current_revision` and `current_result_sha256` for an HTTP 409 response.
History pages use an exclusive `after_revision` cursor and a maximum limit of
500. Actor names are 1–120 visible characters; comments are at most 4000.

Review operations never modify the jobs row or immutable jury result bytes.
Backups must include the ledger table with jobs and the shared media/result
volume. Schema creation is additive through the coordinated `init` bootstrap;
deploy it only when the service operator has scheduled the change.
