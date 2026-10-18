"""Private hash-only snapshot of every accepted service job; metadata only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


SOURCE = '''
import json, os, re
from datetime import timezone
from pathlib import Path
from sqlalchemy import select, func
from forte.jobs import JobStore
from forte.jobs.store import jobs
store = None
try:
    store = JobStore(os.environ["FORTE_DATABASE_URL"], Path(os.environ["FORTE_DATA_DIR"]) / "results", create_schema=False)
    with store.engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
        with connection.begin():
            captured = connection.execute(select(func.current_timestamp())).scalar_one()
            rows = connection.execute(select(jobs.c.input_sha256, jobs.c.status)).all()
    hashes = [digest for digest, _ in rows if isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest)]
    print(json.dumps({"schema_version": "forte.service-content-exposure-inventory.v1",
                      "coverage_state": "complete_snapshot" if len(hashes) == len(rows) else "incomplete_invalid_hash_metadata",
                      "scope": "all_accepted_jobs_since_service_start",
                      "captured_at_utc": captured.astimezone(timezone.utc).isoformat(),
                      "snapshot_isolation": "PostgreSQL REPEATABLE READ",
                      "accepted_job_count": len(rows), "unique_content_count": len(set(hashes)),
                      "active_job_count": sum(status not in {"done", "failed", "cancelled"} for _, status in rows),
                      "records": [{"content_hash": value} for value in hashes],
                      "exposure_policy": "conservatively reserve every accepted content; generated contents may not match corpus",
                      "later_uploads_included": False}))
except Exception as error:
    print(json.dumps({"coverage_state": "failed", "error_type": type(error).__name__}))
finally:
    if store is not None:
        store.close()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new private JSON snapshot; never overwritten")
    parser.add_argument("--aggregate-output", type=Path, default=None)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to((Path.cwd() / "benchmarks/private").resolve()) or args.output.suffix != ".json":
        parser.error("accepted content hashes must remain private JSON")
    if args.output.exists():
        parser.error("private snapshot already exists; choose a new filename")
    result = subprocess.run(["wsl", "-d", "Ubuntu", "--", "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                             "forte-a100-smoke-20261017", "docker exec -i forte-worker-1 python /app/deploy/entrypoint.py python -"],
                            input=SOURCE, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    report = json.loads(result.stdout) if result.returncode == 0 else {"coverage_state": "failed", "error_code": "team_metadata_query_failed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(report, indent=2).encode("utf-8")
    args.output.write_bytes(data)
    aggregate = {key: report.get(key) for key in ("coverage_state", "scope", "captured_at_utc", "snapshot_isolation",
                                                  "accepted_job_count", "unique_content_count", "active_job_count", "later_uploads_included")}
    aggregate.update({"schema_version": "1.0", "source_inventory_sha256": hashlib.sha256(data).hexdigest(),
                      "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                      "limits": "snapshot of committed accepted jobs only; later uploads require a fresh snapshot and registry ingestion",
                      "media_or_results_or_names_exported": False})
    if args.aggregate_output:
        if args.aggregate_output.resolve() == args.output.resolve():
            parser.error("aggregate and private hash snapshot must be separate")
        args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate_output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate))
    return 0 if report.get("coverage_state") == "complete_snapshot" else 1


if __name__ == "__main__":
    raise SystemExit(main())
