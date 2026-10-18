"""Kill a real leased Worker process and recover via an explicitly synthetic pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from forte.jobs import JobStore, LeaseLost


def run_recovery_check() -> dict:
    project = Path(__file__).resolve().parents[2]
    fixture = Path(__file__).with_name("worker_fault_fixture.py")
    child = None
    store = None
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix="forte-synthetic-worker-qa-") as directory:
        work = Path(directory)
        if work.resolve().parent != Path(tempfile.gettempdir()).resolve() or not work.name.startswith("forte-synthetic-worker-qa-"):
            raise RuntimeError("unexpected synthetic runtime directory")
        database = "sqlite:///" + (work / "jobs.db").as_posix()
        store = JobStore(database, work / "results", lease_seconds=1.5,
                         job_timeout_seconds=40, max_attempts=3)
        source = work / "synthetic.mov"
        source.write_bytes(b"synthetic worker process fixture; not real media")
        job_id = "synthetic-worker-recovery"
        store.create_job(job_id, source, hashlib.sha256(source.read_bytes()).hexdigest(),
                         source.stat().st_size, "synthetic.mov")
        marker = work / "marker.json"
        environment = {**os.environ, "PYTHONPATH": str(project / "src") + os.pathsep + str(project),
                       "FORTE_SYNTHETIC_FAULT_FIXTURE": "explicitly-enabled-for-qa",
                       "FORTE_DATA_DIR": str(work), "FORTE_DATABASE_URL": database,
                       "FORTE_LEASE_SECONDS": "1.5", "FORTE_WORKER_HEARTBEAT_SECONDS": ".3",
                       "FORTE_JOB_TIMEOUT_SECONDS": "40", "FORTE_WORKER_POLL_SECONDS": ".05",
                       "FORTE_WORKER_ID": "synthetic-first-worker"}
        command = [sys.executable, str(fixture), "--mode", "stall", "--marker", str(marker)]
        try:
            child = subprocess.Popen(command, cwd=project, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            marker_deadline = time.monotonic() + 15
            while not marker.is_file() and time.monotonic() < marker_deadline and child.poll() is None:
                time.sleep(.025)
            if not marker.is_file():
                return {"status": "FAIL", "error_code": "first_worker_did_not_claim"}
            first = store.get_job(job_id)
            if first is None or first.attempt != 1 or first.lease_owner != "synthetic-first-worker":
                return {"status": "FAIL", "error_code": "first_attempt_not_durably_owned"}
            killed = time.perf_counter_ns()
            child.kill()  # Abrupt OS process termination; no graceful lease release.
            child.communicate(timeout=5)
            child = None
            marker.unlink()
            # Wait beyond the final real heartbeat lease. The replacement owns
            # recovery; no synthetic clock advancement occurs in this check.
            time.sleep(1.7)
            environment["FORTE_WORKER_ID"] = "synthetic-replacement-worker"
            replacement = subprocess.run([sys.executable, str(fixture), "--mode", "complete", "--marker", str(marker)],
                                         cwd=project, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            record = store.get_job(job_id)
            if replacement.returncode != 0 or record is None or record.status != "done" or record.attempt != 2:
                return {"status": "FAIL", "error_code": "replacement_did_not_publish_attempt_two"}
            fenced = False
            try:
                store.publish(job_id, "synthetic-first-worker", 1,
                              {"job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []})
            except LeaseLost:
                fenced = True
            return {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "PASS" if fenced else "FAIL", "old_attempt_fenced": fenced,
                    "attempts": record.attempt, "actual_worker_process_abruptly_killed": True,
                    "synthetic_pipeline": True, "real_gpu_or_video_analysis": False,
                    "storage": "SQLite durable file", "lease_seconds": 1.5,
                    "kill_to_replacement_publication_ms": (time.perf_counter_ns() - killed) / 1e6,
                    "total_check_ms": (time.perf_counter_ns() - started) / 1e6,
                    "scope": "actual Worker/job-store/publication behavior with synthetic pipeline only"}
        except Exception as exc:
            return {"status": "FAIL", "error_code": "fixture_process_error", "exception_type": type(exc).__name__}
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=5)
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("output must be a JSON report")
    report = run_recovery_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
