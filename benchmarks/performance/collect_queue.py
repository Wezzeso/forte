"""Collect only technical queue durations for benchmark-owned jobs over team SSH."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


REMOTE_SOURCE = '''
from datetime import datetime
import json, os
from pathlib import Path
from forte.jobs import JobStore
mapping = json.loads(MAPPING_LITERAL)
store = None
try:
    store = JobStore(os.environ["FORTE_DATABASE_URL"], Path(os.environ["FORTE_DATA_DIR"]) / "results", create_schema=False)
    rows = []
    for sample_id, job_id in mapping.items():
        record = store.get_job(job_id)
        if record is None:
            rows.append({"sample_id": sample_id, "status": "unmeasured"})
            continue
        claimed = record.stage_times.get("claimed")
        claimed = datetime.fromisoformat(claimed) if claimed else None
        queue_ms = None
        if record.attempt == 1 and claimed is not None:
            delta = (claimed - record.accepted_at).total_seconds() * 1000
            queue_ms = delta if delta >= 0 else None
        rows.append({"sample_id": sample_id, "status": record.status, "attempt": record.attempt,
                     "acceptance_to_first_claim_ms": queue_ms,
                     "recovery_queue_note": "unmeasured for recovered attempts" if record.attempt > 1 else None})
    print(json.dumps({"status": "measured", "samples": rows}))
except Exception as error:
    print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
finally:
    if store is not None:
        store.close()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-job-map", type=Path, required=True)
    parser.add_argument("--ssh-alias", default="forte-a100-smoke-20261017")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    parser.add_argument("--worker-container", default="forte-worker-1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) for value in (args.ssh_alias, args.worker_container)):
        parser.error("invalid team SSH/container reference")
    if args.output.suffix.lower() != ".json" or args.output.resolve() == args.private_job_map.resolve():
        parser.error("output must be a separate JSON aggregate report")
    try:
        mapping = json.loads(args.private_job_map.read_text(encoding="utf-8"))["sample_jobs"]
        if not isinstance(mapping, dict) or any(not re.fullmatch(r"s[0-9]+", key) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) for key, value in mapping.items()):
            raise ValueError("unexpected opaque correlation shape")
        source = REMOTE_SOURCE.replace("MAPPING_LITERAL", repr(json.dumps(mapping)))
        remote_command = f"docker exec -i {args.worker_container} python /app/deploy/entrypoint.py python -"
        result = subprocess.run(["wsl", "-d", args.wsl_distro, "--", "ssh", "-T", "-o", "BatchMode=yes",
                                 "-o", "ConnectTimeout=12", args.ssh_alias, remote_command],
                                input=source, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode != 0:
            report = {"status": "failed", "error_code": "team_ssh_or_container_command_failed"}
        else:
            report = json.loads(result.stdout)
        report.update({"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "clock_domain": "same team GPU host UTC for API and worker containers",
                       "clock_assumption": "containers share host kernel clock; no client/server clock subtraction",
                       "measurement": "durable accepted_at to recorded claimed UTC; dispatch after claim excluded",
                       "ntp_state": "not measured by this collector"})
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__, "scope": "queue_numeric_metadata_only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "samples": len(report.get("samples", []))}))
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
