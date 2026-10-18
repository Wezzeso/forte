"""Read only opaque ID/static code for A20's known generated-MOV fixture."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


SOURCE = '''
import json, os, re
from pathlib import Path
from sqlalchemy import select, func
from forte.jobs import JobStore
from forte.jobs.store import jobs
store = None
try:
    store = JobStore(os.environ["FORTE_DATABASE_URL"], Path(os.environ["FORTE_DATA_DIR"]) / "results", create_schema=False)
    condition = (jobs.c.original_filename == "synthetic.mov") & (jobs.c.status == "failed")
    with store.engine.connect() as connection:
        count = connection.execute(select(func.count()).select_from(jobs).where(condition)).scalar_one()
        row = connection.execute(select(jobs.c.job_id, jobs.c.error_code, jobs.c.attempt).where(condition).order_by(jobs.c.accepted_at.desc()).limit(1)).first()
    if row is None:
        output = {"status": "not_found", "matching_failed_synthetic_records": count}
    else:
        code = row.error_code
        code = code if isinstance(code, str) and re.fullmatch("[a-z][a-z0-9_]{0,63}", code) else "unrecognized_static_error_code"
        output = {"status": "measured", "job_id": row.job_id, "error_code": code,
                  "matching_failed_synthetic_records": count, "unique_match": count == 1}
        work_root = (Path(os.environ["FORTE_DATA_DIR"]) / "work").resolve()
        audit = (work_root / row.job_id / ("attempt-" + str(row.attempt)) / "analysis.private.json").resolve()
        output["audit_available"] = bool(audit.is_relative_to(work_root) and audit.is_file())
        if output["audit_available"]:
            raw = json.loads(audit.read_text(encoding="utf-8"))
            phase = raw.get("phase")
            static_code = raw.get("error_code")
            safe = {"phase": phase if phase in {"coarse", "refined", "ready", "done", "failed"} else "unrecognized_phase",
                    "error_code": static_code if isinstance(static_code, str) and re.fullmatch("[a-z][a-z0-9_]{0,63}", static_code) else None}
            for key in ("coarse_requests", "refinement_requests"):
                safe[key] = raw.get(key) if type(raw.get(key)) is int and raw[key] >= 0 else None
            from forte.observability import StageTimer
            from forte.observability.timing import STAGES
            timings = raw.get("timings", {})
            stages = timings.get("stages_ms", {}) if isinstance(timings, dict) else {}
            import math
            safe["stages_ms"] = {key: float(value) for key in STAGES if type(value := stages.get(key)) in {int, float} and math.isfinite(value) and value >= 0}
            output["fixed_audit_fields"] = safe
    print(json.dumps(output))
except Exception as error:
    print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
finally:
    if store is not None:
        store.close()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = subprocess.run(["wsl", "-d", "Ubuntu", "--", "ssh", "-T", "-o", "BatchMode=yes",
                                 "-o", "ConnectTimeout=12", "forte-a100-smoke-20261017",
                                 "docker exec -i forte-worker-1 python /app/deploy/entrypoint.py python -"],
                                input=SOURCE, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        report = json.loads(result.stdout) if result.returncode == 0 else {"status": "failed", "error_code": "team_ssh_or_container_failed"}
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__}
    report.update({"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "scope": "known A20 generated synthetic.mov; opaque ID and static error_code only",
                   "media_or_prediction_or_error_detail_inspected_or_exported": False, "model_inference_called": False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "job_id", "error_code", "unique_match") if key in report}))
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
