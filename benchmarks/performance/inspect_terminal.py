"""Inspect fixed technical fields for benchmark-owned terminal jobs only."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


REMOTE_SOURCE = '''
import json, math, os, re, shutil
from pathlib import Path
from forte.jobs import JobStore
from forte.observability.timing import STAGES
from forte.observability.model_metrics import sanitize_model_metrics
mapping = json.loads(MAPPING_LITERAL)
store = None
def static_code(value):
    return value if isinstance(value, str) and re.fullmatch("[a-z][a-z0-9_]{0,63}", value) else None
try:
    root = (Path(os.environ["FORTE_DATA_DIR"]) / "work").resolve()
    store = JobStore(os.environ["FORTE_DATABASE_URL"], Path(os.environ["FORTE_DATA_DIR"]) / "results", create_schema=False)
    rows, handoffs = [], []
    for sample_id, job_id in mapping.items():
        record = store.get_job(job_id)
        if record is None:
            rows.append({"sample_id": sample_id, "status": "not_found"})
            continue
        row = {"sample_id": sample_id, "status": record.status, "attempt": record.attempt,
               "error_code": static_code(record.error_code), "audit_available": False}
        audit = (root / job_id / ("attempt-" + str(record.attempt)) / "analysis.private.json").resolve()
        terminal = record.status in {"done", "failed"}
        row["terminal_verified"] = terminal
        if terminal and audit.is_relative_to(root) and audit.is_file():
            raw = json.loads(audit.read_text(encoding="utf-8"))
            row["audit_available"] = True
            phase = raw.get("phase")
            safe = {"phase": phase if phase in {"coarse", "refined", "ready", "done", "failed"} else "unrecognized_phase",
                    "error_code": static_code(raw.get("error_code"))}
            for key in ("coarse_requests", "refinement_requests"):
                safe[key] = raw.get(key) if type(raw.get(key)) is int and raw[key] >= 0 else None
            timings = raw.get("timings", {})
            stages = timings.get("stages_ms", {}) if isinstance(timings, dict) else {}
            safe["stages_ms"] = {key: float(value) for key in STAGES if type(value := stages.get(key)) in {int, float} and math.isfinite(value) and value >= 0}
            safe["model_requests"] = sanitize_model_metrics(timings.get("model_requests")) if isinstance(timings, dict) else None
            row["fixed_audit_fields"] = safe
            if shutil.which("python3"):
                handoffs.append({"sample_id": sample_id, "job_id": job_id, "terminal_status": record.status,
                                 "remote_audit_path": str(audit), "remote_work_root": str(root), "remote_frame_root": str(root),
                                 "remote_python": "python3"})
        rows.append(row)
    print(json.dumps({"status": "measured", "samples": rows, "private_handoffs": handoffs}))
except Exception as error:
    print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
finally:
    if store is not None:
        store.close()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-job-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-handoff", type=Path)
    parser.add_argument("--private-handoff-directory", type=Path)
    args = parser.parse_args()
    try:
        mapping = json.loads(args.private_job_map.read_text(encoding="utf-8"))["sample_jobs"]
        if not isinstance(mapping, dict) or any(not re.fullmatch(r"s[0-9]+", key) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) for key, value in mapping.items()):
            raise ValueError("unexpected opaque correlation shape")
        if args.private_handoff:
            args.private_handoff.resolve().relative_to((Path.cwd() / "benchmarks/private").resolve())
        if args.private_handoff_directory:
            args.private_handoff_directory.resolve().relative_to((Path.cwd() / "benchmarks/private").resolve())
        if args.private_handoff and args.private_handoff_directory:
            raise ValueError("choose one private handoff destination")
        source = REMOTE_SOURCE.replace("MAPPING_LITERAL", repr(json.dumps(mapping)))
        result = subprocess.run(["wsl", "-d", "Ubuntu", "--", "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                                 "forte-a100-smoke-20261017", "docker exec -i forte-worker-1 python /app/deploy/entrypoint.py python -"],
                                input=source, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        report = json.loads(result.stdout) if result.returncode == 0 else {"status": "failed", "error_code": "team_ssh_or_container_failed"}
        handoffs = report.pop("private_handoffs", [])
        handoff_paths = []
        for handoff in handoffs:
            sample_id = handoff.pop("sample_id")
            destination = args.private_handoff if args.private_handoff and len(handoffs) == 1 else args.private_handoff_directory / (sample_id + ".json") if args.private_handoff_directory else None
            if destination is None:
                continue
            handoff.update({"schema_version": "forte.proposal-artifact-handoff.v1", "terminal_verified": True,
                            "artifact_available": True, "ssh_target": "forte-a100-smoke-20261017",
                            "ssh_wsl_distribution": "Ubuntu", "remote_container": "forte-worker-1", "remote_docker_sudo": False})
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
            handoff_paths.append(str(destination.resolve()))
            report["private_handoff_status"] = "written"
        if args.private_handoff_directory and handoff_paths:
            index = {"schema_version": "forte.proposal-artifact-handoff-index.v1", "terminal_verified": True,
                     "artifact_count": len(handoff_paths), "handoff_paths": handoff_paths}
            (args.private_handoff_directory / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
            report["private_handoff_count"] = len(handoff_paths)
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__}
    report.update({"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "scope": "benchmark-owned terminal technical metadata only",
                   "media_or_prediction_or_error_detail_exported": False, "model_inference_called": False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "samples": len(report.get("samples", [])),
                      "private_handoff_status": report.get("private_handoff_status", "not_written")}))
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
