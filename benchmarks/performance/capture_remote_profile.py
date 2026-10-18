"""Capture whitelisted image/health/profile flags without exposing container secrets."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

import httpx


FLAGS = {"FORTE_STAGED_SCREENING": "bool", "FORTE_CONCISE_OBSERVATIONS": "bool",
         "FORTE_COMPACT_FRAME_IDS": "bool", "FORTE_MAX_INFLIGHT_PER_JOB": "int",
         "FORTE_MAX_CONCURRENT_CLIPS": "int", "FORTE_JOB_TIMEOUT_SECONDS": "int"}


def sanitized_profile(raw):
    safe = []
    for item in raw:
        env = dict(value.split("=", 1) for value in item.get("Config", {}).get("Env", []) if "=" in value)
        flags = {}
        for name, kind in FLAGS.items():
            value = env.get(name)
            if kind == "bool":
                flags[name] = value.lower() == "true" if isinstance(value, str) and value.lower() in {"true", "false"} else None
            else:
                flags[name] = int(value) if isinstance(value, str) and re.fullmatch("[0-9]{1,6}", value) else None
        state = item.get("State", {})
        health = state.get("Health", {}).get("Status")
        image = item.get("Image")
        name = item.get("Name", "").lstrip("/")
        safe.append({"container_role": "api" if name == "forte-api-1" else "worker" if name == "forte-worker-1" else "unrecognized",
                     "image_id": image if isinstance(image, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", image) else None,
                     "health": health if health in {"healthy", "unhealthy", "starting"} else "unmeasured",
                     "running": state.get("Running") is True, "profile_flags": flags})
    return safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-staged-concise", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("profile evidence already exists; choose a new filename")
    result = subprocess.run(["wsl", "-d", "Ubuntu", "--", "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                             "forte-a100-smoke-20261017", "docker inspect forte-api-1 forte-worker-1"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode != 0:
        report = {"status": "failed", "error_code": "team_container_inspection_failed"}
    else:
        containers = sanitized_profile(json.loads(result.stdout))
        worker = next((row for row in containers if row["container_role"] == "worker"), {})
        flags = worker.get("profile_flags", {})
        expected = {"FORTE_STAGED_SCREENING": True, "FORTE_CONCISE_OBSERVATIONS": True,
                    "FORTE_COMPACT_FRAME_IDS": True, "FORTE_MAX_INFLIGHT_PER_JOB": 2,
                    "FORTE_MAX_CONCURRENT_CLIPS": 2}
        response = httpx.get("https://hackathon.wezeso.me/readyz", timeout=15, follow_redirects=False, trust_env=False)
        ready = response.status_code == 200
        health_ok = (len(containers) == 2 and {row["container_role"] for row in containers} == {"api", "worker"}
                     and len({row["image_id"] for row in containers}) == 1
                     and all(row["running"] and row["health"] == "healthy" and row["image_id"] for row in containers))
        profile_ok = all(flags.get(key) == value for key, value in expected.items())
        report = {"status": "verified" if ready and health_ok and (profile_ok or not args.require_staged_concise) else "failed",
                  "schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "containers": containers, "strict_public_readiness_http_status": response.status_code,
                  "required_staged_concise_profile_matches": profile_ok,
                  "scope": "actual team Docker image IDs/health and fixed profile flags; all other environment values discarded",
                  "model_configuration_or_edge_protocol_attested_here": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "profile_matches": report.get("required_staged_concise_profile_matches")}))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
