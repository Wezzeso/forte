"""Select the actual worker CPU image for future acceptance operations.

Prepare/trace launch an offline CPU helper with the actual worker's non-secret
configuration and source/config mounts. Execute/report reuse the existing
launcher with the worker as its image/secret/data provider, avoiding API-image
assumptions. No service activation or inference occurs merely by preparation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import uuid

from deploy import qwen38_acceptance_host as base

ENV_FIELDS = frozenset({"FORTE_DATA_DIR", "FORTE_MODEL_BASE_URL", "FORTE_MODEL_NAME",
    "FORTE_MODEL_CANDIDATE_LAUNCH_CONFIG", "FORTE_RULES_PATH", "FORTE_TAXONOMY_PATH",
    "FORTE_MODEL_MANIFEST", "FORTE_WORKER_POLL_SECONDS", "FORTE_LEASE_SECONDS",
    "FORTE_WORKER_HEARTBEAT_SECONDS", "FORTE_JOB_TIMEOUT_SECONDS", "FORTE_WINDOW_SECONDS",
    "FORTE_STRIDE_SECONDS", "FORTE_SAMPLE_FPS", "FORTE_MAX_FRAMES_PER_WINDOW",
    "FORTE_MAX_CONCURRENT_JOBS", "FORTE_MAX_CONCURRENT_CLIPS", "FORTE_MAX_INFLIGHT_PER_JOB",
    "FORTE_COMPACT_FRAME_IDS", "FORTE_CONCISE_OBSERVATIONS", "FORTE_STAGED_SCREENING",
    "FORTE_MAX_REFINEMENT_PASSES", "FORTE_TRUSTED_SCENE_CONTEXT"})
MOUNT_PREFIXES = ("/app", "/run/configs", "/release")


def prepare(args):
    root = args.ops_root.resolve()
    base.require(root.is_dir())
    worker = base.inspect(args.worker_container)
    image = worker["Image"]
    base.require(base.re.fullmatch(r"sha256:[a-f0-9]{64}", image))
    values = json.loads(args.arguments_file.read_text(encoding="utf-8"))
    base.require(isinstance(values, list) and all(isinstance(value, str) and "\x00" not in value for value in values))
    base.require("--cpu-image-sha256" not in values)
    name = "forte-acceptance-rebind-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--name", name, "--read-only", "--network", "none",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--user", "0:0",
        "--entrypoint", "/app/.venv/bin/python", "--env", "PYTHONPATH=/ops:/app",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env",
        "FORTE_ACCEPTANCE_CONTEXT_CPU_IMAGE_SHA256=" + image.removeprefix("sha256:"),
        "--volume", str(root) + ":/ops", "--workdir", "/app"]
    # Secret, database and GPU-related environment values are never forwarded.
    for item in worker.get("Config", {}).get("Env", []):
        key = item.partition("=")[0]
        if key in ENV_FIELDS:
            command.extend(["--env", item])
    for mount in worker.get("Mounts", []):
        destination = mount.get("Destination", "")
        if any(destination == prefix or destination.startswith(prefix + "/") for prefix in MOUNT_PREFIXES):
            source = mount.get("Name") if mount.get("Type") == "volume" else mount.get("Source")
            base.require(source and ":" not in source and ":" not in destination)
            command.extend(["--volume", source + ":" + destination + ":ro"])
    forwarded = [args.action, *values]
    if args.action == "prepare":
        forwarded.extend(["--cpu-image-sha256", image.removeprefix("sha256:")])
    try:
        process = subprocess.run(command + [image, "-m", "benchmarks.performance.qwen38_acceptance_rebind", *forwarded],
                                 capture_output=True, timeout=180)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=20)
        raise RuntimeError("selected_worker_metadata_helper_interrupted") from None
    try:
        result = json.loads(process.stdout)
    except (TypeError, ValueError):
        result = {"status": "blocked", "reason_code": "selected_worker_metadata_helper_failed"}
    allowed = {"status", "reason_code", "expected_attempts", "source_jobs", "maximum_generation_http_attempts",
               "configured_source_equivalence_verified", "historical_attempts_reused", "loaded_model_verified", "model_calls_made"}
    print(json.dumps({key: value for key, value in result.items() if key in allowed}, sort_keys=True))
    return process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "trace", "execute", "report"))
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--worker-container", default="forte-worker-1")
    parser.add_argument("--arguments-file", type=Path)
    parser.add_argument("--population", type=Path, default=Path("benchmarks/private/qwen38/acceptance-six-a12/population.json"))
    parser.add_argument("--bound-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--reservation", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action in {"prepare", "trace"}:
            base.require(args.arguments_file is not None)
            return prepare(args)
        base.require(args.bound_root is not None and args.run_root is not None)
        # The actual worker supplies CPU image, secrets and shared data mounts.
        # No API container identity is substituted or claimed here.
        args.api_container = args.worker_container
        return base.run(args)
    except Exception:
        print(json.dumps({"status": "blocked", "reason_code": "selected_worker_acceptance_prerequisite_missing",
                          "private_values_exported": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
