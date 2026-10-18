"""Python 3.10-compatible remote CPU launcher; secrets stay in mounted files.

The launcher performs no service handoff, admission change, GPU operation or
reservation. Bind is offline. Execute requires the exact root release already
validated by the acceptance module. All child stderr/private output is captured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import uuid


def require(value):
    if not value:
        raise RuntimeError("acceptance_host_prerequisite_missing")


def inspect(container):
    require(re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", container))
    process = subprocess.run(["docker", "inspect", container], capture_output=True, timeout=15)
    require(process.returncode == 0)
    rows = json.loads(process.stdout)
    require(len(rows) == 1)
    return rows[0]


def container_path(root, relative, *, must_exist=True):
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and (path.exists() or not must_exist))
    return "/ops/" + path.relative_to(root).as_posix()


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(source.read()).hexdigest()


def run(args):
    root = args.ops_root.resolve()
    require(root.is_dir() and not root.is_symlink())
    api = inspect(args.api_container)
    image = api["Image"]
    require(re.fullmatch(r"sha256:[a-f0-9]{64}", image))
    helper_name = "forte-acceptance-ops-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--name", helper_name, "--user", "0:0", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true", "--entrypoint", "/app/.venv/bin/python",
        "--network", "forte_private" if args.action == "execute" else "none",
        "--env", "PYTHONPATH=/ops:/app", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--volume", str(root) + ":/ops", "--workdir", "/ops"]
    module = "benchmarks.performance.qwen38_acceptance_ops"
    if args.action == "bind":
        require(args.runtime_config and args.worker_recipe and args.comparison_plan and args.arm)
        forwarded = ["bind", "--population", container_path(root, args.population),
            "--runtime-config", container_path(root, args.runtime_config),
            "--worker-recipe", container_path(root, args.worker_recipe),
            "--comparison-plan", container_path(root, args.comparison_plan), "--arm", args.arm,
            "--output", container_path(root, args.bound_root, must_exist=False)]
    elif args.action == "trace-bind":
        require(args.worker_recipe and args.model_release and args.model_reservation and args.source_archive)
        forwarded = ["trace-bind", "--bound-root", container_path(root, args.bound_root),
            "--worker-recipe", container_path(root, args.worker_recipe),
            "--model-release", container_path(root, args.model_release),
            "--model-reservation", container_path(root, args.model_reservation),
            "--source-archive", container_path(root, args.source_archive),
            "--app-image-sha256", image.removeprefix("sha256:"),
            "--output", container_path(root, args.bound_root / "trace-binding.json", must_exist=False)]
    elif args.action == "execute":
        require(args.release and args.reservation)
        # Read only Docker mount metadata. No key bytes enter this process or argv.
        mounts = [row for row in api.get("Mounts", []) if row.get("Destination") == "/run/secrets/api_keys"]
        require(len(mounts) == 1 and mounts[0].get("Type") == "bind")
        key_file = Path(mounts[0]["Source"])
        require(key_file.is_file() and not key_file.is_symlink())
        worker = inspect(args.worker_container)
        worker_command = worker.get("Config", {}).get("Cmd", [])
        require(worker.get("State", {}).get("Running") is True and worker["Image"] == image
                and "evaluation.qwen38_worker_trace" in worker_command and "--binding" in worker_command)
        binding_path = worker_command[worker_command.index("--binding") + 1]
        config_mounts = [row for row in worker.get("Mounts", []) if binding_path == row.get("Destination")
                         or binding_path.startswith(row.get("Destination", "") + "/")]
        require(config_mounts)
        mount = max(config_mounts, key=lambda row: len(row["Destination"]))
        mounted_binding = Path(mount["Source"]) / binding_path.removeprefix(mount["Destination"]).lstrip("/")
        require(checksum(mounted_binding) == checksum(root / args.bound_root / "trace-binding.json"))
        command.extend(["--volume", str(key_file) + ":/run/acceptance-api-key:ro"])
        forwarded = ["execute", "--plan", container_path(root, args.bound_root / "plan.json"),
            "--budget", container_path(root, args.bound_root / "model-budget.json"),
            "--release", container_path(root, args.release), "--reservation", container_path(root, args.reservation),
            "--output", container_path(root, args.run_root, must_exist=False),
            "--api-key-file", "/run/acceptance-api-key"]
    else:
        mounts = [row for row in api.get("Mounts", []) if row.get("Destination") == "/var/lib/forte"]
        require(len(mounts) == 1)
        mount = mounts[0]
        source = mount.get("Name") if mount["Type"] == "volume" else mount["Source"]
        command.extend(["--volume", str(source) + ":/var/lib/forte:ro"])
        forwarded = ["export-report", "--run", container_path(root, args.run_root),
            "--data-root", "/var/lib/forte", "--reference", container_path(root, args.population.parent / "reference.private.json"),
            "--private-output", container_path(root, args.run_root / "terminals.private.json", must_exist=False),
            "--public-output", container_path(root, args.run_root / "aggregate.json", must_exist=False)]
    try:
        process = subprocess.run(command + [image, "-m", module] + forwarded,
            capture_output=True, timeout=1900 if args.action == "execute" else 420 if args.action == "report" else 120)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # Terminate only this uniquely named helper. No service/container from
        # the deployment is stopped, and uncertain submissions stay consumed.
        subprocess.run(["docker", "rm", "--force", helper_name], capture_output=True, timeout=20)
        raise RuntimeError("acceptance_host_helper_interrupted") from None
    # Never forward Docker stderr or arbitrary stdout; only known aggregate fields.
    try:
        result = json.loads(process.stdout)
    except (TypeError, ValueError):
        result = {"status": "blocked", "reason_code": "acceptance_host_child_failed"}
    allowed = {"status", "reason_code", "expected_attempts", "completed", "failed_or_unattempted",
        "source_jobs", "model_calls_made", "maximum_generation_http_attempts", "minimum_initial_model_requests",
        "failed_or_timeout", "missing_or_not_attempted", "server_verified", "less_than_ten_seconds",
        "fresh_repeatability", "ui_ready_less_than_ten_seconds", "targets", "server_receipts_exported",
        "server_receipts_export_failed", "provided_binary_diagnostic"}
    public = {key: value for key, value in result.items() if key in allowed}
    public["private_key_values_exported"] = False
    print(json.dumps(public, sort_keys=True))
    return process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bind", "trace-bind", "execute", "report"))
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--api-container", default="forte-api-1")
    parser.add_argument("--worker-container", default="forte-worker-1")
    parser.add_argument("--population", type=Path, default=Path("benchmarks/private/qwen38/acceptance-six-a12/population.json"))
    parser.add_argument("--bound-root", type=Path, default=Path("benchmarks/private/qwen38/acceptance-six-a12/bound"))
    parser.add_argument("--run-root", type=Path, default=Path("benchmarks/private/qwen38/acceptance-six-a12/run"))
    for name in ("runtime-config", "worker-recipe", "comparison-plan", "release", "reservation",
                 "model-release", "model-reservation", "source-archive"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--arm", choices=("qwen38_tuned", "qwen38_untuned"))
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception:
        print(json.dumps({"status": "blocked", "reason_code": "acceptance_host_prerequisite_missing",
                          "private_key_values_exported": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
