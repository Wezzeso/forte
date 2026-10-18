"""Root-operated startup supervisor; separate from immutable model source.

Default inspect mode is CPU/read-only. Verify mode hashes artifacts without
GPU access. Start mode is only for the root's later exclusive GPU handoff and
generated-call reservation. This script never changes admission or workers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time


class OpsError(RuntimeError):
    pass


def need(condition, code):
    if not condition:
        raise OpsError(code)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def private_write(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command(args, timeout=30):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    need(result.returncode == 0, "startup_control_command_failed")
    return result.stdout


def inspect_container(name):
    return json.loads(command(["docker", "inspect", name]))[0]


def expected_identity(config):
    need(config["backend"] in {"transformers_base_diagnostic", "transformers_peft"},
         "startup_backend_invalid")
    if config["backend"] == "transformers_peft":
        return {"model_id": config["model_id"], "model_revision": config["model_revision"],
            "processor_revision": config["processor_revision"], "backend": config["backend"],
            "model_manifest_sha256": config["base_manifest"]["sha256"],
            "tokenizer_sha256": config["tokenizer_sha256"], "chat_template_sha256": config["chat_template_sha256"],
            "adapter_sha256": config["adapter_sha256"], "adapter_config_sha256": config["adapter_config_sha256"],
            "training_run_sha256": config["training_run"]["sha256"],
            "quantization_sha256": canonical(config["quantization"]),
            "generation_sha256": canonical(config["generation"]), "config_sha256": canonical(config),
            "image_sha256": config["runtime_image_sha256"], "source_sha256": config["source_archive"]["sha256"]}
    return {"model_id": config["model_id"], "model_revision": config["model_revision"],
        "processor_revision": config["processor_revision"], "backend": config["backend"], "arm": "qwen38_untuned",
        "diagnostic_only": True, "production_selected": False, "trained": False,
        "adapter_sha256": None, "adapter_config_sha256": None, "training_run_sha256": None,
        "model_manifest_sha256": config["base_manifest"]["sha256"], "tokenizer_sha256": config["tokenizer_sha256"],
        "chat_template_sha256": config["chat_template_sha256"], "quantization_sha256": canonical(config["quantization"]),
        "generation_sha256": canonical(config["generation"]), "runtime_profile_sha256": config["runtime_profile_sha256"],
        "image_sha256": config["runtime_image_sha256"], "source_sha256": config["source_archive"]["sha256"],
        "config_sha256": canonical(config)}


def safe_line(line):
    """Never persist arbitrary runtime text, generated descriptions or paths."""
    try:
        value = json.loads(line)
    except (ValueError, TypeError):
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Timeout)):", line.strip())
        return {"event": "runtime_exception_class", "error_type": match[1]} if match else None
    allowed = {"candidate_snapshot_verification_started", "candidate_load_started",
               "candidate_adapter_loaded_and_tensor_verified", "candidate_ready",
               "untuned_diagnostic_ready", "untuned_start_failed", "candidate_start_failed"}
    if not isinstance(value, dict) or value.get("event") not in allowed:
        return None
    result = {"event": value["event"]}
    for key in ("reason_code", "error_type"):
        if isinstance(value.get(key), str) and re.fullmatch(r"[A-Za-z0-9_.]{1,100}", value[key]):
            result[key] = value[key]
    return result


def preflight(plan, *, activation=False):
    config_path = Path(plan["remote_config_path"])
    need(digest(config_path) == plan["config_file_sha256"], "startup_config_raw_hash_mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    need(canonical(config) == plan["config_sha256"], "startup_config_semantic_hash_mismatch")
    image = json.loads(command(["docker", "image", "inspect", "sha256:" + plan["image_sha256"]]))[0]
    need(image["Id"] == "sha256:" + plan["image_sha256"], "startup_image_mismatch")
    net = json.loads(command(["docker", "network", "inspect", plan["network_preflight"]["network_name"]]))[0]
    need(net["Internal"] is True and net["Id"] == plan["network_preflight"]["network_id"],
         "startup_internal_network_mismatch")
    alias = plan["network_preflight"]["alias"]
    for cid in (net.get("Containers") or {}):
        member = inspect_container(cid)
        connection = member["NetworkSettings"]["Networks"][net["Name"]]
        need(alias not in (connection.get("Aliases") or []) + (connection.get("DNSNames") or []),
             "startup_unique_alias_collision")
    argv = plan["gpu_container_create_argv"]
    need(not any(flag in {"-p", "--publish", "-P", "--publish-all"}
                 or flag.startswith(("--publish=", "--publish-all=", "-p")) for flag in argv)
         and argv[argv.index("--network") + 1] == net["Name"]
         and argv[argv.index("--network-alias") + 1] == alias, "startup_plan_network_or_ports_invalid")
    if activation:
        api = inspect_container("forte-api-1")
        # Read only this public boolean; never emit container environment.
        paused = any(value.lower() == "forte_admission_paused=true" for value in api["Config"].get("Env", []))
        need(paused, "startup_api_admission_must_be_paused")
        need(inspect_container("forte-worker-1")["State"]["Running"] is False,
             "startup_worker_must_be_stopped")
        need(datetime.now(timezone.utc).timestamp() + 2520 < datetime(2026, 10, 18, 6, 1, 25,
             tzinfo=timezone.utc).timestamp(), "startup_fixed_stop_headroom_insufficient")
    return config, net


def supervise(plan, output: Path, release_id: str, *, wall_seconds=2400):
    need(re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", release_id) is not None, "startup_root_release_id_required")
    need(60 <= wall_seconds <= 2400, "startup_wall_budget_invalid")
    config, net = preflight(plan, activation=True)
    expected = expected_identity(config)
    cache = Path(plan["private_cache_directory"]["path"])
    need(not cache.exists(), "startup_fresh_cache_required")
    cache.mkdir(mode=0o700)
    need(cache.stat().st_uid == 1000, "startup_cache_owner_mismatch")
    for relative in ("xdg/torch/kernels", "hf", "triton", "cuda", "torchinductor"):
        (cache / relative).mkdir(mode=0o700, parents=True, exist_ok=False)
    argv = plan["gpu_container_create_argv"]
    name = argv[argv.index("--name") + 1]
    record = {"schema_version": "forte.qwen38.generated-startup-ready.v1", "status": "starting",
              "root_release_id": release_id, "container_name": name, "network_name": net["Name"],
              "network_id": net["Id"], "network_alias": plan["network_preflight"]["alias"],
              "source_archive_sha256": plan["source_archive_sha256"], "runtime_image_sha256": plan["image_sha256"],
              "config_file_sha256": plan["config_file_sha256"], "config_sha256": plan["config_sha256"],
              "runtime_profile_sha256": plan["runtime_profile_sha256"], "no_published_ports": True,
              "model_bearer_auth_configured": False, "generated_startup_upper_bound": 6,
              "cache_subdirectories_created": ["xdg/torch/kernels", "hf", "triton", "cuda", "torchinductor"],
              "cache_directory_mode": "0700", "cache_owner_uid": 1000,
              "private_media_used": False, "private_quality_calls": 0, "automatic_retry": False,
              "queue_idle_evidence_owner": "fresh_root_handoff", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    attach = None
    reader = None
    log = output / "startup.safe-events.jsonl"
    started = time.monotonic()
    try:
        command(argv)
        command(["docker", "start", name])
        attach = subprocess.Popen(["docker", "attach", "--sig-proxy=false", "--no-stdin", name],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        def collect():
            descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                for line in attach.stdout:
                    safe = safe_line(line)
                    if safe:
                        safe["elapsed_seconds"] = round(time.monotonic() - started, 3)
                        stream.write(json.dumps(safe) + "\n")
                        stream.flush()

        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        while time.monotonic() - started < wall_seconds:
            state = inspect_container(name)["State"]
            need(state["Running"] is True, "generated_startup_container_exited_before_ready")
            checked = subprocess.run(plan["readiness_probe_argv"], capture_output=True, text=True, timeout=15)
            if checked.returncode == 0:
                state_path = ("/run/forte/qwen38-untuned-ready.json"
                              if config["backend"] == "transformers_base_diagnostic"
                              else "/run/forte/qwen38-ready.json")
                raw = command(["docker", "exec", name, "/opt/forte-features/bin/python", "-B", "-c",
                    "import sys; from pathlib import Path; print(Path(sys.argv[1]).read_text())", state_path])
                need(len(raw.encode()) <= 1024 * 1024, "startup_ready_record_too_large")
                ready = json.loads(raw)
                need(ready.get("ready") is True and ready.get("config_sha256") == plan["config_sha256"],
                     "startup_actual_ready_config_mismatch")
                need(all(ready.get(key) == value for key, value in expected.items()),
                     "startup_actual_ready_identity_mismatch")
                attempts = ready.get("schema_smoke", {}).get("metrics", {}).get("attempts")
                warmup = ready.get("warmup")
                need(type(attempts) is int and attempts in (1, 2) and isinstance(warmup, list) and len(warmup) == 4,
                     "startup_generated_attempt_accounting_invalid")
                actual_path = output / "startup.actual-ready.json"
                private_write(actual_path, ready)
                record.update(status="passed", generated_startup_posts=4 + attempts,
                    actual_ready_identity={key: ready[key] for key in expected},
                    expected_runtime_identity=expected,
                    actual_ready_record_sha256=digest(actual_path), actual_state_path=state_path,
                    initialization_seconds=ready.get("structured_decoding", {}).get("initialization_seconds"),
                    public_grammar_warmup=ready.get("structured_decoding", {}).get("public_schema_warmup"),
                    startup_elapsed_seconds=time.monotonic() - started,
                    request_timeout_seconds_unchanged=config["limits"]["request_timeout_seconds"])
                private_write(output / "startup.ready.json", record)
                return record
            time.sleep(15)
        raise OpsError("generated_startup_wall_timeout")
    except Exception as error:
        # Classify this before any private quality release. Never retry or
        # reinterpret generated cold-JIT failure as a private model verdict.
        record.update(status="failed", reason_code=str(error) if isinstance(error, OpsError) else "startup_control_error",
                      error_type=type(error).__name__, startup_elapsed_seconds=time.monotonic() - started,
                      classification="generated_startup_failure_not_private_quality_failure")
        try:
            subprocess.run(plan["failure_stop_argv"], capture_output=True, timeout=50)
            if inspect_container(name)["State"]["Running"]:
                subprocess.run(plan["failure_kill_only_if_still_running_argv"], capture_output=True, timeout=15)
            record["container_stopped_verified"] = inspect_container(name)["State"]["Running"] is False
        except Exception:
            record["container_stopped_verified"] = False
        private_write(output / "startup.failed.json", record)
        raise
    finally:
        if attach and attach.poll() is None:
            attach.terminate()  # --sig-proxy=false leaves the model process alone.
            try:
                attach.wait(timeout=5)
            except subprocess.TimeoutExpired:
                attach.kill()
        if reader:
            reader.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("inspect", "verify", "start"), default="inspect")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root-release-id")
    parser.add_argument("--detach", action="store_true",
                        help="POSIX start supervision survives an SSH disconnect")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    need(not args.detach or args.mode == "start", "startup_detach_requires_start_mode")
    if args.mode == "inspect":
        preflight(plan)
        print(json.dumps({"status": "passed", "inspection_only": True, "gpu_access": False}))
        return 0
    need(args.output is not None and not args.output.exists(), "startup_new_private_output_required")
    args.output.mkdir(mode=0o700)
    if args.detach:
        need(hasattr(os, "fork") and hasattr(os, "setsid"), "startup_posix_detach_required")
        need(re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", args.root_release_id or "") is not None,
             "startup_root_release_id_required")
        pid = os.fork()
        if pid:
            print(json.dumps({"status": "dispatched", "supervisor_pid": pid,
                              "output_directory": str(args.output), "model_ready": False}))
            return 0
        os.setsid()
        source = os.open(os.devnull, os.O_RDONLY)
        target = os.open(args.output / "controller.safe-summary.jsonl",
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.dup2(source, 0)
        os.dup2(target, 1)
        os.dup2(target, 2)
        os.close(source)
        os.close(target)
        private_write(args.output / "controller.dispatch.json", {
            "schema_version": "forte.qwen38.detached-startup-controller.v1",
            "supervisor_pid": os.getpid(), "root_release_id": args.root_release_id,
            "plan_sha256": digest(args.plan), "maximum_startup_wall_seconds": 2400,
            "ssh_disconnect_independent": True, "automatic_retry": False})
    if args.mode == "verify":
        preflight(plan)
        result = subprocess.run(plan["cpu_full_artifact_verify_argv"], capture_output=True, text=True, timeout=1200)
        # Only static metadata from verifier JSON is retained; no raw stderr.
        value = json.loads(result.stdout.splitlines()[-1]) if result.stdout.strip() else {}
        proof = {"status": "passed" if result.returncode == 0 and value.get("verified") is True else "failed",
                 "gpu_access": False, "model_loaded": False, "config_file_sha256": plan["config_file_sha256"],
                 "source_archive_sha256": plan["source_archive_sha256"], "runtime_image_sha256": plan["image_sha256"],
                 "files_verified": value.get("files_verified"), "error_type": value.get("error_type")}
        private_write(args.output / "artifact-verification.json", proof)
        print(json.dumps(proof))
        return 0 if proof["status"] == "passed" else 2
    result = supervise(plan, args.output, args.root_release_id or "")
    print(json.dumps({key: result[key] for key in ("status", "container_name", "generated_startup_posts",
                                                  "startup_elapsed_seconds", "private_quality_calls")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "reason_code": str(error) if isinstance(error, OpsError)
                          else "startup_supervisor_error", "error_type": type(error).__name__, "raw_runtime_text_logged": False}))
        raise SystemExit(2)
