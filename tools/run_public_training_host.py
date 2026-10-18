"""Linux-only isolated public-training launcher; never operates production containers.

Run on the authorized host with Python's standard library. The source manifest is
an external JSON file of the form {"files": {"training/__init__.py": "sha256"}}.
Its exact bytes, the config bytes and every file in the dedicated source closure
are pinned. All config paths must be absolute host paths; bind mounts retain them.
Only the output's existing parent and exposure registry are writable in the job.
Containers and private logs are retained for audit. No sealed evaluation is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

HARD_STOP = datetime(2026, 10, 18, 5, 45, tzinfo=timezone.utc)
LOCK_PATH = Path("/data/claude/training/public-20261018/gpu-training.lock")
WRITABLE_ROOT = LOCK_PATH.parent
SERVICES = ("forte-api-1", "forte-worker-1", "forte-db-1", "forte-ingress-1", "vlm30")
PHASES = {"rf-fit": ("training.specialists.public_train", ["--defer-test"]),
          "crop-smoke": ("training.public_crop_lora", ["--memory-smoke"]),
          "crop-fit": ("training.public_crop_lora", [])}
SHA = re.compile(r"[a-f0-9]{64}\Z")
NAME = re.compile(r"forte-public-[a-z0-9][a-z0-9_.-]{0,100}\Z")
SAFE_STATE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
SOURCE_SUFFIXES = {".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml"}
FORBIDDEN_PARTS = {"private", "secrets", ".git", ".env", "__pycache__"}
MIN_START_FREE_MIB = 6656
MIN_FREE_MIB = 1536
MAX_JOB_MIB = 5632


class GuardError(RuntimeError):
    """Safe finite error codes only; command output belongs in private logs."""


def require(condition, code):
    if not condition:
        raise GuardError(code)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc(value: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0,
                "deadline_must_be_utc")
        return stamp
    except (ValueError, TypeError, AttributeError):
        raise GuardError("deadline_invalid") from None


def absolute_path(value, *, kind=None) -> Path:
    require(isinstance(value, str) and value.startswith("/") and
            not any(c in value for c in (",", "\n", "\r", "\x00")), "absolute_local_path_required")
    path = Path(value)
    require(path.is_absolute() and str(path) == str(path.resolve()) and
            not path.is_symlink(), "canonical_nonsymlink_path_required")
    if kind == "file":
        require(path.is_file(), "required_file_missing")
    if kind == "dir":
        require(path.is_dir(), "required_directory_missing")
    return path


def pinned_json(path: Path, expected: str) -> dict:
    require(isinstance(expected, str) and bool(SHA.fullmatch(expected)), "sha256_pin_required")
    require(path.is_file() and sha256(path) == expected, "input_hash_mismatch")
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        raise GuardError("json_input_invalid") from None
    require(isinstance(data, dict), "json_object_required")
    return data


def verify_sources(root: Path, manifest: Path, expected: str) -> dict:
    require(root.is_dir() and not root.is_symlink(), "source_directory_missing")
    require(not manifest.is_relative_to(root), "source_manifest_must_be_external")
    data = pinned_json(manifest, expected)
    files = data.get("files")
    require(isinstance(files, dict) and bool(files), "source_manifest_files_required")
    found = set()
    for item in root.rglob("*"):
        require(not item.is_symlink(), "source_symlink_forbidden")
        if item.is_file():
            found.add(item.relative_to(root).as_posix())
    require(found == set(files), "source_closure_mismatch")
    for name, expected_file in files.items():
        require(isinstance(name, str), "source_relative_path_invalid")
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and name == rel.as_posix() and
                not any(x in {"", ".", ".."} for x in name.split("/")) and
                "\\" not in name and not any(x.lower() in FORBIDDEN_PARTS for x in rel.parts) and
                rel.suffix in SOURCE_SUFFIXES, "source_relative_path_invalid")
        require(isinstance(expected_file, str) and SHA.fullmatch(expected_file), "source_hash_invalid")
        require(sha256(root / name) == expected_file, "source_file_hash_mismatch")
    require("training/__init__.py" in files, "source_training_package_missing")
    return {"file_count": len(files), "manifest_sha256": expected}


def validate_inputs(args, *, now=None) -> dict:
    require(sys.platform == "linux", "linux_host_required")
    require(re.fullmatch(r"sha256:[a-f0-9]{64}", args.image) is not None, "immutable_image_required")
    require(NAME.fullmatch(args.name) is not None, "isolated_container_name_required")
    require(1 <= args.max_wall_seconds <= 7800, "wall_limit_invalid")
    stop = utc(args.stop_at_utc)
    require(stop <= HARD_STOP, "deadline_exceeds_authorization")
    require(stop > (now or datetime.now(timezone.utc)), "absolute_deadline_elapsed")
    config_path = absolute_path(str(args.config), kind="file")
    source = absolute_path(str(args.source_dir), kind="dir")
    manifest = absolute_path(str(args.source_manifest), kind="file")
    report_path = absolute_path(str(args.report))
    require(not report_path.exists() and report_path.parent.is_dir(), "fresh_report_path_required")
    source_receipt = verify_sources(source, manifest, args.expected_source_manifest_sha256)
    config = pinned_json(config_path, args.expected_config_sha256)
    require(utc(config.get("stop_at_utc")) <= stop, "config_deadline_exceeds_launcher")
    budget = config.get("gpu_memory_budget_mib")
    require(type(budget) is int and 1024 <= budget <= 5120, "config_allocator_budget_exceeds_authorization")
    require(config.get("device", "cuda:0") == "cuda:0", "only_device_zero_authorized")
    if args.phase == "rf-fit":
        require(config.get("defer_sealed_test") is True, "sealed_test_must_be_deferred")
        require(config.get("model_class") in {"RFDETRSmall", "RFDETRNano"}, "detector_class_invalid")
    if args.phase in {"rf-fit", "crop-fit"}:
        require(type(config.get("max_fit_seconds")) in (int, float) and
                0 < config["max_fit_seconds"] <= 7200, "fit_time_limit_invalid")
    dataset = absolute_path(config.get("dataset_dir"), kind="dir")
    output = absolute_path(config.get("output_dir"))
    registry = absolute_path(config.get("test_exposure_registry_dir"), kind="dir")
    require(not output.exists() and output.parent.is_dir(), "fresh_output_directory_required")
    require(all(path != WRITABLE_ROOT and path.is_relative_to(WRITABLE_ROOT)
                for path in (output.parent, registry)), "writable_mount_outside_authorized_root")
    readonly = [source, config_path, dataset, manifest]
    pins = {str(config_path): args.expected_config_sha256,
            str(manifest): args.expected_source_manifest_sha256}
    for key, expected_key in (("pretrain_weights", "expected_pretrain_weights_sha256"),
                              ("model_manifest", "expected_model_manifest_sha256"),
                              ("memory_smoke_run", "expected_memory_smoke_run_sha256")):
        if config.get(key):
            path = absolute_path(config[key], kind="file")
            expected = config.get(expected_key)
            require(isinstance(expected, str) and SHA.fullmatch(expected) and sha256(path) == expected,
                    "artifact_pin_mismatch")
            readonly.append(path)
            pins[str(path)] = expected
    if config.get("model_dir"):
        readonly.append(absolute_path(config["model_dir"], kind="dir"))
    require((args.phase == "rf-fit" and bool(config.get("pretrain_weights"))) or
            (args.phase != "rf-fit" and bool(config.get("model_dir")) and bool(config.get("model_manifest"))),
            "phase_model_inputs_missing")
    require(not any(output.is_relative_to(x) for x in readonly if x.is_dir()),
            "output_inside_readonly_input")
    require(not any(read.is_relative_to(write) or (read.is_dir() and write.is_relative_to(read))
                    for write in (output.parent, registry) for read in readonly),
            "writable_readonly_input_overlap")
    require(not report_path.is_relative_to(source), "report_inside_source_forbidden")
    return {"config": config, "config_path": config_path, "source": source,
            "manifest": manifest, "report": report_path, "output": output, "registry": registry,
            "readonly": sorted(set(readonly), key=lambda p: (len(p.parts), str(p))),
            "pins": pins, "source_receipt": source_receipt, "stop": stop}


def command(args, *, timeout=30):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise GuardError("host_command_unavailable_or_timed_out") from None
    require(result.returncode == 0, "host_command_failed")
    return result.stdout.strip()


def inspect_container(name):
    result = subprocess.run(["docker", "container", "inspect", name], capture_output=True,
                            text=True, timeout=15)
    if result.returncode:
        # Absence must be distinguished from a broken Docker daemon.
        command(["docker", "info", "--format", "{{.ServerVersion}}"])
        require("No such" in result.stderr, "container_inspection_failed")
        return None
    try:
        return json.loads(result.stdout)[0]
    except (ValueError, KeyError, IndexError):
        raise GuardError("container_inspection_invalid") from None


def production_bindings():
    result = {}
    for name in SERVICES:
        data = inspect_container(name)
        require(data is not None and data["State"]["Running"], "production_service_not_running")
        result[name] = {"container_id": data["Id"], "image_id": data["Image"],
                        "started_at": data["State"]["StartedAt"]}
    return result


def readiness():
    code = ("import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen("
            "'http://127.0.0.1:8000/readyz',timeout=5))))")
    data = json.loads(command(["docker", "exec", "forte-api-1", "python", "-c", code]))
    components = data.get("components")
    require(data.get("status") == "ready" and isinstance(components, dict) and bool(components)
            and all(type(value) is bool and value for value in components.values()), "production_not_ready")
    return {"status": "ready", "all_components_ready": True}


def queue_state():
    sql = ("SELECT count(*) FILTER (WHERE status='processing' AND lease_owner IS NOT NULL),"
           "count(*) FILTER (WHERE status='processing' AND lease_owner IS NULL) FROM jobs;")
    raw = command(["docker", "exec", "forte-db-1", "psql", "-U", "forte", "-d", "forte", "-At", "-c", sql])
    try:
        active, queued = map(int, raw.split("|"))
    except ValueError:
        raise GuardError("queue_probe_invalid") from None
    return {"active": active, "queued": queued}


def gpu_state():
    raw = command(["nvidia-smi", "--id=0", "--query-gpu=memory.total,memory.used,memory.free",
                   "--format=csv,noheader,nounits"], timeout=5)
    try:
        total, used, free = map(int, raw.split(","))
        apps = command(["nvidia-smi", "--id=0", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"], timeout=5)
        processes = {int(row.split(",")[0]): int(row.split(",")[1])
                     for row in apps.splitlines() if row.strip()}
    except ValueError:
        raise GuardError("gpu_probe_invalid") from None
    return {"total_mib": total, "used_mib": used, "free_mib": free}, processes


def container_gpu_mib(name, processes):
    raw = command(["docker", "top", name, "-eo", "pid"], timeout=5)
    try:
        pids = {int(row.strip()) for row in raw.splitlines()[1:] if row.strip()}
    except ValueError:
        raise GuardError("container_pid_probe_invalid") from None
    return sum(processes.get(pid, 0) for pid in pids)


def check_sample(sample, job_mib):
    require(sample["free_mib"] >= MIN_FREE_MIB, "global_free_memory_guard")
    require(job_mib <= MAX_JOB_MIB, "actual_training_memory_guard")


def container_args(args, checked, *, preflight):
    module, phase_flags = PHASES[args.phase]
    name = args.name + "-preflight" if preflight else args.name
    result = ["docker", "create", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}",
              "--network", "none", "--read-only", "--cpus", "4", "--memory", "24g",
              "--shm-size", "1g", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
              "--tmpfs", "/tmp:rw,size=2147483648,mode=1777", "--workdir", str(checked["source"])]
    if args.phase in {"crop-smoke", "crop-fit"}:
        result += ["--tmpfs", "/triton-cache:rw,nosuid,nodev,exec,size=536870912,mode=1777"]
    if not preflight:
        result += ["--gpus", "device=0"]
    env = {"NVIDIA_VISIBLE_DEVICES": "void" if preflight else "0",
           "CUDA_VISIBLE_DEVICES": "" if preflight else "0", "HOME": "/tmp",
           "HF_HOME": "/tmp/hf", "RF_HOME": "/tmp/rf", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled",
           "DO_NOT_TRACK": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(checked["source"]), "OMP_NUM_THREADS": "2",
           "PYTORCH_ALLOC_CONF": "backend:native"}
    if args.phase in {"crop-smoke", "crop-fit"}:
        env["TRITON_CACHE_DIR"] = "/triton-cache"
    for key, value in env.items():
        result += ["--env", f"{key}={value}"]
    # Preflight receives no writable host paths at all.
    if not preflight:
        for path in (checked["output"].parent, checked["registry"]):
            result += ["--mount", f"type=bind,src={path},dst={path}"]
    else:
        registry = checked["registry"]
        result += ["--mount", f"type=bind,src={registry},dst={registry},readonly"]
    for path in checked["readonly"]:
        result += ["--mount", f"type=bind,src={path},dst={path},readonly"]
    argv = [module, "--config", str(checked["config_path"])] + (["--preflight"] if preflight else phase_flags)
    if preflight:
        bootstrap = ("import os,runpy,sys,torch; "
                     "assert not any(x.startswith('nvidia') for x in os.listdir('/dev')); "
                     "assert not torch.cuda.is_initialized(); "
                     f"sys.argv={argv!r}; "
                     "\ntry:\n runpy.run_module(sys.argv[0],run_name='__main__')"
                     "\nfinally:\n assert not torch.cuda.is_initialized()\n")
        command_args = ["-c", bootstrap]
    else:
        command_args = ["-m"] + argv
    return result + ["--entrypoint", "python3", args.image] + command_args


def stop_owned(name, expected_id):
    """Only kill a container whose exact ID was returned by our own create."""
    data = inspect_container(name)
    require(data is not None and data["Id"] == expected_id, "owned_container_identity_changed")
    if data["State"]["Running"]:
        command(["docker", "kill", expected_id], timeout=20)
    data = inspect_container(name)
    require(data is not None and data["Id"] == expected_id and not data["State"]["Running"],
            "owned_container_did_not_stop")
    return data


def compact_progress(output: Path, elapsed: float):
    summary = {}
    for name in ("run.json", "progress.json"):
        path = output / name
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            continue
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        if isinstance(data.get("state"), str) and SAFE_STATE.fullmatch(data["state"]):
            summary["state"] = data["state"]
        for key in ("epoch", "optimizer_steps", "physical_batches"):
            if type(data.get(key)) is int and data[key] >= 0:
                summary[key] = data[key]
        losses = data.get("recent_losses")
        loss = losses[-1] if isinstance(losses, list) and losses else data.get("recent_loss")
        if type(loss) in (int, float) and math.isfinite(loss):
            summary["recent_loss"] = loss
    return summary


def emit(kind, value):
    print("PUBLIC_TRAINING_" + kind + " " + json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def private_file(path):
    return os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w")


def write_report(path, report):
    temporary = path.with_name(path.name + ".tmp")
    with private_file(temporary) as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def wait_container(name, container_id, log, *, deadline, wall_started, wall_seconds,
                   output=None, monitor_gpu=False, receipt=None):
    require(time.monotonic() - wall_started < wall_seconds, "hard_wall_time_exceeded")
    require(datetime.now(timezone.utc) < deadline, "absolute_deadline_reached")
    command(["docker", "start", container_id])
    follower = subprocess.Popen(["docker", "logs", "--follow", container_id], stdout=log,
                                stderr=subprocess.STDOUT, text=True)
    last_progress = None
    try:
        while True:
            elapsed = time.monotonic() - wall_started
            require(elapsed < wall_seconds, "hard_wall_time_exceeded")
            require(datetime.now(timezone.utc) < deadline, "absolute_deadline_reached")
            data = inspect_container(name)
            require(data is not None and data["Id"] == container_id, "owned_container_identity_changed")
            if not data["State"]["Running"]:
                return data
            if monitor_gpu:
                sample, processes = gpu_state()
                # Process can exit between inspect and top; recheck rather than masking a guard error.
                try:
                    own_mib = container_gpu_mib(name, processes)
                except GuardError:
                    latest = inspect_container(name)
                    if latest and latest["Id"] == container_id and not latest["State"]["Running"]:
                        return latest
                    raise
                receipt["minimum_sampled_global_free_mib"] = min(receipt["minimum_sampled_global_free_mib"], sample["free_mib"])
                receipt["maximum_sampled_training_mib"] = max(receipt["maximum_sampled_training_mib"], own_mib)
                receipt["gpu_samples"] += 1
                check_sample(sample, own_mib)
            if output:
                progress = compact_progress(output, elapsed)
                if progress and progress != last_progress:
                    emit("PROGRESS", dict(progress, elapsed_seconds=round(elapsed, 3)))
                    last_progress = progress
            time.sleep(2)
    finally:
        follower.terminate()
        try:
            follower.wait(timeout=10)
        except subprocess.TimeoutExpired:
            follower.kill()
            follower.wait(timeout=10)


def execute(args):
    import fcntl
    checked = validate_inputs(args)
    started = time.monotonic()
    report = {"schema_version": "forte.public-training-host.v1", "phase": args.phase,
              "passed": False, "failure": None, "image_id": args.image,
              "config_sha256": args.expected_config_sha256, "source": checked["source_receipt"],
              "network": "none", "sealed_test_deferred": True, "promotion_eligible": False,
              "limits": {"wall_seconds": args.max_wall_seconds, "stop_at_utc": args.stop_at_utc,
                         "start_free_mib": MIN_START_FREE_MIB, "running_free_mib": MIN_FREE_MIB,
                         "actual_training_max_mib": MAX_JOB_MIB},
              "gpu_samples": 0, "maximum_sampled_training_mib": 0,
              "minimum_sampled_global_free_mib": 2**31 - 1}
    owned = []
    before = None
    lock = None
    interrupted = {"signal": None}
    previous_handlers = {}

    def interrupt(signum, _frame):
        interrupted["signal"] = signum
        if signum == signal.SIGALRM:
            raise GuardError("absolute_deadline_reached" if datetime.now(timezone.utc) >= checked["stop"]
                             else "hard_wall_time_exceeded")
        raise GuardError("host_signal_interruption")

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, interrupt)
    # A wall-clock alarm interrupts even a stuck nvidia-smi or Docker probe.
    previous_handlers[signal.SIGALRM] = signal.signal(signal.SIGALRM, interrupt)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, min(args.max_wall_seconds,
                     (checked["stop"] - datetime.now(timezone.utc)).total_seconds())))
    try:
        before = production_bindings()
        report["production_before"] = before
        report["readiness_before"] = readiness()
        require(inspect_container(args.name) is None and inspect_container(args.name + "-preflight") is None,
                "isolated_container_name_already_exists")
        image = json.loads(command(["docker", "image", "inspect", args.image]))[0]
        require(image["Id"] == args.image, "immutable_image_mismatch")
        emit("START", {"phase": args.phase, "image_id": args.image, "state": "cpu_preflight"})
        preflight_id = command(container_args(args, checked, preflight=True))
        require(SHA.fullmatch(preflight_id) is not None, "created_container_id_invalid")
        owned.append((args.name + "-preflight", preflight_id))
        log_path = checked["report"].with_name(checked["report"].stem + ".preflight.private.log")
        with private_file(log_path) as log:
            preflight = wait_container(args.name + "-preflight", preflight_id, log,
                                       deadline=checked["stop"], wall_started=started,
                                       wall_seconds=min(args.max_wall_seconds, 900))
        report["cpu_preflight_exit_code"] = preflight["State"]["ExitCode"]
        require(preflight["State"]["ExitCode"] == 0, "cpu_preflight_failed")
        verify_sources(checked["source"], checked["manifest"], args.expected_source_manifest_sha256)
        for path, expected in checked["pins"].items():
            require(sha256(Path(path)) == expected, "input_changed_during_preflight")
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock = LOCK_PATH.open("a")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GuardError("training_gpu_lock_busy") from None
        report["gpu_lock_held"] = True
        require(before == production_bindings(), "production_changed_during_cpu_preflight")
        readiness()
        report["queue_before"] = queue_state()
        require(report["queue_before"] == {"active": 0, "queued": 0}, "live_queue_not_idle")
        gpu_before, _ = gpu_state()
        report["gpu_before"] = gpu_before
        require(gpu_before["free_mib"] >= MIN_START_FREE_MIB, "insufficient_spare_gpu_memory")
        require(not checked["output"].exists(), "output_created_during_preflight")
        require(time.monotonic() - started < args.max_wall_seconds, "hard_wall_time_exceeded")
        require(datetime.now(timezone.utc) < checked["stop"], "absolute_deadline_reached")
        container_id = command(container_args(args, checked, preflight=False))
        require(SHA.fullmatch(container_id) is not None, "created_container_id_invalid")
        owned.append((args.name, container_id))
        report["container_id"] = container_id
        emit("START", {"phase": args.phase, "state": "isolated_gpu_start",
                        "gpu_free_before_mib": gpu_before["free_mib"], "gpu_lock_held": True})
        log_path = checked["report"].with_name(checked["report"].stem + ".training.private.log")
        with private_file(log_path) as log:
            trained = wait_container(args.name, container_id, log, deadline=checked["stop"],
                                     wall_started=started, wall_seconds=args.max_wall_seconds,
                                     output=checked["output"], monitor_gpu=True, receipt=report)
        report["training_exit_code"] = trained["State"]["ExitCode"]
        require(trained["State"]["ExitCode"] == 0, "training_container_failed")
        report["final_progress"] = compact_progress(checked["output"], time.monotonic() - started)
        expected_state = {"rf-fit": "trained_unselected_pending_test", "crop-fit": "trained_unselected",
                          "crop-smoke": "memory_smoke_passed"}[args.phase]
        require(report["final_progress"].get("state") == expected_state and
                report["final_progress"].get("optimizer_steps", 0) > 0,
                "training_completion_receipt_invalid")
        report["passed"] = True
    except GuardError as exc:
        report["failure"] = str(exc)
    except BaseException:
        report["failure"] = "unexpected_host_launcher_failure"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_errors = []
        # Defer further termination signals until our own containers have been stopped.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        for name, container_id in reversed(owned):
            try:
                stop_owned(name, container_id)
            except BaseException:
                cleanup_errors.append("owned_container_cleanup_failed")
        report["owned_containers_retained_stopped"] = bool(owned) and not cleanup_errors
        try:
            verify_sources(checked["source"], checked["manifest"], args.expected_source_manifest_sha256)
            for path, expected in checked["pins"].items():
                require(sha256(Path(path)) == expected, "immutable_input_changed")
            report["immutable_inputs_unchanged"] = True
        except BaseException:
            cleanup_errors.append("immutable_inputs_changed")
        if before is not None:
            try:
                after = production_bindings()
                report["production_after"] = after
                report["production_unchanged"] = before == after
                require(before == after, "production_binding_changed")
                report["readiness_after"] = readiness()
                report["queue_after"] = queue_state()
                report["gpu_after"], _ = gpu_state()
            except BaseException:
                cleanup_errors.append("production_postcheck_failed")
        if cleanup_errors:
            report["postcheck_failures"] = cleanup_errors
            report["passed"] = False
        if lock:
            lock.close()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if not report["gpu_samples"]:
            report["minimum_sampled_global_free_mib"] = None
        write_report(checked["report"], report)
        emit("FINAL", report)
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    return 0 if report["passed"] else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=PHASES, default="rf-fit")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--expected-config-sha256", required=True)
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--expected-source-manifest-sha256", required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--name", required=True)
    result.add_argument("--max-wall-seconds", type=int, default=7800)
    result.add_argument("--stop-at-utc", default="2026-10-18T05:45:00Z")
    return result


def main():
    args = parser().parse_args()
    try:
        return execute(args)
    except GuardError as exc:
        emit("FINAL", {"passed": False, "failure": str(exc), "gpu_container_started": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
