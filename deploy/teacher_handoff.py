"""Prepare private terminal teacher handoffs through a read-only CPU exporter.

Run on the team host after the bounded teacher process exits. No GPU call,
review decision, source mutation, account operation or external network is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def run(arguments: list[str]) -> str:
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("artifact_exporter_command_failed")
    return result.stdout.strip()


def prepare(root: Path, source_root: Path, image: str, ssh_target: str, name: str,
            code_sha256: str, metadata_sha256: str) -> dict:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("artifact_exporter_identity_invalid")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", ssh_target):
        raise ValueError("artifact_ssh_alias_invalid")
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (code_sha256, metadata_sha256)):
        raise ValueError("artifact_producer_archive_pin_invalid")
    if root.is_symlink() or not root.is_absolute() or str(root) == "/":
        raise ValueError("artifact_output_root_invalid")
    phase2 = json.loads((root / "phase2.status.json").read_text())
    smoke = json.loads((root / "smoke.status.json").read_text())
    if phase2.get("returncode") != 0 or smoke.get("returncode") != 0:
        raise RuntimeError("successful_terminal_teacher_receipt_required")
    if phase2["aggregate"].get("model_requests") + smoke["aggregate"].get("model_requests") != 32:
        raise RuntimeError("teacher_bounded_call_receipt_mismatch")
    if run(["docker", "ps", "-q", "--filter", "label=com.docker.compose.project=forte-teacher"]):
        raise RuntimeError("teacher_process_still_running")
    audits = sorted((root / "train").glob("*/teacher.private.json"))
    if not audits:
        raise RuntimeError("teacher_private_audits_missing")
    existing = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
    if existing.returncode:
        run(["docker", "run", "-d", "--name", name, "--network", "none", "--read-only",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--user", "1000:1000",
             "--cpus", "1", "--memory", "256m", "--log-driver", "none",
             "--label", "forte.role=private-artifact-exporter",
             "--mount", f"type=bind,source={root},target=/teacher/w2,readonly",
             "--mount", f"type=bind,source={source_root},target=/private/sources,readonly",
             "--entrypoint", "python", image, "-c", "import time;time.sleep(86400)"])
    observed = json.loads(run(["docker", "inspect", name]))[0]
    if observed["Image"] != image or not observed["HostConfig"]["ReadonlyRootfs"] or observed["HostConfig"]["NetworkMode"] != "none":
        raise RuntimeError("artifact_exporter_boundary_mismatch")
    if any(m.get("RW") for m in observed["Mounts"]):
        raise RuntimeError("artifact_exporter_writable_mount_forbidden")
    handoff_root = root.parent / "handoffs"
    handoff_root.mkdir(mode=0o700, exist_ok=True)
    entries = []
    for audit in audits:
        raw = audit.read_bytes();payload = json.loads(raw);job_id = payload["job_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id) or audit.parent.name != job_id:
            raise RuntimeError("teacher_job_binding_invalid")
        relative = audit.relative_to(root).as_posix()
        entry = {"schema_version": "forte.proposal-artifact-handoff.v1", "terminal_verified": True,
                 "artifact_available": True, "terminal_status": "done", "job_id": job_id,
                 "ssh_target": ssh_target, "ssh_wsl_distribution": "Ubuntu", "remote_container": name,
                 "remote_python": "python", "remote_docker_sudo": False,
                 "remote_work_root": "/teacher/w2", "remote_audit_path": "/teacher/w2/" + relative,
                 "remote_frame_root": "/teacher/w2/" + audit.parent.relative_to(root).as_posix() + "/frames",
                 "audit_sha256": hashlib.sha256(raw).hexdigest(),
                 "audit_phase_unchanged": payload.get("phase"),
                 "producer_cpu_image": image, "exporter_cpu_image": observed["Image"],
                 "producer_public_code_archive_sha256": code_sha256,
                 "producer_private_metadata_archive_sha256": metadata_sha256,
                 "human_reviewed": False, "training_authorized": False}
        target = handoff_root / (job_id + ".json")
        encoded = json.dumps(entry, sort_keys=True, indent=2) + "\n"
        if target.exists():
            if target.read_text() != encoded:
                raise RuntimeError("immutable_teacher_handoff_already_differs")
        else:
            with target.open("x") as output:output.write(encoded)
            target.chmod(0o600)
        entries.append(entry)
    index = handoff_root / "index.private.json"
    encoded = json.dumps({"schema_version": "forte.proposal-artifact-handoff-index.v1", "entries": entries}, sort_keys=True, indent=2) + "\n"
    if index.exists():
        if index.read_text() != encoded:
            raise RuntimeError("immutable_teacher_handoff_index_already_differs")
    else:
        with index.open("x") as output:output.write(encoded)
        index.chmod(0o600)
    return {"handoff_ready": True, "audit_count": len(entries), "terminal_verified": True,
            "exporter_cpu_image": observed["Image"], "exporter_network": "none",
            "exporter_root_and_mounts_read_only": True, "private_content_printed": False, "gpu_calls": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/forte-teacher/w2"))
    parser.add_argument("--source-root", type=Path, default=Path("/data/forte-training-sources"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--container", default="forte-w2-artifact-exporter")
    parser.add_argument("--code-archive-sha256", required=True)
    parser.add_argument("--metadata-archive-sha256", required=True)
    args = parser.parse_args()
    try:result = prepare(args.root, args.source_root, args.image, args.ssh_target, args.container,
                        args.code_archive_sha256, args.metadata_archive_sha256)
    except Exception as exc:
        print(json.dumps({"handoff_ready": False, "reason_code": "teacher_handoff_blocked", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True));return 0


if __name__ == "__main__":raise SystemExit(main())
