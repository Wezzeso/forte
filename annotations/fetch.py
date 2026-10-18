"""Terminal-handoff-gated SSH download of private proposal artifacts only.

The CLI returns aggregate counters/static errors. SSH binary output and stderr
never enter agent-visible tool output; originals and downloaded JPEGs remain in
the excluded local workspace. This code never starts or restarts the workbench.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, BinaryIO, Callable

from .proposals import MAX_AUDIT_BYTES, _sha256, import_proposal
from .rebase import _portable_relative, _remote_path
from .store import ReviewError, ReviewStore, canonical

MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_FRAMES = 8192
MAX_TRANSFER_MANIFEST_BYTES = 4 * 1024 * 1024


def _handoff(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != "forte.proposal-artifact-handoff.v1":
        raise ReviewError("artifact_handoff_schema_invalid")
    if value.get("terminal_verified") is not True or value.get("artifact_available") is not True or value.get("terminal_status") not in {"done", "failed"}:
        raise ReviewError("terminal_artifact_handoff_required")
    if not isinstance(value.get("job_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value["job_id"]):
        raise ReviewError("artifact_job_id_invalid")
    if not isinstance(value.get("ssh_target"), str) or value["ssh_target"].startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]{1,256}", value["ssh_target"]):
        raise ReviewError("artifact_ssh_target_invalid")
    work = _remote_path(value.get("remote_work_root"))
    frames = _remote_path(value.get("remote_frame_root"))
    audit = _remote_path(value.get("remote_audit_path"))
    if str(work) == "/" or not frames.is_relative_to(work) or not audit.is_relative_to(work) or audit.name not in {"analysis.private.json", "teacher.private.json"} or value["job_id"] not in audit.relative_to(work).parts:
        raise ReviewError("artifact_remote_roots_invalid")
    port = value.get("ssh_port", 22)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ReviewError("artifact_ssh_port_invalid")
    distribution = value.get("ssh_wsl_distribution")
    if distribution is not None and (not isinstance(distribution, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", distribution)):
        raise ReviewError("artifact_wsl_distribution_invalid")
    for field in ("ssh_identity_path", "ssh_config_path", "ssh_known_hosts_path"):
        if field in value:
            if distribution is not None:
                _remote_path(value[field])
            else:
                path = Path(value[field])
                if not path.is_absolute() or not path.is_file():
                    raise ReviewError("artifact_ssh_reference_unavailable")
    container = value.get("remote_container")
    if container is not None and (not isinstance(container, str) or container.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", container)):
        raise ReviewError("artifact_container_invalid")
    if "remote_docker_sudo" in value and not isinstance(value["remote_docker_sudo"], bool):
        raise ReviewError("artifact_docker_authority_invalid")
    python = value.get("remote_python", "python3")
    if not isinstance(python, str) or not re.fullmatch(r"(?:/[A-Za-z0-9_.-]+)*/?python(?:[0-9.]+)?", python):
        raise ReviewError("artifact_python_reference_invalid")
    if "audit_sha256" in value and (not isinstance(value["audit_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", value["audit_sha256"])):
        raise ReviewError("artifact_audit_hash_invalid")
    return value


def ssh_command(handoff: dict[str, Any]) -> list[str]:
    """Build an argv vector; all remote paths travel in JSON stdin, not a shell."""
    source = Path(__file__).with_name("_remote_export.py").read_text(encoding="utf-8")
    remote = [handoff.get("remote_python", "python3"), "-c", source]
    if handoff.get("remote_container"):
        remote = ["docker", "exec", "-i", handoff["remote_container"], *remote]
        if handoff.get("remote_docker_sudo"):
            remote = ["sudo", "-n", *remote]
    wsl = handoff.get("ssh_wsl_distribution")
    command = [shutil.which("wsl") or "wsl.exe", "--distribution", wsl, "--exec", "ssh"] if wsl else [shutil.which("ssh") or "ssh"]
    command.extend(["-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15"])
    for field, flag in (("ssh_config_path", "-F"), ("ssh_identity_path", "-i")):
        if handoff.get(field):
            command.extend([flag, handoff[field] if wsl else str(Path(handoff[field]).resolve())])
    if handoff.get("ssh_known_hosts_path"):
        command.extend(["-o", "UserKnownHostsFile=" + (handoff["ssh_known_hosts_path"] if wsl else str(Path(handoff["ssh_known_hosts_path"]).resolve()))])
    if "ssh_port" in handoff:
        command.extend(["-p", str(handoff["ssh_port"])])
    command.extend([handoff["ssh_target"], shlex.join(remote)])
    return command


def _ssh_transport(handoff: dict[str, Any], config: dict[str, Any], output: BinaryIO, stderr: BinaryIO) -> None:
    if shutil.which("wsl" if handoff.get("ssh_wsl_distribution") else "ssh") is None:
        raise ReviewError("artifact_ssh_unavailable")
    try:
        result = subprocess.run(ssh_command(handoff), input=canonical(config).encode("utf-8"), stdout=output, stderr=stderr, timeout=180, check=False)
    except subprocess.TimeoutExpired:
        raise ReviewError("artifact_transfer_timeout") from None
    except OSError:
        raise ReviewError("artifact_transport_unavailable") from None
    if result.returncode:
        raise ReviewError("artifact_transfer_failed_inspect_private_stderr")


def _extract(archive_path: Path, staging: Path, *, expected_job_id: str) -> dict[str, Any]:
    if archive_path.stat().st_size > MAX_TOTAL_BYTES + MAX_AUDIT_BYTES + MAX_TRANSFER_MANIFEST_BYTES + MAX_FRAMES * 2048:
        raise ReviewError("artifact_archive_size_limit")
    names, total, count = set(), 0, 0
    try:
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                count += 1
                if count > MAX_FRAMES + 2 or not member.isfile() or member.name in names:
                    raise ReviewError("artifact_archive_member_invalid")
                path = PurePosixPath(member.name)
                if path.is_absolute() or any(part in {"", ".", ".."} for part in member.name.split("/")):
                    raise ReviewError("artifact_archive_path_invalid")
                relative = _portable_relative(PurePosixPath("/transfer") / path, PurePosixPath("/transfer"))
                if member.name == "analysis.private.json":
                    limit = MAX_AUDIT_BYTES
                elif member.name == "transfer.json":
                    limit = MAX_TRANSFER_MANIFEST_BYTES
                elif len(relative) > 1 and relative[0] == "frames":
                    limit = MAX_FRAME_BYTES
                else:
                    raise ReviewError("artifact_archive_member_unexpected")
                if member.size < 0 or member.size > limit:
                    raise ReviewError("artifact_archive_member_size_limit")
                total += member.size
                if total > MAX_TOTAL_BYTES + MAX_TRANSFER_MANIFEST_BYTES:
                    raise ReviewError("artifact_archive_total_size_limit")
                destination = staging.joinpath(*relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.resolve().is_relative_to(staging.resolve()):
                    raise ReviewError("artifact_archive_path_escape")
                source = archive.extractfile(member)
                if source is None:
                    raise ReviewError("artifact_archive_member_unavailable")
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                    output.flush(); os.fsync(output.fileno())
                if destination.stat().st_size != member.size:
                    raise ReviewError("artifact_archive_member_truncated")
                names.add(member.name)
    except ReviewError:
        raise
    except (OSError, ValueError, tarfile.TarError):
        raise ReviewError("artifact_archive_invalid") from None
    if not {"analysis.private.json", "transfer.json"} <= names:
        raise ReviewError("artifact_archive_manifest_missing")
    try:
        manifest = json.loads((staging / "transfer.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != "forte.proposal-transfer.v1" or manifest["job_id"] != expected_job_id or manifest["audit_sha256"] != _sha256(staging / "analysis.private.json"):
            raise ReviewError("artifact_transfer_manifest_binding_invalid")
        files = manifest["files"]
        if not isinstance(files, list) or manifest["frame_count"] != len(files) or not 0 < len(files) <= MAX_FRAMES:
            raise ReviewError("artifact_transfer_frame_count_invalid")
        expected = {"analysis.private.json", "transfer.json"}
        for file in files:
            if not isinstance(file, dict) or not isinstance(file.get("path"), str) or file["path"] in expected:
                raise ReviewError("artifact_transfer_file_binding_invalid")
            path = staging.joinpath(*PurePosixPath(file["path"]).parts)
            if file["path"] not in names or not file["path"].startswith("frames/") or path.stat().st_size != file["size"] or _sha256(path) != file["sha256"]:
                raise ReviewError("artifact_transfer_file_checksum_invalid")
            expected.add(file["path"])
        if expected != names:
            raise ReviewError("artifact_transfer_contains_unreferenced_members")
        if manifest.get("total_payload_bytes") != (staging / "analysis.private.json").stat().st_size + sum(file["size"] for file in files):
            raise ReviewError("artifact_transfer_payload_size_invalid")
    except ReviewError:
        raise
    except (KeyError, TypeError, ValueError, OSError):
        raise ReviewError("artifact_transfer_manifest_invalid") from None
    return manifest


def fetch_proposal(
    store: ReviewStore,
    handoff_path: Path,
    *,
    source_roots: list[Path] | None = None,
    transport: Callable[[dict[str, Any], dict[str, Any], BinaryIO, BinaryIO], None] | None = None,
) -> dict[str, Any]:
    """Fetch only after explicit terminal handoff; return aggregate diagnostics.

    ``transport`` is dependency injection for synthetic tests only. Production
    CLI always uses verified-host SSH with a fixed remote Python exporter.
    """
    try:
        if handoff_path.stat().st_size > 65536:
            raise ReviewError("artifact_handoff_size_limit")
        handoff = _handoff(json.loads(handoff_path.read_text(encoding="utf-8-sig")))
    except ReviewError:
        raise
    except (OSError, TypeError, ValueError):
        raise ReviewError("artifact_handoff_unavailable") from None
    private_root = store.workspace / "proposals"
    private_root.mkdir(exist_ok=True)
    if not private_root.resolve().is_relative_to(store.workspace):
        raise ReviewError("artifact_private_root_escape")
    config = {key: handoff[key] for key in ("job_id", "remote_audit_path", "remote_work_root", "remote_frame_root")}
    config.update(max_audit_bytes=MAX_AUDIT_BYTES, max_frame_bytes=MAX_FRAME_BYTES, max_total_bytes=MAX_TOTAL_BYTES, max_frames=MAX_FRAMES)
    with tempfile.TemporaryDirectory(prefix=".transfer-", dir=private_root) as temporary:
        temporary_root = Path(temporary).resolve()
        if not temporary_root.is_relative_to(private_root.resolve()):
            raise ReviewError("artifact_private_staging_escape")
        archive_path = temporary_root / "download.tar"
        stderr_path = private_root / (handoff["job_id"] + ".transport.stderr.private.txt")
        with archive_path.open("xb") as output, stderr_path.open("wb") as stderr:
            (transport or _ssh_transport)(handoff, config, output, stderr)
            output.flush(); os.fsync(output.fileno())
        staging = temporary_root / "extracted"
        staging.mkdir()
        manifest = _extract(archive_path, staging, expected_job_id=handoff["job_id"])
        if handoff.get("audit_sha256") not in {None, manifest["audit_sha256"]}:
            raise ReviewError("artifact_handoff_audit_hash_mismatch")
        final = private_root / (handoff["job_id"] + "-" + manifest["audit_sha256"][:24])
        if not final.resolve().is_relative_to(private_root.resolve()):
            raise ReviewError("artifact_private_destination_escape")
        if final.exists():
            if not final.is_dir() or _sha256(final / "analysis.private.json") != manifest["audit_sha256"]:
                raise ReviewError("artifact_existing_private_destination_conflict")
        else:
            staging.rename(final)
        if not (final / "frames").resolve().is_relative_to(final.resolve()):
            raise ReviewError("artifact_private_frame_root_escape")
        source_map = {row["content_hash"]: row["media_paths"][0] for row in store.list_records() if row["source_role"] != "jury_test"}
        approved_sources = source_roots or list({Path(value).parent for value in source_map.values()})
        try:
            imported = import_proposal(store, final / "analysis.private.json", proposal_roots=[private_root], source_roots=approved_sources, frame_roots=[final / "frames"], local_source_by_hash=source_map, remote_frame_root=handoff["remote_frame_root"], local_frame_root=final / "frames")
        except ReviewError:
            raise ReviewError("artifact_verified_private_download_retained_import_requires_local_action") from None
    return {"transfer_status": "verified", "downloaded_frames": manifest["frame_count"], "downloaded_payload_bytes": manifest["total_payload_bytes"], **imported}
