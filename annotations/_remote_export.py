"""Fixed SSH-side exporter. Stdout is a private bounded tar, never diagnostics."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile


def _path(value, root=None, directory=False):
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value or any(part in {".", ".."} for part in value.split("/")):
        raise ValueError("invalid path")
    result = Path(value).resolve(strict=True)
    if str(result) == "/" or (root is not None and not result.is_relative_to(root)):
        raise ValueError("invalid root")
    if directory and not result.is_dir() or not directory and not result.is_file():
        raise ValueError("invalid file")
    return result


def _read(path, limit):
    if path.stat().st_size > limit:
        raise ValueError("size limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("size limit")
    return raw


def _add(archive, name, raw):
    member = tarfile.TarInfo(name)
    member.size, member.mode, member.mtime = len(raw), 0o600, 0
    archive.addfile(member, io.BytesIO(raw))


def export(config, output):
    work_root = _path(config["remote_work_root"], directory=True)
    frame_root = _path(config["remote_frame_root"], work_root, directory=True)
    audit_path = _path(config["remote_audit_path"], work_root)
    raw = _read(audit_path, config["max_audit_bytes"])
    audit = json.loads(raw)
    if audit.get("schema_version") != "forte.local-proposal.v1" or audit.get("job_id") != config["job_id"] or audit.get("asset", {}).get("job_id") != config["job_id"]:
        raise ValueError("audit binding")
    frames = audit.get("frames", audit.get("frame_map"))
    if isinstance(frames, dict):
        if any(frame.get("frame_id") != key for key, frame in frames.items()):
            raise ValueError("frame binding")
        frames = list(frames.values())
    if not isinstance(frames, list) or not 0 < len(frames) <= config["max_frames"]:
        raise ValueError("frame count")
    identities, paths, files, total = set(), set(), [], len(raw)
    with tarfile.open(fileobj=output, mode="w|") as archive:
        _add(archive, "analysis.private.json", raw)
        for frame in frames:
            identity, expected = frame.get("frame_id"), frame.get("jpeg_sha256")
            if not isinstance(identity, str) or not identity or identity in identities or frame.get("job_id") != config["job_id"] or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError("frame binding")
            path = _path(frame["path"], frame_root)
            relative = path.relative_to(frame_root)
            if not relative.parts or str(relative) in paths:
                raise ValueError("frame alias")
            image = _read(path, config["max_frame_bytes"])
            if hashlib.sha256(image).hexdigest() != expected or not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
                raise ValueError("frame checksum")
            total += len(image)
            if total > config["max_total_bytes"]:
                raise ValueError("total size")
            name = "frames/" + relative.as_posix()
            _add(archive, name, image)
            files.append({"path": name, "sha256": expected, "size": len(image)})
            identities.add(identity); paths.add(str(relative))
        transfer = {"schema_version": "forte.proposal-transfer.v1", "job_id": config["job_id"], "audit_sha256": hashlib.sha256(raw).hexdigest(), "frame_count": len(files), "total_payload_bytes": total, "files": files}
        _add(archive, "transfer.json", json.dumps(transfer, allow_nan=False).encode("utf-8"))


def main():
    try:
        raw = sys.stdin.buffer.read(65537)
        if len(raw) > 65536:
            raise ValueError("config size")
        export(json.loads(raw), sys.stdout.buffer)
        return 0
    except Exception:
        sys.stderr.write("REMOTE_PROPOSAL_EXPORT_REJECTED\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
