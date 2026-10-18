"""Hash and safely stage a private flat TAR of SHA256-named original MOVs.

Run on the team host only. No media pixels, paths, source filenames, labels or
exception strings are emitted. Existing matching files are verified and kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import uuid

NAME = re.compile(r"([0-9a-f]{64})\.mov\Z")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def stage(archive_path: Path, target: Path, expected_sha256: str,
          max_total_bytes: int = 16 * 1024**3, max_files: int = 1000) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or max_total_bytes <= 0 or max_files <= 0:
        raise ValueError("source_stage_configuration_invalid")
    if archive_path.is_symlink() or not archive_path.is_file() or digest(archive_path) != expected_sha256:
        raise ValueError("source_archive_digest_mismatch")
    if not target.is_absolute() or target.is_symlink():
        raise ValueError("source_target_invalid")
    target.mkdir(mode=0o700, parents=False, exist_ok=True)
    resolved = target.resolve(strict=True)
    if not resolved.is_dir() or resolved != target.absolute():
        raise ValueError("source_target_invalid")
    staged = kept = total = 0
    with tarfile.open(archive_path, "r:*") as archive:
        members = []
        names = set()
        for member in archive:
            if len(members) >= max_files:
                raise ValueError("source_archive_member_budget_exceeded")
            if not member.isfile() or not NAME.fullmatch(member.name) or member.name in names or member.size <= 0:
                raise ValueError("source_archive_member_invalid")
            names.add(member.name)
            total += member.size
            if total > max_total_bytes:
                raise ValueError("source_archive_byte_budget_exceeded")
            members.append(member)
        if not members:
            raise ValueError("source_archive_member_budget_exceeded")
        for member in members:
            destination = resolved / member.name
            expected = NAME.fullmatch(member.name).group(1)
            if destination.is_symlink():
                raise ValueError("source_existing_symlink_forbidden")
            if destination.exists():
                if not destination.is_file() or destination.stat().st_size != member.size or digest(destination) != expected:
                    raise ValueError("source_existing_content_mismatch")
                kept += 1
                continue
            temporary = resolved / ("." + expected + ".partial-" + uuid.uuid4().hex)
            try:
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("source_archive_member_unreadable")
                value, count = hashlib.sha256(), 0
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with source, os.fdopen(fd, "wb") as output:
                    while chunk := source.read(1024 * 1024):
                        count += len(chunk)
                        if count > member.size:
                            raise ValueError("source_archive_member_size_mismatch")
                        value.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if count != member.size or value.hexdigest() != expected:
                    raise ValueError("source_member_digest_mismatch")
                # link is atomic and refuses to overwrite an existing target.
                os.link(temporary, destination)
                temporary.unlink()
                os.chmod(destination, 0o400)
                staged += 1
            finally:
                temporary.unlink(missing_ok=True)
    return {"status": "verified", "files": len(members), "bytes": total,
            "new_files": staged, "matching_files_preserved": kept,
            "private_media_opened_for_hash_only": True, "gpu_used": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-total-bytes", type=int, default=16 * 1024**3)
    args = parser.parse_args()
    try:
        result = stage(args.archive, args.target, args.expected_sha256, args.max_total_bytes)
    except Exception as exc:
        result = {"status": "blocked", "reason_code": "private_source_stage_failed",
                  "error_type": type(exc).__name__}
    print(json.dumps(result, sort_keys=True))
    return int(result["status"] != "verified")


if __name__ == "__main__":
    raise SystemExit(main())
