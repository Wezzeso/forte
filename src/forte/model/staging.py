"""Explicit build/provisioning-time pinned artifact downloader, never inference.

Only the public Hugging Face repository in a reviewed manifest is contacted.
Streams download directly onto the authorized deployment disk. Every completed
file is checksum-verified before atomic publication. No video path is accepted.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import re
import shutil
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

from .manifest import ManifestError, load_manifest, verify_local_snapshot


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(root: Path, entry: dict, model_id: str, revision: str, deadline: float) -> int:
    target = root / entry["path"]
    expected_size, expected_digest = entry["size_bytes"], entry["sha256"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_size and _digest(target) == expected_digest:
        return 0
    partial = target.with_name(target.name + ".part")
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{quote(entry['path'], safe='/')}"
    opener = build_opener(ProxyHandler({}))
    for attempt in range(3):
        if time.monotonic() >= deadline:
            raise ManifestError("model_staging_deadline_exceeded")
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset == expected_size:
                if _digest(partial) == expected_digest:
                    os.replace(partial, target)
                    return 0
                partial.unlink()
                offset = 0
            elif offset > expected_size:
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "ForteLocalArtifactStager/1", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with opener.open(Request(url, headers=headers), timeout=min(60, max(1, deadline-time.monotonic()))) as response:
                status = response.getcode()
                if status == 206 and offset:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{expected_size}"):
                        raise ManifestError("model_staging_range_invalid")
                    mode = "ab"
                elif status == 200:
                    offset, mode = 0, "wb"
                else:
                    raise ManifestError("model_staging_http_failure")
                downloaded = offset
                with partial.open(mode) as output:
                    while block := response.read(8 * 1024 * 1024):
                        if time.monotonic() >= deadline:
                            raise ManifestError("model_staging_deadline_exceeded")
                        downloaded += len(block)
                        if downloaded > expected_size:
                            raise ManifestError("model_staging_size_exceeded")
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            if downloaded != expected_size or _digest(partial) != expected_digest:
                partial.unlink(missing_ok=True)
                raise ManifestError("model_staging_checksum_or_size_mismatch")
            os.replace(partial, target)
            return downloaded - offset
        except (URLError, OSError, TimeoutError):
            if attempt == 2:
                raise ManifestError("model_staging_network_or_disk_failure") from None
            if deadline - time.monotonic() <= attempt + 1:
                raise ManifestError("model_staging_deadline_exceeded") from None
            time.sleep(attempt + 1)
    raise ManifestError("model_staging_attempts_exhausted")


def stage_local_snapshot(snapshot: str | Path, manifest_path: str | Path = "models/manifest.json",
                         *, max_workers: int = 4, timeout_sec: float = 1800) -> dict:
    manifest = load_manifest(manifest_path)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", manifest.get("model_id", "")):
        raise ManifestError("model_staging_repository_invalid")
    if not 1 <= max_workers <= 4 or not 0 < timeout_sec <= 7200:
        raise ManifestError("model_staging_budget_invalid")
    root = Path(snapshot).resolve()
    root.mkdir(parents=True, exist_ok=True)
    total = sum(entry["size_bytes"] for entry in manifest["files"])
    pending = []
    for entry in manifest["files"]:
        target = root / entry["path"]
        if not (target.is_file() and target.stat().st_size == entry["size_bytes"]
                and _digest(target) == entry["sha256"]):
            pending.append(entry)
    remaining = sum(entry["size_bytes"] for entry in pending)
    if shutil.disk_usage(root).free < remaining + 2 * 1024**3:
        raise ManifestError("model_staging_insufficient_disk")
    started = time.monotonic()
    deadline = started + timeout_sec
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download, root, entry, manifest["model_id"], manifest["revision"], deadline)
                   for entry in pending]
        try:
            downloaded = sum(future.result() for future in futures)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    verification = verify_local_snapshot(root, manifest_path)
    return {"verified": True, "model_id": verification.model_id, "revision": verification.revision,
            "processor_revision": verification.processor_revision, "files_verified": verification.files_verified,
            "bytes_verified": verification.bytes_verified, "bytes_downloaded": downloaded,
            "artifact_bytes": total, "elapsed_sec": time.monotonic() - started,
            "local_inference_verified": False}
