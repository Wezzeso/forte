"""Explicit stdlib-only staging of separately reviewed public image encoders.

This module never imports ML libraries, reads videos, instantiates a model, or
changes the canonical Qwen snapshot. Successful hashing establishes artifact
identity only. These assets remain candidates until separately evaluated.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .manifest import ManifestError
from .staging import _digest, _download


APPROVED_ENCODERS = {
    "google/siglip2-base-patch16-256": "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab",
    "google/siglip2-base-patch32-256": "94dffa8cb1179de3e03f091dbc3917e5d5a9ae84",
}
_REQUIRED_FILES = {"model.safetensors", "config.json", "preprocessor_config.json", "README.md"}


class EncoderManifestError(ManifestError):
    """Sanitized candidate artifact failure, without raw paths or content."""


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def _reject_constant(_: str) -> None:
    raise ValueError("nonfinite JSON")


def load_encoder_manifest(path: str | Path, *,
                          expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Validate an image-only manifest without weakening Qwen's verifier."""
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 1024 * 1024:
            raise ValueError("manifest too large")
        if expected_manifest_sha256 is not None and (
                not re.fullmatch(r"[a-f0-9]{64}", expected_manifest_sha256)
                or hashlib.sha256(raw).hexdigest() != expected_manifest_sha256):
            raise ValueError("manifest identity mismatch")
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys,
                           parse_constant=_reject_constant)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("unsupported schema")
        model_id = value.get("model_id")
        if (value.get("artifact_kind") != "image_encoder"
                or model_id not in APPROVED_ENCODERS
                or value.get("revision") != APPROVED_ENCODERS[model_id]
                or value.get("processor_revision") != value["revision"]
                or value.get("license") != "apache-2.0"
                or value.get("license_verified") is not True
                or value.get("selected_adapter") is not None):
            raise ValueError("unapproved candidate identity or license")
        contract = value.get("loading_contract")
        if (not isinstance(contract, dict)
                or contract.get("architecture") != "SiglipVisionModel"
                or contract.get("image_processor") != "SiglipImageProcessor"
                or contract.get("local_files_only") is not True
                or contract.get("trust_remote_code") is not False
                or contract.get("tokenizer_required") is not False):
            raise ValueError("image-only offline loading contract required")
        files = value.get("files")
        if not isinstance(files, list) or len(files) != len(_REQUIRED_FILES):
            raise ValueError("exact image artifact set required")
        names = set()
        total = 0
        for entry in files:
            name = entry.get("path")
            if (name not in _REQUIRED_FILES or name in names
                    or type(entry.get("size_bytes")) is not int
                    or not 0 < entry["size_bytes"] <= 8 * 1024**3
                    or not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", ""))):
                raise ValueError("invalid candidate artifact")
            names.add(name)
            total += entry["size_bytes"]
        if names != _REQUIRED_FILES or type(value.get("total_bytes")) is not int or value["total_bytes"] != total:
            raise ValueError("candidate artifact total mismatch")
        return value
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise EncoderManifestError("encoder_manifest_invalid_or_unavailable") from None


def _check_paths(root: Path, manifest: dict[str, Any]) -> None:
    for entry in manifest["files"]:
        target = root / entry["path"]
        if target.is_symlink() or target.with_name(target.name + ".part").is_symlink():
            raise EncoderManifestError("encoder_snapshot_symlink_forbidden")


def verify_encoder_snapshot(snapshot: str | Path, manifest_path: str | Path, *,
                            expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Hash all four local assets; this function never contacts the network."""
    manifest = load_encoder_manifest(manifest_path, expected_manifest_sha256=expected_manifest_sha256)
    root = Path(snapshot)
    if root.is_symlink() or not root.is_dir():
        raise EncoderManifestError("encoder_snapshot_missing_or_symlink")
    _check_paths(root, manifest)
    try:
        for entry in manifest["files"]:
            target = root / entry["path"]
            if not target.is_file() or target.stat().st_size != entry["size_bytes"]:
                raise EncoderManifestError("encoder_snapshot_artifact_missing_or_size_mismatch")
            if _digest(target) != entry["sha256"]:
                raise EncoderManifestError("encoder_snapshot_checksum_mismatch")
    except OSError:
        raise EncoderManifestError("encoder_snapshot_artifact_unavailable") from None
    return {"verified": True, "artifact_kind": "image_encoder", "candidate": True,
            "model_id": manifest["model_id"], "revision": manifest["revision"],
            "processor_revision": manifest["processor_revision"], "license": manifest["license"],
            "files_verified": len(manifest["files"]), "bytes_verified": manifest["total_bytes"],
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            "local_inference_verified": False, "selected_for_production": False}


def stage_encoder_snapshot(snapshot: str | Path, manifest_path: str | Path, *,
                           expected_manifest_sha256: str | None = None,
                           max_workers: int = 2, timeout_sec: float = 1800) -> dict[str, Any]:
    """Explicit provisioning action; caller must hold the benchmark transfer gate."""
    manifest = load_encoder_manifest(manifest_path, expected_manifest_sha256=expected_manifest_sha256)
    if (type(max_workers) is not int or not 1 <= max_workers <= 4
            or not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 7200):
        raise EncoderManifestError("encoder_staging_budget_invalid")
    root = Path(snapshot)
    if root.is_symlink():
        raise EncoderManifestError("encoder_snapshot_symlink_forbidden")
    try:
        root.mkdir(parents=True, exist_ok=True)
        _check_paths(root, manifest)
        pending = [entry for entry in manifest["files"] if not (
            (root / entry["path"]).is_file()
            and (root / entry["path"]).stat().st_size == entry["size_bytes"]
            and _digest(root / entry["path"]) == entry["sha256"])]
        if shutil.disk_usage(root).free < sum(entry["size_bytes"] for entry in pending) + 512 * 1024**2:
            raise EncoderManifestError("encoder_staging_insufficient_disk")
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_download, root, entry, manifest["model_id"], manifest["revision"],
                                   started + timeout_sec) for entry in pending]
            try:
                downloaded = sum(future.result() for future in futures)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        result = verify_encoder_snapshot(root, manifest_path,
                                         expected_manifest_sha256=expected_manifest_sha256)
        return {**result, "bytes_downloaded": downloaded, "elapsed_sec": time.monotonic() - started}
    except OSError:
        raise EncoderManifestError("encoder_staging_network_or_disk_failure") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "stage"))
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    try:
        kwargs = {"expected_manifest_sha256": args.manifest_sha256}
        if args.command == "verify":
            result = verify_encoder_snapshot(args.snapshot, args.manifest, **kwargs)
        else:
            result = stage_encoder_snapshot(args.snapshot, args.manifest, max_workers=args.workers,
                                            timeout_sec=args.timeout, **kwargs)
    except ManifestError as exc:
        print(json.dumps({"verified": False, "error_code": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
