"""Offline verification of pinned public model artifacts (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestError(RuntimeError):
    """Sanitized artifact-validation failure."""


@dataclass(frozen=True)
class SnapshotVerification:
    model_id: str
    revision: str
    processor_revision: str
    files_verified: int
    bytes_verified: int
    manifest_sha256: str


def _reject_constant(_: str) -> None:
    raise ValueError("nonfinite JSON")


def load_manifest(path: str | Path = "models/manifest.json") -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 1024 * 1024:
            raise ValueError("manifest too large")
        value = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise ValueError("manifest object required")
        if value.get("license") != "apache-2.0" or value.get("license_verified") is not True:
            raise ValueError("baseline license not verified")
        if not re.fullmatch(r"[a-f0-9]{40}", value.get("revision", "")):
            raise ValueError("unpinned model revision")
        if value.get("processor_revision") != value["revision"]:
            raise ValueError("processor/model revision mismatch")
        if value.get("selected_adapter") is not None:
            raise ValueError("baseline verifier requires untuned checkpoint")
        files = value.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("artifact list missing")
        names = set()
        for entry in files:
            name = entry.get("path", "")
            if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                    or name.startswith("/") or ".." in Path(name).parts or name in names
                    or not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", ""))
                    or type(entry.get("size_bytes")) is not int or entry["size_bytes"] <= 0):
                raise ValueError("invalid artifact entry")
            names.add(name)
        if not any(name.endswith(".safetensors") for name in names):
            raise ValueError("weights absent from manifest")
        if not {"config.json", "preprocessor_config.json", "tokenizer_config.json"}.issubset(names):
            raise ValueError("processor absent from manifest")
        return value
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise ManifestError("model_manifest_invalid_or_unavailable") from None


def verify_local_snapshot(snapshot: str | Path, manifest_path: str | Path = "models/manifest.json") -> SnapshotVerification:
    """Hash every artifact, without network calls or diagnostic path output.

    Hugging Face snapshots normally use sibling content-addressed blob symlinks;
    artifact names still come only from the reviewed manifest.
    """
    manifest = load_manifest(manifest_path)
    root = Path(snapshot)
    if not root.is_dir():
        raise ManifestError("model_snapshot_missing")
    verified_bytes = 0
    for entry in manifest["files"]:
        try:
            artifact = root / entry["path"]
            if not artifact.is_file() or artifact.stat().st_size != entry["size_bytes"]:
                raise ManifestError("model_snapshot_artifact_missing_or_size_mismatch")
            digest = hashlib.sha256()
            with artifact.open("rb") as handle:
                while block := handle.read(8 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ManifestError("model_snapshot_checksum_mismatch")
            verified_bytes += entry["size_bytes"]
        except OSError:
            raise ManifestError("model_snapshot_artifact_unavailable") from None
    return SnapshotVerification(
        model_id=manifest["model_id"], revision=manifest["revision"],
        processor_revision=manifest["processor_revision"],
        files_verified=len(manifest["files"]), bytes_verified=verified_bytes,
        manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
    )
