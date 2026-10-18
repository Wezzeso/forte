"""Stage a public pinned Qwen snapshot without loading a model or touching serving.

Only public Hugging Face URLs are used; this module never reads account tokens.
Completed files are immutable: a mismatch fails rather than replacing a file.
Interrupted downloads remain as ``.part`` files and are resumed with HTTP Range.
The same verifier can run with no network access before private inference.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WEIGHT_BYTES = 55_563_006_776
REQUIRED_METADATA = frozenset({
    "LICENSE", "README.md", "config.json", "generation_config.json",
    "chat_template.jinja", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "preprocessor_config.json", "video_preprocessor_config.json",
})
SHARDS = frozenset(f"model-{i:05d}-of-00018.safetensors" for i in range(1, 19))
MARKER = ".forte-public-snapshot.json"
LOCK = ".forte-stage.lock"
DEFAULT_RESERVE_BYTES = 8 * 1024**3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(name: str) -> PurePosixPath:
    if not isinstance(name, str):
        raise ValueError("unsafe_manifest_artifact_path")
    value = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name
            or value.is_absolute() or value.as_posix() != name
            or any(part in {".", ".."} for part in value.parts)):
        raise ValueError("unsafe_manifest_artifact_path")
    return value


def validate_manifest(manifest: dict) -> dict:
    """Validate the approved model/revision, exact shards and explicit hashes."""
    if (manifest.get("model_id", manifest.get("repo_id")) != MODEL_ID
            or manifest.get("revision") != REVISION):
        raise ValueError("unapproved_model_or_revision")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("missing_file_inventory")
    paths: set[str] = set()
    total = 0
    weights_total = 0
    for item in files:
        name = item.get("path")
        _artifact_path(name)
        if name in paths or name in {MARKER, LOCK} or name.endswith(".part"):
            raise ValueError("duplicate_or_reserved_artifact_path")
        paths.add(name)
        size = item.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise ValueError("invalid_artifact_size")
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
            raise ValueError("missing_artifact_sha256")
        if name.endswith(".safetensors"):
            weights_total += size
        elif PurePosixPath(name).suffix not in {".json", ".jinja", ".txt", ".md", ""} and name != ".gitattributes":
            raise ValueError("unexpected_artifact_format")
        total += size
    if {name for name in paths if name.endswith(".safetensors")} != SHARDS:
        raise ValueError("weight_shard_inventory_mismatch")
    if not REQUIRED_METADATA <= paths:
        raise ValueError("required_processor_or_config_missing")
    if weights_total != WEIGHT_BYTES:
        raise ValueError("weight_byte_inventory_mismatch")
    if manifest.get("total_bytes", total) != total:
        raise ValueError("total_byte_inventory_mismatch")
    return {**manifest, "model_id": MODEL_ID, "total_bytes": total}


def _local_path(snapshot: Path, name: str) -> Path:
    rel = _artifact_path(name)
    path = snapshot.joinpath(*rel.parts)
    # No symlink may redirect writes or reads to baseline/other storage.
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError("symlink_in_snapshot_path")
        if candidate == snapshot:
            break
    if not path.resolve().is_relative_to(snapshot.resolve()):
        raise ValueError("artifact_escaped_snapshot")
    return path


def validate_snapshot_metadata(snapshot: Path, manifest: dict) -> dict:
    """Validate architecture, shard index and native processor configuration."""
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if (config.get("model_type") != "qwen3_5"
            or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
            or config.get("language_model_only") is not False):
        raise ValueError("architecture_mismatch")
    layers = config.get("text_config", {}).get("layer_types", [])
    if (len(layers) != 64 or layers.count("linear_attention") != 48
            or layers.count("full_attention") != 16):
        raise ValueError("hybrid_layer_inventory_mismatch")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or set(weight_map.values()) != SHARDS:
        raise ValueError("safetensors_index_shard_mismatch")
    processors = {}
    for filename, field, expected in (
        ("preprocessor_config.json", "image_processor_type", "Qwen2VLImageProcessorFast"),
        ("video_preprocessor_config.json", "video_processor_type", "Qwen3VLVideoProcessor"),
    ):
        processor = json.loads((snapshot / filename).read_text(encoding="utf-8"))
        if (processor.get("processor_class") != "Qwen3VLProcessor"
                or processor.get(field) != expected or processor.get("patch_size") != 16
                or processor.get("temporal_patch_size") != 2 or processor.get("merge_size") != 2):
            raise ValueError("processor_config_mismatch")
        processors[filename] = {key: processor[key] for key in (
            "processor_class", field, "patch_size", "temporal_patch_size", "merge_size")}
    tokenizer = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    if tokenizer.get("tokenizer_class") != "Qwen2Tokenizer":
        raise ValueError("tokenizer_class_mismatch")
    template = (snapshot / "chat_template.jinja").read_text(encoding="utf-8")
    if tokenizer.get("chat_template") != template:
        raise ValueError("tokenizer_chat_template_mismatch")
    return {"model_type": "qwen3_5", "architecture": "Qwen3_5ForConditionalGeneration",
            "language_layers": 64, "linear_attention_layers": 48, "full_attention_layers": 16,
            "indexed_tensors": len(weight_map), "indexed_shards": len(SHARDS),
            "processors": processors, "tokenizer_class": "Qwen2Tokenizer",
            "tokenizer_template_matches_file": True,
            "processor_execution_verified": False, "model_load_verified": False}


def verify_snapshot(snapshot: Path, manifest: dict) -> dict:
    """Hash every declared file and verify metadata, without model imports/network."""
    manifest = validate_manifest(manifest)
    snapshot = Path(snapshot).absolute()
    allowed = {item["path"] for item in manifest["files"]} | {MARKER, LOCK}
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink_in_snapshot_path")
        if path.is_file() and path.relative_to(snapshot).as_posix() not in allowed:
            raise ValueError("undeclared_snapshot_artifact")
    results = []
    for item in manifest["files"]:
        path = _local_path(snapshot, item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError("missing_or_wrong_size_artifact: " + item["path"])
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError("artifact_hash_mismatch: " + item["path"])
        results.append({"path": item["path"], "size_bytes": path.stat().st_size,
                        "sha256": actual, "verified": True})
    metadata = validate_snapshot_metadata(snapshot, manifest)
    return {"schema_version": "forte.qwen38.snapshot-verification.v1", "verified": True,
            "captured_at_utc": utc_now(), "model_id": MODEL_ID, "revision": REVISION,
            "processor_revision": REVISION, "snapshot": str(snapshot),
            "files_verified": len(results), "bytes_verified": sum(row["size_bytes"] for row in results),
            "files": results, "metadata": metadata, "private_data_read": False,
            "model_loaded": False, "production_selected": False}


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _check_public_url(url: str) -> None:
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in {None, 443}
            or not (hostname == "huggingface.co" or hostname.endswith(".huggingface.co")
                    or hostname.endswith(".xethub.hf.co") or hostname.endswith(".hf.co"))):
        raise ValueError("non_public_model_download_url")


class _PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@contextmanager
def _stage_lock(snapshot: Path):
    path = snapshot / LOCK
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        lock = json.loads(path.read_text(encoding="utf-8"))
        # Resume a crashed process only on the same machine when it no longer exists.
        if lock.get("hostname") != socket.gethostname():
            raise RuntimeError("snapshot_locked_by_other_host") from None
        try:
            os.kill(int(lock["pid"]), 0)
        except ProcessLookupError:
            path.unlink()
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        else:
            raise RuntimeError("snapshot_staging_already_running") from None
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), "created_at_utc": utc_now()}, stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _download_file(snapshot: Path, item: dict, *, retries: int, timeout: float) -> dict:
    final = _local_path(snapshot, item["path"])
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = _local_path(snapshot, item["path"] + ".part")
    expected_size = item["size_bytes"]
    if final.exists():
        if final.stat().st_size != expected_size or sha256_file(final) != item["sha256"]:
            raise ValueError("existing_artifact_mismatch_preserved: " + item["path"])
        return {"path": item["path"], "downloaded": False, "verified": True}
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{quote(item['path'], safe='/')}"
    # Do not use the system proxy, auth handlers, netrc, HF cache or credential files.
    from urllib.request import ProxyHandler
    opener = build_opener(ProxyHandler({}), _PublicRedirects())
    resumed_bytes = partial.stat().st_size if partial.exists() else 0
    for attempt in range(retries + 1):
        current = partial.stat().st_size if partial.exists() else 0
        if current > expected_size:
            raise ValueError("oversized_partial_preserved: " + item["path"])
        try:
            if current < expected_size:
                headers = {"User-Agent": "forte-public-artifact-stager/1", "Accept-Encoding": "identity"}
                if current:
                    headers["Range"] = f"bytes={current}-"
                with opener.open(Request(url, headers=headers), timeout=timeout) as response:
                    _check_public_url(response.url)
                    status = response.status
                    if current and status == 206:
                        expected_range = f"bytes {current}-{expected_size - 1}/{expected_size}"
                        if response.headers.get("Content-Range") != expected_range:
                            raise ValueError("invalid_partial_content_range")
                        mode = "ab"
                    elif status == 200:
                        # Server ignored Range; restart the partial atomically on this handle.
                        mode, current = "wb", 0
                    else:
                        raise ValueError("unexpected_download_status")
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) != expected_size - current:
                        raise ValueError("download_content_length_mismatch")
                    with partial.open(mode) as stream:
                        while block := response.read(8 * 1024**2):
                            current += len(block)
                            if current > expected_size:
                                raise ValueError("download_exceeded_expected_size")
                            stream.write(block)
                        stream.flush()
                        os.fsync(stream.fileno())
            if partial.stat().st_size != expected_size:
                raise URLError("incomplete_public_artifact")
            actual = sha256_file(partial)
            if actual != item["sha256"]:
                raise ValueError("download_hash_mismatch_partial_preserved: " + item["path"])
            if final.exists():
                raise RuntimeError("artifact_appeared_during_download")
            partial.replace(final)
            result = {"path": item["path"], "size_bytes": expected_size, "sha256": actual,
                      "downloaded": True, "resumed_bytes": resumed_bytes, "verified": True}
            print(json.dumps({"event": "public_artifact_verified", **result}), flush=True)
            return result
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError):
            if attempt >= retries:
                raise RuntimeError("public_download_failed: " + item["path"]) from None
            time.sleep(min(2**attempt, 15))
    raise RuntimeError("download_retry_exhausted")


def stage_snapshot(snapshot: Path, manifest: dict, *, workers: int = 4, retries: int = 3,
                   timeout: float = 120, reserve_bytes: int = DEFAULT_RESERVE_BYTES) -> dict:
    manifest = validate_manifest(manifest)
    snapshot = Path(snapshot).absolute()
    if not 1 <= workers <= 8 or reserve_bytes < 0:
        raise ValueError("invalid_staging_limits")
    if snapshot.is_symlink() or any(parent.is_symlink() for parent in snapshot.parents):
        raise ValueError("symlink_in_snapshot_root")
    snapshot.mkdir(parents=True, exist_ok=True)
    marker = snapshot / MARKER
    identity = {"model_id": MODEL_ID, "revision": REVISION}
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError("snapshot_directory_identity_mismatch")
    else:
        if any(snapshot.iterdir()):
            raise ValueError("refuse_unmarked_nonempty_snapshot_directory")
        _atomic_json(marker, identity)
    with _stage_lock(snapshot):
        remaining = 0
        for item in manifest["files"]:
            final = _local_path(snapshot, item["path"])
            partial = _local_path(snapshot, item["path"] + ".part")
            if not final.exists():
                remaining += max(0, item["size_bytes"] - (partial.stat().st_size if partial.exists() else 0))
        free = shutil.disk_usage(snapshot).free
        if free < remaining + reserve_bytes:
            raise RuntimeError("insufficient_free_disk_for_snapshot_and_reserve")
        started = time.monotonic()
        downloads = []
        # Inspect every small config before downloading 55GB of weights.
        for item in manifest["files"]:
            if item["path"] not in SHARDS:
                downloads.append(_download_file(snapshot, item, retries=retries, timeout=timeout))
        validate_snapshot_metadata(snapshot, manifest)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download_file, snapshot, item, retries=retries, timeout=timeout)
                       for item in manifest["files"] if item["path"] in SHARDS]
            for future in as_completed(futures):
                downloads.append(future.result())
        receipt = verify_snapshot(snapshot, manifest)
        receipt.update({"staging_started_free_bytes": free, "staging_required_remaining_bytes": remaining,
                        "disk_reserve_bytes": reserve_bytes, "staging_elapsed_seconds": time.monotonic() - started,
                        "download_results": downloads, "network_scope": "public_pinned_huggingface_artifacts_only",
                        "credentials_used": False, "gpu_accessed": False})
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--disk-reserve-gib", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    result = (verify_snapshot(args.snapshot, manifest) if args.verify_only else stage_snapshot(
        args.snapshot, manifest, workers=args.workers, reserve_bytes=args.disk_reserve_gib * 1024**3))
    result["manifest_sha256"] = sha256_file(args.manifest)
    result["stager_sha256"] = sha256_file(Path(__file__))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.receipt, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"files", "download_results"}}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Never print HTTP request/signed redirect URLs or credential-bearing tracebacks.
        print(json.dumps({"event": "public_staging_failed", "error_type": type(error).__name__,
                          "error_code": str(error) if re.fullmatch(r"[a-z_]+(?:: [a-zA-Z0-9_.-]+)?", str(error)) else None}),
              file=sys.stderr, flush=True)
        raise SystemExit(1)
