"""Bounded public MEVA acquisition. REQUEST is embedded by the local launcher."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(item, target, measured):
    parsed = urllib.parse.urlparse(item["source_url"])
    if parsed.scheme != "https" or parsed.hostname not in {
        "mevadata-public-01.s3.amazonaws.com", "mevadata.org", "gitlab.kitware.com"
    }:
        raise ValueError("Unexpected public source host")
    expected_size = item["size_bytes"]
    expected_hash = item.get("sha256")
    if target.exists():
        actual_hash = digest(target)
        if target.stat().st_size != expected_size or (expected_hash and actual_hash != expected_hash):
            raise ValueError("Existing destination differs; refusing overwrite: " + target.name)
        measured.append(dict(item, sha256=actual_hash, reused=True))
        return
    partial = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        raise ValueError("Partial destination exists; refusing overwrite: " + partial.name)
    h = hashlib.sha256()
    count = 0
    request = urllib.request.Request(item["source_url"], headers={"User-Agent": "Forte-public-MEVA-subset"})
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("xb") as stream:
        if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) != expected_size:
            raise ValueError("Public source size changed: " + target.name)
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > expected_size:
                raise ValueError("Public source exceeds bound: " + target.name)
            h.update(chunk)
            stream.write(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    actual_hash = h.hexdigest()
    if count != expected_size or (expected_hash and actual_hash != expected_hash):
        raise ValueError("Public source identity changed: " + target.name)
    partial.replace(target)
    measured.append(dict(item, sha256=actual_hash, reused=False))
    print(json.dumps({"downloaded": target.name, "bytes": count}), flush=True)


def main(request):
    target = Path(request["destination"])
    if str(target) != "/data/forte-public-data/meva-counter-train-v1":
        raise ValueError("Unexpected destination")
    if target.exists() and target.is_symlink():
        raise ValueError("Refuse symlink destination")
    if len(request["videos"]) != 4 or sum(x["size_bytes"] for x in request["videos"]) > request["max_video_bytes"]:
        raise ValueError("Video inventory exceeds authorization")
    if sum(x["size_bytes"] for x in request["metadata"]) > request["max_metadata_bytes"]:
        raise ValueError("Metadata inventory exceeds bound")
    target.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema_version="forte.meva-public-subset-receipt.v1", started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), request=request, metadata=[], videos=[], status="started", no_gpu_or_model_calls=True, media_frame_review_performed=False)
    receipt_path = target / "acquisition-receipt.json"
    if receipt_path.exists():
        raise ValueError("Receipt already exists; refusing overwrite")
    try:
        license_item = next(x for x in request["metadata"] if x["path"] == "meva-license.txt")
        if license_item["sha256"] != request["license_expected_sha256"]:
            raise ValueError("License identity differs")
        fetch(license_item, target / "metadata" / license_item["path"], receipt["metadata"])
        # Verify currently served license bytes even if a local copy was reused.
        with urllib.request.urlopen(license_item["source_url"], timeout=30) as response:
            fresh_license = response.read(100000)
        if hashlib.sha256(fresh_license).hexdigest() != request["license_expected_sha256"]:
            raise ValueError("Public license changed; stop before media")
        for item in request["metadata"]:
            if item is not license_item:
                fetch(item, target / "metadata" / item["path"], receipt["metadata"])
        for item in request["videos"]:
            fetch(item, target / "originals" / item["path"], receipt["videos"])
        (target / "ATTRIBUTION.txt").write_text(request["attribution"] + "\nOriginal bytes retained. No media modifications.\n", encoding="utf-8")
        receipt["status"] = "complete_original_bytes_hashed"
    except Exception as exc:
        receipt["status"] = "failed_preserved_partial_bytes"
        receipt["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        receipt["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    print(json.dumps({"status": receipt["status"], "receipt_path": str(receipt_path), "receipt_sha256": digest(receipt_path), "video_count": len(receipt["videos"]), "video_bytes": sum(x["size_bytes"] for x in receipt["videos"]), "metadata_count": len(receipt["metadata"]), "error": receipt.get("error")}), flush=True)
    if receipt["status"] != "complete_original_bytes_hashed":
        raise SystemExit(1)


main(REQUEST)
