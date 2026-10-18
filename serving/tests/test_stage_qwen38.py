"""Public snapshot integrity, immutable directories and safe resumable transfer."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

import pytest

from serving import stage_qwen38 as stage


def test_real_pinned_metadata_is_consistent():
    root = Path(__file__).resolve().parents[2] / "configs/serving/qwen38"
    manifest = stage.validate_manifest(json.loads((root / "public-candidate.manifest.json").read_text()))
    metadata = stage.validate_snapshot_metadata(root / "public-snapshot", manifest)
    assert len(manifest["files"]) == 32
    assert manifest["total_bytes"] == 55_586_114_863
    assert metadata["indexed_shards"] == 18
    assert metadata["linear_attention_layers"] == 48


@pytest.mark.parametrize("name", ["../baseline/config.json", "/etc/passwd", "x\\config.json", "a/../x", "C:/x", None])
def test_manifest_paths_cannot_escape(name):
    with pytest.raises(ValueError, match="unsafe_manifest_artifact_path"):
        stage._artifact_path(name)


def test_existing_directory_is_preserved(tmp_path):
    root = Path(__file__).resolve().parents[2] / "configs/serving/qwen38"
    manifest = json.loads((root / "public-candidate.manifest.json").read_text())
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    original = baseline / "config.json"
    original.write_bytes(b"existing baseline bytes")
    with pytest.raises(ValueError, match="unmarked_nonempty"):
        stage.stage_snapshot(baseline, manifest)
    assert original.read_bytes() == b"existing baseline bytes"


def test_completed_mismatch_is_not_overwritten(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"old")
    item = {"path": path.name, "size_bytes": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
    with pytest.raises(ValueError, match="existing_artifact_mismatch_preserved"):
        stage._download_file(tmp_path, item, retries=0, timeout=1)
    assert path.read_bytes() == b"old"


def test_partial_download_uses_range_and_verifies_final_bytes(tmp_path, monkeypatch):
    payload = b"complete public test bytes"
    partial = tmp_path / "config.json.part"
    partial.write_bytes(payload[:8])
    requests = []

    class Response(BytesIO):
        status = 206
        url = "https://cas-bridge.xethub.hf.co/public-test"
        headers = {"Content-Range": f"bytes 8-{len(payload) - 1}/{len(payload)}", "Content-Length": str(len(payload) - 8)}

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return Response(payload[8:])

    monkeypatch.setattr(stage, "build_opener", lambda *args: Opener())
    item = {"path": "config.json", "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    receipt = stage._download_file(tmp_path, item, retries=0, timeout=1)
    assert requests[0].headers["Range"] == "bytes=8-"
    assert (tmp_path / "config.json").read_bytes() == payload
    assert receipt["resumed_bytes"] == 8
    assert not partial.exists()


@pytest.mark.parametrize("url", ["http://huggingface.co/x", "https://example.com/file", "https://user:secret@huggingface.co/x", "https://huggingface.co.attacker.example/x"])
def test_download_redirects_reject_non_public_destinations(url):
    with pytest.raises(ValueError, match="non_public_model_download_url"):
        stage._check_public_url(url)


def test_wrong_content_range_preserves_partial(tmp_path, monkeypatch):
    partial = tmp_path / "config.json.part"
    partial.write_bytes(b"abc")

    class Response(BytesIO):
        status = 206
        url = "https://huggingface.co/test"
        headers = {"Content-Range": "bytes 0-2/6"}

    class Opener:
        def open(self, request, timeout):
            return Response(b"def")

    monkeypatch.setattr(stage, "build_opener", lambda *args: Opener())
    item = {"path": "config.json", "size_bytes": 6, "sha256": hashlib.sha256(b"abcdef").hexdigest()}
    with pytest.raises(ValueError, match="invalid_partial_content_range"):
        stage._download_file(tmp_path, item, retries=0, timeout=1)
    assert partial.read_bytes() == b"abc"


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path, monkeypatch):
    payload = b"abcdef"
    (tmp_path / "config.json.part").write_bytes(b"abc")

    class Response(BytesIO):
        status = 200
        url = "https://huggingface.co/test"
        headers = {"Content-Length": "6"}

    class Opener:
        def open(self, request, timeout):
            return Response(payload)

    monkeypatch.setattr(stage, "build_opener", lambda *args: Opener())
    item = {"path": "config.json", "size_bytes": 6, "sha256": hashlib.sha256(payload).hexdigest()}
    stage._download_file(tmp_path, item, retries=0, timeout=1)
    assert (tmp_path / "config.json").read_bytes() == payload


def test_download_hash_failure_does_not_promote_partial(tmp_path, monkeypatch):
    class Response(BytesIO):
        status = 200
        url = "https://huggingface.co/test"
        headers = {"Content-Length": "3"}

    class Opener:
        def open(self, request, timeout):
            return Response(b"bad")

    monkeypatch.setattr(stage, "build_opener", lambda *args: Opener())
    item = {"path": "config.json", "size_bytes": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
    with pytest.raises(ValueError, match="download_hash_mismatch_partial_preserved"):
        stage._download_file(tmp_path, item, retries=0, timeout=1)
    assert (tmp_path / "config.json.part").read_bytes() == b"bad"
    assert not (tmp_path / "config.json").exists()

