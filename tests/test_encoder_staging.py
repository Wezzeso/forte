"""Synthetic bytes only: image-encoder artifact checks never instantiate ML."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forte.model.encoder_staging import (
    APPROVED_ENCODERS, EncoderManifestError, load_encoder_manifest,
    stage_encoder_snapshot, verify_encoder_snapshot,
)
from forte.model.manifest import ManifestError, load_manifest


class EncoderStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="forte-public-encoder-test-")
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "candidate"
        self.snapshot.mkdir()
        self.path = self.root / "encoder.manifest.json"
        self.contents = {name: ("synthetic artifact " + name).encode() for name in
                         ("model.safetensors", "config.json", "preprocessor_config.json", "README.md")}
        model_id, revision = next(iter(APPROVED_ENCODERS.items()))
        self.manifest = {"schema_version": 1, "artifact_kind": "image_encoder", "model_id": model_id,
            "revision": revision, "processor_revision": revision, "license": "apache-2.0",
            "license_verified": True, "selected_adapter": None,
            "loading_contract": {"architecture": "SiglipVisionModel", "image_processor": "SiglipImageProcessor",
                "local_files_only": True, "trust_remote_code": False, "tokenizer_required": False},
            "files": [{"path": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                      for name, raw in self.contents.items()], "total_bytes": sum(map(len, self.contents.values()))}
        self.save()
        for name, raw in self.contents.items():
            (self.snapshot / name).write_bytes(raw)

    def save(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_image_manifest_without_tokenizer_is_separate_from_qwen(self):
        self.assertEqual(load_encoder_manifest(self.path)["artifact_kind"], "image_encoder")
        with self.assertRaises(ManifestError):
            load_manifest(self.path)

    def test_offline_hash_proves_only_candidate_identity(self):
        with patch("forte.model.staging.build_opener", side_effect=AssertionError("network forbidden")):
            result = verify_encoder_snapshot(self.snapshot, self.path)
        self.assertEqual(result["files_verified"], 4)
        self.assertEqual(result["manifest_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertFalse(result["local_inference_verified"])
        self.assertFalse(result["selected_for_production"])

    def test_expected_manifest_digest_is_enforced(self):
        with self.assertRaises(EncoderManifestError):
            load_encoder_manifest(self.path, expected_manifest_sha256="0"*64)

    def test_unapproved_revision_license_or_loading_contract_rejected(self):
        for field, bad in (("revision", "a"*40), ("license", "unknown"), ("license_verified", False),
                           ("artifact_kind", "language_model")):
            original = self.manifest[field]
            with self.subTest(field=field):
                self.manifest[field] = bad
                self.save()
                with self.assertRaises(EncoderManifestError):
                    load_encoder_manifest(self.path)
            self.manifest[field] = original
        self.manifest["loading_contract"]["trust_remote_code"] = True
        self.save()
        with self.assertRaises(EncoderManifestError):
            load_encoder_manifest(self.path)

    def test_manifest_paths_extra_code_and_total_mismatch_rejected(self):
        for bad in ("../model.safetensors", "nested/model.safetensors", "model.py"):
            with self.subTest(path=bad):
                self.manifest["files"][0]["path"] = bad
                self.save()
                with self.assertRaises(EncoderManifestError):
                    load_encoder_manifest(self.path)
        self.manifest["files"][0]["path"] = "model.safetensors"
        self.manifest["total_bytes"] += 1
        self.save()
        with self.assertRaises(EncoderManifestError):
            load_encoder_manifest(self.path)

    def test_duplicate_json_keys_rejected(self):
        self.path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        with self.assertRaises(EncoderManifestError):
            load_encoder_manifest(self.path)

    def test_tamper_fails_without_outputting_bytes(self):
        target = self.snapshot / "model.safetensors"
        target.write_bytes(b"x"*target.stat().st_size)
        with self.assertRaisesRegex(EncoderManifestError, "checksum"):
            verify_encoder_snapshot(self.snapshot, self.path)

    def test_verified_files_never_download_again(self):
        with patch("forte.model.encoder_staging._download", side_effect=AssertionError("network forbidden")):
            result = stage_encoder_snapshot(self.snapshot, self.path)
        self.assertEqual(result["bytes_downloaded"], 0)

    def test_only_missing_artifact_is_staged_then_verified(self):
        (self.snapshot / "model.safetensors").unlink()
        calls = []
        def fake_download(root, entry, model_id, revision, deadline):
            calls.append(entry["path"])
            raw = self.contents[entry["path"]]
            (root / entry["path"]).write_bytes(raw)
            return len(raw)
        with patch("forte.model.encoder_staging._download", side_effect=fake_download):
            result = stage_encoder_snapshot(self.snapshot, self.path)
        self.assertEqual(calls, ["model.safetensors"])
        self.assertEqual(result["bytes_downloaded"], len(self.contents["model.safetensors"]))
        self.assertEqual(result["files_verified"], 4)

    def test_stdlib_only_cli_without_site_packages(self):
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        result = subprocess.run([sys.executable, "-S", "-m", "forte.model.encoder_staging", "verify",
            "--snapshot", str(self.snapshot), "--manifest", str(self.path)],
            env=env, capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["files_verified"], 4)


if __name__ == "__main__":
    unittest.main()
