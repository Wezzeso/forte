"""Candidate artifact/request safety tests: synthetic bytes, no CUDA calls."""

import base64
import asyncio
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from serving.qwen38_identity import (CandidateError, REVISION, canonical_sha256, file_sha256,
                                      load_expected_runtime_identity, load_json, verify_candidate)
from serving.qwen38_launch import verify_private_namespace
from serving.qwen38_service import create_app, parse_request
from serving.tests.qwen38_adapter_fixtures import (adapter_config_fixture, archive_fixture,
    inventory_fixture, write_adapter_fixture)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = json.loads(Path("configs/serving/qwen38/candidate.runtime.example.json").read_text())
        self.config.update(runtime_image_sha256="a" * 64,
                           package_versions={name: "1.2.3" for name in self.config["package_versions"]})
        self.config["package_versions"]["xgrammar"] = "0.2.7"
        self.config["model_path"] = str(self.root / "base")
        adapter = self.root / "adapter"
        adapter.mkdir()
        self.config["adapter_path"] = str(adapter)
        (adapter / "adapter_config.json").write_text(json.dumps(adapter_config_fixture(self.config["model_path"])))
        # Generated zero tensor payloads; topology and source binding are real.
        write_adapter_fixture(adapter / "adapter_model.safetensors")
        checksums = {p.name: file_sha256(p) for p in adapter.iterdir()}
        self.config["adapter_sha256"] = canonical_sha256(checksums)
        self.config["adapter_config_sha256"] = checksums["adapter_config.json"]
        manifest = {"files": [{"path": "tokenizer.json", "sha256": self.config["tokenizer_sha256"]},
                               {"path": "chat_template.jinja", "sha256": self.config["chat_template_sha256"]}]}
        self.bind("base_manifest", "manifest.json", manifest)
        archive_fixture(self.root / "source.tar")
        self.config["source_archive"] = {"path": str(self.root / "source.tar"),
                                         "sha256": file_sha256(self.root / "source.tar")}
        self.run = {"schema_version": "forte.qwen38.silver-run.v1", "status": "candidate_unselected",
                    "dataset_origin": "local_teacher_silver", "human_verified": False,
                    "base_revision": REVISION, "processor_revision": REVISION,
                    "base_manifest_sha256": self.config["base_manifest"]["sha256"],
                    "gpu_executed": True, "seed": 20261017,
                    "code_archive_sha256": self.config["source_archive"]["sha256"],
                    "runtime_image_sha256": self.config["runtime_image_sha256"],
                    "optimizer_steps": 2, "finite_loss_verified": True, "gradients_verified": True,
                    "frozen_tensors_unchanged": True, "peft_load_verified": True,
                    "reload_forward_verified": True, "changed_adapter_tensors": 3,
                    "quantization": deepcopy(self.config["quantization"]),
                    "processor_limits": {"max_pixels_per_image": self.config["limits"]["max_pixels_per_image"]},
                    "adapted_module_inventory": inventory_fixture(),
                    "changed_adapter_groups": {"self_attn": True, "linear_attn": True, "mlp": True},
                    "adapter_checksums": checksums, "adapter_tree_sha256": self.config["adapter_sha256"]}
        self.bind_run()

    def bind(self, key, name, value):
        path = self.root / name
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        self.config[key] = {"path": str(path), "sha256": file_sha256(path)}

    def bind_run(self):
        self.bind("training_run", "run.json", self.run)

    def verify(self):
        with patch("serving.qwen38_identity.verify_candidate_base", return_value={"files_verified": 32}):
            return verify_candidate(self.config)

    def test_exact_hashes_do_not_claim_loaded_or_selected(self):
        result = self.verify()
        self.assertTrue(result["verified"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["production_selected"])
        path = self.root / "runtime.json"
        path.write_text(json.dumps(self.config))
        expected = load_expected_runtime_identity(path)
        self.assertTrue(all(result[key] == value for key, value in expected.items()))

    def test_adapter_bytes_and_unlisted_files_fail(self):
        path = Path(self.config["adapter_path"]) / "adapter_model.safetensors"
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(CandidateError, "candidate_adapter_checksum_mismatch"):
            self.verify()
        (path.parent / "unexpected.bin").write_bytes(b"unexpected")
        with self.assertRaisesRegex(CandidateError, "candidate_adapter_extra_or_missing_files"):
            self.verify()

    def test_preparation_and_quantization_mismatch_cannot_be_served(self):
        self.run["optimizer_steps"] = 0
        self.bind_run()
        with self.assertRaisesRegex(CandidateError, "two_real_optimizer_steps"):
            self.verify()
        self.run["optimizer_steps"] = 2
        self.run["quantization"]["format"] = "none"
        self.bind_run()
        with self.assertRaisesRegex(CandidateError, "training_quantization_mismatch"):
            self.verify()

    def test_processor_budget_and_hybrid_coverage_are_bound(self):
        self.run["processor_limits"]["max_pixels_per_image"] = 65536
        self.bind_run()
        with self.assertRaisesRegex(CandidateError, "training_processor_budget_mismatch"):
            self.verify()
        self.run["processor_limits"]["max_pixels_per_image"] = self.config["limits"]["max_pixels_per_image"]
        self.run["changed_adapter_groups"]["linear_attn"] = False
        self.bind_run()
        with self.assertRaisesRegex(CandidateError, "hybrid_adapter_inventory_mismatch"):
            self.verify()

    def test_duplicate_json_keys_rejected(self):
        path = self.root / "duplicate.json"
        path.write_text('{"status":"candidate_unselected","status":"trained"}')
        with self.assertRaisesRegex(CandidateError, "duplicate_json_key"):
            load_json(path)
        path.write_text('{"score":1e999}')
        with self.assertRaisesRegex(CandidateError, "nonfinite_number"):
            load_json(path)

    def request(self):
        image = Image.new("RGB", (16, 16), "red")
        buffer = BytesIO()
        image.save(buffer, "JPEG")
        return {"model": self.config["served_model_name"], "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Synthetic test"}, {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}}]}],
            "structured_outputs": {"json": {"type": "object"}}, "max_tokens": 32, "seed": 20261017}

    def test_jpeg_request_bound_to_template_and_generation(self):
        document = self.request()
        messages, images, schema, maximum = parse_request(document, self.config)
        self.assertEqual(maximum, 32)
        self.assertEqual(messages[0]["content"][1], {"type": "image"})
        for image in images:
            image.close()
        for override in ({"seed": 17}, {"max_tokens": 5000}, {"chat_template_kwargs": {"enable_thinking": True}},
                         {"tools": []}):
            with self.subTest(override=override), self.assertRaises(CandidateError):
                parse_request({**document, **override}, self.config)

    def test_remote_media_and_remote_schema_never_fetch(self):
        document = self.request()
        document["messages"][0]["content"][1]["image_url"]["url"] = "https://example.invalid/private.jpg"
        with self.assertRaisesRegex(CandidateError, "jpeg_data_required"):
            parse_request(document, self.config)
        document = self.request()
        document["structured_outputs"]["json"] = {"$ref": "https://example.invalid/schema"}
        with self.assertRaisesRegex(CandidateError, "remote_schema_reference_forbidden"):
            parse_request(document, self.config)

    def test_default_route_forbidden(self):
        path = self.root / "route"
        path.write_text("Iface Destination Gateway Flags RefCnt Use Metric Mask\neth0 00000000 0100000A 0003 0 0 0 00000000\n")
        with self.assertRaisesRegex(CandidateError, "default_network_route_forbidden"):
            verify_private_namespace(path)
        path.write_text("Iface Destination Gateway Flags RefCnt Use Metric Mask\neth0 0000000A 00000000 0001 0 0 0 00FFFFFF\n")
        verify_private_namespace(path)

    def test_http_surface_uses_candidate_readiness_and_bounds_media(self):
        import httpx

        class SyntheticRuntime:
            def __init__(self, config):
                self.config, self.model, self.ready = config, object(), True
                self.requests = 0
            def readiness(self):
                return {"ready": self.ready, "backend": "transformers_peft", "model_loaded": True}
            def generate(self, document, **kwargs):
                self.requests += 1
                return {"synthetic_test_only": True}

        runtime = SyntheticRuntime(self.config)
        async def run():
            with patch.dict(os.environ, {"VLLM_API_KEY": ""}):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(runtime)),
                                             base_url="http://127.0.0.1") as client:
                    ready = await client.get("/readyz")
                    self.assertEqual(ready.status_code, 200)
                    self.assertEqual(ready.json()["backend"], "transformers_peft")
                    good = await client.post("/v1/chat/completions", json=self.request())
                    self.assertEqual(good.status_code, 200)
                    bad = await client.post("/v1/chat/completions", json={"url": "https://example.invalid/image.jpg"})
                    self.assertEqual(bad.status_code, 400)
                    self.assertEqual(runtime.requests, 1)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
