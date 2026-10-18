"""No model/private calls: untuned diagnostic contracts and mock HTTP only."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from forte.model.adapter import LocalModelConfig, LocalModelProvider, ModelUnavailable
from serving.qwen38_comparison_profile import require_matching_profiles, runtime_profile_sha256
from serving.qwen38_identity import CandidateError, MODEL_ID, REVISION, validate_config
from serving.qwen38_untuned import (UntunedRuntime, expected_untuned_identity,
                                    validate_untuned_config, verify_untuned)
from serving.qwen38_untuned_provider import UntunedModelProvider


def config_fixture():
    config = {"schema_version": "forte.qwen38.untuned-runtime.v1", "backend": "transformers_base_diagnostic",
        "arm": "qwen38_untuned", "diagnostic_only": True, "production_selected": False,
        "model_id": MODEL_ID, "model_revision": REVISION, "processor_revision": REVISION,
        "served_model_name": "Qwen3.8-27B-forte-untuned-diagnostic", "model_path": "/models/base",
        "adapter_sha256": None, "adapter_config_sha256": None, "training_run_sha256": None,
        "base_manifest": {"path": "/release/base-manifest.json", "sha256": "a" * 64},
        "source_archive": {"path": "/release/source.tar", "sha256": "b" * 64},
        "tokenizer_sha256": "c" * 64, "chat_template_sha256": "d" * 64, "runtime_image_sha256": "e" * 64,
        "package_versions": {key: "synthetic-version" for key in ("torch", "transformers", "peft", "fastapi", "uvicorn", "pillow", "jsonschema", "bitsandbytes")},
        "quantization": {"format": "nf4", "compute_dtype": "bfloat16", "double_quant": True},
        "generation": {"do_sample": False, "temperature": 0, "seed": 20261017, "max_new_tokens": 4096,
                       "enable_thinking": False, "preserve_thinking": False, "num_beams": 1},
        "device": "cuda:0", "host": "127.0.0.1", "port": 8000,
        "limits": {"max_images": 16, "max_pixels_per_image": 1048576, "max_source_pixels_per_image": 64000000,
                   "max_context_tokens": 32768, "max_body_bytes": 100663296, "request_timeout_seconds": 240},
        "prewarm_shapes": [{"frames": 1, "width": 448, "height": 448}]}
    config["structured_decoding"] = {"backend": "xgrammar_hf", "version": "0.2.7",
        "schema_policy": "forte.decision-json-subset.v1", "any_order": False, "any_whitespace": False,
        "strict_mode": True, "max_cache_entries": 8, "max_cache_bytes": 67108864,
        "fact_order": "canonical_requested_fact_ids.v1"}
    config["package_versions"]["xgrammar"] = "0.2.7"
    config["runtime_profile_sha256"] = runtime_profile_sha256(config)
    return config


def ready_fixture(config):
    return {**expected_untuned_identity(config), "verified": True, "ready": True,
            "structured_decoding": {**config["structured_decoding"], "fresh_matcher_per_generate": True,
                                    "strict_post_generation_validation": True, "requests_bound_to_grammar": 3},
            "model_files_verified": True, "model_loaded": True, "prewarm_completed": True,
            "schema_smoke_completed": True, "namespace_default_route_absent": True,
            "adapter_loaded_tensor_count": 0, "runtime_load": {"model_gpu_loaded": True,
                "model_type": "qwen3_5", "attn_implementation": "sdpa", "cuda_version": "12.9",
                "gpu_compute_capability": [8, 0], "quantization": config["quantization"],
                "nf4_module_count": 42, "actual_nf4_state_verified": True,
                "hybrid_kernel_path_verified": False}}


class UntunedContracts(unittest.TestCase):
    def setUp(self):
        self.config = config_fixture()

    def test_base_only_identity_is_explicitly_untrained_and_nonproduction(self):
        identity = expected_untuned_identity(self.config)
        self.assertEqual(identity["backend"], "transformers_base_diagnostic")
        self.assertIsNone(identity["adapter_sha256"])
        self.assertIsNone(identity["training_run_sha256"])
        self.assertIs(identity["trained"], False)
        self.assertIs(identity["production_selected"], False)

    def test_untuned_config_never_passes_existing_trained_candidate_gate(self):
        with self.assertRaises(CandidateError):
            validate_config(self.config)

    def test_no_dummy_adapter_or_zero_step_training_workaround_is_accepted(self):
        for field, value in (("adapter_path", "/fake/adapter"), ("training_run", {"optimizer_steps": 0}),
                             ("adapter_sha256", "f" * 64), ("adapter_config_sha256", "f" * 64),
                             ("training_run_sha256", "f" * 64), ("production_selected", True),
                             ("diagnostic_only", False), ("model_revision", "1" * 40)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.config)
                changed[field] = value
                with self.assertRaises(CandidateError):
                    validate_untuned_config(changed)

    def test_unpinned_or_changed_effective_generation_and_pixel_profile_fails(self):
        for field in ("generation", "limits", "quantization", "package_versions", "structured_decoding"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.config)
                if field == "generation":
                    changed[field]["max_new_tokens"] = 2048
                elif field == "limits":
                    changed[field]["max_pixels_per_image"] = 524288
                elif field == "quantization":
                    changed[field] = {"format": "none", "compute_dtype": "bfloat16", "double_quant": False}
                elif field == "structured_decoding":
                    changed[field]["any_order"] = True
                else:
                    changed[field]["transformers"] = "different-version"
                with self.assertRaises(CandidateError):
                    validate_untuned_config(changed)

    def test_same_runtime_profile_excludes_only_arm_adapter_and_served_alias(self):
        trained = copy.deepcopy(self.config)
        trained.update(schema_version="forte.qwen38.runtime.v1", backend="transformers_peft",
                       served_model_name="trained-alias", adapter_sha256="f" * 64,
                       adapter_config_sha256="f" * 64, training_run_sha256="f" * 64)
        self.assertEqual(require_matching_profiles(self.config, trained), self.config["runtime_profile_sha256"])
        trained["limits"]["max_pixels_per_image"] //= 2
        with self.assertRaises(CandidateError):
            require_matching_profiles(self.config, trained)

    def test_verification_never_opens_adapter_or_training_artifact(self):
        manifest = {"files": [{"path": "tokenizer.json", "sha256": "c" * 64},
                               {"path": "chat_template.jinja", "sha256": "d" * 64}]}
        with (patch("serving.qwen38_untuned._bound_file", side_effect=lambda entry: Path(entry["path"])) as bound,
              patch("serving.qwen38_untuned.verify_candidate_base", return_value={"verified": True, "files_verified": 32}) as base,
              patch("serving.qwen38_untuned.load_json", return_value=manifest)):
            identity = verify_untuned(self.config)
        self.assertEqual(bound.call_count, 2)
        self.assertEqual(base.call_count, 1)
        self.assertIs(identity["model_files_verified"], True)
        self.assertIs(identity["model_loaded"], False)
        self.assertIsNone(identity["training_run_sha256"])

    def test_unverified_snapshot_cannot_construct_runtime(self):
        with self.assertRaises(CandidateError):
            UntunedRuntime(self.config, expected_untuned_identity(self.config))

    def test_runtime_reuses_shared_loader_and_generates_without_adapter(self):
        class Parameter:
            requires_grad = True
        class FakeBase:
            peft_config = None
            _forte_candidate_load_receipt = {"synthetic_only": True}
            def __init__(self):
                self.parameter = Parameter()
            def named_parameters(self):
                return [("model.layers.0.weight", self.parameter)]
            def parameters(self):
                return [self.parameter]
            def requires_grad_(self, value):
                self.parameter.requires_grad = value
            def eval(self):
                pass
        identity = {**expected_untuned_identity(self.config), "verified": True, "model_files_verified": True}
        runtime = UntunedRuntime(self.config, identity)
        base = FakeBase()
        processor = SimpleNamespace(tokenizer=object())
        with patch("serving.qwen38_untuned.load_processor_and_base", return_value=(processor, base)) as load, \
                patch("serving.qwen38_grammar.NativeGrammar") as grammar:
            runtime.load()
            grammar.assert_called_once_with(processor.tokenizer, base, self.config["structured_decoding"])
            grammar.return_value.precompile_public_schema.assert_called_once_with(max_images=16)
        load.assert_called_once_with(self.config)
        self.assertIs(runtime.model, base)
        self.assertEqual(runtime.readiness()["adapter_loaded_tensor_count"], 0)
        self.assertFalse(runtime.readiness()["ready"])
        self.assertFalse(base.parameter.requires_grad)


class UntunedProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = config_fixture()
        self.ready = ready_fixture(self.config)
        self.requests = []

    def transport(self):
        def handler(request):
            self.requests.append(request)
            if request.url.path == "/health":
                return httpx.Response(200, json={"loaded": True})
            if request.url.path == "/readyz":
                return httpx.Response(200, json=self.ready)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": self.config["served_model_name"]}]})
            if request.url.path == "/v1/chat/completions":
                return httpx.Response(200, json={"model": self.config["served_model_name"],
                    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": '{"ok":true}'}}]})
            return httpx.Response(404)
        return httpx.MockTransport(handler)

    async def test_exact_identity_loaded_warmup_and_zero_adapter_required(self):
        async with UntunedModelProvider(self.config, media_root=Path("."), transport=self.transport()) as provider:
            await provider.check_ready(deadline=time.monotonic() + 5)
            self.assertEqual(provider.serving_attestation["arm"], "qwen38_untuned")
            self.assertIsNone(provider.serving_attestation["adapter_sha256"])
            self.assertEqual(provider.config.expected_runtime_identity, ())
            self.assertTrue(provider.config.require_readiness)

    async def test_every_identity_pin_and_warmth_failure_is_rejected(self):
        original = copy.deepcopy(self.ready)
        changes = (("model_revision", "1" * 40), ("processor_revision", "1" * 40),
            ("runtime_profile_sha256", "f" * 64), ("tokenizer_sha256", "f" * 64),
            ("image_sha256", "f" * 64), ("config_sha256", "f" * 64),
            ("adapter_sha256", "f" * 64), ("adapter_loaded_tensor_count", 1),
            ("prewarm_completed", False), ("schema_smoke_completed", False),
            ("namespace_default_route_absent", False), ("runtime_load", None),
            ("diagnostic_only", 1), ("trained", 0))
        for key, value in changes:
            with self.subTest(key=key):
                self.ready = {**original, key: value}
                async with UntunedModelProvider(self.config, media_root=Path("."), transport=self.transport()) as provider:
                    with self.assertRaises(ModelUnavailable):
                        await provider.check_ready(deadline=time.monotonic() + 5)
                    self.assertIsNone(provider.serving_attestation)

    async def test_absent_null_adapter_field_is_still_incomplete_identity(self):
        del self.ready["adapter_sha256"]
        async with UntunedModelProvider(self.config, media_root=Path("."), transport=self.transport()) as provider:
            with self.assertRaises(ModelUnavailable):
                await provider.check_ready(deadline=time.monotonic() + 5)

    async def test_default_provider_does_not_accept_untuned_diagnostic_readiness(self):
        config = LocalModelConfig(media_root=Path("."), base_url="http://127.0.0.1:8000/v1",
                                 model=self.config["served_model_name"], expected_revision=REVISION)
        async with LocalModelProvider(config, transport=self.transport()) as provider:
            with self.assertRaises(ModelUnavailable):
                await provider.check_ready(deadline=time.monotonic() + 5)

    async def test_pipeline_injection_preserves_recipe_and_does_not_read_trained_config(self):
        from forte.config import WorkerSettings
        from serving.qwen38_untuned_provider import make_untuned_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = WorkerSettings(database_url="sqlite://", data_dir=root / "runtime",
                model_base_url="http://127.0.0.1:1/v1", model_name="baseline-alias", model_api_key=None,
                worker_id="synthetic", heartbeat_file=root / "heartbeat", rules_path=Path("rules/catalog.json"),
                taxonomy_path=Path("annotations/taxonomy.v1.json"), model_manifest_path=Path("models/manifest.json"),
                model_candidate_launch_config_path=root / "unavailable-trained-config.json",
                trusted_scene_context="bank_cash_workstation", compact_frame_ids=True,
                concise_observations=True, staged_screening=True)
            pipeline = make_untuned_pipeline(settings, self.config, transport=self.transport())
            try:
                self.assertIsInstance(pipeline.provider.delegate, UntunedModelProvider)
                self.assertEqual(settings.model_name, "baseline-alias")
                self.assertIsNotNone(settings.model_candidate_launch_config_path)
                self.assertIsNone(pipeline.settings.model_candidate_launch_config_path)
                for field in ("window_seconds", "stride_seconds", "sample_fps", "max_frames_per_window",
                              "max_refinement_passes", "compact_frame_ids", "concise_observations",
                              "staged_screening", "max_concurrent_clips", "max_inflight_per_job"):
                    self.assertEqual(getattr(settings, field), getattr(pipeline.settings, field))
                self.assertEqual(pipeline.audit_provenance["arm"], "qwen38_untuned")
                self.assertIsNone(pipeline.audit_provenance["adapter_sha256"])
                await pipeline.provider.check_ready(deadline=time.monotonic() + 5)
            finally:
                await pipeline.aclose()

    async def test_inherited_media_parser_request_and_generation_controls_are_used(self):
        from PIL import Image
        from forte.shared.schemas import ClipRequest, FrameRef

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owned = root / "synthetic_job"
            owned.mkdir()
            path = owned / "frame.jpg"
            Image.new("RGB", (16, 16), (100, 50, 25)).save(path, "JPEG")
            frame = FrameRef(frame_id="synthetic_frame", pts_sec=0, path=path, width=16, height=16,
                             job_id=owned.name, jpeg_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            clip = ClipRequest(job_id=owned.name, clip_id="synthetic_clip", frame_refs=[frame])
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
            async with UntunedModelProvider(self.config, media_root=root, transport=self.transport()) as provider:
                response = await provider.infer(clip, prompt="Synthetic fixture only", response_schema=schema,
                                                deadline=time.monotonic() + 5)
            payload = json.loads(next(request.content for request in self.requests if request.method == "POST"))
            self.assertEqual(payload["max_tokens"], self.config["generation"]["max_new_tokens"])
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False, "preserve_thinking": False})
            self.assertEqual(response.data, {"ok": True})
            self.assertIsNone(response.metrics.serving_attestation["adapter_sha256"])


if __name__ == "__main__":
    unittest.main()
