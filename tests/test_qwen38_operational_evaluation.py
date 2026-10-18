"""Operational orchestration tests: generated fixtures and no model/network."""

import asyncio
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation.io import digest
from evaluation.qwen38_comparison import ARMS
from evaluation.qwen38_pipeline import ReservedProvider, prepare, require_released_runtime, trace_pipeline_stages
from evaluation.qwen38_server_export import build_server_receipt, native_ledger
from evaluation.qwen38_worker_trace import TracedPipeline
from forte.config import WorkerSettings
from forte.shared.schemas import VideoAsset
from forte.video.decode import frame_identity


def worker_recipe():
    return {"window_seconds": 8.0, "stride_seconds": 4.0, "sample_fps": 2.0,
        "max_frames_per_window": 16, "max_concurrent_jobs": 1, "max_concurrent_clips": 1,
        "max_inflight_per_job": 1, "compact_frame_ids": True, "concise_observations": True,
        "staged_screening": False, "max_refinement_passes": 8, "job_timeout_seconds": 285.0,
        "trusted_scene_context": "bank_cash_workstation"}


class FullPipelinePreparationTests(unittest.TestCase):
    def test_candidate_runtime_image_source_and_base_must_match_release(self):
        pins = {"base_manifest_sha256": "1"*64, "source_archive_sha256": "2"*64, "runtime_image_sha256": "3"*64}
        identity = {"model_id": "Qwen/Qwen3.8-27B", "model_revision": "a"*40, "adapter_sha256": "4"*64}
        proof = {"model_files_verified": True, "model_manifest_sha256": "1"*64,
                 "source_sha256": "2"*64, "image_sha256": "3"*64}
        require_released_runtime(proof, identity, pins)
        for key in ("model_manifest_sha256", "source_sha256", "image_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_released_runtime({**proof, key: "f"*64}, identity, pins)

    def test_exact_six_population_prepares_true_window_plus_refinement_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, records, attempts = [], [], []
            for index in range(6):
                path = root / f"source{index}.mov"
                path.write_bytes(f"invented source identity {index}".encode())
                content = hashlib.sha256(path.read_bytes()).hexdigest()
                row = {"sample_id": f"s{index}", "content_sha256": content, "split": "validation"}
                records.append(row)
                sources.append({**row, "owned_path": str(path)})
                for arm in ARMS:
                    attempts.append({**row, "attempt_id": f"a_{arm}_{index}", "arm": arm, "repeat_index": 0})
            recipe = {"sampling": worker_recipe(), "prompt_sha256": "a" * 64,
                "generation": {"max_new_tokens": 4096, "seed": 20261017}, "evidence": {},
                "timing": {}, "concurrency": 1, "supported_envelope": {}}
            identities = {arm: {"model_id": "synthetic_8b" if arm == "baseline_8b" else "synthetic_27b",
                "model_revision": "1" * 40, "adapter_sha256": "b" * 64 if arm == "qwen38_tuned" else None} for arm in ARMS}
            comparison = {"schema_version": "forte.qwen38_comparison_plan.v1", "records": records,
                "attempts": attempts, "arms": identities, "recipe": recipe, "recipe_sha256": digest(recipe),
                "provided_labels_sha256": "c" * 64}
            spec = {"arm": "qwen38_untuned", "worker": worker_recipe(), "provider": {},
                "runtime_bindings": {}, "sources": sources}
            def fake_probe(path, **kwargs):
                return SimpleNamespace(content_hash=hashlib.sha256(Path(path).read_bytes()).hexdigest(), duration_sec=15.5)
            with patch("evaluation.qwen38_pipeline.probe_video", side_effect=fake_probe):
                plan = prepare(comparison, spec, role_config_sha256="d" * 64)
            self.assertEqual(len(plan["source_attempts"]), 6)
            self.assertEqual(plan["cases"][0]["coarse_windows"], 3)
            self.assertEqual(len(plan["attempts"]), 6 * (3 + 8) * 2)
            self.assertEqual(plan["quality_model_calls_during_prepare"], 0)

    def test_stages_wrap_exact_functions_without_changing_return_values(self):
        frame = SimpleNamespace()
        assembled = SimpleNamespace(frames=[])
        with patch("forte.pipeline.decode_frames", return_value=[frame]) as decode, patch("forte.pipeline.assemble_result", return_value=assembled) as assemble:
            import forte.pipeline as pipeline
            with trace_pipeline_stages() as (counts, evidence):
                self.assertEqual(pipeline.decode_frames("synthetic"), [frame])
                self.assertIs(pipeline.assemble_result(job_id="synthetic"), assembled)
            self.assertIs(pipeline.decode_frames, decode)
            self.assertIs(pipeline.assemble_result, assemble)
            self.assertEqual(dict(counts), {"decode": 1, "evidence": 1})
            self.assertEqual(evidence, [])


class ReservedProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_model_request_consumes_slot_and_unused_slots_remain_visible(self):
        identity = {"model_id": "synthetic", "model_revision": "a" * 40, "adapter_sha256": None}
        class Fake:
            config = SimpleNamespace(max_attempts=1)
            serving_attestation = identity
            async def check_ready(self, **kwargs):
                pass
            async def infer(self, *args, **kwargs):
                return await self._request("POST", "/chat/completions")
            async def _request(self, *args, **kwargs):
                raise RuntimeError("private content must not enter logs")
            async def aclose(self):
                pass
        slots = [{"attempt_id": f"slot{i}"} for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            provider = ReservedProvider(Fake(), slots, guard=lambda: identity,
                progress_path=Path(directory) / "slots.json")
            with self.assertRaises(RuntimeError):
                await provider.infer(object(), deadline=time.monotonic()+5)
            self.assertEqual(provider.calls, 1)
            self.assertEqual([row["status"] for row in provider.rows], ["failed", "not_attempted"])
            self.assertNotIn("private content", (Path(directory) / "slots.json").read_text())


class WorkerTraceTests(unittest.IsolatedAsyncioTestCase):
    async def run_wrapper(self, root, *, include_decode=True):
        import forte.pipeline as pipeline_module
        import forte.model.adapter as adapter_module
        identity = {"model_id": "synthetic", "model_revision": "a"*40, "adapter_sha256": None}
        audit = root / "audit.json"
        audit.write_text('{"synthetic_fixture":true}')
        original_result = SimpleNamespace(json_bytes=b'{"synthetic_result_bytes":true}', metadata_path=audit)
        class FakeProvider:
            def __init__(self, config, *args):
                self.config = config
                self.serving_attestation = identity
            async def _request(self, *args, **kwargs):
                return 200, b"private generated output must remain unprinted"
        class FakePipeline:
            def __init__(self, settings, provider):
                self.raw = provider
                self.provider, self.rules = object(), []
            async def analyze(self, job, **kwargs):
                if include_decode:
                    pipeline_module.decode_frames("fixture")
                adapter_module._prepare_media("fixture")
                await self.raw._request("POST", "/chat/completions", deadline=time.monotonic()+5)
                pipeline_module.assemble_result(job_id=job.job_id)
                return original_result
            async def aclose(self):
                pass
        settings = WorkerSettings(database_url="sqlite://", data_dir=root / "runtime",
            model_base_url="http://127.0.0.1:8000/v1", model_name="Qwen3-VL-8B-Instruct", model_api_key=None,
            worker_id="synthetic", heartbeat_file=root / "heartbeat", rules_path=Path("rules/catalog.json"),
            taxonomy_path=Path("annotations/taxonomy.v1.json"), model_manifest_path=Path("models/manifest.json"),
            max_concurrent_jobs=1, max_concurrent_clips=1, max_inflight_per_job=1)
        slot = {"attempt_id": "synthetic_model_slot", "status": "not_attempted"}
        binding = {"provider_max_attempts": 2, "expected_model_identity": identity, "wrapper_sha256": "1"*64,
            "pipeline_sha256": "2"*64, "trace_source_archive": {"sha256": "3"*64}, "trace_runtime_image_sha256": "4"*64}
        session = SimpleNamespace(binding=binding, settings=settings, root=root, rows=[slot], release={},
            claim_source=lambda job: {"attempt_id": "synthetic_source_attempt"},
            claim_model_slot=lambda source_id: slot, flush_slots=lambda: None)
        job = SimpleNamespace(job_id="synthetic_job", attempt=1, content_hash="5"*64)
        with (patch("evaluation.qwen38_worker_trace.ManifestBoundBaselineProvider", FakeProvider),
              patch("evaluation.qwen38_worker_trace.AnalysisPipeline", FakePipeline),
              patch("evaluation.qwen38_worker_trace.require_released_runtime"),
              patch("forte.pipeline.decode_frames", return_value=[]),
              patch("forte.pipeline.assemble_result", return_value=SimpleNamespace(frames=[])),
              patch("forte.model.adapter._prepare_media", return_value=[])):
            traced = TracedPipeline(settings, session)
            if not include_decode:
                with self.assertRaises(RuntimeError):
                    await traced.analyze(job)
                returned = None
            else:
                returned = await traced.analyze(job)
        witness = json.loads((root / "synthetic_job.attempt-1.json").read_text())
        return returned, original_result, witness, slot

    async def test_wrapper_records_actual_stages_and_preserves_result_object_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            returned, original, witness, slot = await self.run_wrapper(Path(directory))
            self.assertIs(returned, original)
            self.assertEqual(witness["status"], "done")
            self.assertEqual(witness["stage_calls"], {"decode": 1, "preprocessing": 1, "model": 1, "evidence": 1})
            self.assertEqual(slot["status"], "done")
            self.assertFalse(witness["result_bytes_changed_by_wrapper"])
            self.assertNotIn("private generated output", json.dumps(witness))

    async def test_missing_actual_decode_fails_witness_instead_of_backfilling_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            returned, original, witness, slot = await self.run_wrapper(Path(directory), include_decode=False)
            self.assertEqual(witness["status"], "failed")
            self.assertEqual(witness["stage_calls"]["decode"], 0)


class ServerExportTests(unittest.TestCase):
    def fixture(self, root):
        from PIL import Image
        content = "1" * 64
        image = root / "frame.jpg"
        Image.new("RGB", (16, 16), (200, 100, 50)).save(image, "JPEG", quality=95)
        sha = hashlib.sha256(image.read_bytes()).hexdigest()
        asset = VideoAsset(job_id="synthetic_job", owned_path=str(root / "source.mov"), content_hash=content,
            duration_sec=1, width=16, height=16, fps=30, time_base="1/30", source_start_sec=0,
            rotation_deg=0, codec="h264", frame_count=1, stream_index=0)
        frame_id = frame_identity(asset, 0)
        frame = {"frame_id": frame_id, "pts_sec": 0, "path": str(image), "width": 16, "height": 16,
            "job_id": "synthetic_job", "jpeg_sha256": sha, "source_pts": 0, "rotation_deg": 0}
        result = {"job_id": "synthetic_job", "status": "done", "verdict": "violation", "violations": [{"type": "invented fixture",
            "timecode_sec": 0.0, "description": "Invented scalar fixture, no real event truth.",
            "evidence_frames": [{"timestamp_sec": 0.0, "image_base64": base64.b64encode(image.read_bytes()).decode()}]}]}
        raw = json.dumps(result).encode()
        audit = {"asset": asset.model_dump(mode="json"), "phase": "done", "job_id": "synthetic_job",
            "publication_ready": True, "coverage": {"complete": True}, "frames": [frame]}
        identity = {"model_id": "synthetic", "model_revision": "a" * 40, "adapter_sha256": None}
        binding = {"attempt_id": "synthetic_attempt", "job_id": "synthetic_job", "content_sha256": content,
            "conditions_sha256": "b" * 64, "recipe_sha256": "c" * 64, "expected_model_identity": identity}
        witness = {"schema_version": "forte.qwen38-fresh-stage-witness.v1", "status": "done",
            "job_id": "synthetic_job", "source_attempt_id": "synthetic_attempt",
            "result_sha256": hashlib.sha256(raw).hexdigest(), "audit_sha256": digest(audit),
            "fresh_output_directory_created": True, "stage_calls": {"decode": 1, "preprocessing": 1, "model": 1, "evidence": 1},
            "cache_hits": {"decode": 0, "model": 0, "evidence": 0, "whole_result": 0}, "actual_model_identity": identity}
        native = {"content_sha256": content, "origin_pts": 0, "time_base_num": 1, "time_base_den": 30,
            "source_complete": True, "tail_included": True, "decoded_source_frame_count": 1,
            "frames": [{"frame_id": frame_id, "source_pts": 0, "timestamp_sec": 0.0}]}
        return audit, raw, binding, witness, native

    def test_receipt_is_compatible_with_existing_acceptance_binder(self):
        from benchmarks.performance.qwen38_acceptance import bind_server_receipt, evidence_hashes, normalized
        with tempfile.TemporaryDirectory() as directory:
            audit, raw, binding, witness, native = self.fixture(Path(directory))
            receipt = build_server_receipt(audit=audit, result_bytes=raw, binding=binding, witness=witness, native_source=native)
            result = json.loads(raw)
            plan = {"conditions_sha256": binding["conditions_sha256"], "recipe_sha256": binding["recipe_sha256"],
                    "expected_model_identity": binding["expected_model_identity"]}
            terminal = {"attempt_id": binding["attempt_id"], "job_id": binding["job_id"], "content_sha256": binding["content_sha256"],
                "normalized_result_sha256": digest(normalized(result)), "evidence_sha256": evidence_hashes(result)}
            self.assertTrue(bind_server_receipt(plan, terminal, receipt)["server_receipt_verified"])
            self.assertFalse(receipt["client_upload_or_ui_latency_established_by_export"])

    def test_historical_audit_cannot_invent_fresh_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            audit, raw, binding, witness, native = self.fixture(Path(directory))
            del witness["stage_calls"]
            with self.assertRaises(ValueError):
                build_server_receipt(audit=audit, result_bytes=raw, binding=binding, witness=witness, native_source=native)

    def test_processing_witness_or_missing_actual_preprocessing_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            audit, raw, binding, witness, native = self.fixture(Path(directory))
            witness["status"] = "processing"
            with self.assertRaises(ValueError):
                build_server_receipt(audit=audit, result_bytes=raw, binding=binding, witness=witness, native_source=native)
            witness["status"] = "done"
            witness["stage_calls"]["preprocessing"] = 0
            with self.assertRaises(ValueError):
                build_server_receipt(audit=audit, result_bytes=raw, binding=binding, witness=witness, native_source=native)

    def test_result_hash_source_frame_identity_or_bound_attempt_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            audit, raw, binding, witness, native = self.fixture(Path(directory))
            witness["source_attempt_id"] = "another_attempt"
            with self.assertRaises(ValueError):
                build_server_receipt(audit=audit, result_bytes=raw, binding=binding, witness=witness, native_source=native)

    def test_native_ledger_decodes_generated_source_and_preserves_actual_pts(self):
        import av
        from PIL import Image
        from fractions import Fraction
        from forte.video.probe import probe_video
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.mp4"
            with av.open(str(path), "w") as output:
                stream = output.add_stream("mpeg4", rate=4)
                stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
                for index in range(8):
                    image = Image.new("RGB", (32,24), (index*20,20,50))
                    frame = av.VideoFrame.from_image(image)
                    image.close()
                    frame.pts, frame.time_base = index, Fraction(1,4)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            asset = probe_video(path, job_id="synthetic")
            ledger = native_ledger(asset, deadline=time.monotonic()+10)
            self.assertEqual(ledger["decoded_source_frame_count"], 8)
            self.assertTrue(ledger["source_complete"])
            self.assertTrue(ledger["tail_included"])
            self.assertEqual([row["timestamp_sec"] for row in ledger["frames"]], [index/4 for index in range(8)])


if __name__ == "__main__":
    unittest.main()
