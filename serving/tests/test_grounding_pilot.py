"""Generated metadata and fake execution boundary tests, without ML/GPU loads."""
from contextlib import nullcontext
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from forte.shared.schemas import VideoAsset
from forte.video.batches import SourceFrameIdentity, source_frame_id
from training.grounding.contracts import PhysicalQuery, expected_sampled_frame_indices
from serving.grounding_pilot import (GroundingAttempt, GroundingPilot, GroundingPilotConfig,
    GroundingPilotError, MANIFEST_SHA256, MODEL_ID, REVISION, _verify_clock_audit, audit_source_clock, canonical_sha256,
    decode_sampled_source, expected_pair_timestamps, source_clock_values, validate_native_ledger)

ROOT = Path(__file__).resolve().parents[2]


def fixture(path, *, ticks=(0, 100, 300, 600, 900, 1000), origin=5000):
    content = hashlib.sha256(b"generated-placeholder-only").hexdigest()
    asset = VideoAsset(job_id="generated-grounding", owned_path=str(path), content_hash=content,
        duration_sec=1.1, width=224, height=224, fps=6., time_base="1/1000",
        source_start_sec=origin / 1000, frame_count=len(ticks))
    frames = tuple(SourceFrameIdentity(source_frame_id(content, origin + tick, 1, 1000),
        content, asset.job_id, 0, origin + tick, "1/1000", 1, 1000, origin, tick / 1000,
        origin + ticks[index + 1] if index + 1 < len(ticks) else None,
        ticks[index + 1] / 1000 if index + 1 < len(ticks) else None,
        None if index + 1 < len(ticks) else asset.duration_sec,
        224, 224, 0, index == 0) for index, tick in enumerate(ticks))
    return asset, frames


def cfg(**kwargs):
    return GroundingPilotConfig(Path("unused"), Path("unused"), **kwargs)


class GroundingMetadataTests(unittest.TestCase):
    def test_import_and_help_use_no_optional_ml_packages(self):
        result = subprocess.run([sys.executable, "-S", "-c",
            "import json,sys; import serving.grounding_pilot; print(json.dumps([x for x in "
            "('torch','transformers','numpy','av','PIL') if x in sys.modules]))"],
            cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])
        result = subprocess.run([sys.executable, "-S", "-m", "serving.grounding_pilot", "--help"],
            cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertIn("--audit-source", result.stdout)

    def test_bad_profiles_are_rejected_before_allocation(self):
        for changes in ({"target_size": 256}, {"sample_fps": 1.}, {"device": "cpu"},
                        {"process_memory_fraction": .20}, {"max_source_duration_sec": 143.},
                        {"max_new_tokens": 257}, {"min_free_bytes": 8*1024**3}):
            with self.subTest(changes=changes), self.assertRaises(GroundingPilotError):
                cfg(**changes)

    def test_vfr_tick_clock_preserves_actual_nonzero_origin_ratios(self):
        asset, frames = fixture("generated.mov")
        validate_native_ledger(asset, frames)
        indices = expected_sampled_frame_indices(asset, frames)
        self.assertEqual(indices, (0, 3, 5))
        values = source_clock_values(frames, indices, "native_pts_ticks")
        self.assertEqual(values["indices"], (0, 600, 1000))
        self.assertEqual(values["rate"], Fraction(1000))
        self.assertEqual(expected_pair_timestamps(values["indices"], values["rate"]), (.3, 1.))
        with self.assertRaisesRegex(GroundingPilotError, "grounding_vfr_not_supported_by_cfr_profile"):
            source_clock_values(frames, indices, "strict_cfr")

    def test_cfr_ordinals_and_native_ticks_agree_but_keep_distinct_semantics(self):
        asset, frames = fixture("generated.mov", ticks=(0, 200, 400, 600, 800, 1000))
        indices = expected_sampled_frame_indices(asset, frames)
        self.assertEqual(indices, (0, 2, 5))  # Exact .5-second tie chooses .4, not .6.
        cfr = source_clock_values(frames, indices, "strict_cfr")
        ticks = source_clock_values(frames, indices, "native_pts_ticks")
        self.assertEqual(cfr["rate"], Fraction(5))
        self.assertEqual(expected_pair_timestamps(cfr["indices"], cfr["rate"]),
                         expected_pair_timestamps(ticks["indices"], ticks["rate"]))
        self.assertNotIn("fps_num", ticks["proof_fields"])
        self.assertNotIn("source_tick_indices", cfr["proof_fields"])

    def test_source_identity_and_tail_fail_closed(self):
        asset, frames = fixture("generated.mov")
        for altered in ((replace(frames[0], frame_id="9"*64), *frames[1:]),
                        (*frames[:-1], replace(frames[-1], next_source_pts=99999))):
            with self.assertRaises(Exception):
                validate_native_ledger(asset, altered)

    def test_stream_selection_reuses_authoritative_contract_with_original_tail(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "generated.mov"
            path.write_bytes(b"generated-placeholder-only")
            for ticks in ((0, 100, 300, 600, 900, 1000), (0, 200, 400, 600, 800, 1000)):
                asset, frames = fixture(path, ticks=ticks)
                batch = NS(frames=frames, source_complete=True, tail_included=True,
                    frame_pixels=lambda index: memoryview(bytes([index]) * (224*224*3)))
                with patch("forte.video.batches.iter_rgb_batches", return_value=(value for value in [batch])):
                    ledger, indices, pixels = decode_sampled_source(asset, cfg(clock_policy="native_pts_ticks"),
                        deadline=time.monotonic()+2)
                self.assertEqual(ledger, frames)
                self.assertEqual(indices, expected_sampled_frame_indices(asset, frames))
                self.assertEqual(len(pixels), len(indices)*224*224*3)
                self.assertEqual(pixels[-1], len(frames)-1)

    def test_cpu_source_audit_does_not_claim_processor_or_inference_proof(self):
        asset, frames = fixture("generated.mov")
        indices = expected_sampled_frame_indices(asset, frames)
        with patch("serving.grounding_pilot.decode_sampled_source", return_value=(frames, indices, b"generated")):
            audit = audit_source_clock(asset, cfg(clock_policy="native_pts_ticks"), deadline=time.monotonic()+1)
        self.assertFalse(audit["processor_clock_verified"])
        self.assertFalse(audit["model_inference_performed"])
        self.assertFalse(audit["is_cfr"])
        self.assertEqual(audit["native_frame_count"], len(frames))
        self.assertEqual(len(audit["source_ledger_sha256"]), 64)
        self.assertNotIn("pixels", audit)

    def test_actual_processor_audit_receipt_is_hash_and_flags_bound(self):
        audit = {"schema_version": "forte.grounding-processor-clock-audit.v1",
            "transformers_version": "5.17.0", "clock_policy": "native_pts_ticks",
            "processor_clock_sha256": "a"*64, "model_id": MODEL_ID, "processor_revision": REVISION,
            "model_manifest_sha256": MANIFEST_SHA256, "public_processor_assets_sha256": {"config.json": "b"*64},
            "target_runtime_verified": True, "packages": {name: "synthetic" for name in
                ("transformers", "torch", "torchvision", "numpy", "pillow", "tokenizers")},
            "runner_source_sha256": hashlib.sha256((ROOT / "serving/grounding_pilot.py").read_bytes()).hexdigest(),
            "cuda_initialized": False, "model_weights_loaded": False,
            "model_forward_calls": 0, "private_source_files_read": 0}
        audit_args = {"expected_processor_assets": audit["public_processor_assets_sha256"],
                      "expected_runtime_packages": audit["packages"]}
        for name in ("generated_only", "cpu_only", "pixel_values_identical_under_clock_reencoding",
            "video_grid_identical_under_clock_reencoding", "nonzero_origin_verified", "vfr_verified",
            "odd_tail_verified", "processor_timestamps_verified", "rendered_timestamp_tokens_verified"):
            audit[name] = True
        with TemporaryDirectory() as directory:
            path = Path(directory) / "generated-audit.json"
            path.write_text(json.dumps(audit))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            _verify_clock_audit(path, digest, "a"*64, **audit_args)
            for changed in ("b"*64, "c"*64):
                with self.assertRaises(GroundingPilotError):
                    _verify_clock_audit(path, changed, "a"*64, **audit_args)
            audit["vfr_verified"] = False
            path.write_text(json.dumps(audit))
            with self.assertRaises(GroundingPilotError):
                _verify_clock_audit(path, hashlib.sha256(path.read_bytes()).hexdigest(), "a"*64, **audit_args)

    def test_failed_attempt_repr_and_safe_summary_never_reveal_private_payload(self):
        attempt = GroundingAttempt("failed", "grounding_output_invalid_json", None, (), {}, {}, {},
            "generated-private-observation", (123, 456), "stop")
        self.assertNotIn("generated-private", repr(attempt))
        self.assertNotIn("123", json.dumps(attempt.safe_summary()))
        self.assertEqual(attempt.private_dict()["raw_response"], "generated-private-observation")


class _Tensor:
    def __init__(self, values=None, shape=(1, 5)):
        self.values, self.shape = values, shape
    def __getitem__(self, key):
        return _Tensor([11, 12, 2]) if isinstance(key, tuple) else self
    def tolist(self):
        return self.values
    def detach(self):
        return self
    def cpu(self):
        return self


class _Inputs(dict):
    def to(self, device):
        return self


class GroundingProducerTests(unittest.TestCase):
    def run_fake(self, response, *, ledger_pin=None):
        asset, frames = fixture("generated.mov", ticks=(0, 200, 400, 600, 800, 1000))
        indices = expected_sampled_frame_indices(asset, frames)
        pilot = GroundingPilot(cfg())
        pilot._model_load_calls = 1
        pilot._attestation = {"processor_clock_sha256": "a"*64, "input_recipe_sha256": "b"*64}
        pilot._torch = NS(cuda=NS(synchronize=lambda device: None), inference_mode=nullcontext)
        pilot._model = NS(generate=lambda **kwargs: _Tensor(), generation_config=NS(eos_token_id=2))
        def calculate(values, rate, merge):
            return expected_pair_timestamps(tuple(values), Fraction(rate))
        expected = calculate(list(indices), 5., 2)
        expanded = "".join(f"<{stamp:.1f} seconds>" for stamp in expected)
        tokenizer = NS(decode=lambda values, **kwargs: response if isinstance(values, list) else expanded)
        class Processor:
            _calculate_timestamps = staticmethod(calculate)
            def __init__(self):
                self.tokenizer = tokenizer
            def apply_chat_template(self, *args, **kwargs):
                return "synthetic-video-template"
            def __call__(self, **kwargs):
                return _Inputs(input_ids=_Tensor(), pixel_values_videos=_Tensor(),
                    video_grid_thw=_Tensor([[2, 14, 14]]))
        pilot._processor = Processor()
        fake_numpy = NS(frombuffer=lambda *args, **kwargs: NS(reshape=lambda *shape: NS(copy=lambda: object())), uint8=object())
        fake_transformers = NS(StoppingCriteria=object, StoppingCriteriaList=list)
        fake_video = NS(VideoMetadata=lambda **kwargs: NS(**kwargs))
        query = PhysicalQuery(query_id="CASH_COUNT_VISIBLE", text="Visible hands count banknotes.")
        with (patch.dict(sys.modules, {"numpy": fake_numpy, "transformers": fake_transformers,
                                      "transformers.video_utils": fake_video}),
              patch("forte.video.probe.content_sha256", return_value=asset.content_hash),
              patch("pathlib.Path.is_file", return_value=True),
              patch("pathlib.Path.stat", return_value=NS(st_size=100)),
              patch("serving.grounding_pilot.decode_sampled_source", return_value=(frames, indices, b"generated")),
              patch.object(pilot, "memory_stats", return_value={})):
            return pilot.run(asset, query, deadline=time.monotonic()+2,
                expected_source_ledger_sha256=ledger_pin)

    def test_success_constructs_actual_strict_a08_result_and_stage_counts(self):
        attempt = self.run_fake("[[0.1,0.8]]")
        self.assertEqual(attempt.status, "done", attempt.error_code)
        self.assertEqual(attempt.raw_result.native_frame_count, 6)
        self.assertEqual(len(attempt.raw_result.source_ledger_sha256), 64)
        self.assertEqual(attempt.raw_result.intervals[0].proposed_start_sec, .1)
        self.assertIsNone(attempt.raw_result.legal_verdict)
        self.assertEqual([attempt.attestation[name] for name in
            ("decode_calls", "processor_calls", "generation_calls", "model_load_calls")], [1,1,1,1])
        self.assertEqual(attempt.attestation["source_result_cache_hits"], 0)

    def test_invalid_response_retains_raw_tokens_but_has_no_validated_interval_claim(self):
        attempt = self.run_fake("generated private non-JSON observations")
        self.assertEqual(attempt.status, "failed")
        self.assertEqual(attempt.error_code, "grounding_output_invalid_json")
        self.assertIsNone(attempt.raw_result)
        self.assertEqual(attempt.generated_token_ids, (11,12,2))
        self.assertNotIn("observations", json.dumps(attempt.safe_summary()))

    def test_preflight_ledger_mismatch_never_calls_processor_or_generation(self):
        attempt = self.run_fake("[]", ledger_pin="c"*64)
        self.assertEqual(attempt.error_code, "grounding_preflight_source_ledger_changed")
        self.assertEqual(attempt.attestation["generation_calls"], 0)
        self.assertEqual(attempt.attestation["processor_calls"], 0)

    def test_empty_proposal_remains_unmeasured_without_legal_absence(self):
        attempt = self.run_fake("[]")
        self.assertEqual(attempt.status, "done", attempt.error_code)
        self.assertEqual(attempt.raw_result.proposal_status, "no_grounding_proposal")
        self.assertIsNone(attempt.raw_result.legal_verdict)

    def test_not_loaded_cancelled_and_busy_are_static_failures(self):
        asset, _ = fixture("generated.mov")
        query = PhysicalQuery(query_id="TEST_QUERY", text="Generated marker moves.")
        pilot = GroundingPilot(cfg())
        self.assertEqual(pilot.run(asset, query, deadline=time.monotonic()+1).error_code, "grounding_not_loaded")
        event = threading.Event()
        event.set()
        self.assertEqual(pilot.run(asset, query, deadline=time.monotonic()+1, cancel_event=event).error_code,
                         "grounding_cancelled")
        pilot._lock.acquire()
        try:
            self.assertEqual(pilot.run(asset, query, deadline=time.monotonic()+1).error_code, "grounding_busy")
        finally:
            pilot._lock.release()


if __name__ == "__main__":
    unittest.main()
