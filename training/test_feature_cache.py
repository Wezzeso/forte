"""Source/coverage/corruption checks for the private native-frame cache."""
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from forte.shared.schemas import VideoAsset
from forte.video.batches import FrameTransform, SourceFrameIdentity, source_frame_id
from training.guards import TrainingGateError, canonical_sha256
from training.temporal.cache import (SCHEMA, _verify_manifest, load_feature_cache,
                                      sequence_from_source_frames, write_feature_cache)
from training.temporal.extract_features import validate_prepared_inputs


def fixtures(count=3):
    asset = VideoAsset(job_id="synthetic-cache", owned_path="synthetic.mov", content_hash="a" * 64,
        duration_sec=count * .04, width=64, height=48, time_base="1/1000", source_start_sec=5.)
    frames = [SourceFrameIdentity(source_frame_id(asset.content_hash, 5000 + i * 40, 1, 1000),
        asset.content_hash, asset.job_id, 0, 5000 + i * 40, "1/1000", 1, 1000, 5000,
        i * .04, 5000 + (i + 1) * 40 if i + 1 < count else None,
        (i + 1) * .04 if i + 1 < count else None, count * .04 if i + 1 == count else None,
        64, 48, 0, i == 0) for i in range(count)]
    recipes = {"encoder_recipe": {"encoder_manifest_sha256": "b" * 64, "synthetic": True},
               "decoder_recipe": {"synthetic": "decoder"}, "sampling_recipe": {"mode": "native"},
               "crop_recipe": {"views": "global_only"}}
    return asset, frames, recipes


def payload_fixture():
    asset, frames, recipes = fixtures()
    seq, spec, sha = sequence_from_source_frames(asset, frames, group_id="synthetic-group", split="train", **recipes)
    result = {"schema_version": SCHEMA, "content_sha256": seq.content_sha256,
        "group_id": seq.group_id, "split": seq.split, "sequence": asdict(seq),
        "tensor_file": "features.safetensors", "tensor_dtype": "float16", "tensor_shape": [3, 1, 768],
        "feature_spec": spec, "feature_spec_sha256": sha,
        "coverage": {"source_complete": True, "tail_included": True, "selection_mode": "native",
                     "decoded_source_frame_count": 3, "delivered_count": 3}}
    result["payload_sha256"] = canonical_sha256(result)
    return result


def rehash(payload):
    payload["payload_sha256"] = canonical_sha256({k: v for k, v in payload.items() if k != "payload_sha256"})


class CacheMetadataTests(unittest.TestCase):
    def test_nonzero_origin_and_native_tail(self):
        asset, frames, recipes = fixtures()
        seq, spec, sha = sequence_from_source_frames(asset, frames, group_id="synthetic-group", split="train", **recipes)
        self.assertEqual(seq.origin_pts, 5000)
        self.assertEqual(seq.frames[-1].source_end_bound_sec, asset.duration_sec)
        self.assertEqual(sha, canonical_sha256(spec))

    def test_complete_manifest_reconstructs_exact_source(self):
        seq = _verify_manifest(payload_fixture())
        self.assertEqual(seq.frames[1].source_pts, 5040)
        self.assertEqual(seq.frames[1].view_roles, ("global",))

    def test_skipped_native_frame_is_rejected(self):
        asset, frames, recipes = fixtures()
        with self.assertRaisesRegex(TrainingGateError, "cache_native_frame_gap"):
            sequence_from_source_frames(asset, [frames[0], frames[2]], group_id="g", split="train", **recipes)

    def test_spliced_source_is_rejected(self):
        asset, frames, recipes = fixtures()
        frames[1] = replace(frames[1], content_hash="c" * 64)
        with self.assertRaisesRegex(TrainingGateError, "cache_source_binding_mismatch"):
            sequence_from_source_frames(asset, frames, group_id="g", split="train", **recipes)

    def test_missing_real_eof_is_rejected(self):
        asset, frames, recipes = fixtures()
        frames[-1] = replace(frames[-1], source_end_bound_sec=None)
        with self.assertRaisesRegex(TrainingGateError, "cache_original_tail_missing"):
            sequence_from_source_frames(asset, frames, group_id="g", split="train", **recipes)

    def test_mutated_manifest_and_partial_coverage_are_rejected(self):
        payload = payload_fixture()
        payload["split"] = "validation"
        with self.assertRaisesRegex(TrainingGateError, "cache_manifest_digest_mismatch"):
            _verify_manifest(payload)
        payload = payload_fixture()
        payload["coverage"]["tail_included"] = False
        rehash(payload)
        with self.assertRaisesRegex(TrainingGateError, "cache_coverage_incomplete"):
            _verify_manifest(payload)

    def test_hash_consistent_path_escape_is_rejected(self):
        payload = payload_fixture()
        payload["tensor_file"] = "../features.safetensors"
        rehash(payload)
        with self.assertRaisesRegex(TrainingGateError, "cache_tensor_path_invalid"):
            _verify_manifest(payload)

    def test_cache_encoder_identity_cannot_disagree_with_sequence(self):
        payload = payload_fixture()
        payload["sequence"]["encoder_manifest_sha256"] = "d" * 64
        rehash(payload)
        with self.assertRaisesRegex(TrainingGateError, "cache_sequence_recipe_mismatch"):
            _verify_manifest(payload)

    def test_more_than_4096_native_frames_are_preserved(self):
        asset, frames, recipes = fixtures(4273)
        seq, _, _ = sequence_from_source_frames(asset, frames, group_id="g", split="train", **recipes)
        self.assertEqual(len(seq.frames), 4273)


class FeatureInputScopeTests(unittest.TestCase):
    def payload(self):
        asset, _, _ = fixtures()
        asset.owned_path = "/private/sources/" + asset.content_hash + ".mov"
        payload = {"schema_version": "forte.private-feature-inputs.v1", "source_mount": "/private/sources",
            "source_count": 1, "final_check_excluded": True, "organizer_labels_in_model_inputs": False,
            "records": [{"asset": asset.model_dump(mode="json"), "content_sha256": asset.content_hash,
                "source_file": asset.content_hash + ".mov", "group_id": "synthetic-group", "split": "train",
                "source_bytes": 123}]}
        payload["payload_sha256"] = canonical_sha256(payload)
        return payload

    def test_hash_named_train_input_is_valid(self):
        self.assertEqual(len(validate_prepared_inputs(self.payload())), 1)

    def test_final_check_cannot_enter_extraction_inputs(self):
        payload = self.payload()
        payload["records"][0]["split"] = "final_check"
        rehash(payload)
        with self.assertRaisesRegex(TrainingGateError, "feature_input_assignment_invalid"):
            validate_prepared_inputs(payload)

    def test_source_path_cannot_be_replaced_by_a_friendly_name(self):
        payload = self.payload()
        payload["records"][0]["asset"]["owned_path"] = "/private/sources/synthetic-original-name.mov"
        rehash(payload)
        with self.assertRaisesRegex(TrainingGateError, "feature_input_source_binding_invalid"):
            validate_prepared_inputs(payload)

    def test_cross_partition_group_cannot_enter_cache(self):
        payload = self.payload()
        second = dict(payload["records"][0])
        second["asset"] = dict(second["asset"], content_hash="c" * 64,
                               owned_path="/private/sources/" + "c" * 64 + ".mov")
        second.update(content_sha256="c" * 64, source_file="c" * 64 + ".mov", split="validation")
        payload["records"].append(second)
        payload["source_count"] = 2
        rehash(payload)
        with self.assertRaisesRegex(TrainingGateError, "feature_input_assignment_invalid"):
            validate_prepared_inputs(payload)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                     "optional actual Torch/safetensors runtime required")
class CacheTensorTests(unittest.TestCase):
    def test_real_roundtrip_corruption_and_partial_stream(self):
        import torch
        asset, frames, recipes = fixtures()
        batch = SimpleNamespace(features=torch.zeros(3, 768, dtype=torch.float16), frames=tuple(frames),
            transforms=tuple(FrameTransform(64, 48, 256, 192, 256, 256, 0, 32, 0) for _ in frames),
            encoder_recipe_sha256=canonical_sha256(recipes["encoder_recipe"]), views_per_frame=1,
            selection_mode="native", segment_id="synthetic-source", delivered_count=3,
            source_complete=True, segment_complete=True, tail_included=True, decoded_source_frame_count=3)
        with tempfile.TemporaryDirectory(prefix="forte-cache-test-") as root:
            destination = Path(root) / "complete"
            entry = write_feature_cache(destination, asset, [batch], group_id="g", split="train", **recipes)
            seq, values, manifest = load_feature_cache(Path(entry["manifest_path"]))
            self.assertEqual(len(seq.frames), 3)
            self.assertEqual(tuple(values.shape), (3, 1, 768))
            self.assertFalse(manifest["feature_coverage_implies_visibility"])
            with self.assertRaisesRegex(TrainingGateError, "cache_destination_already_exists"):
                write_feature_cache(destination, asset, [batch], group_id="g", split="train", **recipes)
            tensor = destination / "features.safetensors"
            raw = bytearray(tensor.read_bytes()); raw[-1] ^= 1; tensor.write_bytes(raw)
            with self.assertRaisesRegex(TrainingGateError, "cache_tensor_digest_mismatch"):
                load_feature_cache(destination / "manifest.json")
            batch.source_complete = False
            partial = Path(root) / "partial"
            with self.assertRaisesRegex(TrainingGateError, "cache_coverage_incomplete"):
                write_feature_cache(partial, asset, [batch], group_id="g", split="train", **recipes)
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
