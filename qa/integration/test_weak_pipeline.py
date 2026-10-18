"""Synthetic CPU cache→optimizer→checkpoint integration; no media or GPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

HAS_ML = all(importlib.util.find_spec(name) for name in ("torch", "safetensors"))


@unittest.skipUnless(HAS_ML, "isolated optional Torch/safetensors environment required")
class WeakPipelineIntegration(unittest.TestCase):
    def test_dense_native_cache_fit_and_bound_checkpoint_roundtrip(self):
        import torch
        from forte.shared.schemas import VideoAsset
        from forte.video.batches import FrameTransform, SourceFrameIdentity, source_frame_id
        from training.guards import canonical_sha256, file_sha256
        from training.temporal.axes import load_axes
        from training.temporal.cache import write_feature_cache, load_feature_cache
        from training.temporal.checkpoint import save_checkpoint, load_checkpoint, sequence_inputs
        from training.temporal.head import FrozenFeatureWeakMILHead
        from training.temporal.train_weak import fit

        torch.set_num_threads(2)
        torch.manual_seed(902)
        count = 97  # Dense head input is separate from the 96-reference VLM guard.
        asset = VideoAsset(job_id="synthetic-weak-integration", owned_path="synthetic.mov",
            content_hash="a" * 64, duration_sec=count * .04, width=64, height=48,
            time_base="1/1000", source_start_sec=5.)
        frames = tuple(SourceFrameIdentity(
            source_frame_id(asset.content_hash, 5000 + index * 40, 1, 1000),
            asset.content_hash, asset.job_id, 0, 5000 + index * 40, "1/1000", 1, 1000, 5000,
            index * .04, 5000 + (index + 1) * 40 if index + 1 < count else None,
            (index + 1) * .04 if index + 1 < count else None,
            asset.duration_sec if index + 1 == count else None, 64, 48, 0, index == 0)
            for index in range(count))
        encoder_recipe = {"model_id": "synthetic-public-encoder", "model_revision": "b" * 40,
                          "encoder_manifest_sha256": "c" * 64, "synthetic": True}
        encoded = SimpleNamespace(features=torch.randn(count, 768, dtype=torch.float16),
            frames=frames, transforms=tuple(FrameTransform(64, 48, 256, 192, 256, 256, 0, 32, 0) for _ in frames),
            encoder_recipe_sha256=canonical_sha256(encoder_recipe), views_per_frame=1,
            selection_mode="native", segment_id="synthetic-whole-source", delivered_count=count,
            source_complete=True, segment_complete=True, tail_included=True, decoded_source_frame_count=count)
        root = Path(__file__).resolve().parents[2]
        axis = root / "training/temporal/fact-axis.v1.json"
        taxonomy = root / "annotations/taxonomy.v1.json"
        fact_ids, event_ids = load_axes({"fact_axis_manifest": str(axis),
            "expected_fact_axis_manifest_sha256": file_sha256(axis), "taxonomy_path": str(taxonomy),
            "expected_taxonomy_sha256": file_sha256(taxonomy)})
        with tempfile.TemporaryDirectory(prefix="forte-weak-integration-") as temporary:
            folder = Path(temporary)
            cache = write_feature_cache(folder / "cache", asset, [encoded], encoder_recipe,
                {"fixture": "decoder"}, {"mode": "native"}, {"views": "global_only"},
                "synthetic-provisional-group", "train")
            sequence, values, manifest = load_feature_cache(Path(cache["manifest_path"]))
            original = values.clone()
            self.assertEqual(tuple(values.shape), (97, 1, 768))
            self.assertEqual(sequence.frames[-1].source_end_bound_sec, asset.duration_sec)
            model = FrozenFeatureWeakMILHead(768, hidden_dim=8, event_count=len(event_ids))
            before = model.video_instances.weight.detach().clone()
            untouched_fact_head = model.frame_states.weight.detach().clone()
            metrics = fit(model, [(sequence, values, manifest, 1, [], ((3, 4, 5),))],
                device="cpu", epochs=2, max_steps=2, learning_rate=.001, weight_decay=.01,
                teacher_weight=0, coarse_anchor_weight=.5, seed=17)
            self.assertEqual(metrics["optimizer_steps"], 2)
            self.assertFalse(torch.equal(before, model.video_instances.weight))
            torch.testing.assert_close(values, original, atol=0, rtol=0)
            torch.testing.assert_close(model.frame_states.weight, untouched_fact_head, atol=0, rtol=0)
            identity = {"encoder_id": encoder_recipe["model_id"], "encoder_revision": encoder_recipe["model_revision"],
                "encoder_artifacts_sha256": sequence.encoder_manifest_sha256,
                "feature_spec_sha256": manifest["feature_spec_sha256"], "taxonomy_sha256": file_sha256(taxonomy),
                "fact_ids": fact_ids, "event_type_ids": event_ids}
            architecture = {"kind": "FrozenFeatureWeakMILHead", "embedding_dim": 768, "hidden_dim": 8,
                            "event_count": len(event_ids), "mil_temperature": .25}
            receipt = save_checkpoint(model, folder / "checkpoint", identity=identity,
                architecture=architecture, provenance={"synthetic": True, "human_verified": False})
            restored, bound, metadata = load_checkpoint(folder / "checkpoint",
                expected_manifest_sha256=receipt["manifest_sha256"], device="cpu")
            self.assertEqual(bound.fact_ids, fact_ids)
            self.assertFalse(metadata["selected"])
            inputs = sequence_inputs(sequence, values)
            with torch.inference_mode():
                actual, expected = restored(**inputs), model(**inputs)
            self.assertEqual(tuple(actual["frame_logits"].shape), (1, 97, 19, 3))
            torch.testing.assert_close(actual["video_logit"], expected["video_logit"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
