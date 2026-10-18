"""Actual synthetic CPU optimizer/checkpoint tests; no assets, media or GPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None


@unittest.skipUnless(HAS_TORCH, "optional isolated Torch runtime required")
class WeakHeadRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(123)

    def inputs(self, frames=7, dim=12):
        import torch
        return dict(features=torch.randn(1, frames, 1, dim, requires_grad=True),
                    view_mask=torch.ones(1, frames, 1, dtype=torch.bool),
                    frame_mask=torch.ones(1, frames, dtype=torch.bool),
                    pts_sec=torch.arange(frames, dtype=torch.float64).unsqueeze(0) / 30,
                    view_roles=torch.zeros(1, frames, 1, dtype=torch.long),
                    normalized_boxes=torch.tensor([0., 0., 1., 1.]).expand(1, frames, 1, 4).clone())

    def test_binary_optimizer_preserves_frozen_features_and_fact_heads(self):
        import torch
        from training.temporal.head import FrozenFeatureWeakMILHead, weak_video_mil_loss
        model = FrozenFeatureWeakMILHead(12, hidden_dim=16, event_count=2)
        values = self.inputs()
        before = model.video_instances.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        outputs = model(**values)
        self.assertEqual(tuple(outputs["frame_logits"].shape), (1, 7, 19, 3))
        self.assertEqual(tuple(outputs["boundary_logits"].shape), (1, 7, 2, 2))
        loss, _ = weak_video_mil_loss(outputs, torch.tensor([1]), values["frame_mask"])
        loss.backward()
        self.assertIsNone(values["features"].grad)
        self.assertIsNone(model.window_states.weight.grad)
        self.assertIsNone(model.frame_states.weight.grad)
        optimizer.step()
        self.assertFalse(torch.equal(model.video_instances.weight, before))

    def test_nan_right_padding_does_not_contaminate_valid_outputs(self):
        import torch
        from training.temporal.head import FrozenFeatureWeakMILHead
        model = FrozenFeatureWeakMILHead(12, hidden_dim=16).eval()
        values = self.inputs()
        original = model(**values)
        padded = {key: value.detach().clone() for key, value in values.items()}
        for key, value in padded.items():
            shape = list(value.shape)
            shape[1] = 3
            tail = torch.zeros(shape, dtype=value.dtype)
            if key in {"features", "pts_sec", "normalized_boxes"}:
                tail[:] = float("nan")
            padded[key] = torch.cat((value, tail), dim=1)
        output = model(**padded)
        self.assertTrue(torch.isfinite(output["video_logit"]).all())
        torch.testing.assert_close(output["video_logit"], original["video_logit"], atol=1e-6, rtol=1e-6)
        padded["pts_sec"][0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid_temporal_valid_timestamp"):
            model(**padded)

    def test_all_masked_sparse_targets_are_zero_and_finite(self):
        import torch
        from training.temporal.head import FrozenFeatureWeakMILHead, sparse_loss
        output = FrozenFeatureWeakMILHead(12, hidden_dim=16, event_count=2)(**self.inputs())
        target = {"window_states": torch.full((1, 19), -100), "frame_states": torch.full((1, 7, 19), -100),
                  "frame_support": torch.full((1, 7, 19), -100), "event_boundaries": torch.full((1, 7, 2, 2), -100)}
        loss, _ = sparse_loss(output, target)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()

    def test_complete_4273_frame_recording_is_one_bag(self):
        import torch
        from training.temporal.head import FrozenFeatureWeakMILHead, weak_video_mil_loss
        from training.temporal.weak_targets import mil_logit_reference
        model = FrozenFeatureWeakMILHead(12, hidden_dim=16)
        inputs = self.inputs(frames=4273)
        outputs = model(**inputs)
        self.assertEqual(tuple(outputs["video_logit"].shape), (1,))
        self.assertEqual(tuple(outputs["video_instance_logits"].shape), (1, 4273))
        expected = mil_logit_reference(outputs["video_instance_logits"][0].detach().tolist())
        self.assertAlmostEqual(float(outputs["video_logit"].detach()[0]), expected, places=5)
        loss, _ = weak_video_mil_loss(outputs, torch.tensor([1]), inputs["frame_mask"])
        loss.backward()
        self.assertIsNone(inputs["features"].grad)

    def test_coarse_anchor_objective_has_no_outside_negative_labels(self):
        import torch
        from training.temporal.coarse_targets import coarse_positive_anchor_loss
        instances = torch.zeros((1, 7), requires_grad=True)
        loss = coarse_positive_anchor_loss({"video_instance_logits": instances}, ((2, 3, 4),), temperature=.25)
        loss.backward()
        self.assertTrue(torch.all(instances.grad[0, [0, 1, 5, 6]] == 0))
        self.assertTrue(torch.all(instances.grad[0, [2, 3, 4]] < 0))

    def test_real_fit_and_safetensors_reload_identical(self):
        import torch
        from training.temporal.head import FrozenFeatureWeakMILHead
        from training.temporal.checkpoint import save_checkpoint, load_checkpoint, sequence_inputs
        from training.temporal.train_weak import fit
        from training.test_temporal import fixtures
        from training.guards import TrainingGateError
        sequence, _, facts = fixtures()
        features = torch.randn(3, 2, 768)
        model = FrozenFeatureWeakMILHead(768, hidden_dim=16, event_count=1)
        metrics = fit(model, [(sequence, features, {}, 1, [])], device="cpu", epochs=2, max_steps=2,
                      learning_rate=.001, weight_decay=.01, teacher_weight=.1, seed=1)
        self.assertEqual(metrics["optimizer_steps"], 2)
        architecture = {"kind": "FrozenFeatureWeakMILHead", "embedding_dim": 768, "hidden_dim": 16,
                        "event_count": 1, "mil_temperature": .25}
        identity = {"encoder_id": "synthetic-encoder", "encoder_revision": "a" * 40,
                    "encoder_artifacts_sha256": "b" * 64, "feature_spec_sha256": "c" * 64,
                    "taxonomy_sha256": "d" * 64, "fact_ids": facts, "event_type_ids": ("synthetic-event",)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint"
            receipt = save_checkpoint(model, path, identity=identity, architecture=architecture,
                                      provenance={"synthetic": True, "human_verified": False})
            restored, bound, metadata = load_checkpoint(path, expected_manifest_sha256=receipt["manifest_sha256"])
            self.assertEqual(bound.head_sha256, receipt["weights_sha256"])
            args = sequence_inputs(sequence, features)
            with torch.inference_mode():
                torch.testing.assert_close(restored(**args)["video_logit"], model(**args)["video_logit"], atol=0, rtol=0)
            (path / "head.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(TrainingGateError, "weak_checkpoint_weights_pin_mismatch"):
                load_checkpoint(path, expected_manifest_sha256=receipt["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
