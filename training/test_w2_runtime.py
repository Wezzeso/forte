"""Executable CPU checks for W2 weighting, freezing and pinned warm starts."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None


@unittest.skipUnless(HAS_TORCH, "optional isolated Torch runtime required")
class W2RuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(123)
        torch.set_num_threads(2)

    def targets(self):
        import torch
        target = {"window_states": torch.full((1, 19), -100), "frame_states": torch.full((1, 3, 19), -100),
                  "frame_support": torch.full((1, 3, 19), -100), "event_boundaries": torch.full((1, 3, 2, 2), -100)}
        target["frame_states"][0, 1, 0] = 2
        target["frame_support"][0, 1, 0] = 0
        return target

    def test_weighted_unknown_masks_and_disabled_boundary_gradients(self):
        import torch
        from training.temporal.head import sparse_loss
        outputs = {"window_logits": torch.zeros((1, 19, 3), requires_grad=True),
                   "frame_logits": torch.zeros((1, 3, 19, 3), requires_grad=True),
                   "support_logits": torch.zeros((1, 3, 19), requires_grad=True),
                   "boundary_logits": torch.zeros((1, 3, 2, 2), requires_grad=True)}
        loss, _ = sparse_loss(outputs, self.targets(), component_weights={"window": 1., "frame": 2.5, "support": 2.5, "boundary": 0.})
        self.assertAlmostEqual(float(loss.detach()), 2.5 * (math.log(3) + math.log(2)), places=5)
        loss.backward()
        self.assertIsNone(outputs["boundary_logits"].grad)
        self.assertTrue(torch.all(outputs["frame_logits"].grad[0, 0] == 0))
        self.assertTrue(torch.any(outputs["frame_logits"].grad[0, 1, 0] != 0))
        self.assertTrue(torch.all(outputs["support_logits"].grad[0, 0] == 0))

    def test_real_optimizer_keeps_unsupervised_branch_weights_exact(self):
        import torch
        from training.temporal.head import FrozenFeatureWeakMILHead, weak_video_mil_loss, sparse_loss
        from training.temporal.supervision import freeze_unsupervised_branches
        from training.temporal.checkpoint import sequence_inputs
        from training.test_temporal import fixtures
        sequence, _, _ = fixtures()
        inputs = sequence_inputs(sequence, torch.randn(3, 2, 768))
        model = FrozenFeatureWeakMILHead(768, hidden_dim=16, event_count=2)
        weights = {"window": 1., "frame": 2.5, "support": 2.5, "boundary": 0.}
        flags = freeze_unsupervised_branches(model, {"silver_frame_fact_labels": 1, "silver_support_negative_labels": 1}, weights=weights, teacher_weight=.1)
        self.assertFalse(flags["window_states"]["trainable"])
        self.assertFalse(flags["boundary"]["trainable"])
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=.001, weight_decay=.1)
        output = model(**inputs)
        binary, _ = weak_video_mil_loss(output, torch.tensor([1]), inputs["frame_mask"])
        silver, _ = sparse_loss(output, self.targets(), component_weights=weights)
        (binary + .1 * silver).backward()
        optimizer.step()
        for name in ("window_states.weight", "window_states.bias", "boundary.weight", "boundary.bias"):
            torch.testing.assert_close(model.state_dict()[name], before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.frame_states.weight, before["frame_states.weight"]))
        self.assertFalse(torch.equal(model.support.weight, before["support.weight"]))

    def test_pinned_warm_start_preserves_parent_and_rejects_feature_drift(self):
        import torch
        from training.temporal.head import FrozenFeatureWeakMILHead
        from training.temporal.checkpoint import save_checkpoint
        from training.temporal.train_weak import load_parent_checkpoint
        from training.guards import TrainingGateError
        model = FrozenFeatureWeakMILHead(12, hidden_dim=16, event_count=2)
        architecture = {"kind": "FrozenFeatureWeakMILHead", "embedding_dim": 12, "hidden_dim": 16, "event_count": 2, "mil_temperature": .25}
        identity = {"encoder_id": "synthetic", "encoder_revision": "a"*40, "encoder_artifacts_sha256": "b"*64,
                    "feature_spec_sha256": "c"*64, "taxonomy_sha256": "d"*64,
                    "fact_ids": tuple(f"f{x}" for x in range(19)), "event_type_ids": ("e0", "e1")}
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)/"parent"
            receipt = save_checkpoint(model, destination, identity=identity, architecture=architecture,
                provenance={"metrics": {"optimizer_steps": 336}, "trained_components": {"video_binary_mil": True}})
            config = {"initial_checkpoint_dir": str(destination), "expected_initial_checkpoint_manifest_sha256": receipt["manifest_sha256"]}
            restored, parent = load_parent_checkpoint(config, architecture=architecture, identity=identity)
            self.assertEqual(parent["optimizer_steps"], 336)
            self.assertEqual(parent["head_sha256"], receipt["weights_sha256"])
            torch.testing.assert_close(restored.video_instances.weight, model.video_instances.weight, rtol=0, atol=0)
            with self.assertRaisesRegex(TrainingGateError, "weak_initial_checkpoint_binding_mismatch"):
                load_parent_checkpoint(config, architecture=architecture, identity={**identity, "feature_spec_sha256": "e"*64})


if __name__ == "__main__":
    unittest.main()
