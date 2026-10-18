"""No GPU: predeclared W2 settings must survive runtime relocation unchanged."""
import json
from pathlib import Path
import tempfile
import unittest

from training.guards import TrainingGateError, file_sha256
from training.temporal.protocol import SETTINGS, validate_protocol_config, require_protocol_supervision


class W2ProtocolTests(unittest.TestCase):
    def test_settings_and_protocol_bytes_are_independently_pinned(self):
        original = Path("training/temporal/w2-protocol.v1.json")
        protocol = json.loads(original.read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            path.write_bytes(original.read_bytes())
            config = {key: protocol[key] for key in SETTINGS}
            config.update(protocol_manifest=str(path), expected_protocol_manifest_sha256=file_sha256(path), initial_checkpoint_dir="/runs/W1")
            self.assertEqual(validate_protocol_config(config)["max_steps"], 392)
            changed = {**config, "coarse_anchor_weight": .75}
            with self.assertRaisesRegex(TrainingGateError, "weak_predeclared_protocol_settings_changed"):
                validate_protocol_config(changed)
            path.write_text("{}")
            with self.assertRaisesRegex(TrainingGateError, "weak_protocol_manifest_pin_mismatch"):
                validate_protocol_config(config)

    def test_original_W1_config_does_not_gain_new_requirements(self):
        self.assertIsNone(validate_protocol_config({"experiment_id": "W1"}))

    def test_W2_cannot_silently_fall_back_to_binary_only_targets(self):
        with self.assertRaisesRegex(TrainingGateError, "weak_W2_declared_supervision_not_available"):
            require_protocol_supervision({"experiment_id": "W2"}, {"recording_bags": 28, "provided_coarse_positive_anchor_bags": 15})
        require_protocol_supervision({"experiment_id": "W2"}, {"recording_bags": 28, "provided_coarse_positive_anchor_bags": 15,
                                                               "silver_frame_fact_labels": 8, "silver_support_negative_labels": 4})


if __name__ == "__main__":
    unittest.main()
