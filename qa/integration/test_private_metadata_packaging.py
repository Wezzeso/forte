"""Synthetic authorization checks for private W2 metadata relocation."""
from copy import deepcopy
import unittest

from qa.integration.package_private_w1 import validate_w2_approval


class W2ApprovalTests(unittest.TestCase):
    def setUp(self):
        names = ("machine_annotations", "runtime_configuration", "question_policy",
                 "teacher_terminal_receipt", "proposal_selection_index", "exposure_registry")
        self.entries = {name: {"sha256": str(index + 1) * 64} for index, name in enumerate(names)}
        self.approval = {"experiment_id": "W2", "compute_released": True,
                         "compute_release_authorization": "root_explicit_W2_synthetic_test",
                         "expected_artifact_sha256": {k: v["sha256"] for k, v in self.entries.items()}}
        self.teacher = self.entries["machine_annotations"]["sha256"]

    def test_exact_new_release_and_all_artifact_pins(self):
        validate_w2_approval(self.approval, self.entries, self.teacher)

    def test_w1_release_cannot_authorize_w2(self):
        for field, value in (("experiment_id", "W1"),
                             ("compute_released", False),
                             ("compute_release_authorization", "root_explicit_W1_only")):
            with self.subTest(field=field):
                approval = deepcopy(self.approval)
                approval[field] = value
                with self.assertRaisesRegex(ValueError, "w2_specific_compute_approval_missing"):
                    validate_w2_approval(approval, self.entries, self.teacher)

    def test_wrong_or_missing_independent_teacher_pin(self):
        for pin in (None, "0" * 64):
            with self.subTest(pin_present=pin is not None):
                with self.assertRaisesRegex(ValueError, "w2_independent_teacher_pin_missing"):
                    validate_w2_approval(self.approval, self.entries, pin)

    def test_missing_selection_or_changed_snapshot_rejected(self):
        entries = deepcopy(self.entries)
        del entries["proposal_selection_index"]
        with self.assertRaisesRegex(ValueError, "w2_required_metadata_missing"):
            validate_w2_approval(self.approval, entries, self.teacher)
        entries = deepcopy(self.entries)
        entries["exposure_registry"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "w2_independent_artifact_pin_mismatch"):
            validate_w2_approval(self.approval, entries, self.teacher)


if __name__ == "__main__":
    unittest.main()
