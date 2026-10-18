"""Synthetic coarse-anchor scope, Unicode, duplicate and timing checks."""
import copy
import unittest

import test_reviewed_split as fixture_module
from ingest_coarse_anchors import build_anchor_exports


class CoarseAnchorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ReviewedSplitTests("test_complete_review_is_candidate_without_authority")
        self.fixture.setUp()
        _, self.split, _ = self.fixture.build()
        self.original, self.audited, anchors = [], [], {}
        for index, record in enumerate(self.fixture.records):
            positive = self.fixture.rows[index]["has_violation"]
            name = ("Х-" if index % 2 else "X-") + str(index + 1)
            seconds = float(2 + index % 3) if positive else None
            self.original.append({"row_id": index + 2, "source_filename": name, "timestamp_sec": seconds, "has_violation": positive})
            self.audited.append({"row_id": index + 2, "source_filename": name, "join_state": "matched", "recording_id": record["recording_id"]})
            if positive:
                anchors["X-" + str(index + 1)] = seconds
        self.source = {"schema_version": "forte.user-coarse-anchor-input.v1", "remaining_rows_explicitly_negative": True,
                       "expected_rows": len(self.original), "expected_positive_rows": len(anchors),
                       "positive_anchor_seconds": anchors, "provenance": {"origin": "synthetic-user-message"}}

    def tearDown(self):
        self.fixture.tearDown()

    def build(self):
        return build_anchor_exports(self.source, self.fixture.records, self.original, self.audited, self.split,
                                    source_sha256="a" * 64, workbook_sha256="b" * 64)

    def test_unicode_join_and_duplicate_anchors_preserved(self):
        outputs, report = self.build()
        self.assertTrue(report["ok"])
        self.assertEqual(sum(map(len, outputs.values())), 20)
        self.assertEqual(report["duplicate_contents_with_distinct_coarse_anchors"], 1)

    def test_scope_is_approximate_and_outside_unknown(self):
        outputs, _ = self.build()
        for row in [row for rows in outputs.values() for row in rows]:
            self.assertFalse(row["event_type_known"])
            self.assertFalse(row["boundaries_known"])
            self.assertFalse(row["legal_ground_truth"])
            self.assertIsNone(row["uncertainty_sec"])
            self.assertEqual(row["outside_anchor_interval_label"], "unknown")

    def test_workbook_timestamp_disagreement_rejected(self):
        key = next(iter(self.source["positive_anchor_seconds"]))
        self.source["positive_anchor_seconds"][key] += 1
        with self.assertRaisesRegex(ValueError, "user_anchor_disagrees_with_original_workbook"):
            self.build()

    def test_binary_disagreement_rejected(self):
        self.original[0]["has_violation"] = not self.original[0]["has_violation"]
        with self.assertRaisesRegex(ValueError, "user_binary_table_disagrees_with_original_workbook"):
            self.build()

    def test_anchor_outside_duration_rejected(self):
        index = next(index for index, row in enumerate(self.original) if row["has_violation"])
        self.original[index]["timestamp_sec"] = 100
        self.source["positive_anchor_seconds"]["X-" + str(index + 1)] = 100
        with self.assertRaisesRegex(ValueError, "user_anchor_outside_source_duration"):
            self.build()

    def test_confusable_alias_collision_rejected(self):
        key = next(iter(self.source["positive_anchor_seconds"]))
        self.source["positive_anchor_seconds"][key.replace("X", "Х")] = self.source["positive_anchor_seconds"][key]
        with self.assertRaisesRegex(ValueError, "coarse_anchor_source_identity_or_time_invalid"):
            self.build()

    def test_frozen_assignment_changed_rejected(self):
        self.split["records"][0]["split"] = "changed"
        with self.assertRaisesRegex(ValueError, "coarse_anchor_frozen_assignment_digest_mismatch"):
            self.build()

    def test_validation_and_final_policies_do_not_condition_inference(self):
        outputs, _ = self.build()
        self.assertTrue(all(row["allowed_use"] == "evaluation_only_do_not_condition_inference" for row in outputs["validation"]))
        self.assertTrue(all(row["allowed_use"] == "held_out_do_not_train_or_condition_inference" for row in outputs["final_check"]))


if __name__ == "__main__":
    unittest.main()
