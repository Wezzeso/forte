"""Synthetic checks only; no provided recordings, labels, or model weights are read."""
from __future__ import annotations

import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from training.data import completion_labels, video_metadata
from training.guards import (TrainingGateError, canonical_sha256, locked_records,
                             local_file, validate_target_scope, validate_training_inputs)


def fixtures():
    from forte.decision import PROMPT_VERSION, CONFIG_VERSION
    split_rows = [
        {"sample_id": "s-train", "content_sha256": "a" * 64, "group_id": "g-train", "split": "train"},
        {"sample_id": "s-validation", "content_sha256": "b" * 64, "group_id": "g-val", "split": "validation"},
        {"sample_id": "s-final", "content_sha256": "c" * 64, "group_id": "g-final", "split": "final_check"},
    ]
    split = {"state": "locked", "locked": True, "training_authorized": True,
             "scenario_groups_reviewed": True, "approved_by": "synthetic-reviewer",
             "locked_at": "2026-10-17T00:00:00Z", "source_manifest_sha256": "d" * 64,
             "records": split_rows, "records_sha256": canonical_sha256(sorted(split_rows, key=lambda r: r["sample_id"]))}
    frames = [{"path": "first.png", "frame_id": "f1", "pts_sec": 0.25},
              {"path": "last.png", "frame_id": "f2", "pts_sec": 1.75}]
    review = {"content_sha256": "a" * 64, "group_id": "g-train", "split": "train", "review_state": "reviewed",
              "split_lock_verified": True, "split_manifest_sha256": "e" * 64,
              "source_role": "organizer_development", "full_review_complete": True,
              "review_status": "human_reviewed", "reviewer_id": "synthetic-reviewer", "reviewed_at": "2026-10-17",
              "frames": frames, "events": [], "negative_windows": [{"start_sec": 0, "end_sec": 2,
              "negative_verified": True, "context_complete": True, "visibility": "clear"}], "unresolved_items": []}
    record = {"sample_id": "w1", "source_recording_id": "s-train", "content_sha256": "a" * 64,
              "group_id": "g-train", "split": "train",
              "data_origin": "development", "annotation_record_sha256": canonical_sha256(review),
              "review_status": "human_reviewed", "reviewer_id": "synthetic-reviewer", "reviewed_at": "2026-10-17",
              "start_sec": 0, "end_sec": 2, "negative_verified": True, "frames": frames,
              "requested_fact_ids": ["synthetic_fact"], "reviewed_event_ids": [], "prompt": "Return requested facts.",
              "prompt_version": PROMPT_VERSION, "config_version": CONFIG_VERSION,
              "target": {"facts": [{"fact_id": "synthetic_fact", "state": "absent", "supporting_frame_ids": ["f1"]}],
                         "candidates": []}}
    manifest = {"schema_version": "forte.training.v1", "split_manifest_sha256": "e" * 64,
                "annotations_sha256": "f" * 64, "approved_by": "synthetic-reviewer", "records": [record]}
    return manifest, split, [review]


class GuardTests(unittest.TestCase):
    def validate(self, manifest, split, reviews):
        return validate_training_inputs(manifest, split, reviews, split_hash="e" * 64,
                                        annotations_hash="f" * 64, media_root=Path.cwd(), verify_frames=False)

    def assert_blocked(self, code, manifest, split, reviews):
        with self.assertRaisesRegex(TrainingGateError, code):
            self.validate(manifest, split, reviews)

    def test_synthetic_reviewed_train_passes(self):
        self.assertEqual(len(self.validate(*fixtures())), 1)

    def test_unlocked_split_is_blocked(self):
        manifest, split, reviews = fixtures()
        split["state"] = "proposed_unlocked"
        self.assert_blocked("split_not_locked", manifest, split, reviews)

    def test_duplicate_hash_leak_is_blocked(self):
        manifest, split, reviews = fixtures()
        split["records"][1]["content_sha256"] = "a" * 64
        split["records_sha256"] = canonical_sha256(sorted(split["records"], key=lambda r: r["sample_id"]))
        self.assert_blocked("duplicate_content_leakage", manifest, split, reviews)

    def test_scenario_leak_is_blocked(self):
        manifest, split, reviews = fixtures()
        split["records"][1]["group_id"] = "g-train"
        split["records_sha256"] = canonical_sha256(sorted(split["records"], key=lambda r: r["sample_id"]))
        self.assert_blocked("scenario_group_leakage", manifest, split, reviews)

    def test_final_check_training_is_blocked(self):
        manifest, split, reviews = fixtures()
        manifest["records"][0]["content_sha256"] = "c" * 64
        self.assert_blocked("training_partition_forbidden", manifest, split, reviews)

    def test_jury_data_is_blocked(self):
        manifest, split, reviews = fixtures()
        manifest["records"][0]["data_origin"] = "jury"
        self.assert_blocked("jury_or_unknown_origin_forbidden", manifest, split, reviews)

    def test_teacher_label_is_not_human_review(self):
        manifest, split, reviews = fixtures()
        reviews[0]["review_status"] = "teacher_provisional"
        self.assert_blocked("source_review_incomplete", manifest, split, reviews)

    def test_ambiguous_positive_intervals_are_not_negative(self):
        manifest, split, reviews = fixtures()
        reviews[0]["negative_windows"] = []
        manifest["records"][0]["annotation_record_sha256"] = canonical_sha256(reviews[0])
        self.assert_blocked("negative_window_unverified", manifest, split, reviews)

    def test_unreviewed_derived_frame_is_blocked(self):
        manifest, split, reviews = copy.deepcopy(fixtures())
        reviews[0]["frames"] = []
        manifest["records"][0]["annotation_record_sha256"] = canonical_sha256(reviews[0])
        self.assert_blocked("training_frames_not_in_review", manifest, split, reviews)

    def test_unsupported_target_frame_is_blocked(self):
        manifest, split, reviews = fixtures()
        manifest["records"][0]["target"]["facts"][0]["supporting_frame_ids"] = ["invented"]
        self.assert_blocked("target_fact_support_invalid", manifest, split, reviews)

    def test_prompt_loss_and_visual_loss_are_masked(self):
        self.assertEqual(completion_labels([1, 9, 9, 2, 3], [1, 9, 9], {9}), [-100, -100, -100, 2, 3])

    def test_processor_prefix_mismatch_is_blocked(self):
        with self.assertRaisesRegex(TrainingGateError, "processor_prefix_alignment_failed"):
            completion_labels([1, 9, 2], [1, 8], {9})

    def test_visual_token_in_completion_is_blocked(self):
        with self.assertRaisesRegex(TrainingGateError, "completion_contains_visual_token"):
            completion_labels([1, 2, 9], [1, 2], {9})

    def test_vfr_pts_survive_metadata(self):
        frames = [{"pts_sec": 0.3}, {"pts_sec": 1.7}, {"pts_sec": 2.125}]
        metadata = video_metadata(frames)
        self.assertEqual([index / metadata["fps"] for index in metadata["frames_indices"]], [0.3, 1.7, 2.125])

    def test_media_root_and_urls_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(TrainingGateError, "media_path_invalid"):
                local_file("https://example.invalid/video", root)
            with self.assertRaisesRegex(TrainingGateError, "media_outside_approved_root"):
                local_file("../escape.png", root)

    def test_split_digest_matches_a10(self):
        from evaluation.audit import records_digest
        _, split, _ = fixtures()
        self.assertEqual(records_digest(split["records"]), split["records_sha256"])
        locked_records(split)

    def test_isolated_runtime_metadata_check_never_initializes_cuda(self):
        from training.runtime_check import verify_runtime
        from training.guards import load_json
        lock_path = Path(__file__).parent / "runtime-lock.json"
        lock = load_json(lock_path)
        versions = {**lock["core_packages"], **lock["addon_packages"]}
        with patch("training.runtime_check.sys.platform", "linux"), \
             patch("training.runtime_check.sys.version_info", (3, 12, 3)), \
             patch("training.runtime_check.importlib.metadata.version", side_effect=versions.__getitem__):
            result = verify_runtime(lock_path)
        self.assertFalse(result["cuda_initialized"])
        self.assertFalse(result["model_weights_opened"])

    def test_newer_cuda13_stack_fails_training_runtime_lock(self):
        from training.runtime_check import verify_runtime
        with patch("training.runtime_check.sys.platform", "linux"), \
             patch("training.runtime_check.sys.version_info", (3, 12, 3)), \
             patch("training.runtime_check.importlib.metadata.version", return_value="2.14.1"):
            with self.assertRaisesRegex(TrainingGateError, "training_runtime_dependency_version_mismatch"):
                verify_runtime(Path(__file__).parent / "runtime-lock.json")

    def test_full_19_fact_training_scope_passes(self):
        validate_target_scope([{"requested_fact_ids": [f"synthetic-{index}" for index in range(19)]}])

    def test_four_fact_screening_is_not_a_training_target(self):
        with self.assertRaisesRegex(TrainingGateError, "training_requires_full_19_fact_targets"):
            validate_target_scope([{"requested_fact_ids": [f"synthetic-{index}" for index in range(4)]}], 4)


if __name__ == "__main__":
    unittest.main()
