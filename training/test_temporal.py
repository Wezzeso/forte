"""Architecture-independent synthetic supervision checks. No ML/GPU imported."""
from __future__ import annotations

from dataclasses import replace
import unittest

from training.guards import TrainingGateError
from training.temporal.targets import (FeatureFrame, FeatureSequence, IGNORE, STATE_IDS,
                                        build_targets, canonical_source_frame_id)


def fid(index):
    return canonical_source_frame_id("a" * 64, (0, 350, 1100)[index], 1, 1000)


def fixtures():
    frames = tuple(FeatureFrame(fid(index), pts, pts / 1000, pts / 1000 + .04,
                               (True, False), ("global", "roi"), ((0, 0, 1, 1), (.1, .1, .8, .8)),
                               next_source_pts=pts + 40)
                   for index, pts in enumerate((0, 350, 1100)))
    sequence = FeatureSequence("a" * 64, "synthetic-group", "train", 1, 1000, 0, 768,
                               "b" * 64, "c" * 64, "d" * 64, "e" * 64, frames)
    facts = tuple(f"synthetic-fact-{index}" for index in range(19))
    window = {"content_sha256": "a" * 64, "group_id": "synthetic-group", "split": "train",
              "review_status": "human_reviewed", "reviewer_id": "synthetic-reviewer", "reviewed_at": "2026-10-17",
              "start_sec": 0.0, "end_sec": 1.2,
              "target": {"facts": [{"fact_id": fact, "state": "unclear", "supporting_frame_ids": []}
                                     for fact in facts], "candidates": []}}
    return sequence, window, facts


class TemporalTargetsTests(unittest.TestCase):
    def build(self, sequence, window, facts):
        return build_targets(sequence, window, fact_ids=facts, event_type_ids=("synthetic-event",))

    def test_explicit_unclear_and_unmarked_frames_are_distinct(self):
        target = self.build(*fixtures())
        self.assertEqual(target.window_states, [STATE_IDS["unclear"]] * 19)
        self.assertTrue(all(value == IGNORE for row in target.frame_states for value in row))
        self.assertTrue(all(value == IGNORE for row in target.frame_support for value in row))

    def test_negative_video_label_never_generates_absent_facts(self):
        sequence, window, facts = fixtures()
        window.update(has_violation=False, negative_verified=True)
        target = self.build(sequence, window, facts)
        self.assertNotIn(STATE_IDS["absent"], target.window_states)

    def test_supported_present_does_not_label_other_frames(self):
        sequence, window, facts = fixtures()
        window["target"]["facts"][0].update(state="present", supporting_frame_ids=[fid(1)])
        target = self.build(sequence, window, facts)
        self.assertEqual([row[0] for row in target.frame_support], [IGNORE, 1, IGNORE])
        self.assertEqual([row[0] for row in target.frame_states], [IGNORE, IGNORE, IGNORE])

    def test_absent_without_visibility_context_remains_masked(self):
        sequence, window, facts = fixtures()
        window["target"]["facts"][0].update(state="absent", supporting_frame_ids=[fid(0)])
        target = self.build(sequence, window, facts)
        self.assertEqual(target.window_states[0], IGNORE)
        self.assertEqual(target.frame_support[0][0], IGNORE)

    def test_reviewed_absent_context_and_explicit_support_negative(self):
        sequence, window, facts = fixtures()
        window["target"]["facts"][0].update(state="absent", supporting_frame_ids=[fid(0)])
        window["fact_context"] = {facts[0]: {"review_status": "human_reviewed", "context_complete": True,
                                           "visibility_adequate": True}}
        window["support_reviews"] = [{"review_status": "human_reviewed", "frame_id": fid(2),
                                      "fact_id": facts[0], "supports_state": False}]
        target = self.build(sequence, window, facts)
        self.assertEqual(target.window_states[0], STATE_IDS["absent"])
        self.assertEqual([row[0] for row in target.frame_support], [1, IGNORE, 0])

    def test_only_exact_reviewed_event_frames_get_boundary_labels(self):
        sequence, window, facts = fixtures()
        window["target"]["facts"][0].update(state="present", supporting_frame_ids=[fid(1)])
        window["target"]["candidates"] = [{"candidate_id": "synthetic-candidate", "event_type_id": "synthetic-event",
          "start_frame_id": fid(0), "end_frame_id": fid(2),
          "supporting_fact_ids": [facts[0]], "supporting_frame_ids": [fid(1)], "visibility_notes": []}]
        target = self.build(sequence, window, facts)
        self.assertEqual([row[0] for row in target.event_boundaries], [[1, IGNORE], [IGNORE, IGNORE], [IGNORE, 1]])
        self.assertEqual(target.coverage()["boundary_negative_labels"], 0)

    def test_nominal_fps_time_substitution_fails(self):
        sequence, _, _ = fixtures()
        broken = replace(sequence.frames[1], pts_sec=.5)
        sequence = replace(sequence, frames=(sequence.frames[0], broken, sequence.frames[2]))
        with self.assertRaisesRegex(TrainingGateError, "feature_source_pts_mismatch"):
            sequence.validate()

    def test_roi_only_features_fail(self):
        sequence, _, _ = fixtures()
        sequence = replace(sequence, frames=tuple(replace(frame, view_mask=(False, True)) for frame in sequence.frames))
        with self.assertRaisesRegex(TrainingGateError, "feature_global_view_missing"):
            sequence.validate()

    def test_final_partition_and_teacher_targets_fail(self):
        sequence, window, facts = fixtures()
        with self.assertRaisesRegex(TrainingGateError, "temporal_training_partition_forbidden"):
            self.build(replace(sequence, split="final_check"), window, facts)
        window["review_status"] = "local_teacher_provisional"
        with self.assertRaisesRegex(TrainingGateError, "temporal_target_not_human_reviewed"):
            self.build(sequence, window, facts)

    def test_support_conflicts_fail_instead_of_overwriting(self):
        sequence, window, facts = fixtures()
        window["target"]["facts"][0].update(state="present", supporting_frame_ids=[fid(1)])
        window["support_reviews"] = [{"review_status": "human_reviewed", "frame_id": fid(1),
                                      "fact_id": facts[0], "supports_state": False}]
        with self.assertRaisesRegex(TrainingGateError, "temporal_annotation_conflict"):
            self.build(sequence, window, facts)

    def test_canonical_frame_identity_matches_existing_decoder_protocol(self):
        from forte.shared.schemas import VideoAsset
        from forte.video.decode import frame_identity
        asset = VideoAsset(job_id="synthetic-job", owned_path="synthetic.mov", content_hash="a" * 64,
                           duration_sec=2.0, width=32, height=32, time_base="1/1000")
        self.assertEqual(frame_identity(asset, 350), fid(1))

    def test_invented_frame_id_is_rejected_at_cache_boundary(self):
        sequence, _, _ = fixtures()
        sequence = replace(sequence, frames=(replace(sequence.frames[0], frame_id="invented"), *sequence.frames[1:]))
        with self.assertRaisesRegex(TrainingGateError, "feature_source_frame_id_mismatch"):
            sequence.validate()

    def test_feature_spec_changes_invalidate_identity(self):
        from training.temporal.feature_spec import build_feature_spec
        recipe = {"encoder_manifest_sha256": "b" * 64, "feature_kind": "pooler_output", "l2_normalization": False}
        spec, first = build_feature_spec(recipe, decoder_recipe_sha256="c" * 64,
                                        sampling_recipe_sha256="d" * 64, crop_recipe_sha256="e" * 64)
        _, crop_changed = build_feature_spec(recipe, decoder_recipe_sha256="c" * 64,
                                             sampling_recipe_sha256="d" * 64, crop_recipe_sha256="f" * 64)
        recipe["l2_normalization"] = True
        _, normalization_changed = build_feature_spec(recipe, decoder_recipe_sha256="c" * 64,
                                                      sampling_recipe_sha256="d" * 64, crop_recipe_sha256="e" * 64)
        self.assertEqual(len({first, crop_changed, normalization_changed}), 3)
        self.assertFalse(spec["encoder_recipe"]["l2_normalization"])

    def test_canonical_head_state_axis_is_shared(self):
        from forte.decision.head_contract import STATE_ORDER, FULL_FACT_COUNT
        self.assertEqual(tuple(STATE_IDS), STATE_ORDER)
        self.assertEqual(len(fixtures()[2]), FULL_FACT_COUNT)

    def test_eof_bound_is_not_fabricated_next_frame(self):
        sequence, _, _ = fixtures()
        last = replace(sequence.frames[-1], next_source_pts=None, next_pts_sec=None, source_end_bound_sec=1.2)
        replace(sequence, frames=(*sequence.frames[:-1], last)).validate()
        last = replace(last, source_end_bound_sec=None)
        with self.assertRaisesRegex(TrainingGateError, "feature_eof_bound_missing"):
            replace(sequence, frames=(*sequence.frames[:-1], last)).validate()


if __name__ == "__main__":
    unittest.main()
