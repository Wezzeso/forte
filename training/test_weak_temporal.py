"""Synthetic weak supervision metadata/MIL checks; no model execution."""
from __future__ import annotations

import unittest
from dataclasses import replace

from training.guards import TrainingGateError, canonical_sha256
from training.test_temporal import fixtures, fid
from training.temporal.targets import IGNORE, build_targets
from training.temporal.weak_targets import (build_weak_supervision, mil_logit_reference,
                                             teacher_windows_from_machine_record)


def provided(sequence, value=True):
    return {"content_sha256": sequence.content_sha256, "group_id": sequence.group_id, "split": "train",
            "provided_video_label": value, "provided_label_source_sha256": "f" * 64}


def teacher(sequence, facts):
    return {"schema_version": "forte.machine-annotations.v1", "annotation_origin": "local_model",
            "review_status": "machine_consistency_checked", "human_reviewed": False,
            "content_sha256": sequence.content_sha256,
            "provenance": {"model_id": "synthetic-local-teacher", "model_revision": "a" * 40,
                           "inference_location": "team_local_service",
                           "model_manifest_sha256": "b" * 64, "prompt_sha256": "c" * 64,
                           "source_proposal_sha256": "d" * 64},
            "windows": [{"window_id": "synthetic-window", "start_sec": .3, "end_sec": 1.2,
                         "frame_ids": [fid(1), fid(2)], "fact_supervision_mask": {facts[0]: True, facts[1]: False},
                         "context_masks": {"continuous_visibility": False, "procedure_complete": False},
                         "target": {"facts": [{"fact_id": facts[0], "state": "present", "supporting_frame_ids": [fid(1)]},
                                                {"fact_id": facts[1], "state": "unclear", "supporting_frame_ids": []}],
                                    "candidates": []}}]}


class WeakTemporalTests(unittest.TestCase):
    def build(self, sequence, facts, record=None, label=True):
        windows = teacher_windows_from_machine_record(sequence, record) if record is not None else []
        return build_weak_supervision(sequence, provided(sequence, label), teacher_windows=windows,
                                      fact_ids=facts, event_type_ids=("synthetic-event",),
                                      expected_label_source_sha256="f" * 64)

    def test_binary_label_alone_creates_no_fact_or_event_targets(self):
        sequence, _, facts = fixtures()
        positive = self.build(sequence, facts, label=True)
        negative = self.build(sequence, facts, label=False)
        self.assertEqual(positive.provided_video_label, 1)
        self.assertEqual(negative.provided_video_label, 0)
        self.assertEqual(positive.teacher_windows, ())
        self.assertEqual(negative.teacher_windows, ())
        self.assertFalse(positive.human_verified)

    def test_coarse_anchor_selects_actual_pts_without_inventing_boundaries(self):
        from training.temporal.coarse_targets import anchor_frame_indices
        sequence, _, _ = fixtures()
        sequence = replace(sequence, frames=sequence.frames[:-1] + (
            replace(sequence.frames[-1], next_pts_sec=None, next_source_pts=None, source_end_bound_sec=1.2),))
        self.assertEqual(anchor_frame_indices(sequence, .5, .2), (1,))
        with self.assertRaisesRegex(TrainingGateError, "weak_coarse_anchor_has_no_observed_frames"):
            anchor_frame_indices(sequence, .8, .05)

    def test_teacher_masks_and_actual_window_scope_are_preserved(self):
        sequence, _, facts = fixtures()
        target = self.build(sequence, facts, teacher(sequence, facts)).teacher_windows[0]
        self.assertEqual(target.sequence_frame_indices, (1, 2))
        self.assertEqual(target.targets.window_states[0], 1)
        self.assertTrue(all(value == IGNORE for value in target.targets.window_states[1:]))
        self.assertEqual([row[0] for row in target.targets.frame_support], [1, IGNORE])
        self.assertFalse(target.human_verified)

    def test_weak_teacher_does_not_pass_human_target_gate(self):
        sequence, _, facts = fixtures()
        normalized = teacher_windows_from_machine_record(sequence, teacher(sequence, facts))[0]
        with self.assertRaisesRegex(TrainingGateError, "temporal_target_not_human_reviewed"):
            build_targets(sequence, normalized, fact_ids=facts, event_type_ids=("synthetic-event",))

    def test_missing_teacher_provenance_is_rejected(self):
        sequence, _, facts = fixtures()
        record = teacher(sequence, facts)
        record["provenance"]["model_revision"] = None
        with self.assertRaisesRegex(TrainingGateError, "weak_teacher_identity_missing"):
            self.build(sequence, facts, record)

    def test_exact_window_prompt_hash_overrides_missing_recording_hash(self):
        sequence, _, facts = fixtures()
        record = teacher(sequence, facts)
        record["provenance"]["prompt_sha256"] = None
        record["windows"][0]["provenance"] = {"prompt_sha256": "c" * 64,
                                                  "prompt_hash_source": "original_audit_window"}
        result = self.build(sequence, facts, record)
        self.assertEqual(len(result.teacher_windows), 1)

    def test_heldout_label_is_forbidden(self):
        sequence, _, facts = fixtures()
        record = provided(sequence)
        record["split"] = "validation"
        with self.assertRaisesRegex(TrainingGateError, "weak_training_partition_forbidden"):
            build_weak_supervision(sequence, record, teacher_windows=[], fact_ids=facts,
                                   event_type_ids=(), expected_label_source_sha256="f" * 64)

    def test_mil_score_is_stable_and_not_average_positive_instance_labelling(self):
        self.assertGreater(mil_logit_reference([-10, -10, 10]), 9)
        self.assertAlmostEqual(mil_logit_reference([3, 3, 3]), 3)
        self.assertLess(mil_logit_reference([-5, -5]), 0)
        self.assertTrue(mil_logit_reference([1000, -1000]) > 999)

    def test_negative_bag_masks_contradictory_event_without_changing_facts(self):
        sequence, _, facts = fixtures()
        record = teacher(sequence, facts)
        record["windows"][0]["target"]["candidates"] = [{"candidate_id": "synthetic-event-proposal",
            "event_type_id": "synthetic-event", "start_frame_id": fid(1), "end_frame_id": fid(2),
            "supporting_fact_ids": [facts[0]], "supporting_frame_ids": [fid(1)], "visibility_notes": []}]
        target = self.build(sequence, facts, record, label=False)
        self.assertEqual(target.masked_teacher_event_conflicts, 1)
        self.assertEqual(target.teacher_windows[0].targets.window_states[0], 1)
        self.assertEqual(target.teacher_windows[0].targets.event_brackets, ())
        self.assertEqual(target.teacher_windows[0].targets.coverage()["boundary_negative_labels"], 0)

    def focal_fixture(self):
        sequence, _, facts = fixtures()
        facts = ("CASH_VISIBLE",) + facts[1:]
        record = teacher(sequence, facts)
        record["windows"][0]["target"]["facts"][0]["supporting_frame_ids"] = [fid(2)]
        definition = {"fact_id": "ANCHOR__CASH_VISIBLE", "description": "Synthetic exact anchor assessment.", "required": True}
        item = {"frame_id": fid(1), "fact_id": "CASH_VISIBLE", "state": "unclear",
                "supporting_frame_ids": [], "context_frame_ids": [fid(1), fid(2)],
                "review_status": "machine_consistency_checked", "scope": "exact_anchor_frame",
                "derivation": "direct_anchor_teacher_question", "localizable_category": "physical_anchor_atom",
                "supervision_mask": True, "human_verified": False, "question_definition": definition,
                "question_fact_id": definition["fact_id"], "provenance": {"prompt_sha256": "c" * 64,
                    "focal_question_sha256": canonical_sha256(definition)}}
        record["windows"][0]["frame_fact_targets"] = [item]
        record["windows"][0]["support_reviews"] = [{**item, "supports_state": False, "teacher_state": "unclear",
            "focal_frame_id": fid(1), "negative_reference_kind": "focal_frame_explicit_unclear"}]
        return sequence, facts, record

    def test_explicit_focal_unclear_is_not_absent_and_can_mark_no_decisive_support(self):
        sequence, facts, record = self.focal_fixture()
        target = self.build(sequence, facts, record).teacher_windows[0].targets
        self.assertEqual([row[0] for row in target.frame_states], [2, IGNORE])
        self.assertEqual([row[0] for row in target.frame_support], [0, 1])
        self.assertEqual(target.window_states[0], 1)

    def test_focal_definition_hash_tamper_fails(self):
        sequence, facts, record = self.focal_fixture()
        record["windows"][0]["frame_fact_targets"][0]["question_definition"]["description"] = "Changed"
        with self.assertRaisesRegex(TrainingGateError, "temporal_silver_frame_state_not_directly_localized"):
            self.build(sequence, facts, record)

    def test_unmarked_negative_support_is_rejected(self):
        sequence, facts, record = self.focal_fixture()
        del record["windows"][0]["support_reviews"][0]["negative_reference_kind"]
        with self.assertRaisesRegex(TrainingGateError, "temporal_silver_support_negative_not_explicit"):
            self.build(sequence, facts, record)

    def test_conflicting_window_and_direct_anchor_support_fails_closed(self):
        sequence, facts, record = self.focal_fixture()
        record["windows"][0]["target"]["facts"][0]["supporting_frame_ids"] = [fid(1)]
        with self.assertRaisesRegex(TrainingGateError, "temporal_annotation_conflict"):
            self.build(sequence, facts, record)

    def test_explicit_conflict_mask_keeps_focal_unclear_without_inventing_window_labels(self):
        sequence, facts, record = self.focal_fixture()
        record["windows"][0]["target"]["facts"][0]["supporting_frame_ids"] = [fid(1)]
        review = record["windows"][0]["support_reviews"][0]
        review.update(supervision_mask=False, mask_reason="direct_anchor_window_support_conflict")
        target = self.build(sequence, facts, record).teacher_windows[0].targets
        self.assertEqual(target.window_states[0], IGNORE)
        self.assertEqual([row[0] for row in target.frame_states], [2, IGNORE])
        self.assertEqual([row[0] for row in target.frame_support], [IGNORE, IGNORE])

    def test_dataset_exact_frame_disagreement_masks_both_receipts_without_mutation(self):
        from copy import deepcopy
        from training.temporal.supervision import mask_cross_window_conflicts, class_coverage
        sequence, facts, first = self.focal_fixture()
        second = deepcopy(first)
        answer = second["windows"][0]["frame_fact_targets"][0]
        answer.update(state="absent", supporting_frame_ids=[fid(1)])
        review = second["windows"][0]["support_reviews"][0]
        review.update(supports_state=True, teacher_state="absent")
        a = self.build(sequence, facts, first).teacher_windows[0]
        b = self.build(sequence, facts, second).teacher_windows[0]
        data, audit = mask_cross_window_conflicts([(sequence, None, {}, 1, [a, b], ())])
        self.assertEqual(audit["conflicting_frame_state_pairs"], 1)
        self.assertEqual(audit["conflicting_support_pairs"], 1)
        self.assertEqual(audit["masked_target_observations"], 4)
        for window in data[0][4]:
            self.assertEqual(window.targets.frame_states[0][0], IGNORE)
            self.assertEqual(window.targets.frame_support[0][0], IGNORE)
        self.assertEqual(a.targets.frame_states[0][0], 2)
        self.assertEqual(a.targets.frame_support[0][0], 0)
        counts = class_coverage(data, fact_ids=facts, event_ids=("synthetic-event",))
        self.assertEqual(counts["cross_window_conflicting_pair_observations"], 0)
        self.assertEqual(counts["unique_direct_frame_fact_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
