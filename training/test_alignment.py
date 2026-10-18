"""Synthetic compact-ID training parity; no media files, model or GPU needed."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from forte.decision import DecisionConfig, build_prompt
from forte.shared.schemas import ClipRequest
from training.alignment import provider_record, provider_records
from training.guards import TrainingGateError


def canonical_window():
    request = ClipRequest.model_validate({
        "job_id": "synthetic-job", "clip_id": "synthetic-window", "start_sec": 1.0, "end_sec": 2.0,
        "frame_refs": [{"frame_id": "original-long-frame-a", "pts_sec": 1.0, "path": "a.jpg", "width": 32, "height": 32},
                       {"frame_id": "original-long-frame-b", "pts_sec": 2.0, "path": "b.jpg", "width": 32, "height": 32}],
        "fact_definitions": [{"fact_id": "synthetic-fact", "description": "Visible synthetic criterion."}],
    })
    event_definitions = {"synthetic-event": {"name_ru": "Синтетическое событие",
                         "required_fact_states": {"synthetic-fact": "present"}}}
    bundle = build_prompt(request, reviewed_event_ids=["synthetic-event"], event_definitions=event_definitions)
    return {"sample_id": "synthetic-window", "start_sec": 1.0, "end_sec": 2.0,
            "frames": [{"frame_id": "original-long-frame-a", "pts_sec": 1.0, "path": "a.jpg"},
                       {"frame_id": "original-long-frame-b", "pts_sec": 2.0, "path": "b.jpg"}],
            "requested_fact_ids": ["synthetic-fact"], "reviewed_event_ids": ["synthetic-event"],
            "clip_request": request.model_dump(mode="json"), "event_definitions": event_definitions,
            "prompt": bundle.prompt, "prompt_hash": bundle.prompt_hash,
            "prompt_version": bundle.prompt_version, "config_version": bundle.config_version,
            "target": {"facts": [{"fact_id": "synthetic-fact", "state": "present",
                                   "supporting_frame_ids": ["original-long-frame-a"], "notes": None}],
                       "candidates": [{"candidate_id": "synthetic_candidate", "event_type_id": "synthetic-event",
                                       "start_frame_id": "original-long-frame-a", "end_frame_id": "original-long-frame-b",
                                       "supporting_fact_ids": ["synthetic-fact"],
                                       "supporting_frame_ids": ["original-long-frame-a", "original-long-frame-b"],
                                       "visibility_notes": []}]}}


class AlignmentTests(unittest.TestCase):
    def align(self, row):
        return provider_record(row, config=DecisionConfig(compact_frame_ids=True), media_root=Path.cwd())

    def test_default_is_unchanged_independent_copy(self):
        row = canonical_window()
        result = provider_record(row, config=DecisionConfig(), media_root=Path.cwd())
        self.assertEqual(result, row)
        result["frames"][0]["frame_id"] = "changed-copy"
        self.assertEqual(row["frames"][0]["frame_id"], "original-long-frame-a")

    def test_all_reference_fields_and_prompt_align_without_source_mutation(self):
        row = canonical_window()
        snapshot = copy.deepcopy(row)
        result = self.align(row)
        self.assertEqual(row, snapshot)
        self.assertEqual([frame["frame_id"] for frame in result["frames"]], ["f00", "f01"])
        self.assertEqual(result["target"]["facts"][0]["supporting_frame_ids"], ["f00"])
        candidate = result["target"]["candidates"][0]
        self.assertEqual((candidate["start_frame_id"], candidate["end_frame_id"]), ("f00", "f01"))
        self.assertEqual(candidate["supporting_frame_ids"], ["f00", "f01"])
        self.assertNotIn("original-long-frame-a", result["prompt"])
        self.assertIn('"compact_frame_ids":true', result["prompt"])
        self.assertEqual(result["clip_request"], row["clip_request"])

    def test_authoritative_mapping_called_exactly_once_and_restoration_used(self):
        from forte.decision import compact_frame_request, restore_frame_ids
        with patch("training.alignment.compact_frame_request", wraps=compact_frame_request) as compact, \
             patch("training.alignment.restore_frame_ids", wraps=restore_frame_ids) as restore:
            self.align(canonical_window())
        self.assertEqual(compact.call_count, 1)
        self.assertEqual(restore.call_count, 1)

    def test_missing_frozen_request_is_blocked(self):
        row = canonical_window()
        del row["clip_request"]
        with self.assertRaisesRegex(TrainingGateError, "compact_training_clip_request_missing"):
            self.align(row)

    def test_changed_canonical_prompt_is_blocked(self):
        row = canonical_window()
        row["prompt"] += " modified"
        with self.assertRaisesRegex(TrainingGateError, "compact_training_canonical_prompt_mismatch"):
            self.align(row)

    def test_changed_source_frame_order_is_blocked(self):
        row = canonical_window()
        row["clip_request"]["frame_refs"].reverse()
        with self.assertRaisesRegex(TrainingGateError, "compact_training_frame_binding_mismatch"):
            self.align(row)

    def test_already_aliased_manifest_is_blocked(self):
        row = canonical_window()
        row["compact_frame_ids"] = True
        with self.assertRaisesRegex(TrainingGateError, "training_manifest_must_use_canonical_ids"):
            self.align(row)

    def test_aggregate_fingerprint_distinguishes_selected_modes(self):
        rows = [canonical_window()]
        _, canonical = provider_records(rows, config=DecisionConfig(), media_root=Path.cwd())
        _, compact = provider_records(rows, config=DecisionConfig(compact_frame_ids=True), media_root=Path.cwd())
        self.assertEqual(compact["aligned_windows"], 1)
        self.assertNotEqual(canonical["training_alignment_sha256"], compact["training_alignment_sha256"])
        self.assertEqual(set(compact), {"compact_frame_ids", "concise_observations", "aligned_windows", "training_alignment_sha256"})

    def test_concise_omits_only_optional_fact_notes_in_copy(self):
        from forte.decision.structured import CONCISE_PROMPT_VERSION
        row = canonical_window()
        row["target"]["facts"][0]["notes"] = "Исходное дополнительное наблюдение."
        row["target"]["candidates"][0]["visibility_notes"] = ["Краткое ограничение видимости."]
        snapshot = copy.deepcopy(row)
        result = provider_record(row, config=DecisionConfig(concise_observations=True), media_root=Path.cwd())
        self.assertEqual(row, snapshot)
        self.assertIsNone(result["target"]["facts"][0]["notes"])
        self.assertEqual(result["target"]["facts"][0]["state"], "present")
        self.assertEqual(result["target"]["facts"][0]["supporting_frame_ids"], ["original-long-frame-a"])
        self.assertEqual(result["target"]["candidates"], row["target"]["candidates"])
        self.assertEqual(result["prompt_version"], CONCISE_PROMPT_VERSION)

    def test_concise_only_does_not_alias_frame_ids(self):
        with patch("training.alignment.compact_frame_request") as compact:
            result = provider_record(canonical_window(), config=DecisionConfig(concise_observations=True),
                                     media_root=Path.cwd())
        compact.assert_not_called()
        self.assertEqual([frame["frame_id"] for frame in result["frames"]],
                         ["original-long-frame-a", "original-long-frame-b"])

    def test_compact_and_concise_modes_combine(self):
        row = canonical_window()
        row["target"]["facts"][0]["notes"] = "Optional observation."
        result = provider_record(row, config=DecisionConfig(compact_frame_ids=True, concise_observations=True),
                                 media_root=Path.cwd())
        self.assertIsNone(result["target"]["facts"][0]["notes"])
        self.assertEqual(result["target"]["facts"][0]["supporting_frame_ids"], ["f00"])
        self.assertEqual(result["target"]["candidates"][0]["end_frame_id"], "f01")

    def test_concise_refuses_to_truncate_long_visibility_note(self):
        row = canonical_window()
        row["target"]["candidates"][0]["visibility_notes"] = ["x" * 161]
        with self.assertRaisesRegex(TrainingGateError, "concise_training_visibility_requires_review"):
            provider_record(row, config=DecisionConfig(concise_observations=True), media_root=Path.cwd())
        self.assertEqual(len(row["target"]["candidates"][0]["visibility_notes"][0]), 161)

    def test_concise_refuses_to_drop_additional_visibility_notes(self):
        row = canonical_window()
        row["target"]["candidates"][0]["visibility_notes"] = ["First limitation.", "Second limitation."]
        with self.assertRaisesRegex(TrainingGateError, "concise_training_visibility_requires_review"):
            provider_record(row, config=DecisionConfig(concise_observations=True), media_root=Path.cwd())

    def test_concise_preserves_all_fact_states_and_candidates(self):
        row = canonical_window()
        for fact_id, state, support in (("synthetic-absent", "absent", ["original-long-frame-b"]),
                                        ("synthetic-unclear", "unclear", [])):
            row["clip_request"]["fact_definitions"].append({"fact_id": fact_id, "description": "Synthetic criterion."})
            row["requested_fact_ids"].append(fact_id)
            row["target"]["facts"].append({"fact_id": fact_id, "state": state,
                                           "supporting_frame_ids": support, "notes": "Optional local note."})
        row["prompt"] = build_prompt(ClipRequest.model_validate(row["clip_request"]),
                                     reviewed_event_ids=row["reviewed_event_ids"],
                                     event_definitions=row["event_definitions"]).prompt
        result = provider_record(row, config=DecisionConfig(concise_observations=True), media_root=Path.cwd())
        self.assertEqual([fact["state"] for fact in result["target"]["facts"]], ["present", "absent", "unclear"])
        self.assertEqual([fact["supporting_frame_ids"] for fact in result["target"]["facts"]],
                         [["original-long-frame-a"], ["original-long-frame-b"], []])
        self.assertEqual(len(result["target"]["candidates"]), 1)


if __name__ == "__main__":
    unittest.main()
