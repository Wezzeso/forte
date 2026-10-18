"""Synthetic automated-first-pass checks; never reads supplied annotations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from .machine import attach_provided_focus, automatic_first_pass, build_annotation_worklist, build_group_proposals, canonical_receipt_sha256, summarize_record
from .store import ReviewError, RevisionConflict
from . import test_proposals as proposal_fixtures


class MachineChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = proposal_fixtures.ProposalChecks("test_import_is_idempotent_by_audit_bytes")
        self.fixture.setUp()
        from forte.shared.schemas import VideoAsset
        from forte.video.decode import frame_identity
        self.fixture.payload["asset"]["time_base"] = "1/30"
        asset = VideoAsset.model_validate(self.fixture.payload["asset"])
        identity = frame_identity(asset, 120)
        self.fixture.payload["frames"][0].update(frame_id=identity, source_pts=120)
        self.fixture.fact["supporting_frame_ids"] = [identity]
        self.fixture.candidate["supporting_frame_ids"] = [identity]
        self.fixture.payload["window_decisions"][0]["frame_ids"] = [identity]
        self.fixture.ingest()
        self.store = self.fixture.store

    def tearDown(self):
        self.fixture.tearDown()

    def record(self):
        return self.store.list_records()[0]

    def summary(self):
        return summarize_record(self.record(), self.store.workspace)

    def direct_target(self, state="present"):
        identity = self.fixture.payload["frames"][0]["frame_id"]
        definition = {"fact_id": "ANCHOR__CASH_VISIBLE", "description": "INDEPENDENT EXACT-FRAME QUESTION: evaluate physical fact CASH_VISIBLE at frame f00, timestamp 4.000000000 seconds ONLY. Synthetic criterion.", "required": True}
        raw = {"frame_id": identity, "pts_sec": 4.0, "fact_id": "CASH_VISIBLE", "state": state, "supporting_frame_ids": [identity] if state != "unclear" else [], "context_frame_ids": [identity], "scope": "exact_anchor_frame", "derivation": "direct_anchor_teacher_question", "localizable_category": "physical_anchor_atom", "human_verified": False, "eligible": True, "question_fact_id": "ANCHOR__CASH_VISIBLE", "question_definition": definition, "provenance": {"prompt_sha256": "c" * 64, "focal_question_sha256": canonical_receipt_sha256(definition)}}
        self.fixture.payload["window_decisions"][0].update(prompt_hash="c" * 64, complete=True, anchor_frame_ids=[identity], frame_fact_targets=[raw])
        return raw

    def test_machine_summary_is_useful_without_human_review_but_is_not_ground_truth(self):
        row = self.summary()
        self.assertEqual(row["review_status"], "machine_consistency_checked")
        self.assertFalse(row["human_reviewed"])
        self.assertFalse(row["training_authorized"])
        self.assertFalse(row["organizer_labels_used"])
        self.assertEqual(len(row["windows"]), 1)
        self.assertIsNone(row["machine_verdict"])
        self.assertIn("attempt_not_successfully_completed", row["limitations"])
        self.assertIn("technical_coverage_incomplete", row["limitations"])

    def test_unknown_fact_vocabulary_is_not_filled_from_binary_or_missing_labels(self):
        row = self.summary()
        window = row["windows"][0]
        self.assertEqual(len(window["target"]["facts"]), 1)
        self.assertEqual(window["requested_fact_ids"], ["CASH_VISIBLE"])
        self.assertIn("partial_fact_vocabulary_not_filled", window["quality_flags"])

    def test_absent_without_full_context_is_explicitly_masked(self):
        self.fixture.fact["state"] = "absent"
        self.fixture.ingest()
        window = self.summary()["windows"][0]
        self.assertEqual(window["observed_facts"][0]["state"], "absent")
        self.assertEqual(window["target"]["facts"][0]["state"], "unclear")
        self.assertFalse(window["fact_supervision_mask"]["CASH_VISIBLE"])
        self.assertEqual(self.summary()["explanation_ru"]["physical_observations"], [])

    def test_readable_what_why_uses_catalog_phrases_and_source_support_only(self):
        self.fixture.fact["notes"] = "synthetic-free-text-must-not-become-explanation"
        self.fixture.ingest()
        explanation = self.summary()["explanation_ru"]
        self.assertEqual(explanation["physical_observations"][0]["fact_id"], "CASH_VISIBLE")
        self.assertEqual(explanation["physical_observations"][0]["supporting_frame_ids"], [self.fixture.payload["frames"][0]["frame_id"]])
        self.assertNotIn("synthetic-free-text", json.dumps(explanation))
        self.assertFalse(explanation["organizer_labels_used_for_explanation"])
        self.assertTrue(all(rule["grounding_status"] == "requires_temporal_and_exception_verification" for event in explanation["events"] for rule in event["rule_candidates"]))

    def test_candidate_times_are_not_rounded_to_unobserved_source_frames(self):
        row = self.summary()
        self.assertEqual(row["windows"][0]["target"]["candidates"], [])
        self.assertEqual(row["events"][0]["anchor_frame_id"], self.fixture.payload["frames"][0]["frame_id"])
        self.assertEqual(row["events"][0]["anchor_sec"], 4)
        self.assertFalse(row["events"][0]["exact_target_endpoints"])

    def test_second_bins_do_not_turn_missing_frames_into_negative_observations(self):
        row = self.summary()
        self.assertEqual(len(row["per_second"]), 10)
        self.assertEqual(row["per_second"][4]["source_pts_sec"], [4.0])
        self.assertEqual(row["per_second"][4]["state"], "partial_machine_observation")
        self.assertEqual(row["per_second"][0]["state"], "unknown_no_sampled_frame")
        self.assertEqual(row["per_second"][0]["event_state"], "abstain")
        self.assertTrue(row["source_pts_identity_checked"])

    def test_exact_prompt_hash_is_bound_to_window_and_never_reconstructed(self):
        self.assertIsNone(self.summary()["windows"][0]["provenance"]["prompt_sha256"])
        window = self.fixture.payload["window_decisions"][0]
        window.update(prompt_hash="a" * 64, prompt_version="synthetic-v1", config_version="synthetic-config")
        self.fixture.ingest()
        row = self.summary()
        self.assertEqual(row["windows"][0]["provenance"]["prompt_sha256"], "a" * 64)
        self.assertEqual(row["windows"][0]["provenance"]["prompt_hash_source"], "original_audit_window")
        self.assertIsNone(row["provenance"]["prompt_sha256"])

    def test_automatic_worklist_pins_assignments_and_excludes_final_check(self):
        record = self.record()
        split = {"records": [{"content_sha256": record["content_hash"], "split": "train", "scenario_group_id": "synthetic-scenario"}], "assignments_sha256": "b" * 64, "state": "automated_provisional"}
        work = build_annotation_worklist([record], [self.summary()], split, "c" * 64)
        self.assertEqual(work["records"][0]["split"], "train")
        self.assertEqual(work["records"][0]["scenario_group_id"], "synthetic-scenario")
        self.assertFalse(work["records"][0]["human_review_required"])
        self.assertEqual(work["assignments_sha256"], "b" * 64)
        split["records"][0]["split"] = "final_check"
        work = build_annotation_worklist([record], [self.summary()], split, "c" * 64)
        self.assertEqual(work["records"], [])
        self.assertEqual(work["excluded_final_check_or_test_sources"], 1)

    def test_latest_attempt_is_not_replaced_by_longer_stale_model_output(self):
        from .machine import _latest_proposal
        old = {"audit_sha256": "a" * 64, "coverage": {"complete": True}, "window_decisions": [{}, {}], "imported_at": "2026-01-01T00:00:00+00:00"}
        current = {"audit_sha256": "b" * 64, "coverage": {"complete": False}, "window_decisions": [{}], "imported_at": "2026-01-02T00:00:00+00:00"}
        self.assertEqual(_latest_proposal({"local_model_proposals": [old, current]}), current)

    def test_inspection_focus_keeps_full_request_fact_semantics(self):
        self.fixture.payload["window_decisions"][0].update(focal_start_sec=4.0, focal_end_sec=4.9, anchor_frame_ids=[self.fixture.payload["frames"][0]["frame_id"]])
        self.fixture.ingest()
        window = self.summary()["windows"][0]
        self.assertEqual(window["start_sec"], 3.0)
        self.assertEqual(window["end_sec"], 5.0)
        self.assertEqual(window["inspection_focus"]["fact_scope"], "full_request_context_with_cited_frame_support")
        self.assertFalse(window["inspection_focus"]["focal_exclusive_labels"])

    def test_explicit_frame_atom_requires_direct_question_and_cited_exact_frame(self):
        raw = self.direct_target()
        self.fixture.ingest()
        targets = self.summary()["windows"][0]["frame_fact_targets"]
        self.assertEqual(targets[0]["state"], "present")
        self.assertTrue(targets[0]["supervision_mask"])
        self.assertEqual(targets[0]["question_definition"], raw["question_definition"])
        raw["fact_id"] = "PROCEDURE_CONTEXT_COMPLETE"
        self.fixture.ingest()
        self.assertNotIn("frame_fact_targets", self.summary()["windows"][0])

    def test_explicit_unclear_is_not_absence_and_has_bound_no_decisive_support_receipt(self):
        raw = self.direct_target("unclear")
        self.fixture.fact.update(state="unclear", supporting_frame_ids=[])
        self.fixture.ingest()
        window = self.summary()["windows"][0]
        target, review = window["frame_fact_targets"][0], window["support_reviews"][0]
        self.assertEqual(target["state"], "unclear")
        self.assertTrue(target["supervision_mask"])
        self.assertEqual(target["supporting_frame_ids"], [])
        self.assertFalse(review["supports_state"])
        self.assertEqual(review["teacher_state"], "unclear")
        self.assertEqual(review["frame_id"], review["focal_frame_id"])
        self.assertEqual(review["negative_reference_kind"], "focal_frame_explicit_unclear")
        self.assertEqual(review["question_definition"], raw["question_definition"])
        self.assertEqual(review["provenance"]["focal_question_sha256"], canonical_receipt_sha256(raw["question_definition"]))

    def test_actual_teacher_question_extractor_interoperates_without_model_call(self):
        from types import SimpleNamespace
        from forte.shared.schemas import ClipRequest, FactDefinition, FactDecision, FrameRef
        from training.teacher_sampling import ANCHOR_PHYSICAL_FACT_IDS, extract_anchor_targets, with_anchor_questions
        anchor = FrameRef.model_validate(self.fixture.payload["frames"][0])
        request = ClipRequest(job_id=anchor.job_id, clip_id="synthetic-direct", start_sec=3, end_sec=5, frame_refs=[anchor], fact_definitions=[FactDefinition(fact_id=fact_id, description="Synthetic generic criterion") for fact_id in ANCHOR_PHYSICAL_FACT_IDS], rule_definitions=[])
        question_request = with_anchor_questions(request, anchor)
        batch = SimpleNamespace(request=question_request, facts=[FactDecision(fact_id="ANCHOR__CASH_VISIBLE", state="unclear")], prompt_hash="c" * 64, prompt_version="synthetic", config_version="synthetic")
        original = self.fixture.payload["window_decisions"][0]
        original.update(frame_fact_targets=extract_anchor_targets(batch, anchor), prompt_hash=batch.prompt_hash, anchor_frame_ids=[anchor.frame_id], complete=True)
        self.fixture.fact.update(state="unclear", supporting_frame_ids=[])
        self.fixture.ingest()
        row = self.summary()
        window = row["windows"][0]
        self.assertTrue(window["frame_fact_targets"][0]["supervision_mask"])
        self.assertFalse(window["support_reviews"][0]["supports_state"])
        from jsonschema import Draft202012Validator
        schema = json.loads(Path(__file__).with_name("machine.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(row)

    def test_focal_question_or_prompt_tampering_produces_no_frame_or_support_labels(self):
        for tamper in ("prompt", "question", "anchor"):
            with self.subTest(tamper=tamper):
                raw = self.direct_target("unclear")
                if tamper == "prompt":
                    raw["provenance"]["prompt_sha256"] = "d" * 64
                elif tamper == "question":
                    raw["question_definition"]["description"] += " changed"
                else:
                    raw["pts_sec"] = 4.1
                self.fixture.ingest()
                window = self.summary()["windows"][0]
                self.assertNotIn("frame_fact_targets", window)
                self.assertNotIn("support_reviews", window)

    def test_unclear_window_support_conflict_does_not_export_contradictory_negative(self):
        self.direct_target("unclear")
        self.fixture.ingest()
        window = self.summary()["windows"][0]
        self.assertEqual(window["frame_fact_targets"][0]["state"], "unclear")
        self.assertEqual(window["support_reviews"][0]["mask_reason"], "direct_anchor_window_support_conflict")
        self.assertFalse(window["support_reviews"][0]["supervision_mask"])
        self.assertIn("direct_anchor_window_support_conflict", window["quality_flags"])

    def test_coarse_provided_focus_is_separate_and_validation_never_conditions_worklist(self):
        source, row = self.record(), self.summary()
        row["split"] = "train"
        manifest = self.fixture.root / "provided-focus.jsonl"
        raw = {"schema_version": "forte.user-coarse-anchor.v1", "content_sha256": source["content_hash"], "recording_ids": source["recording_ids"], "split": "train", "provided_video_label": True, "coarse_anchor_sec": [4.0], "anchor_origin": "user_provided_coarse_table", "anchor_source_sha256": "a" * 64, "assignments_sha256": "b" * 64, "target_scope": "class_agnostic_approximate_event_anchor", "event_type_known": False, "boundaries_known": False, "legal_ground_truth": False}
        work = {"assignments_sha256": "b" * 64, "records": [{"content_sha256": source["content_hash"], "split": "train"}]}
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        attach_provided_focus([row], work, [manifest], [source])
        self.assertEqual(row["provided_focus"]["coarse_anchor_sec"], [4.0])
        self.assertNotIn("provided_video_label", work["records"][0]["provided_focus"])
        self.assertFalse(row["provided_focus"]["event_type_known"])
        raw["split"] = "validation"
        row = self.summary(); row["split"] = "validation"
        work["records"] = [{"content_sha256": source["content_hash"], "split": "validation"}]
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        attach_provided_focus([row], work, [manifest], [source])
        self.assertIn("provided_evaluation_focus", row)
        self.assertNotIn("provided_focus", work["records"][0])
        raw["split"] = "final_check"
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(ReviewError):
            attach_provided_focus([row], work, [manifest], [source])

    def test_changed_frame_bytes_mask_supervision_without_inventing_replacement(self):
        self.fixture.frame.write_bytes(b"changed-synthetic-frame")
        row = self.summary()
        self.assertIn("source_frame_missing_changed_or_unbound", row["limitations"])
        self.assertEqual(row["windows"][0]["target"]["facts"][0]["state"], "unclear")
        self.assertFalse(row["windows"][0]["fact_supervision_mask"]["CASH_VISIBLE"])

    def test_machine_publication_preserves_all_human_and_split_fields(self):
        before = self.record()
        row = self.summary()
        result = self.store.publish_machine_summary(before["sample_id"], row, expected_revision=before["revision"])
        self.assertEqual(result["machine_summaries_updated"], 1)
        after = self.record()
        for key, value in before.items():
            if key not in {"revision", "updated_at"}:
                self.assertEqual(after[key], value)
        self.assertEqual(after["machine_first_pass"]["summary_sha256"], row["summary_sha256"])

    def test_machine_summary_updates_are_idempotent_and_revision_fenced(self):
        before, row = self.record(), self.summary()
        self.store.publish_machine_summary(before["sample_id"], row, expected_revision=before["revision"])
        current = self.record()
        result = self.store.publish_machine_summary(current["sample_id"], self.summary(), expected_revision=current["revision"])
        self.assertEqual(result["machine_summaries_unchanged"], 1)
        row["summary_sha256"] = "1" * 64
        with self.assertRaises(RevisionConflict):
            self.store.publish_machine_summary(before["sample_id"], row, expected_revision=before["revision"])

    def test_machine_authority_cannot_be_mislabeled_human(self):
        row = self.summary()
        row["human_reviewed"] = True
        with self.assertRaises(ReviewError):
            self.store.publish_machine_summary(self.record()["sample_id"], row, expected_revision=self.record()["revision"])

    def test_automatic_groups_cover_all_aliases_with_explicit_unmodeled_fallback(self):
        other = self.fixture.sources / "other.mov"
        other.write_bytes(b"other-synthetic-source")
        self.store.seed([{"recording_id": "other-recording", "content_hash": hashlib.sha256(other.read_bytes()).hexdigest(), "group_id": "other-group", "source_path": str(other), "duration_sec": 8.0, "source_role": "synthetic"}], self.fixture.sources)
        records = self.store.list_records()
        rows = [row for record in records if (row := summarize_record(record, self.store.workspace))]
        groups = build_group_proposals(records, rows, "0" * 64)
        self.assertEqual({alias for group in groups["groups"] for alias in group["recording_ids"]}, {"synthetic-recording", "other-recording"})
        self.assertEqual(groups["provenance"]["exact_content_only_source_count"], 1)
        self.assertFalse(groups["provenance"]["scenario_independence_validated"])
        self.assertEqual(groups, build_group_proposals(list(reversed(records)), list(reversed(rows)), "0" * 64))

    def test_private_machine_exports_do_not_require_or_create_a_split_lock(self):
        manifest = self.fixture.root / "synthetic-manifest.jsonl"
        manifest.write_text("synthetic-manifest", encoding="utf-8")
        result = automatic_first_pass(self.store, manifest, publish_workbench=True)
        self.assertEqual(result["machine_summaries"], 1)
        self.assertEqual(result["human_reviews_added"], 0)
        self.assertEqual(result["training_authorizations_added"], 0)
        self.assertFalse(self.record()["split_lock_verified"])
        output = self.store.workspace / "automatic" / "machine_annotations.jsonl"
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["annotation_origin"], "local_model")

    def test_new_export_track_preserves_consumed_default_byte_pin(self):
        manifest = self.fixture.root / "synthetic-manifest.jsonl"
        manifest.write_text("synthetic-manifest", encoding="utf-8")
        automatic_first_pass(self.store, manifest)
        original = (self.store.workspace / "automatic" / "machine_annotations.jsonl").read_bytes()
        automatic_first_pass(self.store, manifest, export_track="w2")
        self.assertEqual((self.store.workspace / "automatic" / "machine_annotations.jsonl").read_bytes(), original)
        self.assertTrue((self.store.workspace / "automatic" / "w2" / "machine_annotations.jsonl").is_file())
        with self.assertRaises(ReviewError):
            automatic_first_pass(self.store, manifest, export_track="../escape")

    def test_versioned_export_selects_exact_terminal_audit_set(self):
        manifest = self.fixture.root / "synthetic-manifest.jsonl"
        manifest.write_text("synthetic-manifest", encoding="utf-8")
        index = self.fixture.root / "terminal-index.json"
        receipt = {"schema_version": "forte.proposal-artifact-handoff-index.v1", "entries": [{"audit_sha256": self.record()["local_model_proposals"][0]["audit_sha256"], "terminal_verified": True, "artifact_available": True, "terminal_status": "done"}]}
        index.write_text(json.dumps(receipt), encoding="utf-8")
        result = automatic_first_pass(self.store, manifest, export_track="w2", proposal_index=index)
        self.assertEqual(result["machine_summaries"], 1)
        row = json.loads((self.store.workspace / "automatic" / "w2" / "machine_annotations.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(row["proposal_selection_index_sha256"], hashlib.sha256(index.read_bytes()).hexdigest())
        with self.assertRaises(ReviewError):
            automatic_first_pass(self.store, manifest, proposal_index=index)
        receipt["entries"][0]["audit_sha256"] = "f" * 64
        index.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaises(ReviewError):
            automatic_first_pass(self.store, manifest, export_track="w2", proposal_index=index)


if __name__ == "__main__":
    unittest.main()
