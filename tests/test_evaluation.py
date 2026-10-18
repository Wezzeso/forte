"""Behavioral tests use invented scalar events; no private media or labels."""

import copy
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluation.audit import audit_split, records_digest
from evaluation.cli import main
from evaluation.io import write_json
from evaluation.metrics import evaluate, match_events
from evaluation.selection import select_candidate
from evaluation.synthetic import fixtures
from evaluation.exploratory import aggregate as exploratory_aggregate
from evaluation.strict_targets import evaluate_strict_targets
from evaluation.io import digest
from evaluation.weak_supervision import evaluate_weak_binary, provided_label_digest, select_weak_candidate, validate_weak_attestation, verify_provided_evaluation_for_manifest, weak_assignment_digest
from evaluation.coarse_anchors import evaluate_coarse_anchors


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.truth, self.predictions, self.manifest, self.metadata = fixtures()

    def run_report(self, predictions=None, **changes):
        return evaluate(self.truth, self.predictions if predictions is None else predictions, metadata=self.metadata, split_manifest=self.manifest, verified_rule_ids={"synthetic_rule_a"}, **changes)

    def test_failed_and_missing_jobs_remain_in_denominators(self):
        report = self.run_report()
        self.assertEqual(report["completion"]["required"], 8)
        self.assertEqual(report["completion"]["completed"], 5)
        self.assertEqual(report["completion"]["missing_predictions"], 1)
        whole = report["whole_video"]
        self.assertEqual([whole[k] for k in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative")], [2, 1, 0, 2, 2, 1])
        self.assertEqual(whole["confusion_denominator_completed_reviewed"], 5)
        self.assertEqual(whole["effective_positive_recall_including_failures"], 0.5)
        self.assertEqual(whole["effective_correct_rate_including_failures"], 0.5)

    def test_event_mismatches_and_failed_positive_events(self):
        diagnostics = self.run_report()["events"]["by_tolerance"]
        primary = next(row for row in diagnostics if row["tolerance_sec"] == 1.0)
        self.assertEqual([primary[k] for k in ("tp", "fp", "fn")], [2, 2, 3])
        self.assertEqual(primary["matched_anchor_error_sec"]["count"], 2)

    def test_matching_repairs_greedy_cardinality_failure(self):
        truth = [{"type": "a", "anchor_sec": 1}, {"type": "a", "anchor_sec": 2}]
        pred = [{"type": "a", "anchor_sec": 1.5}, {"type": "a", "anchor_sec": 0.5}]
        pairs = match_events(truth, pred, 0.6)
        self.assertEqual({(i, j) for i, j, _ in pairs}, {(0, 1), (1, 0)})

    def test_matching_minimizes_error_and_does_not_duplicate(self):
        truth = [{"type": "a", "anchor_sec": 1}, {"type": "a", "anchor_sec": 2}]
        pred = [{"type": "a", "anchor_sec": 2}, {"type": "a", "anchor_sec": 1}, {"type": "b", "anchor_sec": 1}]
        pairs = match_events(truth, pred, 2)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(sum(error for _, _, error in pairs), 0)

    def test_duplicate_detection_is_false_positive(self):
        truth = [{"type": "a", "anchor_sec": 1}]
        pred = [{"type": "a", "anchor_sec": 1}, {"type": "a", "anchor_sec": 1.1}]
        self.assertEqual(len(match_events(truth, pred, 1)), 1)

    def test_human_review_is_required(self):
        self.truth[0]["review_status"] = "local_teacher_provisional"
        report = self.run_report()
        self.assertEqual(report["ground_truth"]["human_reviewed_resolved"], 7)
        self.assertEqual(report["ground_truth"]["unreviewed_or_unresolved"], 1)

    def test_unresolved_legal_grounding_is_not_assumed_correct(self):
        self.truth[0]["events"][0]["rule_id"] = None
        report = self.run_report()
        self.assertEqual(report["grounding_and_evidence"]["unresolved_truth_rule_mappings"], 1)

    def test_deadline_uses_all_required_samples(self):
        report = self.run_report()
        self.assertEqual(report["completion"]["late_completed"], 1)
        self.assertEqual(report["completion"]["on_time_completed"], 4)
        self.assertEqual(report["completion"]["deadline_completion_rate"], 0.5)
        self.assertEqual(report["completion"]["unprocessed_at_deadline_including_unknown_timing"], 4)
        self.assertEqual(report["whole_video_within_deadline"]["tn"], 1)
        self.assertEqual(report["whole_video_within_deadline"]["unprocessed_negative"], 2)

    def test_unknown_timing_does_not_earn_cutoff_success(self):
        del self.predictions[0]["elapsed_sec"]
        report = self.run_report()
        self.assertEqual(report["completion"]["completed"], 5)
        self.assertEqual(report["completion"]["timing_unknown"], 1)
        self.assertEqual(report["whole_video_within_deadline"]["unprocessed_positive"], 3)

    def test_unreliable_anchor_is_excluded_from_timing_error(self):
        self.truth[0]["events"][0]["anchor_reliable"] = False
        primary = next(row for row in self.run_report()["events"]["by_tolerance"] if row["tolerance_sec"] == 1)
        self.assertEqual(primary["tp"], 2)
        self.assertEqual(primary["matched_anchor_error_sec"]["count"], 1)

    def test_invalid_done_cannot_become_clean_negative(self):
        predictions = copy.deepcopy(self.predictions)
        predictions[0]["verdict"] = "no_violation"
        report = self.run_report(predictions)
        self.assertEqual(report["completion"]["completed"], 4)
        self.assertEqual(report["whole_video"]["unprocessed_positive"], 3)
        predictions[0]["verdict"] = "violation"
        predictions[0]["events"][0]["anchor_sec"] = 11
        self.assertEqual(self.run_report(predictions)["completion"]["completed"], 4)

    def test_empty_denominators_are_null(self):
        report = evaluate([], [], metadata={"inference_kind": "synthetic"})
        self.assertIsNone(report["completion"]["rate"])
        self.assertIsNone(report["whole_video"]["precision_completed"])

    def test_unlocked_manifest_never_authorizes_training(self):
        self.manifest["state"] = "proposed_unlocked"
        audit = audit_split(self.manifest)
        self.assertTrue(audit["integrity_passed"])
        self.assertFalse(audit["training_gate_passed"])

    def test_content_group_and_scenario_leakage_are_detected(self):
        first, second = self.manifest["records"][:2]
        second.update(split="train", content_sha256=first["content_sha256"], group_id=first["group_id"])
        first["scenario_group_id"] = second["scenario_group_id"] = "scenario_shared"
        audit = audit_split(self.manifest)
        self.assertFalse(audit["integrity_passed"])
        for key in ("content_leakage", "group_leakage", "scenario_leakage", "records_digest_mismatch"):
            self.assertGreater(audit["error_counts"][key], 0)

    def test_child_windows_inherit_recording_split(self):
        first, second = self.manifest["records"][:2]
        second.update(split="train", parent_sample_id=first["sample_id"])
        self.assertEqual(audit_split(self.manifest)["error_counts"]["derived_split_leakage"], 1)

    def test_duplicate_sample_records_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.truth, [self.predictions[0], self.predictions[0]])

    def test_canonical_digest_ignores_record_order(self):
        self.assertEqual(records_digest(self.manifest["records"]), records_digest(list(reversed(self.manifest["records"]))))

    def test_synthetic_report_cannot_select_a_checkpoint(self):
        baseline = self.run_report()
        candidate = copy.deepcopy(baseline)
        decision = select_candidate(baseline, candidate)
        self.assertEqual(decision["selected"], "baseline")
        self.assertIn("baseline:real_local_inference_unmeasured", decision["blockers"])

    def test_comparable_real_reviewed_error_reduction_can_select(self):
        # Pure gate-policy test, not a real model run or a release claim.
        self.metadata["inference_kind"] = "real_local"
        baseline = self.run_report()
        candidate_predictions = copy.deepcopy(self.predictions)
        candidate_predictions[3].update(verdict="no_violation", events=[])
        candidate = self.run_report(candidate_predictions)
        decision = select_candidate(baseline, candidate)
        self.assertTrue(decision["candidate_earned_selection"])
        self.assertIn("video_false_positives", decision["improved_axes"])

    def test_config_change_blocks_apples_to_oranges_selection(self):
        baseline = self.run_report()
        candidate = copy.deepcopy(baseline)
        candidate["provenance"]["conditions_sha256"] = "different_sampling"
        self.assertIn("incomparable_conditions_sha256", select_candidate(baseline, candidate)["blockers"])

    def test_stdout_contains_only_aggregate_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "truth.json", self.truth)
            write_json(root / "pred.json", self.predictions)
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(["evaluate", "--truth", str(root / "truth.json"), "--predictions", str(root / "pred.json"), "--output", str(root / "report.json")])
            self.assertEqual(result, 0)
            self.assertNotIn("synthetic_01", output.getvalue())
            self.assertNotIn("synthetic_a", output.getvalue())

    def test_exploratory_provided_labels_include_failure_without_accuracy_claim(self):
        rows = [{"sample_id": f"s{i}", "content_hash": f"h{i}", "proposed_split": "train"} for i in range(6)]
        selection = {"baseline": rows[0], "burst": rows[1:]}
        labels = {f"h{i}": i % 2 == 0 for i in range(6)}
        terminals = [{"sample_id": "s0", "status": "done", "verdict": "violation", "public_contract_validated": True, "upload_start_to_result_sec": 10}, {"sample_id": "s1", "status": "done", "verdict": "violation", "public_contract_validated": True, "upload_start_to_result_sec": 301}, {"sample_id": "s2", "status": "failed"}]
        report = exploratory_aggregate(selection, terminals, labels)
        self.assertEqual(report["completion"]["completed"], 2)
        self.assertEqual(report["completion"]["failed_or_unprocessed"], 4)
        self.assertEqual(report["whole_video_against_provided_labels"]["tp"], 1)
        self.assertEqual(report["whole_video_against_provided_labels"]["fp"], 1)
        self.assertEqual(report["whole_video_against_provided_labels"]["unprocessed_positive"], 2)
        self.assertEqual(report["whole_video_within_deadline_against_provided_labels"]["unprocessed_negative"], 3)
        self.assertFalse(report["heldout_accuracy_claim"])
        self.assertFalse(report["human_reviewed_ground_truth"])

    def test_exploratory_rejects_heldout_and_duplicate_content(self):
        rows = [{"sample_id": f"s{i}", "content_hash": f"h{i}", "proposed_split": "train"} for i in range(6)]
        selection = {"baseline": rows[0], "burst": rows[1:]}
        rows[0]["proposed_split"] = "final_check"
        with self.assertRaises(ValueError):
            exploratory_aggregate(selection, [], {})
        rows[0]["proposed_split"] = "train"
        rows[1]["content_hash"] = rows[0]["content_hash"]
        with self.assertRaises(ValueError):
            exploratory_aggregate(selection, [], {})


class StrictTargetTests(unittest.TestCase):
    """Invented scalar/frame-index fixtures, with no model/API/media access."""

    def setUp(self):
        review = {"review_status": "human_reviewed", "review_origin": "human", "reviewer_id": "synthetic_fixture_author", "reviewed_at": "2026-10-17T00:00:00Z"}
        frame_pts = [0, 1 / 30, 2 / 30]
        frames = [{"frame_id": f"f{i}", "pts_sec": value} for i, value in enumerate(frame_pts)]
        truth_event = {"event_type_id": "synthetic_a", "anchor_sec": 1 / 30, "anchor_frame_id": "f1", "rule_id": "synthetic_rule", "rule_grounding_status": "verified_applicable", "temporal_verified": True, "exceptions_verified": True, "evidence_frame_ids": ["f1"]}
        rows = [{"sample_id": "synthetic_positive", "content_sha256": "a" * 64, "group_id": "synthetic_gp", "split": "validation"}, {"sample_id": "synthetic_negative", "content_sha256": "b" * 64, "group_id": "synthetic_gn", "split": "validation"}]
        self.manifest = {"state": "locked", "scenario_groups_reviewed": True, "approved_by": "synthetic_fixture_author", "locked_at": "2026-10-17T00:00:00Z", "source_manifest_sha256": "c" * 64, "records": rows, "records_sha256": records_digest(rows)}
        self.manifest_hash = digest(self.manifest)
        self.truth = [{**row, **review, "verdict": "violation" if i == 0 else "no_violation", "events": [truth_event] if i == 0 else [], "full_review_complete": True, "negative_verified": i == 1, "split_lock_verified": True, "split_manifest_sha256": self.manifest_hash, "unresolved_items": [], "frames": frames, "source_timing": {"original_timeline_verified": True, "original_frame_pts_sec": frame_pts}} for i, row in enumerate(rows)]
        pred_event = {**truth_event, "timestamp_ms": 33, "local_review": {**review, "evidence_supported": True, "timestamp_correspondence_verified": True, "rule_applicability_verified": True}}
        self.expected = [{"attempt_id": "synthetic_attempt_p", "sample_id": "synthetic_positive"}, {"attempt_id": "synthetic_attempt_n", "sample_id": "synthetic_negative"}]
        self.attempts = [{**planned, "status": "done", "verdict": "violation" if i == 0 else "no_violation", "events": [pred_event] if i == 0 else [], "public_contract_validated": True, "upload_start_to_full_result_ms": 9999.9, "timing_origin": "client_upload_start_to_full_result_byte"} for i, planned in enumerate(self.expected)]
        self.metadata = {"inference_kind": "synthetic", "execution_verified": True, "exposure_audit_passed": True, "exposure_registry_sha256": "synthetic", "model_revision": "synthetic", "conditions_sha256": "synthetic", "rules_sha256": "synthetic", "attempt_manifest_frozen_before_run": True, "uncached_inputs_verified": True, "heldout_use_disclosed": True}

    def report(self, **kwargs):
        return evaluate_strict_targets(self.truth, self.expected, self.attempts, split_manifest=self.manifest, split_manifest_sha256=self.manifest_hash, metadata=self.metadata, verified_rule_ids={"synthetic_rule"}, **kwargs)

    def test_perfect_synthetic_case_never_becomes_real_accuracy_claim(self):
        result = self.report()
        self.assertEqual(result["status"], "SYNTHETIC_ONLY")
        self.assertEqual(result["all_attempt_metrics"]["all_criteria_passed"], 2)
        self.assertFalse(result["claim_limits"]["observed_reviewed_sample_all_criteria_passed"])
        self.assertFalse(result["claim_limits"]["universal_100_percent_accuracy_claim_permitted"])

    def test_ten_second_boundary_is_strict_and_not_rounded(self):
        self.attempts[0]["upload_start_to_full_result_ms"] = 10000
        result = self.report()
        self.assertEqual(result["all_attempt_metrics"]["strict_latency_passed"], 1)
        self.attempts[0]["upload_start_to_full_result_ms"] = 9999.9999
        self.assertEqual(self.report()["all_attempt_metrics"]["strict_latency_passed"], 2)

    def test_named_strict_ten_second_policy_cannot_be_relaxed(self):
        with self.assertRaises(ValueError):
            self.report(deadline_ms=10000.001)
        with self.assertRaises(ValueError):
            self.report(deadline_ms=20000)

    def test_failed_and_missing_attempts_never_drop_from_denominator(self):
        self.expected.append({"attempt_id": "synthetic_missing", "sample_id": "synthetic_positive"})
        self.attempts[0]["status"] = "failed"
        result = self.report()
        self.assertEqual(result["denominators"]["expected_attempts"], 3)
        self.assertEqual(result["denominators"]["failed_or_unprocessed"], 2)
        self.assertEqual(result["denominators"]["missing_attempt_results"], 1)
        self.assertEqual(result["whole_video"]["unprocessed_positive"], 2)
        self.assertEqual(result["events"]["fn"], 2)
        self.assertEqual(result["all_attempt_metrics"]["all_criteria_rate"], 1 / 3)

    def test_evidence_review_is_required_separately_from_detection(self):
        del self.attempts[0]["events"][0]["local_review"]
        result = self.report()
        self.assertEqual(result["all_attempt_metrics"]["detection_passed"], 2)
        self.assertEqual(result["all_attempt_metrics"]["grounding_and_evidence_passed"], 1)

    def test_millisecond_display_does_not_create_one_ms_action_accuracy(self):
        event = self.attempts[0]["events"][0]
        event.update(anchor_frame_id="f2", anchor_sec=0.067, timestamp_ms=67, evidence_frame_ids=["f2"])
        result = self.report(max_anchor_error_ms=1)
        timing = result["timestamp_diagnostics"]
        self.assertAlmostEqual(timing["error_to_human_selected_original_frame_anchor_ms"]["max"], 1000 / 30)
        self.assertAlmostEqual(timing["adjacent_original_frame_spacing_ms"]["max"], 1000 / 30)
        self.assertLess(timing["display_rounding_error_ms"]["max"], 0.5)
        self.assertEqual(result["all_attempt_metrics"]["all_criteria_passed"], 1)
        self.assertFalse(result["claim_limits"]["one_millisecond_action_timing_claim_permitted"])

    def test_missing_source_timeline_is_not_inferred_from_sampled_frames(self):
        self.truth[0]["source_timing"]["original_timeline_verified"] = False
        result = self.report()
        self.assertIn("source_frame_binding_or_timestamp_review_incomplete", result["evidence_blocker_counts"])
        self.assertEqual(result["all_attempt_metrics"]["all_criteria_passed"], 1)

    def test_unreviewed_truth_cannot_pass_even_if_outputs_agree(self):
        self.metadata["inference_kind"] = "real_local"  # Simulated gate input only.
        self.truth[0]["review_status"] = "local_teacher_provisional"
        result = self.report()
        self.assertEqual(result["status"], "UNMEASURED")
        self.assertFalse(result["claim_limits"]["observed_reviewed_sample_all_criteria_passed"])

    def test_unexpected_attempt_is_not_silently_excluded(self):
        self.attempts.append({"attempt_id": "unexpected", "sample_id": "synthetic_positive"})
        with self.assertRaises(ValueError):
            self.report()

    def test_grounding_requires_verified_applicable_source_and_exceptions(self):
        self.truth[0]["events"][0]["exceptions_verified"] = False
        result = self.report()
        self.assertIn("truth_grounding_or_temporal_review_incomplete", result["evidence_blocker_counts"])
        self.assertEqual(result["all_attempt_metrics"]["all_criteria_passed"], 1)

    def test_millisecond_field_must_match_rounded_source_pts(self):
        self.attempts[0]["events"][0]["timestamp_ms"] = 1
        self.assertIn("numeric_millisecond_representation_mismatch", self.report()["evidence_blocker_counts"])

    def test_requested_millisecond_representation_cannot_be_omitted(self):
        del self.attempts[0]["events"][0]["timestamp_ms"]
        result = self.report()
        self.assertIn("numeric_millisecond_representation_mismatch", result["evidence_blocker_counts"])
        self.assertEqual(result["all_attempt_metrics"]["all_criteria_passed"], 1)

    def test_wrong_timing_origin_cannot_pass_end_to_end_target(self):
        self.attempts[0]["timing_origin"] = "worker_compute_only"
        result = self.report()
        self.assertEqual(result["all_attempt_metrics"]["strict_latency_passed"], 1)
        self.assertEqual(result["all_attempt_metrics"]["timing_unknown_or_invalid"], 1)

    def test_vfr_spacing_uses_actual_original_pts(self):
        self.truth[0]["source_timing"]["original_frame_pts_sec"] = [0, 0.01, 0.05]
        self.truth[0]["frames"] = [{"frame_id": "f0", "pts_sec": 0}, {"frame_id": "f1", "pts_sec": 0.01}, {"frame_id": "f2", "pts_sec": 0.05}]
        self.truth[0]["events"][0]["anchor_sec"] = 0.01
        self.attempts[0]["events"][0].update(anchor_sec=0.01, timestamp_ms=10)
        result = self.report()
        self.assertEqual(result["timestamp_diagnostics"]["adjacent_original_frame_spacing_ms"]["max"], 40)


class WeakOperationalTests(unittest.TestCase):
    """Synthetic binary-label policy tests; no private corpus or actual model."""

    def setUp(self):
        rows = [{"sample_id": f"weak_val_{i}", "recording_id": f"weak_val_{i}", "content_sha256": digest(f"synthetic_content_{i}"), "group_id": f"synthetic_group_{i}", "split": "validation"} for i in range(4)]
        rows.append({"sample_id": "weak_train", "recording_id": "weak_train", "content_sha256": digest("synthetic_train"), "group_id": "synthetic_train_group", "split": "train"})
        self.source_pin = digest("synthetic_provided_workbook")
        self.labels = [{"content_sha256": row["content_sha256"], "provided_video_label": i % 2 == 0, "provided_label_origin": "organizer_workbook", "provided_label_source_sha256": self.source_pin, "target_scope": "video_binary_mil_only", "teacher_used_as_binary_ground_truth": False, "split": "validation"} for i, row in enumerate(rows[:4])]
        self.label_hash = provided_label_digest(self.labels)
        self.pin = weak_assignment_digest(rows)
        self.manifest = {"records": rows, "assignments_sha256": self.pin, "state": "automated_provisional", "human_reviewed": False, "locked": False, "training_authorized": False, "group_review_origin": "local_model_and_metadata", "provided_label_manifests": {"validation": {"sha256": self.label_hash}}}
        self.config = {"authorized": True, "authorization_reference": "synthetic_explicit_automatic_firstpass", "allow_provisional_groups": True, "validation_assignments_frozen": True, "expected_assignments_sha256": self.pin, "expected_provided_label_source_sha256": self.source_pin, "expected_provided_label_manifests_sha256": {"validation": self.label_hash}, "allowed_target": "video_binary_mil"}
        self.gate = {"ok": True, "expected_assignments_sha256": self.pin, "provided_label_source_sha256": self.source_pin, "provided_label_manifest_sha256": self.label_hash, "exposure_registry_tip_sha256": digest("synthetic_exposure_tip"), "coverage_snapshot_sha256": digest("synthetic_fresh_inventory"), "label_split": "validation"}
        self.expected = [{"attempt_id": f"weak_a_{i}", "sample_id": row["sample_id"]} for i, row in enumerate(rows[:4])]
        self.predictions = [{**row, "status": "done", "prediction_validated": True, "verdict": "violation" if i in {0, 3} else "no_violation"} for i, row in enumerate(self.expected)]
        self.metadata = {"inference_kind": "synthetic", "execution_verified": True, "model_revision": "synthetic_head", "conditions_sha256": digest("synthetic_frozen_conditions"), "comparison_scope": "weak_model_only", "expected_attempts_frozen_before_run": True, "threshold_selection_sha256": digest("synthetic_threshold_zero_logodds")}

    def report(self, predictions=None):
        return evaluate_weak_binary(self.labels, self.expected, self.predictions if predictions is None else predictions, split_manifest=self.manifest, experiment_config=self.config, weak_gate_proof=self.gate, metadata=self.metadata, provided_labels_sha256=self.label_hash)

    def test_operational_track_needs_no_human_event_annotations(self):
        self.metadata["inference_kind"] = "real_local"  # Pure gate simulation only.
        result = self.report()
        self.assertTrue(result["model_selection_eligible"])
        self.assertFalse(result["provenance"]["human_reviewed_groups"])
        self.assertEqual(result["ground_truth_scopes"]["event_types"], "UNKNOWN")
        self.assertFalse(result["target_observations"]["human_event_grounding_100_percent_target_met"])

    def test_actual_binary_confusion_is_separate_from_teacher_truth(self):
        result = self.report()
        whole = result["whole_video_per_unique_content"]
        self.assertEqual([whole[key] for key in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertFalse(result["ground_truth_scopes"]["teacher_support_agreement_is_accuracy"])
        self.assertFalse(result["target_observations"]["universal_accuracy_claim_permitted"])

    def test_weak_head_completion_is_not_public_service_or_e2e_completion(self):
        result = self.report()
        self.assertEqual(result["all_attempt_completion"]["classification_predictions_completed"], 4)
        self.assertEqual(result["all_attempt_completion"]["public_results_completed"], 0)
        self.assertEqual(result["end_to_end_10s"]["passed"], 0)

    def test_failed_and_missing_predictions_remain_in_all_attempts(self):
        self.predictions[0]["status"] = "failed"
        result = self.report(self.predictions[:3])
        self.assertEqual(result["all_attempt_completion"]["required"], 4)
        self.assertEqual(result["all_attempt_completion"]["classification_failed_or_unprocessed"], 2)
        self.assertEqual(result["whole_video_per_unique_content"]["unprocessed_positive"], 1)
        self.assertEqual(result["whole_video_per_unique_content"]["unprocessed_negative"], 1)

    def test_provided_label_source_cannot_be_teacher(self):
        self.labels[0]["provided_label_origin"] = "local_teacher"
        result = self.report()
        self.assertFalse(result["evaluation_gate_passed"])
        self.assertIn("labels_not_independent_organizer_binary", result["gate_blocker_counts"])

    def test_stale_assignment_pin_or_label_artifact_is_rejected(self):
        self.manifest["records"][0]["group_id"] = "changed_group"
        self.assertIn("validation_assignment_pin_mismatch", self.report()["gate_blocker_counts"])

    def test_latest_exposure_gate_is_required_without_human_gate(self):
        self.gate["ok"] = False
        self.assertFalse(self.report()["evaluation_gate_passed"])

    def test_head_can_earn_provisional_selection_below_perfect_target(self):
        self.metadata["inference_kind"] = "real_local"  # Synthetic policy simulation.
        baseline = self.report()
        candidate_predictions = copy.deepcopy(self.predictions)
        candidate_predictions[2]["verdict"] = "violation"
        candidate = self.report(candidate_predictions)
        self.assertFalse(candidate["target_observations"]["provided_binary_detection_target_met"])
        decision = select_weak_candidate(baseline, candidate)
        self.assertTrue(decision["candidate_earned_provisional_selection"])
        self.assertEqual(decision["selection_scope"], "provisional_weak_model_only")
        self.assertFalse(decision["human_event_ground_truth_established"])

    def test_synthetic_result_cannot_select_real_checkpoint(self):
        baseline = self.report()
        self.assertFalse(select_weak_candidate(baseline, baseline)["candidate_earned_provisional_selection"])

    def test_weak_selection_can_disclose_better_total_error_with_class_tradeoff(self):
        self.metadata["inference_kind"] = "real_local"  # Synthetic policy simulation.
        for row in self.predictions:
            row["verdict"] = "no_violation"
        baseline = self.report()
        candidate_rows = copy.deepcopy(self.predictions)
        for i in (0, 1, 2):
            candidate_rows[i]["verdict"] = "violation"
        candidate = self.report(candidate_rows)
        selection = select_weak_candidate(baseline, candidate)
        self.assertTrue(selection["candidate_earned_provisional_selection"])
        self.assertEqual(selection["per_class_diagnostics"]["candidate"]["fp"], 1)
        self.assertEqual(selection["guarded_diagnostics"]["candidate"]["unique_video_total_errors_including_failures"], 1)

    def test_attestation_detects_tamper_and_expected_binding_mismatch(self):
        report = self.report()
        self.assertTrue(validate_weak_attestation(report)["ok"])
        self.assertFalse(validate_weak_attestation(report, {"provided_labels_sha256": "wrong"})["ok"])
        report["whole_video_per_unique_content"]["tp"] = 99
        self.assertFalse(validate_weak_attestation(report)["ok"])

    def test_e2e_ten_second_limit_is_strict_and_needs_full_public_result(self):
        for row in self.predictions:
            row.update(public_contract_validated=True, upload_start_to_full_result_ms=9999, timing_origin="client_upload_start_to_full_result_byte")
        self.predictions[0]["upload_start_to_full_result_ms"] = 10000
        result = self.report()
        self.assertEqual(result["end_to_end_10s"]["passed"], 3)
        self.assertEqual(result["end_to_end_10s"]["required_attempts"], 4)

    def test_validation_only_scope_rejects_train_input(self):
        self.expected[0]["sample_id"] = "weak_train"
        with self.assertRaises(ValueError):
            self.report()

    def test_provided_label_digest_matches_actual_A02_contract(self):
        from scripts.build_automated_split import label_records_digest
        self.assertEqual(self.label_hash, label_records_digest(self.labels))

    def test_final_check_labels_are_not_silently_used_for_calibration(self):
        self.labels[0]["split"] = "final_check"
        with self.assertRaises(ValueError):
            self.report()

    def test_A08_adapter_verifies_provided_track_without_inventing_teacher_proof(self):
        self.metadata["inference_kind"] = "real_local"  # Synthetic policy input only.
        identity = {"head_sha256": digest("synthetic_head"), "fact_ids": [f"synthetic_fact_{i}" for i in range(19)]}
        self.metadata["head_identity"] = identity
        self.gate["exposure_registry_file_sha256"] = digest("synthetic_registry_file")
        baseline = self.report()
        candidate_rows = copy.deepcopy(self.predictions)
        candidate_rows[2]["verdict"] = "violation"
        candidate = self.report(candidate_rows)
        selection = select_weak_candidate(baseline, candidate)
        manifest = {key: candidate[key] for key in ("split_assignments_sha256", "provided_labels_sha256", "input_set_sha256", "predictions_sha256", "provided_evaluation_attestation_sha256")}
        manifest.update(identity=identity, threshold_selection_sha256=self.metadata["threshold_selection_sha256"], exposure_manifest_sha256=self.gate["exposure_registry_file_sha256"], selection_attestation_sha256=selection["selection_attestation_sha256"])
        result = verify_provided_evaluation_for_manifest(candidate, manifest, selection)
        self.assertTrue(result["provided_label_evaluation_verified"])
        self.assertFalse(result["teacher_support_agreement_verified"])
        self.assertFalse(result["threshold_file_contents_verified"])
        self.assertFalse(result["independent_fact_event_accuracy_measured"])
        manifest["identity"] = {**identity, "head_sha256": digest("different_head")}
        self.assertFalse(verify_provided_evaluation_for_manifest(candidate, manifest, selection)["provided_label_evaluation_verified"])

    def test_A08_adapter_rejects_missing_selection_or_model_binding(self):
        result = verify_provided_evaluation_for_manifest(self.report(), {})
        self.assertFalse(result["provided_label_evaluation_verified"])
        self.assertIsNone(result["provided_binary_target_met"])


class CoarseAnchorTests(unittest.TestCase):
    """Coarse points never become source-frame-selected event timing truth."""

    def setUp(self):
        base = WeakOperationalTests()
        base.setUp()
        self.manifest, self.expected = base.manifest, base.expected
        self.anchors = [{"content_sha256": row["content_sha256"], "split": "validation", "provided_video_label": i % 2 == 0, "coarse_anchor_sec": [1.0] if i % 2 == 0 else [], "anchor_origin": "user_provided_coarse_table", "anchor_semantics": "approximate_point_not_boundary", "target_scope": "class_agnostic_approximate_event_anchor", "allowed_use": "evaluation_only_do_not_condition_inference", "anchor_source_sha256": digest("synthetic_coarse_input"), "provided_label_source_sha256": base.source_pin, "assignments_sha256": base.pin, "uncertainty_sec": None} for i, row in enumerate(self.manifest["records"][:4])]
        self.pins = {"assignments_sha256": base.pin, "anchor_source_sha256": digest("synthetic_coarse_input"), "provided_label_source_sha256": base.source_pin, "role_manifests": {"validation": {"sha256": provided_label_digest(self.anchors)}}}
        self.threshold = {"schema_version": "forte.weak_video_threshold.v1", "score_axis": "violation_log_odds", "minimum_video_score": 0.0, "comparison": "greater_than_or_equal", "validation_labels_used_for_selection": False}
        self.threshold_sha = digest(self.threshold)
        self.metadata = {"inference_kind": "synthetic", "model_revision": digest("synthetic_head"), "head_identity": {"feature_spec_sha256": digest("synthetic_features")}, "threshold_selection_sha256": self.threshold_sha, "validation_anchors_used_for_training": False, "validation_anchors_used_for_threshold_selection": False, "validation_anchors_used_for_prompting": False}
        self.terminals = [{**row, "status": "done", "prediction_validated": True} for row in self.expected]
        self.temporal = []
        for i, planned in enumerate(self.expected):
            content = self.manifest["records"][i]["content_sha256"]
            frames = [{"frame_id": hashlib.sha256(f"{content}:{j * 10}:1/10".encode()).hexdigest(), "source_pts": j * 10, "pts_sec": float(j), "eventness_logit": 1.0 if j == 1 and i % 2 == 0 else -1.0} for j in range(4)]
            self.temporal.append({**planned, "score_axis": "violation_log_odds", "threshold": 0.0, "head_sha256": self.metadata["model_revision"], "feature_spec_sha256": self.metadata["head_identity"]["feature_spec_sha256"], "time_base": "1/10", "source_start_sec": 0.0, "frames": frames})

    def report(self, **kwargs):
        return evaluate_coarse_anchors(self.anchors, self.temporal, self.expected, self.terminals, split_manifest=self.manifest, anchor_pins=self.pins, metadata=self.metadata, threshold_artifact=self.threshold, threshold_artifact_sha256=self.threshold_sha, **kwargs)

    def test_approximate_anchor_hit_does_not_claim_event_or_100ms_accuracy(self):
        report = self.report()
        self.assertEqual(report["denominators"]["provided_coarse_points"], 2)
        self.assertEqual(report["bands"][0]["hit_points"], 2)
        self.assertFalse(report["exact_event_f1_claim_permitted"])
        self.assertFalse(report["hundred_millisecond_accuracy_claim_permitted"])
        self.assertEqual(report["event_type_ground_truth"], "UNKNOWN")
        self.assertEqual(report["legal_ground_truth"], "UNKNOWN")

    def test_failed_temporal_output_stays_in_point_denominator(self):
        self.terminals[0]["status"] = "failed"
        self.temporal.pop(0)
        report = self.report()
        self.assertEqual(report["denominators"]["provided_coarse_points"], 2)
        self.assertEqual(report["denominators"]["points_with_failed_or_missing_temporal_prediction"], 1)
        self.assertEqual(report["bands"][0]["failure_inclusive_anchor_hit_rate"], 0.5)

    def test_sampling_gap_is_reported_instead_of_silently_excluded(self):
        for row in self.temporal:
            row["frames"] = [row["frames"][0], row["frames"][3]]
        report = self.report(tolerances_sec=(0.25,))
        self.assertEqual(report["bands"][0]["uncovered_or_failed_points"], 2)
        self.assertEqual(report["bands"][0]["failure_inclusive_anchor_hit_rate"], 0)

    def test_all_positive_frames_cannot_manufacture_event_precision(self):
        for row in self.temporal:
            for frame in row["frames"]:
                frame["eventness_logit"] = 1.0
        report = self.report()
        self.assertEqual(report["bands"][0]["failure_inclusive_anchor_hit_rate"], 1)
        self.assertEqual(report["frame_diagnostics"]["above_threshold_frame_fraction"], 1)
        self.assertFalse(report["exact_event_f1_claim_permitted"])

    def test_cross_content_frame_identity_is_rejected(self):
        self.temporal[0]["frames"][1]["frame_id"] = self.temporal[1]["frames"][1]["frame_id"]
        report = self.report()
        self.assertEqual(report["frame_diagnostics"]["invalid_source_frame_rows"], 1)
        self.assertEqual(report["bands"][0]["hit_points"], 1)

    def test_declared_seconds_must_agree_with_raw_source_pts(self):
        self.temporal[0]["frames"][1]["pts_sec"] = 1.5
        self.assertEqual(self.report()["frame_diagnostics"]["invalid_source_frame_rows"], 1)

    def test_validation_anchor_threshold_or_prompt_leakage_is_explicit(self):
        self.metadata["validation_anchors_used_for_prompting"] = True
        self.assertIn("validation_anchor_exclusion_unverified", self.report()["provenance_blocker_counts"])

    def test_final_check_anchor_rows_are_rejected(self):
        self.anchors[0]["split"] = "final_check"
        self.pins["role_manifests"]["validation"]["sha256"] = provided_label_digest(self.anchors)
        with self.assertRaises(ValueError):
            self.report()

    def test_unknown_source_uncertainty_remains_unknown(self):
        report = self.report()
        self.assertEqual(report["source_uncertainty"]["points_unspecified"], 2)
        self.assertEqual(report["source_uncertainty"]["points_specified"], 0)

    def test_stale_anchor_role_pin_is_rejected(self):
        self.anchors[0]["coarse_anchor_sec"] = [2.0]
        with self.assertRaises(ValueError):
            self.report()

    def test_temporal_threshold_cannot_be_refitted_to_validation_anchor(self):
        self.temporal[0]["threshold"] = -10
        with self.assertRaises(ValueError):
            self.report()

    def test_nonzero_source_origin_and_vfr_are_used_without_rounding(self):
        row = self.temporal[0]
        content = self.anchors[0]["content_sha256"]
        row.update(time_base="1/1000", source_start_sec=5.0)
        row["frames"] = [{"frame_id": hashlib.sha256(f"{content}:{pts}:1/1000".encode()).hexdigest(), "source_pts": pts, "pts_sec": pts / 1000 - 5, "eventness_logit": 1 if pts == 5010 else -1} for pts in (5000, 5010, 5050)]
        self.anchors[0]["coarse_anchor_sec"] = [0.01]
        self.pins["role_manifests"]["validation"]["sha256"] = provided_label_digest(self.anchors)
        report = self.report(tolerances_sec=(0.001,))
        self.assertEqual(report["bands"][0]["hit_points"], 2)
        self.assertFalse(report["hundred_millisecond_accuracy_claim_permitted"])


if __name__ == "__main__":
    unittest.main()
