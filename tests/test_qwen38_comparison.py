"""Scalar invented fixtures only: no media, model, network or held-out files."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from evaluation.io import digest, write_json
from evaluation.qwen38_comparison import ARMS, FULL_RESULT_ORIGIN, UI_RESULT_ORIGIN, compare_qwen38, main, validate_plan
from evaluation.weak_supervision import provided_label_digest


def fixtures(repeats=1):
    sources = [{"sample_id": f"opaque_{i}", "content_sha256": str(i + 1) * 64, "split": "validation", "duration_sec": 20} for i in range(2)]
    labels = [{"content_sha256": source["content_sha256"], "split": source["split"], "provided_video_label": i == 0, "provided_label_origin": "organizer_workbook", "target_scope": "video_binary_mil_only", "teacher_used_as_binary_ground_truth": False} for i, source in enumerate(sources)]
    anchors = [{"content_sha256": source["content_sha256"], "split": source["split"], "provided_video_label": i == 0, "anchor_origin": "user_provided_coarse_table", "anchor_semantics": "approximate_point_not_boundary", "target_scope": "class_agnostic_approximate_event_anchor", "allowed_use": "evaluation_only_do_not_condition_inference", "boundaries_known": False, "event_type_known": False, "legal_ground_truth": False, "coarse_anchor_sec": [5, 8] if i == 0 else [], "uncertainty_sec": None} for i, source in enumerate(sources)]
    recipe = {"sampling": {"frame_budget": 8}, "prompt_sha256": "c" * 64, "generation": {"do_sample": False}, "evidence": {"original_jpeg": True}, "timing": {"start": "immediately_before_upload", "end": "usable_rendered_result"}, "concurrency": 1, "supported_envelope": {"duration_sec_max": 143, "codec": "h264", "max_bytes": 268435456}}
    arms = {arm: {"model_id": "synthetic_base_8b" if arm == "baseline_8b" else "synthetic_qwen38", "model_revision": "1" * 40 if arm == "baseline_8b" else "2" * 40, "adapter_sha256": "a" * 64 if arm == "qwen38_tuned" else None} for arm in ARMS}
    attempts = [{"attempt_id": f"a_{arm}_{i}_{repeat}", "arm": arm, "sample_id": source["sample_id"], "content_sha256": source["content_sha256"], "split": source["split"], "repeat_index": repeat} for arm in ARMS for i, source in enumerate(sources) for repeat in range(repeats)]
    plan = {"schema_version": "forte.qwen38_comparison_plan.v1", "records": sources, "arms": arms, "recipe": recipe, "recipe_sha256": digest(recipe), "provided_labels_sha256": provided_label_digest(labels), "coarse_anchors_sha256": provided_label_digest(anchors), "attempts": attempts}
    results = [{**attempt, "comparison_plan_sha256": digest(plan), "recipe_sha256": plan["recipe_sha256"], "model_identity": arms[attempt["arm"]], "status": "done", "public_contract_validated": True, "verdict": "violation" if attempt["sample_id"] == "opaque_0" else "no_violation", "violations": [{"type": "invented", "timecode_sec": 5, "description": "invented fixture", "evidence_frames": []}] if attempt["sample_id"] == "opaque_0" else [], "upload_start_to_full_result_ms": 9999, "timing_origin": FULL_RESULT_ORIGIN, "upload_start_to_usable_result_ms": 9999.5, "ui_timing_origin": UI_RESULT_ORIGIN} for attempt in attempts]
    metadata = {"inference_kind": "synthetic", "attempt_plan_frozen_before_run": True, "validation_reused_for_development": True, "final_check_used": False, "execution_verified": True, "exposure_audit_passed": True, "exposure_audit_sha256": "d" * 64, "fresh_upload_decode_inference_verified": True, "comparison_plan_sha256": digest(plan), "verified_model_identity_receipts": {arm: "e" * 64 for arm in ARMS}, "coarse_anchors_used_to_condition_inference": False}
    return plan, results, labels, anchors, metadata


class QwenComparisonTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.results, self.labels, self.anchors, self.metadata = fixtures()

    def report(self, **kwargs):
        return compare_qwen38(self.plan, self.results, self.labels, metadata=self.metadata, coarse_anchors=self.anchors, **kwargs)

    def test_synthetic_software_success_never_becomes_model_or_accuracy_claim(self):
        report = self.report()
        self.assertEqual(report["status"], "SYNTHETIC_ONLY")
        self.assertFalse(report["claim_limits"]["model_promotion_authorized"])
        self.assertFalse(report["claim_limits"]["heldout_accuracy_claim_permitted"])
        self.assertFalse(report["claim_limits"]["millisecond_semantic_onset_claim_permitted"])
        self.assertFalse(report["arms"]["baseline_8b"]["timing"]["all_required_usable_results_under_10s_observed"])

    def test_missing_failed_and_invalid_success_stay_in_denominators(self):
        self.results[0].update(status="upload_failed")
        self.results[1].update(status="done", verdict="violation", violations=[])
        del self.results[2]
        report = self.report()
        baseline = report["arms"]["baseline_8b"]
        self.assertEqual(baseline["completion"]["required_attempts"], 2)
        self.assertEqual(baseline["completion"]["failed_or_unprocessed"], 2)
        self.assertEqual(baseline["binary_per_attempt"]["unprocessed_positive"], 1)
        self.assertEqual(baseline["binary_per_attempt"]["unprocessed_negative"], 1)
        self.assertEqual(report["arms"]["qwen38_untuned"]["completion"]["missing_attempts"], 1)

    def test_every_failure_class_is_accounted_without_clean_negative_fallback(self):
        for status in ("failed", "timeout", "admission_rejected", "invalid_json", "insufficient_analysis", "not_attempted", "processing"):
            with self.subTest(status=status):
                self.results[0]["status"] = status
                baseline = self.report()["arms"]["baseline_8b"]
                self.assertEqual(baseline["binary_per_attempt"]["unprocessed_positive"], 1)
                self.assertEqual(baseline["completion"]["required_attempts"], 2)
                self.assertEqual(baseline["completion"]["failure_counts"], {status: 1})

    def test_result_attribution_and_unplanned_attempts_fail_closed(self):
        original = copy.deepcopy(self.results)
        for field, value in (("sample_id", "opaque_1"), ("arm", "qwen38_tuned"), ("content_sha256", "f" * 64), ("split", "train"), ("repeat_index", 1), ("recipe_sha256", "f" * 64), ("comparison_plan_sha256", "f" * 64)):
            with self.subTest(field=field):
                self.results = copy.deepcopy(original)
                self.results[0][field] = value
                with self.assertRaises(ValueError):
                    self.report()
        self.results = original + [{**original[0], "attempt_id": "unexpected"}]
        with self.assertRaises(ValueError):
            self.report()

    def test_wrong_loaded_adapter_is_rejected(self):
        self.results[4]["model_identity"] = {**self.results[4]["model_identity"], "adapter_sha256": "f" * 64}
        with self.assertRaises(ValueError):
            self.report()

    def test_preinference_failure_keeps_unknown_actual_model_and_planned_attribution(self):
        intended = self.results[0].pop("model_identity")
        self.results[0].update(status="upload_failed", expected_model_identity=intended)
        baseline = self.report()["arms"]["baseline_8b"]
        self.assertEqual(baseline["completion"]["actual_model_identity_unknown_attempts"], 1)
        self.assertEqual(baseline["binary_per_attempt"]["unprocessed_positive"], 1)
        self.assertNotIn("model_identity", self.results[0])

    def test_unverified_actual_model_success_cannot_earn_confusion_or_latency_credit(self):
        intended = self.results[0].pop("model_identity")
        self.results[0]["expected_model_identity"] = intended
        baseline = self.report()["arms"]["baseline_8b"]
        self.assertEqual(baseline["completion"]["failure_counts"], {"model_identity_unverified": 1})
        self.assertEqual(baseline["binary_per_attempt"]["tp"], 0)
        self.assertEqual(baseline["timing"]["full_result_bytes"]["strictly_under_10000_ms"], 1)

    def test_unknown_actual_model_failure_still_requires_exact_intended_identity(self):
        self.results[0].pop("model_identity")
        self.results[0].update(status="admission_rejected")
        with self.assertRaises(ValueError):
            self.report()

    def test_duplicate_terminal_cannot_be_used_as_retry_replacement(self):
        self.results.append(copy.deepcopy(self.results[0]))
        with self.assertRaises(ValueError):
            self.report()

    def test_first_repeat_is_used_even_if_retry_is_correct(self):
        self.plan, self.results, self.labels, self.anchors, self.metadata = fixtures(repeats=3)
        self.results[0].update(status="timeout")
        baseline = self.report()["arms"]["baseline_8b"]
        self.assertEqual(baseline["binary_per_attempt"]["all_labeled_denominator_including_failures"], 6)
        self.assertEqual(baseline["binary_per_unique_content"]["all_labeled_denominator_including_failures"], 2)
        self.assertEqual(baseline["binary_per_unique_content"]["unprocessed_positive"], 1)
        self.assertEqual(baseline["binary_per_unique_content"]["tp"], 0)
        self.assertEqual(baseline["coarse_point_diagnostics"]["points_with_failed_first_attempt"], 2)

    def test_paired_population_cannot_be_cherry_picked(self):
        self.plan["attempts"].pop()
        with self.assertRaises(ValueError):
            validate_plan(self.plan)

    def test_heldout_role_rejected_in_plan_labels_anchors_and_results(self):
        for population, key in ((self.plan["records"], "split"), (self.plan["attempts"], "split"), (self.labels, "split"), (self.anchors, "split"), (self.results, "split")):
            previous = population[0][key]
            population[0][key] = "final_check"
            with self.assertRaises(ValueError):
                self.report()
            population[0][key] = previous

    def test_coarse_points_never_create_type_or_boundary_gold(self):
        report = self.report()
        coarse = report["arms"]["baseline_8b"]["coarse_point_diagnostics"]
        self.assertEqual(coarse["provided_points"], 2)
        self.assertEqual(coarse["bands"][0]["hit_points"], 1)
        self.assertEqual(coarse["bands"][0]["failure_inclusive_hit_rate"], 0.5)
        self.assertFalse(coarse["exact_event_precision_or_f1_available"])
        self.assertEqual(report["reference_scopes"]["event_types"], "UNMEASURED_NO_GENUINE_REFERENCE")
        self.assertEqual(report["reference_scopes"]["evidence_support"], "UNMEASURED_NO_GENUINE_REFERENCE")

    def test_bytes_only_timer_cannot_pass_usable_result_target(self):
        # Test the gate with invented provenance flags; no real run is asserted.
        self.metadata["inference_kind"] = "real_local"
        for result in self.results:
            del result["upload_start_to_usable_result_ms"]
        report = self.report()
        timing = report["arms"]["baseline_8b"]["timing"]
        self.assertEqual(timing["full_result_bytes"]["strictly_under_10000_ms"], 2)
        self.assertEqual(timing["usable_rendered_result"]["failed_late_or_unmeasured"], 2)
        self.assertFalse(timing["all_required_usable_results_under_10s_observed"])

    def test_ten_second_boundary_wrong_clock_and_before_bytes_ui_time_fail(self):
        self.results[0]["upload_start_to_full_result_ms"] = 10000
        self.results[0]["upload_start_to_usable_result_ms"] = 10000
        self.results[1]["timing_origin"] = "server_model_elapsed"
        timing = self.report()["arms"]["baseline_8b"]["timing"]
        self.assertEqual(timing["full_result_bytes"]["strictly_under_10000_ms"], 0)
        self.assertEqual(timing["usable_rendered_result"]["strictly_under_10000_ms"], 0)

    def test_model_json_malformed_or_outside_source_is_failed_result(self):
        for bad in (None, [], {"verdict": []}, {"verdict": "violation", "violations": [{"type": "a", "timecode_sec": 1000}]}):
            with self.subTest(bad=bad):
                self.results[0]["complete_result"] = bad
                baseline = self.report()["arms"]["baseline_8b"]
                self.assertEqual(baseline["completion"]["failure_counts"], {"invalid_success_payload": 1})
                self.assertEqual(baseline["binary_per_attempt"]["unprocessed_positive"], 1)

    def test_source_label_hash_and_scope_cannot_be_swapped_for_teacher_truth(self):
        self.labels[0]["teacher_used_as_binary_ground_truth"] = True
        self.plan["provided_labels_sha256"] = provided_label_digest(self.labels)
        with self.assertRaises(ValueError):
            self.report()

    def test_aliases_do_not_expand_unique_content_denominator(self):
        self.plan["records"][1]["content_sha256"] = self.plan["records"][0]["content_sha256"]
        with self.assertRaises(ValueError):
            validate_plan(self.plan)

    def test_cli_rejects_heldout_path_before_any_read(self):
        with patch("evaluation.qwen38_comparison.read_json") as read, redirect_stdout(io.StringIO()):
            code = main(["--plan", "plan.json", "--results", "results.json", "--labels", "provided_binary_labels.final_check.jsonl", "--metadata", "meta.json", "--output", "report.json"])
        self.assertEqual(code, 1)
        read.assert_not_called()

    def test_train_and_validation_metrics_keep_distinct_roles(self):
        self.plan["records"][0]["split"] = "train"
        self.labels[0]["split"] = "train"
        self.anchors[0]["split"] = "train"
        for attempt in self.plan["attempts"]:
            if attempt["sample_id"] == "opaque_0":
                attempt["split"] = "train"
        self.plan["provided_labels_sha256"] = provided_label_digest(self.labels)
        self.plan["coarse_anchors_sha256"] = provided_label_digest(self.anchors)
        for result in self.results:
            if result["sample_id"] == "opaque_0":
                result["split"] = "train"
            result["comparison_plan_sha256"] = digest(self.plan)
        self.metadata["comparison_plan_sha256"] = digest(self.plan)
        roles = self.report()["arms"]["baseline_8b"]["binary_per_unique_content_by_role"]
        self.assertEqual(roles["train"]["tp"], 1)
        self.assertEqual(roles["validation"]["tp"], 0)
        self.assertEqual(roles["validation"]["tn"], 1)

    def test_all_failed_real_local_run_is_a_diagnostic_never_a_target_pass(self):
        self.metadata["inference_kind"] = "real_local"
        for result in self.results:
            result["status"] = "timeout"
        report = self.report()
        self.assertEqual(report["status"], "DEVELOPMENT_DIAGNOSTIC_READY")
        self.assertEqual(report["arms"]["qwen38_tuned"]["completion"]["failed_or_unprocessed"], 2)
        self.assertFalse(report["arms"]["qwen38_tuned"]["timing"]["all_required_usable_results_under_10s_observed"])
        self.assertFalse(report["claim_limits"]["model_promotion_authorized"])

    def test_cli_emits_only_aggregate_output_and_preserves_original_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("plan", self.plan), ("results", self.results), ("labels", self.labels), ("metadata", self.metadata)):
                write_json(root / f"{name}.json", value)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--plan", str(root / "plan.json"), "--results", str(root / "results.json"), "--labels", str(root / "labels.json"), "--metadata", str(root / "metadata.json"), "--output", str(root / "report.json")])
            self.assertEqual(code, 2)
            self.assertNotIn("opaque_0", output.getvalue())
            self.assertNotIn("synthetic_qwen38", output.getvalue())
            self.assertNotIn(directory, output.getvalue())
            self.assertTrue((root / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
