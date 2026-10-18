"""Synthetic-only provisional grouping and weak supervision gate checks."""
import copy
import hashlib
import json
import unittest

import test_reviewed_split as fixture_module
from build_automated_split import automatic_scenario_groups, build_automated_candidate
from exposure_registry import append_exposures, read_registry, register_content_manifest, register_accepted_inventory
from split_gate import validate_automated_experiment_inputs, validate_locked_training_inputs


class AutomatedSplitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ReviewedSplitTests("test_complete_review_is_candidate_without_authority")
        self.fixture.setUp()
        _, self.status = self.fixture.inventory_fixture()
        self.fixture.registry = read_registry(self.fixture.registry_path, self.fixture.records)
        self.manifest_sha = "a" * 64
        self.workbook_sha = "b" * 64
        self.proposal = {"schema_version": "forte.automatic-group-proposals.v1", "proposal_origin": "local_model_and_metadata",
                         "method": "synthetic-label-blind", "config_sha256": "c" * 64, "source_manifest_sha256": self.manifest_sha,
                         "groups": [{"proposed_group_id": f"proposal-{index}", "recording_ids": record["recording_ids"],
                                     "method": "local_model_signature" if index < 5 else "exact_content_only",
                                     "confidence_status": "heuristic_unvalidated"} for index, record in enumerate(self.fixture.review)]}

    def tearDown(self):
        self.fixture.tearDown()

    def build(self):
        return build_automated_candidate(self.fixture.records, self.fixture.rows, self.proposal, self.fixture.registry,
                                         manifest_sha256=self.manifest_sha, workbook_sha256=self.workbook_sha)

    def gate(self, candidate, scenario, labels, *, label_split="train", authorization=None, **kwargs):
        authorization = authorization if authorization is not None else {
            "authorized": True, "authorization_reference": "synthetic-explicit-user-first-pass",
            "allow_provisional_groups": True, "validation_assignments_frozen": True,
            "allowed_target": "video_binary_mil", "expected_assignments_sha256": candidate["assignments_sha256"],
            "expected_provided_label_source_sha256": self.workbook_sha,
            "expected_provided_label_manifests_sha256": {role: binding["sha256"] for role, binding in candidate["provided_label_manifests"].items()}}
        return validate_automated_experiment_inputs(candidate, self.fixture.records, scenario,
                                                   experiment_authorization=authorization, registry_path=self.fixture.registry_path,
                                                   coverage_status_path=self.status, provided_label_records=labels,
                                                   expected_provided_label_source_sha256=self.workbook_sha,
                                                   label_split=label_split, **kwargs)

    def test_honest_provisional_candidate_and_explicit_weak_gate(self):
        candidate, scenario, labels, report = self.build()
        self.assertEqual(candidate["state"], "automated_provisional")
        self.assertFalse(candidate["human_reviewed"])
        self.assertFalse(candidate["training_authorized"])
        self.assertFalse(candidate["scenario_review_complete"])
        result = self.gate(candidate, scenario, labels["train"])
        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse(result["event_fact_ground_truth_validated"])
        self.assertFalse(result["scenario_independence_verified"])
        self.assertIn("expected_assignments_sha256", result)

    def test_existing_human_gate_stays_strict(self):
        candidate, scenario, _, _ = self.build()
        result = validate_locked_training_inputs(candidate, self.fixture.records, scenario,
                                                  expected_assignments_sha256=candidate["assignments_sha256"],
                                                  registry_path=self.fixture.registry_path, coverage_status_path=self.status)
        self.assertFalse(result["ok"])
        self.assertIn("split_not_approved_for_training", result["errors"])

    def test_deterministic_proposals_and_split(self):
        self.assertEqual(self.build(), self.build())

    def test_missing_explicit_experiment_authority_rejected(self):
        candidate, scenario, labels, _ = self.build()
        result = self.gate(candidate, scenario, labels["train"], authorization={})
        self.assertIn("explicit_provisional_experiment_authorization_missing", result["errors"])

    def test_group_proposal_does_not_consume_binary_answers(self):
        self.proposal["groups"][0]["provided_video_label"] = True
        with self.assertRaisesRegex(ValueError, "automatic_grouping_cannot_consume_provided_answers"):
            self.build()

    def test_omitted_group_membership_rejected(self):
        self.proposal["groups"].pop()
        with self.assertRaisesRegex(ValueError, "automatic_group_coverage_incomplete"):
            self.build()

    def test_exact_duplicates_merge_conflicting_machine_components_conservatively(self):
        first = self.proposal["groups"][0]
        alias = first["recording_ids"].pop()
        self.proposal["groups"].append({"proposed_group_id": "separate-duplicate-proposal", "recording_ids": [alias]})
        candidate, _, _, _ = self.build()
        rows = [row for row in candidate["records"] if row["content_hash"] == self.fixture.records[0]["content_hash"]]
        self.assertEqual(len({row["scenario_group_id"] for row in rows}), 1)
        self.assertEqual(len({row["split"] for row in rows}), 1)

    def test_exposed_merged_scenario_not_final_check(self):
        self.proposal["groups"][0]["recording_ids"] += self.proposal["groups"].pop(1)["recording_ids"]
        candidate, scenario, _, _ = self.build()
        exposed_members = self.proposal["groups"][0]["recording_ids"]
        self.assertTrue(all(row["split"] != "final_check" for row in candidate["records"] if row["recording_id"] in exposed_members))

    def test_no_fake_human_origin_in_machine_proposals(self):
        self.proposal["human_reviewed"] = True
        with self.assertRaisesRegex(ValueError, "automatic_groups_cannot_claim_human_approval"):
            self.build()

    def test_independent_binary_label_conflict_rejected(self):
        self.fixture.rows[1]["has_violation"] = not self.fixture.rows[0]["has_violation"]
        with self.assertRaisesRegex(ValueError, "provided_binary_label_conflict_or_unknown"):
            self.build()

    def test_validation_labels_separate_from_train(self):
        candidate, scenario, labels, _ = self.build()
        result = self.gate(candidate, scenario, labels["validation"], label_split="validation")
        self.assertTrue(result["ok"], result["errors"])
        wrong_role = self.gate(candidate, scenario, labels["validation"])
        self.assertIn("provided_label_partition_or_group_mismatch", wrong_role["errors"])

    def test_assignment_pin_mismatch_rejected(self):
        candidate, scenario, labels, _ = self.build()
        authorization = {"authorized": True, "authorization_reference": "synthetic-authority", "allow_provisional_groups": True,
                         "validation_assignments_frozen": True, "allowed_target": "video_binary_mil",
                         "expected_assignments_sha256": "f" * 64, "expected_provided_label_source_sha256": self.workbook_sha}
        result = self.gate(candidate, scenario, labels["train"], authorization=authorization)
        self.assertIn("provisional_assignment_pin_mismatch", result["errors"])

    def test_labels_cannot_be_upgraded_to_fact_targets(self):
        candidate, scenario, labels, _ = self.build()
        labels["train"][0]["facts"] = {"synthetic_fact": "present"}
        result = self.gate(candidate, scenario, labels["train"])
        self.assertIn("binary_label_used_as_unsupported_event_truth", result["errors"])

    def test_later_final_check_exposure_invalidates_weak_track(self):
        candidate, scenario, labels, _ = self.build()
        final_hash = labels["final_check"][0]["content_hash"]
        append_exposures(self.fixture.registry_path, self.fixture.records, {final_hash}, trial_id="synthetic-later-user-upload", phase="local-trial", purpose="local_model_attempt")
        result = self.gate(candidate, scenario, labels["train"])
        self.assertIn("exposed_scenario_in_final_check", result["errors"])

    def test_all_exposed_contents_never_become_unseen(self):
        self.fixture.registry["content_hashes"] = {row["content_hash"] for row in self.fixture.records}
        candidate, _, labels, report = self.build()
        self.assertFalse(labels["final_check"])
        self.assertFalse(candidate["final_check_independence_verified"])

    def test_single_provisional_group_has_no_train_validation_claim(self):
        self.proposal["groups"] = [{"proposed_group_id": "single-group", "recording_ids": [row["recording_id"] for row in self.fixture.records]}]
        with self.assertRaisesRegex(ValueError, "insufficient_provisional_groups_for_train_validation"):
            self.build()

    def test_frozen_validation_content_cannot_be_requested_as_training(self):
        candidate, scenario, labels, _ = self.build()
        result = self.gate(candidate, scenario, labels["train"], requested_content_hashes={labels["validation"][0]["content_hash"]})
        self.assertIn("requested_content_outside_allowed_partition", result["errors"])

    def test_module_importable_for_training_image(self):
        from scripts.split_gate import validate_automated_experiment_inputs as imported_gate
        from scripts.build_automated_split import label_records_digest as imported_digest
        self.assertTrue(callable(imported_gate))
        self.assertTrue(callable(imported_digest))

    def content_registration_fixture(self, roles=("train", "validation")):
        candidate, _, labels, _ = self.build()
        manifest = self.fixture.private / "source-manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in self.fixture.records))
        candidate["source_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        split_path = self.fixture.private / "frozen-split.json"
        split_path.write_text(json.dumps(candidate))
        worklist = self.fixture.private / "worklist.json"
        worklist.write_text(json.dumps({"assignments_sha256": candidate["assignments_sha256"],
                                       "records": [{"content_sha256": row["content_hash"], "split": role, "recording_ids": row["recording_ids"], "source_path": "not-opened.mov"} for role in roles for row in labels[role]]}))
        return candidate, manifest, split_path, worklist

    def test_direct_feature_exposure_registration_no_media_and_idempotent(self):
        candidate, manifest, split_path, worklist = self.content_registration_fixture()
        def register():
            return register_content_manifest(worklist, self.fixture.registry_path, manifest, split_path,
                                             expected_assignments_sha256=candidate["assignments_sha256"], trial_id="synthetic-feature-run",
                                             purpose="feature_extraction_attempt", allowed_splits={"train", "validation"})
        first = register()
        self.assertTrue(first["ok"])
        self.assertFalse(first["media_opened"])
        self.assertFalse(first["model_calls_made"])
        before = self.fixture.registry_path.read_bytes()
        self.assertEqual(register()["events_appended"], 0)
        self.assertEqual(before, self.fixture.registry_path.read_bytes())

    def test_direct_exposure_final_check_rejected_without_append(self):
        candidate, manifest, split_path, worklist = self.content_registration_fixture(("final_check",))
        before = self.fixture.registry_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "non_api_exposure_input_outside_allowed_roles"):
            register_content_manifest(worklist, self.fixture.registry_path, manifest, split_path,
                                      expected_assignments_sha256=candidate["assignments_sha256"], trial_id="synthetic-bad-feature-run",
                                      purpose="feature_extraction_attempt", allowed_splits={"train", "validation"})
        self.assertEqual(before, self.fixture.registry_path.read_bytes())

    def test_teacher_reservation_is_train_only(self):
        candidate, manifest, split_path, worklist = self.content_registration_fixture()
        with self.assertRaisesRegex(ValueError, "bounded_teacher_exposure_must_be_train_only"):
            register_content_manifest(worklist, self.fixture.registry_path, manifest, split_path,
                                      expected_assignments_sha256=candidate["assignments_sha256"], trial_id="synthetic-teacher",
                                      purpose="bounded_teacher_attempt", allowed_splits={"train", "validation"})

    def test_accepted_inventory_never_erases_direct_feature_exposure(self):
        candidate, manifest, split_path, worklist = self.content_registration_fixture(("train",))
        register_content_manifest(worklist, self.fixture.registry_path, manifest, split_path,
                                  expected_assignments_sha256=candidate["assignments_sha256"], trial_id="synthetic-feature-union",
                                  purpose="feature_extraction_attempt")
        before = read_registry(self.fixture.registry_path, self.fixture.records)["content_hashes"]
        register_accepted_inventory(self.fixture.private / "accepted.json", self.fixture.registry_path, manifest, self.status)
        after = read_registry(self.fixture.registry_path, self.fixture.records)["content_hashes"]
        self.assertTrue(before <= after)


if __name__ == "__main__":
    unittest.main()
