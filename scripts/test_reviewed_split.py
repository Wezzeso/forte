"""Synthetic-only checks: python -m unittest discover -s scripts -p test_reviewed_split.py."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from audit_data import check_split, json_bytes
from build_reviewed_split import analyze_review, build_candidate, exposure_coverage, review_snapshot
from exposure_registry import append_exposures, read_registry, register_accepted_inventory, register_selection
from split_gate import validate_locked_training_inputs


class ReviewedSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.private = self.root / "private"
        self.private.mkdir()
        self.registry_path = self.private / "exposure.jsonl"
        self.records, self.rows, self.review, self.audits = [], [], [], []
        for index in range(20):
            content_hash = hashlib.sha256(f"synthetic-{index}".encode()).hexdigest()
            ids = [f"recording-{index}"] + ([f"recording-{index}-duplicate"] if index < 2 else [])
            for recording_id in ids:
                row_id = len(self.rows) + 2
                self.records.append({"recording_id": recording_id, "content_hash": content_hash,
                                     "group_id": f"group-{index}", "workbook_row_ids": [row_id], "duration_sec": 10})
                self.rows.append({"row_id": row_id, "has_violation": bool(index % 2)})
            proposal = {"kind": "scenario_group_proposal", "proposed_group_id": f"scenario-{index}",
                        "scenario_reviewed": True, "reviewer_id": "synthetic-human"}
            self.review.append({"sample_id": f"sample-{index}", "content_hash": content_hash, "recording_ids": ids,
                                "review_origin": "human", "reviewer_id": "synthetic-human", "review_state": "in_progress",
                                "full_review_complete": False, "revision": 2, "proposals": [proposal]})
            self.audits.append({"sample_id": f"sample-{index}", "revision": 2, "actor": "synthetic-human",
                                "reviewer_id": "synthetic-human", "review_origin": "human", "action": "review_update", "proposals": [copy.deepcopy(proposal)]})
        append_exposures(self.registry_path, self.records, {self.records[0]["content_hash"]}, trial_id="synthetic-baseline", phase="baseline", purpose="benchmark_attempt")
        self.registry = read_registry(self.registry_path, self.records)

    def tearDown(self):
        self.temporary.cleanup()

    def build(self):
        return build_candidate(self.records, self.rows, self.review, self.audits, self.registry)

    def assert_rejected(self, error):
        report, candidate, scenario = self.build()
        self.assertFalse(report["ready"])
        self.assertIn(error, report["errors"])
        self.assertIsNone(candidate)
        self.assertIsNone(scenario)

    def test_complete_review_is_candidate_without_authority(self):
        report, candidate, scenario = self.build()
        self.assertTrue(report["ready"])
        self.assertTrue(candidate["scenario_review_complete"])
        self.assertFalse(candidate["locked"])
        self.assertFalse(candidate["training_authorized"])
        self.assertIsNone(candidate["approved_by"])
        self.assertEqual(scenario["review_state"], "human_confirmed_proposal")

    def test_deterministic_candidate(self):
        self.assertEqual(self.build(), self.build())

    def test_missing_content_review(self):
        self.review.pop()
        self.assert_rejected("missing_review_content")

    def test_unchecked_confirmation(self):
        self.review[0]["proposals"][0]["scenario_reviewed"] = False
        self.assert_rejected("scenario_confirmation_incomplete")

    def test_empty_group(self):
        self.review[0]["proposals"][0]["proposed_group_id"] = " "
        self.assert_rejected("scenario_confirmation_incomplete")

    def test_missing_author(self):
        self.review[0]["proposals"][0]["reviewer_id"] = None
        self.assert_rejected("scenario_confirmation_not_human_attributed")

    def test_model_origin_is_not_human_review(self):
        self.review[0]["review_origin"] = "local_model"
        self.assert_rejected("scenario_confirmation_not_human_attributed")

    def test_audit_proof_required(self):
        self.audits.pop(0)
        self.assert_rejected("scenario_confirmation_audit_missing")

    def test_conflicting_proposals_rejected(self):
        self.review[0]["proposals"].append(copy.deepcopy(self.review[0]["proposals"][0]))
        self.assert_rejected("missing_or_conflicting_scenario_proposal")

    def test_duplicate_content_aliases_must_all_be_present(self):
        self.review[0]["recording_ids"].pop()
        self.assert_rejected("duplicate_recording_alias_coverage_mismatch")

    def test_duplicate_review_content(self):
        self.review.append(copy.deepcopy(self.review[0]))
        self.assert_rejected("duplicate_review_content")

    def test_unknown_review_content(self):
        extra = copy.deepcopy(self.review[0])
        extra["content_hash"] = "f" * 64
        self.review.append(extra)
        self.assert_rejected("unknown_review_content")

    def test_later_notes_editor_may_differ_from_proposal_author(self):
        self.review[0]["reviewer_id"] = "synthetic-later-editor"
        self.review[0]["revision"] = 3
        self.assertTrue(self.build()[0]["ready"])

    def test_wrong_audit_actor_does_not_attest_human_confirmation(self):
        self.audits[0]["actor"] = "local_model"
        self.assert_rejected("scenario_confirmation_audit_missing")

    def test_model_proposals_do_not_count(self):
        self.review[0]["local_model_proposals"] = self.review[0]["proposals"]
        self.review[0]["proposals"] = []
        self.assert_rejected("missing_or_conflicting_scenario_proposal")

    def test_exposure_propagates_to_whole_merged_scenario(self):
        for item in (self.review[1]["proposals"][0], self.audits[1]["proposals"][0]):
            item["proposed_group_id"] = "scenario-0"
        report, candidate, _ = self.build()
        self.assertTrue(report["ready"])
        protected = set(self.review[0]["recording_ids"] + self.review[1]["recording_ids"])
        self.assertTrue(all(row["split"] != "final_check" for row in candidate["records"] if row["recording_id"] in protected))

    def test_insufficient_unseen_scenarios_fails_closed(self):
        self.registry["content_hashes"] = {record["content_hash"] for record in self.records}
        self.assert_rejected("insufficient_unexposed_scenario_groups")

    def test_checker_rejects_exposed_final_check_scenario(self):
        _, candidate, scenario = self.build()
        content_hash = next(row["content_hash"] for row in candidate["records"] if row["split"] == "final_check")
        report = check_split(candidate, self.records, scenario=scenario, excluded_final_hashes={content_hash})
        self.assertFalse(report["ok"])
        self.assertIn("exposed_scenario_in_final_check", report["errors"])

    def test_exposure_append_is_idempotent(self):
        before = self.registry_path.read_bytes()
        result = append_exposures(self.registry_path, self.records, {self.records[0]["content_hash"]}, trial_id="synthetic-baseline", phase="baseline", purpose="benchmark_attempt")
        self.assertEqual(result["events_appended"], 0)
        self.assertEqual(before, self.registry_path.read_bytes())

    def test_later_trial_preserves_old_log_prefix(self):
        before = self.registry_path.read_bytes()
        result = append_exposures(self.registry_path, self.records, {self.records[0]["content_hash"]}, trial_id="synthetic-repeat", phase="paired", purpose="benchmark_attempt")
        self.assertEqual(result["events_appended"], 1)
        self.assertTrue(self.registry_path.read_bytes().startswith(before))
        self.assertEqual(result["distinct_exposed_contents"], 1)

    def test_registry_tamper_detected(self):
        event = json.loads(self.registry_path.read_text())
        event["trial_id"] = "tampered"
        self.registry_path.write_text(json.dumps(event) + "\n")
        with self.assertRaisesRegex(ValueError, "exposure_registry_chain_invalid"):
            read_registry(self.registry_path, self.records)

    def test_unknown_exposure_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "exposure_attempt_unknown_content"):
            append_exposures(self.registry_path, self.records, {"f" * 64}, trial_id="synthetic-extra", phase="paired", purpose="benchmark_attempt")

    def test_sqlite_snapshot_projects_only_scenario_metadata(self):
        database = self.private / "review.sqlite3"
        connection = sqlite3.connect(database)
        connection.executescript("CREATE TABLE records(sample_id TEXT,content_hash TEXT,revision INTEGER,payload TEXT); CREATE TABLE audit(audit_id INTEGER,sample_id TEXT,revision INTEGER,actor TEXT,action TEXT,changed_at TEXT,after_payload TEXT);")
        for index, record in enumerate(self.review):
            payload = {**record, "corrections": record["proposals"], "events": [{"description": "SYNTHETIC_UNRELATED_DESCRIPTION"}], "local_model_proposals": ["SYNTHETIC_MODEL_NARRATIVE"]}
            connection.execute("INSERT INTO records VALUES(?,?,?,?)", (record["sample_id"], record["content_hash"], 2, json.dumps(payload)))
            connection.execute("INSERT INTO audit VALUES(?,?,?,?,?,?,?)", (index, record["sample_id"], 2, "synthetic-human", "review_update", "synthetic-time", json.dumps(payload)))
        connection.commit()
        connection.close()
        records, audits = review_snapshot(database)
        self.assertNotIn("SYNTHETIC_UNRELATED_DESCRIPTION", json.dumps([records, audits]))
        self.assertNotIn("SYNTHETIC_MODEL_NARRATIVE", json.dumps([records, audits]))
        self.assertTrue(analyze_review(self.records, records, audits)[0]["ready"])

    def test_missing_accepted_inventory_coverage_rejected(self):
        _, errors = exposure_coverage(self.private / "missing.json", self.records, self.registry, 300)
        self.assertIn("exposure_inventory_coverage_missing", errors)

    def inventory_fixture(self, *, age_sec=0):
        manifest = self.private / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(record) + "\n" for record in self.records))
        inventory = self.private / "accepted.json"
        hashes = [self.records[0]["content_hash"], self.records[3]["content_hash"], "e" * 64]
        inventory.write_text(json.dumps({"schema_version": "forte.service-content-exposure-inventory.v1", "coverage_state": "complete_snapshot",
                                         "scope": "all_accepted_jobs_since_service_start", "captured_at_utc": (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age_sec)).isoformat(),
                                         "accepted_job_count": 4, "unique_content_count": 3, "content_hashes": hashes, "active_job_count": 0}))
        status = self.private / "status.json"
        result = register_accepted_inventory(inventory, self.registry_path, manifest, status)
        return result, status

    def test_manual_accepted_known_hashes_backfilled_without_unknown_uploads(self):
        result, status = self.inventory_fixture()
        registry = read_registry(self.registry_path, self.records)
        self.assertEqual(result["known_corpus_contents_in_snapshot"], 2)
        self.assertEqual(len(registry["content_hashes"]), 2)
        aggregate, errors = exposure_coverage(status, self.records, registry, 300)
        self.assertFalse(errors)
        self.assertTrue(aggregate["exposure_coverage_complete_snapshot"])

    def test_stale_coverage_fails(self):
        _, status = self.inventory_fixture(age_sec=600)
        _, errors = exposure_coverage(status, self.records, read_registry(self.registry_path, self.records), 300)
        self.assertIn("exposure_inventory_snapshot_stale", errors)

    def test_duplicate_preserving_accepted_job_snapshot(self):
        _, status = self.inventory_fixture()
        source = self.private / "accepted.json"
        payload = json.loads(source.read_text())
        hashes = payload.pop("content_hashes")
        payload["records"] = [{"content_hash": value} for value in [*hashes, hashes[0]]]
        source.write_text(json.dumps(payload))
        result = register_accepted_inventory(source, self.registry_path, self.private / "manifest.jsonl", status)
        self.assertEqual(result["accepted_jobs_in_snapshot"], 4)
        aggregate, errors = exposure_coverage(status, self.records, read_registry(self.registry_path, self.records), 300)
        self.assertFalse(errors)
        self.assertEqual(aggregate["accepted_known_corpus_contents"], 2)

    def test_inventory_file_digest_tamper_fails(self):
        _, status = self.inventory_fixture()
        source = self.private / "accepted.json"
        source.write_text(source.read_text() + " ")
        _, errors = exposure_coverage(status, self.records, read_registry(self.registry_path, self.records), 300)
        self.assertIn("accepted_job_inventory_digest_mismatch", errors)

    def test_selection_preflight_records_only_metadata_and_checks_manifest(self):
        manifest = self.private / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(record) + "\n" for record in self.records))
        items = [{"recording_id": self.review[index]["recording_ids"][0], "content_hash": self.review[index]["content_hash"], "proposed_split": "train", "path": "not-opened-synthetic-media.mov"} for index in range(6)]
        selection = self.private / "selection.json"
        selection.write_text(json.dumps({"selection_scope": "proposed_train_benchmark_only", "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "baseline": items[0], "burst": items[1:]}))
        result = register_selection(selection, self.registry_path, manifest, trial_id="synthetic-selection", phase="paired")
        self.assertEqual(result["distinct_exposed_contents"], 6)
        manifest.write_text(manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "exposure_selection_manifest_digest_mismatch"):
            register_selection(selection, self.registry_path, manifest, trial_id="synthetic-selection-two", phase="paired")

    def locked_gate_fixture(self):
        _, status = self.inventory_fixture()
        self.registry = read_registry(self.registry_path, self.records)
        _, candidate, scenario = self.build()
        candidate.update(state="locked", locked=True, training_authorized=True, approved_by="synthetic-human-approval")
        return candidate, scenario, status

    def gate(self, split, scenario, status, **kwargs):
        return validate_locked_training_inputs(split, self.records, scenario,
                                              expected_assignments_sha256=kwargs.pop("expected_assignments_sha256", split["assignments_sha256"]),
                                              registry_path=self.registry_path, coverage_status_path=status, **kwargs)

    def test_namespace_importable_helpers(self):
        from scripts.build_reviewed_split import exposure_coverage as imported_coverage
        from scripts.split_gate import validate_locked_training_inputs as imported_gate
        self.assertTrue(callable(imported_coverage))
        self.assertTrue(callable(imported_gate))

    def test_valid_synthetic_approved_training_gate(self):
        split, scenario, status = self.locked_gate_fixture()
        content_hash = next(row["content_hash"] for row in split["records"] if row["split"] == "train")
        result = self.gate(split, scenario, status, requested_content_hashes={content_hash})
        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse(result["training_authority_inferred_from_exposure"])
        self.assertFalse(result["target_facts_validated"])

    def test_unlocked_candidate_cannot_gain_training_permission_from_exposure(self):
        split, scenario, status = self.locked_gate_fixture()
        split.update(state="proposed_unlocked", locked=False, training_authorized=False, approved_by=None)
        result = self.gate(split, scenario, status)
        self.assertFalse(result["ok"])
        self.assertIn("split_not_approved_for_training", result["errors"])

    def test_validation_and_final_contents_rejected_for_training(self):
        split, scenario, status = self.locked_gate_fixture()
        for forbidden in ("validation", "final_check"):
            with self.subTest(forbidden=forbidden):
                content_hash = next(row["content_hash"] for row in split["records"] if row["split"] == forbidden)
                result = self.gate(split, scenario, status, requested_content_hashes={content_hash})
                self.assertIn("requested_content_not_approved_train", result["errors"])

    def test_pinned_assignment_digest_prevents_split_promotion(self):
        split, scenario, status = self.locked_gate_fixture()
        approved_digest = split["assignments_sha256"]
        for row in split["records"]:
            if row["split"] == "validation":
                row["split"] = "train"
        split["assignments_sha256"] = hashlib.sha256(json_bytes(split["records"])).hexdigest()
        result = self.gate(split, scenario, status, expected_assignments_sha256=approved_digest)
        self.assertIn("approved_assignment_binding_mismatch", result["errors"])

    def test_later_repeat_exposure_keeps_approved_snapshot_ancestry(self):
        split, scenario, status = self.locked_gate_fixture()
        append_exposures(self.registry_path, self.records, {self.records[0]["content_hash"]}, trial_id="synthetic-postlock-repeat", phase="paired", purpose="benchmark_attempt")
        result = self.gate(split, scenario, status)
        self.assertTrue(result["ok"], result["errors"])

    def test_later_final_check_exposure_invalidates_training_gate(self):
        split, scenario, status = self.locked_gate_fixture()
        content_hash = next(row["content_hash"] for row in split["records"] if row["split"] == "final_check")
        append_exposures(self.registry_path, self.records, {content_hash}, trial_id="synthetic-protected-upload", phase="local-trial", purpose="local_model_attempt")
        result = self.gate(split, scenario, status)
        self.assertIn("exposed_scenario_in_final_check", result["errors"])

    def test_unknown_locked_registry_tip_rejected(self):
        split, scenario, status = self.locked_gate_fixture()
        split["exposure_registry_tip_sha256"] = "f" * 64
        result = self.gate(split, scenario, status)
        self.assertIn("approved_exposure_snapshot_not_ancestor", result["errors"])

    def test_source_inventory_relocation_requires_identical_bytes(self):
        split, scenario, status = self.locked_gate_fixture()
        original = self.private / "accepted.json"
        moved = self.private / "copied-inventory.json"
        moved.write_bytes(original.read_bytes())
        original.unlink()
        self.assertFalse(self.gate(split, scenario, status)["ok"])
        self.assertTrue(self.gate(split, scenario, status, source_inventory_path_override=moved)["ok"])
        moved.write_text(moved.read_text() + " ")
        result = self.gate(split, scenario, status, source_inventory_path_override=moved)
        self.assertIn("accepted_job_inventory_digest_mismatch", result["errors"])


if __name__ == "__main__":
    unittest.main()
