"""Synthetic behavioral checks. Never reads organizer videos or weak labels."""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import tempfile
import threading
import unittest
from pathlib import Path

from .catalog import event_definitions, load_taxonomy
from .server import make_server
from .store import ReviewError, ReviewStore, RevisionConflict


class WorkflowChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.media = self.root / "synthetic.mov"
        self.media.write_bytes(b"synthetic-local-test-media-only")
        self.frame = self.root / "synthetic-frame.jpg"
        self.frame.write_bytes(b"synthetic-frame-reference-only")
        self.store = ReviewStore(self.root / "private")
        self.content_hash = hashlib.sha256(self.media.read_bytes()).hexdigest()
        self.row = {"recording_id": "synthetic-recording", "content_hash": self.content_hash, "group_id": "synthetic-group", "relative_path": self.media.name, "duration_sec": 10.0, "split": "unassigned", "source_role": "synthetic", "original_weak_labels": [{"timestamp_raw": "original-unreviewed-synthetic-value"}]}
        self.store.seed([self.row], self.root)
        self.record = self.store.list_records()[0]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def review_patch(self) -> dict:
        return {"review_state": "reviewed", "review_origin": "human", "reviewer_id": "synthetic-human", "unresolved_items": [], "events": [], "negative_windows": [], "full_review_complete": False, "verdict": None}

    def negative(self, start: float = 0.0, end: float = 10.0) -> dict:
        return {"start_sec": start, "end_sec": end, "visibility": "clear", "context_complete": True, "negative_verified": True, "criteria_version": "forte.taxonomy.v1"}

    def event_patch(self) -> dict:
        event_id = "CASH_UNATTENDED"
        patch = self.review_patch()
        patch["frames"] = [{"frame_id": "synthetic-frame", "path": str(self.frame), "pts_sec": 4.0, "width": 2, "height": 2}]
        patch["events"] = [{"event_id": "synthetic-event", "event_type_id": event_id, "start_sec": 3.0, "end_sec": 5.0, "anchor_sec": 4.0, "rule_id": None, "rule_grounding_status": "unresolved", "evidence_frame_ids": ["synthetic-frame"], "facts": [{"fact_id": fid, "state": state, "supporting_frame_ids": ["synthetic-frame"]} for fid, state in event_definitions()[event_id]["required_fact_states"].items()]}]
        return patch

    def save(self, patch: dict) -> dict:
        return self.store.save(self.record["sample_id"], patch, expected_revision=self.record["revision"])

    def lock(self, **overrides: object) -> dict:
        manifest = {"locked": True, "approved_by": "synthetic-human", "scenario_review_complete": True, "training_authorized": True, "records": [{"content_hash": self.content_hash, "group_id": "synthetic-group", "split": "train"}]}
        manifest.update(overrides)
        path = self.root / "synthetic-split.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.store.sync_splits(path)
        self.record = self.store.get(self.record["sample_id"])
        return manifest

    def test_catalog_ids_are_unique_and_references_resolve(self) -> None:
        catalog = load_taxonomy()
        self.assertEqual(len(catalog["facts"]), 21)
        self.assertTrue(all("observable_phrase_ru" in row and "absence_phrase_ru" in row for row in catalog["facts"]))
        omission = event_definitions()["CASH_TRANSFER_WITHOUT_OBSERVED_COUNT"]
        self.assertTrue(omission["is_omission"])
        self.assertEqual(omission["required_fact_states"]["PROCEDURE_CONTEXT_COMPLETE"], "present")

    def test_seed_keeps_every_clip_pending_and_preserves_originals(self) -> None:
        self.assertEqual(self.store.aggregate()["review_states"], {"pending": 1})
        self.assertEqual(self.record["source_records"][0]["original_weak_labels"], self.row["original_weak_labels"])
        self.assertEqual(self.record["events"], [])
        self.assertEqual(self.record["negative_windows"], [])
        result = self.store.export_reviewed(self.store.workspace / "reviewed.jsonl")
        self.assertEqual(result["exported_reviewed_records"], 0)

    def test_duplicates_share_one_review_unit(self) -> None:
        duplicate = {**self.row, "recording_id": "synthetic-duplicate"}
        result = self.store.seed([duplicate], self.root)
        self.assertEqual(result["duplicate_recordings_linked"], 1)
        self.assertEqual(self.store.aggregate()["unique_clips"], 1)
        self.assertEqual(self.store.aggregate()["recordings"], 2)

    def test_duplicate_content_cannot_cross_splits(self) -> None:
        with self.assertRaises(ReviewError):
            self.store.seed([{**self.row, "recording_id": "synthetic-duplicate", "split": "test"}], self.root)
        self.assertEqual(self.store.aggregate()["recordings"], 1)

    def test_manifest_cannot_escape_allowed_media_root(self) -> None:
        with self.assertRaises(ReviewError):
            self.store.seed([{**self.row, "source_path": str(self.root.parent / "unavailable.mov")}], self.root)

    def test_original_identity_is_immutable(self) -> None:
        with self.assertRaises(ReviewError):
            self.save({"content_hash": "a" * 64})

    def test_machine_proposal_cannot_mark_itself_reviewed(self) -> None:
        with self.assertRaises(ReviewError):
            self.save({**self.review_patch(), "review_origin": "local_model"})

    def test_concurrent_edits_require_latest_revision(self) -> None:
        self.save({"review_state": "in_progress"})
        with self.assertRaises(RevisionConflict):
            self.save({"review_state": "pending"})

    def test_negative_whole_video_requires_full_verified_coverage(self) -> None:
        with self.assertRaises(ReviewError):
            self.save({**self.review_patch(), "verdict": "no_violation", "full_review_complete": True, "negative_windows": [self.negative(0, 4)]})

    def test_unknown_visibility_cannot_be_verified_negative(self) -> None:
        with self.assertRaises(ReviewError):
            self.save({**self.review_patch(), "negative_windows": [{**self.negative(), "visibility": "unobservable"}]})

    def test_reviewed_complete_negative_exports_only_after_split_lock(self) -> None:
        updated = self.save({**self.review_patch(), "verdict": "no_violation", "full_review_complete": True, "negative_windows": [self.negative()]})
        self.assertEqual(updated["review_status"], "human_reviewed")
        self.assertEqual(self.store.export_reviewed(self.store.workspace / "before.jsonl")["exported_reviewed_records"], 0)
        self.lock()
        self.assertEqual(self.store.export_reviewed(self.store.workspace / "after.jsonl")["exported_reviewed_records"], 1)

    def test_pending_or_unlocked_split_is_rejected(self) -> None:
        with self.assertRaises(ReviewError):
            self.lock(locked=False)
        with self.assertRaises(ReviewError):
            self.lock(scenario_review_complete=False)

    def test_training_split_requires_explicit_authorization(self) -> None:
        with self.assertRaises(ReviewError):
            self.lock(training_authorized=False)

    def test_final_check_and_explicit_scenario_review_alias_are_supported(self) -> None:
        manifest = {"locked": True, "approved_by": "synthetic-human", "scenario_groups_reviewed": True, "records": [{"content_hash": self.content_hash, "group_id": "synthetic-group", "split": "final_check"}]}
        path = self.root / "synthetic-split.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.store.sync_splits(path)
        self.assertEqual(self.store.get(self.record["sample_id"])["split"], "final_check")
        manifest["scenario_review_complete"] = False
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ReviewError):
            self.store.sync_splits(path)

    def test_jury_data_never_becomes_reviewed_training_data(self) -> None:
        other = self.root / "jury-synthetic.mov"
        other.write_bytes(b"jury-synthetic-bytes")
        row = {**self.row, "recording_id": "jury-synthetic", "relative_path": other.name, "content_hash": hashlib.sha256(other.read_bytes()).hexdigest(), "source_role": "jury_test"}
        self.store.seed([row], self.root)
        jury = next(record for record in self.store.list_records() if record["source_role"] == "jury_test")
        with self.assertRaises(ReviewError):
            self.store.save(jury["sample_id"], self.review_patch(), expected_revision=jury["revision"])

    def test_reviewed_event_may_remain_explicitly_ungrounded(self) -> None:
        updated = self.save(self.event_patch())
        self.assertIsNone(updated["verdict"])
        self.assertEqual(updated["events"][0]["rule_grounding_status"], "unresolved")

    def test_positive_verdict_requires_verified_applicable_rule(self) -> None:
        with self.assertRaises(ReviewError):
            self.save({**self.event_patch(), "verdict": "violation"})

    def test_verified_rule_label_cannot_reference_invented_clause(self) -> None:
        patch = self.event_patch()
        patch["events"][0].update(rule_id="invented-synthetic-rule", rule_grounding_status="verified_applicable", temporal_verified=True, exceptions_verified=True)
        with self.assertRaises(ReviewError):
            self.save({**patch, "verdict": "violation"})

    def test_grounded_positive_uses_exact_enabled_a04_binding(self) -> None:
        from rules.catalog import load_catalog
        catalog = load_catalog()
        rule = next(rule for rule in catalog.enabled_rules() if catalog.bindings[rule.rule_id]["event_type_id"] == "CASH_UNATTENDED")
        patch = self.event_patch()
        event = patch["events"][0]
        states = {**event_definitions()[event["event_type_id"]]["required_fact_states"], **catalog.bindings[rule.rule_id]["required_fact_states"]}
        event.update(rule_id=rule.rule_id, rule_grounding_status="verified_applicable", temporal_verified=True, exceptions_verified=True)
        event["facts"] = [{"fact_id": fid, "state": state, "supporting_frame_ids": ["synthetic-frame"]} for fid, state in states.items()]
        updated = self.save({**patch, "verdict": "violation"})
        self.assertEqual(updated["verdict"], "violation")

    def test_partial_negative_window_never_marks_whole_record_negative(self) -> None:
        self.lock()
        self.save({**self.event_patch(), "negative_windows": [self.negative(0, 2)]})
        output = self.store.workspace / "reviewed.jsonl"
        self.store.export_reviewed(output)
        exported = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(exported["negative_verified"])

    def test_unknown_source_frame_id_is_rejected(self) -> None:
        patch = self.event_patch()
        patch["events"][0]["facts"][0]["supporting_frame_ids"] = ["unknown-frame"]
        with self.assertRaises(ReviewError):
            self.save(patch)

    def test_missing_frames_and_out_of_range_times_are_rejected(self) -> None:
        patch = self.event_patch()
        patch["events"][0]["end_sec"] = 11.0
        with self.assertRaises(ReviewError):
            self.save(patch)
        patch = self.event_patch()
        patch["frames"][0]["path"] = str(self.root / "missing.jpg")
        with self.assertRaises(ReviewError):
            self.save(patch)

    def test_negative_windows_cannot_cover_positive_intervals(self) -> None:
        with self.assertRaises(ReviewError):
            self.save({**self.event_patch(), "negative_windows": [self.negative()]})

    def test_export_cannot_escape_private_workspace(self) -> None:
        with self.assertRaises(ReviewError):
            self.store.export_reviewed(self.root / "outside.jsonl")

    def test_audit_retains_original_and_changed_payloads(self) -> None:
        self.save({"review_state": "in_progress", "corrections": [{"reason": "synthetic correction", "original": "unchanged source value", "corrected": None}]})
        with self.store.connect() as db:
            rows = db.execute("SELECT before_payload,after_payload FROM audit ORDER BY audit_id").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(json.loads(rows[-1]["before_payload"])["review_state"], "pending")
        self.assertEqual(json.loads(rows[-1]["after_payload"])["source_records"], self.record["source_records"])

    def test_workbench_blocks_cross_origin_and_supports_byte_ranges(self) -> None:
        server = make_server(self.store, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            html = response.read().decode("utf-8")
            token = re.search(r'name="review-token" content="([^"]+)"', html).group(1)
            self.assertEqual(response.status, 200)
            self.assertIn("frame-ancestors 'none'", response.getheader("Content-Security-Policy"))
            connection.request("GET", "/api/queue")
            response = connection.getresponse(); response.read()
            self.assertEqual(response.status, 403)
            connection.request("GET", "/api/queue", headers={"X-Review-Token": token, "Origin": "https://external.invalid"})
            response = connection.getresponse(); response.read()
            self.assertEqual(response.status, 403)
            connection.request("GET", "/api/queue", headers={"X-Review-Token": token, "Host": "external.invalid"})
            response = connection.getresponse(); response.read()
            self.assertEqual(response.status, 403)
            connection.request("GET", f"/media/{self.record['sample_id']}?token={token}", headers={"Range": "bytes=2-6"})
            response = connection.getresponse()
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), self.media.read_bytes()[2:7])
            connection.request("GET", f"/media/{self.record['sample_id']}?token={token}", headers={"Range": "bytes=999999-"})
            response = connection.getresponse(); response.read()
            self.assertEqual(response.status, 416)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
