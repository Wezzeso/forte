"""Synthetic private-proposal integration checks. No organizer media is read."""

from __future__ import annotations

import contextlib
import asyncio
import hashlib
import http.client
import io
import json
import re
import shutil
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from .cli import main
from .proposals import import_proposal
from .server import make_server
from .store import ReviewError, ReviewStore, RevisionConflict


class ProposalChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.sources, self.frames, self.audits = (self.root / name for name in ("sources", "frames", "audits"))
        for directory in (self.sources, self.frames, self.audits):
            directory.mkdir()
        self.source = self.sources / "synthetic.mov"
        self.source.write_bytes(b"synthetic-source-not-a-real-recording")
        self.frame = self.frames / "synthetic.jpg"
        Image.new("RGB", (2, 2), (0, 60, 100)).save(self.frame, "JPEG")
        self.content_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.store = ReviewStore(self.root / "private")
        self.store.seed([{"recording_id": "synthetic-recording", "content_hash": self.content_hash, "group_id": "synthetic-group", "relative_path": str(self.source), "duration_sec": 10.0, "source_role": "synthetic"}], self.sources)
        self.record = self.store.list_records()[0]
        self.audit_path = self.audits / "analysis.private.json"
        self.fact = {"fact_id": "CASH_VISIBLE", "state": "present", "supporting_frame_ids": ["synthetic:frame:4"]}
        self.candidate = {"candidate_id": "synthetic-candidate", "event_type_id": "CASH_UNATTENDED", "start_sec": 3.0, "end_sec": 5.0, "supporting_facts": [self.fact], "supporting_frame_ids": ["synthetic:frame:4"], "clip_id": "synthetic-clip"}
        self.payload = {
            "schema_version": "forte.local-proposal.v1", "phase": "failed", "error_code": "insufficient_visible_evidence",
            "job_id": "synthetic-job", "attempt": 1, "publication_ready": False,
            "asset": {"job_id": "synthetic-job", "owned_path": str(self.source), "content_hash": self.content_hash, "duration_sec": 10.0, "width": 2, "height": 2, "fps": 30.0},
            "frames": [{"frame_id": "synthetic:frame:4", "path": str(self.frame), "pts_sec": 4.0, "width": 2, "height": 2, "job_id": "synthetic-job", "jpeg_sha256": hashlib.sha256(self.frame.read_bytes()).hexdigest()}],
            "decisions": [self.fact], "candidate_events": [self.candidate], "verified_events": [],
            "window_decisions": [{"clip_id": "synthetic-clip", "start_sec": 3.0, "end_sec": 5.0, "frame_ids": ["synthetic:frame:4"], "facts": [self.fact], "candidate_events": [self.candidate], "generation": 0, "complete": False, "disposition": "unclear"}],
            "coverage": {"complete": False}, "unresolved_intervals": [{"start_sec": 3.0, "end_sec": 5.0, "reason": "synthetic-incomplete-context"}],
            "provenance": {"execution_location": "team_deployed_service", "external_ai_used": False, "model_id": "synthetic-local-model", "model_revision": None, "processor_version": None, "local_run_id": "synthetic-job:1", "actual_snapshot_verified": None},
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self) -> None:
        self.audit_path.write_text(json.dumps(self.payload), encoding="utf-8")

    def ingest(self, **options: object) -> dict:
        self.write()
        return import_proposal(self.store, self.audit_path, proposal_roots=[self.audits], source_roots=[self.sources], frame_roots=[self.frames], **options)

    def test_incomplete_proposal_is_private_unreviewed_and_not_a_label(self) -> None:
        self.assertEqual(self.ingest()["proposals_imported"], 1)
        updated = self.store.get(self.record["sample_id"])
        self.assertEqual(updated["review_state"], "in_progress")
        self.assertEqual(updated["review_origin"], "local_model")
        self.assertIsNone(updated["reviewer_id"])
        self.assertIsNone(updated["reviewed_at"])
        self.assertIsNone(updated["verdict"])
        self.assertFalse(updated["full_review_complete"])
        self.assertFalse(updated["split_lock_verified"])
        for field in ("events", "negative_windows", "frames", "source_records", "split"):
            self.assertEqual(updated[field], self.record[field])
        proposal = updated["local_model_proposals"][0]
        self.assertFalse(proposal["reviewed"])
        self.assertEqual(proposal["phase"], "failed")
        self.assertEqual(proposal["window_decisions"], self.payload["window_decisions"])
        self.assertIsNone(proposal["provenance"]["model_revision"])
        self.assertEqual(self.store.export_reviewed(self.store.workspace / "reviewed.jsonl")["exported_reviewed_records"], 0)

    def test_import_is_idempotent_by_audit_bytes(self) -> None:
        self.ingest()
        revision = self.store.get(self.record["sample_id"])["revision"]
        self.assertEqual(self.ingest(), {"proposals_imported": 0, "already_imported": 1})
        self.assertEqual(self.store.get(self.record["sample_id"])["revision"], revision)

    def test_cli_prints_only_aggregate_counters(self) -> None:
        self.write()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["--workspace", str(self.store.workspace), "import-proposal", "--input", str(self.audit_path), "--proposal-root", str(self.audits), "--source-root", str(self.sources), "--frame-root", str(self.frames)])
        self.assertEqual(status, 0)
        self.assertEqual(set(json.loads(output.getvalue())), {"proposals_imported", "already_imported"})

    def test_source_content_must_match_seeded_identity(self) -> None:
        self.source.write_bytes(b"modified-source")
        with self.assertRaises(ReviewError):
            self.ingest()

    def test_explicit_sample_and_job_bindings_cannot_disagree(self) -> None:
        with self.assertRaises(ReviewError):
            self.ingest(sample_id="foreign-review-unit")
        self.payload["job_id"] = "foreign-job"
        with self.assertRaises(ReviewError):
            self.ingest()

    def test_audit_source_and_frames_require_explicit_roots(self) -> None:
        self.write()
        for key in ("proposal_roots", "source_roots", "frame_roots"):
            parameters = {"proposal_roots": [self.audits], "source_roots": [self.sources], "frame_roots": [self.frames]}
            parameters[key] = [self.store.workspace]
            with self.assertRaises(ReviewError):
                import_proposal(self.store, self.audit_path, **parameters)

    def test_relative_frame_path_is_rejected(self) -> None:
        self.payload["frames"][0]["path"] = "synthetic.jpg"
        with self.assertRaises(ReviewError):
            self.ingest()

    def test_foreign_job_frame_is_rejected(self) -> None:
        self.payload["frames"][0]["job_id"] = "foreign-job"
        with self.assertRaises(ReviewError):
            self.ingest()

    def test_changed_frame_checksum_and_dimensions_are_rejected(self) -> None:
        self.payload["frames"][0]["jpeg_sha256"] = "0" * 64
        with self.assertRaises(ReviewError):
            self.ingest()
        self.payload["frames"][0]["jpeg_sha256"] = hashlib.sha256(self.frame.read_bytes()).hexdigest()
        self.payload["frames"][0]["width"] = 100
        with self.assertRaises(ReviewError):
            self.ingest()

    def test_foreign_fact_or_window_frames_are_rejected(self) -> None:
        self.payload["window_decisions"][0]["frame_ids"] = ["foreign-frame"]
        with self.assertRaises(ReviewError):
            self.ingest()
        self.payload["window_decisions"][0]["frame_ids"] = ["synthetic:frame:4"]
        self.fact["supporting_frame_ids"] = ["foreign-frame"]
        with self.assertRaises(ReviewError):
            self.ingest()

    def test_external_execution_provenance_is_rejected(self) -> None:
        self.payload["provenance"]["external_ai_used"] = True
        with self.assertRaises(ReviewError):
            self.ingest()

    def test_human_work_and_reviewed_records_are_never_replaced(self) -> None:
        self.store.save(self.record["sample_id"], {"review_state": "in_progress", "review_origin": "human", "reviewer_id": "synthetic-human"}, expected_revision=self.record["revision"])
        with self.assertRaises(ReviewError):
            self.ingest()
        self.assertEqual(self.store.get(self.record["sample_id"])["reviewer_id"], "synthetic-human")

    def test_raw_proposals_never_enter_reviewed_training_export(self) -> None:
        self.ingest()
        record = self.store.get(self.record["sample_id"])
        self.store.save(record["sample_id"], {"review_state": "reviewed", "review_origin": "human", "reviewer_id": "synthetic-human", "unresolved_items": []}, expected_revision=record["revision"])
        locked = self.root / "synthetic-locked-split.json"
        locked.write_text(json.dumps({"locked": True, "scenario_groups_reviewed": True, "approved_by": "synthetic-human", "training_authorized": True, "records": [{"content_hash": self.content_hash, "group_id": "synthetic-group", "split": "train"}]}), encoding="utf-8")
        self.store.sync_splits(locked)
        output = self.store.workspace / "reviewed.jsonl"
        self.store.export_reviewed(output)
        exported = json.loads(output.read_text(encoding="utf-8"))
        self.assertNotIn("local_model_proposals", exported)
        self.assertNotIn("local_model_provenance", exported)
        self.assertFalse(exported["negative_verified"])
        self.payload["attempt"] = 2
        with self.assertRaises(ReviewError):
            self.ingest()

    def test_concurrent_browser_save_blocks_stale_import(self) -> None:
        self.store.save(self.record["sample_id"], {"review_state": "in_progress"}, expected_revision=self.record["revision"])
        with self.assertRaises(RevisionConflict):
            self.ingest(expected_revision=self.record["revision"])

    def test_duplicate_fact_ids_across_windows_remain_separately_bound(self) -> None:
        self.payload["decisions"].append(self.fact)
        self.payload["window_decisions"].append(dict(self.payload["window_decisions"][0]))
        self.ingest()
        stored = self.store.get(self.record["sample_id"])["local_model_proposals"][0]
        self.assertEqual(len(stored["window_decisions"]), 2)

    def test_proposal_frame_preview_is_session_protected_and_checksum_bound(self) -> None:
        self.ingest()
        server = make_server(self.store, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            token = re.search(r'name="review-token" content="([^"]+)"', response.read().decode()).group(1)
            route = f"/frames/{self.record['sample_id']}/synthetic%3Aframe%3A4"
            connection.request("GET", route)
            response = connection.getresponse(); response.read()
            self.assertEqual(response.status, 403)
            connection.request("GET", route + "?token=" + token)
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), self.frame.read_bytes())
            self.frame.write_bytes(b"changed-after-import")
            connection.request("GET", route + "?token=" + token)
            response = connection.getresponse(); response.read()
            self.assertEqual(response.status, 400)
        finally:
            connection.close()
            server.shutdown(); server.server_close(); thread.join(timeout=3)

    @unittest.skipUnless(shutil.which("ffprobe"), "requires a local ffprobe for synthetic pipeline integration")
    def test_actual_pipeline_failure_audit_imports_without_becoming_negative(self) -> None:
        try:
            import av
            from forte.config import WorkerSettings
            from forte.pipeline import AnalysisIncomplete, AnalysisPipeline
        except ImportError:
            self.skipTest("full pipeline dependencies unavailable in this interpreter")
        directory = self.root / "synthetic-pipeline"
        directory.mkdir()
        source = directory / "generated.mov"
        with av.open(str(source), "w") as output:
            stream = output.add_stream("mpeg4", rate=4)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for index in range(4):
                with Image.new("RGB", (64, 48), (index * 20, 50, 100)) as image:
                    frame = av.VideoFrame.from_image(image)
                    frame.pts, frame.time_base = index, Fraction(1, 4)
                    for packet in stream.encode(frame):
                        output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.store.seed([{"recording_id": "synthetic-pipeline-recording", "content_hash": digest, "group_id": "synthetic-pipeline-group", "relative_path": str(source), "duration_sec": 1.0, "source_role": "synthetic"}], directory)
        settings = WorkerSettings(database_url="sqlite://", data_dir=directory / "runtime", model_base_url="http://127.0.0.1:1/v1", model_name="synthetic-no-inference", model_api_key=None, worker_id="synthetic", heartbeat_file=directory / "heartbeat", rules_path=Path("rules/catalog.json"), taxonomy_path=Path("annotations/taxonomy.v1.json"), model_manifest_path=Path("models/manifest.json"), trusted_scene_context="bank_cash_workstation")

        class FailingProvider:
            async def infer(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError("synthetic local provider failure")

        async def run() -> None:
            pipeline = AnalysisPipeline(settings, provider=FailingProvider())
            async def stage(_value: str) -> None:
                pass
            try:
                with self.assertRaises((AnalysisIncomplete, RuntimeError)):
                    await pipeline.analyze(SimpleNamespace(job_id="synthetic-proposal", attempt=1, owned_path=source, content_hash=digest), stage=stage, deadline=time.monotonic() + 30, cancel_event=threading.Event())
            finally:
                await pipeline.aclose()

        asyncio.run(run())
        audit = settings.work_dir / "synthetic-proposal" / "attempt-1" / "analysis.private.json"
        result = import_proposal(self.store, audit, proposal_roots=[settings.work_dir], source_roots=[directory], frame_roots=[settings.work_dir])
        self.assertEqual(result["proposals_imported"], 1)
        record = next(row for row in self.store.list_records() if row["content_hash"] == digest)
        self.assertEqual(record["review_state"], "in_progress")
        self.assertIsNone(record["verdict"])
        self.assertEqual(record["negative_windows"], [])


if __name__ == "__main__":
    unittest.main()
