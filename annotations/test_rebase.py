"""Portable path rebasing over synthetic private artifacts only."""

from __future__ import annotations

from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from .cli import main
from .proposals import import_proposal
from .rebase import rebase_proposal
from .store import ReviewError, ReviewStore


class RebaseChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.sources, self.download = self.root / "inputs", self.root / "download"
        self.sources.mkdir(); self.download.mkdir()
        self.source = self.sources / "синтетика.mov"
        self.source.write_bytes(b"unchanged-local-source-synthetic")
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.relative_frame = Path("opaque-job") / "attempt-1" / "frames" / "frame.jpg"
        self.frame = self.download / self.relative_frame
        self.frame.parent.mkdir(parents=True)
        Image.new("RGB", (2, 2), (20, 80, 10)).save(self.frame, "JPEG")
        self.frame_digest = hashlib.sha256(self.frame.read_bytes()).hexdigest()
        self.remote_root = "/var/lib/forte/work"
        self.frame_reference = {"frame_id": "synthetic-frame", "path": self.remote_root + "/" + self.relative_frame.as_posix(), "pts_sec": 4.0, "width": 2, "height": 2, "job_id": "opaque-job", "jpeg_sha256": self.frame_digest}
        self.audit = {
            "schema_version": "forte.local-proposal.v1", "job_id": "opaque-job", "phase": "failed",
            "asset": {"job_id": "opaque-job", "owned_path": "/var/lib/forte/uploads/opaque.mov", "content_hash": self.digest, "duration_sec": 10.0, "width": 2, "height": 2},
            "frames": [deepcopy(self.frame_reference)], "decisions": [], "candidate_events": [], "verified_events": [],
            "window_decisions": [{"clip_id": "synthetic-clip", "start_sec": 3.0, "end_sec": 5.0, "frame_ids": ["synthetic-frame"], "facts": [], "candidate_events": [], "generation": 0, "complete": False, "disposition": "unclear"}],
            "coverage": {"complete": False},
            "provenance": {"execution_location": "team_deployed_service", "external_ai_used": False, "model_id": "synthetic-model", "local_run_id": "opaque-job:1", "model_revision": None},
        }
        self.source_map = {self.digest: self.source}
        self.store = ReviewStore(self.root / "private")
        self.store.seed([{"recording_id": "synthetic-recording", "content_hash": self.digest, "group_id": "synthetic-group", "source_path": str(self.source), "duration_sec": 10.0, "source_role": "synthetic"}], self.sources)
        self.audit_path = self.download / "analysis.private.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def rebase(self, audit: dict | None = None) -> dict:
        return rebase_proposal(self.audit if audit is None else audit, self.source_map, self.remote_root, self.download)

    def test_rebase_is_a_deep_copy_preserving_all_identity_and_timeline_fields(self) -> None:
        original = deepcopy(self.audit)
        source_bytes, frame_bytes = self.source.read_bytes(), self.frame.read_bytes()
        result = self.rebase()
        self.assertEqual(self.audit, original)
        self.assertEqual(result["asset"]["owned_path"], str(self.source))
        self.assertEqual({key: value for key, value in result["asset"].items() if key != "owned_path"}, {key: value for key, value in original["asset"].items() if key != "owned_path"})
        self.assertEqual(result["frames"][0]["path"], str(self.frame))
        self.assertEqual({key: value for key, value in result["frames"][0].items() if key != "path"}, {key: value for key, value in original["frames"][0].items() if key != "path"})
        for key in ("decisions", "candidate_events", "verified_events", "window_decisions", "coverage", "provenance"):
            self.assertEqual(result[key], original[key])
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(self.frame.read_bytes(), frame_bytes)

    def test_import_rebases_in_memory_and_keeps_original_download_hash(self) -> None:
        self.audit_path.write_text(json.dumps(self.audit), encoding="utf-8")
        raw = self.audit_path.read_bytes()
        result = import_proposal(self.store, self.audit_path, proposal_roots=[self.download], source_roots=[self.sources], frame_roots=[self.download], local_source_by_hash=self.source_map, remote_frame_root=self.remote_root, local_frame_root=self.download)
        self.assertEqual(result["proposals_imported"], 1)
        record = self.store.list_records()[0]
        proposal = record["local_model_proposals"][0]
        self.assertEqual(proposal["audit_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(self.audit_path.read_bytes(), raw)
        self.assertEqual(proposal["asset"]["content_hash"], self.digest)
        self.assertEqual(proposal["frames"][0]["jpeg_sha256"], self.frame_digest)
        self.assertIsNone(record["verdict"])
        self.assertEqual(record["review_origin"], "local_model")
        self.assertFalse(record["split_lock_verified"])

    def test_embedded_requests_use_the_same_verified_frame_mapping(self) -> None:
        self.audit["window_decisions"][0]["request"] = {"job_id": "opaque-job", "clip_id": "synthetic-clip", "start_sec": 3.0, "end_sec": 5.0, "frame_refs": [deepcopy(self.frame_reference)]}
        result = self.rebase()
        request_frame = result["window_decisions"][0]["request"]["frame_refs"][0]
        self.assertEqual(request_frame["path"], str(self.frame))
        self.assertEqual(request_frame["frame_id"], "synthetic-frame")
        self.audit["window_decisions"][0]["request"]["frame_refs"][0]["pts_sec"] = 4.5
        with self.assertRaises(ReviewError):
            self.rebase()

    def test_traversal_windows_drives_and_remote_prefix_siblings_are_rejected(self) -> None:
        bad_paths = [
            "/var/lib/forte/work/../escape.jpg",
            "/var/lib/forte/work/./frame.jpg",
            "/var/lib/forte/work-other/frame.jpg",
            "/var/lib/forte/work/C:escape.jpg",
            "/var/lib/forte/work/NUL.jpg",
            "/var/lib/forte/work/folder\\escape.jpg",
            "//var/lib/forte/work/frame.jpg",
            "relative/frame.jpg",
        ]
        for path in bad_paths:
            with self.subTest(kind=bad_paths.index(path)):
                candidate = deepcopy(self.audit)
                candidate["frames"][0]["path"] = path
                with self.assertRaises(ReviewError):
                    self.rebase(candidate)

    def test_local_symlink_escape_is_rejected_when_supported(self) -> None:
        outside = self.root / "outside.jpg"
        outside.write_bytes(self.frame.read_bytes())
        link = self.frame.parent / "linked.jpg"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("local account lacks symlink creation permission")
        self.audit["frames"][0]["path"] = self.remote_root + "/opaque-job/attempt-1/frames/linked.jpg"
        with self.assertRaises(ReviewError):
            self.rebase()

    def test_original_jpeg_checksum_is_required_and_cannot_be_replaced(self) -> None:
        self.audit["frames"][0].pop("jpeg_sha256")
        with self.assertRaises(ReviewError):
            self.rebase()
        self.audit["frames"][0]["jpeg_sha256"] = self.frame_digest
        self.frame.write_bytes(b"downloaded-bytes-were-substituted")
        with self.assertRaises(ReviewError):
            self.rebase()

    def test_source_map_requires_exact_content_and_absolute_local_path(self) -> None:
        with self.assertRaises(ReviewError):
            rebase_proposal(self.audit, {}, self.remote_root, self.download)
        with self.assertRaises(ReviewError):
            rebase_proposal(self.audit, {self.digest: "relative.mov"}, self.remote_root, self.download)
        self.source.write_bytes(b"source-modified")
        with self.assertRaises(ReviewError):
            self.rebase()

    def test_frame_ids_timeline_and_job_binding_are_preserved_or_rejected(self) -> None:
        for field, value in (("job_id", "foreign-job"), ("pts_sec", 11.0)):
            candidate = deepcopy(self.audit)
            candidate["frames"][0][field] = value
            with self.assertRaises(ReviewError):
                self.rebase(candidate)

    def test_duplicate_ids_and_multiple_ids_aliasing_one_local_file_are_rejected(self) -> None:
        self.audit["frames"].append(deepcopy(self.frame_reference))
        with self.assertRaises(ReviewError):
            self.rebase()
        self.audit["frames"][1]["frame_id"] = "different-frame-id"
        with self.assertRaises(ReviewError):
            self.rebase()

    def test_cli_source_map_and_path_map_import_without_printing_records(self) -> None:
        self.audit_path.write_text(json.dumps(self.audit), encoding="utf-8")
        sources, paths = self.root / "sources.json", self.root / "paths.json"
        sources.write_text(json.dumps({self.digest: str(self.source)}), encoding="utf-8")
        paths.write_text(json.dumps({"remote_frame_root": self.remote_root, "local_frame_root": str(self.download)}), encoding="utf-8")
        args = ["--workspace", str(self.store.workspace), "import-proposal", "--input", str(self.audit_path), "--proposal-root", str(self.download), "--source-root", str(self.sources), "--frame-root", str(self.download), "--source-map", str(sources)]
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(main(args), 2)
        self.assertNotIn(str(self.source), output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(args + ["--path-map", str(paths)]), 0)
        self.assertEqual(set(json.loads(output.getvalue())), {"proposals_imported", "already_imported"})


if __name__ == "__main__":
    unittest.main()
