"""Synthetic transport tests. No SSH server or organizer media is contacted."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from .fetch import _extract, _handoff, fetch_proposal, ssh_command
from . import _remote_export
from .store import ReviewError, ReviewStore


class FetchChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.source = self.sources / "synthetic.mov"
        self.source.write_bytes(b"synthetic-original-read-only-input")
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        image = io.BytesIO()
        Image.new("RGB", (2, 2), (40, 20, 10)).save(image, "JPEG")
        self.jpeg = image.getvalue()
        self.jpeg_hash = hashlib.sha256(self.jpeg).hexdigest()
        self.job = "synthetic-job"
        self.audit = {"schema_version": "forte.local-proposal.v1", "job_id": self.job, "phase": "failed", "asset": {"job_id": self.job, "owned_path": "/team/uploads/source.mov", "content_hash": self.digest, "duration_sec": 10.0, "width": 2, "height": 2}, "frames": [{"frame_id": "synthetic-frame", "job_id": self.job, "path": "/team/work/synthetic-job/attempt-1/frames/frame.jpg", "pts_sec": 4.0, "width": 2, "height": 2, "jpeg_sha256": self.jpeg_hash}], "decisions": [], "candidate_events": [], "verified_events": [], "window_decisions": [], "coverage": {"complete": False}, "provenance": {"execution_location": "team_deployed_service", "external_ai_used": False, "model_id": "synthetic-model", "model_revision": None, "local_run_id": "synthetic-job:1"}}
        self.handoff = {"schema_version": "forte.proposal-artifact-handoff.v1", "terminal_verified": True, "artifact_available": True, "terminal_status": "failed", "job_id": self.job, "remote_work_root": "/team/work", "remote_frame_root": "/team/work", "remote_audit_path": "/team/work/synthetic-job/attempt-1/analysis.private.json", "ssh_target": "synthetic-test-only"}
        self.handoff_path = self.root / "handoff.private.json"
        self.store = ReviewStore(self.root / "private")
        self.store.seed([{"recording_id": "synthetic-recording", "content_hash": self.digest, "group_id": "synthetic-group", "source_path": str(self.source), "duration_sec": 10.0, "source_role": "synthetic"}], self.sources)
        self.transport_calls = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_handoff(self) -> None:
        self.handoff_path.write_text(json.dumps(self.handoff), encoding="utf-8")

    @staticmethod
    def add(archive, name, data, member_type=None):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        if member_type:
            info.type = member_type
            info.linkname = "../outside"
        archive.addfile(info, io.BytesIO(data))

    def archive(self, output, *, mutated_jpeg=None, extra=None):
        raw = json.dumps(self.audit).encode("utf-8")
        path = "frames/synthetic-job/attempt-1/frames/frame.jpg"
        manifest = {"schema_version": "forte.proposal-transfer.v1", "job_id": self.job, "audit_sha256": hashlib.sha256(raw).hexdigest(), "frame_count": 1, "total_payload_bytes": len(raw) + len(self.jpeg), "files": [{"path": path, "size": len(self.jpeg), "sha256": self.jpeg_hash}]}
        with tarfile.open(fileobj=output, mode="w|") as archive:
            self.add(archive, "analysis.private.json", raw)
            self.add(archive, path, self.jpeg if mutated_jpeg is None else mutated_jpeg)
            self.add(archive, "transfer.json", json.dumps(manifest).encode("utf-8"))
            if extra:
                self.add(archive, *extra)

    def transport(self, handoff, config, output, stderr):
        self.transport_calls += 1
        self.assertEqual(handoff["terminal_verified"], True)
        self.assertEqual(config["job_id"], self.job)
        self.archive(output)

    def fetch(self, transport=None):
        self.write_handoff()
        return fetch_proposal(self.store, self.handoff_path, transport=transport or self.transport)

    def test_terminal_handoff_is_required_before_any_transport(self) -> None:
        for field, invalid in (("terminal_verified", False), ("artifact_available", False), ("terminal_status", "processing")):
            original = self.handoff[field]
            self.handoff[field] = invalid
            with self.assertRaises(ReviewError):
                self.fetch()
            self.handoff[field] = original
        self.assertEqual(self.transport_calls, 0)

    def test_terminal_teacher_process_can_export_intentionally_partial_audit(self) -> None:
        self.handoff["terminal_status"] = "done"
        self.handoff["remote_audit_path"] = "/team/work/synthetic-job/teacher.private.json"
        self.audit["phase"] = "partial"
        result = self.fetch()
        self.assertEqual(result["proposals_imported"], 1)
        record = self.store.list_records()[0]
        self.assertEqual(record["local_model_proposals"][0]["phase"], "partial")
        self.assertIsNone(record["verdict"])
        self.assertFalse(record["full_review_complete"])

    def test_private_download_verifies_rebases_and_imports_only_a_proposal(self) -> None:
        result = self.fetch()
        self.assertEqual(result["transfer_status"], "verified")
        self.assertEqual(result["downloaded_frames"], 1)
        self.assertEqual(result["proposals_imported"], 1)
        self.assertEqual(self.transport_calls, 1)
        record = self.store.list_records()[0]
        self.assertEqual(record["review_state"], "in_progress")
        self.assertEqual(record["review_origin"], "local_model")
        self.assertIsNone(record["verdict"])
        self.assertEqual(record["events"], [])
        self.assertEqual(record["negative_windows"], [])
        self.assertFalse(record["split_lock_verified"])
        proposal = record["local_model_proposals"][0]
        self.assertEqual(proposal["asset"]["owned_path"], str(self.source))
        self.assertTrue(Path(proposal["frames"][0]["path"]).is_relative_to(self.store.workspace))

    def test_repeated_identical_handoff_is_import_idempotent(self) -> None:
        self.fetch()
        result = self.fetch()
        self.assertEqual(result["proposals_imported"], 0)
        self.assertEqual(result["already_imported"], 1)
        self.assertEqual(self.store.aggregate()["local_proposal_count"], 1)

    def test_expected_audit_hash_mismatch_prevents_import(self) -> None:
        self.handoff["audit_sha256"] = "0" * 64
        with self.assertRaises(ReviewError):
            self.fetch()
        self.assertEqual(self.store.aggregate()["local_proposal_count"], 0)

    def test_corrupted_downloaded_jpeg_prevents_import(self) -> None:
        def corrupt(handoff, config, output, stderr):
            self.archive(output, mutated_jpeg=b"corrupted-transfer")
        with self.assertRaises(ReviewError):
            self.fetch(corrupt)
        self.assertEqual(self.store.aggregate()["review_states"], {"pending": 1})

    def test_archives_cannot_escape_or_introduce_links_or_unreferenced_files(self) -> None:
        extras = [("../outside", b"bad"), ("frames/link.jpg", b"", tarfile.SYMTYPE), ("frames/extra.jpg", b"unreferenced")]
        for index, extra in enumerate(extras):
            with self.subTest(kind=index):
                archive_path = self.root / f"bad-{index}.tar"
                with archive_path.open("wb") as output:
                    self.archive(output, extra=extra)
                staging = self.root / f"staging-{index}"
                staging.mkdir()
                with self.assertRaises(ReviewError):
                    _extract(archive_path, staging, expected_job_id=self.job)
                self.assertFalse((self.root / "outside").exists())

    def test_download_does_not_replace_existing_human_work(self) -> None:
        record = self.store.list_records()[0]
        self.store.save(record["sample_id"], {"review_state": "in_progress", "review_origin": "human", "reviewer_id": "synthetic-human"}, expected_revision=record["revision"])
        with self.assertRaisesRegex(ReviewError, "artifact_verified_private_download_retained"):
            self.fetch()
        self.assertEqual(self.store.list_records()[0]["reviewer_id"], "synthetic-human")
        self.assertEqual(self.store.aggregate()["local_proposal_count"], 0)
        self.assertEqual(len(list((self.store.workspace / "proposals").glob("*/analysis.private.json"))), 1)

    def test_invalid_remote_roots_and_ssh_options_are_rejected(self) -> None:
        for field, invalid in (("ssh_target", "-oProxyCommand=bad"), ("remote_work_root", "/"), ("remote_audit_path", "/team/other/analysis.private.json"), ("remote_frame_root", "/team/work/../escape")):
            candidate = deepcopy(self.handoff)
            candidate[field] = invalid
            with self.assertRaises(ReviewError):
                _handoff(candidate)

    def test_wsl_ssh_uses_existing_alias_and_fixed_remote_container_command(self) -> None:
        self.handoff.update(ssh_wsl_distribution="Ubuntu", remote_container="synthetic-worker", remote_docker_sudo=True, ssh_identity_path="/home/synthetic/.ssh/id", remote_python="/opt/venv/bin/python")
        verified = _handoff(self.handoff)
        command = ssh_command(verified)
        self.assertEqual(command[1:5], ["--distribution", "Ubuntu", "--exec", "ssh"])
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertNotIn("-p", command, "an omitted port must preserve the observed SSH alias configuration")
        self.assertIn("/home/synthetic/.ssh/id", command)
        remote = shlex.split(command[-1])
        self.assertEqual(remote[:7], ["sudo", "-n", "docker", "exec", "-i", "synthetic-worker", "/opt/venv/bin/python"])
        self.assertNotIn(self.handoff["remote_audit_path"], command[-1])
        explicit = ssh_command({**verified, "ssh_port": 2222})
        self.assertEqual(explicit[explicit.index("-p") + 1], "2222")

    def test_fixed_exporter_streams_only_the_registered_audit_and_jpegs(self) -> None:
        # Only POSIX->Windows fixture path lookup is adapted here; actual fixed
        # exporter checksum/count/archive logic runs unchanged, with no network.
        remote_fixture = self.root / "remote-fixture"
        frame = remote_fixture / "synthetic-job" / "attempt-1" / "frames" / "frame.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(self.jpeg)
        audit = remote_fixture / "synthetic-job" / "attempt-1" / "analysis.private.json"
        audit.write_text(json.dumps(self.audit), encoding="utf-8")

        def synthetic_posix_lookup(value, root=None, directory=False):
            relative = Path(*value.removeprefix("/team/work").strip("/").split("/")) if value != "/team/work" else Path()
            result = (remote_fixture / relative).resolve()
            if root is not None and not result.is_relative_to(root):
                raise ValueError("synthetic root escape")
            return result

        config = {key: self.handoff[key] for key in ("job_id", "remote_work_root", "remote_frame_root", "remote_audit_path")}
        config.update(max_audit_bytes=1024*1024, max_frame_bytes=1024*1024, max_total_bytes=4*1024*1024, max_frames=8)
        output = io.BytesIO()
        with patch.object(_remote_export, "_path", synthetic_posix_lookup):
            _remote_export.export(config, output)
        with tarfile.open(fileobj=io.BytesIO(output.getvalue()), mode="r:") as archive:
            self.assertEqual(set(archive.getnames()), {"analysis.private.json", "transfer.json", "frames/synthetic-job/attempt-1/frames/frame.jpg"})
            self.assertEqual(archive.extractfile("analysis.private.json").read(), audit.read_bytes())
        frame.write_bytes(b"changed-before-export")
        with patch.object(_remote_export, "_path", synthetic_posix_lookup):
            with self.assertRaises(ValueError):
                _remote_export.export(config, io.BytesIO())


if __name__ == "__main__":
    unittest.main()
