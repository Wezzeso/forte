"""Contract boundaries, using only generated synthetic media."""

from __future__ import annotations

import base64
import copy
import json
import unittest
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from forte.shared.schemas import (
    JOB_RESPONSE_ADAPTER,
    CandidateEvent,
    ClipRequest,
    CompletedResponse,
    EvidenceFrame,
    EvaluationRecord,
    FactDecision,
    FactDefinition,
    FrameRef,
    JobRecord,
    RuleRecord,
    RecordingRecord,
    VerifiedEvent,
    VideoAsset,
)


FIXTURES = Path(__file__).resolve().parents[1] / "contracts" / "fixtures"


def asset() -> VideoAsset:
    return VideoAsset(
        job_id="synthetic-job-001", owned_path="owned/source.mov", content_hash="a" * 64,
        duration_sec=10.0, width=128, height=96, source_start_sec=-0.5,
    )


def frame() -> FrameRef:
    return FrameRef(
        job_id="synthetic-job-001", frame_id="f1", pts_sec=2.5,
        path=FIXTURES / "synthetic-evidence.jpg", width=128, height=96,
    )


def positive_fact() -> FactDecision:
    return FactDecision(fact_id="synthetic_fact", state="present", supporting_frame_ids=["f1"])


def rule(**updates: object) -> RuleRecord:
    data = dict(
        rule_id="synthetic-rule", title="Synthetic fixture document", clause="TEST-1",
        source_url="https://example.invalid/synthetic-rule", applicability="Synthetic fixtures only",
        required_facts=["synthetic_fact"], verification_state="source_verified",
        applicability_verified=True, enabled=True,
    )
    data.update(updates)
    return RuleRecord(**data)


def event() -> VerifiedEvent:
    return VerifiedEvent(
        event_id="event-1", event_type_id="synthetic-event", rule_id="synthetic-rule",
        start_sec=2.0, end_sec=3.0, anchor_frame_id="f1", evidence_frame_ids=["f1"],
        observed_facts=[positive_fact()],
    )


class PublicContractsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positive = json.loads((FIXTURES / "positive.json").read_text(encoding="utf-8"))

    def test_all_polling_fixtures_validate_and_roundtrip(self) -> None:
        for name in ("positive", "negative", "processing", "failed"):
            with self.subTest(name=name):
                raw = (FIXTURES / f"{name}.json").read_bytes()
                parsed = JOB_RESPONSE_ADAPTER.validate_json(raw)
                self.assertEqual(json.loads(raw), parsed.model_dump(mode="json"))

    def test_generated_schema_enforces_fixtures_status_and_verdict_consistency(self) -> None:
        schema_path = FIXTURES.parent / "schemas" / "JobResponse.schema.json"
        validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
        for name in ("positive", "negative", "processing", "failed"):
            with self.subTest(name=name):
                validator.validate(json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))
        missing_status = copy.deepcopy(self.positive)
        del missing_status["status"]
        contradictory_verdict = copy.deepcopy(self.positive)
        contradictory_verdict["verdict"] = "no_violation"
        for data in (missing_status, contradictory_verdict):
            self.assertFalse(validator.is_valid(data))

    def test_positive_fixture_jpeg_really_decodes(self) -> None:
        encoded = self.positive["violations"][0]["evidence_frames"][0]["image_base64"]
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            image.load()
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (128, 96))

    def test_illegal_status_and_verdict_fail(self) -> None:
        for key, value in (("status", "queued"), ("verdict", "unclear"), ("verdict", "normal")):
            data = copy.deepcopy(self.positive)
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                JOB_RESPONSE_ADAPTER.validate_python(data)

    def test_positive_requires_events(self) -> None:
        with self.assertRaises(ValidationError):
            CompletedResponse(job_id="job", verdict="violation", violations=[])

    def test_negative_forbids_events(self) -> None:
        self.positive["verdict"] = "no_violation"
        with self.assertRaises(ValidationError):
            CompletedResponse.model_validate(self.positive)

    def test_jpeg_rejects_placeholder_data_uri_corruption_and_png(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (2, 2)).save(buffer, "PNG")
        actual = self.positive["violations"][0]["evidence_frames"][0]["image_base64"]
        corrupt = base64.b64encode(b"\xff\xd8invalid\xff\xd9").decode("ascii")
        for encoded in ("PLACEHOLDER", "data:image/jpeg;base64," + actual, corrupt,
                        base64.b64encode(buffer.getvalue()).decode("ascii")):
            with self.subTest(encoded=encoded[:30]), self.assertRaises(ValidationError):
                EvidenceFrame(timestamp_sec=0.0, image_base64=encoded)

    def test_exactly_one_to_three_evidence_frames(self) -> None:
        evidence = self.positive["violations"][0]["evidence_frames"]
        for count in (0, 4):
            data = copy.deepcopy(self.positive)
            data["violations"][0]["evidence_frames"] = evidence * count
            with self.subTest(count=count), self.assertRaises(ValidationError):
                CompletedResponse.model_validate(data)

    def test_multiple_events_are_preserved(self) -> None:
        self.positive["violations"].append(copy.deepcopy(self.positive["violations"][0]))
        self.assertEqual(len(CompletedResponse.model_validate(self.positive).violations), 2)

    def test_negative_nonfinite_boolean_and_string_timestamps_fail(self) -> None:
        for value in (-0.1, float("nan"), float("inf"), True, "2.5"):
            data = copy.deepcopy(self.positive)
            data["violations"][0]["timecode_sec"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                CompletedResponse.model_validate(data)

    def test_timeline_boundaries_need_asset_context(self) -> None:
        parsed = CompletedResponse.model_validate(self.positive)
        self.assertIs(parsed.validate_timeline(asset()), parsed)
        self.positive["violations"][0]["timecode_sec"] = 10.01
        parsed = CompletedResponse.model_validate(self.positive)
        with self.assertRaises(ValueError):
            parsed.validate_timeline(asset())
        self.positive["violations"][0]["timecode_sec"] = 2.5
        self.positive["violations"][0]["evidence_frames"][0]["timestamp_sec"] = 10.01
        with self.assertRaises(ValueError):
            CompletedResponse.model_validate(self.positive).validate_timeline(asset())

    def test_unexpected_public_fields_rejected(self) -> None:
        for status in ("processing", "failed"):
            with self.subTest(status=status), self.assertRaises(ValidationError):
                JOB_RESPONSE_ADAPTER.validate_python({"job_id": "j", "status": status, "error": "private"})


class InternalContractsTests(unittest.TestCase):
    def test_final_check_is_canonical_split_with_legacy_input_normalized(self) -> None:
        recording = dict(recording_id="synthetic", content_hash="a" * 64,
                         recording_group_id="group", relative_path="synthetic.mov")
        evaluation = dict(sample_id="synthetic", group_id="group", predictions={},
                          model_version="synthetic", config_version="synthetic", rule_version="synthetic")
        for split in ("train", "validation", "final_check", "test"):
            expected = "final_check" if split == "test" else split
            with self.subTest(split=split):
                self.assertEqual(RecordingRecord(**recording, split=split).model_dump()["split"], expected)
                self.assertEqual(EvaluationRecord(**evaluation, split=split).model_dump()["split"], expected)
        for model in (EvaluationRecord, RecordingRecord):
            enum = model.model_json_schema()["properties"]["split"]["enum"]
            self.assertIn("final_check", enum)
            self.assertNotIn("test", enum)

    def test_present_fact_needs_evidence_and_score_needs_provenance(self) -> None:
        with self.assertRaises(ValidationError):
            FactDecision(fact_id="f", state="present")
        with self.assertRaises(ValidationError):
            FactDecision(fact_id="f", state="unclear", model_score=0.9)
        decision = FactDecision(fact_id="f", state="unclear", model_score=0.9, score_provenance="raw logits")
        self.assertEqual(decision.state, "unclear")

    def test_clip_rejects_unknown_frames_unknown_facts_missing_and_duplicate_facts(self) -> None:
        clip = ClipRequest(job_id=asset().job_id, clip_id="clip", frame_refs=[frame()],
                           fact_definitions=[FactDefinition(fact_id="synthetic_fact", description="Synthetic fact")])
        self.assertEqual(clip.validate_decisions([positive_fact()]), [positive_fact()])
        bad_cases = [[], [positive_fact(), positive_fact()],
                     [FactDecision(fact_id="unrequested", state="absent")],
                     [FactDecision(fact_id="synthetic_fact", state="present", supporting_frame_ids=["invented"])]]
        for decisions in bad_cases:
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                clip.validate_decisions(decisions)

    def test_clip_interval_and_job_binding(self) -> None:
        with self.assertRaises(ValidationError):
            ClipRequest(job_id="other-job", clip_id="clip", frame_refs=[frame()])
        with self.assertRaises(ValidationError):
            ClipRequest(job_id=asset().job_id, clip_id="clip", frame_refs=[frame()], start_sec=3.0, end_sec=4.0)
        with self.assertRaises(ValidationError):
            ClipRequest(job_id=asset().job_id, clip_id="clip", frame_refs=[frame(), frame()])

    def test_rule_requires_source_and_applicability_verification_before_enabling(self) -> None:
        for updates in ({"verification_state": "unverified"}, {"applicability_verified": False},
                        {"required_facts": []}):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                rule(**updates)
        self.assertFalse(rule(enabled=False, applicability_verified=False).enabled)

    def test_verified_event_cross_references_and_required_facts(self) -> None:
        self.assertIsInstance(event().validate_context(asset(), {"f1": frame()}, rule()), VerifiedEvent)
        with self.assertRaises(ValueError):
            event().validate_context(asset(), {}, rule())
        with self.assertRaises(ValueError):
            event().validate_context(asset(), {"f1": frame()}, rule(required_facts=["different_fact"]))
        with self.assertRaises(ValueError):
            event().validate_context(asset(), {"f1": frame()}, rule(enabled=False))
        mismatched = frame().model_dump() | {"frame_id": "different-frame"}
        with self.assertRaises(ValueError):
            event().validate_context(asset(), {"f1": FrameRef.model_validate(mismatched)}, rule())

    def test_reversed_intervals_and_unsupported_anchor_fail(self) -> None:
        with self.assertRaises(ValidationError):
            CandidateEvent(candidate_id="c", event_type_id="e", start_sec=3.0, end_sec=1.0, supporting_facts=[])
        data = event().model_dump()
        data["anchor_frame_id"] = "invented"
        with self.assertRaises(ValidationError):
            VerifiedEvent.model_validate(data)

    def test_aware_job_lease_and_publication_invariants(self) -> None:
        accepted = datetime(2026, 10, 17, tzinfo=timezone.utc)
        data = dict(job_id="job", accepted_at=accepted, deadline_at=accepted + timedelta(minutes=5), owned_path="j.mov")
        self.assertEqual(JobRecord(**data).status, "processing")
        for updates in ({"lease_owner": "worker"}, {"status": "done", "stage": "done"},
                        {"accepted_at": accepted.replace(tzinfo=None)}, {"deadline_at": accepted}):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                JobRecord(**(data | updates))

    def test_frame_path_and_timeline_are_bound_without_opening_private_media(self) -> None:
        self.assertIsInstance(frame().path, str)
        self.assertEqual(frame().validate_timeline(asset()).pts_sec, 2.5)
        data = frame().model_dump()
        data["pts_sec"] = 11.0
        with self.assertRaises(ValueError):
            FrameRef.model_validate(data).validate_timeline(asset())


if __name__ == "__main__":
    unittest.main()
