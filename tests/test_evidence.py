"""Evidence checks use generated geometric JPEGs, never supplied media."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path
from time import monotonic

import pytest
from PIL import Image

from forte.evidence import EvidenceConfig, EvidenceError, assemble_result
from forte.shared.schemas import FactDecision, FrameRef, RuleRecord, VerifiedEvent, VideoAsset


@pytest.fixture
def sample(tmp_path: Path):
    asset = VideoAsset(
        job_id="synthetic-job", owned_path=tmp_path / "synthetic.mov",
        content_hash="a" * 64, duration_sec=5.0, width=48, height=32,
        time_base="1/1000", source_start_sec=2.0,
    )
    refs = {}
    for index, pts in enumerate((2000, 3000, 4000)):
        path = tmp_path / f"synthetic-{index}.jpg"
        image = Image.new("RGB", (48, 32), (index * 80, 50, 180))
        image.save(path, "JPEG")
        identity = hashlib.sha256(f"{asset.content_hash}:{pts}:{asset.time_base}".encode()).hexdigest()
        refs[identity] = FrameRef(
            frame_id=identity, pts_sec=float(index), path=path, width=48,
            height=32, job_id=asset.job_id, source_pts=pts,
            jpeg_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    ids = list(refs)
    rule = RuleRecord(
        rule_id="synthetic-rule", title="Synthetic public normative source", clause="7",
        source_url="https://example.org/synthetic-source", applicability="Synthetic fixture only",
        required_facts=["visible_object"], verification_state="verified",
        applicability_verified=True, enabled=True,
    )
    event = VerifiedEvent(
        event_id="synthetic-event", event_type_id="SYNTHETIC_EVENT", rule_id=rule.rule_id,
        start_sec=0.5, end_sec=1.5, anchor_frame_id=ids[1], evidence_frame_ids=[ids[0], ids[1], ids[2]],
        observed_facts=[FactDecision(
            fact_id="visible_object", state="present", supporting_frame_ids=[ids[1]],
            notes="UNTRUSTED_MODEL_NOTES",
        )],
        limitations=["Synthetic visibility limitation"], description="UNTRUSTED_MODEL_DESCRIPTION",
    )
    calls = []

    def extractor(asset_arg, frame_refs, output_dir):
        calls.append((asset_arg.job_id, tuple(frame.frame_id for frame in frame_refs), output_dir))
        return list(frame_refs)

    def renderer(rule_arg, observations, limitations):
        return f"Visible: {'; '.join(observations)}. Possible rule: {rule_arg.title}, {rule_arg.clause}; {rule_arg.source_url}. {'; '.join(limitations)}"

    args = dict(
        job_id=asset.job_id, asset=asset, events=[event], frames=refs,
        rules={rule.rule_id: rule}, evidence_dir=tmp_path / "evidence",
        analysis_complete=True, extractor=extractor,
        event_labels={"SYNTHETIC_EVENT": "Synthetic observable event"},
        fact_descriptions={"visible_object": "Synthetic colored object is visible"},
        description_renderer=renderer,
    )
    return args, calls


def test_original_jpegs_and_exact_actual_pts(sample):
    args, calls = sample
    result = assemble_result(**args)
    response = result.response
    assert len(calls) == 1
    assert len(calls[0][1]) == 3
    assert response.status == "done" and response.verdict == "violation"
    assert response.violations[0].timecode_sec == 1.0
    assert response.violations[0].type == "Synthetic observable event"
    assert "Synthetic public normative source" in response.violations[0].description
    assert "UNTRUSTED_MODEL" not in response.violations[0].description
    for public, original in zip(response.violations[0].evidence_frames, args["frames"].values(), strict=True):
        assert public.timestamp_sec == original.pts_sec
        assert base64.b64decode(public.image_base64, validate=True) == Path(original.path).read_bytes()
    assert result.stats.payload_bytes == len(result.json_bytes)
    assert result.stats.jpeg_bytes == sum(Path(frame.path).stat().st_size for frame in args["frames"].values())
    assert result.stats.total_sec >= result.stats.encoding_sec >= 0


def test_encoding_and_retrieval_deduplicate_source_frames(sample):
    args, calls = sample
    args["events"].append(args["events"][0].model_copy(update={"event_id": "synthetic-second-event"}))
    result = assemble_result(**args)
    assert len(calls) == 1 and len(calls[0][1]) == 3
    assert result.stats.unique_frame_count == 3 and result.stats.event_count == 2
    assert len(result.response.violations) == 2


def test_completed_bytes_are_immutable_and_response_is_defensive_copy(sample):
    args, _ = sample
    result = assemble_result(**args)
    before = result.json_bytes
    mutable_copy = result.response
    mutable_copy.violations[0].description = "changed"
    mutable_copy.violations.clear()
    assert result.json_bytes == before
    assert result.response.violations[0].description != "changed"
    with pytest.raises(FrozenInstanceError):
        result.json_bytes = b"changed"


def test_completed_negative_requires_explicit_complete_analysis(sample):
    args, calls = sample
    args.update(events=[], analysis_complete=False)
    with pytest.raises(EvidenceError, match="Incomplete"):
        assemble_result(**args)
    args["analysis_complete"] = True
    result = assemble_result(**args)
    assert json.loads(result.json_bytes)["verdict"] == "no_violation"
    assert result.response.violations == [] and calls == []


@pytest.mark.parametrize("field,value", [("job_id", "other-job"), ("job_id", None), ("source_pts", None), ("pts_sec", 1.1)])
def test_rejects_missing_ownership_or_pts_mismatch(sample, field, value):
    args, calls = sample
    anchor = args["events"][0].anchor_frame_id
    args["frames"][anchor] = args["frames"][anchor].model_copy(update={field: value})
    with pytest.raises(EvidenceError):
        assemble_result(**args)
    assert calls == []


def test_rejects_cross_video_frame_identity(sample):
    args, _ = sample
    args["asset"] = args["asset"].model_copy(update={"content_hash": "b" * 64})
    with pytest.raises(EvidenceError, match="identity"):
        assemble_result(**args)


def test_rejects_extractor_timestamp_substitution(sample):
    args, _ = sample
    def substituted(asset, refs, directory):
        return [ref.model_copy(update={"pts_sec": ref.pts_sec + 0.04}) for ref in refs]
    args["extractor"] = substituted
    with pytest.raises(EvidenceError, match="timestamp"):
        assemble_result(**args)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "unknown"])
def test_exact_retrieval_requires_precise_frame_set(sample, mode):
    args, _ = sample
    def substituted(asset, refs, directory):
        if mode == "missing":
            return refs[:-1]
        if mode == "duplicate":
            return [*refs, refs[0]]
        return [ref.model_copy(update={"frame_id": "unsupported"}) for ref in refs]
    args["extractor"] = substituted
    with pytest.raises(EvidenceError):
        assemble_result(**args)


@pytest.mark.parametrize("mode", ["corrupt", "dimensions", "hash"])
def test_rejects_invalid_or_modified_jpeg(sample, mode):
    args, _ = sample
    ref = next(iter(args["frames"].values()))
    if mode == "corrupt":
        Path(ref.path).write_bytes(b"\xff\xd8bad\xff\xd9")
    elif mode == "dimensions":
        Image.new("RGB", (49, 32)).save(ref.path, "JPEG")
    else:
        Image.new("RGB", (48, 32), (0, 0, 0)).save(ref.path, "JPEG")
    with pytest.raises(EvidenceError):
        assemble_result(**args)


@pytest.mark.parametrize("config", [
    EvidenceConfig(max_frame_bytes=10),
    EvidenceConfig(max_total_jpeg_bytes=10),
    EvidenceConfig(max_payload_bytes=100),
    EvidenceConfig(max_jpeg_pixels=100),
    EvidenceConfig(max_description_bytes=10),
    EvidenceConfig(max_label_bytes=10),
])
def test_resource_limits_fail_without_clean_verdict(sample, config):
    args, _ = sample
    with pytest.raises(EvidenceError):
        assemble_result(**args, config=config)


def test_repeated_wire_payload_is_bounded_even_when_jpegs_deduplicate(sample):
    args, _ = sample
    one = assemble_result(**args)
    args["events"].append(args["events"][0].model_copy(update={"event_id": "synthetic-second-event"}))
    with pytest.raises(EvidenceError, match="payload"):
        assemble_result(**args, config=EvidenceConfig(max_payload_bytes=len(one.json_bytes) + 100))


def test_selected_frames_must_cover_every_required_present_fact(sample):
    args, _ = sample
    event = args["events"][0]
    context_id = next(iter(args["frames"]))
    second_fact = FactDecision(fact_id="other_fact", state="present", supporting_frame_ids=[context_id])
    args["events"] = [event.model_copy(update={
        "evidence_frame_ids": [event.anchor_frame_id],
        "observed_facts": [*event.observed_facts, second_fact],
    })]
    rule_id = event.rule_id
    args["rules"][rule_id] = args["rules"][rule_id].model_copy(update={"required_facts": ["visible_object", "other_fact"]})
    with pytest.raises(EvidenceError, match="cover"):
        assemble_result(**args)


def test_unverified_rules_and_unreviewed_observations_are_rejected(sample):
    args, _ = sample
    rule_id = args["events"][0].rule_id
    valid = args["rules"][rule_id]
    args["rules"][rule_id] = valid.model_copy(update={"enabled": False})
    with pytest.raises(EvidenceError):
        assemble_result(**args)
    args["rules"][rule_id] = valid
    args["fact_descriptions"] = {}
    with pytest.raises(EvidenceError, match="reviewed"):
        assemble_result(**args)


def test_description_must_include_verified_source_and_clause(sample):
    args, _ = sample
    args["description_renderer"] = lambda *unused: "An unsupported statement"
    with pytest.raises(EvidenceError, match="source"):
        assemble_result(**args)


@pytest.mark.parametrize("control", [{"deadline": monotonic() - 1}, {"cancelled": lambda: True}])
def test_cancel_and_deadline_propagate(sample, control):
    args, calls = sample
    with pytest.raises(EvidenceError):
        assemble_result(**args, **control)
    assert calls == []


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        EvidenceConfig(max_events=0)
    with pytest.raises(ValueError):
        replace(EvidenceConfig(), timestamp_tolerance_sec=float("nan"))


@pytest.mark.integration
@pytest.mark.skipif(not shutil.which("ffprobe"), reason="FFprobe is required for source correspondence")
def test_default_exact_decoder_and_reviewed_catalog_with_synthetic_media(tmp_path):
    """Synthetic decisions test integration, never actual visual correctness."""
    import av

    from forte.video import decode_frames, probe_video
    from rules.catalog import load_catalog

    path = tmp_path / "generated-colors.mov"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=25)
        stream.width, stream.height = 48, 32
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.max_b_frames = 0
        for index, pts in enumerate((0, 40, 160)):
            with Image.new("RGB", (48, 32), (index * 80, 40, 130)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = pts, Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    asset = probe_video(path, job_id="synthetic-catalog-integration")
    refs = decode_frames(asset, tmp_path / "screen", timestamps_sec=[0.04], use_subprocess=False)
    anchor = refs[0]
    catalog = load_catalog()
    assert catalog.enabled_rules()
    for index, rule in enumerate(catalog.enabled_rules()):
        binding = catalog.bindings[rule.rule_id]
        event = VerifiedEvent(
            event_id=f"synthetic-catalog-event-{index}",
            event_type_id=binding["event_type_id"], rule_id=rule.rule_id,
            start_sec=0.0, end_sec=asset.duration_sec,
            anchor_frame_id=anchor.frame_id, evidence_frame_ids=[anchor.frame_id],
            observed_facts=[FactDecision(
                fact_id=fact_id, state=state, supporting_frame_ids=[anchor.frame_id],
            ) for fact_id, state in binding["required_fact_states"].items()],
            limitations=["unclear_visual_context"],
        )
        result = assemble_result(
            job_id=asset.job_id, asset=asset, events=[event], frames=refs,
            rules=catalog, evidence_dir=tmp_path / f"exact-{index}", analysis_complete=True,
        )
        violation = result.response.violations[0]
        assert violation.timecode_sec == anchor.pts_sec
        assert rule.title in violation.description and rule.source_url in violation.description
        assert "unclear_visual_context" not in violation.description
        assert result.events[0].limitations == ("unclear_visual_context",)
        assert Path(result.frames[0].original_jpeg_path).parent.name == f"exact-{index}"
        assert base64.b64decode(violation.evidence_frames[0].image_base64) == Path(anchor.path).read_bytes()
