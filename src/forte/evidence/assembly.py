"""Worker-only original-evidence assembly and immutable jury projection.

Publication is deliberately separate: a job service atomically stores
``AssemblyResult.json_bytes`` before changing its job status to done. Serving a
completed job therefore never decodes a video or re-encodes a JPEG.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

from forte.shared.schemas import (
    CompletedResponse,
    EvidenceFrame,
    FrameRef,
    PublicViolation,
    RuleRecord,
    VerifiedEvent,
    VideoAsset,
)

from ._jpeg import EvidenceError, read_validated_jpeg
from .types import EventEvidenceMetadata, EvidenceConfig, EvidenceFrameMetadata, EvidenceStats

Extractor = Callable[[VideoAsset, Sequence[FrameRef], Path], Sequence[FrameRef]]
DescriptionRenderer = Callable[[RuleRecord, list[str], list[str]], str]


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Validated immutable bytes, plus private local-review measurements.

    ``response`` returns a fresh validated copy. Mutating that convenience
    object can never change the bytes handed to the publication service.
    """

    json_bytes: bytes
    stats: EvidenceStats
    frames: tuple[EvidenceFrameMetadata, ...]
    events: tuple[EventEvidenceMetadata, ...] = ()

    @property
    def response(self) -> CompletedResponse:
        return CompletedResponse.model_validate_json(self.json_bytes)


def _check_running(deadline: float | None, cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise EvidenceError("Evidence assembly was cancelled")
    if deadline is not None and monotonic() >= deadline:
        raise EvidenceError("Evidence assembly deadline exceeded")


def _owned_frame(frame: FrameRef, asset: VideoAsset, config: EvidenceConfig) -> None:
    frame.validate_timeline(asset)
    # A01 permits absent ownership on general-purpose FrameRefs; publication
    # requires ownership to be explicit, including in every fact support ref.
    if frame.job_id != asset.job_id:
        raise EvidenceError("Evidence frame ownership is missing or mismatched")
    if frame.source_pts is None or asset.time_base is None:
        raise EvidenceError("Evidence requires source PTS and a source time base")
    identity = f"{asset.content_hash}:{frame.source_pts}:{asset.time_base}"
    if frame.frame_id != hashlib.sha256(identity.encode("ascii")).hexdigest():
        raise EvidenceError("Evidence frame identity does not match the owned source video")
    try:
        time_base = Fraction(asset.time_base)
    except (ValueError, ZeroDivisionError) as exc:
        raise EvidenceError("Source time base is invalid") from exc
    if time_base <= 0:
        raise EvidenceError("Source time base is invalid")
    actual_sec = float(frame.source_pts * time_base) - asset.source_start_sec
    if not math.isclose(actual_sec, frame.pts_sec, rel_tol=0, abs_tol=config.timestamp_tolerance_sec):
        raise EvidenceError("Frame source PTS and normalized timestamp disagree")


def _rule_lookup(rules: Any, rule_id: str) -> RuleRecord:
    try:
        value = rules[rule_id] if isinstance(rules, Mapping) else rules.get_rule(rule_id, require_enabled=True)
        return RuleRecord.model_validate(value.model_dump() if isinstance(value, RuleRecord) else value)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise EvidenceError("Event rule is missing or not verified and applicable") from exc


def _reviewed_text(
    key: str, values: Mapping[str, str] | None, catalog: Any, method: str
) -> str:
    try:
        value = values[key] if values is not None else getattr(catalog, method)(key)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise EvidenceError("A reviewed taxonomy label or observation template is missing") from exc
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError("Reviewed taxonomy text must be nonempty")
    return value.strip()


def _fact_phrase(
    fact_id: str, state: str, values: Mapping[str, str] | None, catalog: Any
) -> str:
    if values is not None:
        key = fact_id if state == "present" else f"{fact_id}:{state}"
        return _reviewed_text(key, values, catalog, "fact_description")
    try:
        value = catalog.fact_description(fact_id, state)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise EvidenceError("A reviewed state-specific observation template is missing") from exc
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError("Reviewed observation text must be nonempty")
    return value.strip()


def _required_states(event: VerifiedEvent, rule: RuleRecord, catalog: Any) -> Mapping[str, str]:
    bindings = getattr(catalog, "bindings", None)
    if bindings is None:
        return {fact_id: "present" for fact_id in rule.required_facts}
    try:
        binding = bindings[rule.rule_id]
        if binding["event_type_id"] != event.event_type_id:
            raise EvidenceError("Event type differs from its reviewed rule binding")
        return binding["required_fact_states"]
    except (KeyError, TypeError) as exc:
        raise EvidenceError("Reviewed rule binding is missing") from exc


def assemble_result(
    *,
    job_id: str,
    asset: VideoAsset,
    events: Sequence[VerifiedEvent],
    frames: Mapping[str, FrameRef] | Sequence[FrameRef],
    rules: Any,
    evidence_dir: Path | str,
    analysis_complete: bool,
    config: EvidenceConfig | None = None,
    extractor: Extractor | None = None,
    catalog: Any = None,
    event_labels: Mapping[str, str] | None = None,
    fact_descriptions: Mapping[str, str] | None = None,
    description_renderer: DescriptionRenderer | None = None,
    deadline: float | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AssemblyResult:
    """Assemble one publishable result from verified events and owned frames.

    ``analysis_complete`` is an explicit worker assertion that technical
    coverage finished and the event engine permits publication. An empty event
    list alone is never evidence of a clean recording. ``deadline`` uses the
    ``time.monotonic`` clock. Any failure raises instead of manufacturing
    ``no_violation``.

    ``extractor`` must perform exact source retrieval. The default A05 decoder
    verifies the owned video's hash and PTS, then retrieves all unique requested
    source frames in one pass. Injection is intended for worker configuration
    and synthetic tests, never request-controlled code.
    """
    started = perf_counter()
    config = config or EvidenceConfig()
    _check_running(deadline, cancelled)
    if analysis_complete is not True:
        raise EvidenceError("Incomplete or unresolved analysis cannot publish a completed result")
    asset = VideoAsset.model_validate(asset.model_dump())
    if job_id != asset.job_id:
        raise EvidenceError("Completed result and video belong to different jobs")
    if len(events) > config.max_events:
        raise EvidenceError("Result exceeds the configured event limit")
    validated_events = tuple(VerifiedEvent.model_validate(event.model_dump()) for event in events)
    if len({event.event_id for event in validated_events}) != len(validated_events):
        raise EvidenceError("Verified event IDs must be unique")

    frame_values = frames.values() if isinstance(frames, Mapping) else frames
    frame_map: dict[str, FrameRef] = {}
    for frame in frame_values:
        validated = FrameRef.model_validate(frame.model_dump())
        if validated.frame_id in frame_map:
            raise EvidenceError("Frame IDs must be unique")
        frame_map[validated.frame_id] = validated
    if isinstance(frames, Mapping) and any(key != frame.frame_id for key, frame in frames.items()):
        raise EvidenceError("Frame lookup keys do not match source frame IDs")

    validated_rules: dict[str, RuleRecord] = {}
    taxonomy = catalog if catalog is not None else rules
    event_states: dict[str, Mapping[str, str]] = {}
    requested: dict[str, FrameRef] = {}
    for event in validated_events:
        rule = _rule_lookup(rules, event.rule_id)
        try:
            event.validate_context(asset, frame_map, rule)
        except ValueError as exc:
            raise EvidenceError("Verified event is not bound to valid source evidence and rule support") from exc
        validated_rules[event.rule_id] = rule
        states = _required_states(event, rule, taxonomy)
        fact_states = {fact.fact_id: fact.state for fact in event.observed_facts}
        if len(fact_states) != len(event.observed_facts):
            raise EvidenceError("Verified observations must have unique fact IDs")
        if any(state not in {"present", "absent"} or fact_states.get(fact_id) != state for fact_id, state in states.items()):
            raise EvidenceError("Event does not establish every reviewed rule fact state")
        event_states[event.event_id] = states
        referenced = set(event.evidence_frame_ids)
        for fact in event.observed_facts:
            referenced.update(fact.supporting_frame_ids)
            if fact.fact_id in rule.required_facts and fact.state == "present":
                if not set(fact.supporting_frame_ids).intersection(event.evidence_frame_ids):
                    raise EvidenceError("Selected evidence does not cover every required observation")
        for frame_id in referenced:
            _owned_frame(frame_map[frame_id], asset, config)
        for frame_id in event.evidence_frame_ids:
            requested[frame_id] = frame_map[frame_id]

    validation_sec = perf_counter() - started
    extraction_started = perf_counter()
    extracted: Sequence[FrameRef] = ()
    if requested:
        if extractor is None:
            from forte.video import extract_evidence_frames

            def extractor(source: VideoAsset, refs: Sequence[FrameRef], directory: Path) -> Sequence[FrameRef]:
                return extract_evidence_frames(
                    source, refs, directory,
                    deadline_monotonic=deadline,
                    cancel=cancelled,
                    max_pixels=config.max_jpeg_pixels,
                )
        _check_running(deadline, cancelled)
        extracted = extractor(asset, tuple(requested.values()), Path(evidence_dir))
    extraction_sec = perf_counter() - extraction_started

    extracted_map: dict[str, FrameRef] = {}
    for frame in extracted:
        frame = FrameRef.model_validate(frame.model_dump())
        if frame.frame_id not in requested or frame.frame_id in extracted_map:
            raise EvidenceError("Exact source retrieval returned unknown or repeated frame IDs")
        original = requested[frame.frame_id]
        _owned_frame(frame, asset, config)
        if (
            frame.source_pts != original.source_pts
            or not math.isclose(frame.pts_sec, original.pts_sec, rel_tol=0, abs_tol=config.timestamp_tolerance_sec)
            or frame.width != original.width
            or frame.height != original.height
        ):
            raise EvidenceError("Extracted JPEG does not match its declared source frame")
        extracted_map[frame.frame_id] = frame
    if extracted_map.keys() != requested.keys():
        raise EvidenceError("Exact source retrieval did not return every requested frame")

    encoding_started = perf_counter()
    encoded: dict[str, EvidenceFrame] = {}
    metadata: list[EvidenceFrameMetadata] = []
    total_jpeg_bytes = 0
    for frame_id, frame in extracted_map.items():
        _check_running(deadline, cancelled)
        raw = read_validated_jpeg(
            frame.path,
            max_bytes=config.max_frame_bytes,
            max_pixels=config.max_jpeg_pixels,
            expected_width=frame.width,
            expected_height=frame.height,
        )
        total_jpeg_bytes += len(raw)
        if total_jpeg_bytes > config.max_total_jpeg_bytes:
            raise EvidenceError("Result exceeds the configured total JPEG byte limit")
        digest = hashlib.sha256(raw).hexdigest()
        if frame.jpeg_sha256 is not None and digest != frame.jpeg_sha256:
            raise EvidenceError("Original evidence JPEG integrity check failed")
        encoded[frame_id] = EvidenceFrame(
            timestamp_sec=frame.pts_sec,
            image_base64=base64.b64encode(raw).decode("ascii"),
        )
        metadata.append(EvidenceFrameMetadata(
            frame_id=frame_id,
            timestamp_sec=frame.pts_sec,
            original_jpeg_path=Path(frame.path),
            width=frame.width,
            height=frame.height,
            sha256=digest,
            jpeg_bytes=len(raw),
        ))
    encoding_sec = perf_counter() - encoding_started

    description_started = perf_counter()
    if validated_events and description_renderer is None:
        if callable(getattr(taxonomy, "render_description", None)):
            description_renderer = taxonomy.render_description
        else:
            from rules.catalog import render_description

            description_renderer = render_description
    violations: list[PublicViolation] = []
    # Base64 strings are shared internally, but the required public JSON repeats
    # them for each event. Bound that wire size before allocating the full JSON.
    projected_payload_bytes = len(json.dumps(job_id, ensure_ascii=False).encode("utf-8")) + 100
    for event in validated_events:
        _check_running(deadline, cancelled)
        rule = validated_rules[event.rule_id]
        observations = [
            _fact_phrase(fact_id, state, fact_descriptions, taxonomy)
            for fact_id, state in event_states[event.event_id].items()
        ]
        assert description_renderer is not None
        # Free-form model descriptions and model notes never enter this path.
        approved = getattr(taxonomy, "payload", {}).get("approved_limitations_ru")
        public_limits = list(event.limitations) if approved is None else [text for text in event.limitations if text in approved]
        description = description_renderer(rule, observations, public_limits)
        if (
            not isinstance(description, str)
            or not description.strip()
            or rule.source_url not in description
            or (rule.clause is not None and rule.clause not in description)
        ):
            raise EvidenceError("Description lacks its verified source reference")
        label = _reviewed_text(event.event_type_id, event_labels, taxonomy, "event_type_label")
        if len(description.encode("utf-8")) > config.max_description_bytes:
            raise EvidenceError("Description exceeds the configured text byte limit")
        if len(label.encode("utf-8")) > config.max_label_bytes:
            raise EvidenceError("Event label exceeds the configured text byte limit")
        violation = PublicViolation(
            type=label,
            timecode_sec=frame_map[event.anchor_frame_id].pts_sec,
            description=description,
            evidence_frames=[encoded[frame_id] for frame_id in event.evidence_frame_ids],
        )
        projected_payload_bytes += len(violation.model_dump_json().encode("utf-8")) + 1
        if projected_payload_bytes > config.max_payload_bytes:
            raise EvidenceError("Result exceeds the configured public payload byte limit")
        violations.append(violation)
    description_sec = perf_counter() - description_started

    final_validation_started = perf_counter()
    response = CompletedResponse(
        job_id=job_id,
        verdict="violation" if violations else "no_violation",
        violations=violations,
    ).validate_timeline(asset)
    payload = response.model_dump_json().encode("utf-8")
    if len(payload) > config.max_payload_bytes:
        raise EvidenceError("Result exceeds the configured public payload byte limit")
    _check_running(deadline, cancelled)
    validation_sec += perf_counter() - final_validation_started
    return AssemblyResult(
        json_bytes=payload,
        frames=tuple(metadata),
        events=tuple(EventEvidenceMetadata(
            event_id=event.event_id,
            event_type_id=event.event_type_id,
            rule_id=event.rule_id,
            anchor_frame_id=event.anchor_frame_id,
            evidence_frame_ids=tuple(event.evidence_frame_ids),
            limitations=tuple(event.limitations),
        ) for event in validated_events),
        stats=EvidenceStats(
            event_count=len(violations),
            unique_frame_count=len(metadata),
            jpeg_bytes=total_jpeg_bytes,
            payload_bytes=len(payload),
            extraction_sec=extraction_sec,
            encoding_sec=encoding_sec,
            description_sec=description_sec,
            validation_sec=validation_sec,
            total_sec=perf_counter() - started,
        ),
    )


__all__ = ["AssemblyResult", "assemble_result"]
