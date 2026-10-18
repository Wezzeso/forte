"""Versioned, importable contracts for local visual analysis.

All ``*_sec`` values are seconds on the original video timeline, normalized
against ``VideoAsset.source_start_sec``. They are never frame-index / FPS time.
The models check structure; ``validate_timeline`` and ``validate_decisions``
bind it to owned source media. Neither validation proves a visual observation.
"""

from __future__ import annotations

import base64
import binascii
from datetime import date, datetime
from io import BytesIO
from os import fspath
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

CONTRACT_VERSION = "1.1.0"
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
Identifier = Annotated[str, Field(min_length=1, max_length=256, strict=True)]
NonemptyText = Annotated[str, Field(min_length=1, strict=True)]
Sha256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
LocalPath = Annotated[str, BeforeValidator(lambda value: fspath(value) if isinstance(value, Path) else value)]
PublicStatus = Literal["processing", "done", "failed"]
Verdict = Literal["violation", "no_violation"]
FactState = Literal["present", "absent", "unclear"]
SplitName = Literal["train", "validation", "final_check"]
JobStage = Literal[
    "queued", "claimed", "decoding", "screening", "refining", "assembling", "done", "failed"
]


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class AnalyzeResponse(ContractModel):
    job_id: Identifier


class EvidenceFrame(ContractModel):
    timestamp_sec: Seconds = Field(description="Seconds from the beginning of the original video")
    image_base64: NonemptyText = Field(
        description="Raw base64-encoded JPEG, without a data URI prefix",
        json_schema_extra={"contentEncoding": "base64", "contentMediaType": "image/jpeg"},
    )

    @field_validator("image_base64")
    @classmethod
    def actual_raw_jpeg(cls, value: str) -> str:
        """Reject placeholders, data URIs, corrupt files and other image formats."""
        from PIL import Image, UnidentifiedImageError

        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_base64 must be raw valid base64, without a data URI") from exc
        if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
            raise ValueError("evidence must be a complete JPEG")
        try:
            with Image.open(BytesIO(raw)) as image:
                if image.format != "JPEG":
                    raise ValueError("evidence must decode as JPEG")
                image.verify()
            with Image.open(BytesIO(raw)) as image:
                image.load()
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            raise ValueError("evidence JPEG must actually decode") from exc
        return value


class PublicViolation(ContractModel):
    type: NonemptyText
    timecode_sec: Seconds = Field(description="Event anchor in seconds from the original video start")
    description: NonemptyText
    evidence_frames: Annotated[list[EvidenceFrame], Field(min_length=1, max_length=3)]


class ProcessingResponse(ContractModel):
    model_config = ConfigDict(json_schema_extra={"required": ["job_id", "status"]})
    job_id: Identifier
    status: Literal["processing"] = "processing"


class FailedResponse(ContractModel):
    model_config = ConfigDict(json_schema_extra={"required": ["job_id", "status"]})
    job_id: Identifier
    status: Literal["failed"] = "failed"


class CompletedResponse(ContractModel):
    model_config = ConfigDict(
        json_schema_extra={
            "required": ["job_id", "status", "verdict", "violations"],
            "allOf": [
                {
                    "if": {"properties": {"verdict": {"const": "violation"}}},
                    "then": {"properties": {"violations": {"minItems": 1}}},
                },
                {
                    "if": {"properties": {"verdict": {"const": "no_violation"}}},
                    "then": {"properties": {"violations": {"maxItems": 0}}},
                },
            ],
        }
    )
    job_id: Identifier
    status: Literal["done"] = "done"
    verdict: Verdict
    violations: list[PublicViolation]

    @model_validator(mode="after")
    def verdict_matches_events(self) -> CompletedResponse:
        if (self.verdict == "violation") != bool(self.violations):
            raise ValueError("violation requires events; no_violation requires an empty event list")
        return self

    def validate_timeline(self, asset: VideoAsset) -> CompletedResponse:
        if self.job_id != asset.job_id:
            raise ValueError("result belongs to a different job")
        for event in self.violations:
            if event.timecode_sec > asset.duration_sec:
                raise ValueError("event lies outside the original video timeline")
            if any(frame.timestamp_sec > asset.duration_sec for frame in event.evidence_frames):
                raise ValueError("evidence lies outside the original video timeline")
        return self


JobResponse = Annotated[
    ProcessingResponse | CompletedResponse | FailedResponse, Field(discriminator="status")
]
JOB_RESPONSE_ADAPTER: TypeAdapter[JobResponse] = TypeAdapter(JobResponse)


class VideoAsset(ContractModel):
    job_id: Identifier
    owned_path: LocalPath
    content_hash: Sha256
    duration_sec: PositiveSeconds
    width: Annotated[int, Field(gt=0, strict=True)]
    height: Annotated[int, Field(gt=0, strict=True)]
    fps: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    time_base: str | None = None
    source_start_sec: Annotated[float, Field(allow_inf_nan=False)] = 0.0
    rotation_deg: Literal[0, 90, 180, 270] = 0
    codec: str | None = None
    frame_count: Annotated[int, Field(gt=0, strict=True)] | None = None
    stream_index: Annotated[int, Field(ge=0)] = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class FrameRef(ContractModel):
    frame_id: Identifier
    pts_sec: Seconds
    path: LocalPath
    width: Annotated[int, Field(gt=0, strict=True)]
    height: Annotated[int, Field(gt=0, strict=True)]
    job_id: Identifier | None = None
    rotation_deg: Literal[0, 90, 180, 270] = 0
    source_pts: int | None = None
    jpeg_sha256: Sha256 | None = None

    def validate_timeline(self, asset: VideoAsset) -> FrameRef:
        if self.job_id is not None and self.job_id != asset.job_id:
            raise ValueError("frame belongs to a different job")
        if self.pts_sec > asset.duration_sec:
            raise ValueError("frame lies outside the original video timeline")
        return self


class FactDefinition(ContractModel):
    fact_id: Identifier
    description: NonemptyText
    required: bool = True


class FactDecision(ContractModel):
    fact_id: Identifier
    state: FactState
    supporting_frame_ids: list[Identifier] = Field(default_factory=list)
    model_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    score_provenance: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def supported_decision(self) -> FactDecision:
        _unique(self.supporting_frame_ids, "supporting_frame_ids")
        if self.state == "present" and not self.supporting_frame_ids:
            raise ValueError("present facts require supporting frame IDs")
        if self.model_score is not None and not self.score_provenance:
            raise ValueError("model scores require score_provenance; they are not calibrated probabilities")
        return self


class RuleRecord(ContractModel):
    rule_id: Identifier
    title: NonemptyText
    clause: str | None = None
    source_url: NonemptyText
    source_publisher: str | None = None
    accessed_at: date | None = None
    source_sha256: Sha256 | None = None
    jurisdiction: str = "KZ"
    applicability: NonemptyText
    required_facts: list[Identifier]
    temporal_conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    verification_state: Literal["unverified", "source_verified", "verified", "rejected"] = "unverified"
    applicability_verified: bool = False
    enabled: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def enable_only_verified_applicable_rule(self) -> RuleRecord:
        _unique(self.required_facts, "required_facts")
        if self.enabled and (
            self.verification_state not in {"source_verified", "verified"}
            or not self.applicability_verified
            or not self.required_facts
        ):
            raise ValueError("enabled rules require a verified source, verified applicability, and required facts")
        return self


class ClipRequest(ContractModel):
    job_id: Identifier
    clip_id: Identifier
    frame_refs: Annotated[list[FrameRef], Field(min_length=1)]
    fact_definitions: list[FactDefinition] = Field(default_factory=list)
    rule_definitions: list[RuleRecord] = Field(default_factory=list)
    analysis_version: str = CONTRACT_VERSION
    start_sec: Seconds | None = None
    end_sec: Seconds | None = None

    @model_validator(mode="after")
    def consistent_clip(self) -> ClipRequest:
        _unique([frame.frame_id for frame in self.frame_refs], "frame IDs")
        _unique([fact.fact_id for fact in self.fact_definitions], "fact IDs")
        _unique([rule.rule_id for rule in self.rule_definitions], "rule IDs")
        if any(frame.job_id not in (None, self.job_id) for frame in self.frame_refs):
            raise ValueError("clip frames belong to a different job")
        if (self.start_sec is None) != (self.end_sec is None):
            raise ValueError("clip interval requires both start_sec and end_sec")
        if self.start_sec is not None and self.end_sec is not None:
            if self.end_sec < self.start_sec:
                raise ValueError("clip end_sec precedes start_sec")
            if any(not self.start_sec <= frame.pts_sec <= self.end_sec for frame in self.frame_refs):
                raise ValueError("clip frame lies outside its source interval")
        return self

    def validate_decisions(self, decisions: list[FactDecision]) -> list[FactDecision]:
        requested = {fact.fact_id for fact in self.fact_definitions}
        required = {fact.fact_id for fact in self.fact_definitions if fact.required}
        received = {decision.fact_id for decision in decisions}
        _unique([decision.fact_id for decision in decisions], "decision fact IDs")
        if received - requested:
            raise ValueError("decision references unrequested fact IDs")
        if required - received:
            raise ValueError("model output is missing required facts")
        allowed_frames = {frame.frame_id for frame in self.frame_refs}
        if any(set(decision.supporting_frame_ids) - allowed_frames for decision in decisions):
            raise ValueError("decision references unknown supporting frame IDs")
        return decisions


class CandidateEvent(ContractModel):
    candidate_id: Identifier
    event_type_id: Identifier
    start_sec: Seconds
    end_sec: Seconds
    supporting_facts: list[FactDecision]
    supporting_frame_ids: list[Identifier] = Field(default_factory=list)
    visibility_notes: list[str] = Field(default_factory=list)
    clip_id: Identifier | None = None

    @model_validator(mode="after")
    def ordered_interval(self) -> CandidateEvent:
        if self.end_sec < self.start_sec:
            raise ValueError("candidate end_sec precedes start_sec")
        _unique(self.supporting_frame_ids, "supporting_frame_ids")
        return self


class VerifiedEvent(ContractModel):
    event_id: Identifier
    event_type_id: Identifier
    rule_id: Identifier
    start_sec: Seconds
    end_sec: Seconds
    anchor_frame_id: Identifier
    evidence_frame_ids: Annotated[list[Identifier], Field(min_length=1, max_length=3)]
    observed_facts: list[FactDecision]
    limitations: list[str] = Field(default_factory=list)
    description: str | None = None

    @model_validator(mode="after")
    def supported_event(self) -> VerifiedEvent:
        if self.end_sec < self.start_sec:
            raise ValueError("verified end_sec precedes start_sec")
        _unique(self.evidence_frame_ids, "evidence_frame_ids")
        if self.anchor_frame_id not in self.evidence_frame_ids:
            raise ValueError("anchor frame must be in evidence_frame_ids")
        if not self.observed_facts or not any(fact.state == "present" for fact in self.observed_facts):
            raise ValueError("verified events require observed present facts")
        return self

    def validate_context(
        self, asset: VideoAsset, frames: dict[str, FrameRef], rule: RuleRecord
    ) -> VerifiedEvent:
        if self.end_sec > asset.duration_sec:
            raise ValueError("event interval lies outside original video")
        if rule.rule_id != self.rule_id or not rule.enabled:
            raise ValueError("event requires its enabled verified applicable rule")
        present = {fact.fact_id for fact in self.observed_facts if fact.state == "present"}
        if not set(rule.required_facts) <= present:
            raise ValueError("event does not support all rule-required facts")
        referenced = set(self.evidence_frame_ids)
        for fact in self.observed_facts:
            referenced.update(fact.supporting_frame_ids)
        if referenced - frames.keys():
            raise ValueError("event references unknown source frame IDs")
        for frame_id in referenced:
            if frames[frame_id].frame_id != frame_id:
                raise ValueError("frame lookup key does not match the referenced source frame ID")
            frames[frame_id].validate_timeline(asset)
        anchor = frames[self.anchor_frame_id]
        if not self.start_sec <= anchor.pts_sec <= self.end_sec:
            raise ValueError("anchor timestamp lies outside event interval")
        return self


class JobRecord(ContractModel):
    job_id: Identifier
    status: PublicStatus = "processing"
    stage: JobStage = "queued"
    accepted_at: datetime
    deadline_at: datetime
    owned_path: LocalPath
    content_hash: Sha256 | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    attempt: Annotated[int, Field(ge=0)] = 0
    fencing_version: Annotated[int, Field(ge=0)] = 0
    result_path: LocalPath | None = None
    error_code: str | None = None
    error_detail: str | None = None
    updated_at: datetime | None = None

    @field_validator("accepted_at", "deadline_at", "lease_expires_at", "updated_at")
    @classmethod
    def aware_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("job wall-clock times require an explicit timezone")
        return value

    @model_validator(mode="after")
    def consistent_job(self) -> JobRecord:
        if self.deadline_at <= self.accepted_at:
            raise ValueError("deadline must follow durable acceptance")
        if self.status == "done" and (self.stage != "done" or not self.result_path):
            raise ValueError("done jobs require done stage and an atomically published result_path")
        if self.status == "failed" and self.stage != "failed":
            raise ValueError("failed job requires failed stage")
        if self.status == "processing" and self.stage in {"done", "failed"}:
            raise ValueError("processing jobs cannot have a terminal stage")
        if (self.lease_owner is None) != (self.lease_expires_at is None):
            raise ValueError("lease requires owner and expiry together")
        return self


class ModelManifest(ContractModel):
    model_id: NonemptyText
    revision: NonemptyText
    weight_checksum: Sha256
    processor_version: NonemptyText
    license: NonemptyText
    adapter_checksum: Sha256 | None = None
    serving_config: dict[str, Any] = Field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


class EvaluationRecord(ContractModel):
    sample_id: Identifier
    group_id: Identifier
    split: SplitName
    predictions: dict[str, Any]
    matched_errors: list[dict[str, Any]] = Field(default_factory=list)
    timings: dict[str, Seconds] = Field(default_factory=dict)
    model_version: NonemptyText
    config_version: NonemptyText
    rule_version: NonemptyText

    @field_validator("split", mode="before")
    @classmethod
    def canonical_split(cls, value: object) -> object:
        """Accept the legacy draft name while emitting one canonical split name."""
        return "final_check" if value == "test" else value


class RecordingRecord(ContractModel):
    recording_id: Identifier
    content_hash: Sha256
    recording_group_id: Identifier
    relative_path: NonemptyText
    scenario_group_id: Identifier | None = None
    split: Literal["train", "validation", "final_check", "unassigned"] = "unassigned"
    duration_sec: PositiveSeconds | None = None
    width: Annotated[int, Field(gt=0)] | None = None
    height: Annotated[int, Field(gt=0)] | None = None
    workbook_row_ids: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("split", mode="before")
    @classmethod
    def canonical_split(cls, value: object) -> object:
        return "final_check" if value == "test" else value

    @property
    def content_id(self) -> str:
        return self.content_hash


__all__ = [
    "CONTRACT_VERSION", "Seconds", "PublicStatus", "Verdict", "FactState", "SplitName", "JobStage",
    "AnalyzeResponse", "EvidenceFrame", "PublicViolation", "ProcessingResponse", "FailedResponse",
    "CompletedResponse", "JobResponse", "JOB_RESPONSE_ADAPTER", "VideoAsset", "FrameRef",
    "FactDefinition", "FactDecision", "RuleRecord", "ClipRequest", "CandidateEvent", "VerifiedEvent",
    "JobRecord", "ModelManifest", "EvaluationRecord", "RecordingRecord",
]
