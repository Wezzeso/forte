"""Unconnected prototype contract for a reviewed discriminative fast path.

No model is loaded here and this module is not imported by the live provider.
The feature/head implementation must supply canonical source-frame metadata,
window and per-frame logits, localization supports, and optional *decoded*
temporal proposals. Scores are not calibrated probabilities merely because a
softmax was applied. There are deliberately no production threshold defaults.

Future DecisionProvider mapping:
1. Bind model/feature/taxonomy identities and every source frame via this module.
2. Verify the held-out human-reviewed calibration evidence independently.
3. Choose present/absent only for eligible calibrated classes and source support;
   otherwise emit unclear. Preserve all nineteen requested facts.
4. Missing/occluded views or incomplete procedure context remain unclear even
   with a high score. Feature coverage alone never establishes visual coverage.
5. Emit candidates only from an approved calibrated boundary decoder's explicit
   canonical frame brackets. Window presence cannot invent event timing; missing
   localization remains unresolved. Preserve separate repeated occurrences.
6. Return the existing facts/candidates JSON through validate_decisions; legal
   sources, applicability, temporal verification and evidence stay in A11/A12.

This contract does not authorize training, establish genuine annotation review,
make quality/latency claims, or replace the existing uncertainty checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forte.shared.schemas import ClipRequest

HEAD_CONTRACT_VERSION = "forte.temporal-fact-head.v1"
STATE_ORDER = ("absent", "present", "unclear")
FULL_FACT_COUNT = 19
Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Count = Annotated[int, Field(ge=0, strict=True)]
Identifier = Annotated[str, Field(min_length=1, max_length=256, strict=True)]
StateLogits = tuple[Finite, Finite, Finite]


class HeadContractError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(f"Invalid temporal head binding: {code}")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class HeadIdentity(_Contract):
    encoder_id: Identifier
    encoder_revision: Identifier
    encoder_artifacts_sha256: Digest
    head_sha256: Digest
    feature_spec_sha256: Digest
    taxonomy_sha256: Digest
    # Tensor axes are fixed by this explicit versioned order.
    fact_ids: Annotated[tuple[Identifier, ...], Field(min_length=19, max_length=19)]
    event_type_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def unique_axes(self):
        if len(set(self.fact_ids)) != len(self.fact_ids) or len(set(self.event_type_ids)) != len(self.event_type_ids):
            raise ValueError("head axes must be unique")
        return self


class SourceFrame(_Contract):
    frame_id: Identifier
    pts_sec: Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
    source_pts: Annotated[int, Field(strict=True)] | None = None
    next_source_pts: Annotated[int, Field(strict=True)] | None = None


class TemporalProposal(_Contract):
    """An explicitly paired bracket, not an event inferred from window state."""
    proposal_id: Identifier
    event_type_id: Identifier
    start_frame_id: Identifier
    end_frame_id: Identifier
    supporting_fact_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    supporting_frame_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    decoder_score: Finite


class TemporalHeadOutput(_Contract):
    """Shapes: window[F,3], frame[T,F,3], support[T,F], boundary[T,E,2].

    Frame validity reports usable features. A global-view mask distinguishes
    source context from crops; neither field proves procedure observability.
    ROI features retain the original frame ID/PTS and never become evidence IDs.
    """
    contract_version: Literal["forte.temporal-fact-head.v1"] = HEAD_CONTRACT_VERSION
    identity: HeadIdentity
    job_id: Identifier
    clip_id: Identifier
    frames: Annotated[tuple[SourceFrame, ...], Field(min_length=1)]
    window_logits: tuple[StateLogits, ...]
    frame_logits: tuple[tuple[StateLogits, ...], ...]
    support_logits: tuple[tuple[Finite, ...], ...]
    frame_valid: tuple[bool, ...]
    global_view_available: tuple[bool, ...]
    feature_coverage_complete: bool
    # Set only by a backend that checked A05's actual next-source-PTS chain.
    # Delivery of all requested sparse frames is not native temporal coverage.
    native_source_continuity_verified: bool = False
    # Bound by the checkpoint loader from actual positive supervision counts;
    # unconstrained random output branches are not observations.
    trained_components: dict[str, bool] = Field(default_factory=dict)
    boundary_logits: tuple[tuple[tuple[Finite, Finite], ...], ...] | None = None
    decoder_sha256: Digest | None = None
    proposals: tuple[TemporalProposal, ...] = ()

    @model_validator(mode="after")
    def consistent_shapes(self):
        frame_count, fact_count = len(self.frames), len(self.identity.fact_ids)
        if len({frame.frame_id for frame in self.frames}) != frame_count:
            raise ValueError("source frame IDs must be unique")
        if len(self.window_logits) != fact_count:
            raise ValueError("window logits must match the declared fact axis")
        if any(len(values) != frame_count for values in (
            self.frame_logits, self.support_logits, self.frame_valid, self.global_view_available,
        )):
            raise ValueError("head arrays must match the source frame axis")
        if any(len(values) != fact_count for values in self.frame_logits + self.support_logits):
            raise ValueError("head arrays must match the declared fact axis")
        if self.boundary_logits is not None and (
            len(self.boundary_logits) != frame_count
            or any(len(row) != len(self.identity.event_type_ids) for row in self.boundary_logits)
        ):
            raise ValueError("boundary logits must match frame and event axes")
        if self.proposals and self.decoder_sha256 is None:
            raise ValueError("decoded proposals require an identified decoder")
        return self


def validate_head_binding(
    output: TemporalHeadOutput, request: ClipRequest, expected: HeadIdentity,
    *, expected_decoder_sha256: str | None = None,
) -> None:
    """Reject model, tensor-axis, timestamp and invented-reference mismatches."""
    if output.identity != expected:
        raise HeadContractError("head_identity_mismatch")
    if output.job_id != request.job_id or output.clip_id != request.clip_id:
        raise HeadContractError("request_identity_mismatch")
    if tuple(fact.fact_id for fact in request.fact_definitions) != expected.fact_ids:
        raise HeadContractError("fact_axis_mismatch")
    if tuple((frame.frame_id, frame.pts_sec) for frame in output.frames) != tuple(
        (frame.frame_id, frame.pts_sec) for frame in request.frame_refs
    ):
        raise HeadContractError("source_frame_binding_mismatch")
    if any(frame.source_pts is not None and reference.source_pts is not None
           and frame.source_pts != reference.source_pts
           for frame, reference in zip(output.frames, request.frame_refs)):
        raise HeadContractError("integer_source_pts_binding_mismatch")
    if any(first.pts_sec > second.pts_sec for first, second in zip(output.frames, output.frames[1:])):
        raise HeadContractError("non_chronological_source_frames")
    if output.proposals and (expected_decoder_sha256 is None or output.decoder_sha256 != expected_decoder_sha256):
        raise HeadContractError("unverified_temporal_decoder")
    frames = {frame.frame_id: frame for frame in request.frame_refs}
    proposal_ids = set()
    for proposal in output.proposals:
        if proposal.proposal_id in proposal_ids:
            raise HeadContractError("duplicate_temporal_proposal")
        proposal_ids.add(proposal.proposal_id)
        if proposal.event_type_id not in expected.event_type_ids:
            raise HeadContractError("unknown_temporal_event_type")
        refs = (proposal.start_frame_id, proposal.end_frame_id, *proposal.supporting_frame_ids)
        if any(frame_id not in frames for frame_id in refs):
            raise HeadContractError("unknown_source_frame")
        if not set(proposal.supporting_fact_ids).issubset(expected.fact_ids):
            raise HeadContractError("unknown_fact_support")
        if (len(set(proposal.supporting_fact_ids)) != len(proposal.supporting_fact_ids)
                or len(set(proposal.supporting_frame_ids)) != len(proposal.supporting_frame_ids)):
            raise HeadContractError("duplicate_temporal_support")
        start, end = frames[proposal.start_frame_id].pts_sec, frames[proposal.end_frame_id].pts_sec
        if start > end or any(not start <= frames[frame_id].pts_sec <= end
                              for frame_id in proposal.supporting_frame_ids):
            raise HeadContractError("invalid_temporal_bracket")


class StateThreshold(_Contract):
    """All numerical choices must be selected from genuine reviewed validation."""
    score_space: Literal["raw_logit", "softmax_score"]
    minimum_score: Finite
    minimum_margin: Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
    minimum_support_score: Finite
    support_score_space: Literal["raw_logit", "sigmoid_score"]

    @model_validator(mode="after")
    def consistent_score_spaces(self):
        if self.score_space == "softmax_score" and (
            not 0 <= self.minimum_score <= 1 or self.minimum_margin > 1
        ):
            raise ValueError("softmax thresholds must remain within score bounds")
        if self.support_score_space == "sigmoid_score" and not 0 <= self.minimum_support_score <= 1:
            raise ValueError("sigmoid support thresholds must remain within score bounds")
        return self


class FactCalibration(_Contract):
    present: StateThreshold | None = None
    absent: StateThreshold | None = None
    reviewed_class_counts: dict[Literal["present", "absent", "unclear"], Count]
    reviewed_support_counts: dict[Literal["present", "absent"], Count]


class EventCalibration(_Contract):
    score_space: Literal["raw_logit", "sigmoid_score", "softmax_score"]
    minimum_decoder_score: Finite
    reviewed_event_count: Count
    reviewed_normal_count: Count

    @model_validator(mode="after")
    def consistent_score_space(self):
        if self.score_space != "raw_logit" and not 0 <= self.minimum_decoder_score <= 1:
            raise ValueError("normalized decoder thresholds must remain within score bounds")
        return self


class ReviewedCalibrationManifest(_Contract):
    identity: HeadIdentity
    validation_origin: Literal["human_reviewed_validation", "synthetic", "unreviewed"]
    split_role: Literal["validation"]
    split_manifest_sha256: Digest
    reviewed_annotations_sha256: Digest
    validation_predictions_sha256: Digest
    review_approval_sha256: Digest
    recording_count: Annotated[int, Field(gt=0, strict=True)]
    training_group_overlap_count: Count
    final_check_group_overlap_count: Count
    jury_record_count: Count
    fact_calibration: dict[str, FactCalibration]
    decoder_sha256: Digest | None = None
    event_calibration: dict[str, EventCalibration] = Field(default_factory=dict)


class ReviewedEvidenceVerifier(Protocol):
    def verify(self, manifest: ReviewedCalibrationManifest) -> bool:
        """Independently verify hashes, human review, splits and actual predictions.

        A declared origin/approval string is not proof. The future trusted local
        evidence loader must check the bound manifest and review records. This
        protocol has no permissive production implementation in the prototype.
        """
        ...


@dataclass(frozen=True, slots=True)
class CalibrationReadiness:
    evidence_verified: bool
    threshold_coverage_complete: bool
    eligible_fact_states: Mapping[str, tuple[str, ...]]
    eligible_event_types: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "eligible_fact_states", MappingProxyType(dict(self.eligible_fact_states)))


def calibration_readiness(
    manifest: ReviewedCalibrationManifest | None,
    expected: HeadIdentity,
    *,
    verifier: ReviewedEvidenceVerifier | None,
) -> CalibrationReadiness:
    """Abstain by default; explicit thresholds alone never unlock decisions.

    Eligibility requires reviewed target and alternative-class examples plus
    reviewed source-support labels. Missing classes/events remain unavailable.
    These gates are structural prerequisites, not proof of quality: A10 must
    still approve actual errors, temporal localization and target performance.
    """
    empty = {fact_id: () for fact_id in expected.fact_ids}

    def blocked(code: str) -> CalibrationReadiness:
        return CalibrationReadiness(False, False, empty, (), (code,))

    if manifest is None:
        return blocked("missing_reviewed_calibration")
    if manifest.identity != expected:
        return blocked("calibration_identity_mismatch")
    if manifest.validation_origin != "human_reviewed_validation":
        return blocked("calibration_not_human_reviewed_validation")
    if (manifest.training_group_overlap_count or manifest.final_check_group_overlap_count
            or manifest.jury_record_count):
        return blocked("calibration_split_integrity_failed")
    if set(manifest.fact_calibration) != set(expected.fact_ids):
        return blocked("calibration_fact_axis_mismatch")
    if set(manifest.event_calibration) - set(expected.event_type_ids):
        return blocked("calibration_unknown_event_type")
    try:
        verified = verifier is not None and verifier.verify(manifest) is True
    except Exception:
        return blocked("calibration_evidence_verification_failed")
    if not verified:
        return blocked("calibration_evidence_not_verified")
    eligible = {}
    for fact_id in expected.fact_ids:
        calibration = manifest.fact_calibration[fact_id]
        states = []
        for state in ("present", "absent"):
            alternatives = sum(count for key, count in calibration.reviewed_class_counts.items() if key != state)
            if (getattr(calibration, state) is not None
                    and calibration.reviewed_class_counts.get(state, 0) > 0
                    and calibration.reviewed_support_counts.get(state, 0) > 0
                    and alternatives > 0):
                states.append(state)
        eligible[fact_id] = tuple(states)
    event_types = tuple(event_id for event_id in expected.event_type_ids if (
        manifest.decoder_sha256 is not None
        and event_id in manifest.event_calibration
        and manifest.event_calibration[event_id].reviewed_event_count > 0
        and manifest.event_calibration[event_id].reviewed_normal_count > 0
    ))
    complete = all(len(states) == 2 for states in eligible.values()) and event_types == expected.event_type_ids
    return CalibrationReadiness(True, complete, eligible, event_types,
                                () if complete else ("reviewed_class_or_localization_coverage_incomplete",))
