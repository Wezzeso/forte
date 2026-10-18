"""Separate provisional machine-supervision contract; never human-gold approval.

Organizer binary labels supervise MIL video/eventness only. Sparse local teacher
outputs are silver references for facts, supports and temporal boundaries;
unmarked or context-invalid pairs stay masked. Teacher agreement is not fact or
event accuracy. The existing reviewed-calibration contract remains unchanged.

An autonomous candidate may use machine-selected thresholds after independent
provided-label evaluation and verification of the bound teacher-support evidence.
That is a provisional operating mode, with its provenance retained internally
and in the UI. It cannot add jury fields, fabricate source facts/times, turn a
MIL binary score into nineteen fact labels, or bypass legal/evidence guards.
Target performance is reported separately from structural candidate readiness.

This prototype is not imported by the live provider and performs no inference,
training, file access, threshold selection, or deployment action.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .head_contract import Count, Digest, Finite, HeadIdentity, Identifier, StateThreshold

MACHINE_CONTRACT_VERSION = "forte.machine-candidate-calibration.v1"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class LocalTeacherIdentity(_Contract):
    model_id: Identifier
    revision: Identifier
    model_artifacts_sha256: Digest
    processor_artifacts_sha256: Digest
    proposal_config_sha256: Digest
    execution_scope: Literal["team_local"]


class TeacherSupportAgreement(_Contract):
    reference_kind: Literal["local_teacher_silver"] = "local_teacher_silver"
    compared_count: Count
    agreed_count: Count
    # Selected/frozen with the machine calibration bundle, never defaulted here.
    minimum_agreement: Annotated[float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)]

    @model_validator(mode="after")
    def consistent_counts(self):
        if self.agreed_count > self.compared_count:
            raise ValueError("agreement cannot exceed compared references")
        return self

    @property
    def satisfied(self) -> bool:
        return (self.compared_count > 0 and self.agreed_count > 0
                and self.agreed_count / self.compared_count >= self.minimum_agreement)


class MachineStateCalibration(_Contract):
    threshold: StateThreshold
    silver_support_count: Count
    # Explicit inspected non-support alternatives, never unmarked frames.
    silver_support_negative_count: Count = 0
    support_agreement: TeacherSupportAgreement


class MachineFactCalibration(_Contract):
    present: MachineStateCalibration | None = None
    absent: MachineStateCalibration | None = None
    silver_class_counts: dict[Literal["present", "absent", "unclear"], Count]


class MachineEventCalibration(_Contract):
    score_space: Literal["raw_logit", "sigmoid_score", "softmax_score"]
    minimum_decoder_score: Finite
    silver_event_count: Count
    silver_normal_count: Count
    source_frame_binding_checked: bool
    timing_reference_kind: Literal["teacher_source_frame_brackets"]
    support_agreement: TeacherSupportAgreement

    @model_validator(mode="after")
    def normalized_score_bounds(self):
        if self.score_space != "raw_logit" and not 0 <= self.minimum_decoder_score <= 1:
            raise ValueError("normalized decoder thresholds must remain within score bounds")
        return self


class ProvidedBinaryCalibration(_Contract):
    """A binary video reference can prioritize search, never invent a timestamp."""
    score_axis: Literal["violation_log_odds"]
    minimum_video_score: Finite
    provided_positive_count: Count
    provided_negative_count: Count


class MachineCalibrationManifest(_Contract):
    contract_version: Literal["forte.machine-candidate-calibration.v1"] = MACHINE_CONTRACT_VERSION
    supervision_kind: Literal["provided_binary_plus_local_teacher_silver"]
    provided_label_scope: Literal["video_binary"]
    fact_event_reference_kind: Literal["local_teacher_silver"]
    human_fact_event_gold: Literal[False] = False
    independent_fact_event_accuracy_measured: Literal[False] = False
    identity: HeadIdentity
    teacher_identity: LocalTeacherIdentity
    split_assignments_sha256: Digest
    exposure_manifest_sha256: Digest
    provided_labels_sha256: Digest
    input_set_sha256: Digest
    predictions_sha256: Digest
    teacher_proposals_sha256: Digest
    teacher_support_agreement_sha256: Digest
    threshold_selection_sha256: Digest
    provided_evaluation_attestation_sha256: Digest
    selection_attestation_sha256: Digest
    # A02's automated split may retain human-track locked=False. This records
    # the separate machine experiment freeze and independently pinned checksum.
    assignments_frozen: bool
    training_group_overlap_count: Count
    final_check_group_overlap_count: Count
    jury_record_count: Count
    provided_binary_calibration: ProvidedBinaryCalibration
    fact_calibration: dict[str, MachineFactCalibration]
    decoder_sha256: Digest | None = None
    event_calibration: dict[str, MachineEventCalibration] = Field(default_factory=dict)


def machine_manifest_digest(manifest: MachineCalibrationManifest) -> str:
    data = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class MachineEvidenceVerification(_Contract):
    manifest_sha256: Digest
    split_and_exposure_verified: bool
    provided_label_evaluation_verified: bool
    teacher_support_agreement_verified: bool
    threshold_selection_verified: bool
    # These actual evaluation results may be false without forbidding an honest
    # provisional candidate. Missing results remain unknown, not success.
    provided_binary_target_met: bool | None
    latency_target_met: bool | None
    independent_fact_event_accuracy_measured: Literal[False] = False


class MachineEvidenceVerifier(Protocol):
    def verify(self, manifest: MachineCalibrationManifest) -> MachineEvidenceVerification:
        """Check frozen hashes, machine-group/exposure splits and A10 attestation.

        Verify provided binary labels/predictions independently; verify teacher
        proposal/support matching and threshold selection without calling them
        human gold. No permissive production implementation is supplied here.
        """
        ...


@dataclass(frozen=True, slots=True)
class MachineCandidateReadiness:
    evidence_verified: bool
    can_emit_provisional_facts: bool
    eligible_fact_states: Mapping[str, tuple[str, ...]]
    candidate_event_types: tuple[str, ...]
    reason_codes: tuple[str, ...]
    provided_binary_target_met: bool | None = None
    latency_target_met: bool | None = None
    provisional: Literal[True] = True
    human_fact_event_gold: Literal[False] = False
    independent_fact_event_accuracy_measured: Literal[False] = False
    fact_event_reference_kind: Literal["local_teacher_silver"] = "local_teacher_silver"

    def __post_init__(self):
        object.__setattr__(self, "eligible_fact_states", MappingProxyType(dict(self.eligible_fact_states)))


def machine_candidate_readiness(
    manifest: MachineCalibrationManifest | None,
    expected: HeadIdentity,
    *, verifier: MachineEvidenceVerifier | None,
) -> MachineCandidateReadiness:
    """Allow honestly provisional autonomous progress without fake human review.

    Unknown/unsupported fact classes remain unavailable; the future provider
    maps them to unclear. Full legal relevance/context and source-frame guards
    still apply after this gate. MIL alone never unlocks a fact or event type.
    """
    empty = {fact_id: () for fact_id in expected.fact_ids}

    def blocked(code: str):
        return MachineCandidateReadiness(False, False, empty, (), (code,))

    if manifest is None:
        return blocked("missing_machine_calibration")
    if manifest.identity != expected:
        return blocked("machine_calibration_identity_mismatch")
    if (not manifest.assignments_frozen or manifest.training_group_overlap_count
            or manifest.final_check_group_overlap_count or manifest.jury_record_count):
        return blocked("machine_split_or_exposure_integrity_failed")
    if set(manifest.fact_calibration) != set(expected.fact_ids):
        return blocked("machine_fact_axis_mismatch")
    if set(manifest.event_calibration) - set(expected.event_type_ids):
        return blocked("machine_unknown_event_type")
    binary = manifest.provided_binary_calibration
    if not binary.provided_positive_count or not binary.provided_negative_count:
        return blocked("provided_binary_validation_class_missing")
    if verifier is None:
        return blocked("machine_evidence_not_verified")
    try:
        proof = verifier.verify(manifest)
    except Exception:
        return blocked("machine_evidence_verification_failed")
    if (not isinstance(proof, MachineEvidenceVerification)
            or proof.manifest_sha256 != machine_manifest_digest(manifest)
            or not all((proof.split_and_exposure_verified,
                        proof.provided_label_evaluation_verified,
                        proof.teacher_support_agreement_verified,
                        proof.threshold_selection_verified))):
        return blocked("machine_evidence_not_verified")
    eligible = {}
    for fact_id in expected.fact_ids:
        record = manifest.fact_calibration[fact_id]
        states = []
        for state in ("present", "absent"):
            calibration = getattr(record, state)
            alternatives = sum(count for key, count in record.silver_class_counts.items() if key != state)
            if (calibration is not None and record.silver_class_counts.get(state, 0) > 0
                    and alternatives > 0 and calibration.silver_support_count > 0
                    and calibration.silver_support_negative_count > 0
                    and calibration.support_agreement.satisfied):
                states.append(state)
        eligible[fact_id] = tuple(states)
    events = tuple(event_id for event_id in expected.event_type_ids if (
        manifest.decoder_sha256 is not None and event_id in manifest.event_calibration
        and manifest.event_calibration[event_id].silver_event_count > 0
        and manifest.event_calibration[event_id].silver_normal_count > 0
        and manifest.event_calibration[event_id].source_frame_binding_checked
        and manifest.event_calibration[event_id].support_agreement.satisfied
    ))
    any_facts = any(eligible.values())
    if not any_facts:
        events = ()
    reasons = () if any_facts else ("teacher_supported_fact_classes_unavailable",)
    return MachineCandidateReadiness(
        True, any_facts, eligible, events, reasons,
        provided_binary_target_met=proof.provided_binary_target_met,
        latency_target_met=proof.latency_target_met,
    )
