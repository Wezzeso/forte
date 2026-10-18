"""Synthetic machine-gate fixtures; threshold values are not production choices.

No teacher, encoder, training, reviewed labels, private media or inference runs.
"""

from hashlib import sha256

import pytest
from pydantic import ValidationError

from forte.decision.head_contract import HeadIdentity, ReviewedCalibrationManifest, StateThreshold
from forte.decision.machine_contract import (
    LocalTeacherIdentity, MachineCalibrationManifest, MachineEventCalibration,
    MachineEvidenceVerification, MachineFactCalibration, MachineStateCalibration,
    ProvidedBinaryCalibration, TeacherSupportAgreement,
    machine_candidate_readiness, machine_manifest_digest,
)


def digest(value):
    return sha256(value.encode()).hexdigest()


def identity():
    return HeadIdentity(
        encoder_id="synthetic-encoder", encoder_revision="synthetic-pinned-revision",
        encoder_artifacts_sha256=digest("encoder"), head_sha256=digest("head"),
        feature_spec_sha256=digest("features"), taxonomy_sha256=digest("taxonomy"),
        fact_ids=tuple(f"SYNTHETIC_FACT_{index:02d}" for index in range(19)),
        event_type_ids=("SYNTHETIC_EVENT",),
    )


def manifest():
    # Explicit numerical unit fixtures, never deployment thresholds.
    agreement = TeacherSupportAgreement(compared_count=10, agreed_count=9, minimum_agreement=0.8)
    threshold = StateThreshold(score_space="softmax_score", minimum_score=0.7, minimum_margin=0.1,
                               minimum_support_score=0.7, support_score_space="sigmoid_score")
    state = MachineStateCalibration(threshold=threshold, silver_support_count=1,
        silver_support_negative_count=1, support_agreement=agreement)
    fact = MachineFactCalibration(present=state, absent=state,
                                  silver_class_counts={"present": 1, "absent": 1, "unclear": 1})
    return MachineCalibrationManifest(
        supervision_kind="provided_binary_plus_local_teacher_silver", provided_label_scope="video_binary",
        fact_event_reference_kind="local_teacher_silver", identity=identity(),
        teacher_identity=LocalTeacherIdentity(
            model_id="synthetic-local-teacher", revision="synthetic-pinned-revision",
            model_artifacts_sha256=digest("teacher"), processor_artifacts_sha256=digest("processor"),
            proposal_config_sha256=digest("proposal-config"), execution_scope="team_local",
        ),
        split_assignments_sha256=digest("split"), exposure_manifest_sha256=digest("exposure"),
        provided_labels_sha256=digest("provided-labels"), input_set_sha256=digest("inputs"),
        predictions_sha256=digest("predictions"), teacher_proposals_sha256=digest("teacher-proposals"),
        teacher_support_agreement_sha256=digest("teacher-agreement"), threshold_selection_sha256=digest("selection"),
        provided_evaluation_attestation_sha256=digest("evaluation"), selection_attestation_sha256=digest("selection-attestation"),
        assignments_frozen=True, training_group_overlap_count=0, final_check_group_overlap_count=0, jury_record_count=0,
        provided_binary_calibration=ProvidedBinaryCalibration(
            score_axis="violation_log_odds", minimum_video_score=0.0,
            provided_positive_count=1, provided_negative_count=1,
        ),
        fact_calibration={fact_id: fact for fact_id in identity().fact_ids}, decoder_sha256=digest("decoder"),
        event_calibration={"SYNTHETIC_EVENT": MachineEventCalibration(
            score_space="raw_logit", minimum_decoder_score=0.0, silver_event_count=1, silver_normal_count=1,
            source_frame_binding_checked=True, timing_reference_kind="teacher_source_frame_brackets",
            support_agreement=agreement,
        )},
    )


class SyntheticVerifier:
    """Test double for A10/silver-evidence loaders; no production verifier."""
    def __init__(self, **updates):
        self.updates = updates

    def verify(self, value):
        return MachineEvidenceVerification(**{
            "manifest_sha256": machine_manifest_digest(value),
            "split_and_exposure_verified": True,
            "provided_label_evaluation_verified": True,
            "teacher_support_agreement_verified": True,
            "threshold_selection_verified": True,
            "provided_binary_target_met": False,
            "latency_target_met": False,
            **self.updates,
        })


def test_provisional_machine_candidate_can_progress_without_fake_gold_or_target_success():
    gate = machine_candidate_readiness(manifest(), identity(), verifier=SyntheticVerifier())
    assert gate.evidence_verified and gate.can_emit_provisional_facts
    assert gate.candidate_event_types == ("SYNTHETIC_EVENT",)
    assert gate.provisional and not gate.human_fact_event_gold
    assert not gate.independent_fact_event_accuracy_measured
    assert gate.fact_event_reference_kind == "local_teacher_silver"
    assert gate.provided_binary_target_met is False
    assert gate.latency_target_met is False


def test_machine_manifest_cannot_masquerade_as_reviewed_human_calibration():
    with pytest.raises(ValidationError):
        ReviewedCalibrationManifest.model_validate(manifest().model_dump())
    with pytest.raises(ValidationError):
        MachineCalibrationManifest.model_validate({**manifest().model_dump(), "human_fact_event_gold": True})


def test_missing_evidence_and_unbound_verification_never_unlock_candidates():
    assert not machine_candidate_readiness(None, identity(), verifier=None).can_emit_provisional_facts
    assert not machine_candidate_readiness(manifest(), identity(), verifier=None).can_emit_provisional_facts
    stale = SyntheticVerifier(manifest_sha256=digest("different-manifest"))
    assert not machine_candidate_readiness(manifest(), identity(), verifier=stale).can_emit_provisional_facts


@pytest.mark.parametrize("field", [
    "split_and_exposure_verified", "provided_label_evaluation_verified",
    "teacher_support_agreement_verified", "threshold_selection_verified",
])
def test_each_independent_machine_evidence_check_is_required(field):
    gate = machine_candidate_readiness(manifest(), identity(), verifier=SyntheticVerifier(**{field: False}))
    assert not gate.evidence_verified and not gate.can_emit_provisional_facts


@pytest.mark.parametrize("updates", [
    {"assignments_frozen": False}, {"training_group_overlap_count": 1},
    {"final_check_group_overlap_count": 1}, {"jury_record_count": 1},
])
def test_locked_group_exposure_and_jury_boundaries_remain_required(updates):
    value = manifest().model_copy(update=updates)
    assert not machine_candidate_readiness(value, identity(), verifier=SyntheticVerifier()).evidence_verified


def test_model_identity_and_binary_label_scope_are_not_fact_truth():
    other = identity().model_copy(update={"head_sha256": digest("other-head")})
    assert not machine_candidate_readiness(manifest(), other, verifier=SyntheticVerifier()).evidence_verified
    value = manifest()
    binary_only = {fact_id: MachineFactCalibration(silver_class_counts={}) for fact_id in identity().fact_ids}
    value = value.model_copy(update={"fact_calibration": binary_only})
    gate = machine_candidate_readiness(value, identity(), verifier=SyntheticVerifier())
    assert gate.evidence_verified and not gate.can_emit_provisional_facts
    assert gate.candidate_event_types == ()
    assert all(not states for states in gate.eligible_fact_states.values())


def test_low_teacher_agreement_and_unseen_support_remain_unknown():
    value = manifest()
    records = dict(value.fact_calibration)
    first = records[identity().fact_ids[0]]
    bad = first.present.model_copy(update={"support_agreement": TeacherSupportAgreement(
        compared_count=10, agreed_count=1, minimum_agreement=0.8,
    )})
    records[identity().fact_ids[0]] = first.model_copy(update={"present": bad, "absent": None})
    gate = machine_candidate_readiness(value.model_copy(update={"fact_calibration": records}), identity(),
                                       verifier=SyntheticVerifier())
    assert gate.can_emit_provisional_facts  # Other supported classes can progress.
    assert gate.eligible_fact_states[identity().fact_ids[0]] == ()
    with pytest.raises(TypeError):
        gate.eligible_fact_states[identity().fact_ids[0]] = ("present",)
    assert not TeacherSupportAgreement(compared_count=1, agreed_count=0, minimum_agreement=0.0).satisfied


def test_positive_only_support_does_not_establish_calibrated_support_specificity():
    value = manifest()
    records = {fact_id: record.model_copy(update={state: getattr(record, state).model_copy(
        update={"silver_support_negative_count": 0}) for state in ("present", "absent")})
        for fact_id, record in value.fact_calibration.items()}
    gate = machine_candidate_readiness(value.model_copy(update={"fact_calibration": records}), identity(),
                                       verifier=SyntheticVerifier())
    assert gate.evidence_verified and not gate.can_emit_provisional_facts


def test_event_brackets_require_decoder_support_and_verified_source_bindings():
    value = manifest()
    record = value.event_calibration["SYNTHETIC_EVENT"].model_copy(update={"source_frame_binding_checked": False})
    value = value.model_copy(update={"event_calibration": {"SYNTHETIC_EVENT": record}})
    gate = machine_candidate_readiness(value, identity(), verifier=SyntheticVerifier())
    assert gate.can_emit_provisional_facts and gate.candidate_event_types == ()


def test_machine_loader_failure_does_not_echo_private_text():
    class FailingVerifier:
        def verify(self, value):
            raise RuntimeError("SYNTHETIC_PRIVATE_SENTINEL")
    gate = machine_candidate_readiness(manifest(), identity(), verifier=FailingVerifier())
    assert gate.reason_codes == ("machine_evidence_verification_failed",)
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in repr(gate)


def test_local_teacher_identity_and_agreement_counts_are_strict():
    value = manifest().teacher_identity.model_dump()
    with pytest.raises(ValidationError):
        LocalTeacherIdentity.model_validate({**value, "execution_scope": "external_cloud"})
    with pytest.raises(ValidationError):
        TeacherSupportAgreement(compared_count=1, agreed_count=2, minimum_agreement=0.8)
