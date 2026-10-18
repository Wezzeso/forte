"""Synthetic shape/trust-boundary tests; no calibrated model or quality claim.

Numerical thresholds below are explicit unit fixtures only. No production
calibration artifact, model, private observation, or GPU call is created.
"""

from hashlib import sha256

import pytest
from pydantic import ValidationError

from forte.decision.head_contract import (
    EventCalibration, FactCalibration, HeadContractError, HeadIdentity,
    ReviewedCalibrationManifest, SourceFrame, StateThreshold,
    TemporalHeadOutput, TemporalProposal, calibration_readiness, validate_head_binding,
)
from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef


def digest(value):
    return sha256(value.encode()).hexdigest()


def identity():
    return HeadIdentity(
        encoder_id="synthetic-contract-encoder", encoder_revision="synthetic-pinned-revision",
        encoder_artifacts_sha256=digest("encoder"), head_sha256=digest("head"),
        feature_spec_sha256=digest("feature-spec"), taxonomy_sha256=digest("taxonomy"),
        fact_ids=tuple(f"SYNTHETIC_FACT_{index:02d}" for index in range(19)),
        event_type_ids=("SYNTHETIC_EVENT",),
    )


def request():
    manifest = identity()
    return ClipRequest(
        job_id="synthetic-head-job", clip_id="synthetic-head-clip",
        frame_refs=[FrameRef(frame_id=digest(f"frame-{index}"), pts_sec=float(index),
                             path=f"/synthetic-only/frame-{index}.jpg", width=32, height=32)
                    for index in range(2)],
        fact_definitions=[FactDefinition(fact_id=fact_id, description="Synthetic shape fixture only.")
                          for fact_id in manifest.fact_ids],
    )


def output(*, proposals=False):
    clip, manifest = request(), identity()
    return TemporalHeadOutput(
        identity=manifest, job_id=clip.job_id, clip_id=clip.clip_id,
        frames=tuple(SourceFrame(frame_id=frame.frame_id, pts_sec=frame.pts_sec) for frame in clip.frame_refs),
        window_logits=((0.0, 0.0, 0.0),) * 19,
        frame_logits=(((0.0, 0.0, 0.0),) * 19,) * 2,
        support_logits=((0.0,) * 19,) * 2,
        frame_valid=(True, True), global_view_available=(True, True),
        feature_coverage_complete=True,
        boundary_logits=(((0.0, 0.0),),) * 2,
        decoder_sha256=digest("decoder") if proposals else None,
        proposals=(TemporalProposal(
            proposal_id="synthetic-paired-bracket", event_type_id="SYNTHETIC_EVENT",
            start_frame_id=clip.frame_refs[0].frame_id, end_frame_id=clip.frame_refs[1].frame_id,
            supporting_fact_ids=(manifest.fact_ids[0],),
            supporting_frame_ids=(clip.frame_refs[0].frame_id,), decoder_score=0.0,
        ),) if proposals else (),
    )


def calibration():
    # Synthetic numerical unit fixtures, never used as deployment thresholds.
    threshold = StateThreshold(score_space="softmax_score", minimum_score=0.75,
                               minimum_margin=0.1, minimum_support_score=0.75,
                               support_score_space="sigmoid_score")
    fact = FactCalibration(present=threshold, absent=threshold,
                           reviewed_class_counts={"present": 1, "absent": 1, "unclear": 1},
                           reviewed_support_counts={"present": 1, "absent": 1})
    return ReviewedCalibrationManifest(
        identity=identity(), validation_origin="human_reviewed_validation", split_role="validation",
        split_manifest_sha256=digest("split"), reviewed_annotations_sha256=digest("reviews"),
        validation_predictions_sha256=digest("validation-predictions"), review_approval_sha256=digest("approval"),
        recording_count=1, training_group_overlap_count=0, final_check_group_overlap_count=0, jury_record_count=0,
        fact_calibration={fact_id: fact for fact_id in identity().fact_ids}, decoder_sha256=digest("decoder"),
        event_calibration={"SYNTHETIC_EVENT": EventCalibration(
            score_space="raw_logit", minimum_decoder_score=0.0, reviewed_event_count=1, reviewed_normal_count=1,
        )},
    )


class SyntheticVerifier:
    """Test double for the future independently trusted evidence loader."""
    def __init__(self, result=True):
        self.result = result

    def verify(self, manifest):
        return self.result


def test_head_binds_exact_original_frames_timestamps_and_nineteen_fact_axis():
    validate_head_binding(output(), request(), identity())
    with_proposals = output(proposals=True)
    validate_head_binding(with_proposals, request(), identity(), expected_decoder_sha256=digest("decoder"))
    with pytest.raises(HeadContractError, match="unverified_temporal_decoder"):
        validate_head_binding(with_proposals, request(), identity())


@pytest.mark.parametrize("change,code", [
    ({"job_id": "other-job"}, "request_identity_mismatch"),
    ({"clip_id": "other-clip"}, "request_identity_mismatch"),
    ({"identity": identity().model_copy(update={"head_sha256": digest("other-head")})}, "head_identity_mismatch"),
    ({"frames": (SourceFrame(frame_id="invented", pts_sec=0.0), output().frames[1])}, "source_frame_binding_mismatch"),
    ({"frames": (SourceFrame(frame_id=output().frames[0].frame_id, pts_sec=0.1), output().frames[1])}, "source_frame_binding_mismatch"),
])
def test_head_rejects_changed_source_or_model_binding(change, code):
    with pytest.raises(HeadContractError, match=code):
        validate_head_binding(output().model_copy(update=change), request(), identity())


@pytest.mark.parametrize("field,value", [
    ("window_logits", ((0.0, 0.0, 0.0),) * 18),
    ("frame_logits", (((0.0, 0.0, 0.0),) * 19,)),
    ("support_logits", ((0.0,) * 18,) * 2),
    ("frame_valid", (True,)),
    ("boundary_logits", (((0.0, 0.0),),)),
    ("window_logits", ((float("nan"), 0.0, 0.0),) * 19),
])
def test_head_rejects_bad_shapes_and_nonfinite_logits(field, value):
    with pytest.raises(ValidationError):
        TemporalHeadOutput.model_validate({**output().model_dump(), field: value})


def test_head_rejects_fabricated_supports_and_reversed_temporal_brackets():
    raw = output(proposals=True)
    for update in (
        {"supporting_frame_ids": ("invented",)},
        {"supporting_fact_ids": ("invented",)},
        {"start_frame_id": raw.frames[1].frame_id, "end_frame_id": raw.frames[0].frame_id},
    ):
        changed = raw.model_copy(update={"proposals": (raw.proposals[0].model_copy(update=update),)})
        with pytest.raises(HeadContractError):
            validate_head_binding(changed, request(), identity(), expected_decoder_sha256=digest("decoder"))


def test_missing_synthetic_or_unverified_calibration_never_unlocks_decisive_states():
    missing = calibration_readiness(None, identity(), verifier=None)
    assert not missing.evidence_verified and not missing.threshold_coverage_complete
    assert all(not states for states in missing.eligible_fact_states.values())
    synthetic = calibration().model_copy(update={"validation_origin": "synthetic"})
    assert not calibration_readiness(synthetic, identity(), verifier=SyntheticVerifier()).evidence_verified
    assert not calibration_readiness(calibration(), identity(), verifier=None).evidence_verified
    assert not calibration_readiness(calibration(), identity(), verifier=SyntheticVerifier(False)).evidence_verified


@pytest.mark.parametrize("field", ["training_group_overlap_count", "final_check_group_overlap_count", "jury_record_count"])
def test_calibration_requires_genuine_validation_split_isolation(field):
    manifest = calibration().model_copy(update={field: 1})
    gate = calibration_readiness(manifest, identity(), verifier=SyntheticVerifier())
    assert not gate.evidence_verified
    assert gate.reason_codes == ("calibration_split_integrity_failed",)


def test_calibration_rejects_identity_mismatch_and_missing_fact_threshold_record():
    manifest = calibration()
    different = identity().model_copy(update={"head_sha256": digest("other-head")})
    assert not calibration_readiness(manifest, different, verifier=SyntheticVerifier()).evidence_verified
    subset = dict(manifest.fact_calibration); subset.pop(identity().fact_ids[0])
    assert not calibration_readiness(manifest.model_copy(update={"fact_calibration": subset}), identity(),
                                     verifier=SyntheticVerifier()).evidence_verified


def test_unseen_classes_missing_support_and_missing_localization_remain_unavailable():
    manifest = calibration()
    records = dict(manifest.fact_calibration)
    records[identity().fact_ids[0]] = records[identity().fact_ids[0]].model_copy(update={
        "reviewed_class_counts": {"present": 1, "absent": 0, "unclear": 0},
        "reviewed_support_counts": {"present": 0, "absent": 0},
    })
    partial = manifest.model_copy(update={"fact_calibration": records, "event_calibration": {}})
    gate = calibration_readiness(partial, identity(), verifier=SyntheticVerifier())
    assert gate.evidence_verified and not gate.threshold_coverage_complete
    assert gate.eligible_fact_states[identity().fact_ids[0]] == ()
    assert gate.eligible_event_types == ()
    with pytest.raises(TypeError):
        gate.eligible_fact_states[identity().fact_ids[0]] = ("present",)


def test_structural_eligibility_still_requires_external_quality_approval():
    # The test verifier stands in for evidence checks, not real human review.
    gate = calibration_readiness(calibration(), identity(), verifier=SyntheticVerifier())
    assert gate.evidence_verified and gate.threshold_coverage_complete
    assert all(states == ("present", "absent") for states in gate.eligible_fact_states.values())
    assert gate.eligible_event_types == ("SYNTHETIC_EVENT",)


def test_evidence_verifier_failure_exposes_no_private_exception_text():
    class FailingVerifier:
        def verify(self, manifest):
            raise RuntimeError("SYNTHETIC_PRIVATE_SENTINEL")
    gate = calibration_readiness(calibration(), identity(), verifier=FailingVerifier())
    assert gate.reason_codes == ("calibration_evidence_verification_failed",)
    assert "SYNTHETIC_PRIVATE_SENTINEL" not in repr(gate)
