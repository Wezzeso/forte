"""Synthetic pinned artifact composition; no real labels/model/teacher run."""
from dataclasses import replace
from hashlib import sha256
import json

from evaluation.io import digest as json_digest
from evaluation.weak_supervision import weak_assignment_digest
import pytest

from forte.decision.machine_artifacts import (
    ArtifactMachineEvidenceVerifier, MachineEvidenceFiles, verify_teacher_comparisons,
    teacher_processor_artifacts_sha256,
)
from forte.decision.machine_contract import TeacherSupportAgreement, machine_candidate_readiness
from forte.decision.temporal_provider import TemporalProviderError
from forte.video.batches import source_frame_id
from test_machine_calibration_contract import manifest, digest


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    value = manifest()
    records = {}
    agreement = TeacherSupportAgreement(compared_count=2, agreed_count=2, minimum_agreement=0.8)
    for fact_id, record in value.fact_calibration.items():
        records[fact_id] = record.model_copy(update={state: getattr(record, state).model_copy(
            update={"support_agreement": agreement}) for state in ("present", "absent")})
    value = value.model_copy(update={"fact_calibration": records, "event_calibration": {}, "decoder_sha256": None})
    paths = {key: tmp_path / f"{key}.json" for key in MachineEvidenceFiles.__dataclass_fields__}
    teacher_model = {"model_id": value.teacher_identity.model_id, "revision": "a" * 40,
        "processor_revision": "a" * 40, "license": "apache-2.0", "license_verified": True, "selected_adapter": None,
        "files": [{"path": name, "sha256": digest("synthetic-" + name), "size_bytes": 1}
                  for name in ("head.safetensors", "config.json", "preprocessor_config.json", "tokenizer_config.json")]}
    teacher_manifest_sha = write(paths["teacher_model_manifest"], teacher_model)
    value = value.model_copy(update={"teacher_identity": value.teacher_identity.model_copy(update={
        "revision": "a" * 40, "model_artifacts_sha256": teacher_manifest_sha,
        "processor_artifacts_sha256": teacher_processor_artifacts_sha256(teacher_model)})})
    split_rows = [{"content_sha256": digest("synthetic-validation-recording"), "split": "validation", "group_id": "synthetic-group"}]
    split_sha = weak_assignment_digest(split_rows)
    write(paths["split_assignments"], {"records": split_rows, "assignments_sha256": split_sha, "locked": False})
    write(paths["experiment_config"], {"validation_assignments_frozen": True})
    exposure_sha = write(paths["exposure_manifest"], {"synthetic_only": True})
    gate = {"ok": True, "label_split": "validation", "expected_assignments_sha256": split_sha,
            "exposure_registry_file_sha256": exposure_sha}
    write(paths["weak_gate_proof"], gate)
    proposals_sha = write(paths["teacher_proposals"], {"synthetic_only": True})
    value = value.model_copy(update={"split_assignments_sha256": split_sha,
        "exposure_manifest_sha256": exposure_sha, "teacher_proposals_sha256": proposals_sha})
    comparisons = {"schema_version": "forte.teacher-support-comparisons.v1",
        "head_identity": value.identity.model_dump(mode="json"),
        "teacher_identity": value.teacher_identity.model_dump(mode="json"),
        "fact_comparisons": [], "event_comparisons": []}
    content = split_rows[0]["content_sha256"]
    source_ids = [source_frame_id(content, index, 1, 1) for index in range(3)]
    targets = []
    for fact_id in value.identity.fact_ids:
        for state in ("present", "absent", "unclear"):
            focal = source_ids[("present", "absent", "unclear").index(state)]
            question = {"fact_id": "ANCHOR__" + fact_id, "description": "Synthetic exact-frame question: " + focal, "required": True}
            targets.append({"fact_id": fact_id, "state": state, "frame_id": focal,
                "scope": "exact_anchor_frame", "derivation": "direct_anchor_teacher_question",
                "supervision_mask": True, "human_verified": False, "review_status": "machine_consistency_checked",
                "supporting_frame_ids": [focal] if state != "unclear" else [], "question_definition": question,
                "question_fact_id": "ANCHOR__" + fact_id,
                "provenance": {"prompt_sha256": digest("synthetic-prompt"), "focal_question_sha256": json_digest(question)}})
            comparisons["fact_comparisons"].append({"comparison_id": fact_id + ":" + state,
                "fact_id": fact_id, "state": state, "predicted_state": state,
                "supports_state": True,
                "source_binding_verified": True, "procedure_context_complete": True, "visibility_adequate": True,
                "source_frame_ids": source_ids, "teacher_supporting_frame_ids": [focal] if state != "unclear" else [],
                "predicted_supporting_frame_ids": [focal] if state != "unclear" else []})
            if state != "unclear":
                comparisons["fact_comparisons"].append({"comparison_id": fact_id + ":" + state + ":explicit-non-support",
                    "fact_id": fact_id, "state": state, "predicted_state": state,
                    "supports_state": False, "source_binding_verified": True,
                    "negative_reference_kind": "focal_frame_explicit_unclear", "teacher_state": "unclear",
                    "focal_frame_id": source_ids[2], "prompt_sha256": digest("synthetic-prompt"),
                    "focal_question_sha256": digest("synthetic-focal-question"),
                    "source_frame_ids": source_ids, "teacher_supporting_frame_ids": [],
                    "predicted_supporting_frame_ids": []})
    window = {"model_output_complete": True, "frame_ids": source_ids,
        "provenance": {"prompt_sha256": digest("synthetic-prompt"),
            "proposal_config_sha256": value.teacher_identity.proposal_config_sha256}, "frame_fact_targets": targets,
        "context_masks": {"procedure_complete": True, "continuous_visibility": True}}
    teacher_record = {"schema_version": "forte.machine-annotations.v1", "annotation_origin": "local_model",
        "review_status": "machine_consistency_checked", "human_reviewed": False,
        "source_pts_identity_checked": True, "split": "validation", "content_sha256": content,
        "source_time_base": "1/1", "source_start_sec": 0.0,
        "frames": [{"frame_id": frame_id, "source_pts": index, "pts_sec": float(index)} for index, frame_id in enumerate(source_ids)],
        "windows": [window], "provenance": {"model_id": value.teacher_identity.model_id,
            "model_revision": value.teacher_identity.revision,
            "processor_revision": value.teacher_identity.revision,
            "model_manifest_sha256": value.teacher_identity.model_artifacts_sha256, "actual_snapshot_verified": True,
            "execution_location": "team_deployed_service", "external_ai_used": False,
            "serving_attestation": {"model_files_verified": True,
                "model_manifest_sha256": value.teacher_identity.model_artifacts_sha256,
                "model_revision": value.teacher_identity.revision, "processor_revision": value.teacher_identity.revision}}}
    proposals_sha = write(paths["teacher_proposals"], teacher_record)
    value = value.model_copy(update={"teacher_proposals_sha256": proposals_sha})
    for comparison in comparisons["fact_comparisons"]:
        state = comparison["state"] if comparison["supports_state"] else "unclear"
        target = next(target for target in targets if target["fact_id"] == comparison["fact_id"] and target["state"] == state)
        comparison.update(teacher_reference_kind="frame_fact_target", teacher_record_sha256=json_digest(teacher_record),
            teacher_window_sha256=json_digest(window), teacher_target_sha256=json_digest(target),
            content_sha256=content, focal_frame_id=target["frame_id"], prompt_sha256=digest("synthetic-prompt"),
            focal_question_sha256=target["provenance"]["focal_question_sha256"])
    comparisons_sha = write(paths["teacher_comparisons"], comparisons)
    report = {"schema_version": "forte.teacher-support-agreement.v1",
        "head_identity": value.identity.model_dump(mode="json"), "teacher_identity": value.teacher_identity.model_dump(mode="json"),
        "teacher_proposals_sha256": proposals_sha, "split_assignments_sha256": split_sha,
        "comparison_records_sha256": comparisons_sha, "reference_kind": "local_teacher_silver", "teacher_agreement_is_accuracy": False}
    agreement_sha = write(paths["teacher_support_agreement"], report)
    threshold = {"schema_version": "forte.weak_video_threshold.v1", "score_axis": "violation_log_odds",
        "minimum_video_score": 0.0, "comparison": "greater_than_or_equal",
        "selection_method": "synthetic_test_fixture_only", "head_sha256": value.identity.head_sha256,
        "head_identity_sha256": json_digest(value.identity.model_dump(mode="json")),
        "feature_spec_sha256": value.identity.feature_spec_sha256,
        "provided_validation_labels_sha256": value.provided_labels_sha256,
        "fact_calibration": {key: record.model_dump(mode="json") for key, record in value.fact_calibration.items()},
        "event_calibration": {}, "decoder_sha256": None}
    threshold_sha = write(paths["threshold_selection"], threshold)
    value = value.model_copy(update={"threshold_selection_sha256": threshold_sha, "teacher_support_agreement_sha256": agreement_sha})
    evaluation = {"schema_version": "forte.weak_operational_evaluation.v1", "model_selection_eligible": True,
        **{key: getattr(value, key) for key in ("split_assignments_sha256", "provided_labels_sha256", "input_set_sha256", "predictions_sha256")},
        "provenance": {"head_identity_sha256": json_digest(value.identity.model_dump(mode="json")),
            "threshold_selection_sha256": threshold_sha, "exposure_manifest_sha256": exposure_sha,
            "weak_gate_proof_sha256": json_digest(gate)},
        "target_observations": {"provided_binary_detection_target_met": False, "all_attempt_public_e2e_10s_target_met": False}}
    evaluation["provided_evaluation_attestation_sha256"] = json_digest(evaluation)
    write(paths["provided_evaluation"], evaluation)
    selection = {"selected": "candidate", "candidate_evaluation_attestation_sha256": evaluation["provided_evaluation_attestation_sha256"]}
    selection["selection_attestation_sha256"] = json_digest(selection)
    write(paths["selection"], selection)
    value = value.model_copy(update={"provided_evaluation_attestation_sha256": evaluation["provided_evaluation_attestation_sha256"],
        "selection_attestation_sha256": selection["selection_attestation_sha256"]})
    return value, MachineEvidenceFiles(**paths), report, comparisons


def test_actual_local_file_verifier_composes_provisional_evidence_without_claiming_target_success(tmp_path):
    value, files, _, _ = fixture(tmp_path)
    verifier = ArtifactMachineEvidenceVerifier(files)
    proof = verifier.verify(value)
    assert proof.split_and_exposure_verified and proof.provided_label_evaluation_verified
    assert proof.teacher_support_agreement_verified and proof.threshold_selection_verified
    gate = machine_candidate_readiness(value, value.identity, verifier=verifier)
    assert gate.can_emit_provisional_facts and gate.provided_binary_target_met is False
    assert not gate.human_fact_event_gold and not gate.independent_fact_event_accuracy_measured


@pytest.mark.parametrize("changed", ["threshold_selection", "teacher_proposals", "teacher_support_agreement",
                                     "teacher_comparisons", "exposure_manifest", "weak_gate_proof", "provided_evaluation"])
def test_changed_bound_artifact_cannot_unlock_candidate(tmp_path, changed):
    value, files, _, _ = fixture(tmp_path)
    write(getattr(files, changed), {"changed": True})
    gate = machine_candidate_readiness(value, value.identity, verifier=ArtifactMachineEvidenceVerifier(files))
    assert not gate.evidence_verified and not gate.can_emit_provisional_facts


def test_binary_threshold_file_alone_does_not_authorize_nineteen_fact_thresholds(tmp_path):
    value, files, _, _ = fixture(tmp_path)
    threshold = json.loads(files.threshold_selection.read_text(encoding="utf-8"))
    del threshold["fact_calibration"]
    value = value.model_copy(update={"threshold_selection_sha256": write(files.threshold_selection, threshold)})
    assert ArtifactMachineEvidenceVerifier(files).verify(value).threshold_selection_verified is False


def test_teacher_support_is_recounted_not_trusted_as_an_agreement_boolean(tmp_path):
    value, _, report, comparisons = fixture(tmp_path)
    assert verify_teacher_comparisons(report, comparisons, value)
    comparisons["fact_comparisons"][0]["predicted_supporting_frame_ids"] = []
    assert not verify_teacher_comparisons(report, comparisons, value)


def test_absence_comparison_requires_explicit_complete_procedure_and_visibility(tmp_path):
    value, _, report, comparisons = fixture(tmp_path)
    row = next(row for row in comparisons["fact_comparisons"] if row["state"] == "absent" and row["supports_state"])
    row["procedure_context_complete"] = False
    assert not verify_teacher_comparisons(report, comparisons, value)


def test_missing_or_duplicate_json_artifacts_fail_with_static_no_content_error(tmp_path):
    value, files, _, _ = fixture(tmp_path)
    files.threshold_selection.write_text('{"x":1,"x":"SYNTHETIC_PRIVATE_SENTINEL"}', encoding="utf-8")
    with pytest.raises(TemporalProviderError) as error:
        ArtifactMachineEvidenceVerifier(files).verify(value)
    assert str(error.value) == "optional_temporal_provider_incomplete"


@pytest.mark.parametrize("field", ["teacher_record_sha256", "teacher_window_sha256", "teacher_target_sha256",
                                   "focal_question_sha256", "prompt_sha256", "content_sha256"])
def test_fully_repinned_comparison_must_still_join_actual_teacher_source_target(tmp_path, field):
    value, files, report, comparisons = fixture(tmp_path)
    comparisons["fact_comparisons"][0][field] = digest("synthetic-different-teacher-reference")
    report["comparison_records_sha256"] = write(files.teacher_comparisons, comparisons)
    value = value.model_copy(update={"teacher_support_agreement_sha256": write(files.teacher_support_agreement, report)})
    # Counts and ordinary hash declarations still match. The actual joined
    # teacher source/window/question/state reference must independently match.
    proof = ArtifactMachineEvidenceVerifier(files).verify(value)
    assert not proof.teacher_support_agreement_verified


def test_focal_unclear_negative_joins_actual_exported_support_review_receipt(tmp_path):
    value, files, report, comparisons = fixture(tmp_path)
    teacher = json.loads(files.teacher_proposals.read_text(encoding="utf-8"))
    window = teacher["windows"][0]
    negative = next(row for row in comparisons["fact_comparisons"] if not row["supports_state"])
    target = next(target for target in window["frame_fact_targets"]
                  if target["fact_id"] == negative["fact_id"] and target["state"] == "unclear")
    review = {key: target[key] for key in ("fact_id", "frame_id", "question_definition", "question_fact_id",
        "provenance", "review_status", "supervision_mask", "human_verified")}
    review.update(supports_state=False, teacher_state="unclear", negative_reference_kind="focal_frame_explicit_unclear")
    window["support_reviews"] = [review]
    negative.update(teacher_reference_kind="support_review", teacher_target_sha256=json_digest(review),
                    support_review_sha256=json_digest(review))
    teacher_sha = write(files.teacher_proposals, teacher)
    for row in comparisons["fact_comparisons"]:
        row.update(teacher_record_sha256=json_digest(teacher), teacher_window_sha256=json_digest(window))
    report["teacher_proposals_sha256"] = teacher_sha
    report["comparison_records_sha256"] = write(files.teacher_comparisons, comparisons)
    value = value.model_copy(update={"teacher_proposals_sha256": teacher_sha,
        "teacher_support_agreement_sha256": write(files.teacher_support_agreement, report)})
    assert ArtifactMachineEvidenceVerifier(files).verify(value).teacher_support_agreement_verified
    # A real receipt cannot be transplanted to a different source focal frame.
    negative["focal_frame_id"] = negative["source_frame_ids"][0]
    report["comparison_records_sha256"] = write(files.teacher_comparisons, comparisons)
    value = value.model_copy(update={"teacher_support_agreement_sha256": write(files.teacher_support_agreement, report)})
    assert not ArtifactMachineEvidenceVerifier(files).verify(value).teacher_support_agreement_verified
