"""Generated CPU-only scope contract checks; no private media/model/HTTP."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import pytest

from forte.video.batches import source_frame_id
from training.guards import TrainingGateError, canonical_sha256
from training.scope_teacher import (ACTION_WITNESS_KINDS, CONTEXT_SCHEMA, FIXED_FACT_IDS, STATIC_FACTS,
    compile_request, mask_cross_context_conflicts, prepare_plan, scope_target_text, validate_context,
    materialize_http_payload, training_projection, validate_execution_authorization, validate_response)


def fixture(count=16, offset=0):
    content = "a" * 64
    frames = []
    for index in range(count):
        pts = offset + index * 5
        fid = source_frame_id(content, pts, 1, 10)
        frames.append({"frame_id": fid, "source_pts": pts, "pts_sec": pts / 10,
            "path": f"{content}/{fid}.jpg", "sha256": "b" * 64, "width": 464, "height": 848})
    context = {"schema_version": CONTEXT_SCHEMA, "context_id": "context-" + str(offset), "content_sha256": content,
        "split": "train", "group_id": "synthetic-group", "source_recording_ids": ["synthetic-recording"],
        "time_base": "1/10", "source_start_sec": 0.0, "start_sec": offset / 10,
        "end_sec": offset / 10 + (8 if count == 16 else count / 2), "frames": frames,
        "assignments_sha256": "c" * 64, "source_manifest_sha256": "d" * 64,
        "scope": "requested_frame_set", "data_origin": "organizer_development",
        "sampling_geometry": {"clip_seconds": 8, "stride_seconds": 4, "fps": 2, "max_frames": 16,
            "selection_frozen_before_decode": True, "sampler_config_sha256": "e" * 64}}
    axis = {"fact_ids": list(FIXED_FACT_IDS)}
    taxonomy = {"facts": [{"fact_id": fact, "question_ru": "Public physical question.",
        "definition_ru": "Public visual definition.", "absence_requires_ru": "Adequate relevant scope."}
        for fact in FIXED_FACT_IDS]}
    request = compile_request(context, axis, taxonomy)
    raw = {"scope": "requested_frame_set", "facts": [{"fact_id": fact, "state": "unclear",
        "supporting_frame_ids": [], "visibility_at_requested_frames": "unclear", "action_evidence": None}
        for fact in FIXED_FACT_IDS], "candidates": []}
    teacher = {"model_manifest_sha256": "1" * 64, "runtime_image_sha256": "2" * 64,
        "source_archive_sha256": "3" * 64, "server_ready_receipt_sha256": "4" * 64,
        "inference_location": "team_local_service", "external_ai_used": False, "base_only": True,
        "native_input_verified": True}
    return context, axis, taxonomy, request, raw, teacher


def set_fact(raw, fact, state, support, evidence=None, visibility="adequate"):
    row = next(item for item in raw["facts"] if item["fact_id"] == fact)
    row.update(state=state, supporting_frame_ids=list(support), visibility_at_requested_frames=visibility,
               action_evidence=evidence)


def record(values):
    context, _, _, request, raw, teacher = values
    return validate_response(json.dumps(raw), request, context, finish_reason="stop", teacher_provenance=teacher)


def witness(fact, aliases=("f00", "f01")):
    return {"witness_kind": sorted(ACTION_WITNESS_KINDS[fact])[0], "ordered_frame_ids": list(aliases),
        "direct_visual_sequence": True, "same_entity_or_cash_tracked": True, "required_regions_visible": True,
        "alternative_pose_or_occlusion_excluded": True}


def test_production_geometry_variable_short_cohort_and_no_label_input():
    context, axis, taxonomy, request, _, _ = fixture(3)
    assert len(request["frame_ledger"]) == 3
    assert request["generation"] == {"temperature": 0.0, "seed": 20261017, "max_tokens": 4096, "retries": 0}
    assert request["processor_limits"]["max_input_tokens"] == 8192
    assert request["processor_limits"]["maximum_request_tokens"] == 12288
    assert "synthetic-recording" not in request["system_prompt"]
    context["provided_video_label"] = True
    with pytest.raises(TrainingGateError, match="unapproved_fields"):
        compile_request(context, axis, taxonomy)


def test_source_pts_and_edge_gaps_are_preserved():
    values = fixture()
    coverage = validate_context(values[0])
    assert coverage["unobserved_intervals"][-1] == {"start_sec": 7.5, "end_sec": 8.0}
    assert coverage["continuous_coverage_established"] is False
    values[0]["frames"][1]["source_pts"] += 1
    with pytest.raises(TrainingGateError, match="frame_identity"):
        validate_context(values[0])


def test_fixed19_axis_and_response_no_candidates_or_legal_extra():
    values = fixture()
    values[1]["fact_ids"][0] = "OTHER_PERSON_ENTERS_WORKSPACE"
    with pytest.raises(TrainingGateError, match="fixed19_axis"):
        compile_request(values[0], values[1], values[2])
    values = fixture()
    values[4]["verdict"] = "violation"
    with pytest.raises(TrainingGateError, match="response_schema"):
        record(values)


def test_static_cohort_absence_requires_all_actual_requested_frames_and_no_window_absence():
    values = fixture(4)
    set_fact(values[4], "CASH_VISIBLE", "absent", ["f00", "f01", "f02"])
    with pytest.raises(TrainingGateError, match="all_requested_frames"):
        record(values)
    set_fact(values[4], "CASH_VISIBLE", "absent", ["f00", "f01", "f02", "f03"])
    output = record(values)
    assert output["sampled_fact_supervision_mask"]["CASH_VISIBLE"]
    assert not output["window_fact_supervision_mask"]["CASH_VISIBLE"]
    assert output["window_target"]["facts"][11]["state"] == "unclear"
    assert output["whole_window_absence_targets"] == 0


def test_procedural_context_and_sparse_action_absence_remain_masked():
    values = fixture()
    aliases = [f"f{index:02d}" for index in range(16)]
    for fact in ("WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE", "CASH_COUNT_VISIBLE"):
        set_fact(values[4], fact, "absent", aliases)
    output = record(values)
    assert all(not output["sampled_fact_supervision_mask"][fact] for fact in
               ("WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE", "CASH_COUNT_VISIBLE"))
    assert output["raw_teacher_facts"][2]["state"] == "absent"


@pytest.mark.parametrize("fact", ["CASH_HANDLER_LEAVES_WORKSTATION", "CASH_COUNT_VISIBLE"])
def test_observed_boundary_or_count_sequence_uses_specific_gate_without_invented_native_requirement(fact):
    values = fixture()
    set_fact(values[4], fact, "present", ["f00"])
    assert not record(values)["sampled_fact_supervision_mask"][fact]
    set_fact(values[4], fact, "present", ["f00", "f01"], witness(fact))
    assert record(values)["sampled_fact_supervision_mask"][fact]
    values[4]["facts"][FIXED_FACT_IDS.index(fact)]["action_evidence"]["alternative_pose_or_occlusion_excluded"] = False
    assert not record(values)["sampled_fact_supervision_mask"][fact]


@pytest.mark.parametrize("fact", ["CASH_HANDOVER_VISIBLE", "CASH_PLACED_IN_CLOTHING_POCKET", "CASH_PLACED_IN_PERSONAL_CONTAINER"])
def test_continuous_cash_transition_cannot_be_certified_by_sparse_frames_or_opaque_flags(fact):
    values = fixture()
    set_fact(values[4], fact, "present", ["f00", "f01"], witness(fact))
    assert not record(values)["sampled_fact_supervision_mask"][fact]
    frames = values[0]["frames"][:2]
    values[0]["native_interval_audits"] = [{"start_frame_id": frames[0]["frame_id"],
        "end_frame_id": frames[1]["frame_id"], "native_frame_ids": [frame["frame_id"] for frame in frames],
        "all_native_frames_presented": True, "source_pts_index_verified": True,
        "complete_native_frame_index_sha256": "a" * 64}]
    with pytest.raises(TrainingGateError, match="unapproved_fields"):
        compile_request(values[0], values[1], values[2])


def test_overlapping_static_absence_and_present_conflict_masked_without_raw_rewrite():
    left, right = fixture(8, 0), fixture(8, 5)
    set_fact(left[4], "CASH_VISIBLE", "absent", [f"f{index:02d}" for index in range(8)])
    set_fact(right[4], "CASH_VISIBLE", "present", ["f00"])
    originals = [record(left), record(right)]
    before = canonical_sha256(originals)
    masked, receipt = mask_cross_context_conflicts(originals)
    assert canonical_sha256(originals) == before
    assert receipt["overlapping_static_source_frame_conflict_pairs"] == 1
    assert all(not row["sampled_fact_supervision_mask"]["CASH_VISIBLE"] for row in masked)


def test_unknown_all_masked_no_training_and_incomplete_json_fails_closed():
    values = fixture()
    output = record(values)
    with pytest.raises(TrainingGateError, match="loss_all_masked"):
        scope_target_text(output, channel="sampled")
    with pytest.raises(TrainingGateError, match="incomplete_completion"):
        validate_response(json.dumps(values[4]), values[3], values[0], finish_reason="length", teacher_provenance=values[5])


def test_renderer_rechecks_raw_scope_cannot_enable_action_absence_or_context_by_edit():
    values = fixture()
    set_fact(values[4], "CASH_VISIBLE", "present", ["f00"])
    output = record(values)
    text, spans = scope_target_text(output)
    assert len(spans) == 1 and set(json.loads(text)) == {"facts", "candidates"}
    output["sampled_fact_supervision_mask"]["CASH_COUNT_VISIBLE"] = True
    output["sampled_target"]["facts"][2].update(state="absent", supporting_frame_ids=[frame["frame_id"] for frame in values[0]["frames"]])
    with pytest.raises(TrainingGateError, match="scope_promotion"):
        scope_target_text(output, channel="sampled")


def test_plan_is_bounded_train_only_and_preparation_never_authorizes():
    context, axis, taxonomy, *_ = fixture()
    plan = prepare_plan([context], axis, taxonomy, pins={"source": "a" * 64})
    assert plan["planned_requests"] == 1 and plan["maximum_requests"] == 61 and plan["model_calls_made"] == 0
    assert plan["execution_authorized_by_this_plan"] is False
    with pytest.raises(TrainingGateError, match="request_limit"):
        prepare_plan([deepcopy(context)] * 62, axis, taxonomy, pins={"source": "a" * 64})
    with pytest.raises(TrainingGateError, match="fresh_root_release"):
        validate_execution_authorization(plan, {}, {}, {}, {})


def test_execution_requires_fresh_exact_native_base_and_all_source_token_receipts():
    context, axis, taxonomy, *_ = fixture()
    plan = prepare_plan([context], axis, taxonomy, pins={"source": "a" * 64})
    runtime = {"schema_version": "forte.scope-teacher-native-readiness.v1", "base_only": True, "selected_adapter": None,
        "inference_location": "team_local_service", "external_ai_used": False, "native_input_verified": True,
        "http_generation_verified": True, "offline_boundary_verified": True, "model_files_verified": True,
        "max_model_len": 32768, "runtime_image_sha256": "b" * 64, "source_archive_sha256": "c" * 64,
        "model_manifest_sha256": "d" * 64}
    release = {"schema_version": "forte.scope-teacher-root-release.v1", "released_by_root": True,
        "plan_sha256": plan["plan_sha256"], "maximum_requests": 1, "maximum_retries": 0,
        "exclusive_idle_gpu_handoff_verified": True, "expires_at_utc": "2026-10-18T05:00:00Z",
        "maximum_execution_seconds": 3600, "input_pins": plan["input_pins"],
        **{key: runtime[key] for key in ("runtime_image_sha256", "source_archive_sha256", "model_manifest_sha256")}}
    reservation = {"schema_version": "forte.scope-teacher-reservation.v1", "status": "reserved",
        "plan_sha256": plan["plan_sha256"], "release_sha256": canonical_sha256(release), "central_role_gate_sha256": "e" * 64,
        "attempts": [{key: plan["requests"][0][key] for key in ("attempt_id", "content_sha256", "split", "request_sha256")}],
        "train_roles_verified": True, "fresh_central_exposure_gate_verified": True}
    audit = {"plan_sha256": plan["plan_sha256"], "actual_processor_exercised": True,
        "all_source_jpeg_hashes_verified": True, "no_truncation": True, "no_repeated_frame_pixels_created": True,
        "requests": [{"request_sha256": plan["requests"][0]["request_sha256"], "actual_input_tokens": 7000,
                      "declared_output_tokens": 4096, "maximum_actual_image_pixels": 262144}]}
    release.update(native_readiness_sha256=canonical_sha256(runtime), processor_audit_sha256=canonical_sha256(audit),
                   central_role_gate_sha256=reservation["central_role_gate_sha256"])
    reservation["release_sha256"] = canonical_sha256(release)
    now = datetime(2026, 10, 17, 20, 0, tzinfo=timezone.utc)
    assert validate_execution_authorization(plan, release, reservation, runtime, audit, now=now)["model_calls_made"] == 0
    audit["requests"][0]["actual_input_tokens"] = 8193
    with pytest.raises(TrainingGateError, match="receipt_not_bound"):
        validate_execution_authorization(plan, release, reservation, runtime, audit, now=now)


def test_default_production_projection_never_uses_cohort_absence_as_interval_absence():
    values = fixture(4)
    set_fact(values[4], "CASH_VISIBLE", "absent", ["f00", "f01", "f02", "f03"])
    set_fact(values[4], "CASH_HANDLER_PRESENT", "present", ["f00"])
    output = record(values)
    production, cohort = training_projection(output), training_projection(output, channel="sampled")
    assert not production["fact_supervision_mask"]["CASH_VISIBLE"]
    assert cohort["fact_supervision_mask"]["CASH_VISIBLE"]
    assert "AUXILIARY REQUESTED-FRAME-SET TASK" in cohort["prompt"]
    assert "AUXILIARY REQUESTED-FRAME-SET TASK" not in production["prompt"]
    assert production["production_interval_absence_targets"] == 0
    assert cohort["target"]["facts"][11]["supporting_frame_ids"] == ["f00", "f01", "f02", "f03"]
    assert cohort["frames"][0]["frame_id"] == "f00" and len(cohort["frames"][0]["canonical_frame_id"]) == 64


@pytest.mark.parametrize("mutation", ["duplicate", "reorder"])
def test_renderer_exact19_order_and_uniqueness_cannot_be_changed(mutation):
    values = fixture()
    set_fact(values[4], "CASH_VISIBLE", "present", ["f00"])
    output = record(values)
    if mutation == "duplicate": output["window_target"]["facts"][0] = deepcopy(output["window_target"]["facts"][1])
    else: output["window_target"]["facts"].reverse()
    with pytest.raises(TrainingGateError, match="fixed19"):
        scope_target_text(output)


def test_payload_materialization_is_explicit_authorized_hash_bound_and_never_sends():
    from training.guards import file_sha256
    context, axis, taxonomy, *_ = fixture(1)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frame = context["frames"][0]
        path = root / frame["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"\xff\xd8generated-test-only\xff\xd9")
        frame["sha256"] = file_sha256(path)
        request = compile_request(context, axis, taxonomy)
        with pytest.raises(TrainingGateError, match="not_authorized"):
            materialize_http_payload(context, request, model_id="private/base", media_root=root, authorization={})
        auth = {"status": "execution_prerequisites_verified", "authorized_request_sha256": [canonical_sha256(request)]}
        payload = materialize_http_payload(context, request, model_id="private/base", media_root=root, authorization=auth)
        assert payload["max_tokens"] == 4096 and payload["temperature"] == 0
        assert frame["path"] not in json.dumps(payload)
        path.write_bytes(b"changed")
        with pytest.raises(TrainingGateError, match="jpeg_digest"):
            materialize_http_payload(context, request, model_id="private/base", media_root=root, authorization=auth)
