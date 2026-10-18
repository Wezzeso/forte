"""Generated metadata/pixels only: the optional teacher is never contacted."""

import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import pytest

from forte.shared.schemas import FrameRef, VideoAsset
from forte.video.decode import frame_identity
from training.guards import TrainingGateError, canonical_sha256
from training.teacher_sampling import (
    TeacherSamplingConfig, bind_teacher_requests, build_teacher_plan,
    extract_anchor_targets, with_anchor_questions,
    plan_second_windows, prioritize_train_anchors, run_teacher_record,
    validate_execution_release, validate_teacher_plan,
    validate_exposure_receipt,
)


def _digest_records(records):
    return hashlib.sha256((json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()).hexdigest()


def _inputs():
    records = [{"content_sha256": digit * 64, "split": partition, "group_id": "group-" + digit,
                "scenario_group_id": "scenario-" + digit}
               for digit, partition in [("a", "train"), ("b", "validation"), ("c", "final_check")]]
    split = {"records": records, "assignments_sha256": _digest_records(records)}
    worklist = {"schema_version": "forte.autonomous-annotation-worklist.v1",
                "split_file_sha256": "e" * 64, "assignments_sha256": split["assignments_sha256"],
                "records": [{"sample_id": "sample-" + row["content_sha256"][0], **row,
                             "source_path": "/synthetic/filename-that-must-not-enter-prompt.mov",
                             "duration_sec": 2.25, "has_violation": True,
                             "private_answer": "THIS_MUST_NEVER_BECOME_INPUT"} for row in records]}
    return worklist, split


def _plan(partition="train"):
    worklist, split = _inputs()
    plan = build_teacher_plan(worklist, split, split_sha256="e" * 64,
                              worklist_sha256="f" * 64, partition=partition)
    return plan, split


def _release(plan):
    return {"schema_version": "forte.teacher-execution-release.v1", "released": True,
            "plan_sha256": plan["plan_sha256"], "reference": "synthetic-no-gpu-owner",
            "purpose": "source_bound_teacher_annotation", "expires_at_unix": time.time() + 3600}


@pytest.mark.parametrize("duration", [.01, .25, 1, 8, 8.01, 142.533020833])
def test_each_second_including_partial_tail_is_planned(duration):
    import math
    windows = plan_second_windows(duration)
    assert len(windows) == math.ceil(duration)
    assert windows[0].focal_start_sec == 0
    assert windows[-1].focal_end_sec == duration
    assert windows[-1].timestamps_sec[-1] == duration
    assert all(len(window.timestamps_sec) <= 16 for window in windows)
    assert all(window.anchor_target_sec in window.timestamps_sec for window in windows)
    assert all(a.focal_end_sec == b.focal_start_sec for a, b in zip(windows, windows[1:]))


def test_budget_exhaustion_never_prunes_seconds_or_frame_targets():
    for config in (TeacherSamplingConfig(max_frames=4), TeacherSamplingConfig(max_seconds_per_record=1),
                   TeacherSamplingConfig(max_decode_frames=2)):
        with pytest.raises(TrainingGateError):
            plan_second_windows(8.25, config)


def test_default_worklist_projects_train_only_without_answer_fields():
    plan, split = _plan()
    assert [row["split"] for row in plan["records"]] == ["train"]
    assert "has_violation" not in json.dumps(plan)
    assert "THIS_MUST_NEVER_BECOME_INPUT" not in json.dumps(plan)
    validate_teacher_plan(plan, split, split_sha256="e" * 64)
    with pytest.raises(TrainingGateError):
        build_teacher_plan(*_inputs(), split_sha256="e" * 64, worklist_sha256="f" * 64, partition="final_check")


def test_validation_is_a_separate_plan_and_never_anchor_conditioned():
    plan, _ = _plan("validation")
    assert all(row["split"] == "validation" for row in plan["records"])
    with pytest.raises(TrainingGateError):
        prioritize_train_anchors(plan, {}, anchors_sha256="a" * 64)


def test_mutated_partition_group_hash_and_release_are_rejected():
    plan, split = _plan()
    tampered = copy.deepcopy(plan)
    tampered["records"][0]["split"] = "final_check"
    tampered["plan_sha256"] = canonical_sha256({key: value for key, value in tampered.items() if key != "plan_sha256"})
    with pytest.raises(TrainingGateError):
        validate_teacher_plan(tampered, split, split_sha256="e" * 64)
    with pytest.raises(TrainingGateError):
        validate_teacher_plan(plan, split, split_sha256="0" * 64)
    validate_execution_release(_release(plan), plan)
    for mutation in ({"released": False}, {"expires_at_unix": 0}, {"plan_sha256": "0" * 64}):
        with pytest.raises(TrainingGateError):
            validate_execution_release({**_release(plan), **mutation}, plan)


@pytest.mark.parametrize("mutation", [{"execution_release_sha256": "0" * 64},
    {"registered_content_set_sha256": "0" * 64}, {"registered_content_sha256": ["0" * 64]},
    {"source_manifest_sha256": "0" * 64}, {"allowed_splits": ["validation"]}])
def test_remote_receipt_is_bound_to_central_train_reservation_and_exact_release(mutation):
    plan, _ = _plan()
    release = _release(plan)
    contents = [record["content_sha256"] for record in plan["records"]]
    receipt = {"schema_version": "forte.teacher-exposure-receipt.v1",
        "registration_origin": "authoritative_central_exposure_registry", "ok": True,
        "plan_sha256": plan["plan_sha256"], "assignments_sha256": plan["assignments_sha256"],
        "split_file_sha256": plan["split_file_sha256"], "source_manifest_sha256": "d" * 64,
        "execution_release_reference": release["reference"], "execution_release_sha256": canonical_sha256(release),
        "registered_content_sha256": contents, "registered_content_set_sha256": canonical_sha256(sorted(contents)),
        "allowed_splits": ["train"], "registry_tip_sha256": "e" * 64}
    validate_exposure_receipt(receipt, plan, plan["records"], release, source_manifest_sha256="d" * 64)
    with pytest.raises(TrainingGateError):
        validate_exposure_receipt({**receipt, **mutation}, plan, plan["records"], release,
                                  source_manifest_sha256="d" * 64)


def test_supplied_train_anchor_prioritizes_only_schedule_not_questions():
    plan, split = _plan()
    anchors = {"split_file_sha256": "e" * 64, "assignments_sha256": split["assignments_sha256"],
               "records": [{"content_sha256": "a" * 64, "split": "train", "coarse_anchor_sec": [1.8],
                            "claimed_legal_answer": "DO_NOT_USE_THIS"}]}
    focused = prioritize_train_anchors(plan, anchors, anchors_sha256="1" * 64)
    assert focused["records"][0]["priority_second_indices"] == [1, 0, 2]
    assert focused["records"][0]["inspection_focus"]["independent_localization"] is False
    assert "claimed_legal_answer" not in json.dumps(focused)
    validate_teacher_plan(focused, split, split_sha256="e" * 64)


class _SyntheticProvider:
    def __init__(self):
        self.calls = []

    async def infer(self, request, *, prompt, response_schema, **kwargs):
        self.calls.append((request, prompt, response_schema))
        assert request.rule_definitions == []
        assert "filename-that-must-not-enter-prompt" not in prompt
        assert "THIS_MUST_NEVER_BECOME_INPUT" not in prompt
        assert response_schema["properties"]["candidates"]["maxItems"] == 0
        return SimpleNamespace(data={"facts": [{"fact_id": fact.fact_id, "state": "unclear",
            "supporting_frame_ids": [], "notes": None} for fact in request.fact_definitions], "candidates": []},
            metrics={}, finish_reason="stop")


@pytest.mark.asyncio
async def test_executable_bounded_teacher_one_decode_resumes_and_preserves_source_ids(monkeypatch, tmp_path):
    from forte.video import decode as decoder
    from forte.video import probe as prober
    source = tmp_path / "filename-that-must-not-enter-prompt.mov"
    source.write_bytes(b"synthetic-placeholder-decoder-is-injected")
    content = hashlib.sha256(source.read_bytes()).hexdigest()
    plan, _ = _plan()
    plan["records"][0].update(source_path=str(source), content_sha256=content, priority_second_indices=[1, 0, 2])
    plan["plan_sha256"] = canonical_sha256({key: value for key, value in plan.items() if key != "plan_sha256"})
    decode_calls = []

    def probe(path, *, job_id, **kwargs):
        return VideoAsset(job_id=job_id, owned_path=str(path), content_hash=content, duration_sec=2.25,
                          width=64, height=48, fps=4.0, time_base="1/4", source_start_sec=0.0)

    def decode(asset, output_dir, *, timestamps_sec, **kwargs):
        decode_calls.append(tuple(timestamps_sec))
        points = sorted({min(8, round(target * 4)) for target in timestamps_sec})
        return [FrameRef(frame_id=frame_identity(asset, point), pts_sec=float(Fraction(point, 4)),
                         source_pts=point, job_id=asset.job_id, width=64, height=48,
                         path=str(Path(output_dir) / f"{point}.jpg")) for point in points]

    monkeypatch.setattr(prober, "probe_video", probe)
    monkeypatch.setattr(decoder, "decode_frames", decode)
    provider = _SyntheticProvider()
    kwargs = dict(source_root=tmp_path, output_root=tmp_path / "private", provider=provider,
                  manifest={"model_id": "synthetic-test-only", "revision": "a" * 40, "processor_revision": "a" * 40},
                  manifest_sha256="b" * 64, release=_release(plan), deadline=time.monotonic() + 30)
    first = await run_teacher_record(plan["records"][0], plan, max_model_requests=1, **kwargs)
    assert first["partial_records"] == 1 and first["model_requests"] == 1
    assert len(decode_calls) == 1
    audit_path = next((tmp_path / "private").rglob("teacher.private.json"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["phase"] == "partial" and audit["coverage"]["complete"] is False
    assert audit["window_decisions"][0]["second_index"] == 1
    assert len(audit["window_decisions"][0]["facts"]) == 19
    assert audit["candidate_events"] == [] and audit["publication_ready"] is False
    assert audit["human_reviewed"] is False and audit["training_authorized"] is False
    assert audit["provenance"]["actual_snapshot_verified"] is False
    assert len(audit["window_decisions"][0]["frame_fact_targets"]) == 8
    assert all(target["derivation"] == "direct_anchor_teacher_question"
               and target["scope"] == "exact_anchor_frame" and target["human_verified"] is False
               for target in audit["window_decisions"][0]["frame_fact_targets"])
    second = await run_teacher_record(plan["records"][0], plan, max_model_requests=3, **kwargs)
    assert second["completed_records"] == 1 and second["resumed_seconds"] == 1
    assert second["model_requests"] == 2 and len(provider.calls) == 3
    assert len(decode_calls) == 2  # One union pass per attempt, never per second.
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["coverage"]["complete"] and audit["phase"] == "done"
    assert all(re.fullmatch(r"[a-f0-9]{64}", frame_id) for window in audit["window_decisions"]
               for frame_id in window["anchor_frame_ids"])


def test_anchor_target_does_not_copy_window_presence_or_broadcast_context():
    from forte.shared.schemas import ClipRequest, FactDecision, FactDefinition
    from forte.decision import DecisionBatch
    frames = [FrameRef(frame_id=f"source-{index}", pts_sec=float(index), path=f"synthetic-{index}.jpg",
                       width=64, height=48) for index in range(3)]
    request = ClipRequest(job_id="synthetic", clip_id="synthetic-window", frame_refs=frames,
                          start_sec=0.0, end_sec=2.0,
                          fact_definitions=[FactDefinition(fact_id="ANCHOR__" + key,
                              description="Synthetic direct anchor question") for key in
                              ("CASH_VISIBLE", "CASH_HANDOVER_VISIBLE", "CASH_HANDLER_PRESENT")])
    batch = DecisionBatch(request, facts=(
        FactDecision(fact_id="CASH_VISIBLE", state="present", supporting_frame_ids=["source-0"]),
        FactDecision(fact_id="ANCHOR__CASH_VISIBLE", state="unclear"),
        FactDecision(fact_id="ANCHOR__CASH_HANDOVER_VISIBLE", state="present", supporting_frame_ids=["source-0"]),
        FactDecision(fact_id="ANCHOR__CASH_HANDLER_PRESENT", state="present", supporting_frame_ids=["source-1"]),
    ), candidates=(), prompt_hash="a" * 64)
    targets = {row["fact_id"]: row for row in extract_anchor_targets(batch, frames[1])}
    assert targets["CASH_VISIBLE"]["state"] == "unclear"
    assert targets["CASH_HANDOVER_VISIBLE"]["eligible"] is False
    assert targets["CASH_HANDLER_PRESENT"]["eligible"] is True
    assert "PROCEDURE_CONTEXT_COMPLETE" not in targets
    assert all(row["frame_id"] == "source-1" for row in targets.values())
    assert all(row["provenance"]["focal_question_sha256"] == canonical_sha256(row["question_definition"])
               for row in targets.values())


def test_anchor_question_uses_authoritative_short_frame_alias():
    from forte.shared.schemas import ClipRequest
    from rules.catalog import load_catalog
    frames = [FrameRef(frame_id="a" * 64, pts_sec=0.0, path="synthetic.jpg", width=64, height=48),
              FrameRef(frame_id="b" * 64, pts_sec=0.5, path="synthetic.jpg", width=64, height=48)]
    request = ClipRequest(job_id="synthetic", clip_id="synthetic-window", frame_refs=frames,
                          fact_definitions=load_catalog().fact_definitions(), start_sec=0.0, end_sec=0.5)
    extended = with_anchor_questions(request, frames[1])
    assert len(extended.fact_definitions) == 27 and len(request.fact_definitions) == 19
    questions = [fact for fact in extended.fact_definitions if fact.fact_id.startswith("ANCHOR__")]
    assert len(questions) == 8
    assert all("frame f01" in fact.description and "only this visible frame" in fact.description for fact in questions)
    assert all(frames[1].frame_id not in fact.description for fact in questions)


@pytest.mark.asyncio
async def test_exposure_registry_failure_prevents_provider_construction(monkeypatch, tmp_path):
    from training import teacher_sampling
    from training.guards import file_sha256
    from scripts import exposure_registry
    from forte import model
    from forte.model import manifest

    worklist, split = _inputs()
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    worklist["split_file_sha256"] = file_sha256(split_path)
    plan = build_teacher_plan(worklist, split, split_sha256=file_sha256(split_path), worklist_sha256="f" * 64)
    plan_path, release_path = tmp_path / "plan.json", tmp_path / "release.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    release_path.write_text(json.dumps(_release(plan)), encoding="utf-8")
    constructed = []

    def forbidden_provider(*args, **kwargs):
        constructed.append(True)
        raise AssertionError("Provider constructed before successful source registration")

    def reject_registration(*args, **kwargs):
        raise ValueError("synthetic_exposure_registry_failed")

    monkeypatch.setattr(model, "LocalModelProvider", forbidden_provider)
    monkeypatch.setattr(manifest, "load_manifest", lambda path: {"served_model_name": "synthetic"})
    monkeypatch.setattr(exposure_registry, "register_content_manifest", reject_registration)
    args = SimpleNamespace(plan=plan_path, split=split_path, execution_release=release_path,
        manifest=tmp_path / "manifest.json", output_root=tmp_path / "private", max_records=1,
        max_model_requests=32, record_offset=0, anchor_focus_only=False,
        exposure_registry=tmp_path / "private/registry.jsonl", source_manifest=tmp_path / "manifest.jsonl")
    with pytest.raises(ValueError, match="synthetic_exposure_registry_failed"):
        await teacher_sampling._run_cli(args)
    assert constructed == []
