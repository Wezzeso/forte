"""Metadata-only deterministic expansion and exact execution bounds."""
from copy import deepcopy
import hashlib
import json

import pytest

from training.coverage_expansion import build_expansion, validate_guarded_execution
from training.guards import TrainingGateError, canonical_sha256


def fixture():
    sources, labels, anchors, assignments = [], [], [], []
    for index in range(18):
        content = hashlib.sha256(str(index).encode()).hexdigest()
        positive = index < 4 or index == 17
        sources.append({"content_sha256": content, "recording_id": f"r{index}", "duration_sec": 20.5,
                        "metadata": {"start_time": 0}, "source_path": "never-opened"})
        labels.append({"content_sha256": content, "split": "train", "provided_video_label": positive})
        anchors.append({"content_sha256": content, "split": "train", "provided_video_label": positive,
                        "coarse_anchor_sec": [5.25] if positive else []})
        assignments.append({"content_hash": content, "split": "train", "scenario_group_id": f"g{index}"})
    split = {"records": assignments}
    split["assignments_sha256"] = hashlib.sha256((json.dumps(assignments, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    old = [{"content_sha256": sources[-1]["content_sha256"], "split": "train", "human_reviewed": False}]
    return sources, split, labels, anchors, old


def plan_fixture():
    return build_expansion(*fixture(), input_pins={"automated_split_sha256": "f" * 64})


def execution_fixture():
    result = plan_fixture()
    plan = result["teacher_plan"]
    contract = {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "seed": 20261017, "temperature": 0.0,
        "max_tokens": 4096, "max_attempts": 1, "stream": False, "n": 1,
        "compact_frame_ids": True, "concise_observations": True, "chat_template_kwargs": {}}
    config = {"schema_version": "forte.train-coverage-expansion.v1", "execute": True,
        "max_records": 17, "record_offset": 0, "max_model_requests": 34, "anchor_focus_only": True,
        "expected_model_manifest_sha256": "1" * 64, "expected_native_ledger_sha256": "2" * 64,
        "expected_attempt_plan_sha256": "3" * 64, "producer_runtime_image_sha256": "4" * 64,
        "producer_source_archive_sha256": "5" * 64, "max_runtime_sec": 30, "teacher_contract": contract}
    now = 1792230000.0
    release = {"schema_version": "forte.teacher-execution-release.v1", "released": True, "released_by": "root",
        "purpose": "source_bound_teacher_annotation", "plan_sha256": plan["plan_sha256"],
        "max_model_requests": 34, "max_records": 17, "record_offset": 0, "anchor_focus_only": True,
        "model_manifest_sha256": config["expected_model_manifest_sha256"], "max_runtime_sec": 60,
        "attempt_plan_file_sha256": config["expected_attempt_plan_sha256"],
        "teacher_contract_sha256": canonical_sha256(contract), "expires_at_unix": now + 600,
        "producer_runtime_image_sha256": config["producer_runtime_image_sha256"], "producer_source_archive_sha256": config["producer_source_archive_sha256"],
        "exclusive_idle_gpu_handoff_verified": True, "fresh_role_exposure_verified": True,
        "native_pts_ledger_verified": True, "native_pts_ledger_sha256": config["expected_native_ledger_sha256"]}
    exposure = {"plan_sha256": plan["plan_sha256"], "execution_release_sha256": canonical_sha256(release),
                "maximum_attempts": 34, "attempts": result["attempt_plan"]["attempts"],
                "role_exposure_gate": {"ok": True, "requested_contents": 17, "allowed_split_counts": {"train": 17}}}
    return config, plan, release, exposure, now


def test_exact_uncovered_train_roles_and_no_invented_pts_or_truth():
    result = plan_fixture()
    assert result["aggregate"]["records"] == 17
    assert result["aggregate"]["provided_positive"] == 4
    assert result["aggregate"]["provided_negative"] == 13
    assert len(result["attempt_plan"]["attempts"]) == 34
    assert all(item["source_pts_resolved"] is False and item["source_stream_time_base"] is None
               for item in result["attempt_plan"]["attempts"])
    assert all(len(row["priority_second_indices"]) == 2 for row in result["teacher_plan"]["records"])
    assert result == plan_fixture()


def test_labels_and_source_paths_do_not_become_request_targets():
    plan = plan_fixture()["teacher_plan"]
    assert "provided_video_label" not in json.dumps(plan)
    assert "never-opened" not in json.dumps(plan)
    assert plan["coverage_expansion"]["whole_video_targets"] == 0
    assert plan["coverage_expansion"]["boundary_targets"] == 0
    assert plan["coverage_expansion"]["full_clip_coverage_claimed"] is False


def test_negative_queries_are_fixed_quartiles_and_positive_point_selects_context_only():
    result = plan_fixture()
    records = result["teacher_plan"]["records"]
    positives = [row for row in records if row["inspection_focus"]["supplied_anchor_sec"]]
    negatives = [row for row in records if not row["inspection_focus"]["supplied_anchor_sec"]]
    assert all(row["priority_second_indices"] == [5, 7] for row in positives)
    assert all(row["priority_second_indices"] == [5, 15] for row in negatives)
    assert all(row["inspection_focus"]["organizer_labels_used_for_facts"] is False for row in records)


def test_final_check_or_alias_role_conflict_blocks_plan():
    values = list(fixture())
    values[1]["records"].append({**values[1]["records"][0], "split": "final_check"})
    with pytest.raises(TrainingGateError, match="partition_forbidden"):
        build_expansion(*values, input_pins={"automated_split_sha256": "f" * 64})


def test_exact_released_metadata_passes_without_execution():
    config, plan, release, exposure, now = execution_fixture()
    validate_guarded_execution(config, plan, release, exposure, now=now)


@pytest.mark.parametrize("key,value", [("max_model_requests", 35), ("max_records", 18),
    ("record_offset", 1), ("anchor_focus_only", False), ("execute", False)])
def test_cli_cannot_escape_declared_thirtyfour_attempts(key, value):
    config, plan, release, exposure, now = execution_fixture()
    config[key] = value
    with pytest.raises(TrainingGateError):
        validate_guarded_execution(config, plan, release, exposure, now=now)


@pytest.mark.parametrize("key,value", [("native_pts_ledger_verified", False),
    ("fresh_role_exposure_verified", False), ("exclusive_idle_gpu_handoff_verified", False),
    ("max_model_requests", 36), ("expires_at_unix", 0)])
def test_ledger_handoff_and_stop_limits_fail_closed(key, value):
    config, plan, release, exposure, now = execution_fixture()
    release[key] = value
    exposure["execution_release_sha256"] = canonical_sha256(release)
    with pytest.raises(TrainingGateError):
        validate_guarded_execution(config, plan, release, exposure, now=now)


def test_same_content_wrong_window_or_duplicate_attempt_id_rejected():
    for mutation in ("window", "identity"):
        config, plan, release, exposure, now = execution_fixture()
        if mutation == "window":
            exposure["attempts"][0]["second_index"] = -1
        else:
            exposure["attempts"][1]["attempt_id"] = exposure["attempts"][0]["attempt_id"]
        with pytest.raises(TrainingGateError, match="reserved_attempt_order_mismatch"):
            validate_guarded_execution(config, plan, release, exposure, now=now)


def native_fixture(tmp_path):
    from types import SimpleNamespace
    from training.guards import file_sha256
    from forte.video.batches import source_frame_id
    from training.teacher_sampling import ANCHOR_PHYSICAL_FACT_IDS
    content = "a" * 64
    path = tmp_path / "synthetic.jpg"
    path.write_bytes(b"synthetic-original-jpeg-byte-identity")
    digest = file_sha256(path)
    canonical = source_frame_id(content, 10, 1, 10)
    actual = SimpleNamespace(frame_id="f00", source_pts=10, pts_sec=1.0, width=4, height=8,
                             rotation_deg=0, jpeg_sha256=digest, path=str(path))
    frames = [{"frame_id": canonical, "source_pts": 10, "pts_sec": 1.0, "time_base": "1/10",
               "source_origin_pts": 0, "width": 4, "height": 8, "applied_rotation_deg": 0,
               "content_hash": content, "jpeg_sha256": digest}]
    slot = {"attempt_id": "generated-slot", "content_sha256": content, "split": "train", "second_index": 1,
        "source_pts_resolved": True, "source_stream_time_base": "1/10", "source_origin_pts": 0,
        "relative_context_interval_sec": [0.0, 2.0], "anchor_frame_id": canonical,
        "anchor_source_pts": 10, "anchor_pts_sec": 1.0, "frames": frames}
    questions = [SimpleNamespace(fact_id="ANCHOR__" + fact, description=(
        f"INDEPENDENT EXACT-FRAME QUESTION: evaluate physical fact {fact} at frame f00, "
        "timestamp 1.000000000 seconds ONLY. Frozen criterion.")) for fact in ANCHOR_PHYSICAL_FACT_IDS]
    clip = SimpleNamespace(frame_refs=[actual], start_sec=0.0, end_sec=2.0, fact_definitions=questions)
    return clip, slot


def test_actual_prepost_packet_matches_native_clock_jpeg_and_focal_question(tmp_path):
    from training.coverage_expansion import validate_native_packet
    clip, slot = native_fixture(tmp_path)
    result = validate_native_packet(clip, slot)
    assert result["native_packet_prepost_verified"] is True
    assert result["anchor_alias"] == "f00"


@pytest.mark.parametrize("field,value", [("source_pts", 11), ("pts_sec", 1.1), ("width", 5),
                                       ("rotation_deg", 90), ("frame_id", "f01")])
def test_changed_packet_metadata_cannot_reach_provider(tmp_path, field, value):
    from training.coverage_expansion import validate_native_packet
    clip, slot = native_fixture(tmp_path)
    setattr(clip.frame_refs[0], field, value)
    with pytest.raises(TrainingGateError, match="native_frame_metadata_changed"):
        validate_native_packet(clip, slot)


def test_changed_original_jpeg_or_focal_question_is_rejected(tmp_path):
    from training.coverage_expansion import validate_native_packet
    clip, slot = native_fixture(tmp_path)
    clip.fact_definitions[0].description = "Wrong focal alias/time"
    with pytest.raises(TrainingGateError, match="focal_question_changed"):
        validate_native_packet(clip, slot)
    clip, slot = native_fixture(tmp_path)
    from pathlib import Path
    Path(clip.frame_refs[0].path).write_bytes(b"changed-original-bytes")
    with pytest.raises(TrainingGateError, match="native_jpeg_bytes_changed"):
        validate_native_packet(clip, slot)


@pytest.mark.asyncio
async def test_proxy_checks_before_fake_provider_and_consumes_failed_call(tmp_path):
    from training.coverage_expansion import native_bound_provider_class
    class FakeProvider:
        def __init__(self):
            self.calls = 0
        async def infer(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("synthetic-provider-failure")
    clip, slot = native_fixture(tmp_path)
    provider = native_bound_provider_class(FakeProvider, [slot] * 34, tmp_path)()
    with pytest.raises(RuntimeError, match="synthetic-provider-failure"):
        await provider.infer(clip, prompt="synthetic", response_schema={}, deadline=10)
    assert provider.calls == provider.coverage_attempt_index == 1
    assert json.loads((tmp_path / "attempts/000.json").read_text())["status"] == "provider_failed"
    clip.frame_refs[0].source_pts = 11
    with pytest.raises(TrainingGateError):
        await provider.infer(clip, prompt="synthetic", response_schema={}, deadline=10)
    assert provider.calls == provider.coverage_attempt_index == 1
