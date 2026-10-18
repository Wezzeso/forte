"""Generated-only tests for CPU native-clock preparation; no inference."""

import hashlib
import json
from pathlib import Path

import pytest

from forte.video.probe import probe_video
from training.guards import TrainingGateError, canonical_sha256, file_sha256
from training.prepare_native_ledgers import prepare
from training.teacher_sampling import TeacherSamplingConfig, build_teacher_plan, plan_second_windows
from test_video_batches import make_fixture


def _write(path, body):
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def inputs(tmp_path: Path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source = make_fixture(source_root / "generated.mov", b_frames=2,
                          timestamps=tuple(5000 + index * 40 for index in range(60)))
    content = hashlib.sha256(source.read_bytes()).hexdigest()
    source.rename(source_root / (content + ".mov"))
    source = source_root / (content + ".mov")
    asset = probe_video(source)
    records = [{"content_sha256": content, "split": "train", "scenario_group_id": "generated-group"}]
    assignments = hashlib.sha256((json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    split = {"records": records, "assignments_sha256": assignments}
    split_path = _write(tmp_path / "split.json", split)
    worklist = {"schema_version": "forte.autonomous-annotation-worklist.v1", "records": [
        {**records[0], "sample_id": "generated-source", "duration_sec": asset.duration_sec,
         "source_path": content + ".mov"}], "split_file_sha256": file_sha256(split_path),
        "assignments_sha256": assignments}
    plan = build_teacher_plan(worklist, split, split_sha256=file_sha256(split_path), worklist_sha256="a" * 64)
    plan["records"][0]["priority_second_indices"] = [0, 1]
    plan["plan_sha256"] = canonical_sha256({key: value for key, value in plan.items() if key != "plan_sha256"})
    windows = plan_second_windows(asset.duration_sec, TeacherSamplingConfig())
    attempts = [{"attempt_id": f"generated-{index}", "content_sha256": content,
                 "second_index": index, "split": "train", "requested_relative_timestamps_sec": list(windows[index].timestamps_sec)}
                for index in (0, 1)]
    plan_path = _write(tmp_path / "plan.json", plan)
    attempt_path = _write(tmp_path / "attempt.json", {"teacher_plan_sha256": plan["plan_sha256"], "attempts": attempts})
    return source_root, source, plan_path, attempt_path, split_path


def test_real_native_pts_and_jpeg_hashes_bind_two_generated_windows(tmp_path: Path):
    root, source, plan, attempts, split = inputs(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "private-output"
    result = prepare(plan, attempts, split, root, output, expected_plan_sha256=file_sha256(plan),
                     expected_attempt_sha256=file_sha256(attempts), expected_records=1, max_runtime_sec=30)
    assert result["records"] == 1 and result["windows"] == 2
    assert result["native_source_frames"] == 60 and result["model_calls"] == 0
    assert result["native_pts_resolved"] and result["source_bytes_unchanged"]
    assert source.read_bytes() == original
    resolved = json.loads((output / "resolved-attempt-plan.json").read_text())
    for row in resolved["attempts"]:
        assert row["source_pts_resolved"] and row["source_origin_pts"] > 0
        assert row["frames"] and len(row["frames"]) <= 16
        assert any(frame["frame_id"] == row["anchor_frame_id"] for frame in row["frames"])
        for frame in row["frames"]:
            assert frame["source_pts"] >= frame["source_origin_pts"]
            assert file_sha256(output / frame["jpeg_relative_path"]) == frame["jpeg_sha256"]
    receipt = json.loads((output / "receipt.private.json").read_text())
    assert canonical_sha256(receipt["output_files"]) == result["output_tree_sha256"]
    assert all(file_sha256(output / path) == digest for path, digest in receipt["output_files"].items())


def test_changed_plan_pin_stops_before_source_or_output_mutation(tmp_path: Path):
    root, source, plan, attempts, split = inputs(tmp_path)
    original = source.read_bytes()
    with pytest.raises(TrainingGateError, match="input_pin_changed"):
        prepare(plan, attempts, split, root, tmp_path / "forbidden", expected_plan_sha256="0" * 64,
                expected_attempt_sha256=file_sha256(attempts), expected_records=1)
    assert not (tmp_path / "forbidden").exists()
    assert source.read_bytes() == original
