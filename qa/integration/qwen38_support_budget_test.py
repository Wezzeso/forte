"""Independent generated composition of support-budget, balance, alias and masks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from training.guards import TrainingGateError, canonical_sha256, file_sha256
from training.qwen38_silver import (BALANCED_LOSS, COMPACT_FRAMES, MASK_WHOLE_SUPPORT_FACT,
    STRICT_SUPPORT_BUDGET, apply_supervision_policy, build_silver_rows, collect_release_bindings)
from training.test_qwen38_silver import fixtures
from forte.video.batches import source_frame_id


def composed_fixture():
    config, base, _, _, taxonomy, _ = fixtures()
    teachers, assignments, sources, media = [], [], [], {}
    for index in range(6):
        content = str(index + 1) * 64
        record = deepcopy(base[0])
        recording = f"generated-source-{index}"
        record.update(content_sha256=content, recording_ids=[recording], group_id=f"generated-group-{index}")
        frames = []
        for offset in range(9 if index == 0 else 4):
            fid = source_frame_id(content, offset * 10, 1, 10)
            frame = {"frame_id": fid, "source_pts": offset * 10, "pts_sec": float(offset),
                "jpeg_sha256": "a" * 64, "path": "unused-private-filename-sentinel.jpg"}
            frames.append(frame)
            media[(content, fid)] = {"content_sha256": content, "frame_id": fid, "source_pts": offset * 10,
                "pts_sec": float(offset), "sha256": frame["jpeg_sha256"], "path": f"{content}/{fid}.jpg", "split": "train"}
        record["frames"] = frames
        context = [frame["frame_id"] for frame in frames]
        window = record["windows"][0]
        window.update(window_id=f"generated-window-{index}", frame_ids=context)
        for fact_index, (target, review) in enumerate(zip(window["frame_fact_targets"], window["support_reviews"])):
            state = ("present" if index < 3 else "absent") if fact_index == 0 else ("present" if index % 2 == 0 else "absent")
            supports = context.copy() if index == fact_index == 0 else [context[1]]
            target.update(frame_id=context[1], state=state, pts_sec=1.0,
                context_frame_ids=context.copy(), supporting_frame_ids=supports.copy())
            review.update(frame_id=context[1], state=state, teacher_state=state, supports_state=True, pts_sec=1.0,
                context_frame_ids=context.copy(), supporting_frame_ids=supports.copy())
        teachers.append(record)
        assignments.append({"content_hash": content, "recording_id": recording,
                            "scenario_group_id": record["group_id"], "split": "train"})
        sources.append({"content_sha256": content, "recording_id": recording})
    return config, teachers, {"records": assignments}, sources, taxonomy, media


def derive_pinned(tmp_path):
    values = composed_fixture()
    raw_hash = canonical_sha256(values[1])
    with pytest.raises(TrainingGateError, match="support_exceeds_frame_budget"):
        build_silver_rows(*values)
    explicit = deepcopy(values)
    explicit[0]["support_budget_policy"] = STRICT_SUPPORT_BUDGET
    with pytest.raises(TrainingGateError, match="support_exceeds_frame_budget"):
        build_silver_rows(*explicit)
    config = values[0]
    config.update(support_budget_policy=MASK_WHOLE_SUPPORT_FACT, silver_loss_policy=BALANCED_LOSS,
        silver_balance_minimum_class_contents=2, frame_alias_policy=COMPACT_FRAMES)
    canonical, coverage = build_silver_rows(*values)
    assert canonical_sha256(values[1]) == raw_hash
    effective, report = apply_supervision_policy(config, canonical, require_pins=False)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(report["balance_plan"], sort_keys=True))
    config.update(silver_balance_plan=str(plan), expected_silver_balance_plan_sha256=file_sha256(plan),
        expected_silver_balance_plan_semantic_sha256=report["balance_plan_semantic_sha256"],
        expected_effective_rows_sha256=report["effective_rows_sha256"],
        expected_effective_fact_weights_sha256=report["effective_fact_weights_sha256"])
    checked, checked_report = apply_supervision_policy(config, canonical)
    assert checked == effective and checked_report["effective_rows_sha256"] == report["effective_rows_sha256"]
    return values, canonical, coverage, effective, checked_report


def test_whole_fact_mask_composes_with_pinned_balance_and_aliases(tmp_path):
    values, canonical, coverage, effective, report = derive_pinned(tmp_path)
    assert len(canonical) == len(effective) == 6
    assert coverage["support_budget_excluded_fact_count"] == 1
    first = effective[0]
    assert first["fact_supervision_mask"] == {"CASH_VISIBLE": False, "CASH_HANDOVER_VISIBLE": True}
    assert first["fact_loss_weights"]["CASH_VISIBLE"] == 0
    assert first["fact_loss_weights"]["CASH_HANDOVER_VISIBLE"] > 0
    assert first["target"]["facts"][0] == {"fact_id": "CASH_VISIBLE", "state": "unclear", "supporting_frame_ids": []}
    assert len(first["frames"]) == 4
    assert [frame["frame_id"] for frame in first["frames"]] == ["f00", "f01", "f02", "f03"]
    exclusion = first["support_budget_exclusions"][0]
    original = values[1][0]["windows"][0]["frame_fact_targets"][0]
    assert exclusion["original_supporting_frame_ids"] == original["supporting_frame_ids"]
    assert exclusion["original_support_count"] == 9 and exclusion["source_or_citations_modified"] is False
    assert first["raw_teacher_window_sha256"] == canonical_sha256(values[1][0]["windows"][0])
    assert all(fid not in first["prompt"] for fid in original["supporting_frame_ids"])
    assert "private-filename" not in first["prompt"]
    assert report["objective"]["support_budget_policy"] == MASK_WHOLE_SUPPORT_FACT
    model, runtime = tmp_path / "base.json", tmp_path / "runtime.json"
    model.write_text("{}"); runtime.write_text("{}")
    config = {**values[0], "model_manifest": str(model), "runtime_lock": str(runtime),
              "expected_runtime_lock_sha256": file_sha256(runtime), "quantization": "none"}
    bindings = collect_release_bindings(config, {}, {"coverage": coverage, "supervision": report})
    assert bindings["support_budget_exclusions_sha256"] == coverage["support_budget_exclusions_sha256"]
    assert bindings["effective_rows_sha256"] == canonical_sha256(effective)
    assert bindings["supervision_objective_sha256"] == report["supervision_objective_sha256"]
    for pin in ("expected_effective_rows_sha256", "expected_effective_fact_weights_sha256"):
        with pytest.raises(TrainingGateError, match="effective_input_pin"):
            apply_supervision_policy({**values[0], pin: "0" * 64}, canonical)


def test_masked_nine_frame_fact_has_no_actual_processor_tokens_or_weight(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from PIL import Image
    from training.data import SilverCompletionCollator
    from training.silver_balance import fact_token_loss_weights
    _, _, _, effective, _ = derive_pinned(tmp_path)
    row = effective[0]
    for index, frame in enumerate(row["frames"]):
        path = tmp_path / frame["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (index * 30, 80, 140)).save(path, "JPEG")
    snapshot = Path(__file__).resolve().parents[2] / "configs/serving/qwen38/public-snapshot"
    processor = transformers.AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    collator = SilverCompletionCollator(processor, tmp_path, max_length=8192, max_pixels=4096, model_type="qwen3_5")
    batch, positions = collator.collate_with_fact_spans([row])
    assert set(positions) == {"CASH_HANDOVER_VISIBLE"}
    labels = batch["labels"][0].tolist()
    text = processor.tokenizer.decode([label for label in labels if label != -100])
    assert "CASH_VISIBLE" not in text and "unclear" not in text and "CASH_HANDOVER_VISIBLE" in text
    weights = fact_token_loss_weights(labels, positions, row["fact_loss_weights"])
    assert sum(weights) == pytest.approx(row["fact_loss_weights"]["CASH_HANDOVER_VISIBLE"])
    assert all(weight == 0 for weight, label in zip(weights, labels) if label == -100)
