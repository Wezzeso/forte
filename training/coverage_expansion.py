"""Bounded TRAIN-only coverage expansion planning and teacher execution guard.

Planning opens metadata only. Running requires a fresh root release and central
exposure receipt; this module never grants compute authority. Coarse labels
select contents/context and never supply physical fact or boundary targets.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import math
import time
import hashlib
from fractions import Fraction

from .guards import HEX64, TrainingGateError, canonical_sha256, file_sha256, load_annotations, load_json, require
from .teacher_sampling import (TeacherSamplingConfig, build_teacher_plan, plan_second_windows,
                               validate_teacher_plan)

SCHEMA = "forte.train-coverage-expansion.v1"
FIXED_STOP = datetime(2026, 10, 18, 6, 1, 25, tzinfo=timezone.utc).timestamp()


def _content(row: dict) -> str:
    return row.get("content_sha256", row.get("content_hash"))


def _two_bins(duration: float, anchors: list[float]) -> list[int]:
    count = math.ceil(duration)
    require(count >= 2, "coverage_source_too_short_for_two_windows")
    if anchors:
        require(all(type(value) in {int, float} and math.isfinite(value) and 0 <= value <= duration
                    for value in anchors), "coverage_coarse_anchor_invalid")
        first, last = min(count - 1, int(min(anchors))), min(count - 1, int(max(anchors)))
        if first == last:
            last = first + 2 if first + 2 < count else max(0, first - 2)
            if last == first:
                last = (first + 1) % count
        return [first, last]
    first, last = min(count - 1, int(duration / 4)), min(count - 1, int(duration * 3 / 4))
    return [first, last] if first != last else [first, (first + 1) % count]


def build_expansion(sources: list[dict], split: dict, labels: list[dict], anchors: list[dict],
                    previous_teachers: list[dict], *, input_pins: dict[str, str]) -> dict[str, Any]:
    """Freeze two context queries for each previously uncovered TRAIN content."""
    source_by_hash: dict[str, list[dict]] = {}
    for source in sources:
        source_by_hash.setdefault(_content(source), []).append(source)
    roles: dict[str, set[str]] = {}
    for row in split["records"]:
        roles.setdefault(_content(row), set()).add(row["split"])
    require(all(row.get("split") == "train" and type(row.get("provided_video_label")) is bool for row in labels),
            "coverage_coarse_labels_not_train_only")
    require(len({_content(row) for row in labels}) == len(labels), "coverage_coarse_content_duplicate")
    covered = {_content(row) for row in previous_teachers}
    require(all(row.get("split") == "train" and row.get("human_reviewed") is False for row in previous_teachers),
            "coverage_previous_teacher_scope_invalid")
    by_anchor = {_content(row): row for row in anchors}
    require(len(by_anchor) == len(anchors) and all(row.get("split") == "train" for row in anchors),
            "coverage_anchors_not_train_only")
    selected = [row for row in labels if _content(row) not in covered]
    require(len(selected) == 17 and sum(row["provided_video_label"] for row in selected) == 4,
            "coverage_expected_seventeen_content_inventory_changed")
    selected.sort(key=lambda row: _content(row))
    work_records, context_specs = [], {}
    role_counts = Counter()
    for label in selected:
        content = _content(label)
        require(roles.get(content) == {"train"} and content in source_by_hash, "coverage_partition_forbidden")
        source = sorted(source_by_hash[content], key=lambda row: row["recording_id"])[0]
        duration = source["duration_sec"]
        require(type(duration) in {int, float} and math.isfinite(duration) and duration > 0,
                "coverage_source_duration_invalid")
        anchor_row = by_anchor.get(content)
        require(anchor_row is not None and anchor_row.get("provided_video_label") is label["provided_video_label"],
                "coverage_coarse_anchor_binding_missing")
        points = list(anchor_row.get("coarse_anchor_sec", [])) if label["provided_video_label"] else []
        require(label["provided_video_label"] is False or bool(points), "coverage_positive_coarse_anchor_missing")
        bins = _two_bins(duration, points)
        frozen_rows = [row for row in split["records"] if _content(row) == content]
        frozen_groups = {row.get("scenario_group_id", row.get("group_id")) for row in frozen_rows}
        require(len(frozen_groups) == 1, "coverage_frozen_scenario_group_mismatch")
        group = next(iter(frozen_groups))
        work_records.append({"sample_id": "coverage-" + content[:24], "content_sha256": content,
            "scenario_group_id": group, "split": "train", "duration_sec": duration,
            "source_path": str(Path(content + ".mov"))})
        context_specs[content] = {"bins": bins, "points": points,
            "origin": "supplied_coarse_train_anchor_context_only" if points else "uniform_quarter_three_quarter",
            "source_presentation_origin_metadata": source.get("metadata", {}).get("start_time"),
            "source_stream_time_base": None}
        role_counts["provided_positive" if label["provided_video_label"] else "provided_negative"] += 1
    worklist = {"schema_version": "forte.autonomous-annotation-worklist.v1",
        "assignments_sha256": split["assignments_sha256"], "split_file_sha256": input_pins["automated_split_sha256"],
        "records": work_records}
    sampling = TeacherSamplingConfig(trusted_scene_context="bank_cash_workstation")
    plan = build_teacher_plan(worklist, split, split_sha256=input_pins["automated_split_sha256"],
        worklist_sha256=canonical_sha256(worklist), config=sampling)
    attempts = []
    for record in plan["records"]:
        spec = context_specs[record["content_sha256"]]
        record["priority_second_indices"] = spec["bins"]
        record["inspection_focus"] = {"origin": spec["origin"], "supplied_anchor_sec": spec["points"],
            "independent_localization": False, "organizer_labels_used_for_facts": False,
            "focal_exclusive_labels": False, "boundaries_known": False}
        windows = plan_second_windows(record["duration_sec"], sampling)
        for index in spec["bins"]:
            window = windows[index]
            attempts.append({"attempt_id": f"coverage-{record['content_sha256'][:24]}-{index:06d}",
                "content_sha256": record["content_sha256"], "split": "train", "second_index": index,
                "relative_focal_interval_sec": [window.focal_start_sec, window.focal_end_sec],
                "relative_context_interval_sec": [window.start_sec, window.end_sec],
                "requested_relative_timestamps_sec": list(window.timestamps_sec),
                "source_pts_resolved": False, "source_stream_time_base": None,
                "required_runtime_identity": "original_content_hash_plus_native_pts_time_base_and_jpeg_hashes"})
    plan["coverage_expansion"] = {"scope": "two_selected_contexts_per_uncovered_train_content",
        "expected_records": 17, "expected_attempts": 34, "full_clip_coverage_claimed": False,
        "input_pins": dict(input_pins), "native_pts_resolved_during_authorized_decode": True,
        "physical_targets": "independent_exact_anchor_questions_only", "whole_video_targets": 0,
        "boundary_targets": 0, "legal_targets": 0, "source_metadata_has_time_base": False}
    plan["plan_sha256"] = canonical_sha256({key: value for key, value in plan.items() if key != "plan_sha256"})
    validate_teacher_plan(plan, split, split_sha256=input_pins["automated_split_sha256"])
    return {"schema_version": SCHEMA, "teacher_plan": plan, "worklist": worklist,
        "attempt_plan": {"schema_version": "forte.qwen38-attempt-plan.v1", "phase": "teacher",
            "teacher_plan_sha256": plan["plan_sha256"], "maximum_attempts": 34, "attempts": attempts},
        "aggregate": {"records": 17, "planned_calls": 34, **dict(role_counts),
            "max_frames_per_call": sampling.max_frames, "facts_per_anchor": 8,
            "maximum_anchor_fact_answers": 272, "human_verified": False, "model_calls_made": False,
            "native_pts_resolved": False, "full_clip_coverage_claimed": False}}


def validate_guarded_execution(config: dict, plan: dict, release: dict, exposure: dict, *, now: float | None = None) -> None:
    """Pure metadata guard before the existing private teacher runner is entered."""
    now = time.time() if now is None else now
    require(config.get("schema_version") == SCHEMA and config.get("execute") is True, "coverage_execution_not_enabled")
    require(plan.get("plan_sha256") == canonical_sha256({key: value for key, value in plan.items() if key != "plan_sha256"}),
            "coverage_plan_digest_changed")
    require(len(plan.get("records", [])) == 17 and all(record.get("split") == "train"
        and len(record.get("priority_second_indices", [])) == 2 for record in plan["records"]),
        "coverage_exact_window_count_changed")
    require(config.get("max_records") == 17 and config.get("record_offset") == 0
            and config.get("max_model_requests") == 34 and config.get("anchor_focus_only") is True,
            "coverage_cli_bound_changed")
    require(release.get("schema_version") == "forte.teacher-execution-release.v1"
            and release.get("released") is True and release.get("released_by") == "root"
            and release.get("purpose") == "source_bound_teacher_annotation"
            and release.get("plan_sha256") == plan["plan_sha256"], "coverage_root_release_missing")
    require(release.get("max_model_requests") == 34 and release.get("max_records") == 17
            and release.get("record_offset") == 0 and release.get("anchor_focus_only") is True
            and release.get("model_manifest_sha256") == config.get("expected_model_manifest_sha256"),
            "coverage_release_scope_mismatch")
    require(release.get("attempt_plan_file_sha256") == config.get("expected_attempt_plan_sha256"),
            "coverage_attempt_plan_pin_changed")
    contract = {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "seed": 20261017, "temperature": 0.0,
        "max_tokens": 4096, "max_attempts": 1, "stream": False, "n": 1,
        "compact_frame_ids": True, "concise_observations": True, "chat_template_kwargs": {}}
    require(config.get("teacher_contract") == contract and release.get("teacher_contract_sha256") == canonical_sha256(contract),
            "coverage_teacher_generation_contract_changed")
    require(all(HEX64.fullmatch(str(config.get(key, ""))) is not None for key in
        ("expected_native_ledger_sha256", "expected_model_manifest_sha256", "producer_runtime_image_sha256", "producer_source_archive_sha256"))
        and release.get("producer_runtime_image_sha256") == config["producer_runtime_image_sha256"]
        and release.get("producer_source_archive_sha256") == config["producer_source_archive_sha256"], "coverage_runtime_pins_missing")
    require(type(config.get("max_runtime_sec")) is int and 0 < config["max_runtime_sec"] <= 1800
            and config["max_runtime_sec"] <= release.get("max_runtime_sec", 0)
            and now + config["max_runtime_sec"] + 120 < min(release.get("expires_at_unix", 0), FIXED_STOP),
            "coverage_stop_or_wall_bound_insufficient")
    require(release.get("exclusive_idle_gpu_handoff_verified") is True
            and release.get("fresh_role_exposure_verified") is True
            and release.get("native_pts_ledger_verified") is True
            and release.get("native_pts_ledger_sha256") == config.get("expected_native_ledger_sha256")
            and exposure.get("plan_sha256") == plan["plan_sha256"]
            and exposure.get("execution_release_sha256") == canonical_sha256(release)
            and exposure.get("maximum_attempts") == 34 and len(exposure.get("attempts", [])) == 34,
            "coverage_fresh_exposure_or_handoff_missing")
    gate = exposure.get("role_exposure_gate", {})
    require(gate.get("ok") is True and gate.get("requested_contents") == 17
            and gate.get("allowed_split_counts") == {"train": 17}, "coverage_central_role_gate_missing")
    require([(item["content_sha256"], item.get("second_index")) for item in exposure["attempts"]]
        == [(record["content_sha256"], index) for record in plan["records"] for index in record["priority_second_indices"]]
        and len({item.get("attempt_id") for item in exposure["attempts"]}) == 34,
        "coverage_reserved_attempt_order_mismatch")


def validate_native_packet(clip: Any, slot: dict) -> dict:
    """Compare the actual compact request to the staged native ledger pre-POST."""
    from forte.video.batches import source_frame_id
    require(slot.get("source_pts_resolved") is True and slot.get("split") == "train",
            "coverage_native_slot_unresolved")
    expected = slot["frames"]
    require(len(clip.frame_refs) == len(expected) and bool(expected), "coverage_native_frame_count_changed")
    interval = slot["relative_context_interval_sec"]
    require(clip.start_sec == interval[0] and clip.end_sec == interval[1], "coverage_native_context_changed")
    base = Fraction(slot["source_stream_time_base"])
    require(base > 0, "coverage_native_time_base_invalid")
    anchor_alias = None
    frame_ledger = []
    for index, (actual, frame) in enumerate(zip(clip.frame_refs, expected, strict=True)):
        alias = f"f{index:02d}"
        require(actual.frame_id == alias and actual.source_pts == frame["source_pts"]
            and actual.pts_sec == frame["pts_sec"] and actual.width == frame["width"]
            and actual.height == frame["height"] and actual.rotation_deg == frame["applied_rotation_deg"],
            "coverage_native_frame_metadata_changed")
        canonical = source_frame_id(slot["content_sha256"], actual.source_pts, base.numerator, base.denominator)
        require(canonical == frame["frame_id"] and frame["content_hash"] == slot["content_sha256"]
            and Fraction(frame["time_base"]) == base and frame["source_origin_pts"] == slot["source_origin_pts"]
            and abs(float((actual.source_pts - slot["source_origin_pts"]) * base) - actual.pts_sec) <= 1e-9,
            "coverage_native_frame_clock_changed")
        require(actual.jpeg_sha256 == frame["jpeg_sha256"]
            and file_sha256(Path(actual.path)) == frame["jpeg_sha256"], "coverage_native_jpeg_bytes_changed")
        if canonical == slot["anchor_frame_id"]:
            require(actual.source_pts == slot["anchor_source_pts"] and actual.pts_sec == slot["anchor_pts_sec"],
                    "coverage_native_anchor_clock_changed")
            anchor_alias = alias
        frame_ledger.append({"frame_id": canonical, "alias": alias, "source_pts": actual.source_pts,
            "pts_sec": actual.pts_sec, "jpeg_sha256": actual.jpeg_sha256})
    require(anchor_alias is not None, "coverage_native_anchor_not_in_request")
    from .teacher_sampling import ANCHOR_PHYSICAL_FACT_IDS
    questions = {item.fact_id: item.description for item in clip.fact_definitions if item.fact_id.startswith("ANCHOR__")}
    require(set(questions) == {"ANCHOR__" + fact for fact in ANCHOR_PHYSICAL_FACT_IDS},
            "coverage_native_anchor_questions_missing")
    for fact in ANCHOR_PHYSICAL_FACT_IDS:
        prefix = (f"INDEPENDENT EXACT-FRAME QUESTION: evaluate physical fact {fact} at frame {anchor_alias}, "
                  f"timestamp {slot['anchor_pts_sec']:.9f} seconds ONLY.")
        require(questions["ANCHOR__" + fact].startswith(prefix), "coverage_native_focal_question_changed")
    return {"attempt_id": slot["attempt_id"], "content_sha256": slot["content_sha256"],
        "second_index": slot["second_index"], "native_frames_sha256": canonical_sha256(frame_ledger),
        "anchor_frame_id": slot["anchor_frame_id"], "anchor_alias": anchor_alias,
        "native_packet_prepost_verified": True}


def native_bound_provider_class(provider_class: type, slots: list[dict], output: Path) -> type:
    """Scope a packet gate around the existing provider without core edits."""
    class NativeBoundProvider(provider_class):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.coverage_attempt_index = 0

        async def infer(self, clip: Any, *, prompt: str, response_schema: dict, deadline: float,
                        cancel_event: Any = None) -> Any:
            require(self.coverage_attempt_index < len(slots) == 34, "coverage_post_budget_exhausted")
            index = self.coverage_attempt_index
            packet = await asyncio.to_thread(validate_native_packet, clip, slots[index])
            packet.update(prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                response_schema_sha256=canonical_sha256(response_schema), status="attempted_before_provider_post")
            calls = output / "attempts"
            calls.mkdir(exist_ok=True)
            path = calls / f"{index:03d}.json"
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(packet, sort_keys=True, indent=2) + "\n")
            self.coverage_attempt_index += 1  # failures consume the reserved slot
            try:
                response = await super().infer(clip, prompt=prompt, response_schema=response_schema,
                    deadline=deadline, cancel_event=cancel_event)
            except BaseException:
                packet["status"] = "provider_failed"
                path.write_text(json.dumps(packet, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                raise
            packet["status"] = "provider_completed_pending_structured_validation"
            path.write_text(json.dumps(packet, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            return response
    return NativeBoundProvider


async def guarded_run(config_path: Path) -> dict:
    """Execute only under root's exact fresh release, without silent retries."""
    config = load_json(config_path)
    for key in ("teacher_plan", "attempt_plan", "split", "manifest", "source_manifest", "source_map", "native_ledger", "execution_release", "exposure_receipt"):
        require(file_sha256(Path(config[key])) == config.get("expected_" + key + "_sha256"), "coverage_runtime_file_pin_mismatch")
    plan, release, exposure = (load_json(Path(config[key])) for key in ("teacher_plan", "execution_release", "exposure_receipt"))
    validate_guarded_execution(config, plan, release, exposure)
    native = load_json(Path(config["native_ledger"]))
    require(native.get("schema_version") == "forte.native-resolved-teacher-attempt-plan.v1"
        and len(native.get("attempts", [])) == 34
        and [(slot["content_sha256"], slot["second_index"]) for slot in native["attempts"]]
        == [(row["content_sha256"], index) for row in plan["records"] for index in row["priority_second_indices"]],
        "coverage_native_plan_window_binding_changed")
    require(config["expected_manifest_sha256"] == config["expected_model_manifest_sha256"], "coverage_teacher_manifest_pin_mismatch")
    manifest = load_json(Path(config["manifest"]))
    require(manifest.get("model_id") == config["teacher_contract"]["model_id"]
        and manifest.get("revision") == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
        and manifest.get("processor_revision") == manifest["revision"]
        and manifest.get("selected_adapter") is None, "coverage_pinned_baseline_teacher_required")
    for path, digest in config.get("implementation_pins", {}).items():
        require(file_sha256(Path(path)) == digest, "coverage_implementation_source_changed")
    require(len(config.get("implementation_pins", {})) >= 6, "coverage_implementation_pins_missing")
    require(file_sha256(Path(__file__)) == release.get("coverage_guard_sha256"), "coverage_guard_source_changed")
    source_map = load_json(Path(config["source_map"]))
    require(set(source_map) == {record["content_sha256"] for record in plan["records"]}, "coverage_source_map_scope_changed")
    output = Path(config["output_root"])
    require(not output.exists(), "coverage_output_or_execution_already_exists")
    output.mkdir(parents=True)
    (output / "execution.claim.json").write_text(json.dumps({"config_sha256": file_sha256(config_path),
        "plan_sha256": plan["plan_sha256"], "release_sha256": canonical_sha256(release),
        "reserved_attempts": 34, "started_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    from .teacher_sampling import _run_cli
    args = SimpleNamespace(plan=Path(config["teacher_plan"]), split=Path(config["split"]),
        execution_release=Path(config["execution_release"]), source_root=Path(config["source_root"]),
        output_root=output, manifest=Path(config["manifest"]), endpoint=config["endpoint"],
        max_records=17, record_offset=0, max_model_requests=34, source_map=Path(config["source_map"]),
        anchor_focus_only=True, exposure_registry=Path(config["exposure_registry"]),
        exposure_receipt=Path(config["exposure_receipt"]), source_manifest=Path(config["source_manifest"]),
        max_runtime_sec=config["max_runtime_sec"])
    try:
        import forte.model as model_namespace
        from unittest.mock import patch
        original_provider = model_namespace.LocalModelProvider
        bound_provider = native_bound_provider_class(original_provider, native["attempts"], output)
        with patch.object(model_namespace, "LocalModelProvider", bound_provider):
            result = await _run_cli(args)
    except BaseException:
        (output / "execution.failed.json").write_text(json.dumps({"status": "failed",
            "reserved_attempts": 34, "automatic_replay_allowed": False,
            "requires_terminal_reconciliation": True}), encoding="utf-8")
        raise
    require(result.get("model_requests", 0) <= 34, "coverage_actual_attempts_exceeded")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Requires root release and all exact pins")
    args = parser.parse_args()
    try:
        require(args.execute, "coverage_metadata_preparation_only_use_python_api")
        result = asyncio.run(guarded_run(args.config))
        print(json.dumps(result, allow_nan=False))
        return 0
    except TrainingGateError as error:
        print(json.dumps({"status": "blocked", "reason_code": str(error)}))
        return 2
    except Exception as error:
        print(json.dumps({"status": "failed", "reason_code": "coverage_execution_failed", "error_type": type(error).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
