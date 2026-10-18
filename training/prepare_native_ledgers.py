"""Prepare exact TRAIN source clocks/window JPEG identities without inference.

Only a fresh private output directory is written. Original videos are opened
read-only. No provider, model, torch, network request or pixel logging exists.
The separate execution release remains necessary for every later model call.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

from training.guards import canonical_sha256, file_sha256, load_json, require
from training.teacher_sampling import (
    TeacherSamplingConfig, bind_teacher_requests, plan_second_windows, validate_teacher_plan,
)


SCHEMA = "forte.native-teacher-preparation.v1"


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    path.chmod(0o600)


def _stat(path: Path) -> tuple[int, int, int]:
    observed = path.stat()
    return observed.st_size, observed.st_mtime_ns, observed.st_ino


def collect_native_ledger(asset, path: Path, *, deadline: float) -> tuple[list, dict]:
    """Decode every original frame for metadata only; no RGB/JPEG conversion."""
    import av
    from forte.video.batches import _configuration, _identity, _open_stream, _pairs, _Progress
    from forte.video.probe import operation_slot

    config = _configuration(batch_size=1, target_size=(1, 1), max_batch_bytes=3,
        max_source_frames=100_000, max_output_frames=100_000, thread_count=2,
        deadline_monotonic=deadline, max_runtime_sec=max(.001, deadline - time.monotonic()))
    progress = _Progress()
    rows = []
    with operation_slot(deadline, None):
        with av.open(asset.owned_path, "r", options={"protocol_whitelist": "file,pipe"}) as container:
            stream, origin, base = _open_stream(container, asset, config)
            for frame, next_pts in _pairs(container, stream, asset, config, progress, None):
                source = _identity(frame, next_pts, asset, origin, base)
                require(rows or source.source_pts == origin, "native_preparation_start_missing")
                rows.append(source)
    require(bool(rows) and rows[-1].next_source_pts is None, "native_preparation_tail_missing")
    require(asset.frame_count is None or len(rows) == asset.frame_count, "native_preparation_count_mismatch")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    path.chmod(0o600)
    return rows, {"native_frame_count": len(rows), "ledger_file_sha256": file_sha256(path),
                  "ledger_semantic_sha256": canonical_sha256([asdict(row) for row in rows])}


def select_anchor(record: dict, window, request, duration: float):
    """Mirror the teacher runner's exact native focal-anchor selection."""
    eligible = [frame for frame in request.frame_refs
                if window.focal_start_sec <= frame.pts_sec < window.focal_end_sec
                or window.focal_end_sec == duration and frame.pts_sec == window.focal_end_sec]
    require(bool(eligible), "native_preparation_focal_anchor_missing")
    chosen = min(eligible, key=lambda frame: abs(frame.pts_sec - window.anchor_target_sec))
    points = [point for point in record.get("inspection_focus", {}).get("supplied_anchor_sec", [])
              if window.focal_start_sec <= point < window.focal_end_sec
              or point == window.focal_end_sec == duration]
    origin = "native_second_midpoint"
    if points:
        chosen = min(eligible, key=lambda frame: min(abs(frame.pts_sec - point) for point in points))
        origin = "nearest_native_frame_to_supplied_coarse_train_anchor"
    return chosen, origin


def prepare(plan_path: Path, attempt_path: Path, split_path: Path, source_root: Path, output: Path,
            *, expected_plan_sha256: str, expected_attempt_sha256: str,
            expected_records: int = 17, max_runtime_sec: float = 900,
            runtime_image_sha256: str | None = None, host_source_root: str | None = None) -> dict:
    from forte.video.decode import decode_frames, validate_frame_ref
    from forte.video.probe import check_cancelled, content_sha256, probe_video

    started = time.monotonic()
    deadline = started + max_runtime_sec
    require(math.isfinite(max_runtime_sec) and 0 < max_runtime_sec <= 1800, "native_preparation_runtime_invalid")
    require(file_sha256(plan_path) == expected_plan_sha256
            and file_sha256(attempt_path) == expected_attempt_sha256, "native_preparation_input_pin_changed")
    plan, attempt_plan, split = load_json(plan_path), load_json(attempt_path), load_json(split_path)
    validate_teacher_plan(plan, split, split_sha256=file_sha256(split_path))
    require(plan["partition"] == "train" and len(plan["records"]) == expected_records,
            "native_preparation_train_scope_changed")
    require(attempt_plan.get("teacher_plan_sha256") == plan["plan_sha256"]
            and len(attempt_plan.get("attempts", [])) == expected_records * 2,
            "native_preparation_attempt_scope_changed")
    attempts = {(item["content_sha256"], item["second_index"]): item for item in attempt_plan["attempts"]}
    require(len(attempts) == expected_records * 2 and all(item.get("split") == "train"
            for item in attempts.values()), "native_preparation_attempts_duplicate_or_forbidden")
    for record in plan["records"]:
        indices = record.get("priority_second_indices", [])
        require(record["split"] == "train" and len(indices) == len(set(indices)) == 2
                and all((record["content_sha256"], index) in attempts for index in indices),
                "native_preparation_selected_windows_changed")
    source_root = source_root.resolve(strict=True)
    require(source_root.is_dir() and not output.exists(), "native_preparation_output_or_source_invalid")
    output.mkdir(parents=True, mode=0o700)
    sampling = TeacherSamplingConfig(**plan["sampling"])
    pins = {"teacher_plan_file_sha256": expected_plan_sha256, "teacher_plan_sha256": plan["plan_sha256"],
        "attempt_plan_file_sha256": expected_attempt_sha256, "split_file_sha256": file_sha256(split_path),
        "utility_sha256": file_sha256(Path(__file__)), "teacher_sampling_sha256": file_sha256(
            Path(__file__).with_name("teacher_sampling.py"))}
    _write(output / "preparation.claim.json", {"schema_version": SCHEMA, "status": "running", "pins": pins,
        "started_at": datetime.now(timezone.utc).isoformat(), "model_calls": 0, "gpu_used": False})
    resolved, sources, source_map = [], [], {}
    total_native = total_jpegs = 0
    try:
        for record in plan["records"]:
            check_cancelled(deadline, None)
            content = record["content_sha256"]
            source = (source_root / (content + ".mov")).resolve(strict=True)
            require(source.is_relative_to(source_root) and source.is_file(), "native_preparation_source_escaped")
            initial_stat = _stat(source)
            job_id = "teacher-" + canonical_sha256({"content": content, "plan": plan["plan_sha256"],
                                                    "anchor_focus_only": True})[:24]
            asset = probe_video(source, job_id=job_id, deadline_monotonic=deadline)
            require(asset.content_hash == content, "native_preparation_source_hash_changed")
            require(abs(asset.duration_sec - record["duration_sec"]) <= max(.1, 1 / (asset.fps or 30)),
                    "native_preparation_source_duration_changed")
            ledger_path = output / "native" / (content + ".jsonl")
            ledger, ledger_info = collect_native_ledger(asset, ledger_path, deadline=deadline)
            index = {frame.frame_id: frame for frame in ledger}
            windows = plan_second_windows(asset.duration_sec, sampling)
            selected = [windows[position] for position in record["priority_second_indices"]]
            targets = sorted({point for window in selected for point in window.timestamps_sec})
            refs = decode_frames(asset, output / "jpeg" / content, timestamps_sec=targets,
                max_frames=sampling.max_decode_frames, deadline_monotonic=deadline,
                max_runtime_sec=max(.001, deadline - time.monotonic()), thread_count=2)
            for ref in refs:
                validate_frame_ref(asset, ref)
                require(ref.frame_id in index and ref.source_pts == index[ref.frame_id].source_pts,
                        "native_preparation_jpeg_not_in_native_ledger")
                require(file_sha256(Path(ref.path)) == ref.jpeg_sha256, "native_preparation_jpeg_hash_changed")
                Path(ref.path).chmod(0o600)
            requests = bind_teacher_requests(job_id, refs, selected, [])
            for window, request in zip(selected, requests, strict=True):
                attempt = attempts[(content, window.index)]
                anchor, anchor_selection = select_anchor(record, window, request, asset.duration_sec)
                frame_rows = [{**asdict(index[ref.frame_id]), "jpeg_sha256": ref.jpeg_sha256,
                    "jpeg_relative_path": Path(ref.path).relative_to(output).as_posix()}
                    for ref in request.frame_refs]
                require(len(frame_rows) <= sampling.max_frames, "native_preparation_window_frame_limit")
                resolved.append({"attempt_id": attempt["attempt_id"], "content_sha256": content, "split": "train",
                    "second_index": window.index, "source_pts_resolved": True, "source_stream_time_base": asset.time_base,
                    "source_origin_pts": ledger[0].source_origin_pts, "native_ledger_sha256": ledger_info["ledger_file_sha256"],
                    "relative_focal_interval_sec": [window.focal_start_sec, window.focal_end_sec],
                    "relative_context_interval_sec": [window.start_sec, window.end_sec],
                    "planned_requested_relative_timestamps_sec": attempt["requested_relative_timestamps_sec"],
                    "resolved_requested_relative_timestamps_sec": list(window.timestamps_sec),
                    "anchor_frame_id": anchor.frame_id, "anchor_source_pts": anchor.source_pts,
                    "anchor_pts_sec": anchor.pts_sec, "anchor_selection": anchor_selection,
                    "frames": frame_rows, "selected_frames_sha256": canonical_sha256(frame_rows),
                    "full_clip_coverage_claimed": False})
            require(_stat(source) == initial_stat
                    and content_sha256(source, deadline_monotonic=deadline) == content,
                    "native_preparation_original_changed")
            source_map[content] = str(source)
            sources.append({"content_sha256": content, "split": "train", "asset": asset.model_dump(mode="json"),
                "native_ledger_relative_path": ledger_path.relative_to(output).as_posix(), **ledger_info,
                "selected_jpegs": len(refs), "original_bytes_unchanged": True})
            total_native += len(ledger)
            total_jpegs += len(refs)
        require(len(resolved) == expected_records * 2, "native_preparation_resolved_count_changed")
        _write(output / "source-map.container.json", source_map)
        if host_source_root:
            _write(output / "source-map.host.json", {content: str(Path(host_source_root) / (content + ".mov"))
                                                   for content in source_map})
        _write(output / "sources.json", {"schema_version": SCHEMA, "sources": sources})
        _write(output / "resolved-attempt-plan.json", {"schema_version": "forte.native-resolved-teacher-attempt-plan.v1",
            "pins": pins, "phase": "teacher", "maximum_attempts": len(resolved), "attempts": resolved,
            "model_calls": 0, "full_clip_coverage_claimed": False})
        tree = {path.relative_to(output).as_posix(): file_sha256(path)
                for path in sorted(output.rglob("*")) if path.is_file()}
        packages = {name: importlib.metadata.version(name) for name in ("av", "pillow", "pydantic")}
        receipt = {"schema_version": SCHEMA, "status": "prepared_cpu_only", "pins": pins,
            "records": len(sources), "windows": len(resolved), "native_source_frames": total_native,
            "original_jpegs": total_jpegs, "window_frame_uses": sum(len(row["frames"]) for row in resolved),
            "source_bytes_unchanged": True, "native_pts_resolved": True, "full_clip_coverage_claimed": False,
            "model_calls": 0, "gpu_used": False, "gpu_device_nodes_present": bool(list(Path("/dev").glob("nvidia*"))),
            "runtime_image_sha256": runtime_image_sha256, "packages": packages, "python": platform.python_version(),
            "video_module_sha256": {name: file_sha256(Path(importlib.import_module(name).__file__))
                for name in ("forte.video.probe", "forte.video.decode", "forte.video.batches", "forte.video.sampling")},
            "output_tree_sha256": canonical_sha256(tree), "output_files": tree,
            "resolved_attempt_plan_sha256": tree["resolved-attempt-plan.json"],
            "elapsed_sec": time.monotonic() - started, "completed_at": datetime.now(timezone.utc).isoformat()}
        _write(output / "receipt.private.json", receipt)
        summary = {key: value for key, value in receipt.items() if key not in {"pins", "output_files"}}
        summary["receipt_file_sha256"] = file_sha256(output / "receipt.private.json")
        _write(output / "aggregate.json", summary)
        return summary
    except BaseException:
        _write(output / "failure.json", {"schema_version": SCHEMA, "status": "failed",
            "completed_records": len(sources), "resolved_windows": len(resolved), "model_calls": 0,
            "elapsed_sec": time.monotonic() - started})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "attempt-plan", "split", "source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-attempt-sha256", required=True)
    parser.add_argument("--expected-records", type=int, default=17)
    parser.add_argument("--max-runtime-sec", type=float, default=900)
    parser.add_argument("--runtime-image-sha256")
    parser.add_argument("--host-source-root")
    args = parser.parse_args()
    try:
        result = prepare(args.plan, args.attempt_plan, args.split, args.source_root, args.output,
            expected_plan_sha256=args.expected_plan_sha256, expected_attempt_sha256=args.expected_attempt_sha256,
            expected_records=args.expected_records, max_runtime_sec=args.max_runtime_sec,
            runtime_image_sha256=args.runtime_image_sha256, host_source_root=args.host_source_root)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"status": "native_preparation_failed", "error_type": type(error).__name__,
                          "private_details_omitted": True}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
