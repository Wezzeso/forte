"""Private source/native/result receipt exporter; no model or network calls.

The native PTS ledger is an additional bounded CPU decode after the result. It
does not replace an upload/client clock or the actual pipeline's decode trace.
"""

from __future__ import annotations

import argparse
import base64
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

from evaluation.io import digest, read_json
from forte.model.adapter import strict_json_object
from forte.shared.schemas import CompletedResponse, FrameRef, VideoAsset
from forte.video.decode import frame_identity, validate_frame_ref
from forte.video.probe import check_cancelled, content_sha256


def write_private_new(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as target:
        path.chmod(0o600)
        json.dump(value, target, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        target.write("\n")


def native_ledger(asset: VideoAsset, *, deadline: float, max_frames: int = 100_000) -> dict:
    """Decode only the chosen video stream, preserving raw native timestamps."""
    import av

    source = Path(asset.owned_path)
    if content_sha256(source, deadline_monotonic=deadline) != asset.content_hash:
        raise ValueError("export_source_hash_changed")
    base = Fraction(asset.time_base)
    origin = Fraction(str(asset.source_start_sec)) / base
    if abs(float(origin) - round(float(origin))) > 1e-5:
        raise ValueError("export_source_origin_not_native_ticks")
    origin_pts = round(float(origin))
    frames = []
    with av.open(str(source), options={"protocol_whitelist": "file,pipe"}) as container:
        selected = [stream for stream in container.streams.video if stream.index == asset.stream_index]
        if len(selected) != 1 or Fraction(selected[0].time_base) != base:
            raise ValueError("export_source_stream_or_clock_mismatch")
        for frame in container.decode(selected[0]):
            check_cancelled(deadline, None)
            if frame.pts is None or frame.is_corrupt or len(frames) >= max_frames:
                raise ValueError("export_native_frame_invalid_or_limit")
            raw_dimensions = ((asset.height, asset.width) if asset.rotation_deg in (90, 270)
                              else (asset.width, asset.height))
            if not frame.time_base or Fraction(frame.time_base) != base or (frame.width, frame.height) != raw_dimensions:
                raise ValueError("export_native_units_or_geometry_changed")
            pts = int(frame.pts)
            if frames and pts <= frames[-1]["source_pts"]:
                raise ValueError("export_native_pts_not_increasing")
            timestamp = float((pts - origin_pts) * base)
            if timestamp < 0 or timestamp > asset.duration_sec + 1e-6:
                raise ValueError("export_native_pts_outside_source")
            frames.append({"frame_id": frame_identity(asset, pts), "source_pts": pts, "timestamp_sec": timestamp})
    if not frames or frames[0]["source_pts"] != origin_pts or (asset.frame_count is not None and len(frames) != asset.frame_count):
        raise ValueError("export_native_source_incomplete")
    if content_sha256(source, deadline_monotonic=deadline) != asset.content_hash:
        raise ValueError("export_source_changed_during_index")
    return {"content_sha256": asset.content_hash, "source_complete": True, "tail_included": True,
            "origin_pts": origin_pts, "time_base_num": base.numerator, "time_base_den": base.denominator,
            "decoded_source_frame_count": len(frames), "frames": frames,
            "ledger_origin": "additional_full_native_cpu_decode_after_result_not_inference_sampling"}


def build_server_receipt(*, audit: dict, result_bytes: bytes, binding: dict, witness: dict,
                         native_source: dict) -> dict:
    """Pure join of genuine run artifacts; never creates missing stage claims."""
    result = strict_json_object(result_bytes)
    CompletedResponse.model_validate(result)
    asset = VideoAsset.model_validate(audit["asset"])
    if not (witness.get("schema_version") == "forte.qwen38-fresh-stage-witness.v1"
            and witness.get("status") == "done"
            and audit.get("phase") == "done" and audit.get("publication_ready") is True
            and audit.get("coverage", {}).get("complete") is True
            and audit.get("job_id") == result["job_id"] == binding["job_id"] == witness["job_id"]
            and asset.content_hash == binding["content_sha256"] == native_source["content_sha256"]
            and witness.get("result_sha256") == hashlib.sha256(result_bytes).hexdigest()
            and witness.get("source_attempt_id") == binding["attempt_id"]
            and witness.get("audit_sha256") == digest(audit)
            and witness.get("fresh_output_directory_created") is True
            and witness.get("cache_hits") == {"decode": 0, "model": 0, "evidence": 0, "whole_result": 0}
            and all(type(witness.get("stage_calls", {}).get(key)) is int and witness["stage_calls"][key] > 0
                    for key in ("decode", "preprocessing", "model", "evidence"))):
        raise ValueError("export_actual_audit_result_fresh_witness_mismatch")
    attested = witness.get("actual_model_identity")
    if attested != binding["expected_model_identity"]:
        raise ValueError("export_actual_model_identity_mismatch")
    native_by_id = {frame["frame_id"]: frame for frame in native_source["frames"]}
    refs = [FrameRef.model_validate(row) for row in audit["frames"]]
    for ref in refs:
        validate_frame_ref(asset, ref)
        if ref.frame_id not in native_by_id or ref.source_pts != native_by_id[ref.frame_id]["source_pts"]:
            raise ValueError("export_audit_frame_not_native")
    # JPEG bytes are joined to exact original frame IDs/source PTS using the
    # source-owned audit refs; matching a timestamp alone never supplies a frame.
    by_jpeg = {}
    for ref in refs:
        path = Path(ref.path)
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != ref.jpeg_sha256:
            raise ValueError("export_source_jpeg_hash_changed")
        by_jpeg[(actual, ref.pts_sec)] = ref
    refs_by_id = {ref.frame_id: ref for ref in refs}
    for entry in witness.get("evidence_sources", []):
        ref = refs_by_id.get(entry["frame_id"])
        if ref is None or abs(ref.pts_sec - entry["timestamp_sec"]) > 1e-7:
            raise ValueError("export_extracted_evidence_source_mismatch")
        if hashlib.sha256(Path(entry["original_jpeg_path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("export_extracted_evidence_jpeg_changed")
        by_jpeg[(entry["sha256"], entry["timestamp_sec"])] = ref
    selected = []
    for event_index, event in enumerate(result["violations"]):
        for frame_index, image in enumerate(event["evidence_frames"]):
            sha = hashlib.sha256(base64.b64decode(image["image_base64"], validate=True)).hexdigest()
            matches = [ref for (value, timestamp), ref in by_jpeg.items()
                       if value == sha and abs(timestamp - image["timestamp_sec"]) <= 1e-7]
            if len(matches) != 1:
                raise ValueError("export_returned_jpeg_not_exact_source_ref")
            ref = matches[0]
            selected.append({"event_index": event_index, "frame_index": frame_index,
                "frame_id": ref.frame_id, "source_pts": ref.source_pts,
                "timestamp_sec": image["timestamp_sec"], "jpeg_sha256": sha})
    from benchmarks.performance.qwen38_acceptance import normalized
    return {"schema_version": "forte.qwen38-acceptance-server-receipt.v1",
        **{key: binding[key] for key in ("attempt_id", "job_id", "content_sha256", "conditions_sha256", "recipe_sha256")},
        "normalized_result_sha256": digest(normalized(result)), "actual_model_identity": attested,
        "cache_hits": witness["cache_hits"], "stage_calls": witness["stage_calls"],
        "cache_scope": "pipeline decode/model-response/evidence/whole-result cache; backend attention KV internals are not independently measured",
        "native_source": native_source, "evidence_frames": selected,
        "decision_projection": {key: audit.get(key) for key in ("decisions", "candidate_events", "verified_events",
            "unresolved_intervals", "refinement_plan", "coverage", "pipeline_profile", "model_name")},
        "source_audit_sha256": digest(audit), "raw_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "execution_trace_bindings": {key: witness[key] for key in ("wrapper_sha256", "pipeline_sha256",
            "trace_source_archive_sha256", "trace_runtime_image_sha256", "trace_binding_sha256") if key in witness},
        "fresh_stage_witness_sha256": digest(witness), "client_upload_or_ui_latency_established_by_export": False}


def export_artifacts(*, audit_path: Path, result_path: Path, binding: dict, witness: dict,
                     output: Path, max_seconds: float = 60) -> dict:
    started = time.monotonic()
    audit = read_json(audit_path)
    asset = VideoAsset.model_validate(audit["asset"])
    ledger = native_ledger(asset, deadline=time.monotonic() + max_seconds)
    receipt = build_server_receipt(audit=audit, result_bytes=result_path.read_bytes(), binding=binding,
                                   witness=witness, native_source=ledger)
    receipt["native_export_elapsed_ms_separate_from_pipeline_result"] = (time.monotonic() - started) * 1000
    write_private_new(output, receipt)
    return {"server_receipt_written": True, "native_source_frames": ledger["decoded_source_frame_count"],
            "selected_evidence_frames": len(receipt["evidence_frames"]), "model_calls_made_by_export": 0,
            "raw_predictions_or_pixels_printed": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "result", "binding", "witness", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = export_artifacts(audit_path=args.audit, result_path=args.result,
            binding=read_json(args.binding), witness=read_json(args.witness), output=args.output)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "server_export_failed", "error_type": type(error).__name__,
                          "raw_predictions_or_pixels_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
