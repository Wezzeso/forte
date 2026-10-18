"""Post-prediction approximate-anchor diagnostics, never exact event truth.

Uses only the validation-role user-provided coarse points and raw-source-PTS
temporal exports. It makes no inference/model/API/GPU call and never supplies
validation anchors to training, prompts or threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from .audit import sample_id
from .io import digest, read_json, read_records, write_json
from .metrics import distribution, number, ratio
from .weak_supervision import provided_label_digest, weak_assignment_digest


def _index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError("missing or duplicate coarse-diagnostic identity")
        result[identifier] = row
    return result


def _source_scores(row: dict, content: str, duration: float | None) -> tuple[list[tuple[float, float]], Counter]:
    """Recheck A05's hash/raw PTS/time-base/origin identity, without media I/O."""
    counts: Counter = Counter()
    time_base = row.get("time_base")
    origin = number(row.get("source_start_sec"))
    try:
        base = Fraction(time_base) if isinstance(time_base, str) else None
    except (ValueError, ZeroDivisionError):
        base = None
    if base is None or base <= 0 or origin is None or not isinstance(row.get("frames"), list):
        return [], Counter({"source_timeline_metadata_invalid": 1})
    frames, seen = [], set()
    for frame in row["frames"]:
        counts["exported_frame_rows"] += 1
        if not isinstance(frame, dict):
            counts["invalid_source_frame_rows"] += 1
            continue
        raw_pts, pts, score = frame.get("source_pts"), number(frame.get("pts_sec")), number(frame.get("eventness_logit"))
        if isinstance(raw_pts, bool) or not isinstance(raw_pts, int) or pts is None:
            counts["invalid_source_frame_rows"] += 1
            continue
        expected_pts = float(raw_pts * base) - origin
        frame_id = hashlib.sha256(f"{content}:{raw_pts}:{time_base}".encode("ascii")).hexdigest()
        tolerance = max(1e-7, 4 * math.ulp(abs(expected_pts)))
        if frame.get("frame_id") != frame_id or not math.isclose(pts, expected_pts, rel_tol=0, abs_tol=tolerance) or pts < -tolerance or (duration is not None and pts > duration + tolerance):
            counts["invalid_source_frame_rows"] += 1
            continue
        if frame_id in seen:
            counts["duplicate_source_frame_rows"] += 1
            continue
        seen.add(frame_id)
        counts["valid_original_source_frames"] += 1
        if score is None or frame.get("frame_valid", True) is not True:
            counts["unusable_or_unscored_frames"] += 1
            continue
        frames.append((max(0.0, pts), score))
    return sorted(frames), counts


def evaluate_coarse_anchors(
    anchors: list[dict], temporal_predictions: list[dict], expected_attempts: list[dict], terminals: list[dict], *,
    split_manifest: dict, anchor_pins: dict, metadata: dict, threshold_artifact: dict,
    threshold_artifact_sha256: str, tolerances_sec: tuple[float, ...] = (1.0, 2.0, 5.0),
) -> dict:
    if not tolerances_sec or any(not math.isfinite(t) or t < 0 for t in tolerances_sec):
        raise ValueError("diagnostic tolerances must be finite and nonnegative")
    tolerances_sec = tuple(sorted(set(float(t) for t in tolerances_sec)))
    expected = _index(expected_attempts, "attempt_id")
    results = _index(terminals, "attempt_id")
    temporal = _index(temporal_predictions, "attempt_id")
    if results.keys() - expected.keys() or temporal.keys() - expected.keys():
        raise ValueError("unexpected validation attempts")
    validation = {sample_id(row): row for row in split_manifest["records"] if row.get("split") == "validation"}
    validation_contents = {row.get("content_sha256", row.get("content_hash")) for row in validation.values()}
    if anchor_pins.get("assignments_sha256") != weak_assignment_digest(split_manifest["records"]) or anchor_pins.get("assignments_sha256") != split_manifest.get("assignments_sha256"):
        raise ValueError("coarse anchor split pin mismatch")
    anchor_digest = provided_label_digest(anchors)
    if anchor_pins.get("role_manifests", {}).get("validation", {}).get("sha256") != anchor_digest:
        raise ValueError("coarse validation anchor artifact mismatch")
    threshold = number(threshold_artifact.get("minimum_video_score"))
    if threshold_artifact.get("schema_version") != "forte.weak_video_threshold.v1" or threshold_artifact.get("score_axis") != "violation_log_odds" or threshold_artifact.get("comparison") != "greater_than_or_equal" or threshold is None:
        raise ValueError("invalid frozen eventness threshold")
    blockers: Counter = Counter()
    for field in ("validation_anchors_used_for_training", "validation_anchors_used_for_threshold_selection", "validation_anchors_used_for_prompting"):
        if metadata.get(field) is not False:
            blockers["validation_anchor_exclusion_unverified"] += 1
    if metadata.get("threshold_selection_sha256") != threshold_artifact_sha256 or threshold_artifact.get("validation_labels_used_for_selection") is not False:
        blockers["frozen_threshold_or_calibration_provenance_unverified"] += 1
    rows_by_content: dict[str, list[str]] = defaultdict(list)
    for aid, planned in expected.items():
        sid = planned.get("sample_id")
        if sid not in validation:
            raise ValueError("only validation attempts may enter coarse diagnostics")
        rows_by_content[validation[sid].get("content_sha256", validation[sid].get("content_hash"))].append(aid)
    if validation_contents - rows_by_content.keys():
        blockers["validation_contents_missing_from_frozen_attempt_plan"] += len(validation_contents - rows_by_content.keys())
    counter: Counter = Counter()
    band_counts = {t: Counter() for t in tolerances_sec}
    nearest_sample, nearest_positive, sampled_spacing = [], [], []
    seen_contents = set()
    for record in anchors:
        content = record.get("content_sha256", record.get("content_hash"))
        if content not in validation_contents or content in seen_contents or record.get("split") != "validation":
            raise ValueError("invalid or duplicate validation anchor content")
        seen_contents.add(content)
        if record.get("anchor_origin") != "user_provided_coarse_table" or record.get("anchor_semantics") != "approximate_point_not_boundary" or record.get("target_scope") != "class_agnostic_approximate_event_anchor" or record.get("allowed_use") != "evaluation_only_do_not_condition_inference":
            raise ValueError("coarse anchor scope/provenance invalid")
        if record.get("anchor_source_sha256") != anchor_pins.get("anchor_source_sha256") or record.get("provided_label_source_sha256") != anchor_pins.get("provided_label_source_sha256") or record.get("assignments_sha256") != anchor_pins.get("assignments_sha256"):
            raise ValueError("coarse anchor source pin mismatch")
        points = record.get("coarse_anchor_sec")
        if not isinstance(points, list) or any(number(point) is None or point < 0 for point in points) or len(set(points)) != len(points):
            raise ValueError("invalid coarse points")
        if not isinstance(record.get("provided_video_label"), bool) or (record["provided_video_label"] != bool(points)):
            raise ValueError("binary/coarse point consistency failure")
        uncertainty = number(record.get("uncertainty_sec"))
        if uncertainty is not None and uncertainty < 0:
            raise ValueError("negative source uncertainty")
        counter["source_uncertainty_unspecified_points"] += len(points) if uncertainty is None else 0
        counter["source_uncertainty_specified_points"] += len(points) if uncertainty is not None else 0
        counter["validation_contents"] += 1
        counter["provided_positive_contents"] += bool(points)
        counter["provided_coarse_points"] += len(points)
        attempt_ids = rows_by_content.get(content, [])
        # Repeats do not increase anchor/video denominators or permit cherry-picking.
        aid = attempt_ids[0] if attempt_ids else None
        result, prediction = results.get(aid, {}), temporal.get(aid, {})
        done = result.get("status") == "done" and (result.get("prediction_validated") is True or result.get("public_contract_validated") is True)
        scored = []
        if done and prediction:
            if prediction.get("sample_id") != expected[aid]["sample_id"] or prediction.get("score_axis") != "violation_log_odds" or number(prediction.get("threshold")) != threshold:
                raise ValueError("temporal prediction identity/threshold mismatch")
            if prediction.get("head_sha256") != metadata.get("model_revision") or (isinstance(metadata.get("head_identity"), dict) and prediction.get("feature_spec_sha256") != metadata["head_identity"].get("feature_spec_sha256")):
                blockers["temporal_model_feature_identity_unverified"] += 1
            duration = number(validation[expected[aid]["sample_id"]].get("duration_sec"))
            scored, source_counts = _source_scores(prediction, content, duration)
            counter.update(source_counts)
        counter["contents_with_usable_temporal_predictions"] += bool(scored)
        counter["points_with_failed_or_missing_temporal_prediction"] += len(points) if not scored else 0
        counter["scored_unique_original_frames"] += len(scored)
        positive_times = [pts for pts, score in scored if score >= threshold]
        counter["above_threshold_original_frames"] += len(positive_times)
        sampled_spacing.extend((b[0] - a[0]) * 1000 for a, b in zip(scored, scored[1:]))
        for point in points:
            if scored:
                nearest_sample.append(min(abs(pts - point) for pts, _ in scored) * 1000)
            if positive_times:
                nearest_positive.append(min(abs(pts - point) for pts in positive_times) * 1000)
            for tolerance in tolerances_sec:
                width = (uncertainty or 0.0) + tolerance
                covered = any(abs(pts - point) <= width for pts, _ in scored)
                hit = any(abs(pts - point) <= width for pts in positive_times)
                band_counts[tolerance]["covered_points"] += covered
                band_counts[tolerance]["hit_points"] += hit
    if seen_contents != validation_contents:
        raise ValueError("validation anchor manifest must cover the frozen validation population")
    points_count = counter["provided_coarse_points"]
    report = {
        "schema_version": "forte.coarse_anchor_diagnostic.v1",
        "scope": "post_prediction_provided_validation_coarse_points_only",
        "status": "SYNTHETIC_ONLY" if metadata.get("inference_kind") == "synthetic" else "APPROXIMATE_DIAGNOSTIC_ONLY" if not blockers else "DIAGNOSTIC_PROVENANCE_INCOMPLETE",
        "event_type_ground_truth": "UNKNOWN", "event_boundary_ground_truth": "UNKNOWN", "legal_ground_truth": "UNKNOWN", "outside_anchor_region_ground_truth": "UNKNOWN",
        "exact_event_f1_claim_permitted": False, "exact_action_onset_claim_permitted": False, "hundred_millisecond_accuracy_claim_permitted": False,
        "denominators": {"validation_unique_contents": counter["validation_contents"], "provided_positive_contents": counter["provided_positive_contents"], "provided_coarse_points": points_count, "contents_with_usable_temporal_predictions": counter["contents_with_usable_temporal_predictions"], "points_with_failed_or_missing_temporal_prediction": counter["points_with_failed_or_missing_temporal_prediction"], "note": "Points within a recording and repeated attempts are not independent events or videos; first frozen attempt per content only."},
        "bands": [{"team_diagnostic_expansion_sec": tolerance, "covered_points": counts["covered_points"], "hit_points": counts["hit_points"], "uncovered_or_failed_points": points_count - counts["covered_points"], "failure_inclusive_anchor_hit_rate": ratio(counts["hit_points"], points_count), "conditional_hit_rate_given_scored_frame_coverage": ratio(counts["hit_points"], counts["covered_points"])} for tolerance, counts in band_counts.items()],
        "distance_to_supplied_coarse_point_ms_not_true_onset_error": {"nearest_scored_source_frame": distribution(nearest_sample), "nearest_above_threshold_source_frame": distribution(nearest_positive)},
        "frame_diagnostics": {"exported_frame_rows": counter["exported_frame_rows"], "valid_original_source_frames": counter["valid_original_source_frames"], "scored_unique_original_frames": counter["scored_unique_original_frames"], "unusable_or_unscored_frames": counter["unusable_or_unscored_frames"], "above_threshold_original_frames": counter["above_threshold_original_frames"], "above_threshold_frame_fraction": ratio(counter["above_threshold_original_frames"], counter["scored_unique_original_frames"]), "invalid_source_frame_rows": counter["invalid_source_frame_rows"], "duplicate_source_frame_rows": counter["duplicate_source_frame_rows"], "source_timeline_metadata_invalid": counter["source_timeline_metadata_invalid"], "actual_scored_sampling_spacing_ms": distribution(sampled_spacing), "sampling_spacing_is_not_action_timing_precision": True},
        "source_uncertainty": {"points_unspecified": counter["source_uncertainty_unspecified_points"], "points_specified": counter["source_uncertainty_specified_points"], "band_rule": "provided point +/- explicitly supplied uncertainty (0 expansion if unspecified) +/- separately declared team diagnostic expansion; unknown uncertainty remains unknown"},
        "bindings": {"split_assignments_sha256": anchor_pins.get("assignments_sha256"), "anchor_source_sha256": anchor_pins.get("anchor_source_sha256"), "validation_anchor_manifest_sha256": anchor_digest, "expected_attempts_sha256": digest(expected_attempts), "temporal_predictions_sha256": digest(temporal_predictions), "threshold_selection_sha256": threshold_artifact_sha256, "head_sha256": metadata.get("model_revision")},
        "provenance_blocker_counts": dict(sorted(blockers.items())),
        "limitations": ["A coarse point is an approximate positive timing cue, not an exact event start/end, class, legal reason or complete event inventory.", "Anchor hit can be inflated by scoring every frame positive; above-threshold frame fraction is reported and no event precision/F1 is inferred.", "Failed/missing temporal results and sampling gaps remain in every provided-point denominator.", "Distances are to supplied approximate points, not human-selected source-frame anchors or true physical onset; no 100 ms accuracy claim follows.", "Validation anchors are post-prediction diagnostics only; they must not condition training, prompts or threshold selection. Final-check data is not read."],
    }
    report["diagnostic_sha256"] = digest(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("anchors", "anchor-pins", "temporal-predictions", "expected-attempts", "terminals", "splits", "metadata", "threshold", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tolerance-sec", type=float, action="append")
    args = parser.parse_args(argv)
    try:
        report = evaluate_coarse_anchors(read_records(args.anchors), read_records(args.temporal_predictions), read_records(args.expected_attempts), read_records(args.terminals), split_manifest=read_json(args.splits), anchor_pins=read_json(args.anchor_pins), metadata=read_json(args.metadata), threshold_artifact=read_json(args.threshold), threshold_artifact_sha256=hashlib.sha256(Path(args.threshold).read_bytes()).hexdigest(), tolerances_sec=tuple(args.tolerance_sec or (1, 2, 5)))
        write_json(args.output, report)
        print(json.dumps({"status": report["status"], "validation_contents": report["denominators"]["validation_unique_contents"], "coarse_points": report["denominators"]["provided_coarse_points"], "exact_event_or_timing_accuracy_claim": False}, sort_keys=True))
        return 0 if report["status"] == "APPROXIMATE_DIAGNOSTIC_ONLY" else 2
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({"error": "coarse_anchor_diagnostic_input_invalid", "private_values_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
