"""Optional all-attempt target evaluation; offline and aggregate-only.

Targets are 100% observed detection/grounding and strictly <10 seconds from
upload start through the final result byte. A finite test cannot prove universal
100% accuracy. Synthetic inputs never authorize a real accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .audit import audit_split, sample_id
from .io import digest, read_json, read_records, write_json
from .metrics import anchor, distribution, event_list, event_type, match_events, number, ratio, reviewed


def _index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError("missing or duplicate record identity")
        result[identifier] = row
    return result


def _source_timeline(truth: dict) -> tuple[dict[str, float], list[float]]:
    timing = truth.get("source_timing", {})
    values = timing.get("original_frame_pts_sec", [])
    if timing.get("original_timeline_verified") is not True or not isinstance(values, list) or not values:
        return {}, []
    timeline = [number(value) for value in values]
    if any(value is None or value < 0 for value in timeline) or any(b <= a for a, b in zip(timeline, timeline[1:])):
        return {}, []
    frames = {}
    for row in truth.get("source_frames", truth.get("frames", [])):
        fid, pts = row.get("frame_id"), number(row.get("pts_sec"))
        if not isinstance(fid, str) or fid in frames or pts is None or not any(abs(pts - value) <= 1e-9 for value in timeline):
            return {}, []
        frames[fid] = pts
    return frames, timeline


def _bound_event(event: dict, frames: dict[str, float], *, truth: bool = False) -> dict | None:
    fid = event.get("anchor_frame_id")
    declared = anchor(event)
    if truth and not fid and declared is not None:
        candidates = [key for key, value in frames.items() if abs(value - declared) <= 1e-9]
        fid = candidates[0] if len(candidates) == 1 else None
    if fid not in frames or declared is None or not event_type(event):
        return None
    pts = frames[fid]
    # Up to half a millisecond is representation rounding, never action error.
    if abs(declared - pts) * 1000 > (1e-6 if truth else 0.500001):
        return None
    return {**event, "anchor_frame_id": fid, "anchor_sec": pts}


def evaluate_strict_targets(
    truth_records: list[dict], expected_attempts: list[dict], attempt_records: list[dict], *,
    split_manifest: dict, split_manifest_sha256: str, metadata: dict,
    verified_rule_ids: set[str], deadline_ms: float = 10_000.0,
    event_match_tolerance_ms: float = 1000.0, max_anchor_error_ms: float | None = None,
) -> dict:
    """No calls, source media reads, training, split locking or per-video output.

    The expected-attempt manifest is frozen before a run. A failed multipart
    acceptance without a durable job still requires an attempt record. Missing
    records count as failed attempts. Event matching uses source-frame PTS rather
    than rounded displayed milliseconds.
    """
    if deadline_ms != 10_000.0 or not math.isfinite(deadline_ms) or event_match_tolerance_ms < 0 or not math.isfinite(event_match_tolerance_ms):
        raise ValueError("invalid target policy")
    if max_anchor_error_ms is not None and (max_anchor_error_ms < 0 or not math.isfinite(max_anchor_error_ms)):
        raise ValueError("invalid source-anchor threshold")
    truth = _index(truth_records, "sample_id")
    expected = _index(expected_attempts, "attempt_id")
    actual = _index(attempt_records, "attempt_id")
    if any(not isinstance(row.get("sample_id"), str) or not row["sample_id"] for row in expected.values()):
        raise ValueError("planned attempts need sample identity")
    if actual.keys() - expected.keys():
        raise ValueError("unexpected attempts cannot be silently dropped")
    split_audit = audit_split(split_manifest)
    split_rows = {sample_id(row): row for row in split_manifest["records"]}
    counts: Counter = Counter()
    blockers: Counter = Counter()
    confusion = Counter({key: 0 for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative")})
    latencies, anchor_errors, frame_gaps, rounding_errors = [], [], [], []
    if not expected:
        blockers["empty_expected_attempt_manifest"] += 1
    if not split_audit["training_gate_passed"]:
        blockers["split_not_locked_or_integrity_failed"] += 1
    if metadata.get("inference_kind") != "real_local" or metadata.get("execution_verified") is not True:
        blockers["real_local_execution_unverified"] += 1
    if metadata.get("exposure_audit_passed") is not True or not metadata.get("exposure_registry_sha256"):
        blockers["latest_exposure_scenario_audit_unverified"] += 1
    if not metadata.get("model_revision") or not metadata.get("conditions_sha256") or not metadata.get("rules_sha256"):
        blockers["model_conditions_or_rule_provenance_missing"] += 1
    if metadata.get("attempt_manifest_frozen_before_run") is not True:
        blockers["all_attempt_denominator_not_frozen_before_run"] += 1
    if metadata.get("uncached_inputs_verified") is not True:
        blockers["uncached_execution_unverified"] += 1
    if metadata.get("heldout_use_disclosed") is not True:
        blockers["heldout_use_or_tuning_not_disclosed"] += 1
    for aid, planned in expected.items():
        sid = planned.get("sample_id")
        annotation, observed = truth.get(sid, {}), actual.get(aid, {})
        split_row = split_rows.get(sid, {})
        truth_ready = (
            reviewed(annotation) and annotation.get("review_origin") == "human"
            and annotation.get("full_review_complete") is True
            and annotation.get("split_lock_verified") is True
            and annotation.get("split_manifest_sha256") == split_manifest_sha256
            and annotation.get("verdict") in {"violation", "no_violation"}
            and not annotation.get("unresolved_items")
            and split_row.get("split") in {"validation", "final_check"}
            and annotation.get("split") == split_row.get("split")
            and annotation.get("content_sha256", annotation.get("content_hash")) == split_row.get("content_sha256", split_row.get("content_hash"))
            and annotation.get("group_id") == split_row.get("group_id")
        )
        if not truth_ready:
            blockers["attempt_truth_not_fully_reviewed_and_locked_heldout"] += 1
        elif annotation["verdict"] == "no_violation" and annotation.get("negative_verified") is not True:
            truth_ready = False
            blockers["full_negative_review_unverified"] += 1
        counts["missing_attempt_results"] += not bool(observed)
        duration_ms = number(observed.get("upload_start_to_full_result_ms"))
        timing_valid = observed.get("timing_origin") == "client_upload_start_to_full_result_byte" and duration_ms is not None and duration_ms >= 0
        complete = observed.get("sample_id") == sid and observed.get("status") == "done" and observed.get("verdict") in {"violation", "no_violation"} and observed.get("public_contract_validated") is True
        try:
            predicted_events = event_list(observed) if complete else []
            expected_events = event_list(annotation) if truth_ready else []
        except ValueError:
            complete = False
            truth_ready = False
            predicted_events, expected_events = [], []
            blockers["malformed_event_arrays"] += 1
        if truth_ready and (annotation["verdict"] == "violation") != bool(expected_events):
            truth_ready = False
            blockers["truth_verdict_events_inconsistent"] += 1
        counts["truth_ready_attempts"] += truth_ready
        if complete and ((observed["verdict"] == "violation") != bool(predicted_events)):
            complete = False
            predicted_events = []
        counts["completed"] += complete
        fast = complete and timing_valid and duration_ms < deadline_ms
        counts["strict_latency_passed"] += fast
        counts["timing_unknown_or_invalid"] += not timing_valid
        if timing_valid:
            latencies.append(duration_ms)
        if truth_ready:
            positive = annotation["verdict"] == "violation"
            counts["reviewed_positive_attempts"] += positive
            counts["reviewed_negative_attempts"] += not positive
            if not complete:
                confusion["unprocessed_positive" if positive else "unprocessed_negative"] += 1
            else:
                predicted_positive = observed["verdict"] == "violation"
                confusion[("tp" if predicted_positive else "fn") if positive else ("fp" if predicted_positive else "tn")] += 1
        frames, timeline = _source_timeline(annotation)
        bound_truth, bound_pred = [], []
        source_valid = True
        for event in expected_events:
            bound = _bound_event(event, frames, truth=True)
            if bound is None:
                source_valid = False
                continue
            bound_truth.append(bound)
            location = min(range(len(timeline)), key=lambda index: abs(timeline[index] - bound["anchor_sec"]))
            if location > 0:
                frame_gaps.append((timeline[location] - timeline[location - 1]) * 1000)
            if location + 1 < len(timeline):
                frame_gaps.append((timeline[location + 1] - timeline[location]) * 1000)
            if event.get("rule_grounding_status") != "verified_applicable" or event.get("rule_id") not in verified_rule_ids or event.get("temporal_verified") is not True or event.get("exceptions_verified") is not True:
                source_valid = False
                blockers["truth_grounding_or_temporal_review_incomplete"] += 1
        for event in predicted_events:
            bound = _bound_event(event, frames)
            if bound is None:
                source_valid = False
                continue
            bound_pred.append(bound)
            pts_ms = bound["anchor_sec"] * 1000
            rounding_errors.append(abs(round(pts_ms) - pts_ms))
            if number(event.get("timestamp_ms")) is None or abs(event["timestamp_ms"] - round(pts_ms)) > 1e-6:
                source_valid = False
                blockers["numeric_millisecond_representation_mismatch"] += 1
        if not source_valid or len(bound_truth) != len(expected_events) or len(bound_pred) != len(predicted_events):
            blockers["source_frame_binding_or_timestamp_review_incomplete"] += 1
        matches = match_events(bound_truth, bound_pred, event_match_tolerance_ms / 1000)
        counts["event_tp"] += len(matches)
        counts["event_fp"] += len(predicted_events) - len(matches)
        counts["event_fn"] += len(expected_events) - len(matches)
        grounding_ok = source_valid
        anchor_threshold_ok = True
        for i, j, error_sec in matches:
            event, expected_event = bound_pred[j], bound_truth[i]
            error_ms = error_sec * 1000
            anchor_errors.append(error_ms)
            if max_anchor_error_ms is not None and error_ms > max_anchor_error_ms:
                anchor_threshold_ok = False
            local = event.get("local_review", {})
            evidence_ids = event.get("evidence_frame_ids", [])
            supported = (
                isinstance(local, dict) and reviewed(local) and local.get("review_origin") == "human"
                and local.get("evidence_supported") is True
                and local.get("timestamp_correspondence_verified") is True
                and local.get("rule_applicability_verified") is True
                and event.get("rule_id") == expected_event.get("rule_id")
                and event.get("rule_id") in verified_rule_ids
                and isinstance(evidence_ids, list) and 1 <= len(evidence_ids) <= 3
                and len(set(evidence_ids)) == len(evidence_ids)
                and event.get("anchor_frame_id") in evidence_ids
                and all(fid in frames for fid in evidence_ids)
            )
            counts["matched_grounding_and_evidence_passed"] += supported
            grounding_ok &= supported
        all_events_match = len(matches) == len(expected_events) == len(predicted_events)
        detection_ok = truth_ready and complete and annotation.get("verdict") == observed.get("verdict") and all_events_match and source_valid
        counts["detection_passed"] += detection_ok
        counts["grounding_and_evidence_passed"] += detection_ok and grounding_ok
        counts["all_criteria_passed"] += fast and detection_ok and grounding_ok and anchor_threshold_ok
    if not counts["reviewed_positive_attempts"] or not counts["reviewed_negative_attempts"]:
        blockers["reviewed_positive_or_negative_coverage_missing"] += 1
    n = len(expected)
    evidence_ready = not blockers
    criteria_satisfied = n > 0 and counts["all_criteria_passed"] == n
    synthetic = metadata.get("inference_kind") == "synthetic"
    status = "SYNTHETIC_ONLY" if synthetic else "UNMEASURED" if not evidence_ready else "PASS" if criteria_satisfied else "FAIL"
    return {
        "schema_version": "forte.strict_targets.v1",
        "status": status,
        "targets": {"observed_detection_rate": 1.0, "observed_grounding_and_evidence_rate": 1.0, "end_to_end_deadline_ms": deadline_ms, "latency_comparison": "strictly_less_than", "latency_origin": "client_upload_start_to_full_result_byte", "event_match_tolerance_ms": event_match_tolerance_ms, "max_original_frame_anchor_error_ms": max_anchor_error_ms},
        "claim_limits": {"universal_100_percent_accuracy_claim_permitted": False, "observed_reviewed_sample_all_criteria_passed": evidence_ready and criteria_satisfied and not synthetic, "synthetic_results_are_software_checks_only": synthetic, "one_millisecond_action_timing_claim_permitted": False},
        "denominators": {"expected_attempts": n, "completed": counts["completed"], "failed_or_unprocessed": n - counts["completed"], "missing_attempt_results": counts["missing_attempt_results"], "fully_reviewed_locked_truth_attempts": counts["truth_ready_attempts"], "distinct_sample_count": len({row.get("sample_id") for row in expected.values()}), "note": "Repeated attempts count for latency/service acceptance but do not increase independent video sample size."},
        "all_attempt_metrics": {"strict_latency_passed": counts["strict_latency_passed"], "strict_latency_rate": ratio(counts["strict_latency_passed"], n), "detection_passed": counts["detection_passed"], "detection_rate": ratio(counts["detection_passed"], n), "grounding_and_evidence_passed": counts["grounding_and_evidence_passed"], "grounding_and_evidence_rate": ratio(counts["grounding_and_evidence_passed"], n), "all_criteria_passed": counts["all_criteria_passed"], "all_criteria_rate": ratio(counts["all_criteria_passed"], n), "timing_unknown_or_invalid": counts["timing_unknown_or_invalid"], "upload_start_to_full_result_ms": distribution(latencies)},
        "whole_video": dict(confusion),
        "events": {"tp": counts["event_tp"], "fp": counts["event_fp"], "fn": counts["event_fn"], "matched_grounding_and_evidence_passed": counts["matched_grounding_and_evidence_passed"]},
        "timestamp_diagnostics": {"numeric_unit": "milliseconds", "display_rounding": "round(source_pts_sec * 1000); representation only", "display_rounding_error_ms": distribution(rounding_errors), "error_to_human_selected_original_frame_anchor_ms": distribution(anchor_errors), "adjacent_original_frame_spacing_ms": distribution(frame_gaps), "sampling_spacing_is_not_source_frame_resolution": True, "source_frame_spacing_does_not_establish_true_action_onset": True, "source_identity_verification": "trusted upstream A05 validation export; offline evaluator does not reread source media or recompute canonical frame identity", "illustration": "At exactly 30 fps, adjacent original frames are about 33.333 ms apart; printing milliseconds does not create 1 ms action timing accuracy."},
        "evidence_blocker_counts": dict(sorted(blockers.items())),
        "input_binding": {"expected_attempt_manifest_sha256": digest(expected_attempts), "split_manifest_sha256": split_manifest_sha256, "split_records_sha256": split_audit["records_sha256"], "model_revision": metadata.get("model_revision"), "conditions_sha256": metadata.get("conditions_sha256"), "rules_sha256": metadata.get("rules_sha256"), "exposure_registry_sha256": metadata.get("exposure_registry_sha256")},
        "limitations": ["100% is a target; even a perfect finite reviewed run establishes only its declared observed sample result, never universal accuracy.", "All failures, missing attempts, invalid payloads and unknown timing fail the strict target; no successful subset substitutes for all attempts.", "Full human review, applicable rule/source review, frame correspondence and locked exposure-safe heldout groups must be real, not model-assigned labels or filled approval flags.", "Event matching tolerance is an explicit team diagnostic, not an organizer tolerance or a claim of 1 ms action localization."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("truth", "expected-attempts", "attempts", "splits", "metadata", "rules", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--event-match-tolerance-ms", type=float, default=1000.0)
    parser.add_argument("--max-anchor-error-ms", type=float)
    args = parser.parse_args(argv)
    try:
        manifest_bytes = Path(args.splits).read_bytes()
        report = evaluate_strict_targets(read_records(args.truth), read_records(args.expected_attempts), read_records(args.attempts), split_manifest=read_json(args.splits), split_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), metadata=read_json(args.metadata), verified_rule_ids=set(read_json(args.rules)["verified_rule_ids"]), event_match_tolerance_ms=args.event_match_tolerance_ms, max_anchor_error_ms=args.max_anchor_error_ms)
        write_json(args.output, report)
        print(json.dumps({"status": report["status"], "expected_attempts": report["denominators"]["expected_attempts"], "all_criteria_passed": report["all_attempt_metrics"]["all_criteria_passed"], "universal_accuracy_claim_permitted": False}, sort_keys=True))
        return 0 if report["status"] == "PASS" else 2
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({"error": "strict_target_input_invalid", "private_values_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
