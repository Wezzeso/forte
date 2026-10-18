"""Benchmark recorded pipeline stages without inference or parameter fitting.

Reads only allowlisted structured fields from fast_eval reports or collected
benchmark envelopes. Labels enter only the final scorer and focus-list selection.
No images, scene descriptions, model prose, or raw exception text are exported.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

from compare_model_structure import FIELDS, LAYOUT_KEYS, TYPE_IDS, _finite, _get


CORE_FIELDS = ("cashier_at_desk", "cashier_in_room", "client_at_desk", "other_people",
               "person_at_door", "cash_in_hands", "cash_bag", "id_card_shown",
               "counting_machine", "detector", "hand_counting")
SOURCES = {"rule", "holistic", "specialist", "probe"}
ERROR_CLASSES = {"TimeoutError", "ReadTimeout", "ConnectTimeout", "ConnectError",
                 "HTTPStatusError", "RuntimeError", "ValueError", "ValidationError",
                 "KeyError", "TypeError", "JSONDecodeError", "CancelledError"}
SUPPRESSION_REASONS = {"type_not_promoted", "vlm_disagrees_or_unclear"}


def number(value: Any) -> int | float | None:
    return value if _finite(value) else None


def event_rows(row: dict, key: str) -> list[dict]:
    output = []
    for raw in row.get(key) or []:
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            raw = {"type_id": raw[0], "t": raw[1]}
        if not isinstance(raw, dict) or raw.get("type_id") not in TYPE_IDS:
            continue
        event = {"type_id": raw["type_id"], "t": number(raw.get("t"))}
        if raw.get("source") in SOURCES:
            event["source"] = raw["source"]
        for field in ("score", "confidence"):
            if number(raw.get(field)) is not None:
                event[field] = raw[field]
        output.append(event)
    return output


def verification_rows(row: dict) -> list[dict] | None:
    if not isinstance(row.get("verifications"), list):
        return None
    output = []
    for raw in row["verifications"]:
        if not isinstance(raw, dict):
            continue
        result = {field: number(raw.get(field)) for field in ("confidence", "t_sec", "cand_t")}
        result["visible"] = raw.get("visible") if type(raw.get("visible")) is bool else None
        if raw.get("type_id") in TYPE_IDS:
            result["type_id"] = raw["type_id"]
        if raw.get("source") in SOURCES:
            result["source"] = raw["source"]
        if raw.get("status") in ("done", "failed"):
            result["status"] = raw["status"]
        if raw.get("error_class") in ERROR_CLASSES:
            result["error_class"] = raw["error_class"]
        output.append(result)
    return output


def label_value(row: dict, labels: dict | None) -> bool | None:
    label = labels.get(row.get("name")) if labels is not None else row.get("label")
    if isinstance(label, dict):
        label = label.get("violation")
    return label if type(label) is bool else None


def decision_rows(row: dict) -> dict:
    raw = row.get("decisions")
    if not isinstance(raw, dict):
        return {"source": "not_recorded", "suppressed": None}
    suppressed = []
    for proposal in raw.get("suppressed") or []:
        if not isinstance(proposal, dict) or proposal.get("type_id") not in TYPE_IDS:
            continue
        reason = proposal.get("reason")
        suppressed.append({"type_id": proposal["type_id"], "t": number(proposal.get("t")),
                           "reason": reason if reason in SUPPRESSION_REASONS else "not_recorded"})
    result = {"source": "recorded_structured_decisions", "suppressed": suppressed,
              "suppressed_truncated": raw.get("suppressed_truncated") if type(raw.get("suppressed_truncated")) is bool else None}
    if type(raw.get("specialists_enabled")) is bool:
        result["specialists_enabled"] = raw["specialists_enabled"]
    return result


def state_stats(row: dict, layout: dict) -> dict:
    states = [s for s in row.get("states") or [] if isinstance(s, dict)]
    n = len(states)
    valid = Counter()
    missing = Counter()
    invalid = Counter()
    positives = Counter()
    enum_values = {"laptop": Counter(), "other_people": Counter()}
    for state in states:
        for key in FIELDS:
            value = _get(state, key)
            if value == "missing":
                missing[key] += 1
            elif value == "invalid":
                invalid[key] += 1
            else:
                valid[key] += 1
                positives[key] += value == "true"
                if key in enum_values:
                    enum_values[key][value] += 1
    expected = list(CORE_FIELDS)
    if layout.get("laptop"):
        expected.append("laptop")
    if layout.get("desk"):
        expected.extend(k for k in FIELDS if k.startswith("desk."))
    for box, field in (("cashier_chair", "cashier_seat"), ("client_chair", "client_seat")):
        if layout.get(box):
            expected.append(field)
    expected_values = n * len(expected)
    valid_expected = sum(valid[key] for key in expected)
    unknown = {key: missing[key] + invalid[key] for key in FIELDS if missing[key] or invalid[key]}
    if enum_values["laptop"]["not_visible"]:
        unknown["laptop"] = unknown.get("laptop", 0) + enum_values["laptop"]["not_visible"]
    return {"retained_frames": n, "source_state_frames": number(row.get("state_frames")),
            "states_truncated": row.get("states_truncated") if type(row.get("states_truncated")) is bool else None,
            "expected_fields_given_layout": expected,
            "expected_field_values": expected_values, "valid_expected_field_values": valid_expected,
            "expected_field_coverage": round(valid_expected / expected_values, 6) if expected_values else None,
            "all_allowlisted_field_coverage": round(sum(valid.values()) / (n * len(FIELDS)), 6) if n else None,
            "missing_field_counts": dict(missing), "invalid_field_counts": dict(invalid),
            "unknown_field_counts": unknown,
            "true_counts": {key: positives[key] for key in FIELDS if key not in enum_values},
            "enum_value_counts": {key: dict(values) for key, values in enum_values.items()},
            "missing_expected_field_counts": {key: missing[key] for key in expected if missing[key]},
            "invalid_expected_field_counts": {key: invalid[key] for key in expected if invalid[key]}}


def stage(status: str, wall_s: Any = None, **fields: Any) -> dict:
    return {"status": status, "wall_s": number(wall_s), "calls": None, **fields}


def video_benchmark(row: dict, labels: dict | None = None) -> dict:
    name = row["name"]
    timing = row.get("timings") if isinstance(row.get("timings"), dict) else {}
    events, candidates = event_rows(row, "events"), event_rows(row, "candidates")
    explicit_processed = row.get("processed")
    if type(explicit_processed) is bool:
        processed = explicit_processed
    else:
        processed = not bool(row.get("error")) and "events" in row
    final_status = "done" if processed else "failed"
    verdict = "violation" if processed and events else "no_violation" if processed else "failed"
    label = label_value(row, labels)
    outcome = ("technical_failure" if not processed else
               "unscored" if label is None else
               "true_positive" if label and events else
               "false_negative" if label else
               "false_positive" if events else "true_negative")
    focus = outcome in ("technical_failure", "false_negative")
    first_gap = ("technical_failure_stage_unknown" if not processed else
                 "no_candidate" if outcome == "false_negative" and not candidates else
                 "candidate_not_accepted" if outcome == "false_negative" else None)
    layout = row.get("layout") if isinstance(row.get("layout"), dict) else {}
    valid_boxes, invalid_boxes = {}, []
    for key in LAYOUT_KEYS:
        box = layout.get(key)
        if isinstance(box, list) and len(box) == 4 and all(_finite(v) for v in box):
            valid_boxes[key] = box
            if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
                invalid_boxes.append(key)
        elif box is not None:
            invalid_boxes.append(key)
    state_info = state_stats(row, valid_boxes)
    verifications = verification_rows(row)
    decisions = decision_rows(row)
    accepted_types = Counter(event["type_id"] for event in events)
    candidate_types = Counter(candidate["type_id"] for candidate in candidates)
    verification_info = {
        "source": "recorded_structured_decisions" if verifications is not None else "candidate_event_comparison_only",
        "decisions": verifications,
        "candidate_type_counts": dict(candidate_types), "accepted_event_type_counts": dict(accepted_types),
        "candidate_types_without_accepted_event": sorted(set(candidate_types) - set(accepted_types)),
        "rejection_reason": "not_inferred_from_candidate_event_difference",
    }
    failures = row.get("failure_codes") or []
    error_classes = row.get("error_classes") or []
    return {"name": name, "processed": processed, "outcome": outcome, "focus": focus,
            "label_used_only_for_scoring": label, "first_observed_gap": first_gap,
            "cause_established": False,
            "recorded_failure_code_count": len(failures) if isinstance(failures, list) else None,
            "recorded_error_classes": sorted({code for code in error_classes if code in ERROR_CLASSES}),
            "stages": {
                "decode": stage("observed" if number(row.get("decode_s")) is not None else "not_recorded", row.get("decode_s")),
                "layout": stage("observed" if layout else "not_recorded", timing.get("layout"), boxes=valid_boxes,
                                missing_or_null_boxes=[key for key in LAYOUT_KEYS if layout.get(key) is None],
                                invalid_boxes=invalid_boxes),
                "states": stage("observed" if state_info["retained_frames"] else "not_recorded", timing.get("states"), **state_info),
                "rules": stage("candidate_output_observed" if "candidates" in row else "not_recorded", timing.get("rules"),
                               source_per_candidate_recorded=all("source" in c for c in candidates) if candidates else None,
                               rule_internal_decisions=decisions),
                "candidates": stage("observed" if "candidates" in row else "not_recorded", count=len(candidates), values=candidates),
                "verification": stage("observed" if verifications is not None else "derived_comparison_only", timing.get("verify"), **verification_info),
                "final": stage(final_status, row.get("analysis_wall_s", row.get("wall_s")), verdict=verdict, events=events,
                               recorded_total_calls=number(row.get("calls")),
                               final_assembly_time_s=None),
            }}


def summarize(videos: list[dict]) -> dict:
    outcomes = Counter(video["outcome"] for video in videos)
    gaps = Counter(video["first_observed_gap"] for video in videos if video["first_observed_gap"])
    def stats(values: list[float]) -> dict:
        return {"recorded_n": len(values), "min": round(min(values), 6) if values else None,
                "median": round(statistics.median(values), 6) if values else None,
                "max": round(max(values), 6) if values else None}

    stage_times = {key: stats([video["stages"][key]["wall_s"] for video in videos
                              if video["stages"][key]["wall_s"] is not None])
                   for key in ("decode", "layout", "states", "rules", "candidates", "verification", "final")}
    verification_counts = Counter()
    suppression_counts = Counter()
    suppression_types = Counter()
    for video in videos:
        for decision in video["stages"]["verification"]["decisions"] or []:
            verification_counts["visible_true" if decision["visible"] is True else
                                "visible_false" if decision["visible"] is False else "visible_unknown"] += 1
            if "status" in decision:
                verification_counts["status_" + decision["status"]] += 1
        for proposal in video["stages"]["rules"]["rule_internal_decisions"]["suppressed"] or []:
            suppression_counts[proposal["reason"]] += 1
            suppression_types[proposal["type_id"]] += 1
    return {"videos": len(videos), "outcome_counts": dict(outcomes),
            "processed": sum(video["processed"] for video in videos),
            "correct": outcomes["true_positive"] + outcomes["true_negative"],
            "fp": outcomes["false_positive"], "fn": outcomes["false_negative"],
            "technical_failures": outcomes["technical_failure"],
            "focus_names": [video["name"] for video in videos if video["focus"]],
            "first_observed_gap_counts": dict(gaps),
            "recorded_stage_time_s": stage_times,
            "recorded_verification_decision_counts": dict(verification_counts),
            "recorded_suppression_reason_counts": dict(suppression_counts),
            "recorded_suppression_type_counts": dict(suppression_types)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--run", action="append", help="Select run keys; default all recorded runs")
    parser.add_argument("--labels", type=Path, help="Optional scoring labels when row labels are absent")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8-sig")) if args.labels else None
    runs, inputs = {}, []
    for path in args.input:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        entries = report.get("runs") or {path.stem: report}
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for key, run in entries.items():
            if args.run and key not in args.run:
                continue
            if not isinstance(run, dict):
                continue
            rows = run.get("results", run.get("videos"))
            if not isinstance(rows, list):
                continue
            if key in runs:
                raise ValueError("duplicate selected run key")
            names = [row.get("name") for row in rows]
            if len(set(names)) != len(names) or not all(isinstance(name, str) for name in names):
                raise ValueError("unique per-video names required")
            videos = [video_benchmark(row, labels) for row in sorted(rows, key=lambda row: row["name"])]
            runs[key] = {"summary": summarize(videos), "videos": videos}
    if not runs:
        raise ValueError("no selected per-video reports")
    output = {"schema": "forte.pipeline-step-benchmark.v1",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "recorded_development_runs_in_sample",
              "inference_executed": False, "parameters_fitted": 0,
              "labels_used_only_for_scoring_and_focus_selection": True,
              "cause_claim": False, "heldout_accuracy_claim": False,
              "measurement_limits": {
                  "per_stage_calls": "not_recorded_null",
                  "rules_internal_decisions": "recorded_suppression_only_when_available_other_internal_decisions_unknown",
                  "final_assembly_time": "not_recorded_null",
                  "missing_box": "may_be_object_not_visible_not_technical_failure",
                  "missing_optional_state_field": "may_follow_missing_layout_box",
                  "unknown_field_counts": "missing_invalid_or_laptop_not_visible_kept_distinct_in_detail",
                  "stage_timings": "source_recorded_may_overlap_do_not_sum",
                  "verification": "candidate_event_comparison_does_not_establish_rejection_reason",
                  "analysis_wall_s": "analyzer_measurement_not_public_API_202_to_done",
              }, "inputs": inputs,
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "runs": runs}
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "runs": {key: run["summary"] for key, run in runs.items()}}))


if __name__ == "__main__":
    main()
