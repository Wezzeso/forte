"""CPU-only contextual replay of paired, dense procedure observations.

  python tools/contextual_procedure_gate.py --report procedure-visible.json \
      --labels labels.json --out contextual-counting.json

The counting signal requires paired cash_recount=yes, paired visitor=absent,
and clear cashier hands, desk and counting-machine regions. The report does not
distinguish manual/machine counting, so both operative regions are required.
Unclear observations or gaps break episodes; no gap filling is performed.

The duration operating point is fixed from NEGATIVES ONLY: the longest observed
negative episode plus one declared 5-fps sample interval. Positives are scored
once after that value is frozen. This is in-sample diagnostic calibration, not
an independently validated zero-FP result, a service change or type enablement.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_validation import file_hash, json_hash, regression_gate, score_results, validate_labels  # noqa: E402
from model_negative_gate import NAME, load_json, number, sha  # noqa: E402
from procedure_visible_probe import BOUNDARY_SCHEMA, sampled_coverage, window_checks  # noqa: E402

TYPE = "COUNT_WITHOUT_CLIENT"
REQUIRED = ("desk_visibility", "counting_machine_visibility", "cashier_hands_visibility", "visitor_hands_visibility")
VIS_KEYS = ("detector_visibility", *REQUIRED)
ACTION_KEYS = ("banknote_to_detector", "card_to_cashier", "cash_recount", "cash_accepted", "cash_stored")
VIS = {"clear", "occluded", "not_visible", "too_small", "unclear"}
YESNO = {"yes", "no", "unclear"}
DISABLED = {"NO_AUTHENTICITY_CHECK": "cash_acceptance", "NO_CLIENT_ID": "cash_acceptance", "NO_RECOUNT": "bag_to_storage"}
MAX_GAP_S = .25  # Existing 5-fps tool admits one source-frame rounding slack.


def project_observations(raw: Any, start: float, end: float) -> list[dict]:
    if not isinstance(raw, list) or not 2 <= len(raw) <= 1000:
        raise ValueError("observation_count_invalid")
    out = []
    for item in raw:
        if not isinstance(item, dict) or not number(item.get("t"), start, end):
            raise ValueError("observation_timestamp_invalid")
        visibility, actions = item.get("visibility"), item.get("actions")
        if not isinstance(visibility, dict) or not isinstance(actions, dict):
            raise ValueError("observation_fields_invalid")
        for key in VIS_KEYS:
            value = visibility.get(key)
            allowed = VIS | ({"absent"} if key == "visitor_hands_visibility" else set())
            if not isinstance(value, str) or value not in allowed:
                raise ValueError("observation_visibility_invalid")
        if any(not isinstance(actions.get(k), str) or actions[k] not in YESNO for k in ACTION_KEYS):
            raise ValueError("observation_actions_invalid")
        if out and item["t"] <= out[-1]["t"]:
            raise ValueError("observation_order_invalid")
        out.append({"t": item["t"], "visibility": {k: visibility[k] for k in VIS_KEYS},
                    "actions": {k: actions[k] for k in ACTION_KEYS},
                    "paired_action_disagreements": item.get("paired_action_disagreements")
                    if type(item.get("paired_action_disagreements")) is int and 0 <= item["paired_action_disagreements"] <= 5 else None})
    return out


def boundaries_only(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("paired_boundaries_required")
    out = []
    for answer in raw:
        if not isinstance(answer, dict) or any(not isinstance(answer.get(k), str) or answer[k] not in spec["enum"]
                                              for k, spec in BOUNDARY_SCHEMA["properties"].items()):
            raise ValueError("boundary_enum_invalid")
        out.append({k: answer[k] for k in BOUNDARY_SCHEMA["properties"]})
    return out


def critical_visible(row: dict) -> bool:
    v = row["visibility"]
    return all(v[k] == "clear" for k in REQUIRED if k != "visitor_hands_visibility") and v["visitor_hands_visibility"] in {"clear", "absent"}


def episodes(observations: list[dict]) -> list[dict]:
    """Uniform local predicate; source names, labels and clip length are absent."""
    groups: list[list[dict]] = []
    current = []
    for row in observations:
        eligible = (row["actions"]["cash_recount"] == "yes" and critical_visible(row)
                    and row["visibility"]["visitor_hands_visibility"] == "absent")
        if not eligible or (current and row["t"] - current[-1]["t"] > MAX_GAP_S):
            if current:
                groups.append(current)
                current = []
        if eligible:
            current.append(row)
    if current:
        groups.append(current)
    out = []
    for group in groups:
        before = [r for r in observations if r["t"] < group[0]["t"]]
        out.append({"start": group[0]["t"], "end": group[-1]["t"],
                    "duration_proxy_s": round(group[-1]["t"] - group[0]["t"], 6), "qualifying_samples": len(group),
                    "prior_visitor_clear_in_window": any(r["visibility"]["visitor_hands_visibility"] == "clear" for r in before),
                    "prior_cash_acceptance_seen_in_window": any(r["actions"]["cash_accepted"] == "yes" and critical_visible(r) for r in before),
                    "continuous_condition_proven": False})
    return out


def replay(document: dict) -> dict:
    """Project all observations without accepting labels or label timestamps."""
    if not isinstance(document, dict):
        raise ValueError("diagnostic_object_required")
    wrapped = isinstance(document.get("data"), dict)
    data = document["data"] if wrapped else document
    source_ok = (document.get("status") == "done" and document.get("model_source_profile_unchanged") is True) if wrapped else data.get("schema") == "forte.procedure-visible-probe.v1"
    policy = data.get("policy") if isinstance(data.get("policy"), dict) else {}
    paired = policy.get("local_action_phrasings") == 2
    fps = policy.get("decode_fps")
    policy_ok = paired and number(fps, 5, 5) and policy.get("sampling_proxy_only") is True
    raw = data.get("results") if isinstance(data.get("results"), list) else []
    rows = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not NAME.fullmatch(item["name"]):
            rows.append({})
            continue
        source_status = item.get("status") if item.get("status") in {"done", "failed", "abstained"} else "invalid"
        row = {"name": item["name"], "source_status": source_status,
               "source_done": source_status == "done", "coverage_complete": False,
               "windows": [], "episodes": [], "failure_codes": []}
        if not row["source_done"]:
            row["failure_codes"].append("source_failed_or_abstained")
        windows = item.get("windows")
        if not isinstance(windows, list) or not 1 <= len(windows) <= 8:
            row["failure_codes"].append("observed_windows_missing_or_invalid")
            windows = []
        for raw_window in windows:
            w = {"processed": False, "episodes": [], "failure_codes": []}
            try:
                if not isinstance(raw_window, dict) or raw_window.get("status") != "done":
                    raise ValueError("source_window_failed")
                start, end = raw_window.get("start"), raw_window.get("end")
                if not number(start, high=86400) or not number(end, high=86400) or end <= start:
                    raise ValueError("window_bounds_invalid")
                obs = project_observations(raw_window.get("observations"), start, end)
                boundary = boundaries_only(raw_window.get("boundaries"))
                coverage = sampled_coverage(obs, start, end, REQUIRED)
                known = sum(critical_visible(r) and r["actions"]["cash_recount"] != "unclear" for r in obs) / len(obs)
                complete = coverage["meets_90pct_sampling_proxy"] and known >= .9
                found = episodes(obs)
                row["episodes"].extend(found)
                w.update(processed=True, start=start, end=end, episodes=found, visibility=coverage,
                         counting_action_known_fraction=round(known, 6), coverage_complete=complete,
                         checks=window_checks(obs, boundary, start, end),
                         sampling_reliability={"samples": len(obs), "clear_context_samples": sum(critical_visible(r) for r in obs),
                             "action_counts": {k: dict(Counter(r["actions"][k] for r in obs)) for k in ACTION_KEYS},
                             "visibility_counts": {k: dict(Counter(r["visibility"][k] for r in obs)) for k in VIS_KEYS},
                             "paired_action_disagreement_histogram": dict(Counter(str(r["paired_action_disagreements"])
                                 for r in obs if r["paired_action_disagreements"] is not None))})
                if not complete:
                    w["failure_codes"].append("partial_visibility_or_action_coverage")
            except (ValueError, KeyError, TypeError):
                w["failure_codes"].append("window_projection_or_source_failure")
            row["windows"].append(w)
        row["coverage_complete"] = (row["source_done"] and bool(row["windows"])
                                    and all(w["processed"] and w.get("coverage_complete") for w in row["windows"]))
        if not row["coverage_complete"]:
            row["failure_codes"].append("video_partial_or_unmeasured_context_coverage")
        rows.append(row)
    return {"rows": rows, "source_complete": source_ok and policy_ok, "paired_consensus_policy_verified": paired,
            "sample_interval_s": 1 / fps if number(fps, 5, 5) else None,
            "source_success": source_ok, "sampling_policy_verified": policy_ok}


def negative_operating_point(labels: dict, measured: dict) -> dict:
    """The only fitted value uses negative episode durations, never positives."""
    negatives = {n for n, label in labels.items() if not label["violation"]}
    rows = [r for r in measured["rows"] if r.get("name") in negatives]
    durations = [e["duration_proxy_s"] for r in rows for e in r["episodes"]]
    maximum = max(durations, default=0.0)
    margin = measured["sample_interval_s"]
    threshold = round(maximum + margin, 6) if margin is not None else None
    names = [r["name"] for r in rows]
    complete = (len(negatives) == 19 and len(names) == 19 and set(names) == negatives
                and len(set(names)) == len(names) and measured["source_complete"]
                and all(r["coverage_complete"] for r in rows))
    reasons = [] if complete else ["not_all_19_negatives_have_complete_contextual_sampling_coverage"]
    return {"threshold_s": threshold, "comparison": "duration_proxy_s >= threshold_s",
            "maximum_observed_negative_episode_s": maximum if margin is not None else None,
            "one_sample_margin_s": margin, "negative_videos_expected": len(negatives),
            "negative_rows_recorded": len(rows), "negative_rows_with_complete_coverage": sum(r["coverage_complete"] for r in rows),
            "negative_episode_count": len(durations), "negative_episode_duration_distribution": {str(k): v for k, v in sorted(Counter(durations).items())},
            "calibration_complete": complete, "reasons": reasons,
            "labels_used_for": "negative_membership_only; no_positive_labels_or_times_in_operating_point",
            "calibration_scope": "in_sample_negative_envelope; zero_negative_fires_is_not_independent_validation",
            "fitted_parameters": 1 if threshold is not None else 0, "margin_tuned": False}


def counting_score(labels: dict, measured: dict, operating_point: dict) -> dict:
    rows = []
    threshold = operating_point["threshold_s"]
    for r in measured["rows"]:
        if "name" not in r:
            rows.append({})
            continue
        selected = [e for e in r["episodes"] if threshold is not None and e["duration_proxy_s"] >= threshold]
        failed = not r["coverage_complete"] or not measured["source_complete"] or threshold is None
        rows.append({"name": r["name"], "events": [{"type_id": TYPE, "t": e["start"]} for e in selected],
                     **({"error": "failed_or_partial_contextual_coverage"} if failed else {})})
    report = score_results(labels, rows, time_tolerance=1)
    gate = regression_gate(report, expected_count=40, expected_clean=19)
    if not operating_point["calibration_complete"]:
        gate["passed"] = False
        gate["reasons"].append("negative_calibration_incomplete")
    return {"summary": report["summary"], "gate": gate,
            "positive_score_passes_after_threshold_freeze": 1,
            "claim_scope": "diagnostic_coverage_denominator_only; not_whole_pipeline_bank_accuracy",
            "type_correctness_measured": False,
            "videos": [{"name": r["name"], "processed": r["processed"], "events": r.get("events", []),
                        "correct": r["correct"], "prediction": r["prediction"]} for r in report["results"]]}


def disabled_scores(labels: dict, measured: dict) -> dict:
    out = {}
    for kind, applicable in DISABLED.items():
        rows, signal_counts = [], Counter()
        for r in measured["rows"]:
            if "name" not in r:
                rows.append({})
                continue
            ev, assessable = [], r["source_done"] and bool(r["windows"]) and measured["source_complete"]
            for w in r["windows"]:
                check = w.get("checks", {}).get(kind)
                if not w["processed"] or not isinstance(check, dict):
                    assessable = False
                    continue
                signal_counts[check["status"]] += 1
                if check["status"] == "no_step_seen_with_sampling_proxy":
                    ev.append({"type_id": kind, "t": w["start"]})
                non_applicable = (check["paired_boundaries_agree"] and check["transaction_kind"] not in {applicable, "unclear"})
                complete = (check["whole_transaction_observed"] and check["paired_boundaries_agree"]
                            and check["required_cash_flow_observed"] and check["visibility"]["meets_90pct_sampling_proxy"]
                            and check["action_known_fraction"] >= .9)
                assessable = assessable and (non_applicable or complete)
            rows.append({"name": r["name"], "events": ev, **({"error": "failed_or_partial_procedure_coverage"} if not assessable else {})})
        scored = score_results(labels, rows, time_tolerance=1)
        out[kind] = {"summary": scored["summary"], "window_signal_counts": dict(signal_counts),
                     "gate": regression_gate(scored, expected_count=40, expected_clean=19),
                     "type_enablement_qualified": False, "type_correctness_measured": False,
                     "interpretation": "sampling_proxy_omission_signals_only; no_rule_or_threshold_change"}
    return out


def reliability_summary(labels: dict, measured: dict) -> dict:
    """Separate technical completion, perception coverage and valid samples."""
    out = {}
    for label_kind, polarity in (("negative", False), ("positive", True)):
        rows = [r for r in measured["rows"] if r.get("name") in labels and labels[r["name"]]["violation"] is polarity]
        windows = [w for r in rows for w in r["windows"]]
        valid = [w for w in windows if w["processed"]]
        samples = sum(w["sampling_reliability"]["samples"] for w in valid)
        clear = sum(w["sampling_reliability"]["clear_context_samples"] for w in valid)
        durations = [e["duration_proxy_s"] for r in rows for e in r["episodes"]]
        per_type = {}
        for kind in DISABLED:
            checks = [w["checks"][kind] for w in valid]
            per_type[kind] = {"window_status_counts": dict(Counter(c["status"] for c in checks)),
                              "whole_transaction_windows": sum(c["whole_transaction_observed"] for c in checks),
                              "paired_boundary_agreement_windows": sum(c["paired_boundaries_agree"] for c in checks),
                              "meets_sampled_visibility_proxy_windows": sum(c["visibility"]["meets_90pct_sampling_proxy"] for c in checks),
                              "action_known_fraction_90pct_windows": sum(c["action_known_fraction"] >= .9 for c in checks)}
        action_counts = {}
        for key in ACTION_KEYS:
            counter = Counter()
            for w in valid:
                counter.update(w["sampling_reliability"]["action_counts"][key])
            action_counts[key] = dict(counter)
        visibility_counts = {}
        for key in VIS_KEYS:
            counter = Counter()
            for w in valid:
                counter.update(w["sampling_reliability"]["visibility_counts"][key])
            visibility_counts[key] = dict(counter)
        disagreement_counts = Counter()
        for w in valid:
            disagreement_counts.update(w["sampling_reliability"]["paired_action_disagreement_histogram"])
        out[label_kind] = {"expected_videos": sum(lab["violation"] is polarity for lab in labels.values()),
                           "recorded_videos": len(rows), "source_status_counts": dict(Counter(r["source_status"] for r in rows)),
                           "windows_recorded": len(windows), "windows_with_valid_observations": len(valid),
                           "windows_failed_or_invalid": len(windows) - len(valid),
                           "videos_with_complete_contextual_coverage": sum(r["coverage_complete"] for r in rows),
                           "sampled_observations": samples, "clear_context_observations": clear,
                           "clear_context_observation_fraction": clear / samples if samples else None,
                           "action_counts_paired_consensus": action_counts, "visibility_counts_paired_consensus": visibility_counts,
                           "paired_action_disagreement_histogram": dict(sorted(disagreement_counts.items())),
                           "eligible_episode_count": len(durations), "eligible_episode_durations_s": durations,
                           "disabled_type_coverage": per_type}
    return out


def evaluate(document: dict, labels: dict) -> dict:
    measured = replay(document)  # No labels accepted by observation projection.
    labels = validate_labels(labels)
    operating_point = negative_operating_point(labels, measured)  # Frozen here.
    scored = counting_score(labels, measured, operating_point)  # One positive scoring pass.
    source_rows = [r for r in measured["rows"] if "name" in r]
    return {"schema": "forte.contextual-procedure-diagnostic.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "labels_sha256": json_hash(labels), "tool_sha256": file_hash(Path(__file__)),
            "procedure_coverage_helper_sha256": file_hash(Path(__file__).with_name("procedure_visible_probe.py")),
            "diagnostic_only": True, "inference_executed": False, "type_enablement_qualified": False,
            "production_change_authorized": False, "heldout_accuracy_claim": False, "cross_validation_claim": False,
            "source": {"source_guard_and_policy_verified": measured["source_complete"],
                       **{k: v for k, v in measured.items() if k not in {"rows", "source_complete"}}},
            "definition": {"action": "paired_cash_recount_yes", "counterparty": "paired_visitor_absent; no_visitor_present",
                           "required_clear": list(REQUIRED[:-1]), "counting_modality_resolved": False,
                           "duration": "last_qualifying_sample_minus_first_qualifying_sample; no_extrapolation",
                           "maximum_consecutive_sample_gap_s": MAX_GAP_S,
                           "unclear_samples_break_runs": True, "continuous_condition_proven": False},
            "operating_point": operating_point, "count_without_client": scored,
            "observation_reliability": reliability_summary(labels, measured),
            "disabled_type_signals": disabled_scores(labels, measured),
            "observations": [{"name": r["name"], "source_status": r["source_status"], "source_done": r["source_done"], "coverage_complete": r["coverage_complete"],
                              "failure_codes": r["failure_codes"], "episodes": r["episodes"],
                              "windows": [{k: v for k, v in w.items() if k != "checks"} for w in r["windows"]]} for r in source_rows],
            "limitations": ["5_fps_local_action_intervals_and_nearest_sample_visibility_are_proxies; between_sample_behavior_is_unobserved",
                            "state_selected_transaction_windows_do_not_exhaust_the_video; abstained_rows_are_unassessed",
                            "paired_visitor_absence_is_model_perception; no_person_tracking_or_cash_ownership_proof",
                            "all_negative_calibration_reuses_development_data; no_LOVO_or_augmented_non_target_qualification",
                            "typed_ground_truth_original_evidence_rule_grounding_and_API_deadline_checks_not_measured"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.resolve() in {args.report.resolve(), args.labels.resolve()}:
        raise ValueError("output_overlaps_input")
    report = evaluate(load_json(args.report), load_json(args.labels))
    report["source_report_sha256"] = file_hash(args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "threshold_s": report["operating_point"]["threshold_s"],
                      "calibration_complete": report["operating_point"]["calibration_complete"],
                      "summary": report["count_without_client"]["summary"], "type_enablement_qualified": False}, allow_nan=False))
    return 0 if report["operating_point"]["calibration_complete"] and report["count_without_client"]["gate"]["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "contextual_diagnostic_input_failed", "error_class": type(exc).__name__}))
        raise SystemExit(2) from None
