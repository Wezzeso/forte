"""Deterministic local diagnostics, with no invented organizer score."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .audit import audit_split, sample_id
from .io import digest


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def f1(tp: int, fp: int, fn: int) -> float | None:
    return ratio(2 * tp, 2 * tp + fp + fn)


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def reviewed(row: dict) -> bool:
    return row.get("review_status") == "human_reviewed" and bool(row.get("reviewer_id")) and bool(row.get("reviewed_at"))


def event_type(event: dict) -> str:
    value = event.get("event_type_id", event.get("type"))
    return value if isinstance(value, str) else ""


def anchor(event: dict) -> float | None:
    return number(event.get("anchor_sec", event.get("timecode_sec", event.get("timestamp_sec"))))


def event_list(row: dict) -> list[dict]:
    value = row.get("events", row.get("violations", []))
    if not isinstance(value, list) or any(not isinstance(event, dict) for event in value):
        raise ValueError("events must be an array of objects")
    return value


def match_events(truth: list[dict], predictions: list[dict], tolerance_sec: float) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, minimum-total-anchor-error, type-aware 1:1 matching.

    Successive shortest augmenting paths handle ambiguous overlapping events.
    Reversed flow edges allow correcting an earlier match; greedy nearest
    matching would miss a valid second event in several common boundary cases.
    """
    if not math.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("tolerance_sec must be finite and nonnegative")
    n, m = len(truth), len(predictions)
    source, sink = n + m, n + m + 1
    graph: list[list[list]] = [[] for _ in range(sink + 1)]

    def edge(u: int, v: int, cost: float) -> list:
        forward = [v, len(graph[v]), 1, cost]
        reverse = [u, len(graph[u]), 0, -cost]
        graph[u].append(forward)
        graph[v].append(reverse)
        return forward

    for i in range(n):
        edge(source, i, 0.0)
    for j in range(m):
        edge(n + j, sink, 0.0)
    references = []
    for i, expected in enumerate(truth):
        a = anchor(expected)
        for j, predicted in enumerate(predictions):
            b = anchor(predicted)
            if a is None or b is None or not event_type(expected) or event_type(expected) != event_type(predicted):
                continue
            error = abs(a - b)
            if error <= tolerance_sec:
                references.append((i, j, error, edge(i, n + j, error)))
    while True:
        distance = [math.inf] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0.0
        for _ in range(len(graph) - 1):
            changed = False
            for u, edges in enumerate(graph):
                if not math.isfinite(distance[u]):
                    continue
                for k, item in enumerate(edges):
                    v, _, capacity, cost = item
                    if capacity and distance[u] + cost < distance[v] - 1e-12:
                        distance[v] = distance[u] + cost
                        previous[v] = (u, k)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        v = sink
        while v != source:
            u, k = previous[v]  # type: ignore[misc]
            item = graph[u][k]
            item[2] -= 1
            graph[v][item[1]][2] += 1
            v = u
    return [(i, j, error) for i, j, error, item in references if item[2] == 0]


def distribution(values: list[float]) -> dict:
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values), "mean": sum(values) / len(values),
        "p50": values[max(0, math.ceil(0.5 * len(values)) - 1)],
        "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)], "max": values[-1],
    }


def _indexed(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        sid = sample_id(row)
        if sid in indexed:
            raise ValueError("duplicate sample_id in evaluation input")
        indexed[sid] = row
    return indexed


def _completed(row: dict, duration: float | None) -> tuple[bool, str]:
    if row.get("status") != "done":
        return False, "not_done"
    verdict = row.get("verdict")
    events = event_list(row)
    if verdict not in {"violation", "no_violation"} or (verdict == "violation") != bool(events):
        return False, "invalid_verdict_events"
    for event in events:
        timestamp = anchor(event)
        if not event_type(event) or timestamp is None or timestamp < 0 or (duration is not None and timestamp > duration):
            return False, "invalid_event"
    return True, "done"


def evaluate(
    truth_records: list[dict], prediction_records: list[dict], *, metadata: dict | None = None,
    split_manifest: dict | None = None, split: str = "validation", tolerances_sec: tuple[float, ...] = (0.5, 1.0, 2.0),
    deadline_sec: float = 300.0, verified_rule_ids: set[str] | None = None,
) -> dict:
    """Return aggregate-only report. No private descriptions or sample names exit.

    Human review requires review_status, reviewer_id and reviewed_at. Missing or
    invalid/failed predictions remain in completion and effective-recall
    denominators. They are never silently recast as no_violation.
    """
    if not math.isfinite(deadline_sec) or deadline_sec <= 0 or not tolerances_sec:
        raise ValueError("invalid diagnostic configuration")
    tolerances_sec = tuple(sorted(set(float(t) for t in tolerances_sec)))
    if any(not math.isfinite(t) or t < 0 for t in tolerances_sec):
        raise ValueError("invalid timestamp tolerance")
    truth_by_id, predictions = _indexed(truth_records), _indexed(prediction_records)
    audit = audit_split(split_manifest) if split_manifest is not None else None
    if split_manifest is not None:
        selected_ids = {sample_id(row) for row in split_manifest["records"] if row.get("split") == split}
        truth_missing = selected_ids - truth_by_id.keys()
    else:
        selected_ids, truth_missing = set(truth_by_id), set()
    selected = sorted(selected_ids)
    counts: Counter[str] = Counter()
    confusion = Counter({key: 0 for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative")})
    deadline_confusion = Counter({key: 0 for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative")})
    event_counts = {t: Counter({key: 0 for key in ("tp", "fp", "fn")}) for t in tolerances_sec}
    types: dict[str, Counter] = defaultdict(Counter)
    errors: dict[float, list[float]] = {t: [] for t in tolerances_sec}
    grounding: Counter[str] = Counter()
    latencies: list[float] = []
    status_counts: Counter[str] = Counter()
    primary = min(tolerances_sec, key=lambda t: abs(t - 1.0))
    scenario_counts: dict[str, Counter] = defaultdict(Counter)
    for sid in selected:
        expected = truth_by_id.get(sid, {})
        predicted = predictions.get(sid, {})
        duration = number(expected.get("duration_sec"))
        try:
            complete, reason = _completed(predicted, duration)
        except ValueError:
            complete, reason = False, "invalid_event_structure"
        status_counts[reason] += 1
        counts["completed"] += complete
        counts["missing_prediction"] += not bool(predicted)
        elapsed = number(predicted.get("upload_start_to_result_sec", predicted.get("elapsed_sec")))
        on_time = complete and elapsed is not None and 0 <= elapsed <= deadline_sec
        if complete and elapsed is not None and elapsed >= 0:
            latencies.append(elapsed)
            counts["timed_completed"] += 1
            counts["on_time_completed"] += elapsed <= deadline_sec
            counts["late_completed"] += elapsed > deadline_sec
        elif complete:
            counts["completion_timing_unknown"] += 1
        is_reviewed = reviewed(expected) and expected.get("verdict") in {"violation", "no_violation"}
        if not is_reviewed:
            counts["unreviewed_or_unresolved_truth"] += 1
            continue
        counts["reviewed"] += 1
        positive = expected["verdict"] == "violation"
        counts["reviewed_positive"] += positive
        counts["reviewed_negative"] += not positive
        truth_events = event_list(expected)
        if positive != bool(truth_events):
            counts["inconsistent_truth"] += 1
        invalid_truth_events = [e for e in truth_events if not event_type(e) or anchor(e) is None or anchor(e) < 0 or (duration is not None and anchor(e) > duration)]
        counts["invalid_truth_events"] += len(invalid_truth_events)
        truth_events = [e for e in truth_events if e not in invalid_truth_events]
        pred_events = event_list(predicted) if complete else []
        counts["reviewed_completed"] += complete
        if not complete:
            confusion["unprocessed_positive" if positive else "unprocessed_negative"] += 1
        else:
            pred_positive = predicted["verdict"] == "violation"
            confusion[("tp" if pred_positive else "fn") if positive else ("fp" if pred_positive else "tn")] += 1
        if not on_time:
            # Late or unknown-timing jobs do not earn organizer-cutoff success.
            deadline_confusion["unprocessed_positive" if positive else "unprocessed_negative"] += 1
        else:
            pred_positive = predicted["verdict"] == "violation"
            deadline_confusion[("tp" if pred_positive else "fn") if positive else ("fp" if pred_positive else "tn")] += 1
        # Scenario tags are controlled labels, not private free text. Unknown tags are ignored.
        allowed_scenarios = {"new_angle", "brief_action", "window_boundary", "occlusion", "normal_lookalike", "omission", "multiple_events", "unsupported_fact", "media_prompt_injection"}
        for tag in set(expected.get("scenario_tags", [])) & allowed_scenarios:
            scenario_counts[tag]["total"] += 1
            scenario_counts[tag]["completed"] += complete
            scenario_counts[tag]["correct_verdict"] += complete and predicted.get("verdict") == expected["verdict"]
        for tolerance in tolerances_sec:
            matches = match_events(truth_events, pred_events, tolerance)
            event_counts[tolerance].update(tp=len(matches), fp=len(pred_events) - len(matches), fn=len(truth_events) - len(matches))
            errors[tolerance].extend(error for i, _, error in matches if truth_events[i].get("anchor_reliable") is not False)
            if tolerance != primary:
                continue
            matched_truth = {i for i, _, _ in matches}
            matched_pred = {j for _, j, _ in matches}
            for i, event in enumerate(truth_events):
                types[event_type(event)]["tp" if i in matched_truth else "fn"] += 1
            for j, event in enumerate(pred_events):
                if j not in matched_pred:
                    types[event_type(event)]["fp"] += 1
            for i, j, _ in matches:
                expected_event, predicted_event = truth_events[i], pred_events[j]
                expected_rule = expected_event.get("rule_id")
                predicted_rule = predicted_event.get("rule_id")
                if expected_rule:
                    grounding["reviewed_rule_mappings"] += 1
                    grounding["correct_rule_mappings" if expected_rule == predicted_rule else "wrong_or_missing_rule_mappings"] += 1
                else:
                    grounding["unresolved_truth_rule_mappings"] += 1
                review = predicted_event.get("local_review", {})
                if isinstance(review, dict) and reviewed(review) and isinstance(review.get("evidence_supported"), bool):
                    grounding["evidence_reviewed"] += 1
                    grounding["evidence_supported" if review["evidence_supported"] else "evidence_unsupported"] += 1
                else:
                    grounding["evidence_unreviewed"] += 1
        # Catalog validity covers all predictions, including unmatched false alarms.
        for event in pred_events:
            grounding["predicted_events"] += 1
            if verified_rule_ids is None:
                grounding["catalog_verification_unmeasured"] += 1
            elif event.get("rule_id") not in verified_rule_ids:
                grounding["unverified_or_missing_rule_ids"] += 1
            else:
                grounding["verified_catalog_rule_ids"] += 1
    tp, fp, fn, tn = (confusion[key] for key in ("tp", "fp", "fn", "tn"))
    diagnostics = []
    for tolerance, values in event_counts.items():
        etp, efp, efn = (values[key] for key in ("tp", "fp", "fn"))
        diagnostics.append({"tolerance_sec": tolerance, **dict(values), "precision": ratio(etp, etp + efp), "recall": ratio(etp, etp + efn), "f1": f1(etp, efp, efn), "matched_anchor_error_sec": distribution(errors[tolerance])})
    metadata = dict(metadata or {})
    return {
        "schema_version": "forte.evaluation.v1",
        "claim_scope": "synthetic_harness_validation" if metadata.get("inference_kind") == "synthetic" else "local_reviewed_diagnostics_only" if metadata.get("inference_kind") == "real_local" and counts["reviewed"] else "evaluation_incomplete_no_accuracy_claim",
        "protocol": {"split": split, "timestamp_tolerances_sec": list(tolerances_sec), "primary_tolerance_sec": primary, "deadline_sec": deadline_sec, "timing_origin": "upload_start", "official_scoring_formula": None},
        "provenance": {key: metadata.get(key) for key in ("model_id", "model_revision", "adapter_sha256", "inference_kind", "conditions_sha256", "rules_sha256", "taxonomy_sha256", "final_check_used_for_tuning")},
        "input_set_sha256": digest(selected), "truth_records_sha256": digest([truth_by_id.get(sid, {}) for sid in selected]),
        "split_audit": audit,
        "ground_truth": {"required_samples": len(selected), "missing_truth": len(truth_missing), "human_reviewed_resolved": counts["reviewed"], "positive": counts["reviewed_positive"], "negative": counts["reviewed_negative"], "unreviewed_or_unresolved": counts["unreviewed_or_unresolved_truth"], "inconsistent_verdict_events": counts["inconsistent_truth"], "invalid_events": counts["invalid_truth_events"]},
        "completion": {"required": len(selected), "completed": counts["completed"], "unprocessed": len(selected) - counts["completed"], "unprocessed_at_deadline_including_unknown_timing": len(selected) - counts["on_time_completed"], "missing_predictions": counts["missing_prediction"], "unexpected_predictions": len(predictions.keys() - selected_ids), "rate": ratio(counts["completed"], len(selected)), "on_time_completed": counts["on_time_completed"], "late_completed": counts["late_completed"], "timing_unknown": counts["completion_timing_unknown"], "deadline_completion_rate": ratio(counts["on_time_completed"], len(selected)), "status_counts": dict(sorted(status_counts.items())), "upload_start_to_result_sec": distribution(latencies)},
        "whole_video": {**dict(confusion), "confusion_denominator_completed_reviewed": counts["reviewed_completed"], "required_reviewed_denominator": counts["reviewed"], "precision_completed": ratio(tp, tp + fp), "recall_completed": ratio(tp, tp + fn), "f1_completed": f1(tp, fp, fn), "effective_positive_recall_including_failures": ratio(tp, tp + fn + confusion["unprocessed_positive"]), "effective_correct_rate_including_failures": ratio(tp + tn, counts["reviewed"])},
        "whole_video_within_deadline": {**dict(deadline_confusion), "required_reviewed_denominator": counts["reviewed"], "effective_positive_recall_including_failures": ratio(deadline_confusion["tp"], counts["reviewed_positive"]), "effective_correct_rate_including_failures": ratio(deadline_confusion["tp"] + deadline_confusion["tn"], counts["reviewed"]), "unknown_timing_counts_as_unprocessed": True},
        "events": {"primary_tolerance_sec": primary, "by_tolerance": diagnostics, "per_type": {kind: {"tp": values["tp"], "fp": values["fp"], "fn": values["fn"]} for kind, values in sorted(types.items())}, "matching": "maximum_cardinality_then_minimum_anchor_error; same type; one-to-one", "failed_jobs_contribute_ground_truth_event_fn": True},
        "grounding_and_evidence": dict(sorted(grounding.items())),
        "robustness": {tag: dict(sorted(values.items())) for tag, values in sorted(scenario_counts.items())},
        "limitations": ["Internal diagnostics are not the organizer's unpublished score.", "Small grouped samples cannot establish production accuracy or reliable tail latency.", "Image correspondence and action support require private local human review; matching timestamps does not establish either.", "A new camera angle requires genuinely different footage, not rotation or cropping.", "Failures remain in completion/effective-recall denominators and are never negative verdicts."],
    }
