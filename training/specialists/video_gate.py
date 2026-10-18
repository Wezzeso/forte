"""Whole-video type promotion proof; binary-only reports cannot unlock types."""
from __future__ import annotations

from pathlib import Path

from .common import file_sha256, load_json, number, require, sha


def _types(events: list) -> set[str]:
    require(isinstance(events, list), "specialist_video_events_required")
    types = set()
    for event in events:
        require(isinstance(event, dict) and isinstance(event.get("type_id"), str)
                and bool(event["type_id"]), "specialist_video_type_invalid")
        require("t" not in event or number(event["t"]), "specialist_video_time_invalid")
        types.add(event["type_id"])
    return types


def _timed_matches(targets: list[float], predictions: list[float], tolerance: float) -> int:
    targets, predictions = sorted(targets), sorted(predictions)
    i, j, matched = 0, 0, 0
    while i < len(targets) and j < len(predictions):
        if predictions[j] < targets[i] - tolerance:
            j += 1
        elif predictions[j] > targets[i] + tolerance:
            i += 1
        else:
            matched, i, j = matched + 1, i + 1, j + 1
    return matched


def _label_events(label: dict) -> list[dict]:
    if "events" in label:
        return label["events"]
    return [{"type_id": label["type_id"], **({"t": label["t"]} if "t" in label else {})}] if label["violation"] else []


def _report(path: Path) -> tuple[dict, dict[str, dict]]:
    report = load_json(path)
    require(report.get("schema") == "forte.fast-evaluation.v2", "specialist_video_report_schema")
    integrity, provenance = report.get("integrity", {}), report.get("provenance", {})
    require(integrity.get("complete") is True and not any(integrity.get(field) for field in
            ("missing", "unexpected", "duplicates", "invalid_rows")), "specialist_video_report_incomplete")
    require(provenance.get("mode") == "live_analyzer_inference"
            and provenance.get("inference_executed") is True, "specialist_video_live_inference_required")
    require(provenance.get("data_hash_scope") == "full_file_sha256", "specialist_video_full_media_hashes_required")
    require(sha(provenance.get("input_inventory_sha256")) and sha(report.get("labels_sha256")),
            "specialist_video_inputs_unbound")
    rows = report.get("results")
    require(isinstance(rows, list) and len(rows) == 40, "specialist_video_full_forty_required")
    indexed = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"]
                and row["name"] not in indexed, "specialist_video_identity_invalid")
        scoring_reasons = {"missed_violation", "false_alarm", "type_mismatch",
                           "event_missing_or_outside_time_tolerance", "timecode_outside_tolerance"}
        require(row.get("processed") is True and not row.get("error")
                and all(reason in scoring_reasons for reason in row.get("error_reasons", []))
                and type(row.get("prediction")) is bool, "specialist_video_inference_failure")
        require(isinstance(row.get("label"), dict) and type(row["label"].get("violation")) is bool,
                "specialist_video_labels_required")
        label = row["label"]
        if label["violation"]:
            if "events" in label:
                truth = _types(label["events"])
            else:
                truth = _types([{"type_id": label.get("type_id")}])
            require(bool(truth), "specialist_video_positive_type_labels_required")
        else:
            truth = _types(label.get("events", []))
            require(not truth and not label.get("type_id"), "specialist_video_clean_label_conflict")
        predicted = _types(row.get("events"))
        if "duration" in row or "duration_s" in row:
            duration = row.get("duration_s", row.get("duration"))
            require(number(duration) and all("t" not in event or event["t"] <= duration
                                            for event in row["events"]), "specialist_video_time_outside_duration")
        require(row["prediction"] == bool(predicted), "specialist_video_prediction_conflict")
        indexed[row["name"]] = {"truth": truth, "predicted": predicted, "label": label,
                                 "clean": not label["violation"], "events": row["events"]}
    clean = sum(row["clean"] for row in indexed.values())
    require(clean == 19, "specialist_video_nineteen_clean_required")
    inputs = provenance.get("inputs")
    require(isinstance(inputs, list) and len(inputs) == 40
            and len({item.get("name") for item in inputs}) == 40
            and {item.get("name") for item in inputs} == set(indexed)
            and all(sha(item.get("sha256")) for item in inputs), "specialist_video_inventory_coverage")
    return report, indexed


def approve_video_types(candidate_path: Path, baseline_path: Path, requested_types: list[str], run: dict) -> dict:
    require(bool(requested_types) and len(set(requested_types)) == len(requested_types)
            and all(isinstance(value, str) and value for value in requested_types),
            "specialist_video_requested_types_invalid")
    candidate, rows = _report(candidate_path)
    baseline, old_rows = _report(baseline_path)
    cp, bp = candidate["provenance"], baseline["provenance"]
    require(cp.get("evaluation_mode") == "candidate_only"
            and set(cp.get("provisional_types", [])) == set(requested_types),
            "specialist_video_candidate_scope_mismatch")
    require(sha(cp.get("analyzer_policy_sha256")), "specialist_video_analyzer_policy_required")
    require(isinstance(candidate.get("config"), dict), "specialist_video_analyzer_config_required")
    from forte.fast.policy import analyzer_policy_sha256
    require(analyzer_policy_sha256(candidate["config"]) == cp["analyzer_policy_sha256"],
            "specialist_video_analyzer_policy_mismatch")
    tolerance = candidate.get("scoring", {}).get("time_tolerance_s")
    require(number(tolerance, 0, 2) and baseline.get("scoring", {}).get("time_tolerance_s") == tolerance,
            "specialist_video_time_protocol_required")
    identity = cp.get("specialist_identity", {})
    require(all(identity.get(key) == run[key] for key in
                ("checkpoint_sha256", "dataset_lock_sha256", "config_sha256"))
            and sha(identity.get("gate_sha256")), "specialist_video_artifact_identity_mismatch")
    require(candidate["labels_sha256"] == baseline["labels_sha256"]
            and cp["input_inventory_sha256"] == bp["input_inventory_sha256"]
            and set(rows) == set(old_rows)
            and all(row["label"] == old_rows[name]["label"] for name, row in rows.items()),
            "specialist_video_baseline_inputs_mismatch")
    require({item["name"]: item["sha256"] for item in cp["inputs"]}
            == {item["name"]: item["sha256"] for item in bp["inputs"]},
            "specialist_video_baseline_media_mismatch")
    # Zero alarms on the entire clean set applies to every emitted type.
    require(all(not row["predicted"] for row in rows.values() if row["clean"]),
            "specialist_video_clean_false_alarm")
    type_metrics, event_metrics = {}, {}
    for type_id in requested_types:
        positives = {name for name, row in rows.items() if type_id in row["truth"]}
        predictions = {name for name, row in rows.items() if type_id in row["predicted"]}
        old_predictions = {name for name, row in old_rows.items() if type_id in row["predicted"]}
        tp, fp, fn = len(positives & predictions), len(predictions - positives), len(positives - predictions)
        old_tp = len(positives & old_predictions)
        require(bool(positives), "specialist_video_type_positive_support_missing")
        recall = tp / len(positives)
        require(recall >= .95 and fp == 0 and tp > old_tp,
                "specialist_video_type_recall_or_gain_failed")
        expected_events, event_tp, event_fp = 0, 0, 0
        for name, row in rows.items():
            targets = [event for event in _label_events(row["label"]) if event["type_id"] == type_id]
            found = [event for event in row["events"] if event["type_id"] == type_id]
            previous = [event for event in old_rows[name]["events"] if event["type_id"] == type_id]
            require(all(number(event.get("t")) for event in targets + found + previous),
                    "specialist_video_explicit_timestamps_required")
            matches = _timed_matches([event["t"] for event in targets], [event["t"] for event in found], tolerance)
            old_matches = _timed_matches([event["t"] for event in targets], [event["t"] for event in previous], tolerance)
            require(matches >= old_matches, "specialist_video_baseline_event_regression")
            expected_events += len(targets)
            event_tp += matches
            event_fp += len(found) - matches
        require(expected_events > 0 and event_tp / expected_events >= .95 and event_fp == 0,
                "specialist_video_event_precision_or_recall_failed")
        event_metrics[type_id] = {"tp": event_tp, "fp": event_fp, "fn": expected_events - event_tp,
                                   "recall": event_tp / expected_events, "time_tolerance_s": tolerance}
        type_metrics[type_id] = {"tp": tp, "fp": fp, "fn": fn, "recall": recall,
                                 "baseline_tp": old_tp, "positive_videos": len(positives)}
    # Preserve correct baseline verdicts; a gain in one type cannot hide a
    # new miss elsewhere in the same whole-video evaluation.
    require(all(bool(row["predicted"]) == bool(row["truth"])
                or bool(old_rows[name]["predicted"]) != bool(old_rows[name]["truth"])
                for name, row in rows.items()), "specialist_video_baseline_verdict_regression")
    return {"approved_types": sorted(requested_types), "video_validation_sha256": file_sha256(candidate_path),
            "baseline_video_validation_sha256": file_sha256(baseline_path),
            "video_type_metrics": type_metrics, "video_clean_count": 19,
            "video_event_metrics": event_metrics,
            "video_input_inventory_sha256": cp["input_inventory_sha256"],
            "analyzer_policy_sha256": cp["analyzer_policy_sha256"],
            "video_labels_sha256": candidate["labels_sha256"]}
