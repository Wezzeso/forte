"""CPU-only scoring, regression gates, and group split planning.

This module never loads media or a model. A replayed report is evidence about
that recorded run only. A split plan is not a cross-validation result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_labels(payload: Any) -> dict[str, dict]:
    labels = payload.get("labels", payload) if isinstance(payload, dict) else payload
    if not isinstance(labels, dict) or not labels:
        raise ValueError("labels must be a non-empty name-to-label mapping")
    for name, label in labels.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(label, dict):
            raise ValueError("every label needs a non-empty clip name and an object")
        if type(label.get("violation")) is not bool:
            raise ValueError(f"{name}: violation must be a JSON boolean")
        if label["violation"] and "t" in label and (not _number(label["t"]) or label["t"] < 0):
            raise ValueError(f"{name}: t must be finite, non-negative seconds")
        if "type_id" in label and (not isinstance(label["type_id"], str) or not label["type_id"]):
            raise ValueError(f"{name}: type_id must be a non-empty string")
        if "events" in label:
            if not isinstance(label["events"], list):
                raise ValueError(f"{name}: events must be a list")
            for event in label["events"]:
                if (not isinstance(event, dict) or not isinstance(event.get("type_id"), str)
                        or not event["type_id"] or not _number(event.get("t")) or event["t"] < 0):
                    raise ValueError(f"{name}: labeled events need type_id and non-negative t")
            if bool(label["events"]) != label["violation"]:
                raise ValueError(f"{name}: labeled events contradict the binary label")
        # Old label files may contain t=0/null on clean videos; it is ignored.
        if not label["violation"] and label.get("type_id"):
            raise ValueError(f"{name}: a clean label cannot have an event type")
    return labels


def _event_scores(row: dict, label: dict, tolerance: float) -> None:
    """Match type labels before scoring time; never credit a wrong type's time."""
    events = row["events"]
    expected = label.get("events")
    typed = expected is not None or "type_id" in label
    if expected is None and "type_id" in label:
        expected = [{"type_id": label["type_id"], **({"t": label["t"]} if "t" in label else {})}]
    if typed:
        expected = expected or []
        row["type_correct"] = Counter(e["type_id"] for e in events) == Counter(e["type_id"] for e in expected)
        if not row["type_correct"]:
            row["error_reasons"].append("type_mismatch")
        # Maximum-cardinality one-to-one matching within the tolerance; sorting
        # makes the interval matching optimal for each type in one dimension.
        errors = []
        matched = 0
        timed = [e for e in expected if "t" in e]
        for kind in sorted({e["type_id"] for e in timed}):
            targets = sorted(e["t"] for e in timed if e["type_id"] == kind)
            predictions = sorted(e["t"] for e in events if e["type_id"] == kind)
            # Dynamic programming: maximize matches then minimize total error.
            dp = {(0, 0): (0, 0.0, [])}
            for i in range(len(targets) + 1):
                for j in range(len(predictions) + 1):
                    if (i, j) == (0, 0):
                        continue
                    candidates = []
                    if i:
                        candidates.append(dp[i - 1, j])
                    if j:
                        candidates.append(dp[i, j - 1])
                    if i and j:
                        error = abs(targets[i - 1] - predictions[j - 1])
                        if error <= tolerance:
                            count, total, chosen = dp[i - 1, j - 1]
                            candidates.append((count + 1, total + error, chosen + [error]))
                    dp[i, j] = min(candidates, key=lambda x: (-x[0], x[1]))
            count, _, chosen = dp[len(targets), len(predictions)]
            matched += count
            errors.extend(chosen)
        if timed:
            row["event_time_labels"] = len(timed)
            row["event_time_matches"] = matched
            row["event_time_errors"] = errors
            row["event_correct"] = row["type_correct"] and matched == len(timed)
            if matched < len(timed):
                row["error_reasons"].append("event_missing_or_outside_time_tolerance")
        # Keep legacy single-time metric, using the corresponding type only.
        if label["violation"] and "t" in label:
            candidates = [e for e in events if e["type_id"] == label["type_id"]] if "type_id" in label else events
            if candidates:
                row["timecode_err"] = min(abs(e["t"] - label["t"]) for e in candidates)
    elif label["violation"] and "t" in label and events:
        row["timecode_err"] = min(abs(e["t"] - label["t"]) for e in events)
    if "timecode_err" in row and row["timecode_err"] > tolerance:
        row["error_reasons"].append("timecode_outside_tolerance")


def score_results(labels: dict, raw_results: list[dict], *, expected_names: list[str] | None = None,
                  time_tolerance: float = 2.0) -> dict:
    labels = validate_labels(labels)
    if not _number(time_tolerance) or time_tolerance < 0:
        raise ValueError("time_tolerance must be finite and non-negative")
    names = sorted(labels if expected_names is None else expected_names)
    if not names or len(names) != len(set(names)) or set(names) - set(labels):
        raise ValueError("expected_names must be unique known label names")
    if not isinstance(raw_results, list):
        raise ValueError("report.results must be a list of per-video results, not an aggregate summary")
    by_name: dict[str, dict] = {}
    duplicate_names = []
    invalid_rows = 0
    for row in raw_results:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            invalid_rows += 1
            continue
        if row["name"] in by_name:
            duplicate_names.append(row["name"])
        by_name[row["name"]] = row
    unexpected = sorted(set(by_name) - set(names))
    missing = sorted(set(names) - set(by_name))
    results = []
    for name in names:
        raw = by_name.get(name, {"name": name, "error": "missing_result"})
        # Do not trust scores or embedded labels supplied by a cached report.
        row = {k: v for k, v in raw.items() if k not in {
            "label", "correct", "prediction", "error_reasons", "type_correct", "event_correct",
            "timecode_err", "event_time_labels", "event_time_matches", "event_time_errors"}}
        row.update(label=labels[name], error_reasons=[])
        events = raw.get("events")
        error = raw.get("error") or (f"status_{raw['status']}" if raw.get("status") in {"failed", "processing"} else None)
        if name in duplicate_names:
            error = "duplicate_result"
        if not error and (not isinstance(events, list) or any(
                not isinstance(e, dict) or not isinstance(e.get("type_id"), str) or not e["type_id"]
                or not _number(e.get("t")) or e["t"] < 0 for e in events)):
            error = "invalid_or_missing_events"
        row["processed"] = error is None
        row["correct"] = False
        row["prediction"] = None
        if error:
            row["error"] = str(error)
            row["error_reasons"].append("processing_failure:" + str(error))
        else:
            row.pop("error", None)
            row["prediction"] = bool(events)
            row["correct"] = bool(events) == labels[name]["violation"]
            if not row["correct"]:
                row["error_reasons"].append("missed_violation" if labels[name]["violation"] else "false_alarm")
            _event_scores(row, labels[name], time_tolerance)
        results.append(row)
    processed = [r for r in results if r["processed"]]
    positive = sum(labels[n]["violation"] for n in names)
    tp = sum(r["label"]["violation"] and r["prediction"] for r in processed)
    fn = sum(r["label"]["violation"] and not r["prediction"] for r in processed)
    fp = sum(not r["label"]["violation"] and r["prediction"] for r in processed)
    tn = sum(not r["label"]["violation"] and not r["prediction"] for r in processed)
    errs = [r["timecode_err"] for r in results if "timecode_err" in r]
    walls = [r["wall_s"] for r in processed if _number(r.get("wall_s")) and r["wall_s"] >= 0]
    typed_names = [n for n in names if "type_id" in labels[n] or "events" in labels[n]]
    time_names = [n for n in names if labels[n]["violation"] and ("t" in labels[n] or any("t" in e for e in labels[n].get("events", [])))]
    summary = {
        "n": len(names), "processed": len(processed), "failed": len(results) - len(processed),
        "correct": sum(r["correct"] for r in results),
        "accuracy": sum(r["correct"] for r in results) / len(names),
        "positive": positive, "negative": len(names) - positive,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "failed_positive": sum(r["label"]["violation"] and not r["processed"] for r in results),
        "failed_negative": sum(not r["label"]["violation"] and not r["processed"] for r in results),
        "recall": tp / positive if positive else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "type_labeled": len(typed_names), "type_correct": sum(r.get("type_correct") is True for r in results),
        "time_labeled_videos": len(time_names), "timecode_scored": len(errs),
        "timecode_err_median": statistics.median(errs) if errs else None,
        "timecode_within_2s": sum(e <= 2 for e in errs),
        "event_time_labels": sum(len(labels[n]["events"]) if "events" in labels[n] else int("type_id" in labels[n]) for n in time_names),
        "event_time_matches": sum(r.get("event_time_matches", 0) for r in results),
        "wall_max": max(walls) if walls else None, "wall_median": statistics.median(walls) if walls else None,
        "wall_scored": len(walls),
    }
    return {"summary": summary, "results": results,
            "integrity": {"complete": not (missing or unexpected or duplicate_names or invalid_rows),
                          "missing": missing, "unexpected": unexpected,
                          "duplicates": sorted(set(duplicate_names)), "invalid_rows": invalid_rows},
            "scoring": {"accuracy_denominator": "all_expected_videos_including_failures",
                        "confusion_matrix": "successfully_processed_only; failures_reported_separately",
                        "recall_denominator": "all_expected_positive_videos_including_failures",
                        "type_time_scope": "only_explicit_labels; no_inferred_ground_truth",
                        "time_tolerance_s": time_tolerance},
            "labels_sha256": json_hash({n: labels[n] for n in names})}


def regression_gate(report: dict, *, expected_count: int | None = None, expected_clean: int | None = None,
                    min_accuracy: float | None = None, max_wall_s: float | None = None,
                    baseline: dict | None = None, require_type_time: bool = False) -> dict:
    summary = report["summary"]
    reasons = []
    if not report["integrity"]["complete"]:
        reasons.append("result_names_not_exact")
    if summary["failed"]:
        reasons.append("processing_failures")
    if summary["fp"]:
        reasons.append("false_alarms")
    if expected_count is not None and summary["n"] != expected_count:
        reasons.append("unexpected_video_count")
    if expected_clean is not None and summary["negative"] != expected_clean:
        reasons.append("unexpected_clean_count")
    if min_accuracy is not None:
        if not _number(min_accuracy) or not 0 <= min_accuracy <= 1:
            raise ValueError("min_accuracy must be between 0 and 1")
        if summary["accuracy"] < min_accuracy:
            reasons.append("accuracy_below_minimum")
    if max_wall_s is not None:
        if not _number(max_wall_s) or max_wall_s <= 0:
            raise ValueError("max_wall_s must be positive")
        if summary["wall_scored"] != summary["n"]:
            reasons.append("latency_missing")
        if summary["wall_max"] is not None and summary["wall_max"] > max_wall_s:
            reasons.append("latency_budget_exceeded")
    if require_type_time and any(any(reason in {"type_mismatch", "event_missing_or_outside_time_tolerance",
                                              "timecode_outside_tolerance"} for reason in r["error_reasons"])
                                 for r in report["results"]):
        reasons.append("type_or_time_regression")
    regressions = []
    if baseline is not None:
        if (not baseline["integrity"]["complete"] or baseline["summary"]["failed"]
                or baseline["labels_sha256"] != report["labels_sha256"]):
            reasons.append("baseline_not_complete_or_labels_differ")
        else:
            previous = {r["name"]: r for r in baseline["results"]}
            regressions = [r["name"] for r in report["results"] if previous[r["name"]]["correct"] and not r["correct"]]
            if regressions:
                reasons.append("previously_correct_video_regressed")
            if summary["accuracy"] < baseline["summary"]["accuracy"]:
                reasons.append("accuracy_below_baseline")
    return {"passed": not reasons, "reasons": reasons, "regressed_videos": regressions,
            "scope": "recorded_run_only; no_unseen_accuracy_or_cross_validation_claim"}


def grouped_folds(manifest: dict, *, n_splits: int = 5, seed: int = 0) -> dict:
    """Build a reproducible plan; explicit source/group IDs keep variants together."""
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("source") or not provenance.get("grouping_method"):
        raise ValueError("manifest needs provenance.source and provenance.grouping_method")
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("manifest.clips must be a non-empty list")
    names: set[str] = set()
    identities: dict[tuple[str, str], str] = {}
    group_roles: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    holdout = []
    for clip in clips:
        if not isinstance(clip, dict) or any(not isinstance(clip.get(k), str) or not clip[k]
                                             for k in ("name", "group_id", "source_id")):
            raise ValueError("each clip needs name, group_id and source_id")
        name, group = clip["name"], clip["group_id"]
        if name in names:
            raise ValueError(f"duplicate clip name: {name}")
        names.add(name)
        role = clip.get("role", "development")
        if role not in {"development", "holdout"}:
            raise ValueError(f"{name}: role must be development or holdout")
        if group in group_roles and group_roles[group] != role:
            raise ValueError(f"group {group} crosses development and holdout")
        group_roles[group] = role
        for key in ("source_id", "content_sha256"):
            identity = clip.get(key)
            if identity:
                identity_key = (key, identity)
                if identity_key in identities and identities[identity_key] != group:
                    raise ValueError(f"{key} crosses groups: {identity}")
                identities[identity_key] = group
        if clip.get("augmentation") and not clip.get("parent_name"):
            raise ValueError(f"{name}: augmentation needs parent_name")
        if role == "holdout":
            holdout.append(name)
        else:
            groups.setdefault(group, []).append(name)
    by_name = {c["name"]: c for c in clips}
    for clip in clips:
        if clip.get("parent_name"):
            parent = by_name.get(clip["parent_name"])
            if parent is None or any(clip[k] != parent[k] for k in ("group_id", "source_id")):
                raise ValueError(f"{clip['name']}: parent must exist with the same group and source")
    if type(n_splits) is not int or n_splits < 2 or n_splits > len(groups):
        raise ValueError("n_splits must be between 2 and the number of development groups")
    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    group_buckets: list[list[str]] = [[] for _ in range(n_splits)]
    ordered = sorted(groups, key=lambda g: (-len(groups[g]), json_hash([seed, g])))
    for group in ordered:
        target = min(range(n_splits), key=lambda i: (len(buckets[i]), i))
        buckets[target].extend(groups[group])
        group_buckets[target].append(group)
    development = set().union(*map(set, buckets))
    folds = [{"fold": i, "train": sorted(development - set(buckets[i])), "evaluate": sorted(buckets[i]),
              "evaluation_groups": sorted(group_buckets[i])} for i in range(n_splits)]
    return {"schema": "forte.grouped-split-plan.v1", "seed": seed, "n_splits": n_splits,
            "manifest_sha256": json_hash(manifest), "provenance": provenance,
            "folds": folds, "untouched_holdout": sorted(holdout),
            "claim_scope": "split_plan_only; separately_train_tune_and_infer_each_fold_before_reporting_cv",
            "heldout_accuracy_claim": False,
            "limitations": ["Source and scenario IDs require human review; metadata cannot prove visual independence.",
                            "Augmented development videos are correlated robustness probes, not independent test examples."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    plan = grouped_folds(json.loads(Path(args.manifest).read_text(encoding="utf-8")), n_splits=args.folds, seed=args.seed)
    Path(args.out).write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
