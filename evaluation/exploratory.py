"""Aggregate proposed-train diagnostics against provided binary labels only.

This command performs no network, inference, frame extraction or model calls.
It does not read a split manifest or evaluate final-check samples. Private
terminal records remain local; only aggregate counts are emitted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .io import digest, read_json, read_records, write_json
from .metrics import number, ratio


def selected_inputs(selection: dict) -> list[dict]:
    baseline, burst = selection.get("baseline"), selection.get("burst")
    if not isinstance(baseline, dict) or not isinstance(burst, list) or len(burst) != 5:
        raise ValueError("expected one baseline and five burst inputs")
    rows = [baseline, *burst]
    if any(not isinstance(row, dict) or row.get("proposed_split") != "train" for row in rows):
        raise ValueError("only explicitly proposed-train inputs are allowed")
    ids = [row.get("sample_id") for row in rows]
    hashes = [row.get("content_hash", row.get("content_sha256")) for row in rows]
    if any(not isinstance(value, str) or not value for value in ids + hashes):
        raise ValueError("selection has missing identities")
    if len(set(ids)) != 6 or len(set(hashes)) != 6:
        raise ValueError("six distinct sample and content identities are required")
    return rows


def provided_labels(path: str | Path, permitted_content_hashes: set[str]) -> tuple[dict[str, bool], int]:
    """Stream workbook-derived metadata, retaining only whitelisted contents.

    No source filename, timestamp, caption or unrelated binary label is
    retained. Conflicting duplicate labels are unresolved rather than chosen.
    """
    labels: dict[str, set[bool]] = defaultdict(set)
    with Path(path).open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            content_hash = row.get("content_hash", row.get("content_sha256"))
            if content_hash not in permitted_content_hashes:
                continue
            value = row.get("has_violation")
            if isinstance(value, bool):
                labels[content_hash].add(value)
    conflicts = sum(len(values) > 1 for values in labels.values())
    return {key: next(iter(values)) for key, values in labels.items() if len(values) == 1}, conflicts


def aggregate(selection: dict, terminal_records: list[dict], labels: dict[str, bool], *, conflicting_label_contents: int = 0, deadline_sec: float = 300) -> dict:
    rows = selected_inputs(selection)
    expected_ids = {row["sample_id"] for row in rows}
    terminal = {}
    for row in terminal_records:
        sid = row.get("sample_id")
        if sid not in expected_ids or sid in terminal:
            raise ValueError("unexpected or duplicate terminal identity")
        terminal[sid] = row
    confusion = Counter({key: 0 for key in ("tp", "fp", "fn", "tn", "unprocessed_positive", "unprocessed_negative")})
    cutoff = Counter({key: 0 for key in confusion})
    counts: Counter = Counter()
    failures: Counter = Counter()
    for selected in rows:
        sid = selected["sample_id"]
        content_hash = selected.get("content_hash", selected.get("content_sha256"))
        item = terminal.get(sid, {})
        status = item.get("status")
        verdict = item.get("verdict")
        complete = status == "done" and verdict in {"violation", "no_violation"} and item.get("public_contract_validated") is True
        elapsed = number(item.get("upload_start_to_result_sec"))
        on_time = complete and elapsed is not None and 0 <= elapsed <= deadline_sec
        counts["completed"] += complete
        counts["on_time"] += on_time
        counts["timing_unknown"] += complete and (elapsed is None or elapsed < 0)
        counts["late"] += complete and elapsed is not None and elapsed > deadline_sec
        counts["missing_terminal"] += not bool(item)
        if not complete:
            # Use a fixed allowlist; never echo arbitrary error text/model output.
            reason = "missing_terminal" if not item else "failed" if status == "failed" else "timeout" if status == "timeout" else "processing" if status == "processing" else "invalid_or_unvalidated_terminal"
            failures[reason] += 1
        if content_hash not in labels:
            counts["missing_or_conflicting_label"] += 1
            continue
        positive = labels[content_hash]
        counts["labeled"] += 1
        counts["positive"] += positive
        counts["negative"] += not positive
        for target, eligible in ((confusion, complete), (cutoff, on_time)):
            if not eligible:
                target["unprocessed_positive" if positive else "unprocessed_negative"] += 1
            else:
                predicted_positive = verdict == "violation"
                target[("tp" if predicted_positive else "fn") if positive else ("fp" if predicted_positive else "tn")] += 1
    for target in (confusion, cutoff):
        target["completed_labeled_denominator"] = sum(target[key] for key in ("tp", "fp", "fn", "tn"))
        target["all_provided_label_denominator"] = counts["labeled"]
    return {
        "schema_version": "forte.provided_label_exploration.v1",
        "claim_scope": "exploratory_proposed_train_against_organizer_provided_binary_labels_only",
        "heldout_accuracy_claim": False,
        "human_reviewed_ground_truth": False,
        "selection_input_sha256": digest([{key: row.get(key) for key in ("sample_id", "content_hash", "content_sha256", "proposed_split")} for row in rows]),
        "selection_scope": {"proposed_split": "train", "distinct_contents": 6, "baseline_single_count": 1, "burst_count": 5, "final_check_media_or_predictions_accessed": False, "final_check_evaluated": False, "training_authorized_by_this_report": False, "split_lock_claimed": False},
        "labels": {"provenance": "organizer_workbook_via_A02_local_workbook_rows", "provided_binary_labeled_contents": counts["labeled"], "positive": counts["positive"], "negative": counts["negative"], "missing_or_conflicting": counts["missing_or_conflicting_label"], "conflicting_contents": conflicting_label_contents, "timestamp_labels_used": False, "private_visual_annotations_used": False},
        "completion": {"required": 6, "completed": counts["completed"], "failed_or_unprocessed": 6 - counts["completed"], "missing_terminal": counts["missing_terminal"], "rate": ratio(counts["completed"], 6), "on_time_completed": counts["on_time"], "late_completed": counts["late"], "timing_unknown": counts["timing_unknown"], "deadline_sec": deadline_sec, "deadline_completion_rate": ratio(counts["on_time"], 6), "failure_counts": dict(sorted(failures.items()))},
        "whole_video_against_provided_labels": {**dict(confusion), "effective_positive_recall_including_failures": ratio(confusion["tp"], counts["positive"]), "effective_correct_rate_including_failures": ratio(confusion["tp"] + confusion["tn"], counts["labeled"])},
        "whole_video_within_deadline_against_provided_labels": {**dict(cutoff), "effective_positive_recall_including_failures": ratio(cutoff["tp"], counts["positive"]), "effective_correct_rate_including_failures": ratio(cutoff["tp"] + cutoff["tn"], counts["labeled"])},
        "event_timestamps": "UNREVIEWED_NOT_EVALUATED", "evidence_support": "UNREVIEWED_NOT_EVALUATED", "legal_grounding": "UNREVIEWED_NOT_EVALUATED",
        "limitations": ["Six duration-selected proposed-train contents are exploratory development diagnostics, not heldout or final-check accuracy.", "Organizer-provided binary labels have not been independently reviewed here; no private action descriptions or event timestamps are evaluated.", "Terminal public-contract validation establishes payload shape, not event truth or evidence support.", "Failures remain separate from clean negatives; cutoff counts also exclude late and unknown-timing results.", "This offline aggregation makes no model call, split-lock claim, training authorization or adapter-selection decision."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default="benchmarks/private/a18-inputs.json")
    parser.add_argument("--terminals", required=True)
    parser.add_argument("--provided-labels", default="data/manifests/workbook_rows.jsonl")
    parser.add_argument("--output", default="benchmarks/quality/proposed-train-provided-labels.json")
    args = parser.parse_args(argv)
    try:
        selection = read_json(args.selection)
        selected = selected_inputs(selection)
        hashes = {row.get("content_hash", row.get("content_sha256")) for row in selected}
        labels, conflicts = provided_labels(args.provided_labels, hashes)
        terminals = read_records(args.terminals)
        if len(terminals) != 6 or any(row.get("status") not in {"done", "failed", "timeout"} for row in terminals):
            raise ValueError("wait for all six terminal records before aggregate reporting")
        report = aggregate(selection, terminals, labels, conflicting_label_contents=conflicts)
        write_json(args.output, report)
        print(json.dumps({"claim_scope": report["claim_scope"], "required": 6, "completed": report["completion"]["completed"], "provided_binary_labels": report["labels"]["provided_binary_labeled_contents"], "heldout_accuracy_claim": False}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"error": "exploratory_input_invalid", "private_values_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
