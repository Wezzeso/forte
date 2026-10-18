"""Audit cached procedure coverage without opening media or contacting a model.

This is diagnostic postprocessing, never a source of video verdicts. Boolean state
fields measure schema completeness, not visibility: ``False`` is not a visibility
certificate. Empty probe lists and teacher ``NONE`` answers are unassessed checks,
not evidence that a procedure was performed. Labels are opened only by the final
scorer and never select windows, types, thresholds, or reported coverage.

Example::

    python tools/procedure_coverage_audit.py --run ref-structured.json \
        --run-name ref1 --probes probes.json --teacher teacher.json --out audit.json

New model-diagnostics bundles may be supplied with ``--diagnostics``. Legacy
teacher windows without a declared uniform grid remain explicitly nonuniform.
Only allowlisted enums, numbers, booleans, names, and hashes leave this program.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


TYPE_FIELDS = {
    "WORKSTATION_UNLOCKED": ("cashier_at_desk", "cashier_seat", "laptop"),
    "CASH_LEFT_UNATTENDED": ("cashier_in_room", "cash_on_desk", "valuables_on_desk"),
    "PREVIOUS_CASH_ON_DESK": ("client_at_desk", "client_seat", "cash_on_desk"),
    "COLLECTOR_NO_DOCUMENTS": ("cash_bag", "id_card_shown", "client_at_desk"),
    "COUNT_WITHOUT_CLIENT": ("client_at_desk", "client_seat", "counting_machine", "hand_counting"),
    "UNAUTHORIZED_PERSON": ("other_people", "person_at_door", "client_at_desk"),
    "CASH_ROOM_DOOR_OPEN": ("door_open",),
    "PERSONAL_ITEMS": ("drink", "personal_items"),
    "NO_CLIENT_ID": ("client_at_desk", "client_seat", "cash_in_hands", "id_card_shown"),
    "NO_AUTHENTICITY_CHECK": ("client_at_desk", "client_seat", "cash_in_hands", "counting_machine", "hand_counting", "detector"),
    "NO_RECOUNT": ("client_at_desk", "client_seat", "cash_bag", "counting_machine", "hand_counting"),
    "KEY_LEFT_IN_SAFE": (),  # No safe/key field exists in the cached state schema.
    "LEFT_DURING_SERVICE": ("cashier_at_desk", "cashier_seat", "client_at_desk", "client_seat"),
    "CASH_HANDOVER_IRREGULAR": ("cash_in_hands", "client_at_desk"),
}
TYPE_REGIONS = {
    "WORKSTATION_UNLOCKED": ("laptop", "cashier_chair"),
    "CASH_LEFT_UNATTENDED": ("desk",),
    "PREVIOUS_CASH_ON_DESK": ("desk", "client_chair"),
    "COLLECTOR_NO_DOCUMENTS": ("desk",),
    "COUNT_WITHOUT_CLIENT": ("counting_machine", "client_chair"),
    "UNAUTHORIZED_PERSON": ("door",),
    "CASH_ROOM_DOOR_OPEN": ("door",),
    "PERSONAL_ITEMS": ("desk",),
    "NO_CLIENT_ID": ("desk",),
    "NO_AUTHENTICITY_CHECK": ("detector",),
    "NO_RECOUNT": ("counting_machine", "safe"),
    "KEY_LEFT_IN_SAFE": ("safe",),
    "LEFT_DURING_SERVICE": ("cashier_chair", "client_chair"),
    "CASH_HANDOVER_IRREGULAR": ("desk",),
}
OMISSION_TYPES = frozenset(("NO_CLIENT_ID", "NO_AUTHENTICITY_CHECK", "NO_RECOUNT"))
NAME_RE = re.compile(r"X-[1-9][0-9]{0,2}\Z")
RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
ANSWER_VALUES = frozenset(("yes", "no", "unclear"))
CASHIER_VALUES = frozenset("counting_banknotes_by_hand using_counting_machine using_small_detector_device taking_cash_from_visitor giving_cash_to_visitor handling_cash_bag looking_at_document_or_card at_the_safe using_laptop leaving_or_away sitting_idle other".split())
VISITOR_VALUES = frozenset("none_present entering_or_leaving handing_cash handing_document_or_card handing_cash_bag receiving_cash waiting_seated standing_near_cashier other".split())
VISIBILITY_THRESHOLD = 0.90  # Brief section 7; not fitted to any labels.


def number(value: Any) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 86400


def fraction(n: int | float, d: int | float) -> float | None:
    return round(n / d, 6) if d else None


def load_json(path: Path) -> Any:
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("oversize_input")
    return json.loads(path.read_text(encoding="utf-8"))


def select_run(value: Any, name: str | None = None) -> dict:
    if name is not None:
        if not RUN_RE.fullmatch(name) or not isinstance(value, dict):
            raise ValueError("invalid_run_selection")
        value = value.get("runs", {}).get(name)
    if not isinstance(value, dict):
        raise ValueError("invalid_run")
    return value


def indexed_rows(value: Any) -> dict[str, dict]:
    if isinstance(value, dict):
        value = value.get("results", value.get("videos"))
    if not isinstance(value, list) or len(value) > 1000:
        raise ValueError("missing_rows")
    result = {}
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not NAME_RE.fullmatch(row["name"]):
            raise ValueError("invalid_row")
        if row["name"] in result:
            raise ValueError("duplicate_row")
        result[row["name"]] = row
    return result


def processed(row: dict) -> bool:
    return row.get("processed") is True and row.get("status") not in ("failed", "processing")


def valid_box(box: Any) -> bool:
    return (isinstance(box, list) and len(box) == 4 and
            all(number(x) and x <= 1000 for x in box) and
            box[0] < box[2] and box[1] < box[3])


def known_state(value: Any, field: str) -> bool:
    if field == "laptop":
        return isinstance(value, str) and value in ("content", "lock_screen", "closed_or_off")
    if field == "other_people":
        return type(value) is int and 0 <= value <= 9
    return type(value) is bool


def states_for(row: dict) -> list[dict]:
    states = row.get("states")
    if not isinstance(states, list) or len(states) > 1000:
        return []
    return [s for s in states if isinstance(s, dict) and number(s.get("t"))]


def session_proposal_counts(states: list[dict]) -> dict[str, int]:
    """Count current state-based probe proposals; never claim session completeness.

    Mirrors probes.sessions/windows for diagnostic comparison, including its
    majority smoothing and two-state minimum. Missing signals cannot prove that
    there was no transaction, so these are lower-bound proposal counts only.
    """
    counts = dict.fromkeys(sorted(OMISSION_TYPES), 0)
    if len(states) < 3:
        return counts
    raw = [s.get("client_seat") is True or s.get("client_at_desk") is True for s in states]
    flags = []
    for i in range(len(raw)):
        around = raw[max(0, i - 1):i + 2]
        flags.append(sum(around) * 2 > len(around))
    start = None
    for i, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            stop = i - 1
            if stop - start >= 2:
                span = states[max(0, start - 2):min(len(states), stop + 3)]
                def seen(field, desk_field=None):
                    return any(s.get(field) is True or (desk_field is not None and isinstance(s.get("desk"), dict) and s["desk"].get(desk_field) is True) for s in span)
                cash = seen("cash_in_hands", "cash")
                bag = seen("cash_bag", "cash_bag")
                counting = seen("counting_machine") or seen("hand_counting")
                counts["NO_CLIENT_ID"] += cash and not bag
                counts["NO_AUTHENTICITY_CHECK"] += cash and counting and not bag
                counts["NO_RECOUNT"] += bag
            start = None
    return counts


def interval_union(intervals: list[tuple[float, float]]) -> float:
    end = -1.0
    total = 0.0
    for lo, hi in sorted(intervals):
        if hi > max(lo, end):
            total += hi - max(lo, end)
        end = max(end, hi)
    return total


def visibility_proof(window: dict, type_id: str) -> dict:
    """Validate explicit time-resolved visibility, never substitute located boxes.

    Current probes do not produce this envelope. This optional contract permits
    future structured evidence without accepting a self-declared percentage.
    Overlapping occluded intervals conservatively invalidate the proof.
    """
    absent = {"status": "unproven", "visible_fraction": None}
    cov = window.get("coverage")
    lo, hi = window.get("start"), window.get("end")
    if not number(lo) or not number(hi) or hi <= lo or not isinstance(cov, dict):
        return absent
    if (cov.get("schema") != "forte.procedure-visibility.v1" or cov.get("type_id") != type_id or
            cov.get("source") not in ("local_model_visibility", "local_detector_visibility") or
            cov.get("interval_start") != lo or cov.get("interval_end") != hi):
        return absent
    segments = cov.get("segments")
    if not isinstance(segments, list) or not segments or len(segments) > 10000:
        return absent
    visible, blocked = [], []
    for segment in segments:
        if not isinstance(segment, dict):
            return absent
        a, b = segment.get("start"), segment.get("end")
        if not number(a) or not number(b) or b <= a or type(segment.get("visible")) is not bool or type(segment.get("occluded")) is not bool:
            return absent
        a, b = max(lo, a), min(hi, b)
        if b > a:
            (visible if segment["visible"] and not segment["occluded"] else blocked).append((a, b))
    # Contradictory segment annotations must never increase apparent visibility.
    overlap = sum(max(0.0, min(b, d) - max(a, c)) for a, b in visible for c, d in blocked)
    if overlap:
        return absent
    ratio = min(1.0, interval_union(visible) / (hi - lo))
    return {"status": "sufficient" if ratio >= VISIBILITY_THRESHOLD else "insufficient", "visible_fraction": round(ratio, 6)}


def teacher_rows(value: Any) -> tuple[dict[str, list[dict]], bool]:
    """Normalize old window lists and new model-diagnostics envelopes."""
    if value is None:
        return {}, False
    uniform = False
    if isinstance(value, dict):
        sampling = value.get("sampling", {})
        uniform = (isinstance(sampling, dict) and sampling.get("teacher_window_selection") == "uniform_grid_all_40_videos_without_labels")
        rows = indexed_rows(value.get("diagnostics", value))
        return {n: r.get("teacher", []) if isinstance(r.get("teacher"), list) else [] for n, r in rows.items()}, uniform
    if not isinstance(value, list) or len(value) > 10000:
        raise ValueError("invalid_teacher_rows")
    out: dict[str, list[dict]] = {}
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not NAME_RE.fullmatch(row["name"]):
            raise ValueError("invalid_teacher_row")
        out.setdefault(row["name"], []).append(row)
    return out, uniform


def teacher_answer_valid(answer: Any) -> bool:
    return (isinstance(answer, dict) and answer.get("violation") in {*TYPE_FIELDS, "NONE"} and
            answer.get("cashier") in CASHIER_VALUES and answer.get("visitor") in VISITOR_VALUES and
            answer.get("safe_door") in ("open", "closed", "not_visible") and
            answer.get("key_in_safe_lock") in ("yes", "no", "unclear") and
            answer.get("door_of_room") in ("open", "closed", "not_visible") and
            type(answer.get("people_in_room")) is int and 0 <= answer["people_in_room"] <= 6)


def audit(run: dict, *, probes: Any = None, teacher: Any = None,
          diagnostics: Any = None, expected_count: int = 40) -> dict:
    """Produce label-free metrics. Ignores labels/correctness stored in run rows."""
    rows = indexed_rows(run)
    if not rows:
        raise ValueError("empty_run")
    probe_rows = indexed_rows(probes) if probes is not None else {}
    teachers, uniform = teacher_rows(diagnostics if diagnostics is not None else teacher)
    if diagnostics is not None:
        diag_rows = indexed_rows(diagnostics.get("diagnostics", diagnostics))
        probe_rows = diag_rows
    foreign = (set(probe_rows) | set(teachers)) - set(rows)
    if foreign:
        raise ValueError("diagnostic_inventory_differs")
    per_video, per_type = [], {}
    for kind, fields in TYPE_FIELDS.items():
        aggregate = Counter()
        kind_rows = []
        for name, row in sorted(rows.items()):
            states = states_for(row)
            proposals = session_proposal_counts(states)
            known = sum(known_state(s.get(f), f) for s in states for f in fields)
            cells = len(states) * len(fields)
            layout = row.get("layout") if isinstance(row.get("layout"), dict) else {}
            located = all(valid_box(layout.get(r)) for r in TYPE_REGIONS[kind])
            raw_probes = probe_rows.get(name, {}).get("probes", [])
            if not isinstance(raw_probes, list) or len(raw_probes) > 10000:
                raise ValueError("invalid_probe_rows")
            windows = [p for p in raw_probes if isinstance(p, dict) and p.get("type_id") == kind]
            count = Counter()
            for window in windows:
                answers = window.get("answers")
                valid = (isinstance(answers, list) and len(answers) == 2 and
                         all(isinstance(a, str) and a in ANSWER_VALUES for a in answers) and
                         type(window.get("fired")) is bool)
                failed = window.get("status") == "failed"
                abstained = window.get("status") == "abstained" or window.get("abstain") == "region_not_located"
                if failed:
                    count["failed_windows"] += 1
                elif abstained:
                    count["abstained_windows"] += 1
                elif not valid:
                    count["invalid_windows"] += 1
                else:
                    count["answered_windows"] += 1
                    count["unclear_windows"] += "unclear" in answers
                    count["disagreed_windows"] += answers in (["yes", "yes"], ["no", "no"])
                    count["fired_windows"] += window["fired"] is True
                    count["inconsistent_fired_windows"] += window["fired"] != (answers == ["no", "yes"])
                proof = visibility_proof(window, kind)
                count["visibility_" + proof["status"] + "_windows"] += 1
            count["proposed_windows"] = len(windows)
            count["cached_state_proposals"] = proposals.get(kind, 0)
            count["cached_state_proposals_without_diagnostic"] = max(0, proposals.get(kind, 0) - len(windows))
            fully_answered = (bool(windows) and count["answered_windows"] == len(windows) and
                              not count["unclear_windows"] and not count["disagreed_windows"] and
                              not count["inconsistent_fired_windows"])
            omission_covered = (kind in OMISSION_TYPES and fully_answered and
                                count["visibility_sufficient_windows"] == len(windows))
            twindows = teachers.get(name, [])
            if not isinstance(twindows, list) or len(twindows) > 10000:
                raise ValueError("invalid_teacher_windows")
            tc = Counter()
            for window in twindows:
                if not isinstance(window, dict):
                    tc["invalid_windows"] += 1
                    continue
                a, b = window.get("a"), window.get("b")
                if window.get("status") == "failed" or not teacher_answer_valid(a) or not teacher_answer_valid(b):
                    tc["invalid_or_failed_windows"] += 1
                    continue
                tc["answered_windows"] += 1
                tc["type_disagreement_windows"] += a["violation"] != b["violation"]
                tc["consensus_type_windows"] += a["violation"] == kind == b["violation"]
                tc["both_none_windows"] += a["violation"] == "NONE" == b["violation"]
                tc["action_unknown_windows"] += any(x["cashier"] == "other" or x["visitor"] == "other" for x in (a, b))
                tc["key_unknown_windows"] += any(x["key_in_safe_lock"] == "unclear" for x in (a, b))
                tc["safe_unknown_windows"] += any(x["safe_door"] == "not_visible" for x in (a, b))
                tc["room_door_unknown_windows"] += any(x["door_of_room"] == "not_visible" for x in (a, b))
            status = "insufficient_coverage" if kind in OMISSION_TYPES and not omission_covered else "diagnostic_only"
            record = {"name": name, "source_processed": processed(row),
                      "state_frames": len(states), "state_required_cells": cells,
                      "state_known_cells": known, "state_unknown_cells": cells - known,
                      "all_required_regions_located": located,
                      "probe": {key: count[key] for key in (
                          "proposed_windows", "cached_state_proposals", "cached_state_proposals_without_diagnostic",
                          "answered_windows", "failed_windows", "abstained_windows", "invalid_windows",
                          "unclear_windows", "disagreed_windows", "fired_windows", "inconsistent_fired_windows",
                          "visibility_unproven_windows", "visibility_insufficient_windows", "visibility_sufficient_windows")},
                      "teacher": {key: tc[key] for key in (
                          "answered_windows", "invalid_windows", "invalid_or_failed_windows", "type_disagreement_windows", "consensus_type_windows",
                          "both_none_windows", "action_unknown_windows", "key_unknown_windows", "safe_unknown_windows", "room_door_unknown_windows")},
                      "omission_interval_coverage_proven": omission_covered,
                      "check_status": status, "check_verdict": None}
            kind_rows.append(record)
            aggregate.update(count)
            aggregate["state_frames"] += len(states)
            aggregate["state_required_cells"] += cells
            aggregate["state_unknown_cells"] += cells - known
            aggregate["videos_all_required_regions_located"] += located
            aggregate["videos_with_proposed_windows"] += bool(windows)
            aggregate["videos_with_proven_omission_coverage"] += omission_covered
            aggregate["videos_with_teacher_windows"] += bool(twindows)
            for key, value in tc.items():
                aggregate["teacher_" + key] += value
        per_type[kind] = {"state_fields": list(fields), "layout_regions": list(TYPE_REGIONS[kind]),
                          "counts": dict(sorted(aggregate.items())),
                          "state_schema_unknown_rate": fraction(aggregate["state_unknown_cells"], aggregate["state_required_cells"]),
                          "probe_unclear_rate": fraction(aggregate["unclear_windows"], aggregate["answered_windows"]),
                          "probe_visibility_proof_rate": fraction(aggregate["visibility_sufficient_windows"], aggregate["proposed_windows"]),
                          "per_video": kind_rows}
    for name, row in sorted(rows.items()):
        per_video.append({"name": name, "source_processed": processed(row), "state_frames": len(states_for(row))})
    return {"schema": "forte.procedure-coverage-audit.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "cached_structured_diagnostics; all_source_videos_uniformly_audited; in_sample",
            "heldout_accuracy_claim": False, "cross_validation_claim": False,
            "labels_used_for_coverage": False, "tuned_accuracy_parameters": 0,
            "type_enablement_qualified": False, "service_flags_changed": False,
            "inventory": {"expected_videos": expected_count, "observed_videos": len(rows),
                          "complete": len(rows) == expected_count,
                          "processed_videos": sum(processed(r) for r in rows.values()),
                          "teacher_video_rows": len(teachers), "probe_video_rows": len(probe_rows)},
            "coverage_policy": {"omission_min_visible_fraction": VISIBILITY_THRESHOLD,
                                "layout_presence_is_visibility_proof": False,
                                "boolean_false_is_visual_absence_proof": False,
                                "state_schema_unknown_rate_measures_schema_only": True,
                                "cached_state_proposals_prove_transaction_coverage": False,
                                "empty_probe_list_is_negative": False,
                                "teacher_none_is_negative": False,
                                "teacher_uniform_grid_declared": uniform,
                                "teacher_uniform_grid_independently_verified": False,
                                "augmentation_coverage_verified": False},
            "videos": per_video, "per_type": per_type}


def score_negative_gate(report: dict, labels: Any, *, expected_negatives: int = 19) -> dict:
    """Scorer only; coverage has already been finalized without labels."""
    if isinstance(labels, dict):
        labels = labels.get("labels", labels)
    if not isinstance(labels, dict):
        raise ValueError("invalid_labels")
    names = {r["name"] for r in report["videos"]}
    if set(labels) != names or any(not isinstance(v, dict) or type(v.get("violation")) is not bool for v in labels.values()):
        raise ValueError("scoring_inventory_differs")
    negatives = {name for name in names if not labels[name]["violation"]}
    out = {}
    for kind, metrics in report["per_type"].items():
        rows = [r for r in metrics["per_video"] if r["name"] in negatives]
        complete = (report["inventory"]["complete"] and len(negatives) == expected_negatives and
                    all(r["source_processed"] for r in rows))
        fired = sum(r["probe"]["fired_windows"] for r in rows)
        covered = sum(r["omission_interval_coverage_proven"] for r in rows)
        sufficient = complete and covered == expected_negatives and kind in OMISSION_TYPES
        out[kind] = {"negative_videos": len(negatives),
                     "source_negative_videos_processed": sum(r["source_processed"] for r in rows),
                     "source_all_19_processed": complete,
                     "probe_negative_videos_with_windows": sum(r["probe"]["proposed_windows"] > 0 for r in rows),
                     "probe_negative_fires": fired,
                     "probe_negative_videos_fired": sum(r["probe"]["fired_windows"] > 0 for r in rows),
                     "teacher_negative_consensus_fires": sum(r["teacher"]["consensus_type_windows"] for r in rows),
                     "negative_videos_with_proven_omission_coverage": covered,
                     "zero_probe_fires_with_complete_source_processing": complete and fired == 0,
                     "omission_negative_gate_evaluable": sufficient,
                     "omission_negative_gate_passed": sufficient and fired == 0,
                     "type_enablement_qualified": False,
                     "status": "diagnostic_only" if sufficient else "insufficient_coverage"}
    return {"labels_used_only_by_final_scorer": True, "expected_negative_videos": expected_negatives,
            "observed_negative_videos": len(negatives), "type_enablement_qualified": False,
            "per_type": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--expected-negatives", type=int, default=19)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.expected_count < 1 or args.expected_count > 1000 or not 0 <= args.expected_negatives <= args.expected_count:
        raise ValueError("invalid_expected_inventory")
    if args.diagnostics and (args.probes or args.teacher):
        raise ValueError("ambiguous_diagnostics_sources")
    report = audit(select_run(load_json(args.run), args.run_name),
                   probes=load_json(args.probes) if args.probes else None,
                   teacher=load_json(args.teacher) if args.teacher else None,
                   diagnostics=load_json(args.diagnostics) if args.diagnostics else None,
                   expected_count=args.expected_count)
    # This file is read only AFTER all label-free metrics have been finalized.
    if args.labels:
        report["scoring"] = score_negative_gate(report, load_json(args.labels), expected_negatives=args.expected_negatives)
    inputs = {"run": args.run, "probes": args.probes, "teacher": args.teacher, "diagnostics": args.diagnostics, "labels": args.labels}
    report["input_sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items() if path}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "done", "inventory": report["inventory"],
                      "per_omission_type": {kind: {"proposed_windows": report["per_type"][kind]["counts"].get("proposed_windows", 0),
                                                  "visibility_proven_windows": report["per_type"][kind]["counts"].get("visibility_sufficient_windows", 0),
                                                  "check_status": "insufficient_coverage"}
                                            for kind in sorted(OMISSION_TYPES)},
                      "type_enablement_qualified": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, KeyError, AttributeError, OSError, json.JSONDecodeError):
        # Do not print exceptions: malformed structured input may contain private text.
        print('{"status":"failed","failure_code":"coverage_audit_input_or_io_failure"}')
        raise SystemExit(1)
