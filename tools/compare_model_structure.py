"""Label-free state, layout, and event diagnostics for locally collected runs.

Accepts fast_eval reports or the structured collector's ``runs`` envelope. It
ignores labels, correctness fields, descriptions, and images. This is a
diagnostic only: it does not fit parameters or change analyzer behavior.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BOOL_FIELDS = (
    "cashier_at_desk", "cashier_in_room", "client_at_desk", "person_at_door",
    "cash_in_hands", "cash_bag", "id_card_shown", "counting_machine", "detector",
    "hand_counting", "cash_on_desk", "valuables_on_desk", "drink", "personal_items",
    "cashier_seat", "client_seat", "desk.cash", "desk.cash_bag", "desk.coins_or_valuables",
    "desk.drink", "desk.coat_or_handbag", "desk.id_or_document",
)
ENUM_FIELDS = {"laptop": ("content", "lock_screen", "closed_or_off", "not_visible")}
COUNT_FIELDS = {"other_people": range(10)}
FIELDS = BOOL_FIELDS + tuple(ENUM_FIELDS) + tuple(COUNT_FIELDS)
LAYOUT_KEYS = ("desk", "laptop", "safe", "door", "counting_machine", "detector",
               "cashier_chair", "client_chair")
TYPE_IDS = frozenset((
    "WORKSTATION_UNLOCKED", "CASH_LEFT_UNATTENDED", "PREVIOUS_CASH_ON_DESK",
    "COLLECTOR_NO_DOCUMENTS", "COUNT_WITHOUT_CLIENT", "UNAUTHORIZED_PERSON",
    "CASH_ROOM_DOOR_OPEN", "PERSONAL_ITEMS", "NO_CLIENT_ID", "NO_AUTHENTICITY_CHECK",
    "NO_RECOUNT", "KEY_LEFT_IN_SAFE", "LEFT_DURING_SERVICE", "CASH_HANDOVER_IRREGULAR",
))


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _get(state: dict, path: str) -> str:
    value: Any = state
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return "missing"
        value = value[key]
    if path in BOOL_FIELDS:
        return str(value).lower() if type(value) is bool else "invalid"
    if path in ENUM_FIELDS:
        return value if value in ENUM_FIELDS[path] else "invalid"
    return str(value) if type(value) is int and value in COUNT_FIELDS[path] else "invalid"


def load_run(path: Path, run: str | None) -> dict:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if "runs" in report:
        if run is None:
            if len(report["runs"]) != 1:
                raise ValueError("multiple runs present; select --reference-run or --candidate-run")
            run = next(iter(report["runs"]))
        report = report["runs"][run]
    rows = report.get("results", report.get("videos"))
    if not isinstance(rows, list):
        raise ValueError("selected run has no per-video rows")
    names = [r.get("name") for r in rows]
    if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("video names must be unique strings")
    return {r["name"]: r for r in rows}


def _processed(row: dict) -> bool:
    if "processed" in row:
        return row["processed"] is True
    return not row.get("error") and "events" in row


def _events(row: dict, key: str = "events") -> list[tuple[str, float]]:
    return sorted((e["type_id"], float(e["t"])) for e in row.get(key, [])
                  if e.get("type_id") in TYPE_IDS and _finite(e.get("t")))


def _verdict(row: dict) -> str:
    if not _processed(row):
        return "failed"
    return "violation" if _events(row) else "no_violation"


def summarize(rows: dict[str, dict]) -> dict:
    values = {field: Counter() for field in FIELDS}
    transitions = {field: [0, 0] for field in FIELDS}
    layouts = {key: Counter() for key in LAYOUT_KEYS}
    failure_codes = Counter()
    event_types = Counter()
    candidate_types = Counter()
    frames = 0
    rows_with_states = 0
    for row in rows.values():
        # Only enum-shaped failure codes are exported by the collector.
        for code in row.get("failure_codes", []):
            if code in {"missing_result", "unprocessed", "model_call_failed", "technical_failure",
                        "schema_invalid", "missing_events", "invalid_verdict"}:
                failure_codes[code] += 1
            else:
                failure_codes["other_failure"] += 1
        event_types.update(k for k, _ in _events(row))
        candidate_types.update(k for k, _ in _events(row, "candidates"))
        states = [s for s in row.get("states", []) if isinstance(s, dict) and _finite(s.get("t"))]
        states.sort(key=lambda s: s["t"])
        rows_with_states += bool(states)
        frames += len(states)
        previous = None
        for state in states:
            flat = {field: _get(state, field) for field in FIELDS}
            for field, value in flat.items():
                values[field][value] += 1
                if previous is not None and value not in ("missing", "invalid") and previous[field] not in ("missing", "invalid"):
                    transitions[field][1] += 1
                    transitions[field][0] += value != previous[field]
            previous = flat
        layout = row.get("layout") or {}
        for key in LAYOUT_KEYS:
            box = layout.get(key)
            if key not in layout:
                status = "missing"
            elif box is None:
                status = "null"
            elif not isinstance(box, list) or len(box) != 4 or not all(_finite(v) for v in box):
                status = "invalid_shape"
            elif not all(0 <= v <= 1000 for v in box):
                status = "outside_normalized_range"
            elif box[2] <= box[0] or box[3] <= box[1]:
                status = "invalid_extent"
            else:
                status = "valid"
            layouts[key][status] += 1
    return {
        "videos": len(rows), "verdict_counts": dict(Counter(_verdict(row) for row in rows.values())),
        "videos_with_states": rows_with_states, "state_frames": frames,
        "state_value_counts": {k: dict(v) for k, v in values.items()},
        "state_transition_counts": {k: {"changes": n, "adjacent_pairs": total,
                                             "fraction": round(n / total, 4) if total else None}
                                    for k, (n, total) in transitions.items()},
        "layout_status_counts": {k: dict(v) for k, v in layouts.items()},
        "event_type_counts": dict(event_types), "candidate_type_counts": dict(candidate_types),
        "failure_code_counts": dict(failure_codes),
    }


def compare(reference: dict[str, dict], candidate: dict[str, dict]) -> dict:
    contingency = {field: Counter() for field in FIELDS}
    videos = []
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            videos.append({"name": name, "missing_in": "reference" if name not in reference else "candidate"})
            continue
        r, c = reference[name], candidate[name]
        rs = sorted((s for s in r.get("states", []) if _finite(s.get("t"))), key=lambda s: s["t"])
        cs = sorted((s for s in c.get("states", []) if _finite(s.get("t"))), key=lambda s: s["t"])
        j = pairs = 0
        disagreements = Counter()
        for state_r in rs:
            while j < len(cs) and cs[j]["t"] < state_r["t"] - 0.26:
                j += 1
            if j == len(cs) or abs(cs[j]["t"] - state_r["t"]) > 0.26:
                continue
            pairs += 1
            for field in FIELDS:
                vr, vc = _get(state_r, field), _get(cs[j], field)
                contingency[field][f"{vr}->{vc}"] += 1
                disagreements[field] += vr != vc
            j += 1
        videos.append({"name": name, "reference_verdict": _verdict(r), "candidate_verdict": _verdict(c),
                       "reference_events": _events(r), "candidate_events": _events(c),
                       "reference_candidate_count": len(_events(r, "candidates")),
                       "candidate_candidate_count": len(_events(c, "candidates")),
                       "aligned_states": pairs,
                       "state_disagreements": {k: v for k, v in disagreements.items() if v}})
    return {"reference": summarize(reference), "candidate": summarize(candidate),
            "state_value_contingency": {k: dict(v) for k, v in contingency.items()}, "videos": videos}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference-run")
    parser.add_argument("--candidate-run")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    output = compare(load_run(args.reference, args.reference_run), load_run(args.candidate, args.candidate_run))
    output.update(schema="forte.model-structure-diagnostic.v1", labels_used=False, parameters_fitted=0,
                  generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  reference_file_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  candidate_file_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest())
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "videos": len(output["videos"]), "labels_used": False}))


if __name__ == "__main__":
    main()
