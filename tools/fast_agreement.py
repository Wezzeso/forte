"""Label-free agreement between a reference and a candidate fast_eval report.

Gate for speed changes (prompt section 6.5 / 8): a candidate passes only when it
reproduces the reference pipeline's verdicts and events and nearly all per-frame
structured states. Labels are never read here, so a speed change cannot be tuned
to them. Only structured fields (booleans, enums, counts, boxes, timestamps,
scores, timings) are read or written; model free text (observation_ru, notes)
is ignored and never printed.

  python tools/fast_agreement.py --reference ref.json --candidate cand.json --out agreement.json
  python tools/fast_agreement.py --summary run.json        # structured per-video summary only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

TEXT_FIELDS = {"observation_ru", "observation", "note", "note_ru", "error"}
EVENT_TOLERANCE_S = 0.5
STATE_TIME_TOLERANCE_S = 0.26


def _rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise ValueError("report has no per-video results")
    return {row["name"]: row for row in rows}


def _processed(row: dict[str, Any]) -> bool:
    if "processed" in row:
        return bool(row["processed"])
    return not row.get("error") and "events" in row


def _verdict(row: dict[str, Any]) -> str:
    if not _processed(row):
        return "failed"
    return "violation" if row.get("events") else "no_violation"


def _events(row: dict[str, Any]) -> list[tuple[str, float]]:
    return sorted((str(e["type_id"]), float(e["t"])) for e in row.get("events") or [])


def match_events(ref: list[tuple[str, float]], cand: list[tuple[str, float]],
                 tolerance: float = EVENT_TOLERANCE_S) -> tuple[list, list, list]:
    """Greedy one-to-one matching by type, nearest time first."""
    pairs = sorted(((abs(rt - ct), i, j) for i, (rk, rt) in enumerate(ref)
                    for j, (ck, ct) in enumerate(cand) if rk == ck and abs(rt - ct) <= tolerance))
    used_r, used_c, matched = set(), set(), []
    for dt, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        matched.append({"type_id": ref[i][0], "t_ref": ref[i][1], "t_cand": cand[j][1], "dt": round(dt, 3)})
    only_ref = [{"type_id": k, "t": t} for i, (k, t) in enumerate(ref) if i not in used_r]
    only_cand = [{"type_id": k, "t": t} for j, (k, t) in enumerate(cand) if j not in used_c]
    return matched, only_ref, only_cand


def flatten_state(state: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in state.items():
        if key == "t" or key in TEXT_FIELDS:
            continue
        if isinstance(value, dict):
            for sub, inner in value.items():
                if sub not in TEXT_FIELDS and not isinstance(inner, (dict, list)):
                    flat[f"{key}.{sub}"] = inner
        elif not isinstance(value, list):
            flat[key] = value
    return flat


def align_states(ref: list[dict], cand: list[dict]) -> list[tuple[dict, dict]]:
    pairs, j = [], 0
    cand = sorted(cand, key=lambda s: s["t"])
    for r in sorted(ref, key=lambda s: s["t"]):
        while j < len(cand) and cand[j]["t"] < r["t"] - STATE_TIME_TOLERANCE_S:
            j += 1
        if j < len(cand) and abs(cand[j]["t"] - r["t"]) <= STATE_TIME_TOLERANCE_S:
            pairs.append((r, cand[j]))
            j += 1
    return pairs


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 3)


def stats(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": _pct(values, 0.5), "p95": _pct(values, 0.95),
            "max": round(max(values), 3) if values else None,
            "mean": round(statistics.fmean(values), 3) if values else None}


def timing_summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys: dict[str, list[float]] = {}
    for row in rows.values():
        if not _processed(row):
            continue
        for key in ("decode_s", "analysis_wall_s", "wall_s"):
            if isinstance(row.get(key), (int, float)):
                keys.setdefault(key, []).append(float(row[key]))
        for key, value in (row.get("timings") or {}).items():
            if isinstance(value, (int, float)):
                keys.setdefault(f"timings.{key}", []).append(float(value))
        if isinstance(row.get("calls"), int):
            keys.setdefault("calls", []).append(float(row["calls"]))
    return {key: stats(values) for key, values in sorted(keys.items())}


def compare(reference: dict[str, Any], candidate: dict[str, Any], *, min_field: float = 0.98) -> dict[str, Any]:
    ref_rows, cand_rows = _rows(reference), _rows(candidate)
    names = sorted(set(ref_rows) | set(cand_rows))
    per_video, field_hits, field_total = [], {}, {}
    verdict_equal = events_equal = candidates_equal = 0
    for name in names:
        r, c = ref_rows.get(name), cand_rows.get(name)
        if r is None or c is None:
            per_video.append({"name": name, "missing_in": "reference" if r is None else "candidate"})
            continue
        matched, only_ref, only_cand = match_events(_events(r), _events(c))
        rc = sorted((str(x["type_id"]), float(x["t"])) for x in r.get("candidates") or [])
        cc = sorted((str(x["type_id"]), float(x["t"])) for x in c.get("candidates") or [])
        _, cand_only_ref, cand_only_cand = match_events(rc, cc)
        v_eq = _verdict(r) == _verdict(c)
        e_eq = not only_ref and not only_cand and _processed(r) and _processed(c)
        verdict_equal += v_eq
        events_equal += e_eq
        candidates_equal += not cand_only_ref and not cand_only_cand
        lr, lc = r.get("layout") or {}, c.get("layout") or {}
        layout_keys_diff = sorted(set(lr) ^ set(lc))
        shared = set(lr) & set(lc)
        layout_max_diff = max((abs(float(a) - float(b)) for k in shared for a, b in zip(lr[k], lc[k])), default=0.0)
        pairs = align_states(r.get("states") or [], c.get("states") or [])
        video_hits = video_total = 0
        disagreements: dict[str, int] = {}
        for rs, cs in pairs:
            fr, fc = flatten_state(rs), flatten_state(cs)
            for key in set(fr) | set(fc):
                field_total[key] = field_total.get(key, 0) + 1
                same = fr.get(key) == fc.get(key)
                field_hits[key] = field_hits.get(key, 0) + same
                video_total += 1
                video_hits += same
                if not same:
                    disagreements[key] = disagreements.get(key, 0) + 1
        per_video.append({
            "name": name, "verdict_ref": _verdict(r), "verdict_cand": _verdict(c), "verdict_equal": v_eq,
            "events_equal": e_eq, "events_matched": matched, "events_only_ref": only_ref,
            "events_only_cand": only_cand, "candidates_only_ref": cand_only_ref,
            "candidates_only_cand": cand_only_cand, "state_pairs": len(pairs),
            "state_frames_ref": len(r.get("states") or []), "state_frames_cand": len(c.get("states") or []),
            "state_agreement": round(video_hits / video_total, 4) if video_total else None,
            "state_disagreements": dict(sorted(disagreements.items())),
            "layout_equal": not layout_keys_diff and layout_max_diff == 0.0,
            "layout_max_diff": round(layout_max_diff, 1), "layout_keys_diff": layout_keys_diff,
            "wall_s_ref": r.get("analysis_wall_s", r.get("wall_s")),
            "wall_s_cand": c.get("analysis_wall_s", c.get("wall_s")),
        })
    per_field = {k: round(field_hits[k] / field_total[k], 4) for k in sorted(field_total)}
    overall = (sum(field_hits.values()) / sum(field_total.values())) if field_total else None
    worst_field = min(per_field.values()) if per_field else None
    compared = [v for v in per_video if "missing_in" not in v]
    gate = {
        "all_present": len(compared) == len(names),
        "verdicts_identical": verdict_equal == len(names),
        "events_identical": events_equal == len(names),
        "state_fields_min": worst_field,
        "state_fields_ok": worst_field is None or worst_field >= min_field,
        "min_field_threshold": min_field,
    }
    gate["passed"] = all(gate[k] for k in ("all_present", "verdicts_identical", "events_identical", "state_fields_ok"))
    ref_summary, cand_summary = reference.get("summary", {}), candidate.get("summary", {})
    load = {}
    for label, s in (("reference", ref_summary), ("candidate", cand_summary)):
        load[label] = {k: s.get(k) for k in ("calls", "prompt_tokens", "completion_tokens", "total_s") if k in s}
        n = max(1, len(_rows(reference if label == "reference" else candidate)))
        load[label].update({f"{k}_per_video": round(s[k] / n, 1) for k in ("calls", "prompt_tokens", "completion_tokens")
                            if isinstance(s.get(k), (int, float))})
    return {
        "schema": "forte.fast-agreement.v1", "labels_used": False,
        "videos": len(names), "verdicts_equal": verdict_equal, "events_equal": events_equal,
        "candidates_equal": candidates_equal,
        "layout_equal": sum(1 for v in per_video if v.get("layout_equal")),
        "state_agreement_overall": round(overall, 4) if overall else None,
        "state_agreement_per_field": per_field, "gate": gate, "load": load,
        "timings": {"reference": timing_summary(ref_rows), "candidate": timing_summary(cand_rows)},
        "config_sha256": {"reference": reference.get("provenance", {}).get("config_sha256"),
                          "candidate": candidate.get("provenance", {}).get("config_sha256")},
        "per_video": per_video,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    rows = _rows(report)
    videos = []
    for name in sorted(rows):
        row = rows[name]
        videos.append({"name": name, "verdict": _verdict(row), "events": _events(row),
                       "candidates": sorted((str(x["type_id"]), float(x["t"])) for x in row.get("candidates") or []),
                       "label": row.get("label"), "correct": row.get("correct"),
                       "wall_s": row.get("analysis_wall_s", row.get("wall_s")),
                       "timings": row.get("timings"), "calls": row.get("calls"),
                       "states": len(row.get("states") or [])})
    keep = ("n", "correct", "accuracy", "tp", "tn", "fp", "fn", "errors", "timecode_err_median",
            "timecode_within_2s", "calls", "prompt_tokens", "completion_tokens", "total_s")
    return {"summary": {k: report.get("summary", {}).get(k) for k in keep},
            "timings": timing_summary(rows), "videos": videos}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference")
    ap.add_argument("--candidate")
    ap.add_argument("--summary", help="Print a structured-only summary of one report")
    ap.add_argument("--min-field", type=float, default=0.98)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.summary:
        result = summarize(json.loads(Path(args.summary).read_text(encoding="utf-8")))
    else:
        if not args.reference or not args.candidate:
            ap.error("--reference and --candidate are required")
        result = compare(json.loads(Path(args.reference).read_text(encoding="utf-8")),
                         json.loads(Path(args.candidate).read_text(encoding="utf-8")), min_field=args.min_field)
        result["reference_path"], result["candidate_path"] = args.reference, args.candidate
    text = json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    brief = {k: result[k] for k in result if k not in ("per_video",) and not (args.summary and k == "videos")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    if not args.summary:
        for video in result["per_video"]:
            if not video.get("events_equal") or not video.get("verdict_equal"):
                print("DIFF", json.dumps({k: video.get(k) for k in ("name", "verdict_ref", "verdict_cand",
                                         "events_only_ref", "events_only_cand")}, ensure_ascii=False))
        return 0 if result["gate"]["passed"] else 1
    for video in result["videos"]:
        print(json.dumps(video, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
