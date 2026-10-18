"""Label-free proposal replay/comparison, followed by a separate offline scorer.

  python tools/structured_proposal_compare.py --labels labels.json \
    --base public_base.json --base-run baseline_a --out proposals.json
  # Add --adapted public_adapter.json --adapted-run adapter_a for paired analysis.

No inference, media, training, threshold search or service configuration changes.
The installed pure rule/session functions are extracted from their source AST;
model, decoder and transport imports are never executed. All rows receive the
same functions. Identifiers enter only alignment, integrity checks and scoring.

Three channels remain distinct: installed rule candidates; high-recall dense
probe windows; and sampled-state omission proposals within those windows.
Neither windows nor omission proposals are verified violations. Zero proposals
on negatives is necessary evidence only: region visibility, full procedure
coverage, typed truth, augmented non-targets and local verification are absent.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_agreement import align_states, flatten_state, match_events  # noqa: E402
from fast_validation import file_hash, json_hash, regression_gate, score_results, validate_labels  # noqa: E402
from model_negative_gate import NAME, TYPE_IDS, load_json, number, projected_rows, sha, token  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROBE_TYPES = ("NO_CLIENT_ID", "NO_AUTHENTICITY_CHECK", "NO_RECOUNT")
STATE_BOOLS = "cashier_at_desk cashier_in_room cashier_seat client_at_desk client_seat cash_in_hands cash_bag id_card_shown counting_machine detector hand_counting valuables_on_desk drink personal_items".split()
OPTIONAL_STATE_BOOLS = {"cashier_seat", "client_seat"}  # Crops can be absent; existing rules use the full-frame OR.
DESK_BOOLS = "cash cash_bag coins_or_valuables drink coat_or_handbag id_or_document".split()
LAPTOP = {"content", "lock_screen", "closed_or_off", "not_visible"}
UNMEASURED = ["decisive_region_visibility_and_90_percent_procedure_coverage", "dense_two_phrasing_local_verification",
              "type_ground_truth", "all_non_target_videos_and_augmented_copies", "negative_distribution_calibration_provenance",
              "grouped_lovo_if_any_parameter_is_learned", "original_evidence_api_schema_and_latency"]


def pure_functions(path: Path, names: tuple[str, ...], **extras) -> dict:
    """Reuse installed definitions exactly, without importing inference modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if set(names) - set(definitions):
        raise ValueError("pure_rule_definition_missing")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *[definitions[n] for n in names]], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": Any, **extras}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def replay_functions() -> tuple[dict, dict, dict]:
    analyzer = ROOT / "src/forte/fast/analyzer.py"
    probes = ROOT / "src/forte/fast/probes.py"

    def candidate(type_id, t, source, note="", score=.5, **ignored):
        # Fixed source-code notes are discarded; model descriptions never enter.
        return SimpleNamespace(type_id=type_id, t=t, source=source, score=score)

    rules = pure_functions(analyzer, ("_b", "_smooth", "_runs", "_presence", "_smooth_list", "_desk", "rule_candidates"), Candidate=candidate)
    windows = pure_functions(probes, ("_b", "_desk", "_smooth", "_runs", "sessions", "windows"))
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in (analyzer, probes)}
    return rules, windows, hashes


def states_only(raw: Any) -> list[dict]:
    if not isinstance(raw, list) or not 3 <= len(raw) <= 1000:
        raise ValueError("state_timeline_missing_or_invalid")
    states = []
    for item in raw:
        if (not isinstance(item, dict) or not number(item.get("t"), high=86400)
                or any(type(item.get(k)) is not bool for k in STATE_BOOLS if k not in OPTIONAL_STATE_BOOLS)
                or any(k in item and type(item[k]) is not bool for k in OPTIONAL_STATE_BOOLS)
                or item.get("laptop") not in LAPTOP or type(item.get("other_people")) is not int
                or not 0 <= item["other_people"] <= 9):
            raise ValueError("required_structured_state_missing_or_invalid")
        desk = item.get("desk")
        if not isinstance(desk, dict) or any(type(desk.get(k)) is not bool for k in DESK_BOOLS):
            raise ValueError("required_desk_state_missing_or_invalid")
        if states and item["t"] <= states[-1]["t"]:
            raise ValueError("state_timestamps_not_strictly_increasing")
        states.append({"t": item["t"], **{k: item[k] for k in STATE_BOOLS if k in item}, "laptop": item["laptop"],
                       "other_people": item["other_people"], "desk": {k: desk[k] for k in DESK_BOOLS}})
    return states


def select_run(document: dict, selected: str | None) -> dict:
    if not isinstance(document, dict):
        raise ValueError("structured_run_object_required")
    if isinstance(document.get("runs"), dict):
        keys = document["runs"]
        if selected is None:
            if len(keys) != 1:
                raise ValueError("explicit_run_selector_required")
            selected = next(iter(keys))
        if selected not in keys or not token(selected):
            raise ValueError("selected_run_missing_or_invalid")
        run = keys[selected]
        if not isinstance(run, dict):
            raise ValueError("invalid_selected_run")
        rows = run.get("videos")
        source_complete = isinstance(run.get("integrity"), dict) and run["integrity"].get("complete") is True
        available = run.get("status") == "available"
    else:
        selected = selected or "run"
        run, rows, source_complete, available = document, document.get("results"), True, True
    if not isinstance(rows, list):
        raise ValueError("selected_run_not_collected_yet")
    config = run.get("config") if isinstance(run.get("config"), dict) else {}
    values = config.get("values") if isinstance(config.get("values"), dict) else config
    # Only the existing rule parameter is needed. Do not infer or fit a value.
    min_away = values.get("min_away")
    if not number(min_away, 1, 1000) or int(min_away) != min_away:
        raise ValueError("recorded_min_away_required")
    return {"run_id": selected, "rows": rows, "rule_config": {"min_away": int(min_away)},
            "config_sha256": sha(config.get("complete_config_json_sha256")) or json_hash(config),
            "private_report_sha256": sha(run.get("private_report_sha256")),
            "source_complete": source_complete and available, "source_sha256": document.get("source_sha256", {})}


def omissions(states: list[dict], windows: list[dict]) -> list[dict]:
    out = []
    for window in windows:
        span = [s for s in states if window["start"] <= s["t"] <= window["end"]]
        if not span:
            continue
        kind = window["type_id"]
        if kind == "NO_CLIENT_ID":
            observed = any(s["id_card_shown"] or s["desk"]["id_or_document"] for s in span if s["t"] < window["t"])
        elif kind == "NO_AUTHENTICITY_CHECK":
            observed = any(s["detector"] for s in span)
        else:
            observed = any(s["counting_machine"] or s["hand_counting"] for s in span)
        if not observed:
            out.append({**window, "sampled_states": len(span),
                        "window_fully_inside_sampled_timeline": states[0]["t"] <= window["start"] and window["end"] <= states[-1]["t"],
                        "coverage_fraction": None, "verified": False})
    return out


def generate(run: dict, rules: dict, probes: dict) -> dict:
    """No labels are accepted here: proposals are generated before scoring."""
    observations = {r["name"]: r for r in projected_rows(run["rows"]) if "name" in r}
    generated, valid_states = [], {}
    for raw in run["rows"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not NAME.fullmatch(raw["name"]):
            generated.append({})
            continue
        name = raw["name"]
        row = {"name": name, "channels": {"recorded_events": observations[name].get("events", [])},
               "source_error": bool(observations[name].get("error")), "error": None}
        try:
            if observations[name].get("error") or raw.get("states_truncated"):
                raise ValueError("source_failed_or_truncated")
            states = states_only(raw.get("states"))
            candidates, _ = rules["rule_candidates"](states, run["rule_config"])
            rule = [{"type_id": c.type_id, "t": c.t} for c in candidates]
            windows = [{k: w[k] for k in ("type_id", "t", "start", "end")} for w in probes["windows"](states, list(PROBE_TYPES))]
            row["channels"].update(installed_rules_all_types=rule, dense_probe_windows=windows,
                                   sampled_state_omission_proposals=omissions(states, windows))
            valid_states[name] = states
            row["state_frames"] = len(states)
        except (ValueError, KeyError, TypeError):
            row["error"] = "proposal_replay_failed_or_incomplete_states"
        generated.append(row)
    return {**{k: run[k] for k in ("run_id", "rule_config", "config_sha256", "private_report_sha256", "source_complete")},
            "generated": generated, "states": valid_states}


def score_channel(labels: dict, generated: list[dict], channel: str, kind: str | None = None) -> dict:
    rows = []
    for item in generated:
        if "name" not in item:
            rows.append({})
            continue
        proposals = item.get("channels", {}).get(channel, [])
        ev = [{"type_id": p["type_id"], "t": p["t"]} for p in proposals if kind is None or p["type_id"] == kind]
        failed = item.get("source_error") if channel == "recorded_events" else item.get("error")
        rows.append({"name": item["name"], "events": ev, **({"error": "source_or_replay_failure"} if failed else {})})
    score = score_results(labels, rows, time_tolerance=1)
    gate = regression_gate(score, expected_count=40, expected_clean=19)
    s = score["summary"]
    negative_count = sum(len(r.get("events", [])) for r in score["results"] if not r["label"]["violation"] and r["processed"])
    positive_count = sum(len(r.get("events", [])) for r in score["results"] if r["label"]["violation"] and r["processed"])
    negative_distribution = Counter(len(r.get("events", [])) for r in score["results"]
                                    if not r["label"]["violation"] and r["processed"])
    recorded_positive = {r["name"] for r in generated if r.get("channels", {}).get("recorded_events") and not r.get("source_error")}
    originally_unreported = [r for r in score["results"] if r["label"]["violation"] and r["name"] not in recorded_positive]
    return {"videos": s["n"], "processed": s["processed"], "failed": s["failed"], "negative_videos": s["negative"],
            "negative_videos_with_proposals": s["fp"], "negative_proposals": negative_count,
            "positive_videos": s["positive"], "positive_videos_with_proposals": s["tp"], "positive_proposals": positive_count,
            "positive_video_proposal_coverage": s["recall"],
            "originally_unreported_positive_videos": len(originally_unreported),
            "additional_positive_videos_with_proposals": sum(r["processed"] and bool(r["prediction"]) for r in originally_unreported),
            "negative_proposal_count_distribution": {str(k): v for k, v in sorted(negative_distribution.items())},
            "zero_negative_proposals_gate": gate, "type_recall_measured": False,
            "interpretation": "positive_video_coverage_proxy; proposals_are_not_confirmed_findings"}


def summarize(labels: dict, replay: dict) -> dict:
    channels = ("recorded_events", "installed_rules_all_types", "dense_probe_windows", "sampled_state_omission_proposals")
    out = {}
    for channel in channels:
        kinds = set(PROBE_TYPES) if channel in {"dense_probe_windows", "sampled_state_omission_proposals"} else set(TYPE_IDS)
        scores = {kind: score_channel(labels, replay["generated"], channel, kind) for kind in sorted(kinds)}
        out[channel] = {"all_types": score_channel(labels, replay["generated"], channel), "by_type": scores}
        if not replay["source_complete"]:
            for value in [out[channel]["all_types"], *scores.values()]:
                value["zero_negative_proposals_gate"]["passed"] = False
                value["zero_negative_proposals_gate"]["reasons"].append("source_integrity_incomplete")
    videos = [{"name": r["name"], "proposal_replay_processed": not bool(r.get("error")),
               "source_processed": not r.get("source_error", True), "channels": r.get("channels", {}),
               "proposal_replay_failure": bool(r.get("error"))} for r in replay["generated"] if "name" in r]
    return {k: replay[k] for k in ("run_id", "rule_config", "config_sha256", "private_report_sha256", "source_complete")} | {"channels": out, "videos": videos}


def compare(base: dict, adapted: dict) -> dict:
    per_video, field_comparisons, field_changes = [], Counter(), Counter()
    before = {r["name"]: r for r in base["generated"] if "name" in r}
    after = {r["name"]: r for r in adapted["generated"] if "name" in r}
    all_names = sorted(set(before) | set(after))
    for name in all_names:
        left, right = before.get(name), after.get(name)
        if left is None or right is None or left.get("error") or right.get("error"):
            per_video.append({"name": name, "comparable": False})
            continue
        pairs = align_states(base["states"][name], adapted["states"][name])
        for a, b in pairs:
            fa, fb = flatten_state(a), flatten_state(b)
            for key in set(fa) | set(fb):
                field_comparisons[key] += 1
                field_changes[key] += fa.get(key) != fb.get(key)
        changes = {}
        for channel, proposals in left["channels"].items():
            match, removed, added = match_events([(p["type_id"], p["t"]) for p in proposals],
                                                [(p["type_id"], p["t"]) for p in right["channels"].get(channel, [])])
            changes[channel] = {"matched": len(match), "removed": removed, "added": added}
        per_video.append({"name": name, "comparable": True, "aligned_state_frames": len(pairs),
                          "base_state_frames": len(base["states"][name]), "adapted_state_frames": len(adapted["states"][name]),
                          "proposal_changes": changes})
    return {"labels_used": False, "videos": len(all_names), "comparable_videos": sum(r["comparable"] for r in per_video),
            "recorded_min_away_equal": base["rule_config"] == adapted["rule_config"],
            "state_changes": {k: {"compared": field_comparisons[k], "changed": field_changes[k]} for k in sorted(field_comparisons)},
            "scope": "recorded_run_comparison; differences_include_model_run_noise; no_causal_adapter_claim", "per_video": per_video}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--base-run")
    parser.add_argument("--adapted", type=Path)
    parser.add_argument("--adapted-run")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    sources = [args.labels, args.base, *([args.adapted] if args.adapted else [])]
    if args.out.resolve() in {p.resolve() for p in sources}:
        raise ValueError("output_overlaps_input")
    if args.adapted_run and not args.adapted:
        raise ValueError("adapted_path_required")
    rules, probes, hashes = replay_functions()
    # Generate first; only then open labels in the separate scoring phase.
    base = generate(select_run(load_json(args.base), args.base_run), rules, probes)
    adapted = generate(select_run(load_json(args.adapted), args.adapted_run), rules, probes) if args.adapted else None
    labels = validate_labels(load_json(args.labels))
    report = {"schema": "forte.structured-proposal-compare.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "tool_sha256": file_hash(Path(__file__)), "pure_rule_source_sha256": hashes,
              "base_source_sha256": file_hash(args.base), "labels_sha256": json_hash(labels),
              "labels_used_for": "scoring_only_after_uniform_proposal_generation", "parameters_fitted": 0,
              "threshold_search_performed": False, "inference_executed": False, "type_enablement_authorized": False,
              "heldout_accuracy_claim": False, "cross_validation_claim": False,
              "unmeasured": UNMEASURED,
              "unsupported_from_recorded_states": ["KEY_LEFT_IN_SAFE_requires_key_and_safe_interaction_states",
                                                    "CASH_HANDOVER_IRREGULAR_requires_cash_transfer_and_hand_trajectories"],
              "base": summarize(labels, base), "adapted": summarize(labels, adapted) if adapted else None,
              "comparison": compare(base, adapted) if adapted else None}
    if args.adapted:
        report["adapted_source_sha256"] = file_hash(args.adapted)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    brief = {"schema": report["schema"], "parameters_fitted": 0, "type_enablement_authorized": False}
    for name in ("base", "adapted"):
        if report[name]:
            brief[name] = {k: v["all_types"] for k, v in report[name]["channels"].items()}
    print(json.dumps(brief, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        print(json.dumps({"status": "proposal_comparison_input_failed", "error_class": type(exc).__name__}))
        raise SystemExit(2) from None
