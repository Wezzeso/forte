"""Qualify recorded local-model runs without loading media or calling a model.

  python tools/model_negative_gate.py --labels labels.json \
      --benchmark public_models.json --diagnostics probes.json --out gate.json

``--benchmark`` is repeatable and accepts the structured collector's
forte.model-benchmark.v1 envelope, or a fast_eval report. Only allowlisted fields
are projected before calling fast_validation; embedded scores/labels, scene
descriptions, images and exception messages are never emitted. Labels are scorer
inputs only. This tool never trains, calibrates or selects parameters.

Diagnostic coverage is explicit, never inferred from a non-firing probe. A new
diagnostic can record each row as processed=true, evaluated_types=[TYPE, ...],
and probes=[{type_id, fired, start, end, t, coverage_fraction, answers}]. Each
applicable window needs coverage_fraction >= .9 and two clear enum answers.
No-window rows must explicitly list the type in not_applicable_types. Legacy
probe reports remain useful observations, but cannot pass the coverage gate.

Passing two 37/40, zero-FP runs proves the recorded development verdict gate
only. Evidence, API/schema, six-job deadlines, augmentation/non-target gates,
leakage tests and grouped training validation remain separate release checks.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_validation import file_hash, json_hash, regression_gate, score_results, validate_labels  # noqa: E402
from forte.fast.catalog import TYPES  # noqa: E402

TYPE_IDS = frozenset(t.type_id for t in TYPES)
SHA = re.compile(r"[0-9a-f]{64}\Z")
REV = re.compile(r"[0-9a-f]{40}\Z")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,199}\Z")
NAME = re.compile(r"[A-Za-z0-9\u0400-\u04ff][A-Za-z0-9\u0400-\u04ff_.-]{0,99}\Z")
RELEASE_CHECKS = [
    "original_jpeg_evidence_relevance_and_time_bounds",
    "deployed_api_schema_and_failure_status",
    "six_parallel_longest_clips_complete_within_285_seconds",
    "no_regression_against_accuracy_qualified_pipeline",
    "all_non_target_videos_and_augmented_copies_for_new_types",
    "negative_distribution_threshold_provenance_for_new_types",
    "augmentation_invariance_leakage_and_unseen_angle_robustness",
    "grouped_lovo_for_every_learned_or_calibrated_parameter",
    "single_use_untouched_lockbox",
    "staging_rollback_and_public_six_upload_check",
]


def number(value: Any, low: float = 0, high: float = 1e12) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def token(value: Any) -> str | None:
    return value if isinstance(value, str) and TOKEN.fullmatch(value) else None


def sha(value: Any) -> str | None:
    return value if isinstance(value, str) and SHA.fullmatch(value) else None


def timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        return None
    return parsed.isoformat()


def load_json(path: Path) -> Any:
    if path.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("input_too_large")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def projected_rows(raw: Any) -> list[dict]:
    """Discard all unconstrained strings before the shared scorer sees a row."""
    if not isinstance(raw, list):
        raise ValueError("per_video_rows_required")
    rows = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not NAME.fullmatch(item["name"]):
            rows.append({})  # Preserve invalid-row integrity failure.
            continue
        row: dict[str, Any] = {"name": item["name"]}
        failed = bool(item.get("error") or item.get("failure_codes") or item.get("error_classes"))
        failed |= item.get("processed") is False
        failed |= "processed" in item and type(item["processed"]) is not bool
        failed |= "status" in item and item["status"] != "done"
        events = item.get("events")
        valid = isinstance(events, list) and len(events) <= 1000
        valid = valid and all(isinstance(e, dict) and isinstance(e.get("type_id"), str) and e["type_id"] in TYPE_IDS
                              and number(e.get("t"), high=86400) for e in events)
        if valid:
            row["events"] = [{"type_id": e["type_id"], "t": e["t"]} for e in events]
            if "verdict" in item:
                failed |= item["verdict"] != ("violation" if events else "no_violation")
            if "prediction" in item:
                failed |= type(item["prediction"]) is not bool or item["prediction"] != bool(events)
        if not valid:
            failed = True
        if failed:
            row["error"] = "source_failure_or_invalid_output"
        for key in ("wall_s", "analysis_wall_s", "decode_s", "calls"):
            if number(item.get(key)):
                row[key] = item[key]
        rows.append(row)
    return rows


def extract_runs(document: dict, source_sha256: str) -> list[dict]:
    """Normalize collector envelopes and raw fast_eval identity metadata."""
    if not isinstance(document, dict):
        raise ValueError("benchmark_object_required")
    envelope = isinstance(document.get("runs"), dict)
    source = document["runs"].items() if envelope else [("run", document)]
    runs = []
    for run_id, raw in source:
        if not token(run_id) or not isinstance(raw, dict):
            raise ValueError("invalid_run_record")
        prov = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
        hashes = raw.get("provenance_sha256") if isinstance(raw.get("provenance_sha256"), dict) else prov
        cfg = raw.get("config") if isinstance(raw.get("config"), dict) else {}
        model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
        repo = token(model.get("repo"))
        served = token(model.get("served_model")) or token(prov.get("model"))
        revision = model.get("observed_revision")
        revision = revision if isinstance(revision, str) and REV.fullmatch(revision) else None
        provenance_config_hash = sha(hashes.get("config_sha256"))
        observed_config_hash = (sha(cfg.get("complete_config_json_sha256")) if envelope else
                                json_hash(raw["config"]) if isinstance(raw.get("config"), dict) else None)
        config_hash = observed_config_hash or provenance_config_hash
        policy_hash = sha(hashes.get("analyzer_policy_sha256")) or sha(document.get("source_manifest_sha256"))
        identity = {"repo": repo, "served_model": served, "observed_revision": revision,
                    "config_sha256": config_hash, "policy_or_source_sha256": policy_hash}
        issues = []
        if not served:
            issues.append("model_identity_missing")
        if not config_hash:
            issues.append("config_identity_missing")
        if provenance_config_hash and observed_config_hash and provenance_config_hash != observed_config_hash:
            issues.append("config_identity_mismatch")
        if not policy_hash:
            issues.append("policy_or_source_identity_missing")
        if envelope:
            if raw.get("status") != "available":
                issues.append("source_report_not_available")
            integrity = raw.get("integrity")
            if not isinstance(integrity, dict) or integrity.get("complete") is not True:
                issues.append("source_integrity_incomplete_or_missing")
            if raw.get("source_model_matches_metadata") is not True:
                issues.append("source_model_identity_unverified")
            if revision is None:
                issues.append("observed_model_revision_unverified")
            report_hash = sha(raw.get("private_report_sha256"))
            generated = timestamp(raw.get("source_generated_at_utc"))
        else:
            if prov.get("inference_executed") is not True:
                issues.append("inference_execution_unverified")
            report_hash = source_sha256
            generated = timestamp(prov.get("generated_at_utc"))
        if generated is None:
            issues.append("independent_run_timestamp_missing")
        if report_hash is None:
            issues.append("private_report_identity_missing")
        media_hash = sha(hashes.get("input_inventory_sha256"))
        if media_hash is None:
            issues.append("media_inventory_identity_missing")
        rows = raw.get("videos") if envelope else raw.get("results")
        runs.append({"run_id": run_id, "identity": identity, "source_sha256": source_sha256,
                     "report_sha256": report_hash, "generated_at_utc": generated,
                     "media_inventory_sha256": media_hash, "issues": issues,
                     "rows": projected_rows(rows) if isinstance(rows, list) else []})
    return runs


def score_run(labels: dict, run: dict) -> dict:
    report = score_results(labels, run["rows"], time_tolerance=1.0)
    gate = regression_gate(report, expected_count=40, expected_clean=19, min_accuracy=37 / 40)
    reasons = sorted(set(gate["reasons"] + run["issues"]))
    negatives = [r for r in report["results"] if not r["label"]["violation"]]
    return {k: run[k] for k in ("run_id", "source_sha256", "report_sha256", "generated_at_utc",
                                "media_inventory_sha256")} | {
        "summary": report["summary"], "all_19_negatives_processed": len(negatives) == 19 and all(r["processed"] for r in negatives),
        "gate": {"passed": not reasons, "reasons": reasons},
        "videos": [{"name": r["name"], "processed": r["processed"], "correct": r["correct"],
                    "verdict": ("violation" if r["prediction"] else "no_violation") if r["processed"] else "failed",
                    "events": r.get("events", []), "failure": not r["processed"]} for r in report["results"]],
    }


def diagnostic_gate(labels: dict, document: dict, source_sha256: str) -> dict:
    """Report type-specific negative evidence; never choose or fit a threshold."""
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise ValueError("diagnostic_results_required")
    types = document.get("types")
    if (not isinstance(types, list) or not types or not all(isinstance(t, str) and t in TYPE_IDS for t in types)
            or len(set(types)) != len(types)):
        raise ValueError("diagnostic_types_required")
    names = [r.get("name") for r in document["results"] if isinstance(r, dict)]
    counts = Counter(n for n in names if isinstance(n, str))
    rows = {r["name"]: r for r in document["results"] if isinstance(r, dict) and isinstance(r.get("name"), str)}
    negatives = sorted(n for n, lab in labels.items() if not lab["violation"])
    exact = len(names) == len(document["results"]) and not (set(rows) - set(labels)) and all(v == 1 for v in counts.values())
    out = {}
    for kind in types:
        processed, covered, fires, windows, abstained = [], [], [], 0, 0
        for name in negatives:
            row = rows.get(name, {})
            probes = row.get("probes")
            applicable = [p for p in probes if isinstance(p, dict) and p.get("type_id") == kind] if isinstance(probes, list) else []
            done = (row.get("processed") is True and not row.get("error") and not row.get("failure_codes")
                    and not row.get("error_classes") and row.get("status", "done") == "done"
                    and isinstance(row.get("evaluated_types"), list) and kind in row["evaluated_types"])
            if done:
                processed.append(name)
            no_interval = (isinstance(row.get("not_applicable_types"), list) and kind in row["not_applicable_types"])
            window_ok = bool(applicable) and not no_interval
            for p in applicable:
                windows += 1
                clear = isinstance(p.get("answers"), list) and len(p["answers"]) == 2 and all(a in {"yes", "no"} for a in p["answers"])
                timing = all(number(p.get(k), high=86400) for k in ("start", "end", "t"))
                timing = timing and p["start"] <= p["t"] <= p["end"] and p["end"] > p["start"]
                complete = (clear and timing and number(p.get("coverage_fraction"), .9, 1)
                            and type(p.get("fired")) is bool and not p.get("abstain") and not p.get("error"))
                if not complete:
                    abstained += 1
                window_ok &= complete
                if p.get("fired") is True:
                    fires.append({"name": name, "t": p["t"] if number(p.get("t"), high=86400) else None})
            if done and (window_ok or (no_interval and not applicable)):
                covered.append(name)
        reasons = []
        if not exact:
            reasons.append("diagnostic_names_invalid_or_duplicate")
        if len(negatives) != 19 or len(processed) != 19:
            reasons.append("not_all_19_negatives_processed")
        if len(covered) != 19:
            reasons.append("negative_applicability_or_coverage_unproven")
        if fires:
            reasons.append("type_fired_on_negative")
        out[kind] = {"negative_videos_expected": len(negatives), "processed_negative_videos": len(processed),
                     "coverage_verified_negative_videos": len(covered), "negative_windows": windows,
                     "abstained_or_invalid_windows": abstained, "fires_on_negatives": fires,
                     "negative_observation_gate": {"passed": not reasons, "reasons": reasons},
                     "type_enablement_qualified": False,
                     "unmeasured": ["all_non_target_positive_videos_and_augmented_copies",
                                    "negative_distribution_threshold_provenance",
                                    "grouped_lovo_if_any_parameter_is_learned_or_calibrated"]}
    return {"source_sha256": source_sha256, "types": out, "parameters_fitted": 0,
            "scope": "negative_observations_only; no_type_enablement_authorization"}


def qualify(labels: dict, benchmarks: list[tuple[dict, str]], diagnostics: list[tuple[dict, str]] = ()) -> dict:
    labels = validate_labels(labels)
    if any(not NAME.fullmatch(n) for n in labels):
        raise ValueError("invalid_label_identifier")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for document, digest in benchmarks:
        for run in extract_runs(document, digest):
            grouped[json_hash(run["identity"])].append(run)
    candidates = []
    for candidate_id, runs in sorted(grouped.items()):
        scored = [score_run(labels, run) for run in runs]
        reasons = []
        if len(runs) < 2:
            reasons.append("fewer_than_two_recorded_runs")
        hashes = [r["report_sha256"] for r in runs]
        times = [r["generated_at_utc"] for r in runs]
        if None in hashes or len(set(hashes)) != len(hashes) or None in times or len(set(times)) != len(times):
            reasons.append("independent_run_evidence_missing_or_duplicate")
        inventories = {r["media_inventory_sha256"] for r in runs}
        if None in inventories or len(inventories) != 1:
            reasons.append("media_inventory_missing_or_differs")
        if not all(r["gate"]["passed"] for r in scored):
            reasons.append("one_or_more_run_gates_failed")
        candidates.append({"candidate_id": candidate_id, "identity": runs[0]["identity"], "runs": scored,
                           "two_run_development_verdict_gate": {"passed": not reasons, "reasons": reasons},
                           "release_qualified": False, "unmeasured_release_requirements": RELEASE_CHECKS,
                           "latency_scope": "analyzer_wall_measurements_are_not_API_202_to_done_latency"})
    return {"schema": "forte.model-negative-gate.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "labels_sha256": json_hash(labels), "scorer_sha256": file_hash(Path(__file__).with_name("fast_validation.py")),
            "tool_sha256": file_hash(Path(__file__)), "scope": "recorded_unique_development_set_in_sample",
            "labels_used_for": "scoring_only", "parameters_fitted": 0,
            "heldout_accuracy_claim": False, "cross_validation_claim": False, "promotion_authorized": False,
            "requirements": {"expected_videos": 40, "expected_negatives": 19, "minimum_correct_per_run": 37,
                             "maximum_false_alarms_per_run": 0, "minimum_independent_runs": 2,
                             "maximum_failures_per_run": 0, "minimum_omission_coverage_fraction": .9},
            "candidates": candidates,
            "diagnostics": [diagnostic_gate(labels, doc, digest) for doc, digest in diagnostics]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--benchmark", action="append", required=True, type=Path)
    parser.add_argument("--diagnostics", action="append", default=[], type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = [args.labels, *args.benchmark, *args.diagnostics]
    if args.out.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("output_overlaps_input")
    report = qualify(load_json(args.labels), [(load_json(p), file_hash(p)) for p in args.benchmark],
                     [(load_json(p), file_hash(p)) for p in args.diagnostics])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "candidates": len(report["candidates"]),
                      "passing_two_run_development_gates": sum(c["two_run_development_verdict_gate"]["passed"] for c in report["candidates"]),
                      "release_qualified": False, "promotion_authorized": False}, allow_nan=False))
    return 0 if any(c["two_run_development_verdict_gate"]["passed"] for c in report["candidates"]) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        # Exception messages may contain private data. Export only a fixed class.
        print(json.dumps({"status": "gate_input_failed", "error_class": type(exc).__name__}))
        raise SystemExit(2) from None
