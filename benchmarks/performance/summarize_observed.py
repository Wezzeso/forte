"""Combine published numeric benchmark artifacts without private result content."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from forte.observability.benchmark import statistics_ms


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def summarize(root):
    phases = {}
    all_rows, all_technical, all_stage_samples = [], [], []
    for phase in ("baseline", "burst"):
        http_path = root / f"gpu-{phase}.json"
        if not http_path.is_file():
            phases[phase] = {"status": "unmeasured"}
            continue
        http = read(http_path)
        stage_path = root / f"gpu-{phase}-stages.json"
        terminal_path = root / f"gpu-{phase}-terminal-technical.json"
        delta_path = root / f"gpu-{phase}-model-timing-delta.json"
        stages = read(stage_path) if stage_path.is_file() else {}
        technical = read(terminal_path) if terminal_path.is_file() else {}
        model = read(delta_path) if delta_path.is_file() else {"status": "unmeasured"}
        rows = http.get("samples", [])
        all_rows.extend(rows)
        all_technical.extend(technical.get("samples", []))
        all_stage_samples.extend(stages.get("samples", []))
        phases[phase] = {"status": "observed", "attempted": http["summary"]["attempted"],
                         "completed": http["summary"]["completed"], "failed": http["summary"]["failed"],
                         "client_timeouts": http["summary"]["timeouts"],
                         "static_error_code_counts": dict(Counter(row.get("error_code") or "none" for row in technical.get("samples", []))),
                         "model_request_lifecycle": model,
                         "http_artifact": f"benchmarks/performance/gpu-{phase}.json",
                         "stage_artifact": f"benchmarks/performance/gpu-{phase}-stages.json",
                         "terminal_artifact": f"benchmarks/performance/gpu-{phase}-terminal-technical.json"}
    def values(key):
        return [row[key] for row in all_rows if isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)]
    post = [row["post"] for row in all_rows if isinstance(row.get("post"), dict)]
    def post_values(key):
        return [row[key] for row in post if isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)]
    processing = [item for row in all_rows for item in row.get("processing_get", [])]
    completed_get = [item for row in all_rows for item in row.get("completed_get", [])]
    def get_stats(rows, key):
        return statistics_ms([row[key] for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row[key], bool)])
    stage_values = {}
    for row in all_stage_samples:
        merged = {}
        # Pipeline and worker timers overlap; each named stage is counted once.
        for name in ("worker_timing", "pipeline_timing"):
            timer = row.get(name)
            if isinstance(timer, dict):
                merged.update(timer.get("stages_ms", {}))
        for name, value in merged.items():
            stage_values.setdefault(name, []).append(value)
    terminal_counts = Counter(row.get("status", "unknown") for row in all_technical)
    outcomes = Counter(row.get("outcome", "unknown") for row in all_rows)
    completed = outcomes["completed"]
    return {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "first root-authorized proposed-train baseline and five distinct uploads; no final-check media",
            "quality_review": "unmeasured; terminal service behavior and lifecycle measurements do not establish reviewed accuracy",
            "phases": phases,
            "total": {"attempted": len(all_rows), "completed": completed, "failed_or_client_timeout": len(all_rows) - completed,
                      "client_outcome_counts": dict(outcomes), "observed_durable_status_counts": dict(terminal_counts),
                      "static_error_code_counts": dict(Counter(row.get("error_code") or "none" for row in all_technical)),
                      "terminal_state_verified_count": sum(row.get("terminal_verified") is True for row in all_technical)},
            "client_terminal_elapsed_ms_failure_inclusive": statistics_ms(values("client_observed_total_ms")),
            "client_upload_start_to_completed_full_result_ms": statistics_ms(values("client_upload_start_to_full_result_ms")),
            "client_total_post_ms_including_upload": statistics_ms(post_values("client_total_ms")),
            "asgi_after_body_durable_acceptance_ms": statistics_ms(post_values("asgi_post_body_acceptance_ms")),
            "processing_get_asgi_lookup_ms": get_stats(processing, "asgi_lookup_ms"),
            "processing_get_client_full_transfer_ms": get_stats(processing, "full_response_ms"),
            "completed_get_asgi_lookup_ms": get_stats(completed_get, "asgi_lookup_ms"),
            "completed_get_client_full_transfer_ms": get_stats(completed_get, "full_response_ms"),
            "lookup_sampling_note": "repeated polls of these same jobs; request distribution is not independent job throughput or population p95",
            "stage_duration_ms_failure_inclusive": {key: statistics_ms(value) for key, value in sorted(stage_values.items())},
            "planning_stage_budgets_ms": {"queue": 5000, "probe_decode_and_preparation": 10000,
                                          "coarse_model": 40000, "refinement": 25000,
                                          "rules_evidence_publication": 5000, "margin": 5000},
            "budget_note": "planning references only; absent publication/evidence on failed jobs is unmeasured, not zero",
            "success_targets": {"completed_video_latency_p95": "unmeasured; fewer than 20 completed video observations",
                                "full_145_second_video_p95": "unmeasured; selected train maximum is not a dedicated 145-second study",
                                "five_upload_complete_success": "achieved" if phases.get("burst", {}).get("completed") == 5 else "missed" if phases.get("burst", {}).get("status") == "observed" else "unmeasured"},
            "measurement_boundaries": {"T0_proxy": "unmeasured", "T1_to_T2": "ASGI-local monotonic after final received request body to durable accepted job",
                                       "queue": "same-host UTC accepted_at to first claimed; dispatch after claim excluded",
                                       "worker_stages": "worker/pipeline local monotonic elapsed; overlapping timers are not summed",
                                       "T5_client": "completed full JSON observed by client, includes network and polling interval",
                                       "prefill_decode": "actual server lifecycle counters, overlapping request sums; GPU-profiler compute unmeasured"},
            "privacy": "no source paths, hashes, provided labels, predictions, evidence, captions, or media exported"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("benchmarks/performance"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/performance/gpu-observed-summary.json"))
    args = parser.parse_args()
    report = summarize(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "written", **report["total"]}))


if __name__ == "__main__":
    main()
