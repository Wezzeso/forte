"""Report genuine server histogram sum/count deltas; never estimate GPU compute."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


FAMILIES = ("request_prefill_time_seconds", "request_decode_time_seconds",
            "request_inference_time_seconds", "request_queue_time_seconds",
            "time_to_first_token_seconds", "e2e_request_latency_seconds")


def delta(before, after, exclusive=False):
    if before.get("status") != "measured" or after.get("status") != "measured":
        return {"status": "unmeasured", "reason": "both successful snapshots required"}
    first, last = before.get("numeric_metric_totals", {}), after.get("numeric_metric_totals", {})

    def value(metrics, name):
        raw = metrics.get(name, {}).get("sum_value")
        return float(raw) if type(raw) in {int, float} and math.isfinite(raw) and raw >= 0 else None

    process_first, process_last = value(first, "process_start_time_seconds"), value(last, "process_start_time_seconds")
    same_process = process_first is not None and process_first == process_last
    reset = process_first is not None and process_last is not None and process_first != process_last
    families, invalid = {}, []
    for family in FAMILIES:
        name = "vllm:" + family
        sums = value(first, name + "_sum"), value(last, name + "_sum")
        counts = value(first, name + "_count"), value(last, name + "_count")
        if None in sums or None in counts:
            families[family] = {"status": "unmeasured"}
            continue
        elapsed, count = sums[1] - sums[0], counts[1] - counts[0]
        if elapsed < -1e-9 or count < 0 or not count.is_integer():
            invalid.append(family)
            families[family] = {"status": "unmeasured", "reason": "counter reset or invalid count"}
            continue
        families[family] = {"status": "measured", "request_observations": int(count),
                            "server_reported_total_ms": max(0, elapsed) * 1000,
                            "server_reported_mean_ms": max(0, elapsed) * 1000 / count if count else None,
                            "p50_ms": None, "p95_ms": None}
    quiet_boundaries = all(value(metrics, name) == 0 for metrics in (first, last)
                           for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"))
    attributed = exclusive and same_process and quiet_boundaries and not reset and not invalid
    return {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if reset or invalid else "measured",
            "same_metric_process_start_observed": same_process, "counter_reset_observed": reset,
            "running_and_waiting_zero_at_both_boundaries": quiet_boundaries,
            "exclusive_gpu_window_operator_declared": exclusive,
            "attributed_to_selected_benchmark": attributed,
            "timing_families": families, "invalid_counter_families": invalid,
            "scope": "server-reported request lifecycle durations; parallel request sums are not job wall time",
            "gpu_compute_time": "unmeasured; server prefill/decode metrics are not a GPU profiler",
            "p50_p95": "unmeasured from sums/counts; no histogram quantiles invented",
            "durable_job_queue": "separate from model request queue"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--exclusive-gpu-window", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = delta(json.loads(args.before.read_text(encoding="utf-8-sig")),
                       json.loads(args.after.read_text(encoding="utf-8-sig")), args.exclusive_gpu_window)
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "attributed_to_selected_benchmark": report.get("attributed_to_selected_benchmark", False)}))
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
