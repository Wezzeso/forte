"""Merge safe numeric Worker telemetry into a benchmark without exposing job IDs."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from forte.observability.benchmark import statistics_ms
from forte.observability.timing import STAGES
from forte.observability.model_metrics import sanitize_model_metrics


def safe_number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def safe_timer(value):
    if not isinstance(value, dict) or value.get("clock") != "process_monotonic":
        return None
    raw_stages = value.get("stages_ms", {})
    stages = {name: number for name in STAGES if (number := safe_number(raw_stages.get(name))) is not None}
    return {"clock": "process_monotonic", "stages_ms": stages,
            "analysis_to_publication_ms": safe_number(value.get("analysis_to_publication_ms")),
            "model_requests": sanitize_model_metrics(value.get("model_requests"))}


def merge(http_path, mapping_path, logs_path, queue_path=None):
    http_report = json.loads(http_path.read_text(encoding="utf-8-sig"))
    if http_report.get("schema_version") != "1.0" or not isinstance(http_report.get("summary"), dict):
        raise ValueError("expected an aggregate HTTP benchmark report")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))["sample_jobs"]
    expected = set(mapping.values())
    by_job = {}
    ignored, matched = 0, 0
    for line in logs_path.read_text(encoding="utf-8-sig").splitlines():
        try:
            # A docker-compose prefix can precede the safe worker JSON record.
            start = line.find("{")
            item = json.loads(line[start:]) if start >= 0 else None
        except (ValueError, TypeError):
            ignored += 1
            continue
        if not isinstance(item, dict) or item.get("job_id") not in expected:
            ignored += 1
            continue
        attempt = item.get("attempt")
        state = item.get("status")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1 or state not in {"done", "failed", "ownership_lost", "interrupted"}:
            ignored += 1
            continue
        matched += 1
        safe = {"attempt": attempt, "worker_outcome": state,
                "worker_timing": safe_timer(item.get("timings")),
                "pipeline_timing": safe_timer(item.get("pipeline_timings"))}
        previous = by_job.get(item["job_id"])
        if previous is None or safe["attempt"] >= previous["attempt"]:
            by_job[item["job_id"]] = safe
    samples = []
    queues = {}
    if queue_path is not None:
        raw_queue = json.loads(queue_path.read_text(encoding="utf-8-sig"))
        if raw_queue.get("clock_domain") != "same team GPU host UTC for API and worker containers":
            raise ValueError("unverified queue clock domain")
        queues = {item.get("sample_id"): item for item in raw_queue.get("samples", []) if isinstance(item, dict)}
    for row in http_report.get("samples", []):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.startswith("s") or not sample_id[1:].isdigit():
            raise ValueError("unexpected opaque sample identifier")
        item = {"sample_id": sample_id, "http_outcome": row.get("outcome"),
                "worker_telemetry": "unmeasured"}
        selected = by_job.get(mapping.get(sample_id))
        if selected is not None:
            item.update(selected)
            item["worker_telemetry"] = "measured"
        item["acceptance_to_first_claim_ms"] = safe_number(queues.get(sample_id, {}).get("acceptance_to_first_claim_ms"))
        samples.append(item)
    aggregate = {}
    for stage in sorted(STAGES):
        values = []
        for sample in samples:
            for field in ("pipeline_timing", "worker_timing"):
                timer = sample.get(field)
                if timer and stage in timer["stages_ms"]:
                    values.append(timer["stages_ms"][stage])
                    break
        if values:
            aggregate[stage] = {**statistics_ms(values), "missing_sample_count": len(samples) - len(values)}
    complete_elapsed = [timer["analysis_to_publication_ms"] for item in samples
                        if item.get("worker_outcome") == "done" and (timer := item.get("worker_timing"))
                        and timer["analysis_to_publication_ms"] is not None]
    elapsed = statistics_ms(complete_elapsed)
    queue_values = [item["acceptance_to_first_claim_ms"] for item in samples if item["acceptance_to_first_claim_ms"] is not None]
    queue_statistics = statistics_ms(queue_values)
    observed_p95 = elapsed["p95_ms"]
    return {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "quality_review": "unmeasured; numeric processing telemetry is not validation ground truth",
            "http_summary": http_report.get("summary"),
            "worker_telemetry_matched_records": matched, "ignored_log_lines": ignored,
            "telemetry_sample_count": len(samples),
            "missing_terminal_worker_records": sum(item["worker_telemetry"] == "unmeasured" for item in samples),
            "stage_duration_ms": aggregate, "completed_worker_analysis_to_publication": elapsed,
            "acceptance_to_first_claim": queue_statistics,
            "selected_mix_analysis_p95_90000ms_reference": {"status": "unmeasured" if observed_p95 is None else "achieved" if observed_p95 <= 90_000 else "missed",
                                                           "scope": "selected duration mix only; does not establish 145-second-video p95"},
            "full_145_second_video_p95_target": "unmeasured; requires duration-filtered sample and sufficient observations",
            "planning_stage_budgets_ms": {"queue": 5000, "probe_decode_and_preparation": 10000,
                                          "coarse_model": 40000, "refinement": 25000,
                                          "rules_evidence_publication": 5000, "margin": 5000},
            "queue_duration": "same-host accepted_at to first claimed UTC; dispatch after claim excluded" if queue_path is not None else "unmeasured here; requires synchronized durable accepted/claimed UTC from same-host queue",
            "prefill_and_gpu_compute": "unmeasured unless actual server metrics separately verified",
            "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-report", type=Path, required=True)
    parser.add_argument("--private-job-map", type=Path, required=True)
    parser.add_argument("--worker-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-report", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = merge(args.http_report, args.private_job_map, args.worker_log, args.queue_report)
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__, "scope": "safe_numeric_stage_merge"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report.get("status", "finished"),
                      "telemetry_sample_count": report.get("telemetry_sample_count", 0),
                      "missing_terminal_worker_records": report.get("missing_terminal_worker_records", 0)}))
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
