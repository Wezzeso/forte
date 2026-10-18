"""CPU arithmetic over frozen geometry, never media, labels or inference.

A measured generated request is a reference observation, not a measured video
latency or a lower bound on future requests. Every projection states that
conditional assumption explicitly. No frozen plan/config/source is changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmarks.performance.repeatability import file_digest, require, write_new
from forte.video.sampling import coverage_windows


def assess(plan, *, reference_seconds, nonmodel_reserve_seconds=15., provider_timeout_seconds=240.):
    require(all(type(value) in {int, float} and math.isfinite(value) and value > 0
                for value in (reference_seconds, nonmodel_reserve_seconds, provider_timeout_seconds)),
            "feasibility_positive_reference_required")
    worker = plan["worker"]
    require(worker["max_concurrent_jobs"] == worker["max_concurrent_clips"] == worker["max_inflight_per_job"] == 1
            and worker["staged_screening"] is False, "feasibility_sequential_nonstaged_recipe_required")
    deadline = worker["job_timeout_seconds"]
    require(0 < nonmodel_reserve_seconds < deadline, "feasibility_nonmodel_reserve_invalid")
    cases = []
    for ordinal, case in enumerate(plan["cases"], 1):
        windows = coverage_windows(case["duration_sec"], window_sec=worker["window_seconds"],
            stride_sec=worker["stride_seconds"], sample_fps=worker["sample_fps"],
            max_frames_per_window=worker["max_frames_per_window"], max_windows=512, max_total_frames=8192)
        require(case["coarse_windows"] == len(windows), "feasibility_frozen_window_geometry_mismatch")
        count = len(windows)
        refined = count + worker["max_refinement_passes"]
        projected = count * reference_seconds
        available = deadline - nonmodel_reserve_seconds
        cases.append({"case_ordinal": ordinal, "duration_sec": case["duration_sec"],
            "initial_requests": count, "frame_counts_per_initial_request": sorted({len(window.timestamps_sec) for window in windows}),
            "maximum_logical_requests_with_refinement": refined,
            "deadline_seconds": deadline, "maximum_average_initial_request_seconds_zero_overhead": deadline / count,
            "maximum_average_initial_request_seconds_with_reserve": available / count,
            "maximum_average_request_seconds_with_all_refinements_and_reserve": available / refined,
            "conditional_initial_model_seconds_at_reference": projected,
            "conditional_deadline_exceeded_by_initial_model_alone": projected > deadline,
            "conditional_seconds_left_for_all_nonmodel_and_refinement": deadline - projected,
            "reference_speedup_required_for_initial_with_reserve": reference_seconds / (available / count),
            "maximum_average_request_seconds_for_ten_second_job_zero_overhead": 10. / count})
    require(sum(row["initial_requests"] for row in cases) == plan["minimum_initial_generation_requests"],
            "feasibility_frozen_initial_count_mismatch")
    return {"schema_version": "forte.qwen38-latency-feasibility.v1", "status": "conditional_metadata_analysis_not_video_measurement",
        "provided_reference_seconds": reference_seconds, "reference_input_scope": "one generated 16-frame full-19-fact request",
        "reference_assumed_equal_cost_for_each_window_only_for_projection": True,
        "sequential_initial_requests": sum(row["initial_requests"] for row in cases),
        "conditional_total_initial_model_seconds": sum(row["conditional_initial_model_seconds_at_reference"] for row in cases),
        "conditional_jobs_exceeding_deadline_before_other_work": sum(row["conditional_deadline_exceeded_by_initial_model_alone"] for row in cases),
        "provider_request_timeout_seconds": provider_timeout_seconds,
        "reference_exceeds_provider_request_timeout": reference_seconds > provider_timeout_seconds,
        "read_timeouts_are_not_automatically_retried": True, "nonmodel_reserve_seconds_assumption": nonmodel_reserve_seconds,
        "cases": cases, "deterministic_infeasibility_proved": False, "video_or_quality_scores_measured": False,
        "actual_private_model_calls": 0, "media_or_labels_read": False, "frozen_artifacts_modified": False,
        "required_next_gate": "review generated full-recipe provider performance before spending six private validation attempts"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-plan", type=Path, required=True)
    parser.add_argument("--reference-seconds", type=float, required=True)
    parser.add_argument("--reference-provenance", required=True)
    parser.add_argument("--provider-timeout-seconds", type=float, default=240.)
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.model_plan.read_text(encoding="utf-8-sig"))
        result = assess(plan, reference_seconds=args.reference_seconds, provider_timeout_seconds=args.provider_timeout_seconds)
        result.update(frozen_plan_file_sha256=file_digest(args.model_plan), reference_provenance=args.reference_provenance)
        if args.reference_receipt:
            observed = json.loads(args.reference_receipt.read_text(encoding="utf-8-sig"))
            require(observed.get("private_media_used") is False and observed.get("frames_generated") == 16
                    and observed.get("facts_validated") == 19 and observed.get("model_post_attempts") == 1
                    and abs(observed["elapsed_seconds"] - args.reference_seconds) < 1e-6,
                    "feasibility_reference_receipt_mismatch")
            metrics = observed["metrics"]
            result["verified_generated_reference"] = {"receipt_file_sha256": file_digest(args.reference_receipt),
                "full_probe_wall_seconds": observed["elapsed_seconds"], "provider_elapsed_seconds": metrics["elapsed_sec"],
                "http_elapsed_seconds": metrics["http_elapsed_sec"], "preprocessing_seconds": metrics["preprocessing_sec"],
                "readiness_seconds": metrics["readiness_sec"], "prompt_tokens": metrics["prompt_tokens"],
                "generated_tokens": metrics["generated_tokens"], "model_post_attempts": observed["model_post_attempts"],
                "server_stage_metrics_available": metrics.get("server_metrics") is not None,
                "video_or_visual_truth_measured": False}
        write_new(args.output, result)
        print(json.dumps({key: result[key] for key in ("status", "sequential_initial_requests",
            "conditional_jobs_exceeding_deadline_before_other_work", "reference_exceeds_provider_request_timeout",
            "actual_private_model_calls")}))
        return 0
    except Exception:
        print(json.dumps({"status": "failed", "reason_code": "feasibility_metadata_or_geometry_invalid", "actual_private_model_calls": 0}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
