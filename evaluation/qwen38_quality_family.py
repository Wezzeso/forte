"""Immutable quality-family population/recipe, with actual per-arm identities.

The warm baseline may run before a trained adapter exists. Each arm later binds
its own real identity to the same already frozen family; no result is rewritten
to pretend that a future adapter hash was known during the baseline run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re

from evaluation.io import digest, read_json, read_records
from evaluation.qwen38_comparison import (ARMS, IDENTITY_FIELDS, MODEL_FIELDS,
    _confusion, _content, _count, _distribution, _index, _normalize, _references, _summary)
from evaluation.qwen38_pipeline import WORKER_FIELDS, file_sha
from evaluation.qwen38_server_export import write_private_new
from evaluation.weak_supervision import provided_label_digest
from forte.video.sampling import coverage_windows


SCHEMA = "forte.qwen38-quality-family.v1"


def require(value, code):
    if not value:
        raise ValueError(code)


def validate_family(family: dict) -> tuple[dict, dict]:
    require(family.get("schema_version") == SCHEMA, "quality_family_schema_invalid")
    records = _index(family.get("records"), "sample_id")
    require(len(records) == 6 and all(row["split"] == "validation" for row in records.values())
        and len({_content(row) for row in records.values()}) == 6, "quality_family_exact_validation_population_required")
    recipe = family.get("recipe")
    require(isinstance(recipe, dict) and family.get("recipe_sha256") == digest(recipe), "quality_family_recipe_changed")
    worker = recipe.get("worker_settings", recipe.get("sampling"))
    require(isinstance(worker, dict) and set(worker) == set(WORKER_FIELDS)
        and worker["max_concurrent_jobs"] == worker["max_concurrent_clips"] == worker["max_inflight_per_job"] == 1
        and worker["staged_screening"] is False and recipe.get("provider_max_attempts") == 2,
        "quality_family_serial_nonstaged_profile_required")
    require(set(recipe["generation"]) >= {"max_new_tokens", "seed"}
        and recipe["generation"]["seed"] == 20261017
        and recipe["generation"]["max_new_tokens"] == 4096,
        "quality_family_generation_changed")
    require(all(isinstance(recipe.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", recipe[key])
        for key in ("source_archive_sha256", "rules_sha256", "prompt_source_sha256"))
        and recipe.get("context_tokens") == 32768,
        "quality_family_source_prompt_rules_pins_missing")
    expected = _index(family.get("attempts"), "attempt_id")
    for arm in ARMS:
        planned = [row for row in expected.values() if row["arm"] == arm]
        require(len(planned) == 6 and {row["sample_id"] for row in planned} == set(records)
            and all(row["repeat_index"] == 0 and row["split"] == "validation"
                and row["content_sha256"] == _content(records[row["sample_id"]]) for row in planned),
            "quality_family_paired_source_attempts_changed")
    require(len(expected) == 18, "quality_family_arm_population_changed")
    return records, expected


def arm_plan(family: dict, spec: dict, *, role_config_sha256: str) -> dict:
    records, expected = validate_family(family)
    arm = spec["arm"]
    require(arm in ARMS, "quality_arm_invalid")
    identity = spec["expected_model_identity"]
    require(isinstance(identity, dict) and set(identity) == set(MODEL_FIELDS)
        and {key: identity[key] for key in MODEL_FIELDS[:2]} == family["base_models"][arm],
        "quality_arm_actual_base_identity_mismatch")
    if arm == "qwen38_tuned":
        require(isinstance(identity["adapter_sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", identity["adapter_sha256"]), "quality_tuned_actual_adapter_missing")
    else:
        require(identity["adapter_sha256"] is None, "quality_untuned_adapter_forbidden")
    recipe = family["recipe"]
    require(spec["runtime_bindings"]["source_archive_sha256"] == recipe["source_archive_sha256"],
            "quality_arm_shared_source_archive_changed")
    sources = _index(spec["sources"], "sample_id")
    require(set(sources) == set(records), "quality_arm_source_population_changed")
    cases, slots = [], []
    for source_attempt in (row for row in expected.values() if row["arm"] == arm):
        sid = source_attempt["sample_id"]
        source = sources[sid]
        require(source["content_sha256"] == _content(records[sid]) and source["split"] == "validation"
            and isinstance(source.get("owned_path"), str) and (Path(source["owned_path"]).is_absolute()
                or PurePosixPath(source["owned_path"]).is_absolute())
            and type(source.get("source_bytes")) is int and source["source_bytes"] > 0,
            "quality_arm_source_identity_or_readonly_path_invalid")
        worker = recipe["worker_settings"]
        windows = len(coverage_windows(source["duration_sec"], window_sec=worker["window_seconds"],
            stride_sec=worker["stride_seconds"], sample_fps=worker["sample_fps"],
            max_frames_per_window=worker["max_frames_per_window"]))
        upper = (windows + worker["max_refinement_passes"]) * recipe["provider_max_attempts"]
        cases.append({**source_attempt, **{key: source[key] for key in ("owned_path", "source_bytes", "duration_sec")},
            "coarse_windows": windows, "maximum_pipeline_inferences": windows + worker["max_refinement_passes"],
            "maximum_model_requests": upper})
        for index in range(upper):
            slots.append({"attempt_id": source_attempt["attempt_id"] + f"-model-{index:04d}",
                "source_attempt_id": source_attempt["attempt_id"], "content_sha256": source_attempt["content_sha256"],
                "split": "validation", "slot_index": index})
    require(sum(case["coarse_windows"] for case in cases) == 56 and len(slots) == 208,
            "quality_arm_exact_56_initial_208_upper_bound_changed")
    return {"schema_version": "forte.qwen38-attempt-plan.v1", "kind": "fullpipeline_quality",
        "phase": "evaluation", "arm": arm, "comparison_plan_sha256": digest(family),
        "comparison_identity_scope": "immutable_quality_family_not_future_adapter_manifest",
        "recipe_sha256": family["recipe_sha256"], "recipe": recipe, "expected_model_identity": identity,
        "worker": recipe["worker_settings"], "provider": spec["provider"], "role_config_sha256": role_config_sha256,
        "runtime_bindings": spec["runtime_bindings"], "cases": cases,
        "source_attempts": [{key: case[key] for key in ("attempt_id", *IDENTITY_FIELDS)} for case in cases],
        "attempts": slots, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_attempts_required": 6, "maximum_private_model_requests": 208,
        "minimum_initial_generation_requests": 56, "source_media_read_during_plan_binding": False,
        "source_hash_recheck_required_by_executor_before_every_source": True,
        "quality_model_calls_during_prepare": 0, "final_check_used": False,
        "upload_start_or_browser_latency_measured": False}


def compare_family(family: dict, arm_plans: list[dict], terminals: list[dict], labels: list[dict], *, metadata: dict) -> dict:
    records, expected = validate_family(family)
    family_sha = digest(family)
    actual = _index(terminals, "attempt_id")
    require(not (actual.keys() - expected.keys()), "quality_family_unplanned_results")
    plans = {plan["arm"]: plan for plan in arm_plans}
    require(set(plans) == set(ARMS) and len(arm_plans) == 3, "quality_family_all_actual_arm_plans_required")
    references = _references(labels, records)
    require(provided_label_digest(labels) == family["provided_labels_sha256"], "quality_family_label_pin_changed")
    blockers = Counter()
    reports = {}
    for arm in ARMS:
        plan = plans[arm]
        require(plan["comparison_plan_sha256"] == family_sha and plan["recipe_sha256"] == family["recipe_sha256"]
            and plan["recipe"] == family["recipe"] and plan["worker"] == family["recipe"]["worker_settings"]
            and plan["runtime_bindings"]["source_archive_sha256"] == family["recipe"]["source_archive_sha256"],
            "quality_arm_family_recipe_changed")
        expected_arm = {aid: row for aid, row in expected.items() if row["arm"] == arm}
        require({row["attempt_id"]: row for row in plan["source_attempts"]} == expected_arm,
                "quality_arm_source_denominator_changed")
        identity = plan["expected_model_identity"]
        require({key: identity[key] for key in MODEL_FIELDS[:2]} == family["base_models"][arm]
            and (identity["adapter_sha256"] is None if arm != "qwen38_tuned"
                 else isinstance(identity["adapter_sha256"], str) and re.fullmatch(r"[a-f0-9]{64}", identity["adapter_sha256"])),
            "quality_arm_actual_model_identity_invalid")
        confusion = _confusion()
        failures = Counter()
        timings = []
        completed = 0
        for aid, planned in expected_arm.items():
            row = actual.get(aid)
            if row is not None:
                require(all(row.get(key) == planned[key] for key in IDENTITY_FIELDS)
                    and row.get("comparison_plan_sha256") == family_sha
                    and row.get("recipe_sha256") == family["recipe_sha256"], "quality_terminal_original_attribution_changed")
                if row.get("status") == "done":
                    require(row.get("model_identity") == identity, "quality_terminal_actual_model_changed")
                else:
                    require(row.get("expected_model_identity") == identity, "quality_terminal_intended_model_changed")
            normalized = _normalize(row, records[planned["sample_id"]])
            completed += normalized["completed"]
            if not normalized["completed"]:
                failures[str(normalized["failure"])] += 1
            label = references.get(planned["content_sha256"])
            if label is not None:
                _count(confusion, label["provided_video_label"], normalized)
            duration = row.get("pipeline_start_to_result_ms") if row else None
            if normalized["completed"] and isinstance(duration, (float, int)) and not isinstance(duration, bool):
                timings.append(duration)
            if row is None:
                blockers["missing_source_terminal"] += 1
            if normalized["completed"] and row.get("server_receipt_verified") is not True:
                blockers["fresh_native_evidence_receipt_unverified"] += 1
        reports[arm] = {"actual_model_identity": identity, "arm_plan_sha256": digest(plan),
            "completion": {"required_attempts": 6, "completed": completed, "failed_or_unprocessed": 6-completed,
                "failure_counts": dict(failures)}, "provided_binary": _summary(confusion),
            "pipeline_start_to_result_ms_not_upload_or_ui": _distribution(timings),
            "upload_start_and_usable_browser_latency": "UNMEASURED_BY_DIRECT_PIPELINE"}
    if len(references) != 6:
        blockers["provided_binary_reference_incomplete"] += 1
    for key in ("family_frozen_before_baseline", "all_arm_plans_frozen_before_respective_execution",
                "actual_root_releases_and_exposure_reservations_verified", "validation_reuse_disclosed"):
        if metadata.get(key) is not True:
            blockers[key + "_unverified"] += 1
    if metadata.get("final_check_used") is not False:
        blockers["final_check_exclusion_unverified"] += 1
    real = metadata.get("inference_kind") == "real_local"
    report = {"schema_version": "forte.qwen38-quality-family-comparison.v1",
        "status": "SYNTHETIC_ONLY" if metadata.get("inference_kind") == "synthetic" else "DEVELOPMENT_DIAGNOSTIC_READY" if real and not blockers else "PARTIAL_OR_PROVENANCE_INCOMPLETE",
        "family_sha256": family_sha, "recipe_sha256": family["recipe_sha256"], "arms": reports,
        "original_terminal_records_sha256": digest(terminals), "provided_labels_sha256": provided_label_digest(labels),
        "blocker_counts": dict(blockers), "metadata_sha256": digest(metadata),
        "claim_limits": {"raw_results_or_original_terminal_bindings_rewritten": False,
            "future_adapter_hash_known_at_baseline_time": False, "validation_is_reused_development": True,
            "human_reverified_binary_or_event_fact_support_legal_boundary_gold": False,
            "model_promotion_authorized": False, "universal_accuracy_claim_permitted": False,
            "final_check_read_or_evaluated": False},
        "limitations": ["Each actual arm plan binds the same family/population/recipe and its own real model identity before that arm executes. The baseline requires no future adapter identity.",
            "All six original source terminals per arm remain in denominators, including failed/missing/insufficient analysis outcomes. No retry success replaces a source attempt.",
            "Provided organizer binary labels are not human-reverified event/type/fact/evidence-support/legal/boundary truth. Validation is reused development data.",
            "Direct pipeline timing excludes upload transport and browser rendering. Native receipt export adds a separate post-result CPU decode.",
            "Artifacts and verification flags establish supplied consistency, not independent proof of execution or model generalization. Retain real root/exposure/runtime/source receipts privately."]}
    report["report_sha256"] = digest(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bind = sub.add_parser("bind-arm")
    for name in ("family", "spec", "role-config", "output"):
        bind.add_argument("--" + name, type=Path, required=True)
    compare = sub.add_parser("compare")
    for name in ("family", "labels", "metadata", "output"):
        compare.add_argument("--" + name, type=Path, required=True)
    compare.add_argument("--arm-plan", type=Path, required=True, action="append")
    compare.add_argument("--terminals", type=Path, required=True, action="append")
    args = parser.parse_args(argv)
    try:
        family = read_json(args.family)
        if args.command == "bind-arm":
            plan = arm_plan(family, read_json(args.spec), role_config_sha256=file_sha(args.role_config))
            write_private_new(args.output, plan)
            summary = {"status": "arm_plan_prepared_not_executed", "source_attempts": 6,
                "maximum_private_model_requests": 208, "model_calls_made": 0}
        else:
            report = compare_family(family, [read_json(path) for path in args.arm_plan],
                [row for path in args.terminals for row in read_records(path)], read_records(args.labels), metadata=read_json(args.metadata))
            write_private_new(args.output, report)
            summary = {"status": report["status"], "required_source_attempts": 18,
                "private_values_printed": False, "model_promotion_authorized": False}
        print(json.dumps(summary, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "quality_family_input_invalid", "error_type": type(error).__name__, "private_values_printed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
