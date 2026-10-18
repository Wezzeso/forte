"""Content-private repeatability bookkeeping; importing this opens no model.

The public report contains counts, timings and differences only. Canonical
predictions, source identities, tensor files and source digests remain private.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import secrets
from collections import Counter

SCHEMA = "forte.fresh-repeatability-plan.v1"
VOLATILE_KEYS = frozenset({"job_id", "clip_id", "attempt_id", "request_id",
    "timings", "wallclock", "created_at", "completed_at", "temporary_path",
    "result_path", "input_path", "owned_path", "jpeg_path", "local_path"})
HEX = re.compile(r"^[a-f0-9]{64}$")


class RepeatabilityError(ValueError):
    """Only fixed diagnostic codes belong in this exception."""


def require(value, code):
    if not value:
        raise RepeatabilityError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_new(path, value):
    """Never replace a prior attempt, frozen denominator or evidence artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(value, target, ensure_ascii=False, sort_keys=True, allow_nan=False)
        target.write("\n")
        target.flush()
        import os
        os.fsync(target.fileno())


def canonical_semantics(value):
    """Preserve ordering, uncertainty, limits, rule refs and exact source PTS.

    Only named operational fields are removed. Unknown fields remain part of
    comparison; silently dropping them would conceal a meaningful difference.
    Human review is a separate artifact and must never be passed here.
    """
    if isinstance(value, dict):
        return {key: canonical_semantics(item) for key, item in value.items()
                if key not in VOLATILE_KEYS}
    if isinstance(value, (list, tuple)):
        return [canonical_semantics(item) for item in value]
    require(value is None or type(value) in {str, int, float, bool}, "canonical_value_invalid")
    require(type(value) is not float or math.isfinite(value), "canonical_nonfinite")
    return value


def scalar_max_difference(first, second):
    """Small synthetic/reference utility; production tensors use torch."""
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        require(len(first) == len(second), "numeric_shape_mismatch")
        return max((scalar_max_difference(a, b) for a, b in zip(first, second)), default=0.)
    require(type(first) in {int, float} and type(second) in {int, float}
        and math.isfinite(first) and math.isfinite(second), "numeric_nonfinite_or_type")
    return abs(float(first) - float(second))


def freeze_plan(path, *, source, bindings, settings, runtime, private_root):
    """Freeze the one-source/three-attempt denominator before any execution."""
    require(source.get("split") in {"train", "validation"}, "repeat_final_or_unknown_split")
    require(HEX.fullmatch(str(source.get("content_sha256", "")))
        and type(source.get("source_bytes")) is int and source["source_bytes"] > 0,
        "repeat_source_binding_invalid")
    require(isinstance(bindings, dict) and bindings and all(
        HEX.fullmatch(str(item.get("sha256", ""))) and isinstance(item.get("path"), str)
        for item in bindings.values()), "repeat_artifact_pins_invalid")
    require(settings.get("threshold") == 0. and settings.get("batch_size", 0) in range(1, 129)
        and settings.get("concurrency") == 1 and settings.get("cache_reuse") is False
        and settings.get("model_lifetime") == "persistent_three_calls",
        "repeat_settings_invalid")
    require(runtime.get("deterministic_algorithms") is True
        and runtime.get("cublas_workspace_config") == ":4096:8"
        and runtime.get("allow_tf32") is False and runtime.get("seed") == 20261017,
        "repeat_deterministic_runtime_required")
    plan = {"schema_version": SCHEMA, "scope": "fresh_decode_encoder_head",
        "source": source, "bindings": bindings, "settings": settings,
        "expected_runtime": runtime, "private_root": str(Path(private_root).resolve()),
        "attempts": [{"attempt_id": "repeat_" + secrets.token_hex(16), "ordinal": i}
                     for i in range(1, 4)],
        "independent_source_count": 1, "expected_attempt_count": 3,
        "model_lifetime": "persistent_three_calls",
        "human_review_included": False, "jury_legal_completion_eligible": False,
        "primary_ten_second_evidence_eligible": False}
    plan["conditions_sha256"] = digest(plan)
    write_new(path, plan)
    return plan


def validate_plan(plan):
    require(plan.get("schema_version") == SCHEMA
        and plan.get("scope") == "fresh_decode_encoder_head", "repeat_plan_schema_invalid")
    require(plan.get("conditions_sha256") == digest({key: value for key, value in plan.items()
        if key != "conditions_sha256"}), "repeat_frozen_plan_changed")
    attempts = plan.get("attempts", [])
    require(len(attempts) == 3 and [row.get("ordinal") for row in attempts] == [1, 2, 3]
        and len({row.get("attempt_id") for row in attempts}) == 3
        and all(re.fullmatch(r"repeat_[a-f0-9]{32}", row.get("attempt_id", "")) for row in attempts),
        "repeat_attempt_denominator_invalid")
    require(plan["source"].get("split") in {"train", "validation"}
        and plan["settings"].get("cache_reuse") is False
        and plan["settings"].get("model_lifetime") == "persistent_three_calls"
        and plan["jury_legal_completion_eligible"] is False, "repeat_scope_invalid")
    return plan


def summarize(plan, records, pairwise):
    """Missing, failed and duplicated observations cannot reduce denominator."""
    validate_plan(plan)
    expected = {row["attempt_id"] for row in plan["attempts"]}
    require(len({row["attempt_id"] for row in records}) == len(records)
        and all(row["attempt_id"] in expected for row in records), "repeat_unplanned_or_duplicate_attempt")
    good = [row for row in records if row.get("status") == "done"
            and row.get("fresh_stages_verified") is True
            and row.get("cache_hits") == {"decode": 0, "encoder": 0, "head": 0}
            and row.get("stage_calls", {}).get("decode") == 1
            and row.get("stage_calls", {}).get("head") == 1
            and row.get("stage_calls", {}).get("encoder_batches", 0) > 0]
    attempted = len(records)
    compared_ids = [frozenset((row.get("first_attempt_id"), row.get("second_attempt_id"))) for row in pairwise]
    good_ids = {row["attempt_id"] for row in good}
    require(len(set(compared_ids)) == len(compared_ids) and all(
        len(pair) == 2 and pair <= good_ids for pair in compared_ids), "repeat_duplicate_or_invalid_pair")
    complete = len(good) == 3 and len(pairwise) == 3
    categories = ("source_identity_equal", "decoded_rgb_equal", "feature_bits_equal",
                  "output_bits_equal", "semantic_equal")
    timing_keys = ("encoder_load_sec", "head_load_sec", "decode_active_sec",
                  "encoder_active_sec", "fresh_feature_pipeline_sec", "head_forward_sec",
                  "source_start_to_binary_output_sec", "artifact_write_sec", "whole_attempt_sec", "child_process_wall_sec")
    timings = {key: [row["timings"][key] for row in records if type(
        row.get("timings", {}).get(key)) in {int, float} and math.isfinite(row["timings"][key])]
        for key in timing_keys}
    safe_version = lambda value: value if isinstance(value, str) and re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,63}", value) else None
    runtime = plan["expected_runtime"]
    gpu = runtime.get("gpu_name")
    gpu = gpu if isinstance(gpu, str) and re.fullmatch(r"(?:NVIDIA )?[A-Za-z0-9 ._-]{1,80}", gpu) else None
    return {"schema_version": "forte.fresh-repeatability-report.v1",
        "scope": "fresh_decode_encoder_head", "model_lifetime": "persistent_three_calls",
        "status": "observed" if complete else "incomplete",
        "frozen_conditions_sha256": plan["conditions_sha256"],
        "expected_attempts": 3, "observed_attempts": attempted,
        "successful_fresh_attempts": len(good), "failed_attempts": attempted - len(good),
        "missing_attempts": 3 - attempted, "independent_source_count": 1,
        "hardware": {"gpu_name": gpu, "driver_version": safe_version(runtime.get("gpu_driver"))},
        "packages": {key: safe_version(runtime.get("packages", {}).get(key)) for key in
            ("torch", "torchvision", "transformers", "pillow", "numpy", "av", "safetensors")},
        "budgets": {"batch_size": plan["settings"]["batch_size"], "concurrency": 1,
            "maximum_source_frames": 10000, "source_timeout_sec": plan["settings"].get("source_timeout_sec"),
            "encoder_model_dtype": "bfloat16", "stored_feature_dtype": "float16", "head_dtype": "float32"},
        "failure_phase_counts": dict(Counter(row.get("phase") if row.get("phase") in
            {"binding", "encoder_load", "head_load", "fresh_features", "head_forward", "artifact_write", "persistent_runtime"}
            else "unknown" for row in records if row not in good)),
        "compared_pairs": len(pairwise), "complete_three_attempt_evidence": complete,
        "pair_differences": {key: sum(row.get(key) is False for row in pairwise) for key in categories},
        "pair_comparison_unavailable": {key: 3 - sum(type(row.get(key)) is bool
            for row in pairwise) for key in categories},
        "maximum_feature_abs_difference": max((row["feature_max_abs_difference"] for row in pairwise
            if row.get("feature_max_abs_difference") is not None), default=None),
        "maximum_output_abs_difference": max((row["output_max_abs_difference"] for row in pairwise
            if row.get("output_max_abs_difference") is not None), default=None),
        "stage_wall_seconds": timings,
        "stage_times_may_overlap": True, "model_load_is_reported_separately": True,
        "semantic_scope": "weak_binary_output_only",
        "binary_threshold_policy": "fixed_zero_logit_diagnostic_not_calibrated_deployment",
        "fact_candidate_rule_evidence_comparison": "unavailable_in_binary_only_path",
        "jury_legal_completion_eligible": False, "primary_ten_second_evidence_eligible": False,
        "universal_determinism_or_quality_claim": False,
        "limitation": "Three observed repeats on one source do not guarantee future outputs or legal correctness."}
