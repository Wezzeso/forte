"""Freeze a NEW ten-attempt grounding pilot; no decoder/model runs on import.

Inspect performs metadata-only duration selection. Root audits native source
clocks before build. Private plans contain source/query material; stdout does not.
This module does not alter previous head or Qwen benchmark artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import secrets

SCHEMA = "forte.grounding-pilot-plan.v1"
REPEAT_QUERY = "CASH_COUNT_VISIBLE"
CLOCK_POLICIES = {"strict_cfr", "native_pts_ticks"}
ROOT = Path(__file__).resolve().parents[2]
HEX64 = re.compile(r"^[a-f0-9]{64}$")


class GroundingPlanError(ValueError):
    """Static diagnostic codes only."""


def require(condition, code):
    if not condition:
        raise GroundingPlanError(code)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def private_path(path):
    path = Path(path).resolve()
    require("private" in path.parts, "grounding_private_output_required")
    return path


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            output.write(raw)
            output.flush()
            import os
            os.fsync(output.fileno())
        # Atomic publication without replacing a prior frozen artifact.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def select_sources(rows, split, *, maximum_duration_sec=32.):
    """Shortest and upper-median eligible TRAIN, solely by duration + stable ID.

    Frame-clock regularity is deliberately not a selection criterion. The
    actual audited clock may subsequently reject a planned attempt, never
    substitute another source. No labels or source filenames are inspected.
    """
    assignments = {}
    for row in split.get("records", []):
        assignments.setdefault(row["content_hash"], set()).add(row["split"])
    training = []
    for row in rows:
        content = row.get("content_sha256")
        require(HEX64.fullmatch(str(content)) is not None, "grounding_source_hash_invalid")
        if row.get("split") != "train":
            continue
        require(assignments.get(content) == {"train"}, "grounding_source_split_mismatch")
        duration = row.get("asset", {}).get("duration_sec")
        require(type(duration) in {int, float} and math.isfinite(duration) and duration > 0,
            "grounding_source_duration_invalid")
        if duration <= maximum_duration_sec:
            training.append(row)
    require(len({row["content_sha256"] for row in training}) == len(training),
        "grounding_source_duplicate_content")
    ordered = sorted(training, key=lambda row: (row["asset"]["duration_sec"], row["sample_id"]))
    require(len(ordered) >= 2, "grounding_two_short_train_sources_unavailable")
    return [ordered[0], ordered[len(ordered) // 2]], len(ordered)


def query_rows(raw):
    from training.grounding.contracts import PhysicalQuery
    rows = raw.get("queries") if isinstance(raw, dict) else raw
    require(isinstance(rows, list) and len(rows) == 4, "grounding_four_fixed_queries_required")
    queries = [PhysicalQuery(**row) for row in rows]
    records = [query.model_dump(mode="json") for query in queries]
    require(len({row["query_id"] for row in records}) == 4
        and REPEAT_QUERY in {row["query_id"] for row in records}, "grounding_query_id_set_invalid")
    return records


def validate_profile(profile):
    require(isinstance(profile, dict) and profile.get("schema_version") == "forte.grounding-pilot-profile.v1",
        "grounding_profile_schema_invalid")
    config = profile.get("runner_config", {})
    require(config.get("clock_policy") in CLOCK_POLICIES
        and config.get("sample_fps") == 2 and config.get("max_sampled_frames") == 66
        and config.get("target_size") == 224 and config.get("max_source_duration_sec") == 32
        and config.get("max_native_frames") == 10000 and config.get("process_memory_fraction") == .10
        and config.get("min_free_bytes") == 9 * 1024 ** 3
        and config.get("device", "cuda:0") == "cuda:0"
        and type(config.get("max_new_tokens")) is int and 1 <= config["max_new_tokens"] <= 128
        and HEX64.fullmatch(str(config.get("manifest_sha256", ""))), "grounding_profile_budget_invalid")
    require(isinstance(config.get("snapshot"), str) and isinstance(config.get("manifest"), str),
        "grounding_profile_local_artifacts_required")
    return config


def code_bindings():
    paths = {Path(__file__), Path(__file__).with_name("grounding_measure.py")}
    for folder in ("src", "serving", "training"):
        for path in (ROOT / folder).rglob("*.py"):
            if not any(part in {"private", "runs", "__pycache__", ".venv"}
                       for part in path.relative_to(ROOT).parts):
                require(not path.is_symlink(), "grounding_code_symlink_invalid")
                paths.add(path)
    paths.update(ROOT / name for name in ("pyproject.toml", "uv.lock") if (ROOT / name).is_file())
    return {"code:" + path.relative_to(ROOT).as_posix(): {"path": str(path), "sha256": file_sha256(path)}
            for path in sorted(paths)}


def load_verified_inputs(inputs_path, split_path, *, inputs_sha256, split_sha256):
    from training.temporal.extract_features import validate_prepared_inputs
    require(file_sha256(inputs_path) == inputs_sha256 and file_sha256(split_path) == split_sha256,
        "grounding_input_file_pin_mismatch")
    payload, split = read_json(inputs_path), read_json(split_path)
    rows = validate_prepared_inputs(payload)
    require(payload.get("split_manifest_sha256") == split_sha256
        and payload.get("assignments_sha256") == split.get("assignments_sha256"),
        "grounding_native_input_split_binding_mismatch")
    return payload, rows, split


def inspect_inputs(args):
    _, rows, split = load_verified_inputs(args.inputs, args.split,
        inputs_sha256=args.inputs_sha256, split_sha256=args.split_sha256)
    selected, candidates = select_sources(rows, split)
    if args.selection_output:
        write_new(private_path(args.selection_output), {"schema_version": "forte.grounding-pilot-selection.v1",
            "input_manifest_sha256": args.inputs_sha256, "split_sha256": args.split_sha256,
            "selection_policy": "shortest_and_upper_median_train_duration_at_most_32_seconds",
            "records": selected, "labels_consulted": False, "clock_regularity_used_for_selection": False})
    return {"status": "metadata_selection_only", "eligible_train_sources": candidates,
        "selected_sources": 2, "selected_durations_sec": [row["asset"]["duration_sec"] for row in selected],
        "expected_matrix_attempts": 8, "expected_additive_repeat_attempts": 2,
        "expected_total_attempts": 10, "gpu_or_decoder_calls_made": 0,
        "clock_audit": "required_before_freeze", "legal_completion_eligible": False}


def validate_clock_audit(audit, selected, *, input_manifest_sha256, clock_policy):
    require(audit.get("schema_version") == "forte.grounding-clock-audit.v1"
        and audit.get("input_manifest_sha256") == input_manifest_sha256
        and audit.get("clock_policy") == clock_policy, "grounding_clock_audit_binding_invalid")
    rows = audit.get("records", [])
    require(isinstance(rows, list) and len(rows) == 2
        and {row.get("content_sha256") for row in rows} == {row["content_sha256"] for row in selected}
        and all(row.get("complete_native_ledger") is True and type(row.get("cfr")) is bool
            and type(row.get("native_frame_count")) is int and 0 < row["native_frame_count"] <= 10000
            and HEX64.fullmatch(str(row.get("ledger_sha256", ""))) for row in rows),
        "grounding_full_native_clock_audit_required")
    return rows


def bind_source_audits(args):
    """Assemble A15's actual CPU receipts; metadata only, no decoder or model."""
    from forte.shared.schemas import VideoAsset
    from forte.video.batches import SourceFrameIdentity
    from training.grounding.contracts import validate_source_ledger
    selection = read_json(args.selection)
    selected = selection["records"]
    require(selection.get("schema_version") == "forte.grounding-pilot-selection.v1"
        and len(selected) == 2 and len(args.source_audits) == 2,
        "grounding_audit_selection_invalid")
    inputs = {row["content_sha256"]: row for row in selected}
    records = []
    for path in args.source_audits:
        receipt = read_json(path)
        content = receipt.get("content_sha256")
        require(content in inputs and receipt.get("schema_version") == "forte.grounding-source-clock-audit.v1"
            and receipt.get("clock_policy") == args.clock_policy
            and receipt.get("model_inference_performed") is False,
            "grounding_cpu_source_audit_invalid")
        frames = tuple(SourceFrameIdentity(**row) for row in receipt["source_ledger"])
        binding = validate_source_ledger(VideoAsset.model_validate(inputs[content]["asset"]), frames)
        gaps = {second.source_pts - first.source_pts for first, second in zip(frames, frames[1:])}
        require(binding.source_ledger_sha256 == receipt.get("source_ledger_sha256")
            and len(frames) == receipt.get("native_frame_count"), "grounding_source_audit_ledger_changed")
        records.append({"content_sha256": content, "cfr": len(gaps) == 1,
            "native_frame_count": len(frames), "complete_native_ledger": True,
            "ledger_sha256": binding.source_ledger_sha256,
            "source_audit_path": str(path.resolve()), "source_audit_sha256": file_sha256(path)})
    audit = {"schema_version": "forte.grounding-clock-audit.v1",
        "input_manifest_sha256": selection["input_manifest_sha256"],
        "clock_policy": args.clock_policy, "records": records,
        "processor_clock_proof_separate": True, "gpu_or_model_calls_made": 0}
    validate_clock_audit(audit, selected, input_manifest_sha256=selection["input_manifest_sha256"],
        clock_policy=args.clock_policy)
    write_new(private_path(args.output), audit)
    return {"status": "cpu_source_audits_bound", "audited_sources": 2,
        "cfr_sources": sum(row["cfr"] for row in records),
        "gapped_or_vfr_sources": sum(not row["cfr"] for row in records),
        "audit_sha256": file_sha256(args.output), "gpu_or_model_calls_made": 0}


def freeze_attempts(selected, queries):
    attempts = [{"attempt_id": "ground_" + secrets.token_hex(16), "source_index": source_index,
        "query_id": query["query_id"], "phase": "matrix", "repeat_ordinal": 1}
        for source_index in range(2) for query in queries]
    for ordinal in (2, 3):
        attempts.append({"attempt_id": "ground_" + secrets.token_hex(16), "source_index": 0,
            "query_id": REPEAT_QUERY, "phase": "repeat", "repeat_ordinal": ordinal})
    return attempts


def build_plan(args):
    _, rows, split = load_verified_inputs(args.inputs, args.split,
        inputs_sha256=args.inputs_sha256, split_sha256=args.split_sha256)
    selected, candidates = select_sources(rows, split)
    require(file_sha256(args.queries) == args.queries_sha256 and file_sha256(args.profile) == args.profile_sha256
        and file_sha256(args.clock_audit) == args.clock_audit_sha256,
        "grounding_profile_query_or_audit_pin_mismatch")
    queries, profile = query_rows(read_json(args.queries)), read_json(args.profile)
    config = validate_profile(profile)
    audit = validate_clock_audit(read_json(args.clock_audit), selected,
        input_manifest_sha256=args.inputs_sha256, clock_policy=config["clock_policy"])
    from serving.grounding_pilot import GroundingPilotConfig
    GroundingPilotConfig(**{**config, "snapshot": Path(config["snapshot"]), "manifest": Path(config["manifest"])})
    manifest = Path(config["manifest"])
    require(file_sha256(manifest) == config["manifest_sha256"], "grounding_model_manifest_pin_mismatch")
    for row in selected:
        media = Path(row["asset"]["owned_path"])
        require(media.is_file() and not media.is_symlink() and media.stat().st_size == row["source_bytes"]
            and file_sha256(media) == row["content_sha256"], "grounding_selected_source_bytes_changed")
    artifacts = {"inputs": args.inputs, "split": args.split, "queries": args.queries,
        "profile": args.profile, "clock_audit": args.clock_audit, "model_manifest": manifest}
    if config.get("clock_audit_path"):
        processor_audit = Path(config["clock_audit_path"])
        require(file_sha256(processor_audit) == config.get("clock_audit_sha256"),
            "grounding_processor_clock_proof_pin_mismatch")
        artifacts["processor_clock_audit"] = processor_audit
    for index, row in enumerate(audit):
        if row.get("source_audit_path"):
            source_audit = Path(row["source_audit_path"])
            require(file_sha256(source_audit) == row.get("source_audit_sha256"),
                "grounding_private_source_audit_file_changed")
            artifacts[f"source_audit_{index}"] = source_audit
    bindings = {key: {"path": str(path.resolve()), "sha256": file_sha256(path)} for key, path in artifacts.items()}
    bindings.update(code_bindings())
    output_root = private_path(args.private_root)
    require(not output_root.exists(), "grounding_attempt_root_exists")
    plan = {"schema_version": SCHEMA, "sources": selected, "queries": queries,
        "query_set_sha256": canonical_sha256(queries), "profile": profile,
        "bindings": bindings, "clock_audit_summaries": audit,
        "attempts": freeze_attempts(selected, queries), "private_root": str(output_root),
        "selection_policy": "shortest_and_upper_median_train_duration_at_most_32_seconds",
        "eligible_train_sources": candidates, "labels_consulted": False,
        "model_lifetime": "one_resident_model_for_all_ten_attempts",
        "fresh_source_decode_per_attempt": True, "source_result_cache_allowed": False,
        "budgets": {"overall_sec": 360., "per_call_sec": 30., "minimum_free_gpu_bytes": 9 * 1024 ** 3,
            "per_process_memory_fraction": .10, "expected_attempts": 10},
        "scope": "sparse_physical_grounding_proposals_only",
        "primary_jury_ten_second_evidence_eligible": False, "legal_timing_eligible": False,
        "hundred_percent_accuracy_evidence_eligible": False}
    plan["conditions_sha256"] = canonical_sha256(plan)
    validate_plan(plan)
    write_new(private_path(args.plan), plan)
    safe = {"schema_version": "forte.grounding-pilot-plan-binding.v1", "status": "frozen_not_executed",
        "conditions_sha256": plan["conditions_sha256"], "expected_attempts": 10,
        "matrix_attempts": 8, "additional_repeat_attempts": 2, "repeated_condition_total_calls": 3,
        "train_source_count": 2, "physical_query_count": 4,
        "clock_policy": config["clock_policy"], "audited_cfr_sources": sum(row["cfr"] for row in audit),
        "audited_gapped_or_vfr_sources": sum(not row["cfr"] for row in audit),
        "gpu_or_model_calls_made": 0, "legal_completion_eligible": False}
    if args.safe_binding:
        write_new(args.safe_binding, safe)
    return safe


def validate_plan(plan):
    require(plan.get("schema_version") == SCHEMA and plan.get("conditions_sha256") == canonical_sha256(
        {key: value for key, value in plan.items() if key != "conditions_sha256"}), "grounding_frozen_plan_changed")
    sources, queries, attempts = plan.get("sources", []), plan.get("queries", []), plan.get("attempts", [])
    require(len(sources) == 2 and all(row.get("split") == "train" for row in sources)
        and len({row["content_sha256"] for row in sources}) == 2, "grounding_plan_train_sources_invalid")
    require(len(queries) == 4 and len({row["query_id"] for row in queries}) == 4
        and canonical_sha256(queries) == plan.get("query_set_sha256"), "grounding_frozen_query_set_changed")
    require(len(attempts) == 10 and len({row.get("attempt_id") for row in attempts}) == 10
        and all(re.fullmatch(r"ground_[a-f0-9]{32}", row.get("attempt_id", "")) for row in attempts),
        "grounding_attempt_denominator_invalid")
    matrix, repeats = attempts[:8], attempts[8:]
    require([(row["source_index"], row["query_id"], row["phase"]) for row in matrix] ==
        [(index, query["query_id"], "matrix") for index in range(2) for query in queries]
        and all(row["repeat_ordinal"] == 1 for row in matrix)
        and [(row["source_index"], row["query_id"], row["phase"], row["repeat_ordinal"]) for row in repeats]
        == [(0, REPEAT_QUERY, "repeat", 2), (0, REPEAT_QUERY, "repeat", 3)],
        "grounding_frozen_matrix_or_repeats_changed")
    require(plan.get("model_lifetime") == "one_resident_model_for_all_ten_attempts"
        and plan.get("fresh_source_decode_per_attempt") is True
        and plan.get("source_result_cache_allowed") is False
        and plan.get("legal_timing_eligible") is False
        and plan.get("budgets") == {"overall_sec": 360., "per_call_sec": 30.,
            "minimum_free_gpu_bytes": 9 * 1024 ** 3, "per_process_memory_fraction": .10, "expected_attempts": 10},
        "grounding_scope_or_budget_changed")
    validate_profile(plan["profile"])
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "build"):
        sub = commands.add_parser(command)
        for name in ("inputs", "split"):
            sub.add_argument("--" + name, type=Path, required=True)
            sub.add_argument("--" + name + "-sha256", required=True)
        if command == "inspect":
            sub.add_argument("--selection-output", type=Path)
        else:
            for name in ("queries", "profile", "clock-audit"):
                sub.add_argument("--" + name, type=Path, required=True)
                sub.add_argument("--" + name + "-sha256", required=True)
            sub.add_argument("--plan", type=Path, required=True)
            sub.add_argument("--private-root", type=Path, required=True)
            sub.add_argument("--safe-binding", type=Path)
    bind = commands.add_parser("bind-audits", help="bind two A15 CPU source-ledger audits; no decode/model")
    bind.add_argument("--selection", type=Path, required=True)
    bind.add_argument("--source-audits", type=Path, nargs=2, required=True)
    bind.add_argument("--clock-policy", choices=sorted(CLOCK_POLICIES), required=True)
    bind.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = inspect_inputs(args) if args.command == "inspect" else bind_source_audits(args) \
            if args.command == "bind-audits" else build_plan(args)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_code": str(error)
            if isinstance(error, GroundingPlanError) else "grounding_plan_operation_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
