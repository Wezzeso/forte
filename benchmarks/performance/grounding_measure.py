"""NEW bounded TimeLens grounding pilot; source/query/raw outputs stay private.

One resident model handles exactly the frozen ten-attempt denominator. This
runner is never part of production inference and cannot establish jury timing,
legal correctness or all-frame event visibility from sparse proposals.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from benchmarks.performance.grounding_plan import (GroundingPlanError, REPEAT_QUERY,
    canonical_sha256, file_sha256, private_path, read_json, require, validate_plan, write_new)

TIMING_KEYS = ("cold_load_sec", "source_verification_sec", "decode_sec", "sampling_sec",
    "processor_sec", "h2d_sec", "transfer_sec", "generate_sec", "first_generated_token_sec", "parse_bind_sec", "total_sec",
    "observed_call_wall_sec", "artifact_write_sec")
GPU_KEYS = ("free_bytes_before", "free_bytes_after", "allocated_bytes", "reserved_bytes",
    "peak_allocated_bytes", "peak_reserved_bytes", "free_bytes", "total_bytes")


def json_value(value):
    """Serialize private contract objects without invoking object reprs."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    require(value is None or type(value) in {str, int, float, bool}, "grounding_private_serialization_type_invalid")
    require(type(value) is not float or math.isfinite(value), "grounding_private_serialization_nonfinite")
    return value


def numeric_fields(value, names):
    value = value if isinstance(value, dict) else {}
    return {key: float(value[key]) for key in names if type(value.get(key)) in {int, float}
            and math.isfinite(value[key]) and value[key] >= 0}


def validate_bindings(plan):
    validate_plan(plan)
    require(all(Path(binding["path"]).is_file() and not Path(binding["path"]).is_symlink()
        and file_sha256(binding["path"]) == binding["sha256"] for binding in plan["bindings"].values()),
        "grounding_pinned_artifact_changed")
    from benchmarks.performance.grounding_plan import query_rows, select_sources, validate_clock_audit
    inputs = read_json(plan["bindings"]["inputs"]["path"])
    split = read_json(plan["bindings"]["split"]["path"])
    selected, _ = select_sources(inputs["records"], split)
    require(canonical_sha256(selected) == canonical_sha256(plan["sources"])
        and query_rows(read_json(plan["bindings"]["queries"]["path"])) == plan["queries"]
        and read_json(plan["bindings"]["profile"]["path"]) == plan["profile"],
        "grounding_plan_input_query_or_profile_rebound")
    validate_clock_audit(read_json(plan["bindings"]["clock_audit"]["path"]), selected,
        input_manifest_sha256=plan["bindings"]["inputs"]["sha256"],
        clock_policy=plan["profile"]["runner_config"]["clock_policy"])
    for row in plan["sources"]:
        media = Path(row["asset"]["owned_path"])
        require(media.is_file() and not media.is_symlink() and media.stat().st_size == row["source_bytes"]
            and file_sha256(media) == row["content_sha256"], "grounding_source_bytes_changed")


def reserve(args):
    """Append central reservations before transferring the receipt to GPU host."""
    from scripts.exposure_registry import append_exposures, read_registry
    plan = validate_plan(read_json(args.plan))
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    for attempt in plan["attempts"]:
        content = plan["sources"][attempt["source_index"]]["content_sha256"]
        append_exposures(args.registry, rows, {content},
            trial_id=attempt["attempt_id"].replace("_", "-"), phase="local-trial", purpose="local_model_attempt")
    state = read_registry(args.registry, rows)
    receipt = {"schema_version": "forte.grounding-pilot-exposure.v1", "registered": True,
        "conditions_sha256": plan["conditions_sha256"],
        "attempt_ids": [row["attempt_id"] for row in plan["attempts"]],
        "source_content_sha256": [row["content_sha256"] for row in plan["sources"]],
        "registry_tip_sha256": state["tip_sha256"], "registry_event_count": state["event_count"]}
    write_new(private_path(args.receipt), receipt)
    return {"status": "reserved_not_executed", "expected_attempts": 10,
        "receipt_sha256": file_sha256(args.receipt), "model_calls_made": 0}


def attempt_record(planned, result, elapsed):
    """Keep raw generation failures comparable without calling them successes."""
    raw = result.private_dict()
    raw = json_value(raw)
    attestation = raw.get("attestation", {})
    calls = {key: attestation.get(key) for key in
        ("decode_calls", "processor_calls", "generation_calls", "source_result_cache_hits")}
    status = raw.get("status")
    valid_status = status in {"done", "failed"}
    ledger = raw.get("source_ledger")
    fresh = calls["decode_calls"] == 1 and calls["source_result_cache_hits"] == 0 \
        and isinstance(ledger, list) and len(ledger) >= 2
    generated = calls["generation_calls"] == 1
    tokens = raw.get("generated_token_ids")
    token_valid = isinstance(tokens, list) and all(type(token) is int and token >= 0 for token in tokens)
    response = raw.get("raw_response")
    raw_result = raw.get("raw_result")
    intervals = raw_result.get("intervals") if isinstance(raw_result, dict) else None
    parsed = status == "done" and isinstance(intervals, list)
    within_budget = elapsed <= 30.
    record = {"attempt_id": planned["attempt_id"], "source_index": planned["source_index"],
        "query_id": planned["query_id"], "phase": planned["phase"], "repeat_ordinal": planned["repeat_ordinal"],
        "status": status if valid_status and within_budget else "failed",
        "error_code": raw.get("error_code") if within_budget else "grounding_call_wall_budget_exceeded",
        "actual_call_started": True, "fresh_decode_verified": fresh,
        "raw_tokens_available": generated and token_valid,
        "raw_text_available": generated and isinstance(response, str),
        "strict_intervals_available": parsed, "raw_tokens_sha256": canonical_sha256(tokens) if generated and token_valid else None,
        "raw_text_sha256": canonical_sha256(response) if generated and isinstance(response, str) else None,
        "semantic_intervals_sha256": canonical_sha256(intervals) if parsed else None,
        "native_source_ledger_sha256": canonical_sha256(raw.get("source_ledger")) if fresh else None,
        "proposal_count": len(intervals) if parsed else None,
        "token_counts": {"generated_tokens": len(tokens) if generated and token_valid else None,
            "prompt_tokens": raw.get("timings", {}).get("prompt_tokens")},
        "no_grounding_proposal": parsed and len(intervals) == 0,
        "stage_calls": calls, "timings": {**numeric_fields(raw.get("timings"), TIMING_KEYS),
            "observed_call_wall_sec": elapsed},
        "gpu_stats": numeric_fields(raw.get("gpu_stats"), GPU_KEYS),
        "within_per_call_wall_budget": within_budget, "private_result": raw}
    return record


def failed_record(planned, code, *, started=False, elapsed=None):
    return {**planned, "status": "failed", "error_code": code, "actual_call_started": started,
        "fresh_decode_verified": False, "raw_tokens_available": False, "raw_text_available": False,
        "strict_intervals_available": False, "proposal_count": None, "no_grounding_proposal": False,
        "stage_calls": {key: None if started else 0 for key in
            ("decode_calls", "processor_calls", "generation_calls", "source_result_cache_hits")},
        "timings": {"observed_call_wall_sec": elapsed} if elapsed is not None else {},
        "gpu_stats": {}, "within_per_call_wall_budget": elapsed is None or elapsed <= 30.}


def repeat_comparisons(plan, records):
    planned_repeat_ids = {row["attempt_id"] for row in plan["attempts"]
        if row["source_index"] == 0 and row["query_id"] == REPEAT_QUERY}
    repeat = [row for row in records if row["attempt_id"] in planned_repeat_ids]
    comparisons = []
    for first, second in itertools.combinations(repeat, 2):
        row = {"first_attempt_id": first["attempt_id"], "second_attempt_id": second["attempt_id"]}
        for name, available, checksum in (
            ("raw_tokens_equal", "raw_tokens_available", "raw_tokens_sha256"),
            ("raw_text_equal", "raw_text_available", "raw_text_sha256"),
            ("semantic_intervals_equal", "strict_intervals_available", "semantic_intervals_sha256"),
            ("native_source_ledger_equal", "fresh_decode_verified", "native_source_ledger_sha256")):
            row[name] = first.get(checksum) == second.get(checksum) if first.get(available) \
                and second.get(available) and first.get(checksum) is not None and second.get(checksum) is not None else None
        comparisons.append(row)
    return comparisons


def summarize(plan, records, *, cold_load, overall_elapsed, load_calls):
    validate_plan(plan)
    expected = [row["attempt_id"] for row in plan["attempts"]]
    require(len({row["attempt_id"] for row in records}) == len(records)
        and all(row["attempt_id"] in expected for row in records), "grounding_unplanned_or_duplicate_result")
    comparisons = repeat_comparisons(plan, records)
    metrics = ("raw_tokens_equal", "raw_text_equal", "semantic_intervals_equal", "native_source_ledger_equal")
    summary = {"schema_version": "forte.grounding-pilot-report.v1",
        "execution_status": "all_planned_attempts_recorded" if len(records) == 10 else "incomplete",
        "conditions_sha256": plan["conditions_sha256"], "scope": "sparse_physical_grounding_proposals_only",
        "expected_attempts": 10, "recorded_attempts": len(records), "missing_attempts": 10 - len(records),
        "actual_started_calls": sum(row.get("actual_call_started") is True for row in records),
        "done_attempts": sum(row.get("status") == "done" for row in records),
        "failed_attempts": sum(row.get("status") != "done" for row in records),
        "fresh_decode_verified_attempts": sum(row.get("fresh_decode_verified") is True for row in records),
        "strictly_parsed_attempts": sum(row.get("strict_intervals_available") is True for row in records),
        "no_grounding_proposal_attempts": sum(row.get("no_grounding_proposal") is True for row in records),
        "actual_stage_call_counts": {key: sum(row.get("stage_calls", {}).get(key, 0)
            if type(row.get("stage_calls", {}).get(key)) is int else 0 for row in records)
            for key in ("decode_calls", "processor_calls", "generation_calls", "source_result_cache_hits")},
        "stage_call_count_unknown_attempts": {key: sum(type(row.get("stage_calls", {}).get(key)) is not int
            for row in records) for key in ("decode_calls", "processor_calls", "generation_calls", "source_result_cache_hits")},
        "model_load_calls": load_calls, "model_lifetime": "one_resident_model_for_all_ten_attempts",
        "model_token_counts": {key: sum(row.get("token_counts", {}).get(key, 0)
            if type(row.get("token_counts", {}).get(key)) in {int, float}
                and math.isfinite(row["token_counts"][key]) and row["token_counts"][key] >= 0 else 0
                for row in records) for key in ("prompt_tokens", "generated_tokens")},
        "static_error_code_counts": dict(Counter(row.get("error_code") if isinstance(row.get("error_code"), str)
            and re.fullmatch(r"grounding_[a-z0-9_]{1,100}", row["error_code"]) else "unclassified_failure"
            for row in records if row.get("status") != "done")),
        "clock_policy": plan["profile"]["runner_config"]["clock_policy"],
        "sample_fps": 2., "source_tail_required": True, "target_size": 224,
        "cold_load_observed_wall_sec": cold_load, "overall_observed_wall_sec": overall_elapsed,
        "overall_budget_sec": 360., "per_call_budget_sec": 30.,
        "memory_fraction": .10, "memory_fraction_cap_scope": "PyTorch_caching_allocator_only",
        "minimum_free_gpu_bytes_preflight": 9 * 1024 ** 3,
        "overall_budget_exceeded": overall_elapsed > 360.,
        "per_call_budget_overrun_attempts": sum(row.get("within_per_call_wall_budget") is False for row in records),
        "stage_wall_seconds": {key: [row["timings"][key] for row in records if key in row.get("timings", {})]
            for key in TIMING_KEYS},
        "gpu_metrics": {key: [row["gpu_stats"][key] for row in records if key in row.get("gpu_stats", {})]
            for key in GPU_KEYS},
        "repeat_expected_calls": 3, "repeat_expected_pairs": 3,
        "repeat_comparable_pairs": {key: sum(type(row.get(key)) is bool for row in comparisons) for key in metrics},
        "repeat_differing_pairs": {key: sum(row.get(key) is False for row in comparisons) for key in metrics},
        "repeat_unavailable_pairs": {key: 3 - sum(type(row.get(key)) is bool for row in comparisons) for key in metrics},
        "sparse_coverage_limitation": "Two-fps plus original tail only supports proposals; unseen frames and true onset are unverified.",
        "empty_intervals_mean_confirmed_no_violation": False,
        "primary_jury_ten_second_evidence_eligible": False, "legal_timing_eligible": False,
        "hundred_percent_accuracy_evidence_eligible": False, "universal_repeatability_guarantee": False,
        "quality_and_full_public_service_latency": "UNMEASURED"}
    return summary, comparisons


def publish_state(path, *, phase, attempt_id=None):
    """Mutable private supervisor metadata, published before entering CUDA."""
    raw = {"phase": phase, "attempt_id": attempt_id, "started_monotonic": time.monotonic()}
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(json.dumps(raw), encoding="utf-8")
    os.replace(temporary, path)


def run_resident_session(plan, *, root, state_path, overall_deadline):
    """One resident child; the parent enforces deadlines even inside CUDA/load."""
    from serving.grounding_pilot import GroundingPilot, GroundingPilotConfig
    from training.grounding.contracts import PhysicalQuery
    from forte.shared.schemas import VideoAsset
    config = plan["profile"]["runner_config"]
    pilot = GroundingPilot(GroundingPilotConfig(**{**config, "snapshot": Path(config["snapshot"]),
        "manifest": Path(config["manifest"])}))
    records, load_attestation, load_calls, cold_load, ready = [], None, 0, None, False
    storage_healthy = True
    try:
        publish_state(state_path, phase="loading")
        load_calls += 1
        cold_started = time.monotonic()
        try:
            load_attestation = pilot.load()
            ready = True
        except Exception:
            ready = False
        cold_load = time.monotonic() - cold_started
        try:
            write_new(root / "session-load.private.json", {"load_attestation": json_value(load_attestation),
                "load_function_calls": load_calls, "cold_load_observed_wall_sec": cold_load, "ready": ready})
        except Exception:
            storage_healthy = False
        queries = {row["query_id"]: PhysicalQuery(**row) for row in plan["queries"]}
        expected_ledgers = {row["content_sha256"]: row["ledger_sha256"] for row in plan["clock_audit_summaries"]}
        for planned in plan["attempts"]:
            if not storage_healthy:
                record = failed_record(planned, "grounding_private_artifact_storage_unavailable")
            elif not ready:
                record = failed_record(planned, "grounding_resident_load_unavailable")
            elif time.monotonic() >= overall_deadline:
                record = failed_record(planned, "grounding_overall_budget_exhausted")
            else:
                asset = VideoAsset.model_validate(plan["sources"][planned["source_index"]]["asset"])
                call_started = time.monotonic()
                try:
                    publish_state(state_path, phase="running", attempt_id=planned["attempt_id"])
                    result = pilot.run(asset, queries[planned["query_id"]],
                        deadline=min(overall_deadline, call_started + 30.), cancel_event=threading.Event(),
                        expected_source_ledger_sha256=expected_ledgers[asset.content_hash])
                    record = attempt_record(planned, result, time.monotonic() - call_started)
                except Exception:
                    record = failed_record(planned, "grounding_runner_exception", started=True,
                        elapsed=time.monotonic() - call_started)
            artifact_started = time.monotonic()
            records.append(record)
            try:
                publish_state(state_path, phase="writing", attempt_id=planned["attempt_id"])
                write_new(root / (planned["attempt_id"] + ".json"), record)
                record["private_attempt_artifact_written"] = True
            except Exception:
                record["private_attempt_artifact_written"] = False
                record["status"] = "failed"
                record["error_code"] = "grounding_private_attempt_artifact_failed"
                storage_healthy = False
            record["timings"]["artifact_write_sec"] = time.monotonic() - artifact_started
    finally:
        try:
            pilot.close()
        except Exception:
            pass
    return {"load_attestation": json_value(load_attestation), "load_function_calls": load_calls,
        "cold_load_observed_wall_sec": cold_load, "process_id": os.getpid(), "records": records}


def watch_process(command, state_path, allowed_attempts, *, overall_sec=360., per_call_sec=30., poll_sec=.05):
    """Own process-group watchdog; never targets the pre-existing Qwen service."""
    started, reason, current, state = time.monotonic(), None, None, {}
    child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=os.name != "nt", creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    while child.poll() is None:
        now = time.monotonic()
        try:
            candidate = read_json(state_path)
            if candidate.get("phase") in {"loading", "running", "writing", "complete"}:
                state = candidate
        except Exception:
            pass
        current = state.get("attempt_id") if state.get("attempt_id") in allowed_attempts else None
        call_start = state.get("started_monotonic")
        if now - started >= overall_sec:
            reason = "grounding_supervisor_overall_timeout"
        elif state.get("phase") == "running" and current is not None and type(call_start) in {int, float} \
                and math.isfinite(call_start) and 0 <= call_start <= now and now - call_start >= per_call_sec:
            reason = "grounding_supervisor_call_timeout"
        if reason:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            break
        time.sleep(poll_sec)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return {"termination_code": reason, "active_attempt_id": current, "last_state": state,
        "child_exit_code": child.poll(), "process_termination_confirmed": child.poll() is not None,
        "observed_wall_sec": time.monotonic() - started, "overall_limit_sec": overall_sec,
        "per_call_limit_sec": per_call_sec, "poll_interval_sec": poll_sec}


def reconcile_records(plan, root, supervision):
    """Every preallocated ID receives a terminal record after any child exit."""
    session = {}
    try:
        session = read_json(root / "session-results.private.json")
    except Exception:
        pass
    found = {row.get("attempt_id"): row for row in session.get("records", []) if isinstance(row, dict)}
    active = supervision.get("active_attempt_id")
    expected_ids = [row["attempt_id"] for row in plan["attempts"]]
    active_index = expected_ids.index(active) if active in expected_ids else -1
    records = []
    for index, planned in enumerate(plan["attempts"]):
        record = found.get(planned["attempt_id"])
        if record is None:
            try:
                candidate = read_json(root / (planned["attempt_id"] + ".json"))
                if candidate.get("attempt_id") == planned["attempt_id"]:
                    record = candidate
                    record["private_attempt_artifact_written"] = True
            except Exception:
                pass
        if not isinstance(record, dict) or record.get("status") not in {"done", "failed"}:
            started = index == active_index and supervision.get("last_state", {}).get("phase") == "running"
            code = supervision.get("termination_code") if started else "grounding_resident_terminated_before_attempt"
            record = failed_record(planned, code or "grounding_attempt_artifact_unavailable", started=started)
            record["private_attempt_artifact_written"] = False
            # Earlier missing records may have executed; unknown is never zero.
            if index < active_index:
                record["stage_calls"] = {key: None for key in record["stage_calls"]}
                record["actual_call_started"] = None
        records.append(record)
    load = {}
    try:
        load = read_json(root / "session-load.private.json")
    except Exception:
        pass
    return records, {**load, **{key: value for key, value in session.items() if key != "records"}}


def finish_report(plan, records, session, supervision):
    load_calls = session.get("load_function_calls")
    load_attestation = session.get("load_attestation")
    safe, comparisons = summarize(plan, records, cold_load=session.get("cold_load_observed_wall_sec"),
        overall_elapsed=supervision["observed_wall_sec"], load_calls=load_calls)
    safe["resident_load_function_calls"] = load_calls
    safe["model_load_calls"] = load_attestation.get("model_load_calls") if isinstance(load_attestation, dict) else None
    safe["supervisor"] = {key: supervision.get(key) for key in ("termination_code", "child_exit_code",
        "process_termination_confirmed", "overall_limit_sec", "per_call_limit_sec", "poll_interval_sec")}
    safe["missing_private_attempt_artifacts"] = sum(row.get("private_attempt_artifact_written") is not True for row in records)
    attested = load_attestation if isinstance(load_attestation, dict) else {}
    safe["resident_runtime"] = {"model_id": "MCG-NJU/TimeLens2-2B",
        "model_revision": attested.get("model_revision") if re.fullmatch(r"[a-f0-9]{40}", str(attested.get("model_revision", ""))) else None,
        "processor_revision": attested.get("processor_revision") if re.fullmatch(r"[a-f0-9]{40}", str(attested.get("processor_revision", ""))) else None,
        "packages": {key: value for key, value in attested.get("packages", {}).items()
            if key in {"torch", "transformers", "torchvision", "numpy", "pillow", "av"}
                and isinstance(value, str) and re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,63}", value)},
        "input_recipe_sha256": attested.get("input_recipe_sha256") if re.fullmatch(r"[a-f0-9]{64}", str(attested.get("input_recipe_sha256", ""))) else None}
    return safe, comparisons


def _private_file(path):
    """Reject links before resolving a child authorization input."""
    original = Path(path).absolute()
    require(original.is_file() and all(not part.is_symlink() for part in (original, *original.parents)),
        "grounding_session_private_file_invalid")
    return private_path(original)


def resident_session(args):
    """Internal subprocess entry; verify parent release before model imports."""
    plan_path = _private_file(args.plan)
    authorization_path = _private_file(args.authorization)
    plan = validate_plan(read_json(plan_path))
    root = private_path(plan["private_root"])
    require(authorization_path == root / "execution-authorization.private.json",
        "grounding_session_authorization_path_invalid")
    authorization = read_json(authorization_path)
    require(authorization.get("schema_version") == "forte.grounding-session-authorization.v1"
        and authorization.get("compute_window_released") is True
        and authorization.get("parent_process_id") == os.getppid()
        and authorization.get("plan_sha256") == file_sha256(plan_path)
        and authorization.get("conditions_sha256") == plan["conditions_sha256"]
        and authorization.get("attempt_ids") == [row["attempt_id"] for row in plan["attempts"]],
        "grounding_parent_session_authorization_required")
    started = authorization.get("overall_started_monotonic")
    deadline = authorization.get("overall_deadline_monotonic")
    now = time.monotonic()
    require(type(started) in {int, float} and type(deadline) in {int, float}
        and math.isfinite(started) and math.isfinite(deadline)
        and 0 <= started <= now < deadline and 0 < deadline - started <= 360.,
        "grounding_session_authorization_expired_or_invalid")
    receipt_path = _private_file(authorization.get("receipt_path", ""))
    receipt = read_json(receipt_path)
    require(file_sha256(receipt_path) == authorization.get("receipt_sha256")
        and receipt.get("registered") is True
        and receipt.get("conditions_sha256") == plan["conditions_sha256"]
        and receipt.get("attempt_ids") == authorization["attempt_ids"]
        and receipt.get("source_content_sha256") == [row["content_sha256"] for row in plan["sources"]],
        "grounding_session_exposure_receipt_mismatch")
    validate_bindings(plan)
    # Atomic, immutable claim prevents replaying one released session.
    write_new(root / "resident-session.started.private.json", {
        "process_id": os.getpid(), "conditions_sha256": plan["conditions_sha256"]})
    require(time.monotonic() < deadline, "grounding_session_authorization_expired_or_invalid")
    state_path = root / "supervisor-state.private.json"
    session = run_resident_session(plan, root=root, state_path=state_path, overall_deadline=deadline)
    write_new(root / "session-results.private.json", session)
    publish_state(state_path, phase="complete")
    return {"status": "resident_session_complete", "recorded_attempts": len(session["records"])}


def execute(args):
    plan = validate_plan(read_json(args.plan))
    require(args.compute_window_released, "grounding_compute_release_required")
    validate_bindings(plan)
    receipt = read_json(private_path(args.receipt))
    require(file_sha256(args.receipt) == args.receipt_sha256 and receipt.get("registered") is True
        and receipt.get("conditions_sha256") == plan["conditions_sha256"]
        and receipt.get("attempt_ids") == [row["attempt_id"] for row in plan["attempts"]]
        and receipt.get("source_content_sha256") == [row["content_sha256"] for row in plan["sources"]],
        "grounding_exposure_receipt_mismatch")
    root = private_path(plan["private_root"])
    require(not root.exists() and not args.report.exists(), "grounding_execution_already_exists")
    root.mkdir(parents=True)
    authorization = root / "execution-authorization.private.json"
    started = time.monotonic()
    write_new(authorization, {"schema_version": "forte.grounding-session-authorization.v1",
        "conditions_sha256": plan["conditions_sha256"], "compute_window_released": True,
        "parent_process_id": os.getpid(), "plan_sha256": file_sha256(args.plan),
        "receipt_path": str(Path(args.receipt).resolve()),
        "attempt_ids": receipt["attempt_ids"], "receipt_sha256": args.receipt_sha256,
        "overall_started_monotonic": started, "overall_deadline_monotonic": started + 360.})
    state_path = root / "supervisor-state.private.json"
    command = [sys.executable, "-m", "benchmarks.performance.grounding_measure", "_session",
        "--plan", str(args.plan.resolve()), "--authorization", str(authorization)]
    try:
        supervision = watch_process(command, state_path, set(receipt["attempt_ids"]))
    except Exception:
        supervision = {"termination_code": "grounding_supervisor_launch_failed", "active_attempt_id": None,
            "last_state": {}, "observed_wall_sec": 0., "process_termination_confirmed": None,
            "overall_limit_sec": 360., "per_call_limit_sec": 30., "poll_interval_sec": .05}
    records, session = reconcile_records(plan, root, supervision)
    safe, comparisons = finish_report(plan, records, session, supervision)
    try:
        write_new(root / "private-results.json", {"session": session, "supervision": supervision,
            "records": records, "repeat_comparisons": comparisons})
        safe["private_aggregate_artifact_written"] = True
    except Exception:
        safe["private_aggregate_artifact_written"] = False
    try:
        write_new(args.report, safe)
        safe["aggregate_report_file_written"] = True
    except Exception:
        safe["aggregate_report_file_written"] = False
    return safe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reservation = commands.add_parser("reserve")
    reservation.add_argument("--plan", type=Path, required=True)
    reservation.add_argument("--manifest", type=Path, default=Path("data/manifests/recordings.jsonl"))
    reservation.add_argument("--registry", type=Path, default=Path("benchmarks/private/model-exposure.jsonl"))
    reservation.add_argument("--receipt", type=Path, required=True)
    launch = commands.add_parser("execute")
    launch.add_argument("--plan", type=Path, required=True)
    launch.add_argument("--receipt", type=Path, required=True)
    launch.add_argument("--receipt-sha256", required=True)
    launch.add_argument("--report", type=Path, required=True)
    launch.add_argument("--compute-window-released", action="store_true")
    session = commands.add_parser("_session", help=argparse.SUPPRESS)
    session.add_argument("--plan", type=Path, required=True)
    session.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "_session":
            result = resident_session(args)
        else:
            result = reserve(args) if args.command == "reserve" else execute(args)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_code": str(error)
            if isinstance(error, GroundingPlanError) else "grounding_measurement_operation_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
