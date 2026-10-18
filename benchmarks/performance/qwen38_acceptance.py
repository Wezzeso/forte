"""Failure-inclusive fresh HTTP acceptance and repeatability; no model imports.

Prepare freezes metadata only. Execute requires an exact, current root release
and exposure reservation. All per-source and per-attempt artifacts stay private.
Report is offline and cannot establish correctness against human-reviewed gold.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlparse

from benchmarks.performance.repeatability import digest, file_digest, require, write_new
from forte.observability.benchmark import (
    UploadStream, sanitized_transport_error, statistics_ms, transfer_response,
)

PRIVATE = Path(__file__).resolve().parents[2] / "benchmarks" / "private"
PLAN_SCHEMA = "forte.qwen38-acceptance-plan.v1"
ARMS = {"baseline_8b", "qwen38_untuned", "qwen38_tuned"}
HEX = re.compile(r"^[a-f0-9]{64}$")
ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def private(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PRIVATE.resolve()), "acceptance_private_path_required")
    return path


def normalized(value):
    """Only job_id fields are excluded; ordering and every other field remain."""
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items() if key != "job_id"}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def valid_digest(value):
    return isinstance(value, str) and bool(HEX.fullmatch(value))


def validate_plan(plan):
    require(plan.get("schema_version") == PLAN_SCHEMA, "acceptance_plan_schema")
    unsigned = {key: value for key, value in plan.items() if key != "conditions_sha256"}
    require(digest(unsigned) == plan.get("conditions_sha256"), "acceptance_plan_changed")
    require(plan.get("arm") in ARMS and 3 <= plan.get("repeats", 0) <= 30,
            "acceptance_arm_or_repeat_count")
    require(valid_digest(plan.get("recipe_sha256")) and valid_digest(plan.get("comparison_plan_sha256")),
            "acceptance_comparison_recipe_required")
    settings = plan["settings"]
    require(settings.get("whole_result_cache") is False and settings.get("concurrency") == 1
            and settings.get("normalization") == ["job_id"], "acceptance_conditions_invalid")
    require(all(type(settings.get(key)) in {int, float} and math.isfinite(settings[key])
                and settings[key] > 0 for key in ("deadline_seconds", "poll_seconds", "request_timeout_seconds")),
            "acceptance_timing_invalid")
    parsed = urlparse(settings["base_url"])
    require(parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username
            and not parsed.password and not parsed.query and not parsed.fragment,
            "acceptance_base_url_invalid")
    model = plan["expected_model_identity"]
    require(isinstance(model, dict) and model.get("model_id") and model.get("model_revision")
            and (model.get("adapter_sha256") is None or valid_digest(model["adapter_sha256"])),
            "acceptance_model_identity_required")
    require(isinstance(plan["cases"], list) and plan["cases"], "acceptance_cases_required")
    cases = {}
    hashes = set()
    for case in plan["cases"]:
        require(ID.fullmatch(case.get("case_id", "")) and case["case_id"] not in cases
                and ID.fullmatch(case.get("sample_id", ""))
                and case.get("split") in {"train", "validation"}
                and valid_digest(case.get("content_sha256"))
                and case["content_sha256"] not in hashes
                and type(case.get("source_bytes")) is int and case["source_bytes"] > 0,
                "acceptance_source_identity_invalid")
        cases[case["case_id"]] = case
        hashes.add(case["content_sha256"])
    require(len({case["sample_id"] for case in cases.values()}) == len(cases), "acceptance_sample_id_reused")
    attempts = plan["attempts"]
    require(len(attempts) == len(cases) * plan["repeats"]
            and len({row["attempt_id"] for row in attempts}) == len(attempts), "acceptance_denominator_invalid")
    for case in cases.values():
        rows = [row for row in attempts if row["case_id"] == case["case_id"]]
        require(sorted(row["repeat_index"] for row in rows) == list(range(plan["repeats"])),
                "acceptance_repeat_indices_invalid")
        for row in rows:
            require(re.fullmatch(r"attempt_[0-9a-f]{32}", row["attempt_id"])
                    and all(row.get(key) == case[key] for key in ("sample_id", "content_sha256", "split"))
                    and all(row.get(key) == plan[key] for key in
                            ("arm", "recipe_sha256", "comparison_plan_sha256")), "acceptance_attempt_identity_invalid")
    return plan


def prepare(spec, output, repeats=3):
    """No decoding, private inference, reservations, or HTTP requests occur."""
    require(spec.get("schema_version") == "forte.qwen38-acceptance-spec.v1", "acceptance_spec_schema")
    plan = {key: spec[key] for key in ("arm", "cases", "expected_model_identity", "recipe_sha256",
                                      "comparison_plan_sha256", "settings")}
    plan.update(schema_version=PLAN_SCHEMA, repeats=repeats, attempts=[])
    for case in plan["cases"]:
        path = Path(case["owned_path"])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == case["source_bytes"]
                and file_digest(path) == case["content_sha256"], "acceptance_source_bytes_changed")
        case["owned_path"] = str(path.resolve())
        for index in range(repeats):
            plan["attempts"].append({"attempt_id": "attempt_" + secrets.token_hex(16),
                "case_id": case["case_id"], "repeat_index": index,
                **{key: case[key] for key in ("sample_id", "content_sha256", "split")},
                **{key: plan[key] for key in ("arm", "recipe_sha256", "comparison_plan_sha256")}})
    plan["conditions_sha256"] = digest(plan)
    validate_plan(plan)
    write_new(private(output), plan)
    return {"status": "prepared_not_launched", "expected_attempts": len(plan["attempts"]),
            "model_calls_made": 0, "conditions_sha256": plan["conditions_sha256"]}


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "acceptance_release_timezone_required")
    return parsed.astimezone(timezone.utc)


def validate_authorization(plan, release, reservation, *, plan_sha256, release_sha256, now=None):
    """Fail closed before constructing a client and immediately before each POST."""
    now = now or datetime.now(timezone.utc)
    from training.qwen38_preflight import FIXED_STOP_UTC, validate_attempt_reservation
    require(release.get("schema_version") == "forte.qwen38-root-release.v1"
            and release.get("released_by_root") is True and release.get("uploads_released") is True
            and release.get("phase") == "acceptance"
            and release.get("attempt_plan_sha256") == plan_sha256
            and release.get("maximum_attempts") == len(plan["attempts"])
            and release.get("exclusive_idle_gpu_handoff_verified") is True
            and valid_digest(release.get("idle_handoff_receipt_sha256"))
            and release.get("expected_model_identity") == plan["expected_model_identity"],
            "acceptance_exact_root_release_required")
    expiry = timestamp(release["expires_at_utc"])
    require(timestamp(release["not_before_utc"]) <= now < expiry <= FIXED_STOP_UTC,
            "acceptance_release_expired_or_not_started")
    require(type(release.get("maximum_execution_seconds")) is int and release["maximum_execution_seconds"] > 0,
            "acceptance_execution_bound_required")
    validate_attempt_reservation(reservation, plan, plan_sha256=plan_sha256, phase="acceptance", now=now)
    require(reservation.get("release_sha256") == release_sha256
            and reservation.get("config_sha256") == release.get("config_sha256")
            and timestamp(reservation["expires_at_utc"]) == expiry
            and all(valid_digest(release.get(key)) and reservation.get(key) == release[key]
                    for key in ("source_archive_sha256", "runtime_image_sha256", "base_manifest_sha256")),
            "acceptance_exact_exposure_reservation_required")


async def run_attempt(client, path, attempt, settings, key):
    """One fresh POST, followed only by its own job polls; no answer cache."""
    row = {**attempt, "status": "failed", "post_attempted": False, "job_id": None,
        "timing_origin": "client_upload_start_to_full_result_byte",
        "upload_start_to_full_result_ms": None, "public_contract_validated": False,
        "upload_start_to_usable_result_ms": None, "ui_timing_origin": None,
        "poll_count": 0, "post": None, "polls": [], "complete_result": None,
        "error_code": None, "verdict": None, "violations": []}
    body = UploadStream(path)
    started = time.perf_counter_ns()
    deadline = asyncio.get_running_loop().time() + settings["deadline_seconds"]
    try:
        row["post_attempted"] = True
        status, headers, payload, measurement = await transfer_response(client, "POST",
            settings["base_url"].rstrip("/") + "/api/v1/analyze", content=body,
            absolute_deadline=deadline, headers={"X-API-Key": key,
                "Content-Type": "multipart/form-data; boundary=" + body.boundary,
                "Content-Length": str(body.length)})
        row["post"] = {**measurement, "http_status": status}
        require(status == 202, "acceptance_post_http_error")
        submitted = json.loads(payload)
        job_id = submitted.get("job_id")
        require(isinstance(job_id, str) and ID.fullmatch(job_id), "acceptance_invalid_job_id")
        row["job_id"] = job_id
        while asyncio.get_running_loop().time() < deadline:
            status, headers, payload, measurement = await transfer_response(client, "GET",
                settings["base_url"].rstrip("/") + "/api/v1/jobs/" + job_id,
                absolute_deadline=deadline, headers={"X-API-Key": key})
            full_byte_elapsed = (time.perf_counter_ns() - started) / 1e6
            row["poll_count"] += 1
            row["polls"].append({**measurement, "http_status": status})
            require(status == 200, "acceptance_poll_http_error")
            parsed = json.loads(payload)
            require(parsed.get("job_id") == job_id, "acceptance_job_identity_mismatch")
            if parsed.get("status") == "processing":
                await asyncio.sleep(min(settings["poll_seconds"], max(0, deadline - asyncio.get_running_loop().time())))
                continue
            require(parsed.get("status") != "failed", "acceptance_job_failed")
            from forte.shared.schemas import CompletedResponse
            CompletedResponse.model_validate(parsed)  # Includes actual JPEG decoding.
            row.update(status="done", public_contract_validated=True, complete_result=parsed,
                verdict=parsed["verdict"], violations=parsed["violations"],
                upload_start_to_full_result_ms=full_byte_elapsed,
                upload_start_to_validated_result_ms=(time.perf_counter_ns() - started) / 1e6,
                normalized_result_sha256=digest(normalized(parsed)),
                evidence_sha256=evidence_hashes(parsed))
            break
        else:
            raise TimeoutError()
    except Exception as error:
        from benchmarks.performance.repeatability import RepeatabilityError
        row["status"] = "timeout" if isinstance(error, TimeoutError) or type(error).__name__.endswith("Timeout") else "failed"
        row["error_code"] = str(error) if isinstance(error, RepeatabilityError) else (
            "acceptance_client_deadline" if row["status"] == "timeout" else "acceptance_transport_or_contract_error")
        row["transport_diagnostic"] = sanitized_transport_error(error)
    finally:
        row["terminal_observed_elapsed_ms"] = (time.perf_counter_ns() - started) / 1e6
        row["upload_iterator"] = body.progress(started)
    return row


def evidence_hashes(result):
    return [{"event_index": event_index, "frame_index": frame_index,
        "timestamp_sec": frame["timestamp_sec"],
        "jpeg_sha256": hashlib.sha256(base64.b64decode(frame["image_base64"], validate=True)).hexdigest()}
        for event_index, event in enumerate(result["violations"])
        for frame_index, frame in enumerate(event["evidence_frames"])]


def bind_server_receipt(plan, row, receipt):
    """Bind private deployment/decode/model/evidence receipts to the exact job.

    This validates receipt consistency; its server provenance remains an
    operator responsibility. Receipt timestamps never replace client latency.
    """
    require(receipt.get("schema_version") == "forte.qwen38-acceptance-server-receipt.v1"
            and receipt.get("attempt_id") == row["attempt_id"] and receipt.get("job_id") == row["job_id"]
            and receipt.get("content_sha256") == row["content_sha256"]
            and receipt.get("conditions_sha256") == plan["conditions_sha256"]
            and receipt.get("recipe_sha256") == plan["recipe_sha256"]
            and receipt.get("normalized_result_sha256") == row["normalized_result_sha256"]
            and receipt.get("actual_model_identity") == plan["expected_model_identity"],
            "acceptance_server_identity_mismatch")
    require(receipt.get("cache_hits") == {"decode": 0, "model": 0, "evidence": 0, "whole_result": 0}
            and all(type(receipt.get("stage_calls", {}).get(key)) is int
                    and receipt["stage_calls"][key] > 0 for key in ("decode", "model", "evidence")),
            "acceptance_fresh_stages_unverified")
    native = receipt["native_source"]
    require(native.get("content_sha256") == row["content_sha256"]
            and native.get("source_complete") is True and native.get("tail_included") is True
            and type(native.get("origin_pts")) is int
            and type(native.get("time_base_num")) is int and native["time_base_num"] > 0
            and type(native.get("time_base_den")) is int and native["time_base_den"] > 0,
            "acceptance_native_identity_invalid")
    frames = native["frames"]
    require(isinstance(frames, list) and frames and type(native.get("decoded_source_frame_count")) is int
            and native["decoded_source_frame_count"] == len(frames)
            and len({frame["frame_id"] for frame in frames}) == len(frames), "acceptance_native_ledger_incomplete")
    for frame in frames:
        require(isinstance(frame.get("frame_id"), str) and 0 < len(frame["frame_id"]) <= 256
                and type(frame.get("source_pts")) is int
                and type(frame.get("timestamp_sec")) in {int, float}
                and math.isfinite(frame["timestamp_sec"]) and frame["timestamp_sec"] >= 0
                and math.isclose(frame["timestamp_sec"],
                (frame["source_pts"] - native["origin_pts"]) * native["time_base_num"] / native["time_base_den"],
                rel_tol=0, abs_tol=1e-6), "acceptance_native_time_mismatch")
    require([frame["source_pts"] for frame in frames] == sorted(frame["source_pts"] for frame in frames),
            "acceptance_native_order_invalid")
    selected = receipt["evidence_frames"]
    require([{key: frame[key] for key in ("event_index", "frame_index", "timestamp_sec", "jpeg_sha256")}
             for frame in selected] == row["evidence_sha256"], "acceptance_evidence_bytes_mismatch")
    by_id = {frame["frame_id"]: frame for frame in frames}
    require(all(frame.get("frame_id") in by_id and frame.get("source_pts") == by_id[frame["frame_id"]]["source_pts"]
                and frame["timestamp_sec"] == by_id[frame["frame_id"]]["timestamp_sec"] for frame in selected),
            "acceptance_evidence_not_native")
    require(isinstance(receipt.get("decision_projection"), dict), "acceptance_decision_projection_required")
    return {"server_receipt_verified": True, "server_receipt_sha256": digest(receipt),
        "model_identity": receipt["actual_model_identity"],
        "native_source_sha256": digest(normalized(native)),
        "decision_sha256": digest(normalized(receipt["decision_projection"])),
        "native_evidence_sha256": digest(normalized(selected))}


async def execute(plan_path, release_path, reservation_path, output, api_key_env):
    plan = validate_plan(read(private(plan_path)))
    release, reservation = read(private(release_path)), read(private(reservation_path))
    plan_hash, release_hash = file_digest(plan_path), file_digest(release_path)
    validate_authorization(plan, release, reservation, plan_sha256=plan_hash, release_sha256=release_hash)
    execution_deadline = time.monotonic() + min(release["maximum_execution_seconds"],
        (timestamp(release["expires_at_utc"]) - datetime.now(timezone.utc)).total_seconds())
    key = os.environ.get(api_key_env)
    require(key, "acceptance_api_key_environment_missing")
    output = private(output)
    require(not output.exists(), "acceptance_run_directory_exists")
    # The same reservation cannot be replayed by selecting a new output folder.
    claim_path = private(plan_path).with_suffix(".execution-claim.json")
    write_new(claim_path, {"conditions_sha256": plan["conditions_sha256"],
        "plan_file_sha256": plan_hash, "release_file_sha256": release_hash,
        "run_directory": str(output), "claimed_at_utc": datetime.now(timezone.utc).isoformat()})
    output.mkdir(parents=True)
    write_new(output / "plan.json", plan)
    write_new(output / "authorization.json", {"release": release, "reservation": reservation})
    import httpx
    cases = {row["case_id"]: row for row in plan["cases"]}
    records = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(plan["settings"]["request_timeout_seconds"]),
                                 follow_redirects=False, trust_env=False) as client:
        for attempt in plan["attempts"]:
            directory = output / attempt["attempt_id"]
            directory.mkdir()
            try:
                require(file_digest(plan_path) == plan_hash and file_digest(release_path) == release_hash,
                        "acceptance_execution_pins_changed")
                validate_authorization(plan, read(private(release_path)), read(private(reservation_path)),
                                       plan_sha256=plan_hash, release_sha256=release_hash)
                case = cases[attempt["case_id"]]
                path = Path(case["owned_path"])
                require(path.is_file() and not path.is_symlink() and path.stat().st_size == case["source_bytes"]
                        and file_digest(path) == case["content_sha256"], "acceptance_source_bytes_changed")
                # A durable intent exists even if the process dies during submission.
                write_new(directory / "submission-intent.json", attempt)
                remaining = execution_deadline - time.monotonic()
                require(remaining > 0, "acceptance_execution_window_expired")
                settings = {**plan["settings"], "deadline_seconds": min(plan["settings"]["deadline_seconds"], remaining)}
                record = await run_attempt(client, path, {**attempt,
                    "expected_model_identity": plan["expected_model_identity"]}, settings, key)
            except Exception:
                record = {**attempt, "status": "not_attempted", "post_attempted": False,
                    "error_code": "acceptance_preflight_failed", "public_contract_validated": False,
                    "expected_model_identity": plan["expected_model_identity"]}
            write_new(directory / "attempt.json", record)
            records.append(record)
    completed = sum(row["status"] == "done" for row in records)
    return {"status": "finished_receipts_required" if completed == len(records) else "incomplete",
        "expected_attempts": len(plan["attempts"]), "completed": completed,
        "failed_or_unattempted": len(records) - completed, "targets": "unverified_until_receipt_report"}


def summarize(plan, records):
    """Missing, duplicate, failed and unverified attempts never shrink scope."""
    expected = {row["attempt_id"]: row for row in plan["attempts"]}
    require(len({row["attempt_id"] for row in records}) == len(records)
            and set(row["attempt_id"] for row in records) <= set(expected), "acceptance_result_identity_conflict")
    by_id = {row["attempt_id"]: row for row in records}
    rows = []
    for attempt_id, attempt in expected.items():
        row = by_id.get(attempt_id, {**attempt, "status": "missing", "post_attempted": False})
        require(all(row.get(key) == value for key, value in attempt.items()), "acceptance_result_plan_mismatch")
        if "expected_model_identity" in row:
            require(row["expected_model_identity"] == plan["expected_model_identity"],
                    "acceptance_expected_model_identity_changed")
        rows.append(row)
    done = [row for row in rows if row.get("status") == "done" and row.get("public_contract_validated")]
    verified = [row for row in done if row.get("server_receipt_verified")]
    values = [row["upload_start_to_full_result_ms"] for row in done]
    require(all(type(value) in {int, float} and math.isfinite(value) and value >= 0 for value in values),
            "acceptance_invalid_result_timing")
    failed = [row for row in rows if row.get("status") in {"failed", "timeout"}]
    job_ids = [row["job_id"] for row in done]
    reused_job_identity = len(set(job_ids)) != len(job_ids)
    complete = len(verified) == len(rows)
    comparisons = []
    for case in plan["cases"]:
        group = [row for row in verified if row["case_id"] == case["case_id"]]
        pairs = list(itertools.combinations(group, 2))
        keys = ("normalized_result_sha256", "native_source_sha256", "decision_sha256", "native_evidence_sha256")
        comparisons.append({"complete": len(group) == plan["repeats"], "verified_attempts": len(group),
            "compared_pairs": len(pairs), "equal_pairs": sum(all(a[key] == b[key] for key in keys) for a, b in pairs)})
    any_difference = any(item["compared_pairs"] != item["equal_pairs"] for item in comparisons)
    over_target = any(value >= 10_000 for value in values) or any(
        row.get("terminal_observed_elapsed_ms", 0) >= 10_000 for row in failed)
    return {"schema_version": "forte.qwen38-acceptance-summary.v1",
        "expected_attempts": len(rows), "post_attempted": sum(bool(row.get("post_attempted")) for row in rows),
        "completed": len(done), "server_verified": len(verified), "failed_or_timeout": len(failed),
        "missing_or_not_attempted": sum(row["status"] in {"missing", "not_attempted"} for row in rows),
        "independent_contents": len(plan["cases"]), "repeats_per_content": plan["repeats"],
        "normalization": ["job_id"], "whole_result_cache_allowed": False,
        "upload_start_to_full_result_ms": statistics_ms(values),
        "less_than_ten_seconds": "failed" if failed or over_target or reused_job_identity else "demonstrated" if complete else "unverified",
        "fresh_repeatability": "failed" if failed or any_difference or reused_job_identity else "demonstrated" if complete else "unverified",
        "fresh_job_identities_distinct": not reused_job_identity,
        "repeatability_cases": comparisons, "quality_against_reviewed_gold": "unverified",
        "scope": "finite released development cases; repeated contents are not independent latency samples",
        "timing_boundary": "client immediately before POST to completed canonical result final byte, including JPEG evidence",
        "server_receipt_provenance": "validated consistency; operator must supply actual private server artifacts",
        "ui_ready_less_than_ten_seconds": "unverified; requires actual upload-origin browser measurement",
        "universal_accuracy_or_repeatability_guarantee": False}


def report(run, private_output, public_output):
    run = private(run)
    plan = validate_plan(read(run / "plan.json"))
    records = []
    for expected in plan["attempts"]:
        directory = run / expected["attempt_id"]
        path = directory / "attempt.json"
        if not path.is_file():
            records.append({**expected, "status": "missing", "post_attempted":
                (directory / "submission-intent.json").is_file(), "public_contract_validated": False,
                "expected_model_identity": plan["expected_model_identity"],
                "error_code": "acceptance_missing_terminal_outcome"})
            continue
        row = read(path)
        row.setdefault("expected_model_identity", plan["expected_model_identity"])
        receipt = directory / "server-receipt.json"
        if row.get("status") == "done" and receipt.is_file():
            try:
                row.update(bind_server_receipt(plan, row, read(receipt)))
            except Exception:
                row.update(server_receipt_verified=False, receipt_error_code="acceptance_server_receipt_invalid")
        records.append(row)
    summary = summarize(plan, records)
    require(Path(public_output).resolve() != Path(private_output).resolve(), "acceptance_report_paths_conflict")
    write_new(private(private_output), {"schema_version": "forte.qwen38-acceptance-terminals.v1",
        "conditions_sha256": plan["conditions_sha256"], "records": records, "summary": summary})
    write_new(public_output, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--spec", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--repeats", type=int, default=3)
    run = sub.add_parser("execute")
    for name in ("plan", "release", "reservation", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--api-key-env", default="FORTE_API_KEY")
    result = sub.add_parser("report")
    for name in ("run", "private-output", "public-output"):
        result.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(read(private(args.spec)), args.output, args.repeats)
        elif args.command == "execute":
            result = asyncio.run(execute(args.plan, args.release, args.reservation, args.output, args.api_key_env))
        else:
            result = report(args.run, args.private_output, args.public_output)
        print(json.dumps(result, ensure_ascii=True))
        if args.command == "execute":
            return int(result["completed"] != result["expected_attempts"])
        return int(args.command == "report" and (result["less_than_ten_seconds"] != "demonstrated"
                                               or result["fresh_repeatability"] != "demonstrated"))
    except Exception:
        print(json.dumps({"status": "failed", "error_code": "acceptance_setup_or_artifact_error",
                          "targets": "unverified"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
