"""Compare authenticated team ingress using generated invalid bytes only."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import httpx

from forte.observability.benchmark import UploadStream, sanitized_transport_error, server_timing, transfer_response


ORIGINS = {"public": "https://hackathon.wezeso.me", "brev": "https://forte-review-wd88a7vvt.gobrev.dev"}
SIZE_BYTES = 28_718_927


async def route_check(client, base, key):
    row = {}
    try:
        for name, headers in (("missing_key", {}), ("authenticated", {"X-API-Key": key})):
            response = await client.get(base + "/api/v1/jobs/generated-transport-unknown", headers=headers)
            row[name + "_http_status"] = response.status_code
        row["authenticated_routing_supported"] = row["missing_key_http_status"] in {401, 403} and row["authenticated_http_status"] == 404
    except Exception as exc:
        row.update({"authenticated_routing_supported": False, "transport_diagnostic": sanitized_transport_error(exc)})
    return row


def generated_fixture(sample_id):
    path = Path("tmp") / ("generated-invalid-transport-" + sample_id + ".mov")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(("FORTE_GENERATED_INVALID_MOV_NO_MEDIA_" + sample_id + "\x00").encode("ascii"))
        stream.truncate(SIZE_BYTES)
    from forte.video.probe import probe_video
    try:
        probe_video(path)
    except Exception:
        return path
    raise ValueError("generated_fixture_unexpectedly_valid")


async def one_upload(client, base, key, *, origin_name, mode, sample_id, pace, private_jobs):
    source = generated_fixture(sample_id)
    body = UploadStream(source, bytes_per_second=pace if mode == "paced" else None)
    started = time.perf_counter_ns()
    row = {"sample_id": sample_id, "origin": origin_name, "mode": mode, "input_bytes": SIZE_BYTES,
           "fixture_locally_verified_invalid": True, "outcome": "transport_failure", "http_status": None,
           "terminal_state": "unmeasured", "terminal_failed_without_verdict": False}
    try:
        status, headers, payload, measurements = await transfer_response(
            client, "POST", base + "/api/v1/analyze", content=body,
            absolute_deadline=asyncio.get_running_loop().time() + 180,
            headers={"X-API-Key": key, "Content-Type": "multipart/form-data; boundary=" + body.boundary,
                     "Content-Length": str(body.length)})
        row.update({"http_status": status, "post": measurements,
                    "client_post_total_ms": (time.perf_counter_ns() - started) / 1e6,
                    "asgi_after_body_acceptance_ms": server_timing(headers, "acceptance")})
        if status != 202:
            row["outcome"] = "post_rejected"
            return row
        job_id = json.loads(payload).get("job_id")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            row["outcome"] = "invalid_acknowledgement"
            return row
        private_jobs[sample_id] = job_id
        row["outcome"] = "durably_accepted"
        until = time.monotonic() + 20
        while time.monotonic() < until:
            response = await client.get(base + "/api/v1/jobs/" + job_id, headers={"X-API-Key": key})
            if response.status_code != 200:
                row["terminal_state"] = "poll_http_error"
                break
            state = response.json()
            if state.get("status") in {"failed", "done"}:
                row["terminal_state"] = state["status"]
                row["terminal_failed_without_verdict"] = state["status"] == "failed" and "verdict" not in state and "violations" not in state
                break
            await asyncio.sleep(.25)
    except Exception as exc:
        row["transport_diagnostic"] = sanitized_transport_error(exc)
    finally:
        row["client_observed_total_ms"] = (time.perf_counter_ns() - started) / 1e6
        row["upload_iterator"] = body.progress(started)
        if "client_post_total_ms" not in row:
            row["client_post_attempt_elapsed_ms"] = row["client_observed_total_ms"]
        if body.last_chunk_yielded_ns is not None and "client_post_total_ms" in row:
            row["client_final_chunk_yield_to_ack_ms"] = row["client_post_total_ms"] - (body.last_chunk_yielded_ns - started) / 1e6
    return row


async def execute(args, key):
    origins = list(ORIGINS) if args.origins == "both" else [args.origins]
    checks, rows, jobs = {}, [], {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(60), trust_env=False, follow_redirects=False) as client:
        for origin in origins:
            checks[origin] = await route_check(client, ORIGINS[origin], key)
            if not checks[origin]["authenticated_routing_supported"]:
                continue
            for mode in (("normal", "paced") if args.modes == "both" else (args.modes,)):
                sample_id = f"s{len(rows) + 1:03d}"
                rows.append(await one_upload(client, ORIGINS[origin], key, origin_name=origin, mode=mode,
                                             sample_id=sample_id, pace=args.pace_bytes_per_second, private_jobs=jobs))
                args.private_job_map.parent.mkdir(parents=True, exist_ok=True)
                args.private_job_map.write_text(json.dumps({"schema_version": "1.0", "sample_jobs": jobs}, indent=2), encoding="utf-8")
    return {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "transport-only generated invalid MOV; one upload per selected mode and authenticated team origin",
            "origins": {name: ORIGINS[name] for name in origins}, "route_checks": checks,
            "conditions": {"request_phase_timeouts_sec": 60, "post_absolute_deadline_sec": 180,
                           "diagnostic_condition_label": args.condition_label, "requested_modes": args.modes,
                           "paced_bytes_per_second": args.pace_bytes_per_second, "sequential_uploads": True,
                           "configuration_changed_by_this_client": False, "private_media_used": False,
                           "configuration_change_provenance": "any separately authorized edge change requires deployment-owner evidence; condition label alone is not verification",
                           "model_inference_requested": False, "novel_video_performance_claim": False},
            "measurement_note": "client monotonic iterator yields and response observations; yielded bytes do not prove socket delivery or server receipt; proxy T0 unmeasured",
            "samples": rows,
            "summary": {"attempted": len(rows), "accepted": sum(row["outcome"] == "durably_accepted" for row in rows),
                        "transport_failures": sum(row["outcome"] == "transport_failure" for row in rows),
                        "terminal_failed_without_verdict": sum(row["terminal_failed_without_verdict"] for row in rows)},
            "limits": ["Generated corrupt media cannot establish valid-video or model capacity.",
                       "One observation per condition is not a percentile or a concurrency study.",
                       "A failure not reproduced in this sequential experiment does not explain the original concurrent failure.",
                       "Response header time is a client observation; an earlier response buffered while the request body streamed cannot be excluded."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--origins", choices=(*ORIGINS, "both"), default="both")
    parser.add_argument("--modes", choices=("normal", "paced", "both"), default="both")
    parser.add_argument("--condition-label", default="original-edge")
    parser.add_argument("--pace-bytes-per-second", type=float, default=512 * 1024)
    parser.add_argument("--private-job-map", type=Path, default=Path("benchmarks/private/a18-generated-transport-jobs.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/performance/generated-transport.json"))
    args = parser.parse_args()
    if not 64 * 1024 <= args.pace_bytes_per_second <= 8 * 1024 * 1024:
        parser.error("pace must remain within the bounded diagnostic range")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", args.condition_label):
        parser.error("condition label must be a short technical label")
    if args.output.exists() or args.private_job_map.exists():
        parser.error("diagnostic artifacts already exist; choose new report and private correlation paths")
    if not args.private_job_map.resolve().is_relative_to((Path.cwd() / "benchmarks/private").resolve()):
        parser.error("opaque job correlation must stay private")
    key = args.api_key_file.read_text(encoding="utf-8").strip().splitlines()[0].split(",")[0].strip()
    if not key:
        parser.error("empty team API key file")
    report = asyncio.run(execute(args, key))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]))
    return 0 if report["summary"]["attempted"] and report["summary"]["attempted"] == report["summary"]["terminal_failed_without_verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
