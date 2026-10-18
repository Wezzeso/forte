"""Repeatable HTTP benchmarks. Reports contain timings, never video content.

Run ``python -m forte.observability.benchmark --help``. httpx is a benchmark-only
dependency; production instrumentation has no third-party dependencies.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import statistics
import time

from .metrics import discover_vllm_metric_names


def statistics_ms(values: list[float]) -> dict:
    """Avoid presenting tiny samples as reliable tail estimates."""
    finite = sorted(value for value in values if math.isfinite(value) and value >= 0)
    return {"count": len(finite), "p50_ms": statistics.median(finite) if finite else None,
            "p95_ms": finite[math.ceil(len(finite) * .95) - 1] if len(finite) >= 20 else None,
            "max_ms": max(finite) if finite else None,
            "p95_minimum_samples": 20,
            "p95_method": "empirical_nearest_rank; descriptive, not a population guarantee"}


def runtime_metadata() -> dict:
    versions = {}
    for name in ("httpx", "fastapi", "starlette", "pydantic", "vllm", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"os": platform.system(), "os_release": platform.release(),
            "architecture": platform.machine(), "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(), "client_installed_versions": versions,
            "server_versions": "unmeasured; client environment is not server evidence"}


def content_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def server_timing(headers, name: str) -> float | None:
    if headers.get("x-forte-timing-origin") != "asgi-local":
        return None
    for metric in headers.get("server-timing", "").split(","):
        match = re.fullmatch(r"\s*" + re.escape(name) + r";dur=([0-9.]+)\s*", metric)
        if match:
            value = float(match.group(1))
            return value if math.isfinite(value) else None
    return None


async def transfer_response(client, method: str, url: str, *, limit: int = 64 * 1024 * 1024,
                            absolute_deadline: float | None = None, **kwargs) -> tuple[int, dict, bytes, dict]:
    operation = _transfer_response(client, method, url, limit=limit, **kwargs)
    if absolute_deadline is None:
        return await operation
    async with asyncio.timeout_at(absolute_deadline):
        return await operation


async def _transfer_response(client, method: str, url: str, *, limit: int, **kwargs):
    started = time.perf_counter_ns()
    first = None
    size = 0
    pieces = []
    async with client.stream(method, url, **kwargs) as response:
        headers_at = time.perf_counter_ns()
        async for chunk in response.aiter_bytes():
            if chunk and first is None:
                first = time.perf_counter_ns()
            size += len(chunk)
            if size > limit:
                raise ValueError("response_limit")
            pieces.append(chunk)
        completed = time.perf_counter_ns()
        measurements = {"response_headers_ms": (headers_at - started) / 1e6,
                        "first_body_byte_ms": None if first is None else (first - started) / 1e6,
                        "full_response_ms": (completed - started) / 1e6,
                        "response_bytes": size}
        return response.status_code, response.headers, b"".join(pieces), measurements


class UploadStream:
    """Multipart stream with bounded reads and an explicitly client-side boundary."""

    def __init__(self, path: Path, *, bytes_per_second: float | None = None):
        self.path = path
        if bytes_per_second is not None and (not math.isfinite(bytes_per_second) or bytes_per_second <= 0):
            raise ValueError("invalid_upload_pace")
        self.bytes_per_second = bytes_per_second
        self.boundary = "forte-" + secrets.token_hex(16)
        self.header = (f"--{self.boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                       "filename=\"sample.mov\"\r\nContent-Type: video/quicktime\r\n\r\n").encode()
        self.footer = f"\r\n--{self.boundary}--\r\n".encode()
        self.length = len(self.header) + path.stat().st_size + len(self.footer)
        self.last_chunk_yielded_ns = None
        self.multipart_bytes_yielded = 0
        self.file_bytes_yielded = 0
        self.first_multipart_chunk_ns = None
        self.last_multipart_chunk_ns = None
        self.first_file_chunk_ns = None
        self.last_file_chunk_ns = None
        self.footer_yielded = False

    def _yielded(self, chunk: bytes, *, file_chunk: bool = False):
        now = time.perf_counter_ns()
        if self.first_multipart_chunk_ns is None:
            self.first_multipart_chunk_ns = now
        self.last_multipart_chunk_ns = now
        self.multipart_bytes_yielded += len(chunk)
        if file_chunk:
            if self.first_file_chunk_ns is None:
                self.first_file_chunk_ns = now
            self.last_file_chunk_ns = now
            self.file_bytes_yielded += len(chunk)

    def progress(self, started_ns: int) -> dict:
        def relative(value):
            return None if value is None else (value - started_ns) / 1e6
        return {"multipart_bytes_yielded": self.multipart_bytes_yielded,
                "file_bytes_yielded": self.file_bytes_yielded,
                "expected_multipart_bytes": self.length,
                "first_multipart_chunk_ms": relative(self.first_multipart_chunk_ns),
                "last_multipart_chunk_ms": relative(self.last_multipart_chunk_ns),
                "first_file_chunk_ms": relative(self.first_file_chunk_ns),
                "last_observed_file_chunk_ms": relative(self.last_file_chunk_ns),
                "final_multipart_chunk_ms": relative(self.last_chunk_yielded_ns),
                "multipart_footer_yielded": self.footer_yielded,
                "pace_bytes_per_second": self.bytes_per_second,
                "semantics": "iterator handed bytes to transport; not proof of socket write or server receipt"}

    async def __aiter__(self):
        pace_started = time.monotonic()
        self._yielded(self.header)
        yield self.header
        with self.path.open("rb") as source:
            while True:
                chunk = await asyncio.to_thread(source.read, 256 * 1024)
                if not chunk:
                    break
                if self.bytes_per_second:
                    target = (self.file_bytes_yielded + len(chunk)) / self.bytes_per_second
                    await asyncio.sleep(max(0, target - (time.monotonic() - pace_started)))
                self._yielded(chunk, file_chunk=True)
                yield chunk
        self.last_chunk_yielded_ns = time.perf_counter_ns()
        self.footer_yielded = True
        self._yielded(self.footer)
        yield self.footer


def sanitized_transport_error(error: BaseException) -> dict:
    """Fixed exception classes/errno only; never messages, URLs, keys or paths."""
    allowed = {"ReadError", "WriteError", "ReadTimeout", "WriteTimeout", "ConnectTimeout", "ConnectError",
               "RemoteProtocolError", "LocalProtocolError", "PoolTimeout", "TimeoutError", "ConnectionResetError",
               "BrokenPipeError", "ConnectionAbortedError", "OSError", "SSLError", "SSLCertVerificationError",
               "CancelledError", "RuntimeError", "Exception"}
    chain, seen = [], set()
    current = error
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        name = type(current).__name__
        item = {"type": name if name in allowed else "unrecognized_exception_type"}
        number = getattr(current, "errno", None)
        if type(number) is int and 0 <= number < 65536:
            item["errno"] = number
        chain.append(item)
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return {"exception_chain": chain, "messages_exported": False}


async def run_sample(client, path: Path, sample_id: str, base: str, key: str,
                     deadline_seconds: float, poll_seconds: float, repeat_completed_get: int = 0,
                     accepted_jobs: dict | None = None, terminal_exports: dict | None = None,
                     planned_attempt_id: str | None = None) -> dict:
    row = {"sample_id": sample_id, "input_bytes": path.stat().st_size,
           "outcome": "failed", "error_code": None, "poll_count": 0,
           "processing_get": [], "completed_get": [], "other_get": [],
           "post_attempted": False, "http_requests_attempted": 0,
           "completed_repeat_failures": 0, "completed_repeat_timeouts": 0}
    started = time.perf_counter_ns()
    loop_clock = asyncio.get_running_loop().time
    deadline = loop_clock() + deadline_seconds
    body = UploadStream(path)
    private_verdict = None
    attempt_id = planned_attempt_id or "attempt_" + secrets.token_hex(16)
    durable_acknowledgement_received = False
    try:
        row["post_attempted"] = True
        row["http_requests_attempted"] += 1
        post_status, post_headers, payload, post = await transfer_response(
            client, "POST", base + "/api/v1/analyze", content=body,
            absolute_deadline=deadline,
            headers={"X-API-Key": key,
                     "Content-Type": "multipart/form-data; boundary=" + body.boundary,
                     "Content-Length": str(body.length)},
        )
        accepted = time.perf_counter_ns()
        row["post"] = {**post, "http_status": post_status,
                       "client_total_ms": (accepted - started) / 1e6,
                       "client_last_chunk_to_ack_ms": None if body.last_chunk_yielded_ns is None
                       else (accepted - body.last_chunk_yielded_ns) / 1e6,
                       "asgi_post_body_acceptance_ms": server_timing(post_headers, "acceptance")}
        if post_status not in {200, 201, 202}:
            row["error_code"] = "post_http_error"
            return row
        submitted = json.loads(payload)
        job_id = submitted.get("job_id")
        # Reject unexpected routing characters before constructing the poll URL.
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            row["error_code"] = "invalid_job_id"
            return row
        durable_acknowledgement_received = True
        if accepted_jobs is not None:
            accepted_jobs[sample_id] = job_id
        while loop_clock() < deadline:
            row["http_requests_attempted"] += 1
            row["poll_count"] += 1
            status, headers, payload, measurement = await transfer_response(
                client, "GET", base + "/api/v1/jobs/" + job_id,
                headers={"X-API-Key": key}, absolute_deadline=deadline,
            )
            measurement["http_status"] = status
            measurement["asgi_lookup_ms"] = server_timing(headers, "lookup")
            if status != 200:
                row["other_get"].append(measurement)
                row["error_code"] = "poll_http_error"
                return row
            parsed = json.loads(payload)
            state = parsed.get("status")
            if state == "processing":
                row["processing_get"].append(measurement)
                await asyncio.sleep(min(poll_seconds, max(0, deadline - loop_clock())))
                continue
            if state == "done":
                row["completed_get"].append(measurement)
                from forte.shared.schemas import CompletedResponse
                try:
                    validated = CompletedResponse.model_validate(parsed)
                    if validated.job_id != job_id:
                        raise ValueError("completed job identity mismatch")
                except ValueError:
                    row["error_code"] = "invalid_completed_contract"
                    return row
                row["completed_wire_contract_checked"] = True
                private_verdict = parsed["verdict"]
                row["outcome"] = "completed"
                row["client_upload_start_to_full_result_ms"] = (time.perf_counter_ns() - started) / 1e6
                row["client_ack_to_full_result_ms"] = (time.perf_counter_ns() - accepted) / 1e6
                for _ in range(repeat_completed_get):
                    row["http_requests_attempted"] += 1
                    row["poll_count"] += 1
                    try:
                        repeat_status, repeat_headers, repeat_payload, repeat_measurement = await transfer_response(
                            client, "GET", base + "/api/v1/jobs/" + job_id, headers={"X-API-Key": key})
                        repeat_measurement["http_status"] = repeat_status
                        repeat_measurement["asgi_lookup_ms"] = server_timing(repeat_headers, "lookup")
                        if repeat_status == 200 and json.loads(repeat_payload).get("status") == "done":
                            row["completed_get"].append(repeat_measurement)
                        else:
                            row["completed_repeat_failures"] += 1
                            row["other_get"].append(repeat_measurement)
                    except Exception as exc:
                        row["completed_repeat_failures"] += 1
                        row["completed_repeat_timeouts"] += int(isinstance(exc, TimeoutError) or type(exc).__name__.endswith("Timeout"))
                return row
            row["other_get"].append(measurement)
            row["error_code"] = "job_failed" if state == "failed" else "invalid_job_status"
            return row
        row["error_code"] = "client_deadline"
    except Exception as exc:
        # Exception messages may contain paths, credentials, or server body text.
        row["error_code"] = ("client_deadline" if isinstance(exc, TimeoutError) else
                             "response_limit" if type(exc) is ValueError and exc.args == ("response_limit",)
                             else "transport_or_protocol_error")
        row["exception_type"] = type(exc).__name__
        row["transport_diagnostic"] = sanitized_transport_error(exc)
    finally:
        row["client_observed_total_ms"] = (time.perf_counter_ns() - started) / 1e6
        row["upload_iterator"] = body.progress(started)
        if terminal_exports is not None:
            terminal_exports[sample_id] = {"sample_id": sample_id,
                "attempt_id": attempt_id,
                "status": "done" if row["outcome"] == "completed" else "timeout" if row.get("error_code") == "client_deadline" or row.get("exception_type", "").endswith("Timeout") else "failed",
                "verdict": private_verdict,
                "public_contract_validated": bool(row.get("completed_wire_contract_checked")),
                "timing_origin": "client_upload_start_to_full_result_byte",
                "upload_start_to_full_result_ms": row.get("client_upload_start_to_full_result_ms") if row.get("completed_wire_contract_checked") else None,
                "terminal_observed_elapsed_ms": row["client_observed_total_ms"],
                "durable_acknowledgement_received": durable_acknowledgement_received,
                "error_code": row.get("error_code"),
                "upload_start_to_result_sec": row.get("client_upload_start_to_full_result_ms", row["client_observed_total_ms"]) / 1000}
    return row


def summarize(rows: list[dict], *, cache_mode: str, model_warmth: str) -> dict:
    def collect(group: str, metric: str):
        return [entry[metric] for row in rows for entry in row.get(group, []) if entry.get(metric) is not None]

    acceptance = [row["post"]["asgi_post_body_acceptance_ms"] for row in rows
                  if row.get("post", {}).get("asgi_post_body_acceptance_ms") is not None]
    done = [row["client_upload_start_to_full_result_ms"] for row in rows if row["outcome"] == "completed"]
    metrics = {"asgi_acceptance_after_body": statistics_ms(acceptance),
               "client_post_total": statistics_ms([row["post"]["client_total_ms"] for row in rows if "post" in row]),
               "processing_asgi_lookup": statistics_ms(collect("processing_get", "asgi_lookup_ms")),
               "processing_full_transfer": statistics_ms(collect("processing_get", "full_response_ms")),
               "completed_first_body_byte": statistics_ms(collect("completed_get", "first_body_byte_ms")),
               "completed_full_transfer": statistics_ms(collect("completed_get", "full_response_ms")),
               "upload_start_to_full_result_upper_bound": statistics_ms(done)}
    targets = []
    for name, metric, cutoff in (("post_body_acceptance_p95", "asgi_acceptance_after_body", 150),
                                 ("processing_lookup_p95", "processing_asgi_lookup", 25),
                                 ("completed_first_body_byte_p95", "completed_first_body_byte", 50)):
        observed = metrics[metric]["p95_ms"]
        targets.append({"target": name, "threshold_ms": cutoff,
                        "status": "unmeasured" if observed is None else "achieved" if observed <= cutoff else "missed",
                        "scope": "observed sample; completed first byte includes client/network overhead"})
    for name, cutoff in (("processing_safety", 240_000), ("conservative_organizer_cutoff", 300_000)):
        if not rows:
            status = "unmeasured"
        elif any(row.get("error_code") == "job_failed" or row.get("client_upload_start_to_full_result_ms", 0) > cutoff
                 or (row.get("error_code") == "client_deadline" and row.get("client_observed_total_ms", 0) >= cutoff)
                 for row in rows):
            status = "missed"
        elif len(done) != len(rows):
            status = "unmeasured"
        else:
            status = "achieved"
        targets.append({"target": name, "threshold_ms": cutoff, "status": status,
                        "scope": "all observed uploads; includes polling interval and full result transfer"})
    return {"attempted": len(rows), "completed": len(done), "failed": len(rows) - len(done),
            "http_request_count": sum(row.get("http_requests_attempted", int("post" in row) + row["poll_count"]) for row in rows),
            "completed_repeat_failures": sum(row.get("completed_repeat_failures", 0) for row in rows),
            "timeouts": sum(int(row.get("error_code") == "client_deadline" or row.get("exception_type", "").endswith("Timeout")) + row.get("completed_repeat_timeouts", 0) for row in rows),
            "metrics": metrics, "targets": targets,
            "inference_speed_claim": "not established by client transfer measurements",
            "principal_benchmark_conditions_declared": cache_mode == "disabled" and model_warmth == "warm",
            "condition_verification": "operator declaration only; novel-content and server cache audit required"}


async def execute(args) -> dict:
    import httpx

    paths = [Path(value) for value in args.files]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("missing_input_file")
    digests = [await asyncio.to_thread(content_digest, path) for path in paths]
    if len(set(digests)) != len(digests):
        raise ValueError("duplicate_content")
    if args.mode == "burst" and len(paths) != 5:
        raise ValueError("burst_requires_five_distinct_files")
    input_metadata = {"status": "unmeasured", "success_count": 0, "failure_count": 0}
    if args.probe_inputs:
        from forte.video.probe import probe_video
        durations, sizes = [], []
        failed_probes = 0
        for path in paths:
            try:
                asset = await asyncio.to_thread(probe_video, path)
                durations.append(asset.duration_sec)
                sizes.append((asset.width, asset.height))
            except Exception:
                failed_probes += 1
        input_metadata = {"status": "measured" if durations else "failed",
                          "success_count": len(durations), "failure_count": failed_probes,
                          "duration_sec_min": min(durations) if durations else None,
                          "duration_sec_max": max(durations) if durations else None,
                          "upright_resolutions": [{"width": width, "height": height, "count": sizes.count((width, height))}
                                                  for width, height in sorted(set(sizes))]}
    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise ValueError("missing_api_key_environment")
    timeout = httpx.Timeout(args.request_timeout_seconds)
    accepted_jobs = {}
    terminal_exports = {}
    attempt_ids = {}
    attempt_plan_digest = None
    if args.private_attempt_plan:
        plan_bytes = args.private_attempt_plan.read_bytes()
        plan = json.loads(plan_bytes)
        records = plan.get("records", [])
        if plan.get("schema_version") != "forte.benchmark-attempt-plan.v1" or len(records) != len(paths):
            raise ValueError("invalid_private_attempt_plan")
        for item in records:
            wire_id, attempt_id = item.get("wire_sample_id"), item.get("attempt_id")
            if not isinstance(wire_id, str) or not isinstance(attempt_id, str) or not re.fullmatch(r"attempt_[0-9a-f]{32}", attempt_id):
                raise ValueError("invalid_private_attempt_plan")
            attempt_ids[wire_id] = attempt_id
        if set(attempt_ids) != {f"s{index:03d}" for index in range(1, len(paths) + 1)} or len(set(attempt_ids.values())) != len(paths):
            raise ValueError("invalid_private_attempt_plan")
        attempt_plan_digest = hashlib.sha256(plan_bytes).hexdigest()
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        if args.mode == "burst":
            rows = await asyncio.gather(*(run_sample(client, path, f"s{index:03d}", args.base_url.rstrip("/"), key,
                                                    args.deadline_seconds, args.poll_seconds, args.repeat_completed_get, accepted_jobs, terminal_exports,
                                                    attempt_ids.get(f"s{index:03d}"))
                                          for index, path in enumerate(paths, 1)))
        else:
            rows = [await run_sample(client, path, f"s{index:03d}", args.base_url.rstrip("/"), key,
                                     args.deadline_seconds, args.poll_seconds, args.repeat_completed_get, accepted_jobs, terminal_exports,
                                     attempt_ids.get(f"s{index:03d}"))
                    for index, path in enumerate(paths, 1)]
        metrics = {"status": "unmeasured", "names": []}
        if args.metrics_url:
            try:
                response = await client.get(args.metrics_url)
                response.raise_for_status()
                metrics = {"status": "measured", "names": discover_vllm_metric_names(response.text)}
            except Exception:
                metrics = {"status": "failed", "names": []}
    summary = summarize(rows, cache_mode=args.cache_mode, model_warmth=args.model_warmth)
    summary["principal_benchmark_conditions_declared"] = bool(
        summary["principal_benchmark_conditions_declared"] and args.novel_content_declared
        and args.input_kind != "synthetic_transport")
    reused_content = bool(args.reused_content_declared)
    summary["primary_slo_evidence_eligible"] = bool(summary["principal_benchmark_conditions_declared"] and not reused_content)
    if reused_content:
        for target in summary["targets"]:
            target["exploratory_observed_status"] = target["status"]
            target["status"] = "ineligible_reused_content"
            target["scope"] = "paired reused-content observation; not primary novel-content SLO evidence"
        summary["condition_verification"] = "content deliberately reused; only exploratory paired comparison is eligible"
    correlation_status = "not_requested"
    if args.private_job_map:
        try:
            args.private_job_map.parent.mkdir(parents=True, exist_ok=True)
            args.private_job_map.write_text(json.dumps({"schema_version": "1.0", "sample_jobs": accepted_jobs}, indent=2), encoding="utf-8")
            correlation_status = "written_to_private_local_file"
        except OSError:
            correlation_status = "write_failed; HTTP measurements preserved"
    terminal_export_status = "not_requested"
    if args.private_terminal_export:
        try:
            args.private_terminal_export.parent.mkdir(parents=True, exist_ok=True)
            args.private_terminal_export.write_text(json.dumps({"schema_version": "1.0", "records": list(terminal_exports.values())}, indent=2), encoding="utf-8")
            terminal_export_status = "written_to_private_local_file"
        except OSError:
            terminal_export_status = "write_failed; HTTP measurements preserved"
    return {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": runtime_metadata(),
            "input_kind": args.input_kind,
            "quality_review": "unmeasured; successful processing is not reviewed video ground truth",
            "private_correlation_status": correlation_status,
            "private_terminal_export_status": terminal_export_status,
            "expected_attempt_plan_sha256": attempt_plan_digest,
            "evidence_scope": "paired_reused_content_exploratory" if reused_content else "transport_only_no_model" if args.input_kind == "synthetic_transport" else "video_pipeline_requires_server_verification",
            "declared_server": {"gpu_model": args.gpu_model, "gpu_count": args.gpu_count,
                                "gpu_vram_gib_each": args.gpu_vram_gib, "model_revision": args.model_revision,
                                "frame_budget": args.frame_budget, "max_pixels_per_frame": args.max_pixels,
                                "metadata_provenance": "operator declaration; confirm against actual server inventory"},
            "input_metadata_aggregate": input_metadata,
            "conditions": {"mode": args.mode, "model_warmth": args.model_warmth,
                           "whole_result_cache": args.cache_mode, "distinct_content_local_hash_check": True,
                           "novel_content_operator_declared": args.novel_content_declared,
                           "reused_content_operator_declared": reused_content,
                           "local_input_probe_before_timing": args.probe_inputs,
                           "deadline_seconds_from_client_upload_start": args.deadline_seconds,
                           "poll_interval_seconds": args.poll_seconds,
                           "repeat_completed_get_per_successful_job": args.repeat_completed_get,
                           "request_timeout_seconds": args.request_timeout_seconds},
            "boundaries": {"T0_proxy_first_body_byte": "unmeasured",
                           "T1_api_last_body_byte": "ASGI-local when Server-Timing header exists",
                           "T2_durable_acceptance": "ASGI-local acceptance hook when installed",
                           "T3_worker_start": "unmeasured by HTTP client; use local worker StageTimer",
                           "T4_result_publication": "unmeasured by HTTP client; poll provides upper bound",
                           "T5_client_final_byte": "measured with client monotonic clock",
                           "client_last_chunk_to_ack": "client iterator yield to ACK; includes transport, not T2-T1"},
            "summary": summary,
            "vllm_metrics_discovery": metrics, "samples": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--mode", choices=("single", "burst"), default="single")
    parser.add_argument("--api-key-env", default="FORTE_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=300)
    parser.add_argument("--request-timeout-seconds", type=float, default=60)
    parser.add_argument("--poll-seconds", type=float, default=.25)
    parser.add_argument("--repeat-completed-get", type=int, default=20)
    parser.add_argument("--model-warmth", choices=("unknown", "warm", "cold"), default="unknown")
    parser.add_argument("--cache-mode", choices=("unknown", "disabled", "hit"), default="unknown")
    parser.add_argument("--metrics-url", default=None)
    parser.add_argument("--gpu-model", default=None)
    parser.add_argument("--gpu-count", type=int, default=None)
    parser.add_argument("--gpu-vram-gib", type=float, default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--frame-budget", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--probe-inputs", action="store_true", help="local FFprobe numeric metadata only, before timing")
    parser.add_argument("--input-kind", choices=("video_unverified", "synthetic_transport"), default="video_unverified")
    history = parser.add_mutually_exclusive_group()
    history.add_argument("--novel-content-declared", action="store_true", help="operator confirms content not used in earlier server runs")
    history.add_argument("--reused-content-declared", action="store_true", help="paired repeated content; primary SLO claims are ineligible")
    parser.add_argument("--private-job-map", type=Path, default=None, help="optional local job correlation under gitignored benchmarks/private")
    parser.add_argument("--private-terminal-export", type=Path, default=None, help="minimal verdict export stays under gitignored benchmarks/private")
    parser.add_argument("--private-attempt-plan", type=Path, default=None, help="frozen expected attempt IDs under gitignored benchmarks/private")
    args = parser.parse_args(argv)
    if not all(math.isfinite(value) and value > 0 for value in (args.deadline_seconds, args.request_timeout_seconds, args.poll_seconds)):
        parser.error("timing settings must be positive")
    if args.repeat_completed_get < 0:
        parser.error("repeat count cannot be negative")
    if args.output.suffix.lower() != ".json":
        parser.error("output must be a .json report")
    if args.output.resolve() in {Path(value).resolve() for value in args.files}:
        parser.error("output must not replace an input")
    if args.private_job_map:
        private_root = Path(__file__).resolve().parents[3] / "benchmarks/private"
        if args.private_job_map.suffix.lower() != ".json" or not args.private_job_map.resolve().is_relative_to(private_root.resolve()):
            parser.error("job correlation must remain in benchmarks/private as JSON")
        if args.private_job_map.resolve() == args.output.resolve():
            parser.error("private correlation must be separate from the aggregate report")
    if args.private_terminal_export:
        private_root = Path(__file__).resolve().parents[3] / "benchmarks/private"
        if args.private_terminal_export.suffix.lower() != ".json" or not args.private_terminal_export.resolve().is_relative_to(private_root.resolve()):
            parser.error("terminal verdict export must remain private JSON")
        if args.private_terminal_export.resolve() == args.output.resolve() or (args.private_job_map and args.private_terminal_export.resolve() == args.private_job_map.resolve()):
            parser.error("private terminal export must have a separate path")
    if args.private_attempt_plan:
        private_root = Path(__file__).resolve().parents[3] / "benchmarks/private"
        if not args.private_attempt_plan.resolve().is_relative_to(private_root.resolve()) or not args.private_attempt_plan.is_file():
            parser.error("frozen attempt plan must exist under benchmarks/private")
        if args.private_attempt_plan.resolve() in {args.output.resolve(), *(value.resolve() for value in (args.private_job_map, args.private_terminal_export) if value)}:
            parser.error("attempt plan must be separate and immutable")
    try:
        report = asyncio.run(execute(args))
    except Exception as exc:
        safe_codes = {"missing_input_file", "duplicate_content", "burst_requires_five_distinct_files",
                      "missing_api_key_environment"}
        code = str(exc) if type(exc) is ValueError and str(exc) in safe_codes else "benchmark_setup_error"
        report = {"schema_version": "1.0", "status": "failed", "error_code": code,
                  "exception_type": type(exc).__name__, "targets": "unmeasured"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    summary = report.get("summary", {})
    print(json.dumps({"attempted": summary.get("attempted", 0), "completed": summary.get("completed", 0),
                      "failed": summary.get("failed", 0), "status": report.get("status", "finished")}))
    return 0 if summary.get("attempted", 0) > 0 and summary.get("failed", 1) == 0 and summary.get("completed_repeat_failures", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
