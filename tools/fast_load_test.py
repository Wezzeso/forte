"""Measure public-API upload-to-result latency without saving result evidence.

Example: python tools/fast_load_test.py --base-url https://service.example \
    --video /owned/input.mov --parallel 6 --report load-report.json

Inputs are used round-robin. By default the run submits at least one job per
input and enough jobs to fill --parallel. This is a latency/schema check, not
an accuracy evaluation. Only send footage authorized for the target service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable, Sequence
from urllib.parse import quote, urlsplit

import httpx
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forte.shared.schemas import AnalyzeResponse, JOB_RESPONSE_ADAPTER  # noqa: E402


def _validate_settings(base_url: str, videos: Sequence[Path], parallel: int,
                       requests: int, deadline_seconds: float, poll_seconds: float) -> None:
    target = urlsplit(base_url)
    if (target.scheme not in {"http", "https"} or not target.hostname
            or target.username or target.password or target.query or target.fragment
            or target.path not in {"", "/"}):
        raise ValueError("base URL must be a bare service origin")
    # Accessing .port also rejects malformed/out-of-range ports.
    _ = target.port
    if not videos or parallel < 1 or requests < 1:
        raise ValueError("inputs, parallelism and request count must be positive")
    if (not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 300
            or not math.isfinite(poll_seconds) or poll_seconds <= 0):
        raise ValueError("deadline must be 0 < seconds <= 300; polling must be positive")


async def run_load(
    base_url: str,
    api_key: str,
    videos: Sequence[Path],
    *,
    parallel: int = 5,
    requests: int | None = None,
    deadline_seconds: float = 240,
    poll_seconds: float = 0.5,
    client: httpx.AsyncClient | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Run bounded jobs; ``client``/``clock`` support offline transport tests.

    Each budget starts before the client semaphore and upload. A completed
    response must be observed and validated within that budget. HTTP failures,
    failed jobs, mismatched IDs and malformed payloads always fail the run.
    Reports contain counters/timings only, never bodies, observations or images.
    """
    videos = [Path(video) for video in videos]
    requests = max(parallel, len(videos)) if requests is None else requests
    _validate_settings(base_url, videos, parallel, requests, deadline_seconds, poll_seconds)
    if not api_key:
        raise ValueError("API key must be provided")
    if client is None:
        limits = httpx.Limits(max_connections=parallel, max_keepalive_connections=parallel)
        async with httpx.AsyncClient(limits=limits, trust_env=False, follow_redirects=False) as owned:
            return await run_load(base_url, api_key, videos, parallel=parallel, requests=requests,
                                  deadline_seconds=deadline_seconds, poll_seconds=poll_seconds,
                                  client=owned, clock=clock)

    semaphore = asyncio.Semaphore(parallel)
    origin = base_url.rstrip("/")
    headers = {"X-API-Key": api_key}
    started = clock()

    async def run_one(index: int) -> dict:
        t0 = clock()
        record = {"request_index": index, "input_index": index % len(videos), "pass": False,
                  "phase": "client_queue", "elapsed_seconds": 0.0, "polls": 0,
                  "completed_schema_valid": False}

        def remaining() -> float:
            left = deadline_seconds - (clock() - t0)
            if left <= 0:
                raise TimeoutError
            return left

        async def work() -> None:
            async with semaphore:
                remaining()
                record["phase"] = "upload"
                with videos[record["input_index"]].open("rb") as media:
                    response = await client.post(
                        origin + "/api/v1/analyze", headers=headers,
                        files={"file": ("load-input.mov", media, "video/quicktime")},
                        timeout=remaining(), follow_redirects=False,
                    )
                remaining()
                if response.status_code != 202:
                    record.update(error_code="upload_http_failure", http_status=response.status_code)
                    return
                try:
                    accepted = AnalyzeResponse.model_validate(response.json())
                except (ValidationError, ValueError):
                    record["error_code"] = "invalid_upload_payload"
                    return
                record["phase"] = "poll"
                while True:
                    response = await client.get(
                        origin + "/api/v1/jobs/" + quote(accepted.job_id, safe="").replace(".", "%2E"),
                        headers=headers, timeout=remaining(), follow_redirects=False,
                    )
                    record["polls"] += 1
                    remaining()
                    if response.status_code != 200:
                        record.update(error_code="poll_http_failure", http_status=response.status_code)
                        return
                    try:
                        result = JOB_RESPONSE_ADAPTER.validate_python(response.json())
                    except (ValidationError, ValueError):
                        record["error_code"] = "invalid_job_payload"
                        return
                    if result.job_id != accepted.job_id:
                        record["error_code"] = "job_id_mismatch"
                        return
                    # Validation can decode evidence JPEGs; include its time too.
                    remaining()
                    if result.status == "done":
                        record.update(terminal_status="done", completed_schema_valid=True, **{"pass": True})
                        return
                    if result.status == "failed":
                        record.update(terminal_status="failed", error_code="analysis_failed")
                        return
                    await asyncio.sleep(min(poll_seconds, remaining()))

        try:
            # Wall-clock cancellation bounds semaphore, upload, requests and sleeps.
            await asyncio.wait_for(work(), timeout=deadline_seconds)
        except TimeoutError:
            record["error_code"] = "deadline_exceeded"
        except httpx.TimeoutException:
            record["error_code"] = "request_timeout"
        except httpx.HTTPError:
            record["error_code"] = "transport_failure"
        except OSError:
            record["error_code"] = "input_unavailable"
        except Exception:
            # Never serialize exceptions: HTTP errors can contain secrets/bodies.
            record["error_code"] = "load_test_failure"
        elapsed = clock() - t0
        if elapsed >= deadline_seconds:
            record.update(error_code="deadline_exceeded", **{"pass": False})
        record["elapsed_seconds"] = round(elapsed, 6)
        return record

    results = await asyncio.gather(*(run_one(index) for index in range(requests)))
    passed = sum(result["pass"] for result in results)
    return {
        "pass": passed == requests,
        "scope": "public API latency and schema; no accuracy claim; result evidence omitted",
        "parallel": parallel,
        "requests": requests,
        "deadline_seconds": deadline_seconds,
        "elapsed_seconds": round(clock() - started, 6),
        "passed": passed,
        "failed": requests - passed,
        "max_request_seconds": max(result["elapsed_seconds"] for result in results),
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--video", type=Path, action="append", required=True,
                        help="authorized MOV input; repeat for different clips")
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--requests", type=int, help="total uploads; default max(parallel, input count)")
    parser.add_argument("--deadline-seconds", type=float, default=240)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--api-key-env", default="FORTE_API_KEY")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _validate_settings(args.base_url, args.video, args.parallel,
                           args.requests if args.requests is not None else max(args.parallel, len(args.video)),
                           args.deadline_seconds, args.poll_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    key = os.environ.get(args.api_key_env, "")
    if not key:
        parser.error("specified API-key environment variable is unset")
    report = asyncio.run(run_load(args.base_url, key, args.video, parallel=args.parallel,
                                 requests=args.requests, deadline_seconds=args.deadline_seconds,
                                 poll_seconds=args.poll_seconds))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("pass", "requests", "passed", "failed", "max_request_seconds")}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
