"""Network reliability smoke checks; never print bodies or private diagnostics."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import time

import httpx


async def run_checks(client, base_url: str, api_key: str, *, corrupt_wait_seconds: float = 10,
                     include_corrupt: bool = True) -> dict:
    base = base_url.rstrip("/")
    cases = []

    async def check(identifier, method, route, headers, expected, **kwargs):
        try:
            response = await client.request(method, base + route, headers=headers, **kwargs)
            cases.append({"id": identifier, "status": "PASS" if response.status_code in expected else "FAIL",
                          "http_status": response.status_code})
            return response
        except Exception as exc:
            cases.append({"id": identifier, "status": "FAIL", "error_code": "transport_error",
                          "exception_type": type(exc).__name__})
            return None

    wrong = "qa-invalid-" + secrets.token_hex(16)
    for key_name, headers in (("missing", {}), ("wrong", {"X-API-Key": wrong})):
        await check(key_name + "_key_post", "POST", "/api/v1/analyze", headers, {401, 403},
                    content=b"synthetic-auth-check")
        await check(key_name + "_key_get", "GET", "/api/v1/jobs/synthetic-unknown", headers, {401, 403})
    await check("unknown_job", "GET", "/api/v1/jobs/synthetic-unknown",
                {"X-API-Key": api_key}, {404})
    if not include_corrupt:
        counts = {status: sum(case["status"] == status for case in cases) for status in ("PASS", "FAIL", "UNMEASURED")}
        return {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "layer": "HTTP auth-only; no accepted media jobs created", "input": "synthetic invalid auth bytes only",
                "cases": cases, "counts": counts, "corrupt_media_terminal_state": "unmeasured; not requested",
                "pre_body_auth_note": "HTTP rejection measured; receive-not-called proof is an ASGI test"}
    corrupt = await check("corrupt_upload_accepted", "POST", "/api/v1/analyze",
                          {"X-API-Key": api_key}, {200, 201, 202},
                          files={"file": ("synthetic-corrupt.mov", b"FORTE_SYNTHETIC_CORRUPT_MEDIA\x00", "video/quicktime")})
    terminal = {"id": "corrupt_media_fails_without_verdict", "status": "UNMEASURED",
                "reason": "worker did not reach terminal state within observation window"}
    if corrupt is not None and corrupt.status_code in {200, 201, 202}:
        try:
            job_id = corrupt.json().get("job_id")
            if not isinstance(job_id, str) or not job_id.isascii() or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in job_id):
                raise ValueError("invalid routing identifier")
            deadline = time.monotonic() + corrupt_wait_seconds
            while time.monotonic() < deadline:
                response = await client.get(base + "/api/v1/jobs/" + job_id, headers={"X-API-Key": api_key})
                state = response.json() if response.status_code == 200 else {}
                if response.status_code != 200:
                    terminal = {"id": terminal["id"], "status": "FAIL", "error_code": "poll_http_error"}
                    break
                if state.get("status") == "failed":
                    terminal = {"id": terminal["id"], "status": "PASS" if "verdict" not in state and "violations" not in state else "FAIL"}
                    break
                if state.get("status") == "done":
                    terminal = {"id": terminal["id"], "status": "FAIL", "error_code": "corrupt_media_returned_verdict"}
                    break
                await asyncio.sleep(min(.1, max(0, deadline - time.monotonic())))
        except Exception as exc:
            terminal = {"id": terminal["id"], "status": "FAIL", "error_code": "protocol_error",
                        "exception_type": type(exc).__name__}
    cases.append(terminal)
    counts = {status: sum(case["status"] == status for case in cases) for status in ("PASS", "FAIL", "UNMEASURED")}
    return {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "layer": "HTTP; actual deployment locality determined by caller",
            "input": "synthetic invalid bytes only", "cases": cases, "counts": counts,
            "unmeasured": ["large valid media", "actual worker process kill", "PostgreSQL process restart",
                           "GPU inference", "five distinct real clips", "egress-disabled real inference"],
            "pre_body_auth_note": "HTTP rejection is measured here; receive-not-called proof is an ASGI test"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="FORTE_API_KEY")
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrupt-wait-seconds", type=float, default=10)
    parser.add_argument("--auth-only", action="store_true")
    args = parser.parse_args(argv)
    key = os.environ.get(args.api_key_env)
    if args.api_key_file:
        key = args.api_key_file.read_text(encoding="utf-8").strip().splitlines()[0].split(",")[0].strip()
    if not key:
        parser.error("API key environment variable is missing")
    if args.corrupt_wait_seconds <= 0:
        parser.error("observation window must be positive")

    async def run():
        async with httpx.AsyncClient(timeout=10, trust_env=False, follow_redirects=False) as client:
            return await run_checks(client, args.base_url, key, corrupt_wait_seconds=args.corrupt_wait_seconds,
                                    include_corrupt=not args.auth_only)

    report = asyncio.run(run())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"]))
    return 1 if report["counts"]["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
