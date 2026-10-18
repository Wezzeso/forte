"""Observe readiness from this client's start; not an invented cold-start time."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import httpx


async def observe(base_url, seconds, interval):
    started = time.perf_counter_ns()
    deadline = asyncio.get_running_loop().time() + seconds
    attempts, errors = 0, 0
    status = "unmeasured"
    async with httpx.AsyncClient(timeout=min(5, seconds), trust_env=False, follow_redirects=False) as client:
        while asyncio.get_running_loop().time() < deadline:
            attempts += 1
            try:
                async with asyncio.timeout_at(deadline):
                    response = await client.get(base_url.rstrip("/") + "/readyz")
                if response.status_code == 200:
                    status = "ready_observed"
                    break
                errors += 1
            except Exception:
                errors += 1
            await asyncio.sleep(min(interval, max(0, deadline - asyncio.get_running_loop().time())))
    return {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status, "requests": attempts, "unready_or_transport_errors": errors,
            "client_observation_ms": (time.perf_counter_ns() - started) / 1e6,
            "cold_service_startup": "unmeasured; no process launch boundary supplied",
            "scope": "readiness observed from client start, includes poll interval and HTTP overhead"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--interval", type=float, default=.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0 for value in (args.seconds, args.interval)):
        parser.error("observation durations must be finite and positive")
    if args.output.suffix.lower() != ".json":
        parser.error("output must be a JSON report")
    report = asyncio.run(observe(args.base_url, args.seconds, args.interval))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "requests": report["requests"]}))
    return 0 if report["status"] == "ready_observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
