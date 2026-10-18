"""End-to-end latency harness for the Forte API (prompt section 4.2 / 6.6).

Latency is measured from the moment POST /api/v1/analyze returns 202 to the first
GET /api/v1/jobs/{id} that returns ``done``, polling every 0.2 s by default.

  sequential: one job at a time over every --video (single-job target)
  parallel:   N simultaneous uploads (concurrency target; completion order recorded)

Reports hold counters, timings, verdicts, type ids and timecodes only. Evidence
images and description text are validated and then discarded, never written.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forte.shared.schemas import JOB_RESPONSE_ADAPTER  # noqa: E402


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 3)


def stats(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": _pct(values, 0.5), "p95": _pct(values, 0.95),
            "max": round(max(values), 3) if values else None,
            "mean": round(statistics.fmean(values), 3) if values else None}


def media_duration(path: Path) -> float | None:
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return container.duration / 1_000_000
    except Exception:
        return None
    return None


def check_result(payload: dict[str, Any], duration: float | None) -> dict[str, Any]:
    """Validate the public contract; return only structured facts."""
    result = JOB_RESPONSE_ADAPTER.validate_python(payload)
    data = result.model_dump(mode="json")
    problems = []
    events = []
    for v in data.get("violations") or []:
        frames = v.get("evidence_frames") or []
        if not 1 <= len(frames) <= 3:
            problems.append("evidence_count")
        for f in frames:
            img = f.get("image_base64") or ""
            if img.startswith("data:") or not img.startswith("/9j/"):
                problems.append("evidence_not_raw_jpeg")
            if duration is not None and not (0 <= f["timestamp_sec"] <= duration + 0.05):
                problems.append("evidence_time_bounds")
            if abs(f["timestamp_sec"] - v["timecode_sec"]) > 1.05:
                problems.append("evidence_far_from_event")
        if duration is not None and not (0 <= v["timecode_sec"] <= duration + 0.05):
            problems.append("timecode_bounds")
        events.append({"type": v["type"], "timecode_sec": v["timecode_sec"], "frames": len(frames)})
    verdict = data.get("verdict")
    if (verdict == "violation") != bool(events):
        problems.append("verdict_events_mismatch")
    return {"verdict": verdict, "events": events, "problems": sorted(set(problems))}


async def run_job(client: httpx.AsyncClient, origin: str, key: str, path: Path, *, poll: float,
                  deadline: float, clock=time.monotonic) -> dict[str, Any]:
    record: dict[str, Any] = {"name": path.stem, "pass": False}
    headers = {"X-API-Key": key}
    t_start = clock()
    try:
        with path.open("rb") as media:
            response = await client.post(origin + "/api/v1/analyze", headers=headers,
                                         files={"file": ("upload.mov", media, "video/quicktime")},
                                         timeout=deadline)
        t_accept = clock()
        record["upload_s"] = round(t_accept - t_start, 3)
        record["accepted_at"] = t_accept
        if response.status_code != 202:
            record["error"] = f"upload_http_{response.status_code}"
            return record
        job_id = response.json()["job_id"]
        polls = 0
        while True:
            if clock() - t_accept > deadline:
                record["error"] = "deadline_exceeded"
                return record
            r = await client.get(origin + f"/api/v1/jobs/{job_id}", headers=headers, timeout=30)
            polls += 1
            if r.status_code != 200:
                record["error"] = f"poll_http_{r.status_code}"
                return record
            body = r.json()
            status = body.get("status")
            if status == "done":
                t_done = clock()
                record.update(latency_s=round(t_done - t_accept, 3), done_at=t_done, polls=polls)
                facts = await asyncio.to_thread(check_result, body, await asyncio.to_thread(media_duration, path))
                record.update(facts)
                record["pass"] = not facts["problems"]
                return record
            if status == "failed":
                record.update(error="job_failed", latency_s=round(clock() - t_accept, 3), polls=polls)
                return record
            await asyncio.sleep(poll)
    except Exception as exc:  # never serialize exception text (may contain bodies)
        record["error"] = "client_" + type(exc).__name__
        return record


async def main_async(args) -> dict[str, Any]:
    key = Path(args.api_key_file).read_text(encoding="utf-8").replace("\n", ",").split(",")[0].strip() \
        if args.api_key_file else os.environ[args.api_key_env]
    videos = [Path(v) for v in args.video]
    if args.video_dir:
        videos += sorted(Path(args.video_dir).glob("*.mov"))
    if args.only:
        wanted = set(args.only.split(","))
        videos = [v for v in videos if v.stem in wanted]
    origin = args.base_url.rstrip("/")
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
        if args.warmup:
            await run_job(client, origin, key, videos[0], poll=args.poll, deadline=args.deadline)
        if args.mode == "sequential":
            for _ in range(args.repeat):
                for path in videos:
                    results.append(await run_job(client, origin, key, path, poll=args.poll, deadline=args.deadline))
                    if args.gap:
                        await asyncio.sleep(args.gap)
        else:
            chosen = videos[: args.parallel] if len(videos) >= args.parallel else \
                [videos[i % len(videos)] for i in range(args.parallel)]
            tasks = []
            for i, path in enumerate(chosen):
                tasks.append(asyncio.create_task(run_job(client, origin, key, path, poll=args.poll,
                                                         deadline=args.deadline)))
                await asyncio.sleep(args.stagger)
            results = list(await asyncio.gather(*tasks))
    accepted = [r for r in results if "accepted_at" in r]
    order_ok = None
    if args.mode == "parallel" and accepted:
        by_accept = [r["name"] + f"#{i}" for i, r in sorted(enumerate(results), key=lambda x: x[1].get("accepted_at", 1e18))]
        by_done = [r["name"] + f"#{i}" for i, r in sorted(enumerate(results), key=lambda x: x[1].get("done_at", 1e18))]
        order_ok = by_accept == by_done
    for r in results:
        r.pop("accepted_at", None)
        r.pop("done_at", None)
    lat = [r["latency_s"] for r in results if r.get("pass")]
    report = {
        "schema": "forte.fast-latency.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "202 accepted -> first GET status=done; poll interval %.2f s" % args.poll,
        "mode": args.mode, "base_url_kind": args.base_url_kind,
        "jobs": len(results), "passed": sum(bool(r.get("pass")) for r in results),
        "failed": sum(not r.get("pass") for r in results),
        "latency_s": stats(lat), "upload_s": stats([r["upload_s"] for r in results if "upload_s" in r]),
        "completion_in_arrival_order": order_ok,
        "within_10s": sum(1 for x in lat if x <= 10.0),
        "elapsed_s": round(time.monotonic() - started, 3),
        "results": results,
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--base-url-kind", default="staging_internal_dns")
    ap.add_argument("--api-key-file")
    ap.add_argument("--api-key-env", default="FORTE_API_KEY")
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--video-dir")
    ap.add_argument("--only")
    ap.add_argument("--mode", choices=["sequential", "parallel"], default="sequential")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between parallel submissions")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--poll", type=float, default=0.2)
    ap.add_argument("--deadline", type=float, default=290.0)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)
    report = asyncio.run(main_async(args))
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    brief = {k: report[k] for k in ("mode", "jobs", "passed", "failed", "latency_s", "upload_s",
                                    "completion_in_arrival_order", "within_10s", "elapsed_s")}
    print(json.dumps(brief, ensure_ascii=False))
    for r in report["results"]:
        print(json.dumps({k: r.get(k) for k in ("name", "pass", "latency_s", "upload_s", "verdict", "events",
                                                 "problems", "error")}, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
