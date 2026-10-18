"""Protocol-v2 smoke: two affirmative local-action phrasings, direct consensus.

Visual principle: question polarity must remain stable across paired visual
measurements; disagreement must not be created by contradictory field semantics.
Every video gets one uniform midpoint 3-second window and 8 original native
frames. This smoke does not measure full transactions, omissions, 90% visibility
coverage, accuracy, or qualification. No labels or correctness select inputs.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


AFFIRMATIVE = (
    "For ad,ai,ar,aa,as answer yes only when that exact action is directly visible "
    "in this local interval. Answer no only when its operative region and hands "
    "are resolved and the action is absent locally; otherwise unclear. All action "
    "keys describe the action OCCURRING, never its absence. JSON only.",
    "Independently inspect what actually happens in each numbered local interval. "
    "For every action key, yes means you SEE that named action happen; no means "
    "a clear view shows it does not happen in that local interval; unclear means "
    "the pixels or visibility cannot decide. Keep these affirmative meanings for "
    "ad,ai,ar,aa,as. Do not invert the yes/no meaning. JSON only.",
)


def direct_consensus(T, a, b, times):
    if any([r["i"] for r in answer["frames"]] != list(range(len(times))) for answer in (a, b)):
        raise ValueError("frame_index_alignment_failed")
    result = []
    for stamp, ra, rb in zip(times, a["frames"], b["frames"]):
        result.append({"t": stamp,
            "visibility": {name: ra[key] if ra[key] == rb[key] else "unclear" for key, name in T.VIS_KEYS.items()},
            "actions": {name: ra[key] if ra[key] == rb[key] else "unclear" for key, name in T.ACTIONS.items()},
            "action_disagreements": {name: ra[key] != rb[key] for key, name in T.ACTIONS.items()}})
    return result


async def run(args):
    sys.path.insert(0, "/forte/src")
    from forte.fast.video import decode
    spec = importlib.util.spec_from_file_location("original_visible_tool", args.original_tool)
    T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
    started = time.monotonic()
    source = Path(args.run)
    rows = T.cached_rows(json.loads(source.read_text(encoding="utf-8")))
    if {p.stem for p in Path(args.videos).glob("*.mov")} != {r["name"] for r in rows}:
        raise ValueError("video_inventory_differs")
    client = T.LocalClient(args.base_url, "q30test", 32, started + args.max_wall_s)
    sem = asyncio.Semaphore(8)

    async def one(row):
        async with sem:
            out = {"name": row["name"], "status": "failed"}
            video = None
            try:
                video = await asyncio.to_thread(decode, str(Path(args.videos) / (row["name"] + ".mov")),
                    5, max_frames=1000, max_memory_bytes=2 * 1024**3, deadline=client.deadline)
                centre = video.duration / 2
                lo, hi = max(0, centre - 1.5), min(video.duration, centre + 1.5)
                available = video.between(lo, hi)
                if len(available) < 8:
                    raise ValueError("uniform_smoke_has_too_few_frames")
                frames = [available[round(i * (len(available) - 1) / 7)] for i in range(8)]
                times = [f.t for f in frames]
                content = T.frame_content(frames, row["layout"])
                answers = await T.paired_asks(client, content,
                    tuple(T.LOCAL_ACTION_PROMPT + suffix for suffix in AFFIRMATIVE), T.frame_schema(8), 1600)
                out.update(status="done", start=lo, end=hi, frame_times=times,
                           paired_answers=answers, consensus=direct_consensus(T, *answers, times))
            except Exception as exc:
                out["error_class"] = T.error_code(exc)
            finally:
                if video is not None:
                    for frame in video.frames: frame.image.close()
            return out
    try:
        results = await asyncio.gather(*(one(row) for row in rows))
    finally:
        await client.http.aclose()
    return {"schema": "forte.procedure-visible-protocol-smoke.v2", "model": "q30test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True,
        "type_enablement_qualified": False, "transaction_coverage_claim": False,
        "continuous_visibility_proven": False, "labels_used": False,
        "sampling": {"videos": 40, "windows_per_video": 1, "frames_per_window": 8,
                     "window_seconds": 3, "selection": "uniform_midpoint_all40", "action_phrasings": "both_affirmative;direct_consensus"},
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_tool_sha256": hashlib.sha256(Path(args.original_tool).read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": {"videos": 40, "done": sum(r["status"] == "done" for r in results),
                    "failed": sum(r["status"] == "failed" for r in results), "calls": client.calls,
                    "attempted_calls": client.attempted_calls, "prompt_tokens": client.prompt_tokens,
                    "completion_tokens": client.completion_tokens, "wall_s": round(time.monotonic() - started, 3)},
        "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "videos", "original-tool", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-wall-s", type=float, default=140)
    args = parser.parse_args()
    if not 30 <= args.max_wall_s <= 150: raise ValueError("invalid_smoke_budget")
    report = asyncio.run(run(args))
    path = Path(args.out); temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({"status": "done", "summary": report["summary"], "type_enablement_qualified": False}))


if __name__ == "__main__":
    try: main()
    except Exception:
        print('{"status":"failed","failure_code":"procedure_protocol_v2_smoke_failed"}')
        raise SystemExit(1)
