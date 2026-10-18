"""Run dense window probes over cached per-video states (structured) and score them.

  python tools/probe_eval.py --run ref1.json --videos /videos --labels labels.json \
      --types NO_AUTHENTICITY_CHECK,NO_CLIENT_ID,NO_RECOUNT --out probes.json

States and layout come from a previous fast_eval report, so only the probe questions are
new model calls. Labels are read only by the scorer at the end. Output holds enums,
windows and timestamps only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def main_async(args) -> dict:
    from forte.fast.analyzer import DEFAULT_CONFIG
    from forte.fast.probes import probe_candidates
    from forte.fast.service import decode_for_analysis
    from forte.fast.vlm import VLMClient

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in run["results"]}
    cfg = {**DEFAULT_CONFIG, **run.get("config", {}), "probe_types": args.types.split(","),
           "probe_frames": args.frames}
    client = VLMClient(args.base_url, args.model, concurrency=48)
    sem = asyncio.Semaphore(args.par)
    names = sorted(rows) if not args.only else args.only.split(",")

    async def one(name: str) -> dict:
        async with sem:
            row = rows[name]
            video, _ = await asyncio.to_thread(decode_for_analysis, str(Path(args.videos) / f"{name}.mov"), cfg)
            try:
                found = await probe_candidates(client, video, row.get("states") or [], row.get("layout") or {}, cfg)
            finally:
                for f in video.frames:
                    f.image.close()
            return {"name": name, "probes": found}

    try:
        results = await asyncio.gather(*(one(n) for n in names))
    finally:
        await client.aclose()
    report = {"schema": "forte.probe-eval.v1", "run": args.run, "types": args.types.split(","),
              "frames": args.frames, "calls": client.calls, "prompt_tokens": client.prompt_tokens,
              "results": results}
    if args.labels:
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        score = {}
        for t in report["types"]:
            pos_hit, pos_win, neg_fire, windows = [], [], [], 0
            for r in results:
                lab = labels.get(r["name"], {})
                for p in r["probes"]:
                    if p["type_id"] != t:
                        continue
                    windows += 1
                    if not p["fired"]:
                        continue
                    if not lab.get("violation"):
                        neg_fire.append((r["name"], p["t"]))
                    else:
                        lt = lab.get("t")
                        near = lt is not None and p["start"] - 3 <= lt <= p["end"] + 3
                        (pos_hit if near else pos_win).append((r["name"], p["t"], lt))
            score[t] = {"windows": windows, "fires_on_negatives": neg_fire,
                        "fires_on_positives_near_label": pos_hit, "fires_on_positives_elsewhere": pos_win}
        report["score_in_sample"] = score
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--types", required=True)
    ap.add_argument("--only")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--base-url", default="http://vlm30:8000/v1")
    ap.add_argument("--model", default="q30")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    report = asyncio.run(main_async(args))
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("types", "calls", "prompt_tokens")}))
    for t, s in (report.get("score_in_sample") or {}).items():
        print(t, json.dumps(s, ensure_ascii=False))
    for r in report["results"]:
        for p in r["probes"]:
            print(r["name"], p["type_id"], round(p["start"], 1), round(p["end"], 1), p.get("answers"), "FIRE" if p["fired"] else "", p.get("abstain", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
