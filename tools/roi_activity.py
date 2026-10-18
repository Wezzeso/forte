"""ROI activity curves (prompt section 7/9.4): frame-difference energy inside layout regions.

  python tools/roi_activity.py --run ref1.json --videos /videos --out activity.json [--labels labels.json]

Pixels only (no model). For every client visit found in the cached structured states it
reports how many seconds each region (detector, counting machine, safe, laptop, desk) showed
motion. Output is numbers only; no frames are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REGIONS = ("detector", "counting_machine", "safe", "laptop", "desk", "door")


def _box(size, box, expand=1.15):
    w, h = size
    x1, y1, x2, y2 = box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(8.0, (x2 - x1) * expand), max(8.0, (y2 - y1) * expand)
    return (max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)), min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2)))


def curves(path: str, layout: dict, fps: float = 10.0, delta: int = 20):
    import av
    from PIL import ImageChops, ImageFilter

    regions = {k: layout[k] for k in REGIONS if k in layout}
    out = {k: [] for k in regions}
    times = []
    prev = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base)
        start = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
        next_t = 0.0
        for frame in container.decode(stream):
            t = float(frame.pts) * tb - start
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / fps
            img = frame.to_image().convert("L")
            times.append(round(t, 3))
            for k, box in regions.items():
                crop = img.crop(_box(img.size, box)).resize((48, 48)).filter(ImageFilter.GaussianBlur(1.0))
                if k in prev:
                    hist = ImageChops.difference(crop, prev[k]).histogram()
                    out[k].append(sum(hist[delta + 1:]) / max(1, sum(hist)))
                else:
                    out[k].append(0.0)
                prev[k] = crop
    return times, out


def analyse(args_tuple):
    name, path, row, fps, active = args_tuple
    from forte.fast.probes import sessions

    layout = row.get("layout") or {}
    times, cur = curves(path, layout, fps)
    dt = 1.0 / fps

    def active_s(key, a, b):
        if key not in cur:
            return None
        return round(sum(dt for t, v in zip(times, cur[key]) if a <= t <= b and v > active), 2)

    visits = []
    for s in sessions(row.get("states") or []):
        visits.append({"start": s["start"], "end": s["end"], "cash": s["cash"], "counting": s["counting"],
                       "bag": s["bag"], **{f"{k}_active_s": active_s(k, s["start"], s["end"]) for k in REGIONS}})
    whole = {f"{k}_active_s": active_s(k, 0, 1e9) for k in REGIONS}
    return {"name": name, "duration": times[-1] if times else 0, "regions": sorted(cur), "whole": whole, "visits": visits}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--active", type=float, default=0.08, help="fraction of ROI pixels moving to count a frame active")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in run["results"]}
    jobs = [(n, str(Path(args.videos) / f"{n}.mov"), rows[n], args.fps, args.active) for n in sorted(rows)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(analyse, jobs))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8")) if args.labels else {}
    for r in results:
        r["label"] = labels.get(r["name"])
    Path(args.out).write_text(json.dumps({"schema": "forte.roi-activity.v1", "fps": args.fps, "active": args.active,
                                          "results": results}, indent=1) + "\n", encoding="utf-8")
    for r in results:
        lab = r["label"] or {}
        tag = f"POS@{lab.get('t')}" if lab.get("violation") else "neg"
        print(r["name"], tag, "regions", ",".join(r["regions"]), "whole", json.dumps(r["whole"]))
        for v in r["visits"]:
            print("   visit", round(v["start"], 1), round(v["end"], 1), "cash" if v["cash"] else "-", "count" if v["counting"] else "-",
                  "bag" if v["bag"] else "-", {k[:-9]: v[k] for k in v if k.endswith("_active_s")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
