"""Local-model 'teacher' over dense windows: closed questions, enum answers only (prompt 6.2).

  python tools/teacher_windows.py --videos /videos --windows X-1@22,X-2@10 --out teacher.json

For each window (video @ centre second) it asks the local VLM, over dense original frames,
which catalog violation (if any) is visible, what the cashier and the visitor are doing,
and the safe state. Two frame samplings per window. Results are enums only; no free text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CASHIER = ["counting_banknotes_by_hand", "using_counting_machine", "using_small_detector_device", "taking_cash_from_visitor",
           "giving_cash_to_visitor", "handling_cash_bag", "looking_at_document_or_card", "at_the_safe", "using_laptop",
           "leaving_or_away", "sitting_idle", "other"]
VISITOR = ["none_present", "entering_or_leaving", "handing_cash", "handing_document_or_card", "handing_cash_bag",
           "receiving_cash", "waiting_seated", "standing_near_cashier", "other"]


def _schema(types):
    enum = lambda vals: {"type": "string", "enum": vals}  # noqa: E731
    return {"type": "object", "properties": {
        "violation": enum(types + ["NONE"]), "cashier": enum(CASHIER), "visitor": enum(VISITOR),
        "safe_door": enum(["open", "closed", "not_visible"]), "key_in_safe_lock": enum(["yes", "no", "unclear"]),
        "door_of_room": enum(["open", "closed", "not_visible"]), "people_in_room": {"type": "integer", "minimum": 0, "maximum": 6}},
        "required": ["violation", "cashier", "visitor", "safe_door", "key_in_safe_lock", "door_of_room", "people_in_room"],
        "additionalProperties": False}


async def main_async(args):
    from forte.fast import prompts as P
    from forte.fast.analyzer import DEFAULT_CONFIG
    from forte.fast.catalog import TYPES
    from forte.fast.service import decode_for_analysis
    from forte.fast.vlm import VLMClient, image_part, text_part

    types = [t.type_id for t in TYPES]
    checklist = "\n".join(f"- {t.type_id}: {t.cue_en}" for t in TYPES)
    schema = _schema(types)
    client = VLMClient(args.base_url, args.model, concurrency=48)
    cfg = dict(DEFAULT_CONFIG)
    windows = []
    for item in args.windows.split(","):
        name, t = item.split("@")
        windows.append((name, float(t)))
    by_video = {}
    for name, t in windows:
        by_video.setdefault(name, []).append(t)
    sem = asyncio.Semaphore(4)
    out = []

    async def video_job(name, centres):
        async with sem:
            video, _ = await asyncio.to_thread(decode_for_analysis, str(Path(args.videos) / f"{name}.mov"), cfg)
            try:
                async def ask(t, offset):
                    frames = video.between(max(0, t - args.half + offset), min(video.duration, t + args.half + offset))
                    if len(frames) > args.frames:
                        step = (len(frames) - 1) / (args.frames - 1)
                        frames = [frames[round(i * step)] for i in range(args.frames)]
                    content = []
                    for f in frames:
                        content += [text_part(f"t={f.t:.1f}s"), image_part(f.image, max_side=args.max_side)]
                    content.append(text_part(
                        f"These {len(frames)} frames (time order) are from a fixed CCTV camera over a bank cash desk. "
                        "The cashier works behind the desk (laptop, banknote counting machine, small banknote detector, safe); "
                        "visitors (clients, cash collectors) come through the door. Answer about what is clearly visible.\n"
                        f"Possible procedure violations:\n{checklist}\n"
                        "violation: the ONE violation clearly visible in these frames, or NONE.\n"
                        "cashier / visitor: main action in these frames. safe_door, key_in_safe_lock, door_of_room, people_in_room: "
                        "as visible. Answer JSON only."))
                    return await client.chat_json(content, system=P.SYSTEM, max_tokens=120, json_schema=schema)
                for t in centres:
                    a, b = await asyncio.gather(ask(t, 0.0), ask(t, args.shift))
                    out.append({"name": name, "t": t, "a": a, "b": b})
                    print(name, t, "A", json.dumps(a), "B", json.dumps(b), flush=True)
            finally:
                for f in video.frames:
                    f.image.close()

    await asyncio.gather(*(video_job(n, c) for n, c in by_video.items()))
    await client.aclose()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--half", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--max-side", type=int, default=848)
    ap.add_argument("--base-url", default="http://vlm30:8000/v1")
    ap.add_argument("--model", default="q30")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = asyncio.run(main_async(args))
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
