"""Dense window probes for procedure steps the per-second states cannot see (flag ``probe_types``).

Each probe follows prompt section 9.4: a structured perception signal proposes a window
(high recall), the local VLM answers a closed question over dense original frames of
that window under two differently worded phrasings, and a candidate is emitted only when
both phrasings agree that the required step is absent / the condition is present.
Windows and regions come from per-video states and layout; nothing references specific
clips, absolute times or pixel coordinates. A probe that cannot see its decisive region
abstains (no finding either way).
"""

from __future__ import annotations

import asyncio
from typing import Any

from forte.fast import prompts as P
from forte.fast.video import DecodedVideo
from forte.fast.vlm import image_part, text_part

ANSWER = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unclear"]}},
          "required": ["answer"], "additionalProperties": False}

INTRO = ("These {n} frames come in time order from a fixed CCTV camera over a bank cash desk "
         "(timestamps before each frame). The cashier sits behind the desk with a laptop, a large "
         "banknote counting machine and a small banknote authenticity detector; clients and cash "
         "collectors sit or stand on the other side. {extra}")

# Each probe: two phrasings. ``fire`` maps (answer_a, answer_b) -> True when the violation holds.
PROBES: dict[str, dict[str, Any]] = {
    "NO_AUTHENTICITY_CHECK": {
        "regions": ("detector", "counting_machine"),
        "a": ("Question: during this cash transaction, does the cashier at any moment place banknotes on, "
              "into or under the SMALL authenticity detector device (not the large counting machine) to check "
              "them? Answer yes if such a check is visible, no if the detector is visible and is never used "
              "for banknotes, unclear if the detector or the cashier's hands cannot be seen well enough."),
        "b": ("Question: is it visible that the banknotes of this transaction are NEVER checked on the small "
              "authenticity detector device (the separate small device, not the counting machine)? Answer yes "
              "if the detector stays unused for banknotes during the whole transaction, no if any banknote is "
              "checked on it, unclear if this cannot be seen."),
        "fire": lambda a, b: a == "no" and b == "yes",
    },
    "NO_CLIENT_ID": {
        "regions": ("desk",),
        "a": ("Question: before or while the client hands cash to the cashier, does the client hand over or "
              "show an identity document (an ID card or passport: a small rigid card or booklet, not banknotes) "
              "that the cashier takes or looks at? Answer yes if a document is presented, no if cash is handed "
              "over and no identity document is presented at any point, unclear if the hands and desk cannot be "
              "seen well enough."),
        "b": ("Question: is it visible that the cashier accepts cash from the client WITHOUT any identity "
              "document (ID card or passport) being presented or examined during this visit? Answer yes if "
              "cash is accepted and no document is ever presented, no if a document is presented, unclear if "
              "this cannot be seen."),
        "fire": lambda a, b: a == "no" and b == "yes",
    },
    "NO_RECOUNT": {
        "regions": ("counting_machine", "safe"),
        "a": ("Question: after the cash bag or cash bundles are received, are the banknotes recounted (put "
              "through the counting machine or counted by hand) before they are put away into the safe or "
              "drawer? Answer yes if a recount is visible, no if the cash is put away without any recount, "
              "unclear if this cannot be seen."),
        "b": ("Question: is it visible that received cash bundles are stored in the safe or drawer WITHOUT "
              "being recounted on the counting machine or by hand? Answer yes if stored without a recount, "
              "no if they are recounted first, unclear if this cannot be seen."),
        "fire": lambda a, b: a == "no" and b == "yes",
    },
}


def _b(state: dict[str, Any], key: str) -> bool:
    value = state.get(key)
    return value is True or value == "yes"


def _desk(state: dict[str, Any], key: str) -> bool:
    return bool((state.get("desk") or {}).get(key))


def _smooth(raw: list[bool], radius: int = 1) -> list[bool]:
    out = []
    for i in range(len(raw)):
        window = raw[max(0, i - radius): i + radius + 1]
        out.append(sum(window) * 2 > len(window))
    return out


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def sessions(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Visits of a client/collector at the desk, with what the states saw during them."""
    if len(states) < 3:
        return []
    ts = [s["t"] for s in states]
    client = _smooth([_b(s, "client_seat") or _b(s, "client_at_desk") for s in states])
    out = []
    for a, b in _runs(client):
        if b - a < 2:
            continue
        lo, hi = max(0, a - 2), min(len(states) - 1, b + 2)
        span = states[lo:hi + 1]
        out.append({
            "start": ts[lo], "end": ts[hi], "first_client": ts[a],
            "cash": any(_b(s, "cash_in_hands") or _desk(s, "cash") for s in span),
            "counting": any(_b(s, "counting_machine") or _b(s, "hand_counting") for s in span),
            "bag": any(_b(s, "cash_bag") or _desk(s, "cash_bag") for s in span),
            "cash_t": next((s["t"] for s in span if _b(s, "cash_in_hands") or _desk(s, "cash")), ts[a]),
            "bag_t": next((s["t"] for s in span if _b(s, "cash_bag") or _desk(s, "cash_bag")), ts[a]),
        })
    return out


def windows(states: list[dict[str, Any]], probe_types: list[str]) -> list[dict[str, Any]]:
    """High-recall windows per probe type, from structured states only."""
    out = []
    for s in sessions(states):
        if "NO_AUTHENTICITY_CHECK" in probe_types and s["cash"] and s["counting"] and not s["bag"]:
            out.append({"type_id": "NO_AUTHENTICITY_CHECK", "start": s["start"], "end": s["end"], "t": s["cash_t"]})
        if "NO_CLIENT_ID" in probe_types and s["cash"] and not s["bag"]:
            out.append({"type_id": "NO_CLIENT_ID", "start": s["start"], "end": s["end"], "t": s["cash_t"]})
        if "NO_RECOUNT" in probe_types and s["bag"]:
            out.append({"type_id": "NO_RECOUNT", "start": s["start"], "end": s["end"] + 20.0, "t": s["bag_t"]})
    return out


def _crop_native(image, box, expand: float = 1.4):
    w, h = image.size
    x1, y1, x2, y2 = box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(24.0, (x2 - x1) * expand), max(24.0, (y2 - y1) * expand)
    region = image.crop((max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)),
                         min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))))
    scale = min(2.0, max(1.0, 224 / max(1, min(region.size))))  # at most 2x (prompt S5)
    if scale > 1:
        region = region.resize((int(region.width * scale), int(region.height * scale)))
    return region


async def run_probe(client, video: DecodedVideo, window: dict[str, Any], layout: dict[str, list[float]],
                    cfg: dict[str, Any]) -> dict[str, Any]:
    spec = PROBES[window["type_id"]]
    frames = video.between(max(0.0, window["start"]), min(video.duration, window["end"]))
    count = int(cfg.get("probe_frames", 16))
    if len(frames) > count:
        step = (len(frames) - 1) / (count - 1)
        frames = [frames[round(i * step)] for i in range(count)]
    visible_regions = [r for r in spec["regions"] if r in layout]
    if not frames or not visible_regions:
        return {**window, "answers": [], "fired": False, "abstain": "region_not_located"}
    content: list[dict[str, Any]] = []
    for f in frames:
        content.append(text_part(f"t={f.t:.1f}s"))
        content.append(image_part(f.image, max_side=int(cfg.get("probe_max_side", 848))))
    mid = frames[len(frames) // 2]
    for region in visible_regions:
        content.append(text_part(f"[close-up of the {region.replace('_', ' ')} at t={mid.t:.1f}s; original pixels]"))
        content.append(image_part(_crop_native(mid.image, layout[region]), max_side=640))
    extra = "Report only what is clearly visible; do not guess."
    intro = text_part(INTRO.format(n=len(frames), extra=extra))

    async def ask(question: str) -> str:
        data = await client.chat_json([*content, intro, text_part(question + ' Answer JSON {"answer": "yes"|"no"|"unclear"}.')],
                                      system=P.SYSTEM, max_tokens=12, json_schema=ANSWER)
        return str(data.get("answer"))

    a, b = await asyncio.gather(ask(spec["a"]), ask(spec["b"]))
    return {**window, "answers": [a, b], "fired": bool(spec["fire"](a, b)), "frames": len(frames)}


async def probe_candidates(client, video: DecodedVideo, states: list[dict[str, Any]],
                           layout: dict[str, list[float]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    probe_types = [t for t in (cfg.get("probe_types") or []) if t in PROBES]
    if not probe_types:
        return []
    return list(await asyncio.gather(*(run_probe(client, video, w, layout, cfg) for w in windows(states, probe_types))))
