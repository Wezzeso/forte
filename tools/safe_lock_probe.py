"""Diagnostic-only native safe/lock perception; never emits violations or candidates.

Uses only names and layout from a cached fast_eval report. Every video receives
the same evenly spaced window schedule. Labels are not accepted. Original native
frames and native crops are sent only to a loopback local VLM; no images or model
free text are saved, printed, or returned in this tool's report.

  python tools/safe_lock_probe.py --run run.json --videos /videos \
    --model gemma4 --out /runs/safe-lock.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from jsonschema import validate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forte.fast.video import decode  # noqa: E402
from forte.fast.vlm import image_part, text_part  # noqa: E402

ENUM = lambda values: {"type": "string", "enum": values}  # noqa: E731
BOX = {"anyOf": [{"type": "null"}, {"type": "array", "items": {
    "type": "number", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4}]}
LOCATE_SCHEMA = {"type": "object", "properties": {
    "lock_box": BOX,
    "lock_visibility": ENUM(["resolved", "occluded", "not_visible", "too_small"]),
    "safe_door": ENUM(["open", "closed", "unclear"])},
    "required": ["lock_box", "lock_visibility", "safe_door"], "additionalProperties": False}

LOCALIZE = ("Image 1 is an original CCTV frame; image 2 is a native-pixel crop of the "
    "safe located independently for this video. In IMAGE 2 locate the lock cylinder/keyhole "
    "itself, whether or not a key is present. Give lock_box=[x1,y1,x2,y2] normalized 0-1000 "
    "relative to IMAGE 2 only. Use null if its location cannot be visually resolved. "
    "Do not assume a key or lock from what safes usually look like. "
    "lock_visibility=resolved only if the actual lock is distinguishable. JSON only.")

QUESTIONS = (
    "For each numbered native lock close-up, classify the actual visible lock. "
    "key_state=inserted only when an unmistakable key visibly protrudes from or is "
    "inserted in that lock. Use empty only when the lock opening is resolved with "
    "no inserted key. Use unclear if pixels, occlusion, alignment or viewing angle "
    "do not distinguish these states. An apparent handle, latch or highlight is not "
    "a key. hand_at_lock=yes means a hand visibly touches the lock/key in that frame.",
    "Independently assess each numbered close-up: which lock condition is directly "
    "supported by pixels, inserted, empty, or unclear? A resolved empty keyhole "
    "supports empty; a clearly distinguishable key entering/protruding from that "
    "keyhole supports inserted. All other cases support unclear. Do not infer "
    "inserted from a safe being open, a hand having been there, or expected behavior. "
    "Report whether a hand currently contacts the lock/key separately.",
)


def classification_schema(count: int) -> dict[str, Any]:
    return {"type": "object", "properties": {"frames": {"type": "array",
        "minItems": count, "maxItems": count, "items": {"type": "object", "properties": {
            "index": {"type": "integer", "minimum": 0, "maximum": count - 1},
            "key_state": ENUM(["inserted", "empty", "unclear"]),
            "lock_visibility": ENUM(["resolved", "occluded", "not_visible", "too_small"]),
            "hand_at_lock": ENUM(["yes", "no", "unclear"])},
            "required": ["index", "key_state", "lock_visibility", "hand_at_lock"],
            "additionalProperties": False}},
        "cashier_near_safe": ENUM(["yes", "no", "unclear"])},
        "required": ["frames", "cashier_near_safe"], "additionalProperties": False}


def ordered_box(box: Any) -> bool:
    return (isinstance(box, list) and len(box) == 4 and all(
        type(value) in (int, float) and math.isfinite(value) for value in box)
        and 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)


def cached_safe_box(source: dict[str, Any]) -> list[float] | None:
    # Inspect processing provenance, never correctness/labels or error text.
    if source.get("processed") is False or source.get("status") == "failed" or source.get("error"):
        raise ValueError("cached_source_failed")
    layout = source.get("layout")
    if not isinstance(layout, dict) or not layout:
        raise ValueError("cached_layout_unavailable")
    box = layout.get("safe")
    if box is not None and not ordered_box(box):
        raise ValueError("invalid_cached_safe_box")
    return box


def native_box(size: tuple[int, int], box: list[float], *, expand: float = 1.0,
               min_side: int = 1) -> tuple[int, int, int, int]:
    if not ordered_box(box):
        raise ValueError("invalid_normalized_box")
    width, height = size
    x1, y1, x2, y2 = [value * axis / 1000 for value, axis in zip(box, (width, height, width, height))]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(min_side, (x2 - x1) * expand), max(min_side, (y2 - y1) * expand)
    result = (max(0, math.floor(cx - bw / 2)), max(0, math.floor(cy - bh / 2)),
              min(width, math.ceil(cx + bw / 2)), min(height, math.ceil(cy + bh / 2)))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("empty_native_crop")
    return result


def evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return items
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def local_base_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = False
    if parsed.scheme != "http" or not loopback or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("base URL must be an HTTP loopback URL")
    return value.rstrip("/")


class LocalEnumClient:
    def __init__(self, base_url: str, model: str, concurrency: int, timeout: float):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout,
                                       trust_env=False, follow_redirects=False)
        self.model, self.sem = model, asyncio.Semaphore(concurrency)
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    async def ask(self, content: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        async with self.sem:
            response = await self.client.post("/chat/completions", json={
                "model": self.model, "seed": 0, "temperature": 0.0, "max_tokens": 700,
                "messages": [{"role": "system", "content": "Inspect visible pixels only. "
                    "Never identify people or infer invisible facts. Return JSON only."},
                    {"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}})
        response.raise_for_status()
        data = response.json()
        self.calls += 1
        usage = data.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        choice = data["choices"][0]
        message = choice["message"]
        raw = message["content"]
        if choice.get("finish_reason") != "stop" or any(message.get(key) for key in ("reasoning", "reasoning_content")):
            raise ValueError("incomplete_or_reasoning_response")
        def reject_constant(value: str):
            raise ValueError("nonfinite_json")
        value = json.loads(raw, parse_constant=reject_constant)
        validate(value, schema)
        return value


async def probe_window(client: LocalEnumClient, frames, safe_box: list[float], centre: float) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {"centre_sec": round(centre, 3), "frame_times_sec": [f.t for f in frames],
                           "status": "failed"}
    crops = []
    try:
        middle = frames[len(frames) // 2]
        roi = native_box(middle.image.size, safe_box)
        safe = middle.image.crop(roi)
        crops.append(safe)
        localized = await client.ask([image_part(middle.image), image_part(safe), text_part(LOCALIZE)], LOCATE_SCHEMA)
        row["localization"] = localized
        if localized["lock_box"] is None or localized["lock_visibility"] != "resolved":
            row["status"] = "abstained"
            row["abstain_reason"] = "lock_not_resolved"
            return row
        lock = native_box(safe.size, localized["lock_box"], expand=2.0, min_side=24)
        native_lock = (roi[0] + lock[0], roi[1] + lock[1], roi[0] + lock[2], roi[1] + lock[3])
        row["safe_crop_native_xyxy"] = list(roi)
        row["lock_crop_native_xyxy"] = list(native_lock)
        content = [image_part(middle.image), text_part("Original full frame for role and safe context."),
                   image_part(safe), text_part("Native safe crop for lock context; no resizing/enhancement.")]
        for index, frame in enumerate(frames):
            region = frame.image.crop(native_lock)
            crops.append(region)
            content.extend([image_part(region), text_part(f"lock_frame index={index}, t={frame.t:.3f}s")])
        suffix = (" Answer about every numbered lock_frame, in index order. "
                  "cashier_near_safe refers to the original full frame, by role only. "
                  "Return only schema fields; do not assert any rule violation.")
        answers = await asyncio.gather(*(client.ask([*content, text_part(question + suffix)],
            classification_schema(len(frames))) for question in QUESTIONS))
        for answer in answers:
            if [item["index"] for item in answer["frames"]] != list(range(len(frames))):
                raise ValueError("frame_index_alignment_failed")
        row["answers"] = answers
        row["paired_key_state_agreements"] = sum(a["key_state"] == b["key_state"]
            for a, b in zip(answers[0]["frames"], answers[1]["frames"]))
        row["status"] = "done"
    except Exception as exc:
        row["error_type"] = type(exc).__name__
    finally:
        for picture in crops:
            picture.close()
        row["wall_s"] = round(time.monotonic() - started, 3)
    return row


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", type=local_base_url, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--par", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if not (2 <= args.windows <= 12 and 2 <= args.frames <= 10 and 0 < args.window_sec <= 10
            and 0 < args.fps <= 10 and 1 <= args.par <= 4 and 0 < args.timeout <= 600):
        parser.error("invalid diagnostic sampling/concurrency limits")
    run_path = Path(args.run)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    source_rows = run["results"]
    names = [row["name"] for row in source_rows]
    if not names or len(names) != len(set(names)) or any(not isinstance(name, str)
            or Path(name).name != name or "/" in name or "\\" in name or ":" in name for name in names):
        raise ValueError("invalid_report_names")
    sem = asyncio.Semaphore(args.par)
    client = LocalEnumClient(args.base_url, args.model, max(4, args.par * 2), args.timeout)

    async def one(source):
        async with sem:
            row: dict[str, Any] = {"name": source["name"], "status": "failed", "windows": []}
            video = None
            try:
                box = cached_safe_box(source)
                if box is None:
                    row.update(status="abstained", abstain_reason="safe_not_located")
                    return row
                video = await asyncio.to_thread(decode, str(Path(args.videos) / f"{source['name']}.mov"),
                                               args.fps, max_frames=1800)
                first, last = video.frames[0].t, video.frames[-1].t
                centres = [first + i * (last - first) / (args.windows - 1) for i in range(args.windows)]
                for centre in centres:
                    selected = evenly_spaced(video.between(centre - args.window_sec / 2,
                                                          centre + args.window_sec / 2), args.frames)
                    if not selected:
                        raise ValueError("empty_uniform_window")
                    row["windows"].append(await probe_window(client, selected, box, centre))
                row["status"] = "failed" if any(w["status"] == "failed" for w in row["windows"]) else "done"
            except Exception as exc:
                row["error_type"] = type(exc).__name__
            finally:
                if video is not None:
                    for frame in video.frames:
                        frame.image.close()
            return row

    started = time.monotonic()
    try:
        results = await asyncio.gather(*(one(row) for row in source_rows))
    finally:
        await client.client.aclose()
    report = {"schema": "forte.safe-lock-perception-diagnostic.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "claim_scope": "perception_diagnostic_only", "enables_types": False,
        "labels_used": False, "accuracy_claim": False, "tuned_parameter_count": 0,
        "config": {"windows": args.windows, "frames": args.frames, "window_sec": args.window_sec,
                   "fps": args.fps, "lock_crop_expand": 2.0, "lock_min_side_px": 24,
                   "native_crops_only": True, "temporal_fusion": False},
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "calls": client.calls, "prompt_tokens": client.prompt_tokens,
        "completion_tokens": client.completion_tokens, "wall_s": round(time.monotonic() - started, 3),
        "results": results}
    failed = sum(row["status"] == "failed" for row in results)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"videos": len(results), "failed": failed, "calls": client.calls,
                      "windows": sum(len(row["windows"]) for row in results)}))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        raise SystemExit(2) from None
