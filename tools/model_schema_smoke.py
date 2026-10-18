"""Synthetic-only preflight for a loopback OpenAI-compatible local VLM.

Does not accept image/video paths, inspect supplied media, or print model text.
Exercises every schema in forte.fast.prompts, strict JSON, the concise verifier
stop sequence, 2/14/18/32-image inputs, and an independently known normalized box.
This is a compatibility diagnostic, never an accuracy or release gate.

  python tools/model_schema_smoke.py --model gemma4 --out /runs/gemma4-smoke.json
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from jsonschema import validate
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forte.fast import prompts as P  # noqa: E402
from forte.fast.catalog import TYPES  # noqa: E402
from forte.fast.vlm import close_truncated_object, image_part, text_part  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    schema: dict[str, Any]
    prompt: str
    image_kind: str = "room"
    images: int = 1
    stop: bool = False


def cases() -> list[Case]:
    cue = next(item for item in TYPES if item.type_id == "PERSONAL_ITEMS")
    verify = P.VERIFY_PROMPT.format(n=2, t=1.0, cue=cue.cue_en, correct=cue.correct_en)
    rectangle_schema = {
        "type": "object", "properties": {"box": {"type": "array",
            "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
        "required": ["box"], "additionalProperties": False,
    }
    capacity_schema = {"type": "object", "properties": {"answer": {
        "type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"], "additionalProperties": False}
    result = [
        Case("LAYOUT_SCHEMA", P.LAYOUT_SCHEMA, P.LAYOUT_PROMPT),
        Case("STATE_SCHEMA", P.STATE_SCHEMA, P.STATE_PROMPT.format(t=0.0)),
        Case("HOLISTIC_SCHEMA", P.HOLISTIC_SCHEMA, P.HOLISTIC_PROMPT.format(
            n=2, duration=1.0, checklist=P.checklist()), images=2),
        Case("VERIFY_SCHEMA", P.VERIFY_SCHEMA, verify, images=2),
        Case("VERIFY_DECISION_SCHEMA", P.VERIFY_DECISION_SCHEMA, verify, images=2),
        Case("LAPTOP_SCHEMA", P.LAPTOP_SCHEMA, P.LAPTOP_Q),
        Case("DOOR_SCHEMA", P.DOOR_SCHEMA, P.DOOR_Q),
        Case("DESK_SCHEMA", P.DESK_SCHEMA, P.DESK_Q),
        Case("SEAT_SCHEMA", P.SEAT_SCHEMA, P.SEAT_Q),
        Case("DOOR_COMPARE", P.DOOR_SCHEMA, P.DOOR_CMP_Q, images=2),
        Case("NORMALIZED_BOX", rectangle_schema,
             "Locate only the red rectangle. Return box=[x1,y1,x2,y2] normalized "
             "to 0-1000 relative to image width and height.", image_kind="rectangle"),
        Case("VERIFY_CONCISE_STOP", P.VERIFY_SCHEMA, verify, images=2, stop=True),
    ]
    result.extend(Case(f"IMAGE_CAPACITY_{count}", capacity_schema,
        "Are all inputs synthetic diagrams with no real people? Answer yes or no.", images=count)
        for count in (14, 18, 32))
    covered = {case.name for case in result}
    schema_names = {name for name in vars(P) if name.endswith("_SCHEMA")}
    if not schema_names <= covered:
        raise ValueError("uncovered_schema")
    return result


def synthetic_image(kind: str) -> Image.Image:
    """All pixels are generated here; no input image or media is read."""
    canvas = Image.new("RGB", (640, 960), "white")
    draw = ImageDraw.Draw(canvas)
    if kind == "rectangle":
        draw.rectangle((160, 240, 480, 720), fill="red")
        return canvas
    draw.rectangle((25, 30, 615, 930), outline="black", width=5)
    shapes = [
        ((90, 320, 550, 650), "#bb9966", "desk"),
        ((140, 340, 285, 425), "#555555", "laptop"),
        ((315, 360, 440, 450), "#999999", "counting_machine"),
        ((455, 375, 515, 430), "#ddddee", "detector"),
        ((90, 120, 250, 260), "#bbbbbb", "safe"),
        ((430, 60, 565, 220), "#e0ffff", "door"),
        ((190, 245, 290, 315), "#dddddd", "cashier_chair"),
        ((335, 670, 435, 750), "#dddddd", "client_chair"),
    ]
    for bounds, color, label in shapes:
        draw.rectangle(bounds, fill=color, outline="black", width=3)
        draw.text((bounds[0] + 4, bounds[1] + 6), label, fill="black")
    draw.rectangle((150, 350, 275, 400), fill="black")
    draw.text((25, 880), "SYNTHETIC DRAWING - no people - no cash", fill="black")
    return canvas


def strict_loads(value: str) -> Any:
    def finite_number(token: str) -> float:
        number = float(token)
        if not math.isfinite(number):
            raise ValueError("nonfinite_number")
        return number

    def reject_constant(token: str) -> None:
        raise ValueError("nonfinite_number")

    return json.loads(value, parse_float=finite_number, parse_constant=reject_constant)


def safe_projection(value: Any, schema: dict[str, Any]) -> Any:
    """Only schema-validated enums, finite numbers, booleans, boxes, and null."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    if isinstance(value, str):
        return value if value in schema.get("enum", []) else None
    if isinstance(value, list):
        return [safe_projection(item, schema.get("items", {})) for item in value]
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return {key: safe_projection(item, properties.get(key, {}))
                for key, item in value.items() if key in properties}
    return None


def text_field_count(value: Any, schema: dict[str, Any]) -> int:
    if isinstance(value, str):
        return int(bool(value) and "enum" not in schema)
    if isinstance(value, list):
        return sum(text_field_count(item, schema.get("items", {})) for item in value)
    if isinstance(value, dict):
        return sum(text_field_count(item, schema.get("properties", {}).get(key, {}))
                   for key, item in value.items())
    return 0


def box_checks(case: Case, value: Any) -> dict[str, Any]:
    if case.name == "NORMALIZED_BOX":
        box = value["box"]
        error = max(abs(actual - expected) for actual, expected in zip(box, [250, 250, 750, 750]))
        return {"normalized_box_ok": error <= 65.0, "max_coordinate_error": round(error, 3)}
    if case.name != "LAYOUT_SCHEMA":
        return {}
    boxes = [box for box in value.values() if box is not None]
    return {"non_null_boxes": len(boxes), "layout_box_bounds_ok": all(
        0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000 for box in boxes)}


async def run_case(client: httpx.AsyncClient, model: str, case: Case,
                   images: dict[str, Image.Image], template_kwargs: dict[str, Any]) -> dict[str, Any]:
    start = time.monotonic()
    row: dict[str, Any] = {"case": case.name, "images": case.images,
                           "passed": False, "status": "failed"}
    try:
        content = []
        for index in range(case.images):
            if case.name.startswith("IMAGE_CAPACITY_"):
                # Different pixels prevent encoder/cache deduplication from
                # masking the capacity needed for distinct video frames.
                picture = images[case.image_kind].copy()
                try:
                    draw = ImageDraw.Draw(picture)
                    draw.rectangle((5, 5, 635, 28), fill=(
                        (case.images * 31 + index * 13) % 256,
                        (case.images * 11 + index * 17) % 256,
                        (case.images * 7 + index * 23) % 256))
                    draw.text((10, 9), f"SYNTHETIC {case.images} IMAGE {index}", fill="black")
                    part = image_part(picture)
                finally:
                    picture.close()
            else:
                part = image_part(images[case.image_kind])
            content += [text_part(f"t={index:.1f}s"), part]
        content.append(text_part(case.prompt + "\nDiagnostic input: all pixels are a synthetic "
            "drawing, no real people or events. Use an empty string for any observation field. "
            "Return only the requested JSON object with no fences or explanation."))
        body: dict[str, Any] = {
            "model": model, "messages": [{"role": "system", "content": P.SYSTEM},
                                           {"role": "user", "content": content}],
            "seed": 0, "temperature": 0.0, "max_tokens": 640,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "out", "schema": case.schema}},
        }
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        if case.stop:
            body["stop"] = [P.VERIFY_STOP]
        response = await client.post("/chat/completions", json=body)
        row["http_status"] = response.status_code
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        raw = message.get("content")
        if not isinstance(raw, str):
            raise ValueError("content_not_string")
        row["reasoning_present"] = any(bool(message.get(key)) for key in ("reasoning", "reasoning_content"))
        row["thought_markup_present"] = any(marker in raw for marker in ("<think", "</think", "<|channel>", "<channel|>"))
        row["finish_reason"] = choice.get("finish_reason") if choice.get("finish_reason") in {
            "stop", "length", "tool_calls", "content_filter"} else "unknown"
        value = strict_loads(close_truncated_object(raw) if case.stop else raw)
        if case.stop:
            row["stop_truncation_observed"] = not raw.rstrip().endswith("}")
            row["observation_field_absent"] = isinstance(value, dict) and "observation_ru" not in value
            if not row["stop_truncation_observed"] or not row["observation_field_absent"]:
                raise ValueError("concise_stop_not_honored")
        validated_schema = P.VERIFY_DECISION_SCHEMA if case.stop else case.schema
        validate(value, validated_schema)
        row["schema_valid"] = True
        if case.stop:
            # A text-first object is cut at observation_ru before decision keys;
            # required-field validation above must fail that response.
            row["required_decisions_complete_before_stop"] = True
        row["result"] = safe_projection(value, validated_schema)
        row["nonempty_free_text_fields_dropped"] = text_field_count(value, validated_schema)
        row.update(box_checks(case, value))
        usage = data.get("usage") or {}
        row["usage"] = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if type(usage.get(key)) is int and usage[key] >= 0}
        row["passed"] = row["finish_reason"] == "stop" and not row["reasoning_present"] and not row["thought_markup_present"] and all(
            row.get(key, True) for key in ("normalized_box_ok", "layout_box_bounds_ok"))
        row["status"] = "passed" if row["passed"] else "failed"
    except Exception as exc:
        # Exception messages can contain server output or invalid generated text.
        row["error_type"] = type(exc).__name__
    row["wall_s"] = round(time.monotonic() - start, 3)
    return row


def local_base_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = False
    if parsed.scheme != "http" or not loopback or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("base URL must be an HTTP loopback URL")
    return value.rstrip("/")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=local_base_url, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Also send enable_thinking=false per request; default tests server defaults")
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 600:
        parser.error("timeout must be between 0 and 600 seconds")
    template_kwargs = {"enable_thinking": False} if args.disable_thinking else {}
    images = {kind: synthetic_image(kind) for kind in ("room", "rectangle")}
    try:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout,
                                     trust_env=False, follow_redirects=False) as client:
            rows = [await run_case(client, args.model, case, images, template_kwargs) for case in cases()]
    finally:
        for picture in images.values():
            picture.close()
    passed = all(row["passed"] for row in rows)
    report = {
        "schema": "forte.synthetic-model-schema-smoke.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "synthetic_only": True, "accuracy_claim": False,
        "inference_executed": True, "passed": passed, "case_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows), "request_template_kwargs": template_kwargs,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
        "geometry_tolerance_normalized": 65, "results": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "case_count": len(rows),
                      "passed_count": report["passed_count"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as error:
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        raise SystemExit(2) from None
