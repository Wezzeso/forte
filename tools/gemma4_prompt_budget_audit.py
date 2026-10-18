"""Synthetic-only audit of the analyzer's exact local VLM request contracts.

Unlike model_schema_smoke.py, this calls the existing analyzer with generated
pixels and captures its requests without adding diagnostic instructions or
changing max_tokens, image ordering, schemas, stop strings, or template kwargs.
It then replays those requests against a loopback-only local model. A length
finish always fails, even when a complete JSON prefix could be recovered.

No image/video input paths are accepted. No model text, pixels, or HTTP error
body is emitted. This is a compatibility diagnostic, never an accuracy claim.

  python tools/gemma4_prompt_budget_audit.py --model gemma4 --out /runs/budget.json
  python tools/gemma4_prompt_budget_audit.py --model gemma4 --dry-run --out plan.json
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

from forte.fast import analyzer as A  # noqa: E402
from forte.fast import prompts as P  # noqa: E402
from forte.fast import vlm as V  # noqa: E402
from forte.fast.video import DecodedVideo, Frame  # noqa: E402


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def strict_loads(value: str) -> Any:
    def finite(token: str) -> float:
        number = float(token)
        if not math.isfinite(number):
            raise ValueError("nonfinite_number")
        return number

    def reject_constant(token: str) -> None:
        raise ValueError("nonfinite_number")

    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = item
        return result

    return json.loads(value, parse_float=finite, parse_constant=reject_constant,
                      object_pairs_hook=unique)


@dataclass(frozen=True)
class Request:
    name: str
    content: list[dict[str, Any]]
    kwargs: dict[str, Any]
    active: bool

    def body(self, model: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.kwargs.get("system"):
            messages.append({"role": "system", "content": self.kwargs["system"]})
        messages.append({"role": "user", "content": self.content})
        body: dict[str, Any] = {
            "model": model, "messages": messages,
            "max_tokens": self.kwargs.get("max_tokens", 512),
            "temperature": self.kwargs.get("temperature", 0.0), "seed": 0,
        }
        schema = self.kwargs.get("json_schema")
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "out", "schema": schema}}
        if self.kwargs.get("stop"):
            body["stop"] = list(self.kwargs["stop"])
        return body

    def metadata(self) -> dict[str, Any]:
        schema = self.kwargs["json_schema"]
        return {
            "case": self.name, "active_in_speed_profile": self.active,
            "images": sum(part.get("type") == "image_url" for part in self.content),
            "max_tokens": self.kwargs.get("max_tokens", 512),
            "stop": list(self.kwargs.get("stop") or []),
            "close_truncated_object": bool(self.kwargs.get("truncated")),
            "schema_sha256": digest(schema),
            "validation_schema_sha256": digest(self.kwargs.get("validate_schema") or schema),
            "prompt_sha256": digest([part["text"] for part in self.content
                                      if part.get("type") == "text"]),
            "system_sha256": digest(self.kwargs.get("system")),
        }


BOXES = {
    "desk": [120, 340, 880, 680], "laptop": [200, 360, 440, 450],
    "safe": [120, 110, 370, 270], "door": [660, 70, 890, 260],
    "counting_machine": [510, 370, 700, 470], "detector": [740, 390, 840, 460],
    "cashier_chair": [270, 270, 440, 330], "client_chair": [530, 700, 700, 790],
}


class CaptureClient:
    """Capture actual analyzer calls; placeholder facts are never model results."""

    def __init__(self) -> None:
        self.requests: list[Request] = []
        self.phase = ""
        self.seats = 0
        self.active = True

    async def chat_json(self, content: list[dict[str, Any]], **kwargs: Any) -> Any:
        schema = kwargs["json_schema"]
        if schema is P.LAYOUT_SCHEMA:
            name, result = "LAYOUT", dict(BOXES)
        elif schema is P.STATE_SCHEMA:
            name = "STATE"
            result = {key: (0 if key == "other_people" else False)
                      for key in schema["required"]}
        elif schema is P.LAPTOP_SCHEMA:
            name, result = "LAPTOP_CROP", {"answer": "C"}
        elif schema is P.DESK_SCHEMA:
            name, result = "DESK_CROP", {key: False for key in schema["required"]}
        elif schema is P.SEAT_SCHEMA:
            name = ("CASHIER_CHAIR_CROP", "CLIENT_CHAIR_CROP")[self.seats]
            self.seats += 1
            result = {"answer": "no"}
        elif schema is P.VERIFY_SCHEMA:
            name = "VERIFY_CONCISE" if kwargs.get("truncated") else "VERIFY_FULL"
            result = {"visible": False, "confidence": 0.0, "t_sec": 5.5}
            if not kwargs.get("truncated"):
                result["observation_ru"] = ""
        elif schema is P.HOLISTIC_SCHEMA:
            name, result = "HOLISTIC", {"violations": []}
        else:
            raise ValueError("unexpected_analyzer_schema")
        self.requests.append(Request(name, content, kwargs, self.active))
        return result


def synthetic_video() -> DecodedVideo:
    """Known generated shapes only, matching source resolution; no file reads."""
    frames: list[Frame] = []
    for index in range(24):
        image = Image.new("RGB", (464, 848), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 455, 839), outline="black", width=3)
        for key, box in BOXES.items():
            bounds = [int(value * (464 if axis % 2 == 0 else 848) / 1000)
                      for axis, value in enumerate(box)]
            draw.rectangle(bounds, fill="#bbbbbb", outline="black", width=2)
            draw.text((bounds[0] + 2, bounds[1] + 2), key, fill="black")
        draw.rectangle((12, 12, 450, 30), fill=(index * 9 % 256, 40, 160))
        draw.text((12, 810), "SYNTHETIC - no people", fill="black")
        frames.append(Frame(t=index / 2, image=image))
    return DecodedVideo(duration=11.5, fps=30.0, width=464, height=848, frames=frames)


async def capture_requests(*, include_inactive: bool = False) -> list[Request]:
    client = CaptureClient()
    video = synthetic_video()
    cfg = {**A.DEFAULT_CONFIG, "use_verify": True, "verify_concise": True,
           "parallel_layout": True, "job_priority": True}
    try:
        await A.locate_layout(client, video, cfg)
        # A one-frame generated video elicits one actual state request and each
        # crop contract; fixed generated boxes keep model perception irrelevant.
        singleton = DecodedVideo(duration=0.0, fps=30.0, width=464, height=848,
                                 frames=[video.frames[0]])
        await A.extract_states(client, singleton, cfg, dict(BOXES))
        candidate = A.Candidate(type_id="PERSONAL_ITEMS", t=5.5, source="rule")
        await A.verify(client, video, candidate, cfg, dict(BOXES))
        if include_inactive:
            client.active = False
            await A.verify(client, video, candidate, {**cfg, "verify_concise": False}, dict(BOXES))
            await A.holistic(client, video, cfg)
        return client.requests
    finally:
        for frame in video.frames:
            frame.image.close()


def projection(value: Any, schema: dict[str, Any]) -> Any:
    """Retain booleans, finite numbers, boxes and declared enums only."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if value in schema.get("enum", []) else None
    if isinstance(value, list):
        return [projection(item, schema.get("items", {})) for item in value]
    if isinstance(value, dict):
        return {key: projection(item, schema.get("properties", {}).get(key, {}))
                for key, item in value.items() if key in schema.get("properties", {})}
    return None


async def replay(client: httpx.AsyncClient, model: str, request: Request) -> dict[str, Any]:
    row = {**request.metadata(), "passed": False, "status": "failed"}
    started = time.monotonic()
    try:
        response = await client.post("/chat/completions", json=request.body(model))
        row["http_status"] = response.status_code
        response.raise_for_status()
        reply = response.json()
        choice = reply["choices"][0]
        message = choice["message"]
        raw = message.get("content")
        if not isinstance(raw, str):
            raise ValueError("content_not_string")
        finish = choice.get("finish_reason")
        row["finish_reason"] = finish if finish in {
            "stop", "length", "tool_calls", "content_filter"} else "unknown"
        row["reasoning_present"] = any(bool(message.get(key))
                                       for key in ("reasoning", "reasoning_content"))
        row["thought_markup_present"] = any(marker in raw for marker in (
            "<think", "</think", "<|channel>", "<channel|>"))
        usage = reply.get("usage") or {}
        row["usage"] = {key: usage[key] for key in (
            "prompt_tokens", "completion_tokens", "total_tokens")
                        if type(usage.get(key)) is int and usage[key] >= 0}
        # Check termination before JSON repair: a length-truncated concise
        # prefix may happen to contain every decision field and still is invalid.
        if finish != "stop":
            raise ValueError("completion_not_stopped")
        if row["reasoning_present"] or row["thought_markup_present"]:
            raise ValueError("unexpected_reasoning")
        if row["usage"].get("completion_tokens", 0) > row["max_tokens"]:
            raise ValueError("completion_budget_exceeded")
        truncated = bool(request.kwargs.get("truncated"))
        value = strict_loads(V.close_truncated_object(raw) if truncated else raw)
        schema = request.kwargs.get("validate_schema") or request.kwargs["json_schema"]
        validate(value, schema)
        row["schema_valid"] = True
        if truncated:
            row["stop_truncation_observed"] = not raw.rstrip().endswith("}")
            row["observation_field_absent"] = "observation_ru" not in value
            if not row["stop_truncation_observed"] or not row["observation_field_absent"]:
                raise ValueError("concise_stop_not_honored")
            row["required_decisions_complete_before_stop"] = True
        if request.name == "LAYOUT":
            boxes = [box for box in value.values() if box is not None]
            row["non_null_boxes"] = len(boxes)
            row["layout_box_bounds_ok"] = all(
                0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000 for box in boxes)
            if not row["layout_box_bounds_ok"]:
                raise ValueError("layout_bounds_invalid")
        row["result"] = projection(value, schema)
        row["passed"], row["status"] = True, "passed"
    except Exception as error:
        # Error messages can include model free text or HTTP response bodies.
        row["error_type"] = type(error).__name__
    row["wall_s"] = round(time.monotonic() - started, 3)
    return row


def local_base_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        loopback = False
    if (parsed.scheme != "http" or not loopback or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError("base URL must be an HTTP loopback URL")
    return value.rstrip("/")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=local_base_url, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true", help="Capture contracts without model inference")
    parser.add_argument("--include-inactive", action="store_true",
                        help="Also test full free-text verifier and holistic paths, disabled in speed profile")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 600:
        parser.error("timeout must be between 0 and 600 seconds")
    requests = await capture_requests(include_inactive=args.include_inactive)
    if args.dry_run:
        rows = [request.metadata() for request in requests]
        passed: bool | None = None
    else:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout,
                                     trust_env=False, follow_redirects=False) as client:
            rows = [await replay(client, args.model, request) for request in requests]
        passed = all(row["passed"] for row in rows)
    report = {
        "schema": "forte.synthetic-exact-prompt-budget.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True, "accuracy_claim": False, "model": args.model,
        "inference_executed": not args.dry_run, "passed": passed,
        "case_count": len(rows),
        "passed_count": sum(row.get("passed", False) for row in rows) if not args.dry_run else None,
        "request_template_kwargs": {}, "diagnostic_instructions_appended": False,
        "length_finish_is_failure": True, "captured_from_analyzer": True,
        "source_sha256": {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for name, module in (("analyzer.py", A), ("prompts.py", P), ("vlm.py", V))},
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "inference_executed", "passed", "case_count", "passed_count")}))
    return 0 if passed is not False else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as error:
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        raise SystemExit(2) from None
