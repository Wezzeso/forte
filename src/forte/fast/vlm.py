"""Minimal async client for the local OpenAI-compatible vLLM server."""

from __future__ import annotations

import asyncio
import base64
import heapq
import io
import itertools
import json
import math
import re
import time
from typing import Any

import httpx
from jsonschema import validate
from PIL import Image


def jpeg_data_url(image: Image.Image, quality: int = 88, max_side: int | None = None) -> str:
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                             Image.BICUBIC)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite model JSON number")
    return number


def _reject_constant(value: str):
    raise ValueError("non-finite model JSON number")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate model JSON key")
        result[key] = value
    return result


def _strict_loads(value: str, *, reject_duplicates: bool = False):
    kwargs = {"object_pairs_hook": _unique_object} if reject_duplicates else {}
    return json.loads(value, parse_float=_finite_float, parse_constant=_reject_constant, **kwargs)


class _StrictProtocolError(ValueError):
    """A static strict-mode error; retrying must not hide a protocol violation."""


def _strict_response_content(data: Any) -> str:
    if not isinstance(data, dict):
        raise _StrictProtocolError("model_protocol_invalid_envelope")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise _StrictProtocolError("model_protocol_invalid_choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise _StrictProtocolError("model_protocol_incomplete_response")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _StrictProtocolError("model_protocol_invalid_message")
    if any(message.get(key) for key in ("reasoning", "reasoning_content")):
        raise _StrictProtocolError("model_protocol_reasoning_present")
    content = message.get("content")
    if not isinstance(content, str) or not content.lstrip().startswith("{"):
        raise _StrictProtocolError("model_protocol_content_not_plain_object")
    return content


def parse_json(text: str) -> Any:
    text = text.strip()
    match = _FENCE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        return _strict_loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return _strict_loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("model did not return JSON")


def close_truncated_object(text: str) -> str:
    """Close a JSON object whose generation was stopped right after a complete field."""
    body = text.strip()
    if body.startswith("{") and not body.endswith("}"):
        body = body.rstrip().rstrip(",").rstrip() + "}"
    return body


class PrioritySemaphore:
    """Counting semaphore that grants waiting slots by (priority, arrival).

    With equal priorities it is FIFO, like asyncio.Semaphore. A lower priority value
    (an earlier job) is served first, so concurrent jobs finish in arrival order.
    """

    def __init__(self, value: int):
        if value < 1:
            raise ValueError("semaphore value must be positive")
        self._value = value
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()

    async def acquire(self, priority: int = 0) -> None:
        if self._value > 0 and not any(not f.done() for _, _, f in self._waiters):
            self._value -= 1
            return
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (int(priority), next(self._seq), future))
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release()  # slot was granted just before cancellation
            raise

    def release(self) -> None:
        while self._waiters:
            _, _, future = heapq.heappop(self._waiters)
            if not future.done():
                future.set_result(None)
                return
        self._value += 1


class VLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 concurrency: int = 16, timeout: float = 120.0):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers,
                                         timeout=httpx.Timeout(timeout, connect=10.0))
        self._model = model
        self._sem = PrioritySemaphore(concurrency)
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ready(self) -> bool:
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def check_ready(self, *, deadline: float | None = None) -> None:
        budget = deadline - time.monotonic() if deadline is not None else 10.0
        if budget <= 0:
            raise TimeoutError("model_readiness_deadline_exceeded")
        if not await asyncio.wait_for(self.ready(), timeout=budget):
            raise RuntimeError("fast_model_not_ready")

    async def chat(self, content: list[dict[str, Any]], *, system: str | None = None,
                   max_tokens: int = 512, temperature: float = 0.0,
                   json_schema: dict[str, Any] | None = None, retries: int = 2,
                   stop: list[str] | None = None, priority: int = 0,
                   strict_response: bool = False, model_override: str | None = None) -> str:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        body: dict[str, Any] = {"model": self._model if model_override is None else model_override, "messages": messages,
                                "max_tokens": max_tokens, "temperature": temperature,
                                "seed": 0}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "out", "schema": json_schema}}
        if stop:
            body["stop"] = list(stop)
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                await self._sem.acquire(priority)
                try:
                    response = await self._client.post("/chat/completions", json=body)
                finally:
                    self._sem.release()
                response.raise_for_status()
                data = response.json()
                strict_content = _strict_response_content(data) if strict_response else None
                usage = data.get("usage") or {}
                self.calls += 1
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                return strict_content if strict_response else data["choices"][0]["message"]["content"] or ""
            except _StrictProtocolError:
                raise
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last = exc
                await asyncio.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"model request failed: {type(last).__name__}")

    async def chat_json(self, content: list[dict[str, Any]], *, truncated: bool = False,
                        validate_schema: dict[str, Any] | None = None,
                        strict_response: bool = False, **kwargs: Any) -> Any:
        if strict_response:
            kwargs["strict_response"] = True
        text = await self.chat(content, **kwargs)
        text = close_truncated_object(text) if truncated else text
        data = _strict_loads(text, reject_duplicates=True) if strict_response else parse_json(text)
        schema = validate_schema if validate_schema is not None else kwargs.get("json_schema")
        if schema is not None:
            validate(data, schema)
        return data


def image_part(image: Image.Image, **kwargs: Any) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": jpeg_data_url(image, **kwargs)}}


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}
