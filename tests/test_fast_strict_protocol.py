"""Strict model protocol is opt-in and rejects failures without exposing model text."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from forte.fast import analyzer as A
from forte.fast import prompts as P
from forte.fast.vlm import VLMClient


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
          "required": ["answer"], "additionalProperties": False}


def _envelope(content='{"answer":"no"}', finish="stop", **message):
    return {"choices": [{"finish_reason": finish, "message": {"content": content, **message}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}


async def _mock_client(reply, seen):
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    client = VLMClient("http://model/v1", "synthetic")
    await client._client.aclose()
    client._client = httpx.AsyncClient(base_url="http://model/v1", transport=httpx.MockTransport(handler))
    return client


def _query(reply, *, seen=None, **kwargs):
    seen = [] if seen is None else seen

    async def scenario():
        client = await _mock_client(reply, seen)
        try:
            return await client.chat_json([], json_schema=SCHEMA, **kwargs)
        finally:
            await client.aclose()

    return asyncio.run(scenario())


def test_strict_valid_json_does_not_change_server_request():
    baseline, strict = [], []
    assert _query(_envelope(), seen=baseline) == {"answer": "no"}
    assert _query(_envelope(), seen=strict, strict_response=True) == {"answer": "no"}
    assert baseline == strict
    assert "strict_response" not in strict[0]


def test_strict_intentional_concise_stop_is_valid():
    seen = []

    async def scenario():
        client = await _mock_client(_envelope('{"visible":true,"confidence":0.9,"t_sec":2.0, '), seen)
        try:
            return await client.chat_json([], json_schema=P.VERIFY_SCHEMA,
                validate_schema=P.VERIFY_DECISION_SCHEMA, stop=[P.VERIFY_STOP],
                truncated=True, strict_response=True)
        finally:
            await client.aclose()

    assert asyncio.run(scenario()) == {"visible": True, "confidence": 0.9, "t_sec": 2.0}
    assert len(seen) == 1 and seen[0]["stop"] == [P.VERIFY_STOP]


@pytest.mark.parametrize("finish", ["length", "tool_calls", "content_filter", None])
@pytest.mark.parametrize("truncated", [False, True])
def test_strict_nonstop_completion_fails_without_retry(finish, truncated):
    seen = []
    content = '{"answer":"no", ' if truncated else '{"answer":"no"}'
    with pytest.raises(ValueError, match="^model_protocol_incomplete_response$"):
        _query(_envelope(content, finish), seen=seen, strict_response=True, truncated=truncated)
    assert len(seen) == 1


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_strict_reasoning_fails_with_static_error_without_retry(field):
    seen = []
    with pytest.raises(ValueError, match="^model_protocol_reasoning_present$"):
        _query(_envelope(**{field: "synthetic private marker"}), seen=seen, strict_response=True)
    assert len(seen) == 1


@pytest.mark.parametrize("content", [
    '```json\n{"answer":"no"}\n```',
    '<think>synthetic</think>{"answer":"no"}',
    'Synthetic explanation: {"answer":"no"}',
    [{"type": "text", "text": '{"answer":"no"}'}],
    None,
])
def test_strict_nonplain_content_fails_without_retry(content):
    seen = []
    with pytest.raises(ValueError, match="^model_protocol_content_not_plain_object$"):
        _query(_envelope(content), seen=seen, strict_response=True)
    assert len(seen) == 1


@pytest.mark.parametrize("content", [
    '{"answer":"yes","answer":"no"}',
    '{"answer":"no","answer":"no"}',
    '{"answer":NaN}',
    '{"answer":Infinity}',
    '{"answer":1e999}',
    '{"answer":"no"} trailing synthetic text',
])
def test_strict_invalid_json_is_not_repaired_or_retried(content):
    seen = []
    with pytest.raises(ValueError):
        _query(_envelope(content), seen=seen, strict_response=True)
    assert len(seen) == 1


def test_strict_concise_missing_decision_is_rejected():
    from jsonschema import ValidationError

    async def scenario():
        client = await _mock_client(_envelope('{"visible":true, '), [])
        try:
            return await client.chat_json([], json_schema=P.VERIFY_SCHEMA,
                validate_schema=P.VERIFY_DECISION_SCHEMA, stop=[P.VERIFY_STOP],
                truncated=True, strict_response=True)
        finally:
            await client.aclose()

    with pytest.raises(ValidationError):
        asyncio.run(scenario())


def test_default_protocol_keeps_legacy_fence_reasoning_and_finish_behavior():
    assert A.DEFAULT_CONFIG["strict_model_protocol"] is False
    assert _query(_envelope('```json\n{"answer":"no"}\n```', "length",
                            reasoning_content="synthetic")) == {"answer": "no"}
    assert _query(_envelope('{"answer":"yes","answer":"no"}')) == {"answer": "no"}


def test_default_wrapper_adds_no_new_keyword_and_flag_is_uniform():
    class LegacyClient:
        # This deliberately rejects unknown keywords, as existing analyzer doubles do.
        async def chat_json(self, content, *, json_schema):
            return {"answer": "no"}

    class RecordingClient:
        def __init__(self):
            self.calls = []

        async def chat_json(self, *args, **kwargs):
            self.calls.append(kwargs)
            return {"answer": "no"}

    async def scenario():
        assert await A._CheckedClient(LegacyClient()).chat_json([], json_schema=SCHEMA) == {"answer": "no"}
        target = RecordingClient()
        checked = A._CheckedClient(target, priority=3, strict_model_protocol=True)
        await checked.chat_json([], json_schema=SCHEMA)
        await checked.chat_json([], json_schema=SCHEMA, strict_response=False)
        assert all(call["strict_response"] is True and call["priority"] == 3 for call in target.calls)

    asyncio.run(scenario())


def test_analyzer_flagged_swallowed_protocol_failure_cannot_return_clean(monkeypatch):
    async def states(client, *args, **kwargs):
        try:
            await client.chat_json([], json_schema=SCHEMA)
        except Exception:
            pass  # Emulate the state helper's unreadable/unknown fallback.
        return [{"t": 0.0, "cashier_at_desk": True}]

    monkeypatch.setattr(A, "extract_states", states)
    monkeypatch.setattr(A, "rule_candidates", lambda *args: ([], []))

    async def scenario(strict):
        client = await _mock_client(_envelope(finish="length"), [])
        try:
            return await A.analyze(client, SimpleNamespace(),
                {"use_crops": False, "strict_model_protocol": strict})
        finally:
            await client.aclose()

    assert asyncio.run(scenario(False)).events == []
    with pytest.raises(RuntimeError, match="^fast_model_observation_failed$"):
        asyncio.run(scenario(True))
