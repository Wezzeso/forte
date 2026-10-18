"""An opt-in local desk adapter must not alter other requests or shared model state."""

from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest
from PIL import Image

from forte.fast import prompts as P
from forte.fast.analyzer import DEFAULT_CONFIG, analyze
from forte.fast.video import DecodedVideo, Frame
from forte.fast.vlm import VLMClient


class RecordingVLM:
    def __init__(self, *, adapter_fault=None, base_fault=False):
        self.calls = []
        self.adapter_fault = adapter_fault
        self.base_fault = base_fault

    async def chat_json(self, content, **kwargs):
        self.calls.append(dict(kwargs))
        schema = kwargs["json_schema"]
        text = " ".join(part["text"] for part in content if part.get("type") == "text")
        if schema is P.LAYOUT_SCHEMA:
            return {key: [100, 100, 400, 400] for key in schema["required"]}
        if schema is P.STATE_SCHEMA:
            t = float(re.search(r"t=([0-9.]+)s", text).group(1))
            return {"cashier_at_desk": t < 5, "cashier_in_room": t < 5,
                    "client_at_desk": False, "other_people": 0, "person_at_door": False,
                    "cash_in_hands": False, "cash_bag": False, "id_card_shown": False,
                    "counting_machine": False, "detector": False, "hand_counting": False}
        if schema is P.LAPTOP_SCHEMA:
            return {"answer": "A"}
        if schema is P.DESK_SCHEMA:
            if kwargs.get("model_override"):
                if self.adapter_fault == "request":
                    raise ValueError("synthetic_adapter_failure")
                if self.adapter_fault == "nonboolean":
                    return {"cash": "yes", "id_or_document": True}
                return {"cash": True, "id_or_document": True, "cash_bag": False,
                        "coins_or_valuables": False, "drink": False, "coat_or_handbag": False}
            if self.base_fault:
                raise ValueError("synthetic_base_failure")
            return {"cash": False, "id_or_document": False, "cash_bag": True,
                    "coins_or_valuables": True, "drink": True, "coat_or_handbag": True}
        if schema is P.SEAT_SCHEMA:
            return {"answer": "no"}
        if schema is P.VERIFY_SCHEMA:
            t = float(re.search(r"around t=([0-9.]+)s", text).group(1))
            return {"visible": True, "confidence": 0.9, "t_sec": t, "observation_ru": ""}
        raise AssertionError("unexpected synthetic schema")


def _run(config, client=None):
    frames = [Frame(float(t), Image.new("RGB", (48, 64), "grey")) for t in range(11)]
    video = DecodedVideo(duration=10.0, fps=30.0, width=48, height=64, frames=frames)
    client = client or RecordingVLM()
    try:
        result = asyncio.run(analyze(client, video, {"use_verify": True, **config}))
    finally:
        for frame in frames:
            frame.image.close()
    return result, client.calls


def test_default_desk_adapter_adds_no_request_keyword():
    assert DEFAULT_CONFIG["desk_adapter_model"] == ""
    _, default = _run({})
    _, explicit_off = _run({"desk_adapter_model": ""})
    assert default == explicit_off
    assert all("model_override" not in call for call in default)


@pytest.mark.parametrize("gated", [False, True])
def test_only_desk_crops_route_to_adapter_for_both_extraction_paths(gated):
    config = {"change_gating": gated, "parallel_layout": True}
    baseline, base_calls = _run(config)
    routed, calls = _run({**config, "desk_adapter_model": "q30desk"})
    for base_state, adapted_state in zip(baseline.states, routed.states):
        expected = {**base_state, "cash_on_desk": True, "id_card_shown": True,
                    "desk": {**base_state["desk"], "cash": True, "id_or_document": True}}
        assert adapted_state == expected
        assert all(adapted_state["desk"][key] == base_state["desk"][key] is True for key in (
            "cash_bag", "coins_or_valuables", "drink", "coat_or_handbag"))
    assert [(event.type_id, event.t) for event in routed.events] == [
        (event.type_id, event.t) for event in baseline.events]
    assert routed.events  # Verification was exercised as well as the perception calls.
    schemas = [call["json_schema"] for call in calls]
    for required in (P.LAYOUT_SCHEMA, P.STATE_SCHEMA, P.LAPTOP_SCHEMA, P.SEAT_SCHEMA, P.VERIFY_SCHEMA):
        assert any(schema is required for schema in schemas)
    desk_calls = [call for call in calls if call["json_schema"] is P.DESK_SCHEMA]
    baseline_desk = [call for call in base_calls if call["json_schema"] is P.DESK_SCHEMA]
    assert len(desk_calls) == 2 * len(baseline_desk) > 0
    adapter_calls = [call for call in desk_calls if call.get("model_override") == "q30desk"]
    assert len(adapter_calls) == len(baseline_desk)
    assert all("model_override" not in call for call in calls if call["json_schema"] is not P.DESK_SCHEMA)


def test_per_request_override_does_not_mutate_concurrent_base_requests():
    async def scenario():
        seen = []
        together = asyncio.Event()

        async def handler(request):
            body = json.loads(request.content)
            seen.append(body)
            if len(seen) == 2:
                together.set()
            await asyncio.wait_for(together.wait(), timeout=1.0)
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer":"no"}'}}]})

        client = VLMClient("http://model/v1", "q30test")
        await client._client.aclose()
        client._client = httpx.AsyncClient(base_url="http://model/v1", transport=httpx.MockTransport(handler))
        schema = {"type": "object", "properties": {"answer": {"enum": ["no"]}},
                  "required": ["answer"], "additionalProperties": False}
        try:
            replies = await asyncio.gather(
                client.chat_json([], json_schema=schema, model_override="q30desk"),
                client.chat_json([], json_schema=schema))
            assert replies == [{"answer": "no"}, {"answer": "no"}]
            assert client._model == "q30test"
            assert sorted(body["model"] for body in seen) == ["q30desk", "q30test"]
            assert all("model_override" not in body for body in seen)
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_base_and_adapter_desk_queries_share_crop_and_run_in_parallel():
    class ParallelVLM(RecordingVLM):
        def __init__(self):
            super().__init__()
            self.crop_requests = {}

        async def chat_json(self, content, **kwargs):
            if kwargs["json_schema"] is P.DESK_SCHEMA:
                row = self.crop_requests.setdefault(id(content), {"ready": asyncio.Event(), "models": []})
                row["models"].append(kwargs.get("model_override", "base"))
                if len(row["models"]) == 2:
                    row["ready"].set()
                await asyncio.wait_for(row["ready"].wait(), timeout=1.0)
            return await super().chat_json(content, **kwargs)

    client = ParallelVLM()
    _run({"desk_adapter_model": "q30desk"}, client)
    assert client.crop_requests
    assert all(sorted(row["models"]) == ["base", "q30desk"] for row in client.crop_requests.values())


def test_invalid_adapter_config_fails_instead_of_silently_using_base():
    with pytest.raises(ValueError, match="^desk_adapter_model_invalid$"):
        _run({"desk_adapter_model": ["q30desk"]})


@pytest.mark.parametrize("fault", ["request", "nonboolean", "base_request"])
def test_either_desk_request_or_invalid_supported_field_fails_video(fault):
    client = RecordingVLM(adapter_fault=fault, base_fault=fault == "base_request")
    with pytest.raises(ValueError):
        _run({"desk_adapter_model": "q30desk"}, client)
