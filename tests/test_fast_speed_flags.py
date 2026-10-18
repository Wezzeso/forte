"""Speed flags must not change what the analyzer decides (no model, no GPU)."""

from __future__ import annotations

import asyncio
import json
import re

import httpx
from PIL import Image

from forte.fast import prompts as P
from forte.fast.analyzer import OBS_RU, analyze
from forte.fast.video import DecodedVideo, Frame
from forte.fast.vlm import PrioritySemaphore, VLMClient, close_truncated_object, parse_json

BOXES = {k: [100, 100, 400, 400] for k in P.LAYOUT_SCHEMA["required"]}


class FakeVLM:
    """Deterministic stand-in: the cashier leaves at t>=6 s and the laptop keeps working content."""

    def __init__(self, layout_delay: float = 0.0):
        self.calls: list[dict] = []
        self.layout_delay = layout_delay

    async def chat_json(self, content, *, system=None, max_tokens=0, json_schema=None, stop=None,
                        truncated=False, validate_schema=None, priority=None):
        text = " ".join(p["text"] for p in content if p.get("type") == "text")
        self.calls.append({"schema": json_schema, "stop": stop, "truncated": truncated, "priority": priority,
                           "validate_schema": validate_schema})
        await asyncio.sleep(0)
        if json_schema is P.LAYOUT_SCHEMA:
            await asyncio.sleep(self.layout_delay)
            return dict(BOXES)
        if json_schema is P.STATE_SCHEMA:
            t = float(re.search(r"t=([0-9.]+)s", text).group(1))
            here = t < 6.0
            return {"cashier_at_desk": here, "cashier_in_room": here, "client_at_desk": False, "other_people": 0,
                    "person_at_door": False, "cash_in_hands": False, "cash_bag": False, "id_card_shown": False,
                    "counting_machine": False, "detector": False, "hand_counting": False}
        if json_schema is P.LAPTOP_SCHEMA:
            return {"answer": "A"}
        if json_schema is P.DESK_SCHEMA:
            return {k: False for k in P.DESK_SCHEMA["required"]}
        if json_schema is P.SEAT_SCHEMA:
            return {"answer": "no"}
        if json_schema is P.VERIFY_SCHEMA:
            t = float(re.search(r"around t=([0-9.]+)s", text).group(1))
            out = {"visible": True, "confidence": 0.9, "t_sec": t}
            if not truncated:
                out["observation_ru"] = "Свободный текст модели."
            return out
        raise AssertionError("unexpected schema")


def _video(seconds: int = 14, fps: float = 2.0) -> DecodedVideo:
    frames = [Frame(t=i / fps, image=Image.new("RGB", (64, 96), (120, 120, 120))) for i in range(int(seconds * fps) + 1)]
    return DecodedVideo(duration=float(seconds), fps=30.0, width=64, height=96, frames=frames)


LIVE = {"require_corroboration": True, "use_verify": True}


def _run(cfg, client=None, **kwargs):
    client = client or FakeVLM()
    return asyncio.run(analyze(client, _video(), {**LIVE, **cfg}, **kwargs)), client


def _decisions(analysis):
    return [(e.type_id, round(e.t, 3)) for e in analysis.events]


def test_parallel_layout_matches_sequential():
    base, _ = _run({})
    fast, _ = _run({"parallel_layout": True}, FakeVLM(layout_delay=0.01))
    assert _decisions(base) == _decisions(fast) and _decisions(base)
    strip = lambda states: [{k: v for k, v in s.items()} for s in states]  # noqa: E731
    assert strip(base.states) == strip(fast.states)
    assert base.layout == fast.layout
    assert "layout" in fast.timings


def test_parallel_layout_sends_full_frames_before_layout_returns():
    client = FakeVLM(layout_delay=0.05)
    _run({"parallel_layout": True}, client)
    schemas = [c["schema"] for c in client.calls]
    first_crop = min(i for i, s in enumerate(schemas) if s in (P.LAPTOP_SCHEMA, P.DESK_SCHEMA, P.SEAT_SCHEMA))
    states_before_crops = sum(1 for s in schemas[:first_crop] if s is P.STATE_SCHEMA)
    assert states_before_crops == len(_video().frames[::2])  # every 1 fps state frame already sent


def test_verify_concise_keeps_decisions_and_uses_template_text():
    full, _ = _run({})
    concise, client = _run({"verify_concise": True})
    assert _decisions(full) == _decisions(concise)
    verify_calls = [c for c in client.calls if c["schema"] is P.VERIFY_SCHEMA]
    assert verify_calls and all(c["stop"] == [P.VERIFY_STOP] and c["truncated"] for c in verify_calls)
    assert all(c["validate_schema"] is P.VERIFY_DECISION_SCHEMA for c in verify_calls)
    assert {e.observation_ru for e in concise.events} == {OBS_RU["WORKSTATION_UNLOCKED"]}
    assert all("observation_ru" not in v for v in concise.verifications)


def test_job_priority_only_when_enabled():
    _, off = _run({}, priority=7)
    assert all(c["priority"] is None for c in off.calls)
    _, on = _run({"job_priority": True}, priority=7)
    assert all(c["priority"] == 7 for c in on.calls)


def test_priority_semaphore_serves_lowest_priority_then_fifo():
    async def scenario():
        sem = PrioritySemaphore(1)
        await sem.acquire(0)
        order = []

        async def waiter(tag, prio):
            await sem.acquire(prio)
            order.append(tag)
            await asyncio.sleep(0)
            sem.release()

        tasks = [asyncio.create_task(waiter(tag, prio)) for tag, prio in
                 (("late-a", 5), ("early", 1), ("late-b", 5), ("mid", 3))]
        await asyncio.sleep(0)
        sem.release()
        await asyncio.gather(*tasks)
        return order

    assert asyncio.run(scenario()) == ["early", "mid", "late-a", "late-b"]


def test_priority_semaphore_skips_cancelled_waiters():
    async def scenario():
        sem = PrioritySemaphore(1)
        await sem.acquire()
        cancelled = asyncio.create_task(sem.acquire(0))
        kept = asyncio.create_task(sem.acquire(1))
        await asyncio.sleep(0)
        cancelled.cancel()
        await asyncio.sleep(0)
        sem.release()
        await asyncio.wait_for(kept, 1.0)
        sem.release()
        await asyncio.wait_for(sem.acquire(), 1.0)  # capacity fully restored
        return True

    assert asyncio.run(scenario())


def test_truncated_decision_object_parses():
    text = '{"visible": true, "confidence": 0.85, "t_sec": 44.2, '
    assert parse_json(close_truncated_object(text)) == {"visible": True, "confidence": 0.85, "t_sec": 44.2}
    whole = '{"visible": false, "confidence": 0.1, "t_sec": 3}'
    assert close_truncated_object(whole) == whole


def test_client_sends_stop_and_same_grammar():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.update(body)
        reply = '{"visible": true, "confidence": 0.9, "t_sec": 12.0, '
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 12}})

    async def scenario():
        client = VLMClient("http://model/v1", "q30")
        await client._client.aclose()
        client._client = httpx.AsyncClient(base_url="http://model/v1", transport=httpx.MockTransport(handler))
        try:
            return await client.chat_json([{"type": "text", "text": "q"}], json_schema=P.VERIFY_SCHEMA,
                                          stop=[P.VERIFY_STOP], truncated=True,
                                          validate_schema=P.VERIFY_DECISION_SCHEMA, priority=3)
        finally:
            await client.aclose()

    data = asyncio.run(scenario())
    assert data == {"visible": True, "confidence": 0.9, "t_sec": 12.0}
    assert seen["stop"] == [P.VERIFY_STOP]
    assert seen["response_format"]["json_schema"]["schema"] == P.VERIFY_SCHEMA
    assert "priority" not in seen  # scheduling stays client-side; the server flag is unchanged


def _scene_video(seconds: int = 14, fps: float = 2.0, leave_at: float = 6.0) -> DecodedVideo:
    from PIL import ImageDraw

    frames = []
    for i in range(int(seconds * fps) + 1):
        t = i / fps
        img = Image.new("RGB", (64, 96), (120, 120, 120))
        if t < leave_at:  # a dark "person" region that disappears when the cashier leaves
            ImageDraw.Draw(img).rectangle([10, 20, 40, 70], fill=(30, 30, 30))
        frames.append(Frame(t=t, image=img))
    return DecodedVideo(duration=float(seconds), fps=30.0, width=64, height=96, frames=frames)


def test_gating_plan_reuses_static_frames_and_refreshes():
    from forte.fast.analyzer import gating_plan

    picks = _scene_video().frames[::2]  # 1 fps, t = 0..14
    owners = gating_plan(picks, {"gating_refresh_s": 5.0})
    assert owners[:6] == [0, 0, 0, 0, 0, 5]          # refresh at 5 s
    assert owners[6] == 6                             # pixels changed at 6 s
    assert owners[7:12] == [6, 6, 6, 6, 11]           # static again, refresh at 11 s


def test_change_gating_keeps_events_and_asks_fewer_frames():
    base = asyncio.run(analyze(FakeVLM(), _scene_video(), dict(LIVE)))
    client = FakeVLM()
    gated = asyncio.run(analyze(client, _scene_video(), {**LIVE, "change_gating": True}))
    assert _decisions(base) == _decisions(gated) and _decisions(base)
    assert [s["t"] for s in base.states] == [s["t"] for s in gated.states]
    asked = sum(1 for c in client.calls if c["schema"] is P.STATE_SCHEMA)
    assert asked < len(base.states)
    assert gated.audit["gating"]["frames"] == len(base.states)


def test_layout_consensus_takes_majority_median():
    from forte.fast.analyzer import locate_layout

    class LayoutVLM:
        def __init__(self):
            self.n = 0

        async def chat_json(self, content, **kwargs):
            self.n += 1
            shift = {1: 0, 2: 10, 3: 400}[self.n]           # third frame is an outlier
            out = {k: None for k in P.LAYOUT_SCHEMA["required"]}
            out["desk"] = [100 + shift, 100, 500 + shift, 400]
            if self.n == 1:
                out["safe"] = [10, 10, 50, 50]               # seen once only -> dropped
            return out

    boxes = asyncio.run(locate_layout(LayoutVLM(), _video(), {"layout_consensus": 3}))
    assert boxes == {"desk": [110.0, 100.0, 510.0, 400.0]}
