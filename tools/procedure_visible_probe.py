"""Dense, visibility-aware procedure diagnostics; no candidates or service flags.

Visual principle declared before experiment: a procedural omission is testable
only across an observed transaction whose decisive region and hands remain
visible. Dense original-frame action observations must establish its boundaries;
absence needs >=90% sampled visibility and two agreeing closed phrasings.

All cached videos receive the same state-based transaction proposal policy. No
labels or label timestamps are accepted. Every retained original native frame at
5 fps is inspected; images and native crops go only to a loopback local model.
Two phrasings inspect LOCAL actions in batches of <=8 frames, with opposite action
polarity. Separate paired questions inspect transaction boundaries. The result is
typed perception, not a verdict, candidate, legal assertion, or enablement gate.

Coverage is an explicitly named SAMPLING PROXY, not continuous visibility. It
cannot exclude an action between 5-fps samples or establish correct device use.
The output therefore never qualifies a violation type, even when the proxy is
>=90%. Empty proposals, unreadable regions, open boundaries and model failures
stay unassessed. Cached booleans select windows; false never proves invisibility.

    python tools/procedure_visible_probe.py --run run.json --videos /videos \
        --model gemma4 --out /runs/procedure-visible.json
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.parse import urlparse


NAME_RE = re.compile(r"X-[1-9][0-9]{0,2}\Z")
RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
FPS = 5.0
CONTEXT_SECONDS = 2.0
MAX_BATCH_FRAMES = 8
VISIBILITY_THRESHOLD = .90
REGIONS = ("desk", "detector", "counting_machine")
VISIBILITY = ("clear", "occluded", "not_visible", "too_small")
YESNO = ("yes", "no", "unclear")
ACTIONS = {"ad": "banknote_to_detector", "ai": "card_to_cashier",
           "ar": "cash_recount", "aa": "cash_accepted", "as": "cash_stored"}
VIS_KEYS = {"d": "detector_visibility", "z": "desk_visibility",
            "m": "counting_machine_visibility", "h": "cashier_hands_visibility",
            "v": "visitor_hands_visibility"}
TYPES = ("NO_AUTHENTICITY_CHECK", "NO_CLIENT_ID", "NO_RECOUNT")
ERRORS = frozenset("AssertionError AttributeError ConnectionError FileNotFoundError IndexError JSONDecodeError KeyError MemoryError OSError OverflowError PermissionError RuntimeError TimeoutError TypeError UnicodeDecodeError ValueError CancelledError ConnectError ConnectTimeout HTTPStatusError NetworkError PoolTimeout ReadError ReadTimeout RemoteProtocolError TransportError ValidationError OtherError".split())


def enum(values):
    return {"type": "string", "enum": list(values)}


def frame_schema(count: int) -> dict:
    fields = {"i": {"type": "integer", "minimum": 0, "maximum": count - 1},
              **{key: enum(VISIBILITY) for key in ("d", "z", "m", "h")},
              "v": enum((*VISIBILITY, "absent")),
              **{key: enum(YESNO) for key in ACTIONS}}
    return {"type": "object", "properties": {"frames": {"type": "array",
            "minItems": count, "maxItems": count, "items": {"type": "object",
            "properties": fields, "required": list(fields), "additionalProperties": False}}},
            "required": ["frames"], "additionalProperties": False}


BOUNDARY_SCHEMA = {"type": "object", "properties": {
    "kind": enum(("cash_acceptance", "cash_issue", "bag_to_storage", "no_cash_transaction", "other", "unclear")),
    "start": enum(("before_transaction", "already_in_progress", "no_cash_transaction", "unclear")),
    "end": enum(("transaction_completed", "still_in_progress", "no_cash_transaction", "unclear")),
    "whole_transaction_observed": enum(YESNO)},
    "required": ["kind", "start", "end", "whole_transaction_observed"], "additionalProperties": False}

LOCAL_ACTION_PROMPT = (
    "Inspect numbered ORIGINAL frames and three native, unresized ROI crops per frame. "
    "The crops follow each full frame in this order: desk, small authenticity detector, "
    "large counting machine. Missing crops are explicitly marked; never infer an absent device. "
    "Return one item per frame in index order. i=index. d,z,m=visibility of detector, "
    "handover desk zone, counting machine. clear means the actual operative surface is "
    "resolved and unoccluded; too_small means pixels cannot resolve contact or small objects. "
    "h=cashier hands; clear requires both operative hands resolved. v=visitor hands; "
    "absent only if no visitor is present. Do not identify people. "
    "Actions are LOCAL visible events from this frame until the next numbered frame; "
    "for the last frame use this frame only. ad=banknotes physically placed on/into/under "
    "the SMALL detector; ai=an identifiable rigid ID card/passport, distinct from banknotes, "
    "passed client-to-cashier or inspected by cashier; ar=banknotes visibly counted by hand "
    "or moving through counting machine; aa=banknotes passed visitor-to-cashier; "
    "as=received cash visibly put into safe/drawer. A hand approaching, paper, device "
    "presence, or a customary procedure is not evidence of an action. Use unclear when "
    "visibility, pixels or temporal gaps cannot decide. "
)
ACTION_PHRASINGS = (
    "For ad,ai,ar,aa,as answer yes only if that exact action is visibly occurring in "
    "the local interval; no only if the relevant area/hands are clear and that action "
    "is visibly absent in this LOCAL interval; otherwise unclear. JSON only.",
    "For ad,ai,ar,aa,as answer the OPPOSITE question: is that exact action absent in "
    "this LOCAL interval? yes only for a clear view with the action absent; no when "
    "the action is directly visible; otherwise unclear. This polarity applies only "
    "to action keys; visibility labels retain their ordinary meaning. JSON only.",
)
BOUNDARY_PHRASINGS = (
    "Inspect the original beginning/end frames and the dense typed local-action timeline. "
    "Is one cash transaction visibly covered from BEFORE cash/bag acceptance or issue "
    "starts until AFTER that transaction finishes? start=before_transaction only if "
    "the transaction has visibly not started in the beginning images; end=transaction_completed "
    "only if the received cash is put away or the issue/handover visibly finishes in the end "
    "images. whole_transaction_observed=yes requires both boundaries plus a coherent "
    "timeline of this same transaction. Use unclear for missing/ambiguous boundaries.",
    "Independently test whether the footage cuts INTO or OUT OF a transaction. "
    "Use already_in_progress if initial cash/bag handling precedes the shown start, "
    "and still_in_progress if cash processing continues beyond the shown end. "
    "Only a clearly earlier pre-transaction state and a later completed state permit "
    "whole_transaction_observed=yes. A visitor sitting down or leaving alone does "
    "not prove cash acceptance/issue boundaries. Base kind on actual visible cash flow.",
)


def error_code(exc: BaseException) -> str:
    value = type(exc).__name__
    return value if value in ERRORS else "OtherError"


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 86400


def local_base_url(value: str) -> str:
    parsed = urlparse(value)
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        local = False
    if parsed.scheme != "http" or not local or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("loopback_url_required")
    return value.rstrip("/")


def box_valid(box: Any) -> bool:
    return (isinstance(box, list) and len(box) == 4 and all(number(v) and v <= 1000 for v in box)
            and box[0] < box[2] and box[1] < box[3])


def crop_box(size: tuple[int, int], box: list[float]) -> tuple[int, int, int, int]:
    if not box_valid(box):
        raise ValueError("invalid_box")
    w, h = size
    return max(0, math.floor(box[0] * w / 1000)), max(0, math.floor(box[1] * h / 1000)), min(w, math.ceil(box[2] * w / 1000)), min(h, math.ceil(box[3] * h / 1000))


def cached_rows(value: Any, run_name: str | None = None) -> list[dict]:
    if run_name is not None:
        if not RUN_RE.fullmatch(run_name) or not isinstance(value, dict):
            raise ValueError("invalid_run_selection")
        value = value.get("runs", {}).get(run_name)
    raw = value.get("results", value.get("videos")) if isinstance(value, dict) else None
    if not isinstance(raw, list) or len(raw) != 40:
        raise ValueError("expected_40_rows")
    out, seen = [], set()
    for row in raw:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not NAME_RE.fullmatch(row["name"]) or row["name"] in seen:
            raise ValueError("invalid_cached_inventory")
        seen.add(row["name"])
        # Do not carry labels, errors, evidence, descriptions, or verdicts forward.
        states = row.get("states")
        if not isinstance(states, list) or len(states) > 1000:
            raise ValueError("invalid_cached_states")
        fields = ("client_at_desk", "client_seat", "cash_in_hands", "cash_bag", "counting_machine", "hand_counting", "detector")
        projected = []
        for state in states:
            if not isinstance(state, dict) or not number(state.get("t")):
                raise ValueError("invalid_state_timestamp")
            item = {"t": state["t"], **{k: state[k] for k in fields if type(state.get(k)) is bool}}
            desk = state.get("desk")
            if isinstance(desk, dict):
                item["desk"] = {k: desk[k] for k in ("cash", "cash_bag") if type(desk.get(k)) is bool}
            projected.append(item)
        if any(a["t"] >= b["t"] for a, b in zip(projected, projected[1:])):
            raise ValueError("state_order_invalid")
        layout = row.get("layout") if isinstance(row.get("layout"), dict) else {}
        out.append({"name": row["name"], "source_processed": row.get("processed") is True and row.get("status") not in ("failed", "processing"),
                    "states": projected, "layout": {key: layout[key] for key in REGIONS if box_valid(layout.get(key))}})
    return sorted(out, key=lambda r: r["name"])


def transaction_windows(states: list[dict], duration: float) -> list[dict]:
    """High-recall union of visitor/cash/bag/counting signals, with fixed context.

    No maximum-length clipping: a long transaction is processed in full or fails
    an explicit run budget. Signal gaps <=2 s are merged before adding 2 s context.
    """
    active = []
    for state in states:
        desk = state.get("desk", {})
        yes = any(state.get(k) is True for k in ("client_at_desk", "client_seat", "cash_in_hands", "cash_bag", "counting_machine", "hand_counting", "detector"))
        if yes or any(desk.get(k) is True for k in ("cash", "cash_bag")):
            active.append(state["t"])
    spans = []
    for stamp in active:
        if not spans or stamp - spans[-1][1] > CONTEXT_SECONDS:
            spans.append([stamp, stamp])
        else:
            spans[-1][1] = stamp
    expanded = []
    for a, b in spans:
        lo, hi = max(0.0, a - CONTEXT_SECONDS), min(duration, b + CONTEXT_SECONDS)
        if expanded and lo <= expanded[-1][1]:
            expanded[-1][1] = max(expanded[-1][1], hi)
        elif hi > lo:
            expanded.append([lo, hi])
    return [{"start": round(a, 6), "end": round(b, 6)} for a, b in expanded]


def consensus(a: str, b: str, *, opposite: bool = False) -> str:
    if opposite:
        b = {"yes": "no", "no": "yes", "unclear": "unclear"}[b]
    return a if a == b and a != "unclear" else "unclear"


def merge_frame_answers(a: dict, b: dict, times: list[float]) -> list[dict]:
    if [r["i"] for r in a["frames"]] != list(range(len(times))) or [r["i"] for r in b["frames"]] != list(range(len(times))):
        raise ValueError("frame_alignment_failed")
    result = []
    for stamp, ra, rb in zip(times, a["frames"], b["frames"]):
        vis = {name: ra[key] if ra[key] == rb[key] else "unclear" for key, name in VIS_KEYS.items()}
        actions = {name: consensus(ra[key], rb[key], opposite=True) for key, name in ACTIONS.items()}
        result.append({"t": stamp, "visibility": vis, "actions": actions,
                       "paired_action_disagreements": sum(ra[key] != {"yes": "no", "no": "yes", "unclear": "unclear"}[rb[key]] for key in ACTIONS)})
    return result


def sampled_coverage(observations: list[dict], start: float, end: float, required: tuple[str, ...]) -> dict:
    """Time-weight nearest-sample cells; report the interpolation assumption."""
    if not observations or end <= start:
        return {"visible_fraction_proxy": None, "sampled_frames": 0, "max_sample_gap_s": None, "meets_90pct_sampling_proxy": False}
    times = [r["t"] for r in observations]
    if any(a >= b for a, b in zip(times, times[1:])) or times[0] < start - 1e-6 or times[-1] > end + 1e-6:
        raise ValueError("invalid_observation_order")
    visible = 0.0
    for i, row in enumerate(observations):
        lo = start if i == 0 else (times[i - 1] + times[i]) / 2
        hi = end if i + 1 == len(times) else (times[i] + times[i + 1]) / 2
        if all(row["visibility"].get(key) == "clear" or
               (key == "visitor_hands_visibility" and row["visibility"].get(key) == "absent")
               for key in required):
            visible += hi - lo
    gaps = [times[0] - start, end - times[-1], *[b - a for a, b in zip(times, times[1:])]]
    max_gap = max(gaps)
    ratio = visible / (end - start)
    # 5-fps retention admits one source-frame timestamp of rounding slack.
    dense = max_gap <= .25
    return {"schema": "forte.procedure-sampled-visibility.v1", "visible_fraction_proxy": round(ratio, 6),
            "sampled_frames": len(times), "max_sample_gap_s": round(max_gap, 6),
            "meets_90pct_sampling_proxy": dense and ratio >= VISIBILITY_THRESHOLD,
            "continuous_visibility_proven": False, "interpolation": "nearest_sample_cells; not_observed_between_samples"}


def window_checks(observations: list[dict], boundaries: list[dict], start: float, end: float) -> dict:
    paired = len(boundaries) == 2 and boundaries[0] == boundaries[1]
    whole = paired and boundaries[0]["whole_transaction_observed"] == "yes" and boundaries[0]["start"] == "before_transaction" and boundaries[0]["end"] == "transaction_completed"
    kind = boundaries[0]["kind"] if paired else "unclear"
    specs = {"NO_AUTHENTICITY_CHECK": (("detector_visibility", "cashier_hands_visibility"), "banknote_to_detector", "cash_acceptance"),
             "NO_CLIENT_ID": (("desk_visibility", "cashier_hands_visibility", "visitor_hands_visibility"), "card_to_cashier", "cash_acceptance"),
             "NO_RECOUNT": (("desk_visibility", "counting_machine_visibility", "cashier_hands_visibility"), "cash_recount", "bag_to_storage")}
    out = {}
    for type_id, (required, action, applicable_kind) in specs.items():
        coverage = sampled_coverage(observations, start, end, required)
        def visible(row):
            return all(row["visibility"].get(key) == "clear" or
                       (key == "visitor_hands_visibility" and row["visibility"].get(key) == "absent")
                       for key in required)
        # A nominal action answer cannot outrank a disagreement about whether
        # its decisive region/hands were actually resolved in that same frame.
        counts = Counter(row["actions"][action] if visible(row) else "unclear" for row in observations)
        known_fraction = (counts["yes"] + counts["no"]) / len(observations) if observations else 0
        required_flow = "cash_stored" if type_id == "NO_RECOUNT" else "cash_accepted"
        flow_observed = any(row["actions"][required_flow] == "yes" for row in observations)
        decisive = whole and kind == applicable_kind and flow_observed and coverage["meets_90pct_sampling_proxy"] and known_fraction >= VISIBILITY_THRESHOLD
        status = "insufficient_coverage"
        if counts["yes"]:
            status = "step_observed"
        elif decisive:
            status = "no_step_seen_with_sampling_proxy"
        out[type_id] = {"status": status, "action": action,
                        "action_frames": {key: counts[key] for key in YESNO},
                        "action_known_fraction": round(known_fraction, 6),
                        "transaction_kind": kind, "paired_boundaries_agree": paired,
                        "required_cash_flow_observed": flow_observed,
                        "whole_transaction_observed": bool(whole), "visibility": coverage,
                        "diagnostic_only": True, "type_enablement_qualified": False}
    return out


class LocalClient:
    def __init__(self, url: str, model: str, concurrency: int, deadline: float):
        import httpx
        self.http = httpx.AsyncClient(base_url=local_base_url(url), trust_env=False, follow_redirects=False)
        self.model, self.sem, self.deadline = model, asyncio.Semaphore(concurrency), deadline
        self.calls = self.attempted_calls = self.prompt_tokens = self.completion_tokens = 0

    async def ask(self, content: list[dict], schema: dict, max_tokens: int) -> dict:
        from jsonschema import validate
        async with self.sem:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("diagnostic_deadline")
            self.attempted_calls += 1
            response = await self.http.post("chat/completions", timeout=min(120, remaining), json={
                "model": self.model, "temperature": 0, "seed": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": "Inspect original visual pixels only. Never identify people or infer invisible facts. Return strict schema JSON only, no reasoning."},
                             {"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
                "chat_template_kwargs": {"enable_thinking": False}})
        response.raise_for_status()
        payload = response.json()
        self.calls += 1
        usage = payload.get("usage", {})
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        choice = payload["choices"][0]
        message = choice["message"]
        if choice.get("finish_reason") != "stop" or any(message.get(k) for k in ("reasoning", "reasoning_content")):
            raise ValueError("reasoning_or_truncation")
        def reject_constant(value):
            raise ValueError("nonfinite_json")
        answer = json.loads(message["content"], parse_constant=reject_constant)
        validate(answer, schema)
        return answer


async def paired_asks(client: LocalClient, content: list[dict], prompts: tuple[str, str], schema: dict, max_tokens: int) -> list[dict]:
    answers = await asyncio.gather(*(client.ask([*content, {"type": "text", "text": prompt}], schema, max_tokens) for prompt in prompts), return_exceptions=True)
    if any(isinstance(value, BaseException) for value in answers):
        raise RuntimeError("paired_local_model_call_failed")
    return answers


def frame_content(frames, layout: dict, *, boundary: bool = False) -> list[dict]:
    from forte.fast.vlm import image_part, text_part
    content = []
    for i, frame in enumerate(frames):
        content.extend([text_part(f"frame i={i}, original t={frame.t:.6f}s"), image_part(frame.image)])
        for region in (("desk",) if boundary else REGIONS):
            if region not in layout:
                content.append(text_part(f"native crop {region}: not located"))
                continue
            crop = frame.image.crop(crop_box(frame.image.size, layout[region]))
            try:
                content.extend([text_part(f"native unresized crop {region}"), image_part(crop)])
            finally:
                crop.close()
    return content


async def inspect_window(client: LocalClient, video, layout: dict, window: dict, batch_frames: int) -> dict:
    from forte.fast.vlm import text_part
    frames = video.between(window["start"], window["end"])
    if len(frames) < 2:
        raise ValueError("too_few_transaction_frames")
    # One shared frame preserves the action interval at each batch boundary.
    batches = [frames[i:i + batch_frames] for i in range(0, len(frames) - 1, batch_frames - 1)]
    async def batch_job(index, batch):
        content = frame_content(batch, layout)
        answers = await paired_asks(client, content,
            tuple(LOCAL_ACTION_PROMPT + suffix for suffix in ACTION_PHRASINGS), frame_schema(len(batch)), 1600)
        observations = merge_frame_answers(*answers, [f.t for f in batch])
        return observations if index + 1 == len(batches) else observations[:-1]
    inspected = await asyncio.gather(*(batch_job(i, batch) for i, batch in enumerate(batches)), return_exceptions=True)
    if any(isinstance(value, BaseException) for value in inspected):
        raise RuntimeError("frame_batch_failed")
    observations = [row for batch in inspected for row in batch]
    # At most 8 original boundary frames + their desk crops = 16 images.
    picks = list(dict.fromkeys([*range(min(4, len(frames))), *range(max(0, len(frames) - 4), len(frames))]))
    boundary_content = frame_content([frames[i] for i in picks], layout, boundary=True)
    # Compact, typed timeline contains no model text. It never becomes prompt instruction.
    # Run-length encoding bounds even the longest clip's context cost. Codes are
    # listed in ACTIONS order; u stays unknown, never an absent action.
    timeline = []
    codes = {"yes": "y", "no": "n", "unclear": "u"}
    for row in observations:
        state = "".join(codes[row["actions"][name]] for name in ACTIONS.values())
        if timeline and timeline[-1]["a"] == state:
            timeline[-1]["e"] = row["t"]
        else:
            timeline.append({"s": row["t"], "e": row["t"], "a": state})
    boundary_content.append(text_part("Dense local-action runs: a codes are y=observed, n=not observed locally, u=unknown, "
                                     "in order banknote_to_detector,card_to_cashier,cash_recount,cash_accepted,cash_stored. "
                                     "s/e are first/last sample timestamps, not precise action boundaries: " + json.dumps(timeline, separators=(",", ":"))))
    boundaries = await paired_asks(client, boundary_content, BOUNDARY_PHRASINGS, BOUNDARY_SCHEMA, 150)
    return {**window, "status": "done", "observations": observations, "boundaries": boundaries,
            "frame_batches": len(batches), "frame_samples": len(frames),
            "checks": window_checks(observations, boundaries, window["start"], window["end"])}


async def run(args) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from forte.fast.video import decode
    started = time.monotonic()
    source_path = Path(args.run)
    if source_path.stat().st_size > 100 * 1024**2:
        raise ValueError("cached_report_too_large")
    rows = cached_rows(json.loads(source_path.read_text(encoding="utf-8")), args.run_name)
    inventory = {p.stem for p in Path(args.videos).glob("*.mov")}
    if inventory != {r["name"] for r in rows}:
        raise ValueError("media_inventory_differs")
    client = LocalClient(args.base_url, args.model, args.concurrency, started + args.max_wall_s)
    sem = asyncio.Semaphore(args.par)
    async def video_job(source):
        async with sem:
            row = {"name": source["name"], "status": "failed", "windows": [], "type_enablement_qualified": False}
            video = None
            try:
                if not source["source_processed"] or not source["states"]:
                    row.update(status="abstained", reason="cached_states_unavailable_or_failed")
                    return row
                if time.monotonic() >= client.deadline:
                    raise TimeoutError("diagnostic_deadline")
                video = await asyncio.to_thread(decode, str(Path(args.videos) / (source["name"] + ".mov")), FPS,
                                               max_frames=1000, max_memory_bytes=2 * 1024**3, deadline=client.deadline)
                windows = transaction_windows(source["states"], video.duration)
                row["proposed_transactions"] = len(windows)
                if len(windows) > 8:
                    raise ValueError("transaction_window_budget_exceeded")
                if not windows:
                    row.update(status="abstained", reason="no_state_transaction_proposal")
                    return row
                row["located_regions"] = {key: key in source["layout"] for key in REGIONS}
                for window in windows:
                    try:
                        result = await inspect_window(client, video, source["layout"], window, args.batch_frames)
                    except Exception as exc:
                        result = {**window, "status": "failed", "error_class": error_code(exc)}
                    row["windows"].append(result)
                row["status"] = "done" if all(w["status"] == "done" for w in row["windows"]) else "failed"
            except Exception as exc:
                row["error_class"] = error_code(exc)
            finally:
                if video is not None:
                    for frame in video.frames:
                        frame.image.close()
            return row
    try:
        results = await asyncio.gather(*(video_job(row) for row in rows))
    finally:
        await client.http.aclose()
    counts = Counter(row["status"] for row in results)
    return {"schema": "forte.procedure-visible-probe.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model, "scope": "local_original_pixel_perception_diagnostic; no_labels_used",
            "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "heldout_accuracy_claim": False, "cross_validation_claim": False,
            "type_enablement_qualified": False, "service_flags_changed": False, "tuned_accuracy_parameters": 0,
            "policy": {"decode_fps": FPS, "native_pixels": True, "upsampling": False,
                       "window_selection": "all_cached_videos; union_of_visitor_cash_bag_counting_signals; fixed_2s_context",
                       "transaction_clipping": False, "max_transaction_windows_per_video": 8,
                       "frames_per_batch": args.batch_frames, "max_images_per_batch": 4 * args.batch_frames,
                       "action_phrasings": 2, "boundary_phrasings": 2,
                       "visibility_threshold_sampling_proxy": VISIBILITY_THRESHOLD,
                       "continuous_visibility_proven": False, "unobserved_between_samples": True},
            "summary": {"videos": len(results), "done": counts["done"], "failed": counts["failed"],
                        "abstained": counts["abstained"], "calls": client.calls, "attempted_calls": client.attempted_calls,
                        "prompt_tokens": client.prompt_tokens, "completion_tokens": client.completion_tokens,
                        "wall_s": round(time.monotonic() - started, 3)}, "results": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--videos", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--par", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-wall-s", type=float, default=900)
    args = parser.parse_args(argv)
    local_base_url(args.base_url)
    if not RUN_RE.fullmatch(args.model) or not 2 <= args.batch_frames <= MAX_BATCH_FRAMES or not 1 <= args.par <= 2 or not 1 <= args.concurrency <= 16 or not 30 <= args.max_wall_s <= 1800:
        raise ValueError("invalid_diagnostic_controls")
    report = asyncio.run(run(args))
    output = Path(args.out)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(output)
    print(json.dumps({"status": "done", "summary": report["summary"], "type_enablement_qualified": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Exception messages and HTTP/model bodies may contain private text.
        print(json.dumps({"status": "failed", "error_class": error_code(exc)}))
        raise SystemExit(1)
