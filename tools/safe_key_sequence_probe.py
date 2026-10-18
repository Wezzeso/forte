"""Contextual safe-key perception only; no candidates, rules, training or evidence.

All 40 source rows follow the same label-free proposal policy. Native safe-ROI
motion is cached as numbers, combined with cached cashier-role departures, then
at most two uniformly spaced proposals receive six before/during/after samples.
All image transport is lossless PNG without resizing or enhancement. Each sample
has its own lock localization, avoiding a static crop on a moving safe door.

Two closed sequence questions must independently support a distinguishable key
in the same resolved lock before and after hand withdrawal AND cashier departure.
Unknowns, proposal overflow and failures remain explicit. This tool never enables
a type, certifies an omission, emits a violation or claims 40-video accuracy.
Labels and label timestamps are not accepted. Host scoring opens labels only
after inference terminates. Output contains enums, booleans, boxes and numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import io
import ipaddress
import json
import math
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlparse


NAME_RE = re.compile(r"X-[1-9][0-9]{0,2}\Z")
FPS = 2.0
MOTION_DELTA = 20
MOTION_FRACTION = .08
ACTIVITY_GAP = 1.0
PROPOSAL_SEPARATION = 2.0
WINDOW_LIMIT = 2
OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
LOCK_EXPANSION = 2.0
LOCK_MIN_SIDE = 24
POST_SEPARATION = .5
MAX_TOKENS = 900
REQUEST_TIMEOUT = 45.0
YESNO = ("yes", "no", "unclear")
VISIBILITY = ("resolved", "occluded", "not_visible", "too_small")
KEY_STATE = ("inserted", "empty", "unclear")
CASHIER = ("at_safe", "away_visible", "left_room", "unclear")
ERRORS = frozenset("AssertionError AttributeError ConnectionError FileNotFoundError IndexError JSONDecodeError KeyError MemoryError OSError OverflowError PermissionError RuntimeError TimeoutError TypeError UnicodeDecodeError ValueError CancelledError ConnectError ConnectTimeout HTTPStatusError NetworkError PoolTimeout ReadError ReadTimeout RemoteProtocolError TransportError ValidationError OtherError".split())
REASONS = frozenset(("source_failed_or_no_states", "safe_not_located", "no_resolved_sequence", "deadline", "proposal_overflow"))
FIELDS = ("cashier_at_desk", "cashier_in_room", "cashier_seat")


def enum(values):
    return {"type": "string", "enum": list(values)}


def object_schema(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def array_schema(item, count):
    return {"type": "array", "items": item, "minItems": count, "maxItems": count}


def localization_schema(count):
    box = {"anyOf": [{"type": "null"}, {"type": "array", "items": {
        "type": "number", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4}]}
    return object_schema({"frames": array_schema(object_schema({
        "i": {"type": "integer", "minimum": 0, "maximum": count - 1},
        "lock_box": box, "lock_visibility": enum(VISIBILITY)}), count)})


def sequence_schema(count):
    return object_schema({"frames": array_schema(object_schema({
        "i": {"type": "integer", "minimum": 0, "maximum": count - 1},
        "key_state": enum(KEY_STATE), "lock_visibility": enum(VISIBILITY),
        "hand_at_lock": enum(YESNO), "cashier_position": enum(CASHIER)}), count),
        "same_physical_lock": enum(YESNO), "same_distinguishable_key": enum(YESNO), "hand_withdrawal_observed": enum(YESNO),
        "cashier_departure_observed": enum(YESNO)})


LOCALIZE = (
    "Each numbered source time has an original full CCTV frame followed by its native safe crop. "
    "For EACH native safe crop independently locate the actual lock cylinder/keyhole, not a handle. "
    "lock_box=[x1,y1,x2,y2] uses normalized 0-1000 coordinates relative to THAT safe crop only. "
    "A safe door can move; do not reuse a preceding location. resolved requires a visually "
    "distinguishable actual lock. Use null and occluded/not_visible/too_small when it cannot "
    "be resolved. Never invent a keyhole from safe shape or customary practice. i=index; JSON only."
)
SEQUENCE_COMMON = (
    "Inspect this ordered before/during/after sequence. Each numbered time has original full-frame "
    "role context, a native safe crop, and its independently localized native lock crop if resolved. "
    "All are lossless PNG source pixels with no enhancement. Role means cashier, never identity. "
    "For every index, key_state=inserted ONLY for a visibly distinguishable actual key entering "
    "or protruding from that actual lock; empty ONLY for a resolved keyhole without a key. "
    "A handle, latch, highlight, open safe, earlier hand movement or expected procedure cannot "
    "support inserted. Otherwise use unclear. lock_visibility=resolved only if key versus no-key "
    "can actually be distinguished in that frame. hand_at_lock records physical hand contact. "
    "cashier_position=at_safe for operative proximity, away_visible for the cashier visibly away "
    "from the safe, left_room only for an observed exit, otherwise unclear. Report exactly the "
    "listed schema fields; no text, identities, legal conclusions or violation assertions. "
    "same_distinguishable_key refers only to visible key shape/position across the shown samples, "
    "not a person. Use unclear when the key cannot be individually distinguished or removal, "
    "replacement or reinsertion is visually ambiguous. Never infer key identity from the lock identity. "
)
PHRASINGS = (
    SEQUENCE_COMMON +
    "Follow the micro-interaction in time: does an observed hand contact end, and does the cashier "
    "then visibly leave the safe while the same actual lock remains observable? Set "
    "hand_withdrawal_observed=yes only for visible contact followed by visible separation; "
    "cashier_departure_observed=yes only for an at-safe to away/exit transition. "
    "same_physical_lock=yes requires visual association of the same lock before and after. "
    "Missing transitions, tiny pixels or occlusion require unclear, not a guessed yes. "
    "Sparse samples never prove continuous persistence between images or exclude unseen removal/reinsertion.",
    SEQUENCE_COMMON +
    "Independently challenge the sequence: can the safe/lock before contact, the hand releasing "
    "it, and the cashier subsequently away be actually followed, or is a transition unseen? "
    "Use no when a clearly visible sequence contradicts a proposed transition and unclear when "
    "coverage cannot decide. A person absent in one image is insufficient for an observed "
    "departure. A key seen only while held does not show a key left behind. Associate the same "
    "physical lock across any door movement before answering same_physical_lock. "
    "Do not infer continuous key visibility or the absence of removal/reinsertion in unsampled gaps."
)


def error_code(exc):
    name = type(exc).__name__
    return name if name in ERRORS else "OtherError"


def number(value, high=86400):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= high


def box_valid(box):
    return (isinstance(box, list) and len(box) == 4 and all(number(v, 1000) for v in box)
            and box[0] < box[2] and box[1] < box[3])


def native_box(size, box, *, expansion=1.0, minimum=1):
    if not box_valid(box):
        raise ValueError("invalid_box")
    width, height = size
    x1, y1, x2, y2 = [v * a / 1000 for v, a in zip(box, (width, height, width, height))]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(minimum, (x2 - x1) * expansion), max(minimum, (y2 - y1) * expansion)
    out = (max(0, math.floor(cx - bw / 2)), max(0, math.floor(cy - bh / 2)),
           min(width, math.ceil(cx + bw / 2)), min(height, math.ceil(cy + bh / 2)))
    if out[0] >= out[2] or out[1] >= out[3]:
        raise ValueError("empty_crop")
    return out


def local_url(value):
    parsed = urlparse(value)
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        local = False
    if parsed.scheme != "http" or not local or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("loopback_required")
    return value.rstrip("/") + "/"


def png_part(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}}


def text(value):
    return {"type": "text", "text": value}


def cached_rows(value):
    rows = value.get("results") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("expected_40_rows")
    result, seen = [], set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or not NAME_RE.fullmatch(row["name"]) or row["name"] in seen):
            raise ValueError("invalid_inventory")
        seen.add(row["name"])
        states = row.get("states")
        if not isinstance(states, list) or len(states) > 1000:
            raise ValueError("invalid_states")
        clean = []
        for state in states:
            if not isinstance(state, dict) or not number(state.get("t")):
                raise ValueError("invalid_state_time")
            clean.append({"t": state["t"], **{k: state[k] for k in FIELDS if type(state.get(k)) is bool}})
        if any(a["t"] >= b["t"] for a, b in zip(clean, clean[1:])):
            raise ValueError("invalid_state_order")
        layout = row.get("layout") if isinstance(row.get("layout"), dict) else {}
        safe = layout.get("safe")
        if safe is not None and not box_valid(safe):
            raise ValueError("invalid_safe_layout")
        result.append({"name": row["name"], "processed": row.get("processed") is True
                       and row.get("status") not in ("failed", "processing") and not bool(row.get("error")),
                       "states": clean, "layout": {"safe": safe}})
    return sorted(result, key=lambda row: row["name"])


def safe_activity(frames, safe_box):
    from PIL import ImageChops
    result, previous = [], None
    try:
        for frame in frames:
            crop = frame.image.crop(native_box(frame.image.size, safe_box))
            try:
                current = crop.convert("L")
            finally:
                crop.close()
            fraction = 0.0
            if previous is not None:
                difference = ImageChops.difference(previous, current)
                try:
                    histogram = difference.histogram()
                    fraction = sum(histogram[MOTION_DELTA + 1:]) / max(1, sum(histogram))
                finally:
                    difference.close()
                previous.close()
            previous = current
            result.append({"t": frame.t, "active_fraction": round(fraction, 6)})
    finally:
        if previous is not None:
            previous.close()
    return result


def proposals(states, activity, first, last):
    anchors = []
    for before, after in zip(states, states[1:]):
        if any(before.get(k) is True and after.get(k) is False for k in FIELDS):
            anchors.append({"anchor": after["t"], "source": "cached_cashier_departure"})
    spans = []
    for row in activity:
        if row["active_fraction"] > MOTION_FRACTION:
            if spans and row["t"] - spans[-1][1] <= ACTIVITY_GAP:
                spans[-1][1] = row["t"]
            else:
                spans.append([row["t"], row["t"]])
    anchors.extend({"anchor": end, "source": "safe_activity_end"} for _, end in spans)
    anchors.sort(key=lambda row: row["anchor"])
    merged = []
    for row in anchors:
        if not first <= row["anchor"] <= last:
            continue
        if merged and row["anchor"] - merged[-1]["anchor"] <= PROPOSAL_SEPARATION:
            # Prefer an explicit role proposal over activity within the same local
            # interaction, uniformly for all videos. No correctness/label input.
            if row["source"] == "cached_cashier_departure":
                merged[-1] = row
        else:
            merged.append(row)
    fallback = not merged
    if fallback:
        merged = [{"anchor": first + (last - first) * fraction, "source": "uniform_no_proposal_control"}
                  for fraction in (1 / 3, 2 / 3)]
    count = len(merged)
    if count > WINDOW_LIMIT:
        merged = [merged[round(i * (count - 1) / (WINDOW_LIMIT - 1))] for i in range(WINDOW_LIMIT)]
    return merged, {"proposed_windows": count, "unassessed_proposals": max(0, count - len(merged)),
                    "proposal_coverage_complete": count <= WINDOW_LIMIT,
                    "fallback_uniform_control": fallback}


def select_frames(video, anchor):
    selected, seen = [], set()
    for offset in OFFSETS:
        frame = video.nearest(min(video.frames[-1].t, max(video.frames[0].t, anchor + offset)))
        if frame.t not in seen:
            selected.append(frame)
            seen.add(frame.t)
    return sorted(selected, key=lambda frame: frame.t)


def sequence_support(answers, localization, times):
    """Strict perception support only; false support is not a negative verdict."""
    if len(answers) != 2 or len(times) < 3:
        raise ValueError("invalid_sequence")
    for answer in answers:
        if [r["i"] for r in answer["frames"]] != list(range(len(times))):
            raise ValueError("sequence_index_alignment")
    if [r["i"] for r in localization["frames"]] != list(range(len(times))):
        raise ValueError("localization_index_alignment")
    merged = []
    for i, stamp in enumerate(times):
        a, b, located = answers[0]["frames"][i], answers[1]["frames"][i], localization["frames"][i]
        item = {"t": stamp, **{key: a[key] if a[key] == b[key] else "unclear"
                               for key in ("key_state", "lock_visibility", "hand_at_lock", "cashier_position")}}
        if located["lock_visibility"] != "resolved" or not box_valid(located["lock_box"]):
            item["key_state"] = "unclear"
            item["lock_visibility"] = located["lock_visibility"]
        if item["lock_visibility"] != "resolved":
            item["key_state"] = "unclear"
        item["paired_disagreements"] = sum(a[k] != b[k] for k in ("key_state", "lock_visibility", "hand_at_lock", "cashier_position"))
        merged.append(item)
    facts = {key: answers[0][key] if answers[0][key] == answers[1][key] else "unclear"
             for key in ("same_physical_lock", "same_distinguishable_key", "hand_withdrawal_observed", "cashier_departure_observed")}
    before = [i for i, row in enumerate(merged) if row["key_state"] == "inserted"
              and row["lock_visibility"] == "resolved" and row["cashier_position"] == "at_safe"]
    touching = [i for i, row in enumerate(merged) if row["hand_at_lock"] == "yes" and row["cashier_position"] == "at_safe"]
    after = [i for i, row in enumerate(merged) if row["key_state"] == "inserted"
             and row["lock_visibility"] == "resolved" and row["hand_at_lock"] == "no"
             and row["cashier_position"] in ("away_visible", "left_room")]
    released = [i for i, row in enumerate(merged) if row["key_state"] == "inserted"
                and row["lock_visibility"] == "resolved" and row["hand_at_lock"] == "no"
                and row["cashier_position"] == "at_safe"]
    support = False
    supporting_indices = []
    if all(facts[k] == "yes" for k in facts):
        for b in before:
            for h in touching:
                if h < b:
                    continue
                for r in released:
                    if r <= h:
                        continue
                    post = [i for i in after if i > r]
                    # Unknown or contradictory observed intermediate lock states
                    # cannot be bridged by a positive pair of endpoint snapshots.
                    if (len(post) >= 2 and times[post[-1]] - times[post[0]] >= POST_SEPARATION
                            and all(merged[i]["key_state"] == "inserted" and merged[i]["lock_visibility"] == "resolved"
                                    for i in range(r, post[-1] + 1))):
                        support = True
                        supporting_indices = [b, h, r, post[0], post[-1]]
                        break
                if support:
                    break
            if support:
                break
    resolved = sum(row["lock_visibility"] == "resolved" and row["key_state"] != "unclear" for row in merged)
    return {"observations": merged, "paired_sequence_facts": facts,
            "key_remains_after_departure_support": support, "supporting_indices": supporting_indices,
            "support_start_sec": times[supporting_indices[0]] if support else None,
            "support_end_sec": times[supporting_indices[-1]] if support else None,
            "resolved_key_frames": resolved, "assessment": "positive_perception_support" if support
            else "resolved_observations_without_complete_support" if resolved else "unresolved",
            "diagnostic_only": True, "type_enablement_qualified": False,
            "sampling_proxy_only": True, "continuous_visibility_proven": False,
            "unseen_key_removal_or_reinsertion_excluded": False}


class LocalClient:
    def __init__(self, url, model, deadline, concurrency=4):
        import httpx
        self.http = httpx.AsyncClient(base_url=local_url(url), trust_env=False, follow_redirects=False)
        self.model, self.deadline = model, deadline
        self.sem = asyncio.Semaphore(concurrency)
        self.calls = self.attempted_calls = self.prompt_tokens = self.completion_tokens = 0

    async def ask(self, content, schema):
        from jsonschema import validate
        async with self.sem:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("diagnostic_deadline")
            self.attempted_calls += 1
            response = await self.http.post("chat/completions", timeout=min(REQUEST_TIMEOUT, remaining), json={
                "model": self.model, "seed": 0, "temperature": 0, "max_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": "Inspect visible source pixels only. Never identify people. Return strict JSON only, no thinking or free text."},
                             {"role": "user", "content": content}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
                "chat_template_kwargs": {"enable_thinking": False}})
        response.raise_for_status()
        payload = response.json()
        self.calls += 1
        usage = payload.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        choice = payload["choices"][0]
        message = choice["message"]
        if choice.get("finish_reason") != "stop" or any(message.get(k) for k in ("reasoning", "reasoning_content")):
            raise ValueError("reasoning_or_truncation")
        def reject(value):
            raise ValueError("nonfinite_json")
        answer = json.loads(message["content"], parse_constant=reject)
        validate(answer, schema)
        return answer


async def inspect_window(client, video, safe_box, proposal):
    started = time.monotonic()
    frames = select_frames(video, proposal["anchor"])
    row = {**proposal, "status": "failed", "frame_times": [f.t for f in frames]}
    safe_crops, lock_crops = [], []
    try:
        if len(frames) < 3:
            raise ValueError("insufficient_sequence_samples")
        locate_content = []
        for i, frame in enumerate(frames):
            crop = frame.image.crop(native_box(frame.image.size, safe_box))
            safe_crops.append(crop)
            locate_content.extend([text(f"index={i}; original source t={frame.t:.6f}s; original full frame"),
                                   png_part(frame.image), text("native safe crop"), png_part(crop)])
        localization = await client.ask([*locate_content, text(LOCALIZE)], localization_schema(len(frames)))
        if [r["i"] for r in localization["frames"]] != list(range(len(frames))):
            raise ValueError("localization_index_alignment")
        content, crop_boxes = [], []
        for i, (frame, safe, located) in enumerate(zip(frames, safe_crops, localization["frames"])):
            content.extend([text(f"index={i}; original source t={frame.t:.6f}s; original full frame"),
                            png_part(frame.image), text("native safe crop"), png_part(safe)])
            safe_native = native_box(frame.image.size, safe_box)
            if located["lock_box"] is not None and not box_valid(located["lock_box"]):
                raise ValueError("invalid_localized_box")
            if located["lock_visibility"] != "resolved" or located["lock_box"] is None:
                crop_boxes.append(None)
                content.append(text("native lock crop unavailable; localization unresolved; do not invent detail"))
                continue
            box = native_box(safe.size, located["lock_box"], expansion=LOCK_EXPANSION, minimum=LOCK_MIN_SIDE)
            crop = safe.crop(box)
            lock_crops.append(crop)
            crop_boxes.append([safe_native[0] + box[0], safe_native[1] + box[1], safe_native[0] + box[2], safe_native[1] + box[3]])
            content.extend([text("independently localized native lock crop"), png_part(crop)])
        answers = await asyncio.gather(*(client.ask([*content, text(prompt)], sequence_schema(len(frames)))
                                         for prompt in PHRASINGS), return_exceptions=True)
        if any(isinstance(answer, BaseException) for answer in answers):
            raise RuntimeError("paired_sequence_call_failed")
        row.update(status="done", localization=localization, native_lock_boxes=crop_boxes, answers=answers,
                   **sequence_support(answers, localization, row["frame_times"]))
    except Exception as exc:
        row["error_class"] = error_code(exc)
    finally:
        for image in [*safe_crops, *lock_crops]:
            image.close()
        row["wall_s"] = round(time.monotonic() - started, 3)
    return row


def policy():
    return {"decode_fps": FPS, "motion_delta": MOTION_DELTA, "motion_fraction": MOTION_FRACTION,
            "activity_merge_gap_s": ACTIVITY_GAP, "proposal_separation_s": PROPOSAL_SEPARATION,
            "window_limit": WINDOW_LIMIT, "frame_offsets_s": list(OFFSETS),
            "lock_crop_expansion": LOCK_EXPANSION, "lock_min_side_px": LOCK_MIN_SIDE,
            "phrasings": 2, "max_output_tokens": MAX_TOKENS, "request_timeout_s": REQUEST_TIMEOUT,
            "parallel_videos": 2, "concurrent_requests": 4, "post_snapshot_separation_s": POST_SEPARATION,
            "native_png_only": True, "enhancement": False, "temporal_fusion": False,
            "labels_used": False, "declared_control_count": 17, "label_fitted_parameters": 0}


def need(condition, code):
    if not condition:
        raise ValueError(code)


def project_report(raw, expected_names):
    """Strict host export projection: no arbitrary strings or model text survive."""
    from jsonschema import validate
    need(isinstance(raw, dict) and raw.get("schema") == "forte.safe-key-sequence-diagnostic.v1"
         and raw.get("model") == "q30test" and raw.get("policy") == policy()
         and raw.get("type_enablement_qualified") is False and raw.get("accuracy_claim") is False
         and raw.get("continuous_visibility_proven") is False, "report_header_invalid")
    rows = raw.get("results")
    need(isinstance(rows, list) and len(rows) == 40, "report_40_rows_required")
    results, seen = [], set()
    for row in rows:
        need(isinstance(row, dict) and row.get("name") in expected_names
             and row["name"] not in seen, "report_inventory_invalid")
        seen.add(row["name"])
        need(row.get("status") in {"done", "failed", "abstained"}, "report_status_invalid")
        out = {"name": row["name"], "status": row["status"], "windows": [], "type_enablement_qualified": False}
        for key, allowed in (("reason", REASONS), ("error_class", ERRORS)):
            if key in row:
                need(row[key] in allowed, "report_enum_invalid")
                out[key] = row[key]
        for key in ("proposal_coverage_complete", "fallback_uniform_control"):
            if key in row:
                need(type(row[key]) is bool, "report_boolean_invalid")
                out[key] = row[key]
        for key in ("proposed_windows", "unassessed_proposals"):
            if key in row:
                need(type(row[key]) is int and 0 <= row[key] <= 1000, "report_count_invalid")
                out[key] = row[key]
        activity = row.get("safe_activity", [])
        need(isinstance(activity, list) and len(activity) <= 600, "report_activity_invalid")
        out["safe_activity"] = []
        for item in activity:
            need(isinstance(item, dict) and number(item.get("t"))
                 and number(item.get("active_fraction"), 1), "report_activity_number_invalid")
            out["safe_activity"].append({"t": item["t"], "active_fraction": item["active_fraction"]})
        need(all(a["t"] < b["t"] for a, b in zip(out["safe_activity"], out["safe_activity"][1:])), "report_activity_order_invalid")
        windows = row.get("windows")
        need(isinstance(windows, list) and len(windows) <= WINDOW_LIMIT, "report_windows_invalid")
        for window in windows:
            need(isinstance(window, dict) and number(window.get("anchor"))
                 and window.get("source") in {"cached_cashier_departure", "safe_activity_end", "uniform_no_proposal_control"}
                 and window.get("status") in {"done", "failed"}, "report_window_invalid")
            w = {"anchor": window["anchor"], "source": window["source"], "status": window["status"]}
            times = window.get("frame_times")
            need(isinstance(times, list) and 0 <= len(times) <= len(OFFSETS)
                 and all(number(t) for t in times) and all(a < b for a, b in zip(times, times[1:])), "report_times_invalid")
            w["frame_times"] = times
            need(number(window.get("wall_s"), 300), "report_window_wall_invalid")
            w["wall_s"] = window["wall_s"]
            if w["status"] == "failed":
                need(window.get("error_class") in ERRORS, "report_window_error_invalid")
                w["error_class"] = window["error_class"]
            else:
                need(len(times) >= 3, "report_done_samples_invalid")
                localization, answers = window.get("localization"), window.get("answers")
                validate(localization, localization_schema(len(times)))
                need(isinstance(answers, list) and len(answers) == 2, "report_answer_count_invalid")
                for answer in answers:
                    validate(answer, sequence_schema(len(times)))
                boxes = window.get("native_lock_boxes")
                need(isinstance(boxes, list) and len(boxes) == len(times), "report_native_boxes_invalid")
                for box in boxes:
                    need(box is None or isinstance(box, list) and len(box) == 4
                         and all(type(v) is int and 0 <= v <= 16384 for v in box)
                         and box[0] < box[2] and box[1] < box[3], "report_native_box_invalid")
                computed = sequence_support(answers, localization, times)
                need(window.get("key_remains_after_departure_support") is computed["key_remains_after_departure_support"],
                     "report_support_mismatch")
                w.update(localization=localization, answers=answers, native_lock_boxes=boxes, **computed)
            out["windows"].append(w)
        support = any(w.get("key_remains_after_departure_support") is True for w in out["windows"])
        need(row.get("key_remains_after_departure_support", False) is support, "report_video_support_mismatch")
        out["key_remains_after_departure_support"] = support
        need(out["status"] != "done" or out["windows"]
             and all(w["status"] == "done" for w in out["windows"])
             and any(w.get("resolved_key_frames", 0) > 0 for w in out["windows"]), "done_video_unresolved_or_failed")
        need(out["status"] == "failed" or all(w["status"] != "failed" for w in out["windows"]), "window_failure_hidden")
        results.append(out)
    need(seen == set(expected_names), "report_inventory_incomplete")
    source_summary = raw.get("summary")
    need(isinstance(source_summary, dict), "report_summary_invalid")
    summary = {}
    for key in ("calls", "attempted_calls", "prompt_tokens", "completion_tokens"):
        value = source_summary.get(key)
        need(type(value) is int and 0 <= value <= 10**9, "report_summary_number_invalid")
        summary[key] = value
    need(summary["calls"] <= summary["attempted_calls"] <= 40 * WINDOW_LIMIT * 3
         and number(source_summary.get("wall_s"), 270), "report_budget_invalid")
    summary.update(videos=40, done=sum(r["status"] == "done" for r in results),
        failed=sum(r["status"] == "failed" for r in results), abstained=sum(r["status"] == "abstained" for r in results),
        positive_support_videos=sum(r["key_remains_after_departure_support"] for r in results),
        support_on_failed_videos=sum(r["status"] == "failed" and r["key_remains_after_departure_support"] for r in results),
        proposal_overflow_videos=sum(r.get("unassessed_proposals", 0) > 0 for r in results), wall_s=source_summary["wall_s"])
    return {"schema": "forte.safe-key-sequence-diagnostic.v1", "model": "q30test", "policy": policy(),
            "diagnostic_only": True, "type_enablement_qualified": False, "accuracy_claim": False,
            "labels_used_for_processing": False, "sampling_proxy_only": True,
            "continuous_visibility_proven": False, "summary": summary, "results": results}


def score_after_inference(report, labels):
    """Host-only descriptive negative gate; never tunes thresholds or decisions."""
    if isinstance(labels, dict) and "labels" in labels:
        labels = labels["labels"]
    names = {row["name"] for row in report["results"]}
    need(isinstance(labels, dict) and set(labels) == names and len(names) == 40
         and all(isinstance(v, dict) and type(v.get("violation")) is bool for v in labels.values()), "scorer_labels_invalid")
    negative = [row for row in report["results"] if labels[row["name"]]["violation"] is False]
    positive = [row for row in report["results"] if labels[row["name"]]["violation"] is True]
    need(len(negative) == 19 and len(positive) == 21, "scorer_label_counts_invalid")
    supported = lambda rows: sum(row["key_remains_after_departure_support"] for row in rows)
    return {"scope": "all40_in_sample_contextual_perception_only", "videos": 40,
            "negative_videos": 19, "positive_videos": 21,
            "negative_support_videos": supported(negative), "positive_support_videos": supported(positive),
            "negative_failed_videos": sum(row["status"] == "failed" for row in negative),
            "negative_support_on_failed_videos": sum(row["status"] == "failed" and row["key_remains_after_departure_support"] for row in negative),
            "negative_abstained_videos": sum(row["status"] == "abstained" for row in negative),
            "negative_overflow_videos": sum(row.get("unassessed_proposals", 0) > 0 for row in negative),
            "observed_zero_negative_support": supported(negative) == 0,
            "negative_assessment_complete": all(row["status"] == "done" and row.get("proposal_coverage_complete") is True for row in negative),
            "unassessed_negative_videos": sum(row["status"] != "done" or row.get("proposal_coverage_complete") is not True for row in negative),
            "type_enablement_qualified": False, "continuous_visibility_proven": False,
            "accuracy_claim": False, "thresholds_fitted": 0,
            "labels_opened_only_after_inference": True,
            "per_video": [{"name": row["name"], "label_violation": labels[row["name"]]["violation"],
                "status": row["status"], "positive_perception_support": row["key_remains_after_departure_support"],
                "unassessed_proposals": row.get("unassessed_proposals", 0)} for row in report["results"]]}


async def run(args):
    from forte.fast.video import decode
    started = time.monotonic()
    source_path = Path(args.run)
    source_bytes = source_path.read_bytes()
    rows = cached_rows(json.loads(source_bytes))
    deadline = started + args.max_wall_s
    client = LocalClient(args.base_url, args.model, deadline)
    sem = asyncio.Semaphore(2)

    async def one(source):
        row = {"name": source["name"], "status": "failed", "windows": [],
               "type_enablement_qualified": False, "proposal_coverage_complete": False}
        video = None
        async with sem:
            try:
                if not source["processed"] or not source["states"]:
                    row["reason"] = "source_failed_or_no_states"
                    return row
                if source["layout"]["safe"] is None:
                    row.update(status="abstained", reason="safe_not_located")
                    return row
                if time.monotonic() >= deadline:
                    row["reason"] = "deadline"
                    raise TimeoutError("diagnostic_deadline")
                video = await asyncio.to_thread(decode, str(Path(args.videos) / (source["name"] + ".mov")),
                    FPS, max_frames=600, max_memory_bytes=1024**3, deadline=deadline)
                activity = await asyncio.to_thread(safe_activity, video.frames, source["layout"]["safe"])
                proposed, selection = proposals(source["states"], activity, video.frames[0].t, video.frames[-1].t)
                row.update(selection)
                row["safe_activity"] = activity
                for proposal in proposed:
                    row["windows"].append(await inspect_window(client, video, source["layout"]["safe"], proposal))
                failed = any(w["status"] == "failed" for w in row["windows"])
                resolved = any(w.get("resolved_key_frames", 0) > 0 for w in row["windows"])
                row["status"] = "failed" if failed else "done" if resolved else "abstained"
                row["key_remains_after_departure_support"] = any(w.get("key_remains_after_departure_support") is True for w in row["windows"])
                if not resolved and not failed:
                    row["reason"] = "no_resolved_sequence"
                elif not row["proposal_coverage_complete"]:
                    row["reason"] = "proposal_overflow"
            except Exception as exc:
                row["error_class"] = error_code(exc)
            finally:
                if video is not None:
                    for frame in video.frames:
                        frame.image.close()
        return row

    try:
        results = await asyncio.gather(*(one(row) for row in rows))
    finally:
        await client.http.aclose()
    return {"schema": "forte.safe-key-sequence-diagnostic.v1", "model": args.model,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True,
            "labels_used_for_processing": False, "type_enablement_qualified": False, "accuracy_claim": False,
            "sampling_proxy_only": True, "continuous_visibility_proven": False,
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(source_bytes).hexdigest(), "policy": policy(),
            "summary": {"videos": 40, "done": sum(r["status"] == "done" for r in results),
                "abstained": sum(r["status"] == "abstained" for r in results),
                "failed": sum(r["status"] == "failed" for r in results),
                "positive_support_videos": sum(r.get("key_remains_after_departure_support") is True for r in results),
                "support_on_failed_videos": sum(r["status"] == "failed" and r.get("key_remains_after_departure_support") is True for r in results),
                "proposal_overflow_videos": sum(r.get("unassessed_proposals", 0) > 0 for r in results),
                "calls": client.calls, "attempted_calls": client.attempted_calls,
                "prompt_tokens": client.prompt_tokens, "completion_tokens": client.completion_tokens,
                "wall_s": round(time.monotonic() - started, 3)}, "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "videos", "model", "out"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--base-url", type=local_url, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-wall-s", type=float, default=240)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_wall_s) or not 10 <= args.max_wall_s <= 240:
        raise ValueError("invalid_wall_budget")
    report = asyncio.run(run(args))
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise ValueError("existing_output_refused")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "done", "summary": report["summary"], "diagnostic_only": True,
                      "type_enablement_qualified": False}))
    # Partial failures remain usable structured diagnostics, never correct cases.
    # The host wrapper projects all40 rows and records the failed denominator.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_class": error_code(exc)}))
        raise SystemExit(2)
