"""Fast violation analyzer: per-second scene states -> rules -> holistic review -> verification."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

from forte.fast import prompts as P
from forte.fast.catalog import BY_ID
from forte.fast.video import DecodedVideo, Frame
from forte.fast.vlm import VLMClient, image_part, text_part


class _CheckedClient:
    """Keep swallowed query failures scoped to this video, even with a shared client."""

    def __init__(self, client, priority: int | None = None, *, strict_model_protocol: bool = False):
        self.client = client
        self.failures: list[str] = []
        # Scheduling only (flag job_priority): earlier jobs get model slots first.
        self.priority = priority
        self.strict_model_protocol = strict_model_protocol

    async def chat_json(self, *args, **kwargs):
        if self.priority is not None:
            kwargs.setdefault("priority", self.priority)
        if self.strict_model_protocol:
            kwargs["strict_response"] = True
        try:
            return await self.client.chat_json(*args, **kwargs)
        except Exception as exc:
            self.failures.append(type(exc).__name__)
            raise


OBS_RU = {
    "WORKSTATION_UNLOCKED": "Кассир покидает рабочее место; на экране ноутбука остаётся открытое рабочее приложение, экран блокировки не включён.",
    "CASH_LEFT_UNATTENDED": "Кассир покидает рабочее место; на столе без присмотра остаются наличные или иные ценности (сумка, монеты).",
    "PREVIOUS_CASH_ON_DESK": "После ухода клиента на столе остаются наличные/предметы предыдущей операции, и в этот момент к кассе подходит следующий клиент.",
    "COLLECTOR_NO_DOCUMENTS": "Посетитель передаёт кассиру инкассаторскую сумку; предъявление или передача документа (удостоверения, препроводительного документа) в кадре не наблюдается.",
    "COUNT_WITHOUT_CLIENT": "Клиент уходит от кассы, после чего кассир продолжает пересчёт/обработку наличных в его отсутствие.",
    "UNAUTHORIZED_PERSON": "Во время обслуживания клиента в помещении кассы появляется третье лицо и подходит к рабочему месту кассира.",
    "PERSONAL_ITEMS:drink": "На рабочем месте кассира находится напиток (кружка/стакан) — личный предмет в зоне работы с наличными.",
    "PERSONAL_ITEMS:personal_items": "На столе кассира находятся личные вещи (верхняя одежда или сумка).",
    "CASH_ROOM_DOOR_OPEN": "Кассир входит в кассовое помещение и не закрывает за собой дверь; дверь остаётся открытой.",
    "NO_CLIENT_ID": "Клиент передаёт наличные, кассир принимает их; предъявление документа, удостоверяющего личность, перед приёмом в кадре не наблюдается.",
    "NO_AUTHENTICITY_CHECK": "Кассир пересчитывает банкноты вручную; проверка банкнот на детекторе подлинности в кадре не наблюдается.",
    "LEFT_DURING_SERVICE": "Кассир встаёт и уходит от рабочего места, пока клиент ещё находится у кассы.",
}


@dataclass
class Candidate:
    type_id: str
    t: float
    source: str                 # "rule" | "holistic"
    note: str = ""
    score: float = 0.5
    note_ru: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.note_ru:
            key = self.type_id
            if self.type_id == "PERSONAL_ITEMS":
                key = "PERSONAL_ITEMS:" + ("drink" if "drink" in self.note else "personal_items")
            self.note_ru = OBS_RU.get(key, "")


@dataclass
class Event:
    type_id: str
    t: float
    observation_ru: str
    confidence: float
    sources: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    events: list[Event]
    states: list[dict[str, Any]]
    candidates: list[Candidate]
    verifications: list[dict[str, Any]]
    timings: dict[str, float]
    layout: dict[str, list[float]] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)


DEFAULT_CONFIG: dict[str, Any] = {
    "state_fps": 1.0,
    "state_max_side": 896,
    "holistic_frames": 24,
    "holistic_max_side": 640,
    "verify_window": 4.0,
    "verify_frames": 10,
    "verify_max_side": 768,
    "verify_threshold": 0.5,
    "use_rules": True,
    "use_holistic": False,
    "use_verify": False,
    "merge_window": 6.0,
    "min_away": 3,
    "max_state_frames": 100,
    "require_corroboration": False,
    "specialists": {"enabled": False},
    # Types whose detector is not reliable enough at this video resolution (see evaluation notes).
    "disabled_types": ["NO_CLIENT_ID", "NO_AUTHENTICITY_CHECK", "NO_RECOUNT", "KEY_LEFT_IN_SAFE",
                       "LEFT_DURING_SERVICE", "CASH_HANDOVER_IRREGULAR"],
    # Speed flags (default off; each is gated by tools/fast_agreement.py against the reference).
    # S3: start full-frame state questions while the layout call runs (same requests, new order).
    "parallel_layout": False,
    # S1: stop verification output after the decision fields; the description is built from the
    # catalog template instead of model free text (same prompt and grammar, shorter generation).
    "verify_concise": False,
    # S4/S12 (client side): model slots are granted to earlier jobs first (scheduling only).
    "job_priority": False,
    # S2: reuse a region's previous answer while its pixels are unchanged (forced refresh every 5 s).
    "change_gating": False,
    "gating_pixel_delta": 15,
    "gating_changed_fraction": 0.01,
    "gating_refresh_s": 5.0,
    # Accuracy flags (default off): layout consensus over k frames; dense window probes (probes.py).
    "layout_consensus": 1,
    "probe_types": [],
    # Protocol guard (default off): strict plain JSON, complete generation, no reasoning.
    "strict_model_protocol": False,
    # Local LoRA route (default off): desk cash/ID fields may use this served alias.
    "desk_adapter_model": "",
}


# --------------------------------------------------------------------------- layout / crops
CROP_KEYS = ("laptop", "safe", "door", "desk")


async def locate_layout(client: VLMClient, video: DecodedVideo, cfg: dict[str, Any] | None = None) -> dict[str, list[float]]:
    """Find fixed objects once per video (boxes in 0..1000 normalized coordinates).

    Flag ``layout_consensus`` = k > 1: ask on k frames spread over the clip and keep, per object,
    the coordinate-wise median of the boxes found on a majority of them (fixed furniture does
    not move; one occluded or misread frame must not move every crop of the video)."""
    k = int((cfg or {}).get("layout_consensus", 1) or 1)
    if k > 1 and len(video.frames) >= 3:
        n = len(video.frames)
        picks = sorted({min(n - 2, max(1, round(1 + i * (n - 3) / (k - 1)))) for i in range(k)})
        found = await asyncio.gather(*(_layout_one(client, video.frames[i]) for i in picks))
        boxes = {}
        for key in P.LAYOUT_SCHEMA["required"]:
            cands = [f[key] for f in found if key in f]
            if len(cands) * 2 > len(found):
                cands = sorted(cands, key=lambda b: b[0])
                boxes[key] = [sorted(b[i] for b in cands)[len(cands) // 2] for i in range(4)]
        return boxes
    return await _layout_one(client, video.frames[min(len(video.frames) - 1, 1)])


async def _layout_one(client: VLMClient, frame: Frame) -> dict[str, list[float]]:
    try:
        data = await client.chat_json([image_part(frame.image, max_side=1024), text_part(P.LAYOUT_PROMPT)],
                                      system=P.SYSTEM, max_tokens=300, json_schema=P.LAYOUT_SCHEMA)
    except Exception:
        return {}
    boxes = {}
    for key, box in (data or {}).items():
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 > x1 and y2 > y1 and 0 <= x1 <= 1000 and 0 <= y2 <= 1000:
                boxes[key] = [x1, y1, x2, y2]
    return boxes


def crop(image, box: list[float], expand: float = 1.6, min_side: int = 448):
    w, h = image.size
    x1, y1, x2, y2 = box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(16.0, (x2 - x1) * expand), max(16.0, (y2 - y1) * expand)
    left, top = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
    right, bottom = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))
    region = image.crop((left, top, right, bottom))
    scale = max(1.0, min_side / max(1, min(region.size)))
    if scale > 1:
        region = region.resize((int(region.width * scale), int(region.height * scale)))
    return region


def crop_parts(frame: Frame, layout: dict[str, list[float]], keys=CROP_KEYS) -> tuple[list[dict], list[str]]:
    parts, names = [], []
    for key in keys:
        if key in layout:
            parts.append(text_part(f"[close-up: {key}]"))
            parts.append(image_part(crop(frame.image, layout[key]), max_side=640))
            names.append(key)
    return parts, names


# --------------------------------------------------------------------------- change gating (S2)
def _region(size: tuple[int, int], box: list[float], expand: float) -> tuple[int, int, int, int]:
    w, h = size
    x1, y1, x2, y2 = box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(16.0, (x2 - x1) * expand), max(16.0, (y2 - y1) * expand)
    return (max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)), min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2)))


def _thumb(image, box: list[float] | None = None, expand: float = 1.0, size: int = 64):
    from PIL import Image as _Image, ImageFilter

    region = image if box is None else image.crop(_region(image.size, box, expand))
    return region.convert("L").resize((size, size), _Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.5))


def _changed(a, b, pixel_delta: int) -> float:
    """Fraction of thumbnail pixels whose blurred grey level moved more than ``pixel_delta``."""
    from PIL import ImageChops

    hist = ImageChops.difference(a, b).histogram()
    total = sum(hist)
    return sum(hist[pixel_delta + 1:]) / total if total else 1.0


def gating_plan(picks: list[Frame], cfg: dict[str, Any], box: list[float] | None = None,
                expand: float = 1.0) -> list[int]:
    """For each sampled frame, the index of the frame whose answer it may reuse.

    A frame is analysed afresh when its region changed versus the last analysed frame
    (more than ``gating_changed_fraction`` of pixels moved by more than
    ``gating_pixel_delta`` grey levels after blurring) or when ``gating_refresh_s``
    elapsed. Comparison is against the last analysed frame, so slow drift cannot
    accumulate unnoticed. Pixels only; no labels or identifiers.
    """
    delta = int(cfg.get("gating_pixel_delta", 15))
    frac = float(cfg.get("gating_changed_fraction", 0.01))
    refresh = float(cfg.get("gating_refresh_s", 5.0))
    owners: list[int] = []
    last_i, last_thumb = None, None
    for i, frame in enumerate(picks):
        thumb = _thumb(frame.image, box, expand)
        if (last_i is None or frame.t - picks[last_i].t >= refresh - 1e-6
                or _changed(thumb, last_thumb, delta) > frac):
            last_i, last_thumb = i, thumb
        owners.append(last_i)
    return owners


# --------------------------------------------------------------------------- states
async def extract_states(client: VLMClient, video: DecodedVideo, cfg: dict[str, Any],
                         layout: dict[str, list[float]] | None = None,
                         layout_task: asyncio.Future | None = None,
                         audit: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Per-frame states. With ``layout_task`` (flag parallel_layout) the full-frame question of
    every frame is sent immediately and only the crop questions wait for the layout; the
    requests themselves are identical to the sequential path."""
    fps = min(float(cfg["state_fps"]), float(cfg.get("max_state_frames", 100)) / max(video.duration, 1.0))
    step = 1.0 / fps
    picks: list[Frame] = []
    t = 0.0
    while t <= video.duration + 1e-6:
        frame = video.nearest(t)
        if not picks or frame.t != picks[-1].t:
            picks.append(frame)
        t += step
    layout = layout or {}
    desk_adapter_model = cfg.get("desk_adapter_model", "")
    if not isinstance(desk_adapter_model, str):
        raise ValueError("desk_adapter_model_invalid")
    desk_adapter_model = desk_adapter_model.strip()

    async def ask(content, schema, max_tokens=60):
        if desk_adapter_model and schema is P.DESK_SCHEMA:
            # Public desk adaptation supervises cash/ID presence only. Keep the
            # base model's four unrelated answers, using the identical crop and
            # six-field schema for both requests. Neither failure is a fallback.
            answers = await asyncio.gather(
                client.chat_json(content, system=P.SYSTEM, max_tokens=max_tokens, json_schema=schema),
                client.chat_json(content, system=P.SYSTEM, max_tokens=max_tokens, json_schema=schema,
                                 model_override=desk_adapter_model), return_exceptions=True)
            for answer in answers:
                if isinstance(answer, BaseException):
                    raise answer
            base, adapted = answers
            if not isinstance(base, dict) or not isinstance(adapted, dict) or any(
                    type(adapted.get(key)) is not bool for key in ("cash", "id_or_document")):
                raise ValueError("desk_adapter_output_invalid")
            return {**base, **{key: adapted[key] for key in ("cash", "id_or_document")}}
        try:
            return await client.chat_json(content, system=P.SYSTEM, max_tokens=max_tokens, json_schema=schema)
        except Exception:  # unreadable -> unknown, never a negative finding
            return {}

    crops = (("laptop", P.LAPTOP_Q, P.LAPTOP_SCHEMA, 1.4, 30), ("desk", P.DESK_Q, P.DESK_SCHEMA, 1.15, 160),
             ("cashier_chair", P.SEAT_Q, P.SEAT_SCHEMA, 1.8, 30), ("client_chair", P.SEAT_Q, P.SEAT_SCHEMA, 1.8, 30))

    async def one(frame: Frame) -> dict[str, Any]:
        full = ask([image_part(frame.image, max_side=cfg["state_max_side"]),
                    text_part(P.STATE_PROMPT.format(t=frame.t))], P.STATE_SCHEMA, 300)
        lay = layout
        if layout_task is not None:
            full = asyncio.ensure_future(full)
            try:
                lay = await asyncio.shield(layout_task) or {}
            except BaseException:
                full.cancel()
                raise
        jobs = [full]
        keys = []
        if cfg.get("use_crops", True):
            for key, q, schema, exp, mt in crops:
                if key in lay:
                    jobs.append(ask([image_part(crop(frame.image, lay[key], exp), max_side=768), text_part(q)],
                                    schema, mt))
                    keys.append(key)
        results = await asyncio.gather(*jobs)
        return assemble(results[0], zip(keys, results[1:]), frame.t)

    def assemble(full_result, keyed, t_frame) -> dict[str, Any]:
        state = dict(full_result or {})
        for key, res in keyed:
            if not res:
                continue
            if key == "laptop":
                state["laptop"] = P.LAPTOP_MAP.get(res.get("answer"), state.get("laptop"))
            elif key == "desk":
                state["desk"] = dict(res)
                state["cash_on_desk"] = bool(res.get("cash"))
                state["cash_bag"] = bool(state.get("cash_bag")) or bool(res.get("cash_bag"))
                state["valuables_on_desk"] = bool(res.get("cash") or res.get("cash_bag") or res.get("coins_or_valuables"))
                state["drink"] = bool(res.get("drink"))
                state["personal_items"] = bool(res.get("coat_or_handbag"))
                state["id_card_shown"] = bool(state.get("id_card_shown")) or bool(res.get("id_or_document"))
            elif key == "cashier_chair":
                state["cashier_seat"] = res.get("answer") == "yes"
            elif key == "client_chair":
                state["client_seat"] = res.get("answer") == "yes"
        state["t"] = t_frame
        return state

    async def gated() -> list[dict[str, Any]]:
        """Flag change_gating: ask only frames whose region changed; reuse the last answer otherwise."""
        full_owner = gating_plan(picks, cfg)
        full_futs = {i: asyncio.ensure_future(ask([image_part(picks[i].image, max_side=cfg["state_max_side"]),
                                                   text_part(P.STATE_PROMPT.format(t=picks[i].t))], P.STATE_SCHEMA, 300))
                     for i in sorted(set(full_owner))}
        crop_futs: dict[tuple[str, int], asyncio.Future] = {}
        crop_owner: dict[str, list[int]] = {}
        try:
            lay = layout
            if layout_task is not None:
                lay = await asyncio.shield(layout_task) or {}
            if cfg.get("use_crops", True):
                for key, q, schema, exp, mt in crops:
                    if key not in lay:
                        continue
                    owners = gating_plan(picks, cfg, lay[key], exp)
                    crop_owner[key] = owners
                    for i in sorted(set(owners)):
                        crop_futs[(key, i)] = asyncio.ensure_future(ask(
                            [image_part(crop(picks[i].image, lay[key], exp), max_side=768), text_part(q)], schema, mt))
            await asyncio.gather(*full_futs.values(), *crop_futs.values())
        except BaseException:
            for future in (*full_futs.values(), *crop_futs.values()):
                future.cancel()
            raise
        if audit is not None:
            audit["gating"] = {"frames": len(picks), "full_asked": len(full_futs),
                               **{f"{k}_asked": len(set(v)) for k, v in crop_owner.items()}}
        return [assemble(full_futs[full_owner[i]].result(),
                         [(key, crop_futs[(key, crop_owner[key][i])].result()) for key in crop_owner], frame.t)
                for i, frame in enumerate(picks)]

    if cfg.get("change_gating"):
        states = await gated()
    else:
        states = list(await asyncio.gather(*(one(f) for f in picks)))
    if not any(set(s) - {"t"} for s in states):
        raise RuntimeError("fast_scene_observations_unavailable")
    return states


def _b(state: dict[str, Any], key: str) -> bool:
    value = state.get(key)
    return value is True or value == "yes"


def _smooth(states: list[dict[str, Any]], key: str, radius: int = 1) -> list[bool]:
    raw = [_b(s, key) for s in states]
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


# --------------------------------------------------------------------------- rules
def _presence(states, seat_key, frame_key):
    if any(seat_key in s for s in states):
        return [_b(s, seat_key) if seat_key in s else _b(s, frame_key) for s in states]
    return [_b(s, frame_key) for s in states]


def _smooth_list(raw: list[bool], radius: int = 1) -> list[bool]:
    out = []
    for i in range(len(raw)):
        window = raw[max(0, i - radius): i + radius + 1]
        out.append(sum(window) * 2 > len(window))
    return out


def _desk(state, key):
    desk = state.get("desk") or {}
    return bool(desk.get(key))


def rule_candidates(states: list[dict[str, Any]], cfg: dict[str, Any] | None = None) -> tuple[list[Candidate], list[float]]:
    """Return candidates and the times at which the cashier entered (for the door check)."""
    cfg = cfg or {}
    if len(states) < 3:
        return [], []
    ts = [s["t"] for s in states]
    n = len(states)
    # presence = chair crop OR full-frame judgement (chair crops miss people who stand)
    cashier = _smooth_list([_b(s, "cashier_seat") or _b(s, "cashier_at_desk") for s in states])
    client = _smooth_list([_b(s, "client_seat") or _b(s, "client_at_desk") for s in states])
    in_room = _smooth_list([_b(s, "cashier_in_room") or _b(s, "cashier_seat") for s in states])
    # anything lying on the desk that belongs to an operation (cash, cash bag, coins, documents)
    on_desk = _smooth_list([_desk(s, "cash") or _desk(s, "cash_bag") or _desk(s, "coins_or_valuables")
                            or _desk(s, "id_or_document") for s in states])
    out: list[Candidate] = []
    entries: list[float] = []

    # --- cashier leaves the workplace (absence lasting to the end, or leaving the room)
    min_away = int(cfg.get("min_away", 3))
    for (a, b) in _runs([not c for c in cashier]):
        if a == 0:
            # an entry = the cashier was outside the room at the start and then sat down
            if 1 <= b < n - 1 and not in_room[0] and not in_room[min(1, n - 1)]:
                entries.append(ts[b + 1])
            continue
        to_end = b >= n - 2
        left_room = any(not in_room[j] for j in range(a, b + 1))
        if not (to_end or left_room) or (b - a + 1) < min_away:
            continue
        span = states[a:b + 1]
        tail = span[len(span) // 2:] if len(span) >= 4 else span
        anchor = ts[a]
        lap = [s.get("laptop") for s in tail if s.get("laptop") not in (None, "not_visible")]
        if lap and sum(v == "content" for v in lap) * 2 > len(lap):
            out.append(Candidate("WORKSTATION_UNLOCKED", anchor, "rule", "laptop shows working content after cashier left", 0.8))
        if sum(_b(s, "valuables_on_desk") or _desk(s, "id_or_document") for s in tail) * 2 > len(tail):
            out.append(Candidate("CASH_LEFT_UNATTENDED", anchor, "rule", "cash/valuables on desk after cashier left", 0.8))

    # --- client sessions
    sessions = [r for r in _runs(client) if r[1] - r[0] >= 2]
    for idx, (a, b) in enumerate(sessions):
        lo = max(0, a - 2)
        sess = states[lo:b + 1]
        cash_t = [s["t"] for s in sess if _b(s, "cash_in_hands") or _desk(s, "cash")]
        bag_t = [s["t"] for s in sess if _b(s, "cash_bag")]
        id_t = [s["t"] for s in sess if _b(s, "id_card_shown")]
        if bag_t:
            if not id_t:
                # anchor: the handover = bag on the desk while the visitor is at the desk
                handover = [s["t"] for s in states[a:b + 1] if _desk(s, "cash_bag")] or \
                           [t for t in bag_t if t >= ts[a]] or bag_t
                out.append(Candidate("COLLECTOR_NO_DOCUMENTS", handover[0], "rule", "cash bag accepted, no document seen", 0.5))
        elif cash_t:
            if not any(t < cash_t[0] for t in id_t):
                out.append(Candidate("NO_CLIENT_ID", cash_t[0], "rule", "no ID shown before the cash", 0.5))
        # counting after the client left (before anyone else arrives)
        end = sessions[idx + 1][0] if idx + 1 < len(sessions) else n
        post = list(range(b + 1, min(end, b + 16)))
        counting = [i for i in post if (_b(states[i], "counting_machine") or _b(states[i], "hand_counting")) and cashier[i]]
        if cash_t and len(counting) >= 2:
            out.append(Candidate("COUNT_WITHOUT_CLIENT", ts[counting[0]], "rule", "counting after the client left", 0.5))
        # previous operation's cash still on the desk when the next client arrives
        if idx + 1 < len(sessions):
            nxt = sessions[idx + 1][0]
            gap = list(range(b + 1, min(n, nxt + 2)))
            if (cash_t or bag_t) and gap and sum(on_desk[j] for j in gap) * 2 >= len(gap):
                out.append(Candidate("PREVIOUS_CASH_ON_DESK", ts[nxt], "rule", "objects of previous operation on desk at next arrival", 0.5))
        # third person during an established session
        crowd = [int(states[i].get("other_people") or 0) >= 2 for i in range(n)]
        runs = [r for r in _runs([crowd[i] and a + 2 <= i <= b for i in range(n)]) if r[1] - r[0] >= 2]
        if runs:
            out.append(Candidate("UNAUTHORIZED_PERSON", ts[runs[0][0]], "rule", "extra person during client session", 0.4))

    # --- personal items / drink at the workplace (persisting)
    for key in ("drink", "personal_items"):
        flags = _smooth(states, key)
        runs = [r for r in _runs(flags) if r[1] - r[0] >= 2]
        if runs:
            out.append(Candidate("PERSONAL_ITEMS", ts[runs[0][0]], "rule", f"{key} visible", 0.5))
    return out, entries


def _changed_fraction(img_a, img_b, box, upper: float = 0.6, thr: int = 25) -> float:
    """Fraction of pixels that changed in the upper part of a box (PIL only; robust to seated people)."""
    from PIL import ImageChops, ImageFilter
    w, h = img_a.size
    x1, y1, x2, y2 = box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000
    region = (int(x1), int(y1), int(x2), int(y1 + (y2 - y1) * upper))
    if region[2] - region[0] < 8 or region[3] - region[1] < 8:
        return 0.0
    ga = img_a.crop(region).convert("L").filter(ImageFilter.GaussianBlur(4))
    gb = img_b.crop(region).convert("L").filter(ImageFilter.GaussianBlur(4))
    diff = ImageChops.difference(ga, gb).point(lambda v: 255 if v > thr else 0)
    hist = diff.histogram()
    total = sum(hist)
    return hist[255] / total if total else 0.0


async def door_check(client: VLMClient, video: DecodedVideo, layout: dict[str, list[float]],
                     entry_t: float, cfg: dict[str, Any] | None = None) -> Candidate | None:
    """The cashier entered: is the door region still different from the opening frame (door left open)?"""
    cfg = cfg or {}
    if "door" not in layout:
        return None
    ref = video.frames[0].image
    tests = [video.nearest(min(entry_t + 4.0, video.duration)).image, video.frames[-1].image]
    thr = float(cfg.get("door_change_threshold", 0.2))
    changed = [_changed_fraction(ref, im, layout["door"]) for im in tests]
    whole = [_changed_fraction(ref, im, [0, 0, 1000, 1000], upper=1.0) for im in tests]
    if all(c >= thr for c in changed) and all(w < 0.6 for w in whole):
        return Candidate("CASH_ROOM_DOOR_OPEN", entry_t, "rule",
                         f"door region changed {min(changed):.2f} vs opening frame", 0.6)
    return None


# --------------------------------------------------------------------------- holistic
def _uniform(video: DecodedVideo, count: int) -> list[Frame]:
    frames = video.frames
    if len(frames) <= count:
        return frames
    step = (len(frames) - 1) / (count - 1)
    return [frames[round(i * step)] for i in range(count)]


async def holistic(client: VLMClient, video: DecodedVideo, cfg: dict[str, Any],
                   state_digest: str | None = None) -> list[Candidate]:
    frames = _uniform(video, int(cfg["holistic_frames"]))
    content: list[dict[str, Any]] = []
    for f in frames:
        content.append(text_part(f"t={f.t:.1f}s"))
        content.append(image_part(f.image, max_side=cfg["holistic_max_side"]))
    prompt = P.HOLISTIC_PROMPT.format(n=len(frames), duration=video.duration, checklist=P.checklist())
    if state_digest:
        prompt += "\nPer-second automatic observations (may contain errors):\n" + state_digest
    content.append(text_part(prompt))
    try:
        data = await client.chat_json(content, system=P.SYSTEM, max_tokens=700,
                                      json_schema=P.HOLISTIC_SCHEMA)
    except Exception:
        return []
    out = []
    for item in data.get("violations", []):
        if item.get("type_id") in BY_ID:
            t = min(max(float(item.get("t_sec", 0.0)), 0.0), video.duration)
            out.append(Candidate(item["type_id"], t, "holistic", str(item.get("observation", ""))[:300], 0.5))
    return out


# --------------------------------------------------------------------------- verify
async def verify(client: VLMClient, video: DecodedVideo, cand: Candidate, cfg: dict[str, Any],
                 layout: dict[str, list[float]] | None = None) -> dict[str, Any]:
    vt = BY_ID[cand.type_id]
    w = float(cfg["verify_window"])
    start = max(0.0, float(cand.evidence.get("start_sec", cand.t - w)))
    end = min(video.duration, float(cand.evidence.get("end_sec", cand.t + w)))
    if not math.isfinite(start) or not math.isfinite(end) or start > end:
        raise ValueError("fast_verification_interval_invalid")
    window = video.between(start, end) or [video.nearest(cand.t)]
    count = int(cfg["verify_frames"])
    if len(window) > count:
        step = (len(window) - 1) / (count - 1)
        window = [window[round(i * step)] for i in range(count)]
    content: list[dict[str, Any]] = []
    for f in window:
        content.append(text_part(f"t={f.t:.1f}s"))
        content.append(image_part(f.image, max_side=cfg["verify_max_side"]))
    if layout:
        parts, _ = crop_parts(video.nearest(cand.t + cfg.get("verify_crop_offset", 1.0)), layout)
        content.extend(parts)
    for snapshot in cand.evidence.get("lock_snapshots", [])[:2]:
        from forte.fast.specialists import validate_box

        box = validate_box(snapshot["box"])
        at = float(snapshot["t"])
        if not math.isfinite(at) or not 0 <= at <= video.duration:
            raise ValueError("fast_verification_snapshot_time_invalid")
        original = video.nearest(at)
        content.append(text_part(f"Original lock close-up, t={original.t:.3f}s; enlargement adds no detail."))
        content.append(image_part(crop(original.image, [v * 1000 for v in box], expand=3.0), max_side=768))
    content.append(text_part(P.VERIFY_PROMPT.format(n=len(window), t=cand.t, cue=vt.cue_en, correct=vt.correct_en)))
    if cand.source == "specialist":
        content.append(text_part(
            "Confirm only what the original pixels support. A detector proposal is not proof. "
            "For a missing action, require the full transaction and clearly visible hands and objects; "
            "occlusion or an unreadable small object means visible=false. Do not infer cash amounts."
        ))
    try:
        if cfg.get("verify_concise"):
            # Same prompt and response grammar as the full path; generation stops once the
            # decision fields (visible, confidence, t_sec) are complete, so greedy decoding
            # produces the same decision tokens without the free-text sentence.
            data = await client.chat_json(content, system=P.SYSTEM, max_tokens=300, json_schema=P.VERIFY_SCHEMA,
                                          stop=[P.VERIFY_STOP], truncated=True,
                                          validate_schema=P.VERIFY_DECISION_SCHEMA)
            data.pop("observation_ru", None)
        else:
            data = await client.chat_json(content, system=P.SYSTEM, max_tokens=300, json_schema=P.VERIFY_SCHEMA)
    except Exception as exc:
        data = {"visible": False, "confidence": 0.0, "t_sec": cand.t, "observation_ru": "", "error": type(exc).__name__}
    data["type_id"] = cand.type_id
    data["cand_t"] = cand.t
    data["source"] = cand.source
    return data


def state_digest(states: list[dict[str, Any]]) -> str:
    keys = ["cashier_at_desk", "cashier_in_room", "client_at_desk", "cash_on_desk", "cash_in_hands",
            "cash_bag", "id_card_shown", "door_open", "drink", "personal_items", "counting_machine", "detector"]
    lines = []
    for s in states:
        flags = [k for k in keys if _b(s, k)]
        lines.append(f"{s['t']:.0f}s: laptop={s.get('laptop','?')} people={s.get('other_people','?')} " + ",".join(flags))
    return "\n".join(lines)


def _merge(events: list[Event], window: float) -> list[Event]:
    events = sorted(events, key=lambda e: (e.type_id, e.t))
    merged: list[Event] = []
    for e in events:
        if merged and merged[-1].type_id == e.type_id and abs(merged[-1].t - e.t) <= window:
            keep = merged[-1] if merged[-1].confidence >= e.confidence else e
            keep.sources = sorted(set(merged[-1].sources) | set(e.sources))
            merged[-1] = keep
        else:
            merged.append(e)
    return sorted(merged, key=lambda e: e.t)


async def analyze(client: VLMClient, video: DecodedVideo, cfg: dict[str, Any] | None = None, *,
                  specialist_analysis: Any = None, priority: int | None = None) -> Analysis:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    client = _CheckedClient(client, priority if cfg.get("job_priority") and priority is not None else None,
                            strict_model_protocol=bool(cfg.get("strict_model_protocol")))
    if specialist_analysis is not None and specialist_analysis.approved_types:
        from forte.fast.policy import analyzer_policy_sha256

        provenance = specialist_analysis.provenance
        if not provenance.get("evaluation_only") and provenance.get("analyzer_policy_sha256") != analyzer_policy_sha256(cfg):
            raise ValueError("specialist_promoted_analyzer_policy_mismatch")
    timings: dict[str, float] = {}
    state_audit: dict[str, Any] = {}
    t0 = time.monotonic()
    if cfg.get("parallel_layout") and cfg.get("use_crops", True):
        async def timed_layout():
            found = await locate_layout(client, video, cfg)
            timings["layout"] = time.monotonic() - t0
            return found

        layout_task = asyncio.ensure_future(timed_layout())
        try:
            states = await extract_states(client, video, cfg, layout_task=layout_task, audit=state_audit)
        finally:
            if not layout_task.done():
                layout_task.cancel()
        layout = layout_task.result() if not layout_task.cancelled() else {}
    else:
        layout = await locate_layout(client, video, cfg) if cfg.get("use_crops", True) else {}
        timings["layout"] = time.monotonic() - t0
        states = await extract_states(client, video, cfg, layout, audit=state_audit)
    timings["states"] = time.monotonic() - t0
    cands: list[Candidate] = []
    audit: dict[str, Any] = {"suppressed": [], "specialists_enabled": specialist_analysis is not None, **state_audit}
    if cfg["use_rules"]:
        rc, entries = rule_candidates(states, cfg)
        cands.extend(rc)
        for et in entries[:2]:
            dc = await door_check(client, video, layout, et, cfg)
            if dc:
                cands.append(dc)
    if cfg["use_holistic"]:
        t1 = time.monotonic()
        cands.extend(await holistic(client, video, cfg, state_digest(states)))
        timings["holistic"] = time.monotonic() - t1
    approved_types: set[str] = set()
    if specialist_analysis is not None:
        from forte.fast.temporal_events import specialist_candidates

        t1 = time.monotonic()
        proposals, specialist_audit = specialist_candidates(specialist_analysis, states, layout, cfg)
        audit["specialists"] = specialist_audit
        audit["specialist_provenance"] = specialist_analysis.provenance
        approved_types = set(specialist_analysis.approved_types)
        for proposal in proposals:
            if proposal.type_id in BY_ID and proposal.type_id in approved_types:
                cands.append(Candidate(proposal.type_id, proposal.t, "specialist", proposal.note,
                                       proposal.score, evidence=proposal.evidence))
        timings["temporal_events"] = time.monotonic() - t1
    disabled = set(cfg.get("disabled_types") or [])
    protected = set(DEFAULT_CONFIG["disabled_types"])
    filtered = []
    for c in cands:
        promoted = c.source == "specialist" and c.type_id in approved_types
        if (c.type_id in disabled or c.type_id in protected) and not promoted:
            audit["suppressed"].append({"type_id": c.type_id, "t": c.t, "reason": "type_not_promoted"})
        else:
            filtered.append(c)
    cands = filtered
    # de-duplicate candidates of the same type close in time before verification
    uniq: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.type_id, c.t)):
        if uniq and uniq[-1].type_id == c.type_id and abs(uniq[-1].t - c.t) <= cfg["merge_window"]:
            if c.source == "specialist":
                uniq[-1] = c
            elif uniq[-1].source != "specialist":
                uniq[-1].source = uniq[-1].source + "+" + c.source if c.source not in uniq[-1].source else uniq[-1].source
            continue
        uniq.append(c)
    verifications: list[dict[str, Any]] = []
    events: list[Event] = []
    needs_verify = [c for c in uniq if cfg["use_verify"] or cfg.get("require_corroboration") or c.source == "specialist"]
    if needs_verify:
        t2 = time.monotonic()
        verifications = list(await asyncio.gather(*(verify(client, video, c, cfg, layout) for c in needs_verify)))
        timings["verify"] = time.monotonic() - t2
        for c, v in zip(needs_verify, verifications):
            if v.get("error"):
                raise RuntimeError("fast_verification_unavailable")
            confidence = float(v.get("confidence", 0))
            if v.get("visible") is True and math.isfinite(confidence) and cfg["verify_threshold"] <= confidence <= 1:
                t = float(v.get("t_sec", c.t))
                if not math.isfinite(t) or not (0 <= t <= video.duration) or abs(t - c.t) > cfg["verify_window"] + 1:
                    t = c.t
                observed = (c.note_ru or c.note) if cfg.get("verify_concise") else str(v.get("observation_ru", ""))
                events.append(Event(c.type_id, t, observed, float(v["confidence"]), c.source.split("+")))
            else:
                audit["suppressed"].append({"type_id": c.type_id, "t": c.t, "reason": "vlm_disagrees_or_unclear"})
    events.extend(Event(c.type_id, c.t, c.note_ru or c.note, c.score, c.source.split("+"))
                  for c in uniq if c not in needs_verify)
    events = _merge(events, cfg["merge_window"])
    if client.failures:
        raise RuntimeError("fast_model_observation_failed")
    timings["total"] = time.monotonic() - t0
    return Analysis(events, states, uniq, verifications, timings, layout, audit)
