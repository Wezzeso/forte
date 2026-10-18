"""Unit tests for the fast analyzer rules and result assembly (no model, no GPU)."""

from __future__ import annotations

from PIL import Image, ImageDraw

from forte.fast.analyzer import Analysis, Event, _changed_fraction, rule_candidates
from forte.fast.catalog import BY_ID, TYPES
from forte.fast.service import build_result, describe
from forte.fast.video import DecodedVideo, Frame
from forte.shared.schemas import CompletedResponse


def _state(t, *, seat=True, room=True, laptop="content", desk=None, client=False, people=0, **extra):
    desk = desk or {}
    s = {"t": float(t), "cashier_seat": seat, "cashier_at_desk": seat, "cashier_in_room": room,
         "client_seat": client, "client_at_desk": client, "other_people": people, "laptop": laptop,
         "desk": {"cash": False, "cash_bag": False, "coins_or_valuables": False, "drink": False,
                  "coat_or_handbag": False, "id_or_document": False, **desk}}
    s["valuables_on_desk"] = bool(s["desk"]["cash"] or s["desk"]["cash_bag"] or s["desk"]["coins_or_valuables"])
    s["cash_on_desk"] = s["desk"]["cash"]
    s.update(extra)
    return s


def _types(states):
    cands, _ = rule_candidates(states)
    return {c.type_id for c in cands}


def test_unlocked_workstation_after_departure():
    states = [_state(t) for t in range(6)] + [_state(t, seat=False, room=t < 9) for t in range(6, 14)]
    assert "WORKSTATION_UNLOCKED" in _types(states)


def test_lock_screen_is_not_a_violation():
    states = [_state(t) for t in range(6)] + \
             [_state(t, seat=False, room=t < 9, laptop="lock_screen") for t in range(6, 14)]
    assert _types(states) == set()


def test_cash_left_on_desk_after_departure():
    states = [_state(t, laptop="closed_or_off") for t in range(5)] + \
             [_state(t, seat=False, room=t < 8, laptop="closed_or_off", desk={"cash": True}) for t in range(5, 12)]
    assert "CASH_LEFT_UNATTENDED" in _types(states)


def test_short_trip_to_safe_is_ignored():
    # away from the seat but still in the room and back after 3 s: not a departure
    states = [_state(t, laptop="closed_or_off") for t in range(5)] + \
             [_state(t, seat=False, laptop="closed_or_off", desk={"cash": True}) for t in range(5, 8)] + \
             [_state(t, laptop="closed_or_off") for t in range(8, 14)]
    assert "CASH_LEFT_UNATTENDED" not in _types(states)


def test_previous_cash_when_next_client_arrives():
    s1 = [_state(t, laptop="closed_or_off", client=True, cash_in_hands=t == 3) for t in range(0, 8)]
    gap = [_state(t, laptop="closed_or_off", desk={"cash": True}) for t in range(8, 12)]
    s2 = [_state(t, laptop="closed_or_off", client=True, desk={"cash": True}) for t in range(12, 18)]
    cands, _ = rule_candidates(s1 + gap + s2)
    prev = [c for c in cands if c.type_id == "PREVIOUS_CASH_ON_DESK"]
    assert prev and prev[0].t == 12.0


def test_door_change_fraction():
    closed = Image.new("RGB", (200, 200), (200, 200, 200))
    opened = closed.copy()
    ImageDraw.Draw(opened).rectangle([0, 0, 100, 120], fill=(40, 40, 40))
    door = [0, 0, 500, 1000]
    assert _changed_fraction(closed, closed, door) == 0.0
    assert _changed_fraction(closed, opened, door) > 0.5


def test_catalog_has_grounded_citations():
    for t in TYPES:
        assert t.url.startswith("https://adilet.zan.kz/") and t.clause and t.rule_ru and t.name_ru


def test_build_result_matches_public_contract():
    img = Image.new("RGB", (64, 96), (120, 120, 120))
    video = DecodedVideo(duration=10.0, fps=30.0, width=64, height=96,
                         frames=[Frame(t=i * 0.5, image=img) for i in range(21)])
    analysis = Analysis(events=[Event("WORKSTATION_UNLOCKED", 7.0, "Кассир ушла.", 0.8, ["rule"])],
                        states=[], candidates=[], verifications=[], timings={})
    result = build_result("job123", video, analysis)
    validated = CompletedResponse.model_validate(result)
    assert validated.verdict == "violation"
    event = validated.violations[0]
    assert event.type == BY_ID["WORKSTATION_UNLOCKED"].name_ru
    assert 1 <= len(event.evidence_frames) <= 3
    assert event.evidence_frames[0].image_base64.startswith("/9j/")
    assert "пункт 71" in event.description

    empty = build_result("job124", video, Analysis([], [], [], [], {}))
    assert CompletedResponse.model_validate(empty).verdict == "no_violation"


def test_description_never_empty():
    assert describe("PERSONAL_ITEMS", "").startswith("На кадрах")
