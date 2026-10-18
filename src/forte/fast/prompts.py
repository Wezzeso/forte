"""Prompts and output schemas. Kept in one place so they can be tuned without code changes."""

from __future__ import annotations

from forte.fast.catalog import TYPES

SYSTEM = ("You are a precise visual inspector of CCTV footage from a bank cashier room. "
          "Report only what is clearly visible. Never identify people. Answer strictly in JSON.")

STATE_PROMPT = """This is one frame (t={t:.1f}s) from a fixed camera in a bank cash room.
Roles: the CASHIER works behind the desk with the laptop and the cash-counting machine;
a CLIENT or CASH COLLECTOR comes in through the door and sits/stands on the opposite side of the desk.
The room has GLASS walls: people seen through the glass (outside the room) or reflections do NOT count.
Answer each field about THIS frame only:
- cashier_at_desk: the cashier is seated at (or standing right at) the cashier side of the desk.
- cashier_in_room: the cashier is inside the room.
- client_at_desk: a person other than the cashier is seated/standing at the client side of the desk.
- other_people: number of people INSIDE the room other than the cashier.
- person_at_door: someone is in the doorway / entering / leaving through the door.
- cash_in_hands: someone is holding banknotes.
- cash_bag: a cloth cash/collection bag (sack) is visible on the desk or in someone's hands.
- id_card_shown: an ID card / passport / document is being handed over or held and looked at.
- counting_machine: banknotes are being put into or taken out of the cash-counting machine.
- detector: banknotes are being passed through the small banknote detector device.
- hand_counting: the cashier is counting banknotes by hand.
"""

LAYOUT_PROMPT = """This is a frame from a fixed camera in a bank cash room. Locate these objects and give
bounding boxes as [x1, y1, x2, y2] with coordinates normalized to 0-1000 (relative to image width/height).
Use null if an object is not visible.
- desk: the cashier's desk top (whole surface).
- laptop: the laptop computer (screen and keyboard).
- safe: the grey metal safe / cabinet with a lockable door.
- door: the room's entrance door (glass door with a handle).
- counting_machine: the black banknote counting machine.
- detector: the small banknote detector device (smaller than the counting machine).
- cashier_chair: the cashier's office chair behind the desk (on the side of the laptop and counting machine).
- client_chair: the visitor's chair on the opposite side of the desk from the cashier (null if there is none)."""

BOX = {"anyOf": [{"type": "null"},
                 {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}]}
LAYOUT_SCHEMA = {
    "type": "object",
    "properties": {k: BOX for k in ("desk", "laptop", "safe", "door", "counting_machine", "detector", "cashier_chair", "client_chair")},
    "required": ["desk", "laptop", "safe", "door", "counting_machine", "detector", "cashier_chair", "client_chair"],
    "additionalProperties": False,
}

CROPS_NOTE = """After the full frame you also get enlarged close-up crops of the same moment: {names}.
Use the close-ups for the laptop screen, the safe door/key, the door and the items on the desk."""

YESNO = {"type": "string", "enum": ["yes", "no", "unclear"]}
STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "cashier_at_desk": {"type": "boolean"},
        "cashier_in_room": {"type": "boolean"},
        "client_at_desk": {"type": "boolean"},
        "other_people": {"type": "integer", "minimum": 0, "maximum": 9},
        "person_at_door": {"type": "boolean"},
        "cash_in_hands": {"type": "boolean"},
        "cash_bag": {"type": "boolean"},
        "id_card_shown": {"type": "boolean"},
        "counting_machine": {"type": "boolean"},
        "detector": {"type": "boolean"},
        "hand_counting": {"type": "boolean"}
    },
    "required": ["cashier_at_desk", "cashier_in_room", "client_at_desk", "other_people", "person_at_door", "cash_in_hands", "cash_bag", "id_card_shown", "counting_machine", "detector", "hand_counting"],
    "additionalProperties": False,
}


def checklist(type_ids: list[str] | None = None) -> str:
    lines = []
    for t in TYPES:
        if type_ids and t.type_id not in type_ids:
            continue
        lines.append(f"- {t.type_id}: VIOLATION if: {t.cue_en} | NOT a violation if: {t.correct_en}")
    return "\n".join(lines)


HOLISTIC_PROMPT = """You are reviewing a short CCTV video from a bank cash room, given as {n} frames in time order;
each frame is preceded by its timestamp. Duration: {duration:.1f}s.
Check the cashier's work against this checklist of possible procedure violations:
{checklist}

Rules: report a violation only if it is clearly visible in the frames. Most videos show correct work.
Return JSON: {{"violations": [{{"type_id": "...", "t_sec": <seconds when it happens>, "observation": "<what is visible, 1 sentence>"}}]}}.
Return {{"violations": []}} if the cashier's actions look correct."""

HOLISTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "violations": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "type_id": {"type": "string", "enum": [t.type_id for t in TYPES]},
                    "t_sec": {"type": "number"},
                    "observation": {"type": "string", "maxLength": 300},
                },
                "required": ["type_id", "t_sec", "observation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["violations"],
    "additionalProperties": False,
}

VERIFY_PROMPT = """These {n} frames (with timestamps) come from a CCTV video of a bank cash room, around t={t:.1f}s.
Question: is the following procedure violation clearly visible here?
VIOLATION: {cue}
CORRECT behaviour (not a violation): {correct}
Answer JSON: {{"visible": true/false, "confidence": 0.0-1.0, "t_sec": <best timestamp of the violation>, "observation_ru": "<одно предложение на русском: что именно видно в кадре, без домыслов и без описания внешности>"}}"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "t_sec": {"type": "number"},
        "observation_ru": {"type": "string", "maxLength": 400},
    },
    "required": ["visible", "confidence", "t_sec", "observation_ru"],
    "additionalProperties": False,
}

# verify_concise: generation stops when the free-text key begins (decision fields complete).
VERIFY_STOP = '"observation'
VERIFY_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "t_sec": {"type": "number"},
        "observation_ru": {"type": "string", "maxLength": 400},
    },
    "required": ["visible", "confidence", "t_sec"],
    "additionalProperties": False,
}


# ---- focused sub-queries (one small question per crop works far better than one big form)
LAPTOP_Q = ("This is a close-up of a laptop screen. Which is it? A) a working application: spreadsheet, document, "
            "table or windows (mostly white/light with rows of text); B) a lock screen: a full-screen photo wallpaper "
            "(landscape, mountains, lake, sky; dark blue tones) possibly with a big clock; C) closed lid or black/off "
            "screen; D) the screen is not visible (blocked by a person or out of view). Answer with one letter.")
LAPTOP_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
                 "required": ["answer"], "additionalProperties": False}
LAPTOP_MAP = {"A": "content", "B": "lock_screen", "C": "closed_or_off", "D": "not_visible"}

DOOR_Q = ("Close-up of the entrance door of a room. Is the door open (door leaf swung away, gap or corridor visible "
          "through the doorway) or closed (leaf shut in its frame)? Answer open, closed or unclear.")
DOOR_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["open", "closed", "unclear"]}},
               "required": ["answer"], "additionalProperties": False}

DESK_Q = ("Close-up of a cashier's desk in a bank. Ignore the laptop, phone, counting machine and detector. "
          "Report what lies on the desk surface: banknotes/cash bundles, a cloth cash bag (sack), coin tubes/rolls, "
          "a cup or drink, a coat/jacket or handbag, an ID card or document. JSON booleans.")
DESK_SCHEMA = {"type": "object", "properties": {
    "cash": {"type": "boolean"}, "cash_bag": {"type": "boolean"}, "coins_or_valuables": {"type": "boolean"},
    "drink": {"type": "boolean"}, "coat_or_handbag": {"type": "boolean"}, "id_or_document": {"type": "boolean"}},
    "required": ["cash", "cash_bag", "coins_or_valuables", "drink", "coat_or_handbag", "id_or_document"],
    "additionalProperties": False}

SEAT_Q = ("Close-up of a chair area. Is a person sitting on this chair or standing right at it? "
          "People seen behind glass walls (outside the room) do not count. Answer yes or no.")
SEAT_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
               "required": ["answer"], "additionalProperties": False}

DOOR_CMP_Q = ("Both images show the same entrance door of a room from a fixed camera. Image 1 is from the start of "
              "the video when the door was closed. Image 2 is later. In Image 2, is the door closed again in the same "
              "position as in Image 1, or is it left open (door leaf swung away, a dark panel or gap in the doorway)? "
              "Ignore people. Answer open, closed or unclear.")
