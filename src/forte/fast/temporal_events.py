"""Conservative candidates from calibrated specialist tracks and explicit events.

Detections are positive observations only. In particular, a missing ID, key or
device contact is never an observed procedural omission. The three omission
rules need an explicit reviewed negative interval, complete visibility of all
required channels, transaction boundaries, and matching actor/object tracks.
These candidates still require the analyzer's independent VLM verification.

The generic detector currently provides neither transaction roles nor complete
visibility. Its normal output can propose persistent desk objects. Key candidates
can also use an explicit sustained cashier-away scene-state sequence, with key
tracks as the separate positive signal. The other rules abstain until an
event/visibility provider is connected. The
optional observations below are an input contract, not invented model outputs.
No torch dependency, media reads, label lookups or filename rules are used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_TYPES = frozenset({
    "PERSONAL_ITEMS", "KEY_LEFT_IN_SAFE", "NO_CLIENT_ID", "NO_AUTHENTICITY_CHECK",
    "NO_RECOUNT", "CASH_HANDOVER_IRREGULAR",
})
CASH_CLASSES = frozenset({"banknote_stack", "single_banknote"})
_EVENT_SOURCES = frozenset({"tracked_event", "human_verified", "verified_interval"})
_COVERAGE_SOURCES = frozenset({"human_verified", "validated_visibility_model"})


@dataclass(frozen=True)
class TemporalObservation:
    """A source-bound, observed action or explicitly reviewed negative interval.

    ``transaction_started`` means the beginning of the entire transaction was
    observed, including any ID exchange. Actor roles must come from an explicit
    role provider; a person count or a hand proximity signal cannot supply one.
    ``cash_split`` uses object_track_id for the parent and related_track_ids for
    the two child note sets. ``cash_retained_personally`` additionally requires
    destination ``personal_pocket`` or ``verified_personal_container``.
    """

    kind: str
    t: float
    end_t: float | None = None
    status: str = "present"  # present / absent / unknown
    confidence: float = 1.0
    transaction_id: str | None = None
    actor_track_id: int | None = None
    actor_role: str | None = None
    object_track_id: int | None = None
    related_track_ids: tuple[int, ...] = ()
    recipient_track_id: int | None = None
    recipient_role: str | None = None
    destination: str | None = None
    source: str = "unverified"

    def __post_init__(self) -> None:
        end = self.t if self.end_t is None else self.end_t
        if (isinstance(self.confidence, bool) or not math.isfinite(self.t) or not math.isfinite(end) or self.t < 0 or end < self.t
                or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("invalid temporal observation time or confidence")
        if self.status not in {"present", "absent", "unknown"}:
            raise ValueError("observation status must be present, absent or unknown")
        ids = (self.actor_track_id, self.object_track_id, self.recipient_track_id, *self.related_track_ids)
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
               for value in ids):
            raise ValueError("track IDs must be nonnegative integers")
        object.__setattr__(self, "end_t", end)


@dataclass(frozen=True)
class CoverageInterval:
    """Explicit visibility proof; false/unknown segments invalidate coverage.

    Channels: id_exchange, authenticity_check, counting, and object_track.
    ``complete`` also asserts adequate resolution, valid detector calibration,
    no occlusion, no dropped frames and no track identity switch in this segment.
    A frame timestamp or successful hand model call is not this assertion.
    """

    scope: str
    start_t: float
    end_t: float
    transaction_id: str
    actor_track_id: int
    object_track_id: int | None = None
    complete: bool = False
    source: str = "unverified"

    def __post_init__(self) -> None:
        if (not math.isfinite(self.start_t) or not math.isfinite(self.end_t)
                or self.start_t < 0 or self.end_t < self.start_t):
            raise ValueError("invalid coverage interval")
        if not isinstance(self.complete, bool):
            raise ValueError("coverage complete must be an explicit boolean")
        if (not self.transaction_id or isinstance(self.actor_track_id, bool)
                or not isinstance(self.actor_track_id, int) or self.actor_track_id < 0
                or self.object_track_id is not None and (isinstance(self.object_track_id, bool)
                    or not isinstance(self.object_track_id, int) or self.object_track_id < 0)):
            raise ValueError("coverage requires a transaction and actor track")


@dataclass(frozen=True)
class SpecialistCandidate:
    type_id: str
    t: float
    score: float
    note: str
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = "specialist_temporal"


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _frames(analysis: Any, layout: Mapping[str, Any] | None = None) -> list[Any]:
    frames = sorted(_get(analysis, "frames", ()), key=lambda f: _get(f, "t", -1))
    times = [_get(f, "t", -1) for f in frames]
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("specialist frame timestamps must be finite and nonnegative")
    if len(set(times)) != len(times):
        raise ValueError("duplicate specialist frame timestamps")
    if layout:
        from forte.fast.specialists import zones_from_layout

        zones = zones_from_layout(layout)
        annotated = []
        for frame in frames:
            detections = []
            for detection in _get(frame, "detections", ()):
                box = _get(detection, "box", ())
                names = _get(detection, "zones", ())
                if not names and len(box) == 4:
                    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    names = tuple(zone.name for zone in zones if zone.contains(center))
                detections.append({key: _get(detection, key) for key in
                                   ("class_name", "confidence", "box", "track_id", "class_id")}
                                  | {"zones": names})
            annotated.append({"t": _get(frame, "t"), "detections": detections,
                              "contacts": _get(frame, "contacts", ())})
        frames = annotated
    return frames


def filter_doorway_people(detections: Iterable[Any]) -> list[Any]:
    """Return positively localized interior people, never doorway-only people.

    This is only a geometric filter. It does not assign roles or declare anyone
    unauthorized; the candidate generator intentionally has no people-count rule.
    """
    interior = {"cash_room", "cashier_workstation", "cash_room_interior"}
    return [d for d in detections if _get(d, "class_name") == "person"
            and interior.intersection(_get(d, "zones", ())) and _get(d, "track_id") is not None]


def _track_records(frames: Sequence[Any], classes: set[str] | frozenset[str],
                   minimum: float) -> dict[int, list[tuple[float, Any]]]:
    tracks: dict[int, list[tuple[float, Any]]] = {}
    for frame in frames:
        for detection in _get(frame, "detections", ()):
            track_id = _get(detection, "track_id")
            confidence = _get(detection, "confidence", 0.0)
            if (_get(detection, "class_name") in classes and isinstance(track_id, int)
                    and not isinstance(track_id, bool) and track_id >= 0
                    and isinstance(confidence, (int, float)) and math.isfinite(confidence)
                    and minimum <= confidence <= 1):
                tracks.setdefault(track_id, []).append((_get(frame, "t"), detection))
    return tracks


def _bound(fact: TemporalObservation) -> bool:
    return bool(fact.source in _EVENT_SOURCES and fact.actor_track_id is not None
                and fact.actor_role == "cashier" and fact.confidence >= 0.9)


def _same(a: TemporalObservation, b: TemporalObservation, *, same_object: bool = True) -> bool:
    return bool(a.transaction_id and a.transaction_id == b.transaction_id
                and a.actor_track_id == b.actor_track_id and _bound(a) and _bound(b)
                and (not same_object or a.object_track_id is not None
                     and a.object_track_id == b.object_track_id))


def _covered(coverage: Sequence[CoverageInterval], anchor: TemporalObservation,
             scope: str, start: float, end: float, *, object_bound: bool = False) -> bool:
    relevant = [c for c in coverage if c.transaction_id == anchor.transaction_id
                and c.actor_track_id == anchor.actor_track_id and c.scope == scope
                and (not object_bound or c.object_track_id == anchor.object_track_id)
                and c.end_t >= start and c.start_t <= end]
    # Explicit occlusion/unknown always wins over another overlapping assertion.
    if any(c.complete is not True or c.source not in _COVERAGE_SOURCES for c in relevant):
        return False
    cursor = start
    for interval in sorted(relevant, key=lambda c: c.start_t):
        if interval.start_t > cursor + 1e-6:
            return False
        cursor = max(cursor, interval.end_t)
        if cursor >= end - 1e-6:
            return True
    return False


def _negative(facts: Sequence[TemporalObservation], anchor: TemporalObservation,
              kinds: set[str], start: float, end: float, *,
              required_absent_kinds: set[str] | None = None) -> TemporalObservation | None:
    relevant = [f for f in facts if f.kind in kinds and f.transaction_id == anchor.transaction_id
                and f.actor_track_id == anchor.actor_track_id and f.actor_role == "cashier"
                and f.t <= end and f.end_t >= start]
    if any(f.status != "absent" for f in relevant):
        return None
    return next((f for f in relevant if (required_absent_kinds is None or f.kind in required_absent_kinds)
                 and _bound(f) and f.t <= start and f.end_t >= end
                 and (f.object_track_id is None or f.object_track_id == anchor.object_track_id)), None)


def _evidence(facts: Sequence[TemporalObservation], *, tracks: Iterable[int] = ()) -> dict[str, Any]:
    return {"start_sec": min(f.t for f in facts), "end_sec": max(f.end_t for f in facts),
            "frame_times": sorted({f.t for f in facts} | {f.end_t for f in facts}),
            "track_ids": sorted(set(tracks) | {f.object_track_id for f in facts if f.object_track_id is not None}),
            "observations": [{"kind": f.kind, "t": f.t, "end_t": f.end_t,
                              "status": f.status, "source": f.source} for f in facts],
            "requires_independent_verification": True}


def _state_departures(states: Sequence[Mapping[str, Any]], max_gap: float,
                      minimum_away: float) -> list[tuple[TemporalObservation, dict[str, Any]]]:
    """Use explicit role states, never disappearances of anonymous person IDs.

    This proof is limited to departure. It asserts no visibility of hands, cash,
    ID exchange or device use and cannot support any omission rule.
    """
    role_keys = ("cashier_in_room", "cashier_at_desk", "cashier_seat")
    ordered = sorted(states, key=lambda s: s.get("t", -1))
    output = []
    for index in range(1, len(ordered)):
        previous, start = ordered[index - 1], ordered[index]
        t0, t1 = previous.get("t"), start.get("t")
        if (not isinstance(t0, (float, int)) or not isinstance(t1, (float, int))
                or not math.isfinite(t0) or not math.isfinite(t1) or t0 < 0
                or not 0 < t1 - t0 <= max_gap
                or not all(previous.get(key) is True for key in role_keys)
                or not all(start.get(key) is False for key in role_keys)):
            continue
        support = [previous]
        last_t = t0
        for state in ordered[index:]:
            at = state.get("t")
            if (not isinstance(at, (float, int)) or not math.isfinite(at)
                    or not 0 < at - last_t <= max_gap
                    or not all(state.get(key) is False for key in role_keys)):
                break
            support.append(state)
            last_t = at
            if at - t1 < minimum_away:
                continue
            # No synthetic person/actor track ID is assigned from scene states.
            departure = TemporalObservation("cashier_departed", float(t1), end_t=float(at),
                                            actor_role="cashier", confidence=.9,
                                            source="vlm_state_sequence")
            evidence = _evidence([departure])
            evidence["state_support"] = [{"t": state["t"], **{key: state[key] for key in role_keys}}
                                         for state in support]
            output.append((departure, evidence))
            break
    return output


def specialist_candidates(analysis: Any, states: Sequence[Mapping[str, Any]] = (),
                          layout: Mapping[str, Any] | None = None, cfg: Mapping[str, Any] | None = None,
                          *, observations: Iterable[TemporalObservation] = (),
                          coverage: Iterable[CoverageInterval] = ()) -> tuple[list[SpecialistCandidate], list[dict[str, Any]]]:
    """Propose gated events, with static reasons for every unsupported type.

    ``states`` may supply an explicit sustained cashier-departure role sequence
    for the key rule. It cannot supply complete-visibility or omission facts.
    Config may provide explicitly constructed ``specialist_observations`` and
    ``specialist_coverage``; plain dictionaries are not a visibility provider.
    Every proposed event still needs independent VLM confirmation downstream.
    """
    cfg = cfg or {}
    frames = _frames(analysis, layout)
    approved = SUPPORTED_TYPES.intersection(_get(analysis, "approved_types", ()))
    classes = frozenset(_get(analysis, "approved_classes", ()))
    facts = tuple(observations) + tuple(cfg.get("specialist_observations", ()))
    covers = tuple(coverage) + tuple(cfg.get("specialist_coverage", ()))
    if any(not isinstance(f, TemporalObservation) for f in facts):
        raise ValueError("specialist observations require TemporalObservation instances")
    if any(not isinstance(c, CoverageInterval) for c in covers):
        raise ValueError("specialist coverage requires CoverageInterval instances")
    facts = tuple(sorted(facts, key=lambda f: (f.t, f.kind)))
    max_gap = float(cfg.get("specialist_max_gap_s", 1.1))
    minimum = float(cfg.get("specialist_object_confidence", 0.9))
    if not math.isfinite(max_gap) or max_gap <= 0 or not 0.9 <= minimum <= 1:
        raise ValueError("invalid specialist continuity/confidence threshold")
    candidates: list[SpecialistCandidate] = []
    reasons: dict[str, str] = {kind: "type_not_approved_by_gate" for kind in SUPPORTED_TYPES - approved}

    def emit(kind: str, at: float, support: Sequence[TemporalObservation], note: str,
             *, tracks: Iterable[int] = ()) -> None:
        candidates.append(SpecialistCandidate(kind, at, min(f.confidence for f in support),
                                               note, _evidence(support, tracks=tracks)))

    def starts(anchor: TemporalObservation) -> list[TemporalObservation]:
        return [f for f in facts if f.kind == "transaction_started" and f.status == "present"
                and _same(anchor, f, same_object=False) and f.end_t <= anchor.t]

    def ends(anchor: TemporalObservation) -> list[TemporalObservation]:
        return [f for f in facts if f.kind == "transaction_completed" and f.status == "present"
                and _same(anchor, f, same_object=False) and f.t >= anchor.end_t]

    # A single light patch is not a cup: require one calibrated object track,
    # explicit desk zone, >=3 observations spanning >=1 second without a gap.
    if "PERSONAL_ITEMS" in approved:
        reasons["PERSONAL_ITEMS"] = "no_persistent_calibrated_desk_object"
        desk_classes = classes.intersection({"cup", "coat", "coat_or_bag", "personal_bag"})
        tracks = _track_records(frames, desk_classes, minimum)
        for track_id, records in tracks.items():
            run: list[tuple[float, Any]] = []
            for record in records:
                zones = set(_get(record[1], "zones", ()))
                if not zones.intersection({"desk", "cash_desk", "work_surface"}):
                    run = []
                    continue
                if run and (record[0] - run[-1][0] > max_gap
                            or _get(record[1], "class_name") != _get(run[-1][1], "class_name")):
                    run = []
                run.append(record)
                if len(run) < 3 or run[-1][0] - run[0][0] < 1.0:
                    continue
                cls = _get(record[1], "class_name")
                evidence = {"start_sec": run[0][0], "end_sec": run[-1][0],
                            "frame_times": [r[0] for r in (run[0], run[len(run)//2], run[-1])],
                            "track_ids": [track_id], "class_name": cls,
                            "boxes": [list(_get(r[1], "box", ())) for r in (run[0], run[-1])],
                            "requires_independent_verification": True}
                candidates.append(SpecialistCandidate("PERSONAL_ITEMS", run[0][0],
                    min(_get(r[1], "confidence") for r in run),
                    "drink: persistent tracked cup on desk" if cls == "cup" else
                    "personal_items: persistent tracked coat or personal bag on desk", evidence))
                break

    if "KEY_LEFT_IN_SAFE" in approved:
        reasons["KEY_LEFT_IN_SAFE"] = "requires_explicit_cashier_departure_and_resolved_lock_before_after"
        key_tracks = _track_records(frames, classes.intersection({"key_in_lock"}), minimum)
        away = float(cfg.get("specialist_min_away_s", 3.0))
        if not math.isfinite(away) or away < 3:
            raise ValueError("specialist departure duration must be at least three seconds")
        departures = [(f, _evidence([f])) for f in facts if f.kind == "cashier_departed"
                      and f.status == "present" and _bound(f)]
        departures.extend(_state_departures(states, max_gap, away))
        for departure, departure_evidence in departures:
            for track_id, records in key_tracks.items():
                before = [(t, d) for t, d in records if departure.t - max_gap <= t < departure.t]
                after_boundary = max(departure.t + .5, departure.end_t)
                after = [(t, d) for t, d in records if departure.t <= t <= after_boundary + max_gap]
                if not before or len(after) < 2 or after[-1][0] < after_boundary:
                    continue
                if any(f.kind == "cashier_returned" and f.status == "present" and _bound(f)
                       and f.actor_track_id == departure.actor_track_id
                       and departure.t < f.t <= after[-1][0] for f in facts):
                    continue
                if any(departure.t < state.get("t", -1) <= after[-1][0]
                       and any(state.get(key) is True for key in
                               ("cashier_in_room", "cashier_at_desk", "cashier_seat")) for state in states):
                    continue
                sequence = before[-1:] + after
                if any(b[0] - a[0] > max_gap for a, b in zip(sequence, sequence[1:])):
                    continue
                if any(not set(_get(d, "zones", ())).intersection({"safe", "safe_lock"}) for _, d in sequence):
                    continue
                # Stable track IDs alone cannot excuse a key moving to another
                # part of a large safe zone. Compare the resolved lock region.
                first_box = _get(sequence[0][1], "box", ())
                if len(first_box) != 4:
                    continue
                cx, cy = (first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2
                tolerance = 2 * max(first_box[2] - first_box[0], first_box[3] - first_box[1], .01)
                moved = False
                for _, detection in sequence[1:]:
                    box = _get(detection, "box", ())
                    if (len(box) != 4 or ((box[0] + box[2]) / 2 - cx) ** 2
                            + ((box[1] + box[3]) / 2 - cy) ** 2 > tolerance ** 2):
                        moved = True
                        break
                if moved:
                    continue
                evidence = dict(departure_evidence)
                evidence["track_ids"] = [track_id]
                evidence["start_sec"] = sequence[0][0]
                evidence["end_sec"] = sequence[-1][0]
                evidence["frame_times"] = [sequence[0][0], departure.t, sequence[-1][0]]
                evidence["boxes"] = [list(_get(d, "box", ())) for _, d in sequence]
                evidence["lock_snapshots"] = [{"t": t, "box": list(_get(d, "box", ()))}
                                              for t, d in (sequence[0], sequence[-1])]
                candidates.append(SpecialistCandidate("KEY_LEFT_IN_SAFE", departure.t,
                    min(departure.confidence, *(float(_get(d, "confidence")) for _, d in sequence)),
                    "key positively resolved in same safe-lock track before and after tracked cashier departure", evidence))

    # Omission rules require an explicit negative assertion, not empty arrays.
    if "NO_CLIENT_ID" in approved:
        reasons["NO_CLIENT_ID"] = "requires_complete_transaction_id_visibility_and_reviewed_negative"
        if {"id_card", "banknote_stack"}.issubset(classes):
            for received in facts:
                if received.kind != "cash_received" or received.status != "present" or received.object_track_id is None:
                    continue
                for start in starts(received):
                    for end in ends(received):
                        negative = _negative(facts, received, {"id_checked", "id_presented"}, start.t, end.t,
                                             required_absent_kinds={"id_checked"})
                        card_seen = any(_get(d, "class_name") == "id_card"
                                        for frame in frames if start.t <= _get(frame, "t") <= end.t
                                        for d in _get(frame, "detections", ()))
                        if (negative and not card_seen and _covered(covers, received, "id_exchange", start.t, end.t)
                                and _covered(covers, received, "object_track", received.t, end.t, object_bound=True)):
                            emit("NO_CLIENT_ID", received.t, [start, received, end, negative],
                                 "cash accepted during a fully observed transaction with explicitly reviewed ID-check absence")

    if "NO_AUTHENTICITY_CHECK" in approved:
        reasons["NO_AUTHENTICITY_CHECK"] = "requires_completed_count_complete_device_visibility_and_reviewed_negative"
        if "detector_device" in classes and classes.intersection(CASH_CLASSES):
            for counted in facts:
                if counted.kind != "manual_count_completed" or counted.status != "present" or counted.object_track_id is None:
                    continue
                for start in starts(counted):
                    for end in ends(counted):
                        negative = _negative(facts, counted, {"authenticity_check"}, start.t, end.t)
                        # Proximity is insufficient to prove a check, but introduces
                        # doubt and therefore blocks an assertion that none occurred.
                        contact = any(_get(c, "class_name") == "detector_device"
                                      for frame in frames if start.t <= _get(frame, "t") <= end.t
                                      for c in _get(frame, "contacts", ()))
                        if (negative and not contact
                                and _covered(covers, counted, "authenticity_check", start.t, end.t)
                                and _covered(covers, counted, "object_track", start.t, end.t, object_bound=True)):
                            emit("NO_AUTHENTICITY_CHECK", end.t, [start, counted, end, negative],
                                 "tracked cash manually counted and operation completed; fully visible authenticity-check interval explicitly reviewed absent")

    if "NO_RECOUNT" in approved:
        reasons["NO_RECOUNT"] = "requires_same_bundle_bag_to_safe_complete_counting_visibility_and_reviewed_negative"
        if {"cash_bag", "banknote_stack", "counting_machine"}.issubset(classes):
            for removed in facts:
                if (removed.kind != "cash_removed_from_bag" or removed.status != "present"
                        or removed.object_track_id is None or not removed.related_track_ids):
                    continue
                for stored in facts:
                    if (stored.kind != "cash_stored_in_safe" or stored.status != "present"
                            or not _same(removed, stored) or stored.t <= removed.end_t):
                        continue
                    negative = _negative(facts, removed, {"counting", "manual_count", "machine_count",
                                                         "manual_count_completed", "machine_count_completed"}, removed.t, stored.t,
                                         required_absent_kinds={"counting"})
                    if (negative and _covered(covers, removed, "counting", removed.t, stored.t)
                            and _covered(covers, removed, "object_track", removed.t, stored.t, object_bound=True)):
                        emit("NO_RECOUNT", stored.t, [removed, stored, negative],
                             "same tracked banknote bundle moved from bag into safe; continuously visible counting interval explicitly reviewed absent",
                             tracks=removed.related_track_ids)

    if "CASH_HANDOVER_IRREGULAR" in approved:
        reasons["CASH_HANDOVER_IRREGULAR"] = "requires_two_tracked_note_sets_and_personal_retention_after_completion"
        cash_tracks = _track_records(frames, classes.intersection(CASH_CLASSES), minimum)
        for split in facts:
            if (split.kind != "cash_split" or split.status != "present" or not _bound(split)
                    or split.object_track_id is None or len(split.related_track_ids) != 2
                    or len(set(split.related_track_ids)) != 2
                    or split.object_track_id in split.related_track_ids):
                continue
            child_a, child_b = split.related_track_ids
            if any(not any(abs(t - split.t) <= max_gap for t, _ in cash_tracks.get(child, ()))
                   for child in (child_a, child_b)):
                continue
            prepared = next((f for f in facts if f.kind == "cash_prepared" and f.status == "present"
                             and _same(split, f) and f.end_t <= split.t), None)
            if prepared is None:
                continue
            for transfer in facts:
                if (transfer.kind != "cash_transferred" or transfer.status != "present"
                        or not _same(split, transfer, same_object=False) or transfer.t < split.end_t
                        or transfer.object_track_id not in split.related_track_ids
                        or transfer.recipient_role != "client" or transfer.recipient_track_id is None
                        or transfer.recipient_track_id == split.actor_track_id):
                    continue
                remainder = child_b if transfer.object_track_id == child_a else child_a
                if not any(abs(t - transfer.t) <= max_gap for t, _ in cash_tracks.get(transfer.object_track_id, ())):
                    continue
                for end in ends(transfer):
                    for retained in facts:
                        if (retained.kind != "cash_retained_personally" or retained.status != "present"
                                or not _same(split, retained, same_object=False) or retained.t < end.t
                                or retained.object_track_id != remainder
                                or retained.destination not in {"personal_pocket", "verified_personal_container"}):
                            continue
                        if not any(abs(t - retained.t) <= max_gap for t, _ in cash_tracks.get(remainder, ())):
                            continue
                        if any(f.kind in {"cash_recombined", "cash_transferred"} and f.status == "present"
                               and _same(split, f, same_object=False) and split.t <= f.t <= retained.t
                               and (f.kind == "cash_recombined" or f.object_track_id == remainder) for f in facts):
                            continue
                        if not all(_covered(covers, anchor, "object_track", split.t, retained.t, object_bound=True)
                                   for anchor in (transfer, retained)):
                            continue
                        emit("CASH_HANDOVER_IRREGULAR", retained.t, [prepared, split, transfer, end, retained],
                             "prepared cash split into two tracked note sets; one transferred to client and other positively retained in verified personal destination after completion",
                             tracks=split.related_track_ids)

    # Collapse repeated windows of the same supporting event, retaining the best
    # score. Different events at different source times remain independently auditable.
    unique: dict[tuple[str, float, tuple[int, ...]], SpecialistCandidate] = {}
    for candidate in candidates:
        key = candidate.type_id, candidate.t, tuple(candidate.evidence.get("track_ids", ()))
        if key not in unique or candidate.score > unique[key].score:
            unique[key] = candidate
    candidates = sorted(unique.values(), key=lambda c: (c.t, c.type_id))
    emitted = {candidate.type_id for candidate in candidates}
    audit = [{"type_id": kind, "status": "candidate" if kind in emitted else "abstained",
              "reason": "requires_independent_vlm_verification" if kind in emitted else reasons[kind]}
             for kind in sorted(SUPPORTED_TYPES)]
    return candidates, audit


# Short adapter name for integrations that already use ``candidates(...)``.
candidates = specialist_candidates
