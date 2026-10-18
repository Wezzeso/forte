"""Adversarial track/event tests; no video, trained weights or GPU required."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from forte.fast.temporal_events import (
    CoverageInterval, TemporalObservation, filter_doorway_people, specialist_candidates,
)


CLASSES = ("cup", "coat_or_bag", "person", "id_card", "banknote_stack", "single_banknote",
           "cash_bag", "key_in_lock", "counting_machine", "detector_device")


def detection(name="cup", track=7, *, zones=("desk",), confidence=.96, box=(.1, .2, .2, .3)):
    return SimpleNamespace(class_name=name, confidence=confidence, track_id=track, zones=zones, box=box)


def frame(t, *detections, contacts=()):
    return SimpleNamespace(t=float(t), detections=detections, contacts=contacts)


def analysis(*frames, approved=("PERSONAL_ITEMS",), classes=CLASSES):
    return SimpleNamespace(frames=frames, approved_types=approved, approved_classes=classes,
                           provenance={}, hand_status="ok")


def fact(kind, t, **kwargs):
    return TemporalObservation(kind, float(t), transaction_id="tx-a", actor_track_id=1,
                               actor_role="cashier", object_track_id=10, source="tracked_event", **kwargs)


def negative(kind, start=0, end=4):
    return replace(fact(kind, start), end_t=float(end), status="absent", source="verified_interval")


def cover(scope, start=0, end=4, *, object_track=10, **kwargs):
    return CoverageInterval(scope, float(start), float(end), "tx-a", 1,
                            object_track_id=object_track, complete=True, source="human_verified", **kwargs)


def result(kind, facts=(), coverage=(), *frames, **kwargs):
    return specialist_candidates(analysis(*frames, approved=(kind,), **kwargs),
                                 observations=facts, coverage=coverage)


def test_calibrated_cup_needs_persistent_same_track_and_desk_zone():
    candidates, audit = specialist_candidates(analysis(*(frame(t, detection()) for t in (0, .5, 1))))
    assert len(candidates) == 1
    assert candidates[0].type_id == "PERSONAL_ITEMS" and "drink" in candidates[0].note
    assert candidates[0].evidence["track_ids"] == [7]
    assert candidates[0].evidence["start_sec"] == 0
    assert candidates[0].evidence["end_sec"] == 1
    assert candidates[0].evidence["requires_independent_verification"] is True
    assert next(a for a in audit if a["type_id"] == "PERSONAL_ITEMS")["status"] == "candidate"


@pytest.mark.parametrize("frames", [
    (frame(0, detection()),),
    tuple(frame(t, detection(track=i)) for i, t in enumerate((0, .5, 1))),
    tuple(frame(t, detection(zones=("door",))) for t in (0, .5, 1)),
    tuple(frame(t, detection(confidence=.5)) for t in (0, .5, 1)),
    tuple(frame(t, detection()) for t in (0, 2, 4)),
    (frame(0, detection()), frame(.5, detection("coat_or_bag")), frame(1, detection())),
])
def test_ambiguous_or_fragmented_desk_objects_abstain(frames):
    assert specialist_candidates(analysis(*frames))[0] == []


def test_promotion_gate_required_for_type_and_class():
    frames = tuple(frame(t, detection()) for t in (0, .5, 1))
    assert specialist_candidates(analysis(*frames, approved=()))[0] == []
    assert specialist_candidates(analysis(*frames, classes=("id_card",)))[0] == []


def test_post_decode_layout_supplies_empty_runtime_zones_without_mutation():
    frames = tuple(frame(t, detection(zones=())) for t in (0, .5, 1))
    candidates, _ = specialist_candidates(analysis(*frames), layout={"desk": [0, 100, 500, 600]})
    assert len(candidates) == 1
    assert frames[0].detections[0].zones == ()


def test_raw_vlm_states_never_supply_specialist_role_or_omission_evidence():
    data = analysis(frame(0), frame(1), approved=("NO_CLIENT_ID", "NO_AUTHENTICITY_CHECK", "KEY_LEFT_IN_SAFE"))
    candidates, audit = specialist_candidates(data, states=[{"t": 0, "id_card_shown": False,
                                                            "cashier_in_room": False}])
    assert candidates == []
    assert all(a["status"] == "abstained" for a in audit)


def test_person_count_does_not_assign_unauthorized_roles_and_doorway_is_filtered():
    outsider = detection("person", 12, zones=("door",))
    interior = detection("person", 13, zones=("cash_room",))
    anonymous = detection("person", None, zones=("cash_room",))
    assert filter_doorway_people((outsider, interior, anonymous)) == [interior]
    assert specialist_candidates(analysis(frame(0, outsider, interior, anonymous),
                                          approved=("UNAUTHORIZED_PERSON",)))[0] == []


def id_case():
    return ([fact("transaction_started", 0), fact("cash_received", 1),
             fact("transaction_completed", 4), negative("id_checked")],
            [cover("id_exchange"), cover("object_track", 1, 4)])


def test_no_id_requires_complete_transaction_explicit_negative_and_track_coverage():
    facts, coverage = id_case()
    candidates, _ = result("NO_CLIENT_ID", facts, coverage)
    assert len(candidates) == 1 and candidates[0].t == 1
    assert candidates[0].evidence["start_sec"] == 0
    assert candidates[0].evidence["end_sec"] == 4
    assert result("NO_CLIENT_ID", facts, ())[0] == []
    assert result("NO_CLIENT_ID", facts[:-1], coverage)[0] == []
    assert result("NO_CLIENT_ID", facts[1:], coverage)[0] == []
    assert result("NO_CLIENT_ID", facts[:2] + facts[3:], coverage)[0] == []


@pytest.mark.parametrize("extra", [
    replace(fact("id_checked", 2), object_track_id=45),
    replace(fact("id_checked", 2), status="unknown", confidence=.2),
    replace(fact("id_presented", 2), object_track_id=45),
])
def test_positive_or_unknown_id_evidence_blocks_omission_even_on_other_object(extra):
    facts, coverage = id_case()
    assert result("NO_CLIENT_ID", facts + [extra], coverage)[0] == []


def test_seen_id_track_blocks_no_id_even_if_not_certain_it_was_checked():
    facts, coverage = id_case()
    assert result("NO_CLIENT_ID", facts, coverage, frame(2, detection("id_card", 45)))[0] == []


def test_visibility_gap_occlusion_or_wrong_track_invalidates_absence():
    facts, coverage = id_case()
    gap = [cover("id_exchange", 0, 1.9), cover("id_exchange", 2, 4), coverage[1]]
    occluded = coverage + [replace(cover("id_exchange", 2, 2.1), complete=False)]
    wrong_track = [coverage[0], cover("object_track", 1, 4, object_track=999)]
    unverified = [replace(coverage[0], source="unverified"), coverage[1]]
    for intervals in (gap, occluded, wrong_track, unverified):
        assert result("NO_CLIENT_ID", facts, intervals)[0] == []


def test_adjacent_complete_visibility_segments_are_valid():
    facts, coverage = id_case()
    intervals = [cover("id_exchange", 0, 2), cover("id_exchange", 2, 4), coverage[1]]
    assert len(result("NO_CLIENT_ID", facts, intervals)[0]) == 1


def test_transactions_and_actors_cannot_be_combined():
    facts, coverage = id_case()
    for changed in (replace(facts[-1], transaction_id="tx-b"),
                    replace(facts[-1], actor_track_id=2), replace(facts[0], actor_role=None)):
        altered = [changed if f.kind == changed.kind else f for f in facts]
        assert result("NO_CLIENT_ID", altered, coverage)[0] == []


def test_actions_cannot_extend_past_claimed_transaction_completion_or_overlap_start():
    facts, coverage = id_case()
    overlong_receipt = [replace(f, end_t=8.) if f.kind == "cash_received" else f for f in facts]
    late_start = [replace(f, end_t=2.) if f.kind == "transaction_started" else f for f in facts]
    assert result("NO_CLIENT_ID", overlong_receipt, coverage)[0] == []
    assert result("NO_CLIENT_ID", late_start, coverage)[0] == []


def test_visibility_complete_rejects_truthy_strings():
    with pytest.raises(ValueError, match="explicit boolean"):
        replace(cover("id_exchange"), complete="false")


def check_case():
    return ([fact("transaction_started", 0), fact("manual_count_completed", 2),
             fact("transaction_completed", 4), negative("authenticity_check")],
            [cover("authenticity_check"), cover("object_track")])


def test_no_authenticity_check_needs_count_completion_and_explicit_full_visibility():
    facts, coverage = check_case()
    candidates, _ = result("NO_AUTHENTICITY_CHECK", facts, coverage)
    assert len(candidates) == 1 and candidates[0].t == 4
    assert result("NO_AUTHENTICITY_CHECK", facts[:1] + facts[2:], coverage)[0] == []
    assert result("NO_AUTHENTICITY_CHECK", facts, coverage[:1])[0] == []
    assert result("NO_AUTHENTICITY_CHECK", facts + [fact("authenticity_check", 3)], coverage)[0] == []


def test_hand_device_proximity_introduces_doubt_but_does_not_prove_an_actual_check():
    facts, coverage = check_case()
    contact = SimpleNamespace(class_name="detector_device", signal="proximity_2d")
    assert result("NO_AUTHENTICITY_CHECK", facts, coverage, frame(2.5, contacts=(contact,)))[0] == []
    # The contact is not transformed into a positive observation or any new event.
    assert specialist_candidates(analysis(frame(2.5, contacts=(contact,))))[0] == []


def recount_case():
    return ([replace(fact("cash_removed_from_bag", 1), related_track_ids=(20,)),
             fact("cash_stored_in_safe", 3), negative("counting", 1, 3)],
            [cover("counting", 1, 3), cover("object_track", 1, 3)])


def test_bag_to_safe_sequence_requires_same_cash_track_order_and_no_count_interval():
    facts, coverage = recount_case()
    candidates, _ = result("NO_RECOUNT", facts, coverage)
    assert len(candidates) == 1 and candidates[0].t == 3
    assert candidates[0].evidence["track_ids"] == [10, 20]
    for changed in (replace(facts[1], object_track_id=11), replace(facts[1], t=.5, end_t=.5)):
        assert result("NO_RECOUNT", [facts[0], changed, facts[2]], coverage)[0] == []
    for kind in ("manual_count", "manual_count_completed", "machine_count", "machine_count_completed"):
        assert result("NO_RECOUNT", facts + [fact(kind, 2)], coverage)[0] == []
    assert result("NO_RECOUNT", facts, coverage[:1])[0] == []


def test_absence_of_only_one_counting_modality_does_not_prove_no_recount():
    facts, coverage = recount_case()
    for modality in ("manual_count", "machine_count", "manual_count_completed", "machine_count_completed"):
        altered = facts[:-1] + [replace(facts[-1], kind=modality)]
        assert result("NO_RECOUNT", altered, coverage)[0] == []


def key_frames():
    return tuple(frame(t, detection("key_in_lock", 40, zones=("safe_lock",))) for t in (0, .5, 1, 1.5))


def test_key_must_be_resolved_in_lock_before_and_after_observed_departure():
    departure = replace(fact("cashier_departed", .75), object_track_id=None)
    candidates, _ = result("KEY_LEFT_IN_SAFE", [departure], (), *key_frames())
    assert len(candidates) == 1 and candidates[0].t == .75
    assert candidates[0].evidence["lock_snapshots"][0]["t"] == .5
    assert candidates[0].evidence["lock_snapshots"][1]["t"] == 1.5
    assert result("KEY_LEFT_IN_SAFE", [], (), *key_frames())[0] == []
    assert result("KEY_LEFT_IN_SAFE", [replace(departure, actor_role=None)], (), *key_frames())[0] == []
    assert result("KEY_LEFT_IN_SAFE", [departure], (), *key_frames()[:2])[0] == []
    assert result("KEY_LEFT_IN_SAFE", [departure, fact("cashier_returned", 1.1)], (), *key_frames())[0] == []


def role_state(t, present):
    return {"t": float(t), "cashier_in_room": present, "cashier_at_desk": present,
            "cashier_seat": present}


def test_explicit_sustained_cashier_role_states_can_support_key_without_invented_person_id():
    states = [role_state(0, True)] + [role_state(t, False) for t in (1, 2, 3, 4)]
    frames = tuple(frame(t, detection("key_in_lock", 40, zones=("safe_lock",)))
                   for t in (0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4))
    candidates, _ = specialist_candidates(analysis(*frames, approved=("KEY_LEFT_IN_SAFE",)), states=states)
    assert len(candidates) == 1 and candidates[0].t == 1
    assert candidates[0].evidence["track_ids"] == [40]
    assert candidates[0].evidence["state_support"] == states
    assert candidates[0].evidence["lock_snapshots"][-1]["t"] >= 4


@pytest.mark.parametrize("states", [
    [role_state(0, True), role_state(1, False), role_state(2, False), role_state(3, False)],
    [role_state(0, None)] + [role_state(t, False) for t in (1, 2, 3, 4)],
    [role_state(0, True), role_state(1, False), role_state(2, None), role_state(3, False), role_state(4, False)],
    [role_state(0, True), role_state(1, False), role_state(3, False), role_state(4, False)],
    [role_state(0, True)] + [role_state(t, 0) for t in (1, 2, 3, 4)],
    [role_state(0, True)] + [{"t": t, "cashier_in_room": False} for t in (1, 2, 3, 4)],
])
def test_short_unknown_missing_or_gapped_cashier_states_cannot_support_key(states):
    frames = tuple(frame(t, detection("key_in_lock", 40, zones=("safe_lock",)))
                   for t in (0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4))
    assert specialist_candidates(analysis(*frames, approved=("KEY_LEFT_IN_SAFE",)), states=states)[0] == []


def test_key_track_that_moves_to_another_lock_region_abstains():
    departure = fact("cashier_departed", .75)
    frames = list(key_frames())
    frames[-1] = frame(1.5, detection("key_in_lock", 40, zones=("safe_lock",), box=(.6, .6, .7, .7)))
    assert result("KEY_LEFT_IN_SAFE", [departure], (), *frames)[0] == []


def partial_case():
    facts = [fact("cash_prepared", 0), replace(fact("cash_split", 1), related_track_ids=(11, 12)),
             replace(fact("cash_transferred", 2), object_track_id=11,
                     recipient_track_id=2, recipient_role="client"),
             fact("transaction_completed", 4),
             replace(fact("cash_retained_personally", 5), object_track_id=12, destination="personal_pocket")]
    covers = [cover("object_track", 1, 5, object_track=11), cover("object_track", 1, 5, object_track=12)]
    frames = tuple(frame(t, detection("banknote_stack", 11), detection("single_banknote", 12))
                   for t in (1, 2, 3, 4, 5))
    return facts, covers, frames


def test_partial_handover_requires_two_tracked_note_sets_and_personal_retention_after_completion():
    facts, coverage, frames = partial_case()
    candidates, _ = result("CASH_HANDOVER_IRREGULAR", facts, coverage, *frames)
    assert len(candidates) == 1 and candidates[0].t == 5
    assert candidates[0].evidence["track_ids"] == [10, 11, 12]
    assert result("CASH_HANDOVER_IRREGULAR", facts, coverage)[0] == []
    assert result("CASH_HANDOVER_IRREGULAR", facts[:-1], coverage, *frames)[0] == []
    assert result("CASH_HANDOVER_IRREGULAR", facts, coverage[:1], *frames)[0] == []
    assert result("CASH_HANDOVER_IRREGULAR", facts, coverage, *frames[:2])[0] == []


@pytest.mark.parametrize("change", [
    {"destination": "authorized_tray"}, {"t": 3., "end_t": 3.},
    {"transaction_id": "tx-b"}, {"object_track_id": 13},
])
def test_temporary_split_authorized_storage_or_cross_transaction_retention_abstains(change):
    facts, coverage, frames = partial_case()
    facts[-1] = replace(facts[-1], **change)
    assert result("CASH_HANDOVER_IRREGULAR", facts, coverage, *frames)[0] == []


def test_split_then_recombine_or_hand_over_remainder_is_not_partial_handover():
    facts, coverage, frames = partial_case()
    for additional in (fact("cash_recombined", 3),
                       replace(fact("cash_transferred", 3), object_track_id=12)):
        assert result("CASH_HANDOVER_IRREGULAR", facts + [additional], coverage, *frames)[0] == []


def test_split_lineage_requires_exactly_two_distinct_child_track_ids():
    facts, coverage, frames = partial_case()
    for children in ((11, 12, 11), (11, 11), (11,), (10, 11)):
        facts[1] = replace(facts[1], related_track_ids=children)
        assert result("CASH_HANDOVER_IRREGULAR", facts, coverage, *frames)[0] == []


def test_out_of_order_events_normalized_and_duplicate_proposals_collapsed():
    facts, coverage = id_case()
    candidates, _ = result("NO_CLIENT_ID", list(reversed(facts)) + [facts[1]], coverage)
    assert len(candidates) == 1


def test_invalid_nonfinite_or_duplicate_time_inputs_rejected():
    with pytest.raises(ValueError):
        fact("cash_received", float("nan"))
    with pytest.raises(ValueError):
        replace(fact("cash_received", 1), confidence=float("inf"))
    with pytest.raises(ValueError):
        cover("id_exchange", 4, 0)
    with pytest.raises(ValueError):
        specialist_candidates(analysis(frame(0), frame(0)))
    with pytest.raises(ValueError, match="TemporalObservation"):
        specialist_candidates(analysis(), cfg={"specialist_observations": [{"kind": "cash_received"}]})
