"""Synthetic source IDs only: no supplied recording or label is inspected."""

from types import SimpleNamespace

import pytest

from forte.events import EventBuildConfig, analyze_rule_relevance, build_events
from forte.shared.schemas import CandidateEvent, ClipRequest, FactDecision, FrameRef, RuleRecord


def frame(index):
    return FrameRef(frame_id=f"f{index}", pts_sec=float(index), path=f"synthetic/f{index}.jpg",
                    width=32, height=32, job_id="synthetic")


FRAMES = {f"f{i}": frame(i) for i in range(21)}


def fact(fact_id="ACTION", state="present", ids=("f5",)):
    return FactDecision(fact_id=fact_id, state=state, supporting_frame_ids=list(ids))


def candidate(candidate_id="c1", start=4.0, end=6.0, kind="TYPE_A", facts=None, ids=None):
    return CandidateEvent(candidate_id=candidate_id, event_type_id=kind, start_sec=start, end_sec=end,
                          supporting_facts=[fact()] if facts is None else facts,
                          supporting_frame_ids=["f5"] if ids is None else ids)


def batch(clip_id="clip1", start=0.0, end=20.0, candidates=(), facts=(), disposition="complete", complete=True):
    return SimpleNamespace(clip_id=clip_id, start_sec=start, end_sec=end, candidates=tuple(candidates),
                           facts=tuple(facts), disposition=disposition, complete=complete)


def rule(rule_id="RULE_A", *, enabled=True, required=("ACTION",), temporal=(), exceptions=()):
    return RuleRecord(rule_id=rule_id, title="Synthetic reviewed fixture", clause="1",
                      source_url="https://example.invalid/synthetic-rule", applicability="Synthetic fixture only",
                      required_facts=list(required), verification_state="verified", enabled=enabled,
                      applicability_verified=True, temporal_conditions=list(temporal), exceptions=list(exceptions))


def catalog(binding=None, records=None):
    return SimpleNamespace(rules={r.rule_id: r for r in (records or [rule()])},
                           bindings=binding or {"RULE_A": {"event_type_id": "TYPE_A"}})


def coverage(complete=True):
    return {"duration_sec": 20.0, "complete": complete, "gaps": [] if complete else [[12.0, 20.0]],
            "missing_window_indices": [] if complete else [3]}


def test_repeated_overlap_is_one_deterministic_event():
    first = batch("clip1", 0.0, 10.0, [candidate()])
    second = batch("clip2", 4.0, 20.0, [candidate("c2", 5.0, 7.0)])
    a = build_events([first, second], catalog(), FRAMES, coverage())
    b = build_events([second, first], catalog(), FRAMES, coverage())
    assert a.verdict == "violation"
    assert len(a.events) == 1
    assert a.events == b.events
    assert a.events[0].anchor_frame_id in a.events[0].evidence_frame_ids
    assert (a.events[0].start_sec, a.events[0].end_sec) == (4.0, 7.0)


def test_separate_same_type_occurrences_and_different_types_survive():
    other_rule = rule("RULE_B")
    rules = catalog({"RULE_A": {"event_type_id": "TYPE_A"}, "RULE_B": {"event_type_id": "TYPE_B"}},
                    [rule(), other_rule])
    candidates = [candidate(), candidate("later", 10.0, 12.0, facts=[fact(ids=("f11",))], ids=["f11"]),
                  candidate("other", 4.0, 6.0, "TYPE_B")]
    result = build_events([batch(candidates=candidates)], rules, FRAMES, coverage())
    assert [event.event_type_id for event in result.events].count("TYPE_A") == 2
    assert len(result.events) == 3


def test_distinct_candidates_in_same_window_are_not_collapsed():
    result = build_events([batch(candidates=[candidate(), candidate("second", 4.0, 6.0)])],
                          catalog(), FRAMES, coverage())
    assert len(result.events) == 2


def test_overlap_chain_does_not_bridge_two_occurrences():
    batches = [batch("one", candidates=[candidate("a", 4.0, 6.0)]),
               batch("two", candidates=[candidate("b", 5.0, 7.0)]),
               batch("three", candidates=[candidate("c", 6.0, 8.0, facts=[fact(ids=("f7",))], ids=["f7"])])]
    assert len(build_events(batches, catalog(), FRAMES, coverage()).events) == 2


def test_positive_with_unclear_interval_retains_limitation():
    batches = [batch("one", 0.0, 10.0, [candidate()]),
               batch("two", 10.0, 20.0, disposition="unclear")]
    result = build_events(batches, catalog(), FRAMES, coverage())
    assert result.can_publish and result.verdict == "violation"
    assert result.unresolved_intervals and result.refinement_requests
    assert "unclear_visual_context" in result.events[0].limitations


def test_normal_is_only_complete_success_without_uncertainty():
    assert build_events([batch()], catalog(), FRAMES, coverage()).verdict == "no_violation"
    assert build_events([batch(disposition="unclear")], catalog(), FRAMES, coverage()).verdict is None
    assert build_events([], catalog(), FRAMES, coverage()).verdict is None
    assert build_events([batch(end=10.0)], catalog(), FRAMES, coverage()).verdict is None
    assert build_events([batch(complete=False)], catalog(), FRAMES, coverage()).verdict is None


def test_technical_coverage_failure_blocks_even_a_supported_positive():
    result = build_events([batch(candidates=[candidate()])], catalog(), FRAMES, coverage(False))
    assert len(result.events) == 1
    assert not result.can_publish and result.verdict is None


@pytest.mark.parametrize("event", [candidate(facts=[fact(state="unclear", ids=())]),
                                  candidate(facts=[fact(ids=("fabricated",))], ids=["fabricated"]),
                                  candidate(facts=[fact(ids=("f15",))], ids=["f15"])])
def test_unsupported_facts_and_missing_source_anchor_are_unresolved(event):
    result = build_events([batch(candidates=[event])], catalog(), FRAMES, coverage())
    assert not result.events
    assert result.verdict is None
    assert result.refinement_requests


def test_unknown_type_and_unverified_rule_never_become_a_nearby_category():
    result = build_events([batch(candidates=[candidate(kind="UNKNOWN")])], catalog(), FRAMES, coverage())
    assert not result.events and result.verdict is None
    result = build_events([batch(candidates=[candidate()])], catalog(records=[rule(enabled=False)]), FRAMES, coverage())
    assert not result.events and result.verdict is None


def test_omission_needs_full_procedure_and_visible_absence():
    binding = {"RULE_A": {"event_type_id": "TYPE_A", "is_omission": True,
                          "required_fact_states": {"COUNT": "absent"}}}
    facts = [fact(), fact("COUNT", "absent", ("f4", "f6")),
             fact("WORK_AREA_CONTINUOUSLY_VISIBLE", ids=("f4", "f6"))]
    result = build_events([batch(candidates=[candidate(facts=facts)])], catalog(binding), FRAMES, coverage())
    assert result.verdict is None
    assert "PROCEDURE_CONTEXT_COMPLETE" in result.refinement_requests[0].required_fact_ids
    facts.append(fact("PROCEDURE_CONTEXT_COMPLETE", ids=("f6",)))
    result = build_events([batch(candidates=[candidate(facts=facts, ids=["f4", "f6"])])],
                          catalog(binding), FRAMES, coverage())
    assert result.verdict == "violation"
    assert result.events[0].anchor_frame_id == "f6"
    assert any("omission_anchor" in text for text in result.events[0].limitations)


def test_insufficient_long_context_returns_bounded_refinement():
    rules = catalog({"RULE_A": {"event_type_id": "TYPE_A", "min_context_sec": 12.0}})
    result = build_events([batch("one", 0.0, 10.0, [candidate()]), batch("two", 10.0, 20.0)],
                          rules, FRAMES, coverage())
    assert not result.events
    request = result.refinement_requests[0]
    assert request.reason == "longer_procedure_context_required"
    assert 0 <= request.start_sec <= request.end_sec <= 20


def test_textual_temporal_conditions_and_exceptions_need_machine_gates():
    result = build_events([batch(candidates=[candidate()])],
                          catalog(records=[rule(temporal=["sequence required"]) ]), FRAMES, coverage())
    assert not result.events and result.verdict is None
    result = build_events([batch(candidates=[candidate()])],
                          catalog(records=[rule(exceptions=["sealed container"]) ]), FRAMES, coverage())
    assert not result.events and result.verdict is None


def test_selected_evidence_covers_every_required_affirmative_fact():
    facts = [fact("A", ids=("f4",)), fact("B", ids=("f5",)), fact("C", ids=("f6",))]
    result = build_events([batch(candidates=[candidate(facts=facts)])],
                          catalog(records=[rule(required=("A", "B", "C"))]), FRAMES, coverage())
    assert result.verdict == "violation"
    assert set(result.events[0].evidence_frame_ids) == {"f4", "f5", "f6"}


def test_event_requiring_four_distinct_evidence_frames_is_not_publishable():
    facts = [fact(str(i), ids=(f"f{i}",)) for i in range(4, 8)]
    event = candidate(end=7.0, facts=facts, ids=["f4", "f5", "f6", "f7"])
    result = build_events([batch(candidates=[event])],
                          catalog(records=[rule(required=tuple(str(i) for i in range(4, 8)))]),
                          FRAMES, coverage())
    assert result.verdict is None
    assert result.unresolved_intervals[0].reason == "evidence_support_exceeds_frame_budget"


def test_taxonomy_required_states_are_enforced():
    config = EventBuildConfig(event_types={"TYPE_A": {"required_fact_states": {"HANDLER": "absent"}}})
    result = build_events([batch(candidates=[candidate()])], catalog(), FRAMES, coverage(), config=config)
    assert result.verdict is None
    assert "HANDLER" in result.refinement_requests[0].required_fact_ids


def test_conflicting_candidate_identity_cannot_retain_an_earlier_positive():
    changed = candidate(end=7.0)
    first = batch(candidates=[candidate()])
    second = batch(candidates=[changed])
    a = build_events([first, second], catalog(), FRAMES, coverage())
    b = build_events([second, first], catalog(), FRAMES, coverage())
    assert not a.events and not b.events
    assert a.verdict is None and b.verdict is None


def test_invalid_source_timeline_is_rejected():
    with pytest.raises(ValueError, match="timeline"):
        build_events([batch()], catalog(), [frame(21)], coverage())


def test_multiple_applicable_rules_ground_one_visual_occurrence():
    rules = catalog({"RULE_A": {"event_type_id": "TYPE_A"}, "RULE_B": {"event_type_id": "TYPE_A"}},
                    [rule(), rule("RULE_B")])
    result = build_events([batch(candidates=[candidate()])], rules, FRAMES, coverage())
    assert len(result.events) == 1
    assert result.events[0].rule_id == "RULE_A"


def test_missing_scheduled_window_blocks_negative_despite_union_coverage():
    report = coverage()
    report["missing_window_indices"] = [2]
    result = build_events([batch()], catalog(), FRAMES, report)
    assert result.verdict is None and not result.coverage_complete


def test_reviewed_catalog_bindings_accept_synthetic_confirmed_facts():
    from rules.catalog import load_catalog

    reviewed = load_catalog()
    for reviewed_rule in reviewed.enabled_rules():
        binding = reviewed.bindings[reviewed_rule.rule_id]
        states = {key: "absent" for key in reviewed.facts}
        states.update(binding["required_fact_states"])
        facts = [fact(key, state, ("f4", "f5", "f6")) for key, state in states.items()]
        event = candidate(kind=binding["event_type_id"], facts=facts, ids=["f4", "f5", "f6"])
        result = build_events([batch(candidates=[event], facts=facts)], reviewed, FRAMES, coverage())
        assert result.verdict == "violation", reviewed_rule.rule_id
        assert len(result.events) == 1
        assert result.events[0].rule_id == reviewed_rule.rule_id
        assert not result.unresolved_intervals


def test_real_decision_and_coverage_envelopes_integrate():
    from forte.decision import DecisionBatch
    from forte.shared.schemas import FactDefinition
    from forte.video.sampling import coverage_report, coverage_windows, make_clip_requests, record_inspection

    windows = coverage_windows(20.0)
    requests = make_clip_requests("synthetic", list(FRAMES.values()), windows,
                                  fact_definitions=[FactDefinition(fact_id="ACTION", description="synthetic action")])
    batches = []
    for request in requests:
        ids = [request.frame_refs[0].frame_id]
        observed = fact("ACTION", "absent", ids)
        batches.append(DecisionBatch(request=request, facts=(observed,), candidates=()))
    inspected = [record_inspection(window, request) for window, request in zip(windows, requests)]
    result = build_events(batches, catalog(), FRAMES, coverage_report(20.0, windows, inspected))
    assert result.coverage_complete and result.verdict == "no_violation"


def test_missing_required_batch_facts_cannot_publish_a_normal_result():
    from forte.decision import DecisionBatch
    from forte.shared.schemas import ClipRequest, FactDefinition

    request = ClipRequest(job_id="synthetic", clip_id="all", frame_refs=list(FRAMES.values()),
                          start_sec=0.0, end_sec=20.0,
                          fact_definitions=[FactDefinition(fact_id="ACTION", description="synthetic action")])
    result = build_events([DecisionBatch(request=request, facts=(), candidates=())], catalog(), FRAMES, coverage())
    assert result.verdict is None and not result.coverage_complete


def _relevance_batch(facts):
    request = ClipRequest(job_id="synthetic", clip_id="scope", frame_refs=list(FRAMES.values()),
                          start_sec=0.0, end_sec=20.0)
    return SimpleNamespace(request=request, facts=tuple(facts), candidates=(), complete=True,
                           start_sec=0.0, end_sec=20.0,
                           disposition="unclear" if any(value.state == "unclear" for value in facts) else "complete")


def test_irrelevant_unclear_is_a_limit_when_every_branch_has_a_scoped_refutation():
    rules = catalog()
    observed = _relevance_batch([fact("ACTION", "absent", tuple(FRAMES)), fact("UNRELATED", "unclear", ())])
    relevance = analyze_rule_relevance(observed, rules)
    assert relevance.all_branches_refuted and not relevance.requires_refinement
    # Fixture request has no requested facts; the helper is tested with full
    # source scope, and builder's fixture mapping checks the same fact support.
    value = batch(facts=observed.facts, disposition="unclear")
    result = build_events([value], rules, FRAMES, coverage())
    assert result.verdict == "no_violation"
    assert result.limitations and not result.unresolved_intervals and not result.refinement_requests


def test_all_unclear_or_one_unrefuted_branch_stays_blocking():
    observed = batch(facts=[fact("ACTION", "unclear", ())], disposition="unclear")
    assert build_events([observed], catalog(), FRAMES, coverage()).verdict is None
    other = rule("RULE_B", required=("OTHER_ACTION",))
    rules = catalog({"RULE_A": {"event_type_id": "TYPE_A"}, "RULE_B": {"event_type_id": "TYPE_B"}},
                    [rule(), other])
    observed = batch(facts=[fact("ACTION", "absent", tuple(FRAMES)), fact("OTHER_ACTION", "unclear", ())],
                     disposition="unclear")
    assert build_events([observed], rules, FRAMES, coverage()).verdict is None


@pytest.mark.parametrize("gate", ["WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE",
                                  "ALL_COUNTING_PATHS_VISIBLE", "PREVIOUSLY_ACCEPTED_CASH_TRACKED"])
def test_context_absence_is_lack_of_knowledge_even_without_literal_unclear(gate):
    states = {"ACTION": "present", gate: "present"}
    rules = catalog({"RULE_A": {"event_type_id": "TYPE_A", "required_fact_states": states}},
                    [rule(required=("ACTION", gate))])
    observed = batch(facts=[fact("ACTION", ids=tuple(FRAMES)), fact(gate, "absent", tuple(FRAMES))])
    relevance = analyze_rule_relevance(observed, rules)
    assert not relevance.all_branches_refuted and relevance.requires_refinement
    result = build_events([observed], rules, FRAMES, coverage())
    assert result.verdict is None and result.refinement_requests


def test_single_frame_opposite_state_does_not_refute_a_larger_request_scope():
    observed = _relevance_batch([fact("ACTION", "absent", ("f5",)), fact("UNRELATED", "unclear", ())])
    assert not analyze_rule_relevance(observed, catalog()).all_branches_refuted


def test_counting_from_another_procedure_cannot_globally_refute_an_omission():
    from rules.catalog import load_catalog

    reviewed = load_catalog()
    rule_id = "RK_NBK231_P19_2_RECEIPT_COUNT"
    rules = SimpleNamespace(rules={rule_id: reviewed.rules[rule_id]}, bindings={rule_id: reviewed.bindings[rule_id]})
    facts = [fact(key, "present" if key == "CASH_COUNT_VISIBLE" else state, tuple(FRAMES))
             for key, state in reviewed.bindings[rule_id]["required_fact_states"].items()]
    facts.append(fact("UNRELATED", "unclear", ()))
    observed = _relevance_batch(facts)
    relevance = analyze_rule_relevance(observed, rules)
    assert not relevance.all_branches_refuted and relevance.requires_refinement


def test_contradictory_candidate_blocks_negative_despite_other_refutations():
    observed = batch(candidates=[candidate(facts=[fact("ACTION", "absent", tuple(FRAMES))])],
                     facts=[fact("ACTION", "absent", tuple(FRAMES)), fact("UNRELATED", "unclear", ())],
                     disposition="unclear")
    assert build_events([observed], catalog(), FRAMES, coverage()).verdict is None
