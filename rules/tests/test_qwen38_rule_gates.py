"""Migration guardrails using synthetic facts and timestamps, never bank media.

These checks exercise the frozen legal boundary independently of any model
family. They do not establish visual accuracy or adjudicate a recording.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forte.events import build_events
from forte.shared.schemas import CandidateEvent, FactDecision, FrameRef
from rules.catalog import CATALOG_PATH, load_catalog
from rules.experimental_catalog import load_experimental_catalog


ROOT = Path(__file__).resolve().parents[2]
CATALOG = load_catalog()
RULE_IDS = tuple(sorted(rule.rule_id for rule in CATALOG.enabled_rules()))
FRAMES = {
    f"synthetic-{index}": FrameRef(
        frame_id=f"synthetic-{index}", pts_sec=float(index),
        path=f"synthetic/qwen38-rule-audit-{index}.jpg", width=32, height=32,
        job_id="synthetic-rule-audit",
    )
    for index in range(21)
}
COVERAGE = {
    "duration_sec": 20.0, "complete": True, "gaps": [],
    "missing_window_indices": [],
}


def scenario(rule_id, *, changes=None, omitted=(), supports=None, interval=(4.0, 6.0)):
    binding = CATALOG.bindings[rule_id]
    states = dict(binding["required_fact_states"])
    states.update(changes or {})
    facts = [
        FactDecision(
            fact_id=key, state=state,
            supporting_frame_ids=[] if state == "unclear" else list(
                (supports or {}).get(key, ("synthetic-4", "synthetic-5", "synthetic-6"))
            ),
        )
        for key, state in states.items() if key not in omitted
    ]
    event = CandidateEvent(
        candidate_id="synthetic-procedure", event_type_id=binding["event_type_id"],
        start_sec=interval[0], end_sec=interval[1], supporting_facts=facts,
        supporting_frame_ids=["synthetic-5"],
    )
    batch = SimpleNamespace(
        clip_id="synthetic-window", start_sec=0.0, end_sec=20.0,
        candidates=(event,), facts=tuple(facts), complete=True,
        disposition="unclear" if any(fact.state == "unclear" for fact in facts) else "complete",
    )
    return build_events([batch], CATALOG, FRAMES, COVERAGE)


GATE_CASES = [
    (rule_id, fact_id, state)
    for rule_id in RULE_IDS
    for fact_id, state in CATALOG.bindings[rule_id]["required_fact_states"].items()
]


def test_default_nineteen_axis_identity_matches_frozen_catalog_and_taxonomy():
    axis = json.loads((ROOT / "training/temporal/fact-axis.v1.json").read_text(encoding="utf-8"))
    assert axis["catalog_sha256"] == hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
    assert axis["taxonomy_sha256"] == hashlib.sha256(
        (ROOT / "annotations/taxonomy.v1.json").read_bytes()
    ).hexdigest()
    assert axis["fact_ids"] == [fact.fact_id for fact in CATALOG.fact_definitions()]
    assert len(axis["fact_ids"]) == 19
    assert axis["state_order"] == ["absent", "present", "unclear"]
    assert set(axis["event_type_ids"]) == {
        CATALOG.bindings[rule_id]["event_type_id"] for rule_id in RULE_IDS
    }


def test_experimental_axes_cannot_relabel_a_default_head():
    default = [fact.fact_id for fact in CATALOG.fact_definitions()]
    experimental = load_experimental_catalog()
    extended = [fact.fact_id for fact in experimental.axis_fact_definitions()]
    assert len(extended) == 37 and extended[:19] == default
    with pytest.raises(ValueError, match="axis order mismatch"):
        experimental.validate_axis_order(default, vocabulary_version=experimental.vocabulary_version)


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_complete_supported_branch_is_publishable_with_original_frame_ids(rule_id):
    result = scenario(rule_id)
    assert result.verdict == "violation"
    assert len(result.events) == 1
    event = result.events[0]
    assert event.rule_id == rule_id
    assert event.anchor_frame_id in event.evidence_frame_ids
    assert 1 <= len(event.evidence_frame_ids) <= 3
    assert set(event.evidence_frame_ids) <= set(FRAMES)


@pytest.mark.parametrize("rule_id,fact_id,required_state", GATE_CASES)
@pytest.mark.parametrize("condition", ["missing", "unclear", "opposite"])
def test_no_branch_survives_a_missing_unclear_or_opposite_required_gate(
    rule_id, fact_id, required_state, condition,
):
    omitted = (fact_id,) if condition == "missing" else ()
    changes = {} if omitted else {
        fact_id: "unclear" if condition == "unclear" else (
            "absent" if required_state == "present" else "present"
        )
    }
    result = scenario(rule_id, changes=changes, omitted=omitted)
    assert not result.events
    assert result.verdict is None  # a rejected positive must not silently become a clean negative
    assert result.refinement_requests


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_public_explanation_uses_only_reviewed_observations_and_frozen_citation(rule_id):
    rule = CATALOG.get_rule(rule_id)
    binding = CATALOG.bindings[rule_id]
    phrases = [CATALOG.fact_description(key, state) for key, state in binding["required_fact_states"].items()]
    explanation = CATALOG.render_description(rule, phrases)
    assert "Наблюдаемые признаки:" in explanation and "Возможные признаки" in explanation
    assert rule.clause in explanation and rule.source_url in explanation
    assert explanation == CATALOG.render_description(rule, reversed(phrases))
    with pytest.raises(ValueError, match="reviewed phrases"):
        CATALOG.render_description(rule, phrases + ["synthetic unreviewed legal conclusion"])


@pytest.mark.parametrize("rule_id", [
    "RK_NBK231_P22_RECEIPT_DEPARTURE", "RK_NBK231_P20_PREVIOUS_CASH_POCKET",
])
def test_action_from_another_time_cannot_ground_the_candidate(rule_id):
    action_ids = CATALOG.bindings[rule_id]["violation_fact_ids"]
    result = scenario(rule_id, supports={key: ("synthetic-16",) for key in action_ids})
    assert not result.events
    assert result.verdict is None
    assert any(request.reason == "action_support_outside_candidate" for request in result.refinement_requests)


@pytest.mark.parametrize("rule_id", [
    "RK_NBK231_P22_RECEIPT_DEPARTURE", "RK_NBK231_P20_PREVIOUS_CASH_POCKET",
])
def test_action_timestamp_cannot_borrow_an_earlier_context_frame(rule_id):
    action_ids = CATALOG.bindings[rule_id]["violation_fact_ids"]
    result = scenario(rule_id, supports={key: ("synthetic-6",) for key in action_ids})
    assert result.verdict == "violation"
    assert result.events[0].anchor_frame_id == "synthetic-6"
    assert "synthetic-6" in result.events[0].evidence_frame_ids

