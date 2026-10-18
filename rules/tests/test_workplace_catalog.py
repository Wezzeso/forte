"""Optional public-source workplace profile, entirely synthetic evidence.

No private footage/annotations/model outputs are imported or read.
"""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from forte.decision import DecisionBatch, build_prompt
from forte.events import build_events
from forte.shared.schemas import CandidateEvent, ClipRequest, FactDecision, FrameRef
from rules.catalog import CATALOG_PATH, load_catalog
from rules.workplace_catalog import (
    PROFILE_PATH, RULE_IDS, TAXONOMY_PATH, VOCABULARY_VERSION, WorkplaceRuleCatalog,
    build_workplace_payload, build_workplace_taxonomy, load_workplace_catalog,
)


CATALOG = load_workplace_catalog()
PERSONAL, CAMERA, SMOKING = RULE_IDS
FRAMES = {
    f"synthetic-{index}": FrameRef(frame_id=f"synthetic-{index}", pts_sec=float(index),
        path=f"synthetic/workplace-{index}.jpg", width=32, height=32, job_id="synthetic-workplace")
    for index in range(9)
}
COVERAGE = {"duration_sec": 8.0, "complete": True, "gaps": [], "missing_window_indices": []}


def scenario(rule_id, *, changes=None, omitted=(), supports=None):
    states = dict(CATALOG.bindings[rule_id]["required_fact_states"])
    states.update(changes or {})
    facts = tuple(FactDecision(fact_id=key, state=state,
        supporting_frame_ids=[] if state == "unclear" else list(
            (supports or {}).get(key, ("synthetic-4", "synthetic-5", "synthetic-6"))))
        for key, state in states.items() if key not in omitted)
    event = CandidateEvent(candidate_id="synthetic-workplace-action",
        event_type_id=CATALOG.bindings[rule_id]["event_type_id"], start_sec=4.0, end_sec=6.0,
        supporting_facts=list(facts), supporting_frame_ids=["synthetic-5"], clip_id="synthetic-profile")
    requested = [definition for definition in CATALOG.fact_definitions(trusted_scene_context="bank_cash_node")
                 if definition.fact_id in {fact.fact_id for fact in facts}]
    request = ClipRequest(job_id="synthetic-workplace", clip_id="synthetic-profile", start_sec=0.0,
                          end_sec=8.0, frame_refs=list(FRAMES.values()), fact_definitions=requested)
    batch = DecisionBatch(request=request, facts=facts, candidates=(event,))
    return build_events([batch], CATALOG, FRAMES, COVERAGE)


def test_optional_raw_profile_is_rejected_by_default_loader_and_never_changes_baseline():
    before = CATALOG_PATH.read_bytes()
    with pytest.raises(ValueError, match="unsupported regulatory catalog schema"):
        load_catalog(PROFILE_PATH)
    assert len(load_catalog().enabled_rules()) == 4
    assert len(load_catalog().fact_definitions()) == 19
    assert CATALOG_PATH.read_bytes() == before
    assert CATALOG.profile_payload["baseline_snapshot"]["catalog_sha256"] == sha256(before).hexdigest()
    assert not CATALOG.deployment_selected


def test_generated_catalogue_and_separate_taxonomy_are_deterministic():
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert payload == build_workplace_payload()
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    assert taxonomy == build_workplace_taxonomy(payload)
    assert len(taxonomy["event_types"]) == len(CATALOG.enabled_rules()) == 3
    assert set(CATALOG.rules) == set(RULE_IDS)
    assert [fact["fact_id"] for fact in taxonomy["facts"]] == CATALOG.vocabulary["fact_ids"]


def test_adding_baseline_requires_explicit_optional_loader_argument():
    combined = load_workplace_catalog(include_baseline=True)
    assert combined.includes_baseline and combined.is_optional
    assert not combined.deployment_selected
    assert len(combined.enabled_rules()) == 7
    assert len(combined.fact_definitions()) == 32
    assert set(combined.rules) == set(CATALOG.rules) | set(load_catalog().rules)
    assert len(CATALOG.enabled_rules()) == 3 and not CATALOG.includes_baseline


def test_expanded_axes_fit_existing_strict_physical_observation_limit_without_legal_generation():
    definitions = CATALOG.axis_fact_definitions(trusted_scene_context="bank_cash_node")
    request = ClipRequest(job_id="synthetic-workplace", clip_id="synthetic-axes", start_sec=0.0,
                          end_sec=8.0, frame_refs=list(FRAMES.values()), fact_definitions=definitions,
                          rule_definitions=CATALOG.enabled_rules())
    bundle = build_prompt(request, reviewed_event_ids=tuple(CATALOG.event_types),
                          event_definitions=CATALOG.event_types)
    assert bundle.response_schema["properties"]["facts"]["maxItems"] == 32
    assert "пункт 12, часть первая" not in bundle.prompt
    assert "подпункт 9 пункта 5 статьи 110" not in bundle.prompt
    assert "пункта 12" not in bundle.prompt and "статьи 110" not in bundle.prompt
    assert "adilet.zan.kz" not in bundle.prompt


def test_thirty_two_axis_identity_preserves_nineteen_axis_prefix_and_requires_new_version():
    ordered = [fact.fact_id for fact in CATALOG.axis_fact_definitions()]
    original = [fact.fact_id for fact in load_catalog().fact_definitions()]
    assert len(ordered) == 32 and ordered[:19] == original
    assert len(CATALOG.vocabulary["new_axis_ids"]) == 13
    assert all(value.startswith("WP_") for value in ordered[19:])
    CATALOG.validate_axis_order(ordered, vocabulary_version=VOCABULARY_VERSION)
    for bad in (original, ordered[::-1]):
        with pytest.raises(ValueError, match="axis order mismatch"):
            CATALOG.validate_axis_order(bad, vocabulary_version=VOCABULARY_VERSION)
    with pytest.raises(ValueError, match="vocabulary version mismatch"):
        CATALOG.validate_axis_order(ordered, vocabulary_version="baseline.v1")


def test_workstation_attestation_alone_cannot_confirm_cash_node_scope():
    for context in (None, "bank_cash_workstation"):
        definitions = {fact.fact_id: fact.description for fact in CATALOG.fact_definitions(trusted_scene_context=context)}
        assert "состояние должно быть unclear" in definitions["WP_CONFIRMED_BANK_CASH_NODE"]
    definitions = {fact.fact_id: fact.description for fact in CATALOG.fact_definitions(trusted_scene_context="bank_cash_node")}
    assert "дополнительно подтвердите по кадрам" in definitions["WP_CONFIRMED_BANK_CASH_NODE"]


@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_positive_exactly_supported_profile_branches_use_original_action_anchor(rule_id):
    action_id = CATALOG.bindings[rule_id]["violation_fact_ids"][0]
    result = scenario(rule_id, supports={action_id: ("synthetic-6",)})
    assert result.verdict == "violation" and len(result.events) == 1
    event = result.events[0]
    assert event.rule_id == rule_id and event.anchor_frame_id == "synthetic-6"
    assert 1 <= len(event.evidence_frame_ids) <= 3
    assert set(event.evidence_frame_ids) <= set(FRAMES)
    phrases = [CATALOG.fact_description(fact.fact_id, fact.state) for fact in event.observed_facts]
    description = CATALOG.render_description(rule_id, phrases)
    assert "Возможные признаки" in description
    assert CATALOG.rules[rule_id].clause in description and CATALOG.rules[rule_id].source_url in description


GATE_CASES = [(rule_id, key, state) for rule_id in RULE_IDS
              for key, state in CATALOG.bindings[rule_id]["required_fact_states"].items()]


@pytest.mark.parametrize("rule_id,fact_id,state", GATE_CASES)
@pytest.mark.parametrize("condition", ["missing", "unclear", "opposite"])
def test_every_applicability_action_continuity_or_exception_gate_is_required(rule_id, fact_id, state, condition):
    omitted = (fact_id,) if condition == "missing" else ()
    changes = {} if omitted else {fact_id: "unclear" if condition == "unclear" else
                                  "absent" if state == "present" else "present"}
    result = scenario(rule_id, changes=changes, omitted=omitted)
    assert not result.events and result.verdict is None
    assert result.refinement_requests


@pytest.mark.parametrize("name,rule_id,changes", [
    ("unknown rectangle is not a communication device", CAMERA, {"WP_COMM_DEVICE_IDENTIFIABLE": "unclear"}),
    ("generic phone does not prove camera", CAMERA, {"WP_RECORDING_CAPABILITY_VISIBLE": "unclear"}),
    ("camera-free handset is a permitted exception", CAMERA, {"WP_ALLOWED_MEDICAL_OR_NONCAMERA_COMM": "present", "WP_RECORDING_CAPABILITY_VISIBLE": "absent"}),
    ("medical inhaler is permitted", PERSONAL, {"WP_ALLOWED_MEDICAL_OR_NONCAMERA_COMM": "present"}),
    ("unknown bag contents require uncertainty", PERSONAL, {"WP_ALLOWED_MEDICAL_OR_NONCAMERA_COMM": "unclear"}),
    ("bank operational equipment is not personal", CAMERA, {"WP_PERSONAL_ITEM_PROVENANCE_CLEAR": "absent"}),
    ("customer handset in lobby is outside scope", CAMERA, {"WP_CONFIRMED_BANK_CASH_NODE": "absent", "WP_WORKER_ROLE_BY_CASH_ACTIONS": "absent"}),
    ("worn clothing does not prove storage", PERSONAL, {"WP_PERSONAL_STORAGE_EPISODE_VISIBLE": "absent"}),
    ("worker appearance does not prove role", PERSONAL, {"WP_WORKER_ROLE_BY_CASH_ACTIONS": "unclear"}),
    ("holding a pen is not tobacco consumption", SMOKING, {"WP_TOBACCO_PRODUCT_CLEAR": "absent", "WP_TOBACCO_CONSUMPTION_VISIBLE": "absent"}),
    ("unknown vapor is not tobacco", SMOKING, {"WP_TOBACCO_PRODUCT_CLEAR": "unclear"}),
    ("holding unlit tobacco does not prove smoking", SMOKING, {"WP_TOBACCO_CONSUMPTION_VISIBLE": "absent"}),
    ("outdoor smoking is outside this branch", SMOKING, {"WP_CONFIRMED_INDOOR_WORKPLACE": "absent"}),
])
def test_ordinary_lookalikes_and_unknowns_never_become_profile_findings(name, rule_id, changes):
    result = scenario(rule_id, changes=changes)
    assert not result.events, name
    assert result.verdict is None, name


def test_reviewed_sources_and_exceptions_cannot_be_removed_or_replaced():
    for key in ("rules", "sources", "bindings", "facts"):
        payload = deepcopy(CATALOG.profile_payload)
        if key == "rules":
            payload[key][0]["source_sha256"] = "0" * 64
        elif key == "sources":
            payload[key][-1]["url"] = "https://example.invalid/fake-source"
        elif key == "bindings":
            del payload[key][PERSONAL]["required_fact_states"]["WP_ALLOWED_MEDICAL_OR_NONCAMERA_COMM"]
        else:
            payload[key]["WP_RECORDING_CAPABILITY_VISIBLE"]["definition_ru"] = "Any rectangle"
        with pytest.raises(ValueError, match="reviewed"):
            WorkplaceRuleCatalog(payload, root=PROFILE_PATH.parent)


def test_unreviewed_model_legal_text_cannot_enter_public_description():
    with pytest.raises(ValueError, match="reviewed phrases"):
        CATALOG.render_description(PERSONAL, ["synthetic arbitrary disciplinary conclusion"])
    rule = CATALOG.rules[PERSONAL]
    with pytest.raises(ValueError, match="differs"):
        CATALOG.render_description(rule.model_copy(update={"clause": "fabricated"}),
                                  [CATALOG.fact_description("WP_PERSONAL_STORAGE_EPISODE_VISIBLE")])


def test_source_archives_integrity_is_verified_before_explicit_profile_load(tmp_path):
    payload = deepcopy(CATALOG.profile_payload)
    for source in payload["sources"]:
        for archive in source["archives"]:
            target = tmp_path / archive["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((PROFILE_PATH.parent / archive["path"]).read_bytes())
    path = tmp_path / "workplace_sources/health_code_content.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        WorkplaceRuleCatalog(payload, root=tmp_path)

