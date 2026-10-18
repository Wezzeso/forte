from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from rules.catalog import CATALOG_PATH, load_catalog
from rules.experimental_catalog import (
    EXPERIMENTAL_CATALOG_PATH,
    ExperimentalRuleCatalog,
    RULE_IDS,
    build_experimental_payload,
    load_experimental_catalog,
)


def test_default_catalog_and_all_source_archives_remain_unchanged():
    before = CATALOG_PATH.read_bytes()
    baseline = load_catalog()
    catalog = load_experimental_catalog()
    assert CATALOG_PATH.read_bytes() == before
    assert len(baseline.enabled_rules()) == 4
    assert len(baseline.fact_definitions()) == 19
    assert catalog.experimental_payload["baseline_snapshot"]["catalog_sha256"] == sha256(before).hexdigest()
    assert catalog.payload["sources"] == baseline.payload["sources"]
    assert catalog.payload["schema_version"] == "forte.experimental.rules.v1"
    assert set(catalog.rules) == set(RULE_IDS)


def test_production_loader_rejects_experimental_raw_schema():
    with pytest.raises(ValueError, match="unsupported regulatory catalog schema"):
        load_catalog(EXPERIMENTAL_CATALOG_PATH)


def test_build_is_deterministic_and_combining_baseline_requires_opt_in():
    raw = json.loads(EXPERIMENTAL_CATALOG_PATH.read_text(encoding="utf-8"))
    assert build_experimental_payload() == raw
    assert len(load_experimental_catalog().rules) == 4
    combined = load_experimental_catalog(include_baseline=True)
    assert len(combined.rules) == 8
    assert combined.is_experimental is True and combined.requires_temporal_wrapper is True


def test_new_axis_manifest_preserves_baseline_and_rejects_nineteen_axis_head():
    baseline = load_catalog()
    catalog = load_experimental_catalog()
    actual = [value.fact_id for value in catalog.axis_fact_definitions()]
    previous = [value.fact_id for value in baseline.fact_definitions()]
    assert len(actual) == 37
    assert actual[:19] == previous
    assert len(catalog.vocabulary["new_axis_ids"]) == 18
    assert all(value.startswith("EXP_") for value in catalog.vocabulary["new_axis_ids"])
    catalog.validate_axis_order(actual, vocabulary_version=catalog.vocabulary_version)
    with pytest.raises(ValueError, match="axis order mismatch"):
        catalog.validate_axis_order(previous, vocabulary_version=catalog.vocabulary_version)
    with pytest.raises(ValueError, match="vocabulary version mismatch"):
        catalog.validate_axis_order(actual, vocabulary_version="baseline.v1")
    with pytest.raises(ValueError, match="axis order mismatch"):
        catalog.validate_axis_order(actual[::-1], vocabulary_version=catalog.vocabulary_version)


def test_phase_only_facts_are_requested_and_have_reviewed_definitions():
    catalog = load_experimental_catalog()
    requested = {value.fact_id for value in catalog.fact_definitions()}
    for rule_id in RULE_IDS:
        temporal = catalog.bindings[rule_id]["experimental_temporal"]
        for phase in temporal["ordered_phases"]:
            assert set(phase["fact_ids"]) <= requested
            for fact_id in phase["fact_ids"]:
                assert catalog.facts[fact_id]["question_ru"]
                assert catalog.facts[fact_id]["definition_ru"]
    without_trust = {value.fact_id: value.description for value in catalog.fact_definitions()}
    assert "должно быть unclear" in without_trust["BANK_CASH_WORKSTATION"]


def test_prior_cash_requires_distinct_ordered_receipt_episodes_and_one_track():
    catalog = load_experimental_catalog()
    temporal = catalog.bindings[RULE_IDS[0]]["experimental_temporal"]
    assert temporal["distinct_procedure_ids"] is True
    assert temporal["strict_phase_order"] is True
    assert list(temporal["phase_procedure_roles"].values()) == ["prior", "prior", "current", "current"]
    assert temporal["track_phase_ids"]["cash"] == [phase["phase_id"] for phase in temporal["ordered_phases"]]
    assert "EXP_PRIOR_CASH_IDLE_ON_ORDINARY_COUNTER" in catalog.rules[RULE_IDS[0]].required_facts
    assert temporal["event_phase_id"] == "prior_cash_idle_during_next_receipt"
    assert temporal["candidate_interval_policy"] == "event_phase_source_bounds"


def test_client_absence_is_confined_to_count_phase_not_initial_contribution():
    catalog = load_experimental_catalog()
    binding = catalog.bindings[RULE_IDS[1]]
    temporal = binding["experimental_temporal"]
    assert binding["required_fact_states"]["EXP_ANY_CLIENT_AREA_PERSON_PRESENT"] == "absent"
    assert temporal["absence_must_cover_procedure"] is False
    assert temporal["absence_phase_ids"] == ["count_with_client_area_empty"]
    assert temporal["track_phase_ids"]["contributor"] == ["contribution", "client_leaves"]
    assert "EXP_ALL_CLIENT_AREAS_CONTINUOUSLY_VISIBLE" in binding["context_fact_ids"]


@pytest.mark.parametrize("rule_id", RULE_IDS[2:])
def test_count_omissions_require_full_operation_and_every_counting_path(rule_id):
    catalog = load_experimental_catalog()
    binding = catalog.bindings[rule_id]
    temporal = binding["experimental_temporal"]
    assert binding["is_omission"] is True
    assert binding["required_fact_states"]["CASH_COUNT_VISIBLE"] == "absent"
    assert temporal["absence_must_cover_procedure"] is True
    assert temporal["counting_paths_must_cover_procedure"] is True
    assert {"PROCEDURE_CONTEXT_COMPLETE", "ALL_COUNTING_PATHS_VISIBLE", "WORK_AREA_CONTINUOUSLY_VISIBLE"} <= set(catalog.rules[rule_id].required_facts)
    assert len(temporal["required_procedure_boundary_fact_ids"]) == 2
    assert temporal["max_visibility_gap_sec"] is None  # no invented statutory duration


def test_every_rule_uses_reviewed_text_and_renders_only_its_phrases():
    catalog = load_experimental_catalog()
    for rule in catalog.enabled_rules():
        binding = catalog.bindings[rule.rule_id]
        phrases = [catalog.fact_description(key, state) for key, state in binding["required_fact_states"].items()]
        rendered = catalog.render_description(rule, phrases)
        assert rule.title in rendered and rule.clause in rendered and rule.source_url in rendered
        assert "Возможные признаки" in rendered
        with pytest.raises(ValueError, match="reviewed phrases"):
            catalog.render_description(rule, ["fabricated legal conclusion"])


def test_no_authority_identity_phone_or_ppe_rule_is_added():
    catalog = load_experimental_catalog()
    assert set(catalog.rules) == set(RULE_IDS)
    assert set(catalog.event_types) == {
        "EXP_PREVIOUS_CASH_OUTSIDE_STORAGE", "EXP_CLIENT_ABSENT_DURING_RECEIPT_COUNT", "CASH_TRANSFER_WITHOUT_OBSERVED_COUNT"
    }
    assert catalog.experimental_payload["manual_annotation_prerequisite"] is False
    assert all(binding["experimental_temporal"]["face_identification_permitted"] is False for binding in catalog.bindings.values())


def test_invalid_temporal_roles_or_missing_omission_scope_fail_closed():
    original = json.loads(EXPERIMENTAL_CATALOG_PATH.read_text(encoding="utf-8"))
    payload = deepcopy(original)
    del payload["bindings"][RULE_IDS[0]]["experimental_temporal"]["phase_procedure_roles"]["prior_receipt_complete"]
    with pytest.raises(ValueError, match="every phase"):
        ExperimentalRuleCatalog(payload, root=EXPERIMENTAL_CATALOG_PATH.parent)
    payload = deepcopy(original)
    payload["bindings"][RULE_IDS[2]]["experimental_temporal"]["absence_must_cover_procedure"] = False
    with pytest.raises(ValueError, match="complete absence"):
        ExperimentalRuleCatalog(payload, root=EXPERIMENTAL_CATALOG_PATH.parent)
