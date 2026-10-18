from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from rules.catalog import CATALOG_PATH, RuleCatalog, load_catalog, load_rules


def test_official_archives_and_shared_contracts_validate():
    catalog = load_catalog()
    assert len(catalog.enabled_rules()) == 4
    assert set(load_rules()) == set(catalog.rules)
    assert all(rule.clause and rule.source_sha256 for rule in catalog.enabled_rules())
    assert {rule.accessed_at.isoformat() for rule in catalog.enabled_rules()} == {"2026-10-17"}


def test_changed_archived_source_fails_closed(tmp_path):
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for source in payload["sources"]:
        for archive in source["archives"]:
            target = tmp_path / archive["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((CATALOG_PATH.parent / archive["path"]).read_bytes())
    altered = tmp_path / payload["sources"][0]["archives"][0]["path"]
    altered.write_bytes(altered.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        RuleCatalog(payload, root=tmp_path)


def test_bank_scope_is_never_inferred_when_operator_context_missing():
    catalog = load_catalog()
    ordinary = {item.fact_id: item.description for item in catalog.fact_definitions()}
    trusted = {item.fact_id: item.description for item in catalog.fact_definitions(trusted_scene_context="bank_cash_workstation")}
    assert "должно быть unclear" in ordinary["BANK_CASH_WORKSTATION"]
    assert "Дополнительно установите по кадрам" in trusted["BANK_CASH_WORKSTATION"]
    assert ordinary.keys() == trusted.keys()


def test_all_enabled_facts_and_event_ids_match_generic_taxonomy():
    catalog = load_catalog()
    taxonomy_path = CATALOG_PATH.parent.parent / "annotations" / "taxonomy.v1.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    fact_ids = {item["fact_id"] for item in taxonomy["facts"]}
    event_ids = {item["event_type_id"] for item in taxonomy["event_types"]}
    assert {item.fact_id for item in catalog.fact_definitions()}.issubset(fact_ids)
    assert {binding["event_type_id"] for binding in catalog.bindings.values()}.issubset(event_ids)


def test_omission_loses_authority_without_complete_counting_visibility():
    catalog = load_catalog()
    payload = copy.deepcopy(catalog.payload)
    rule = next(item for item in payload["rules"] if item["rule_id"] == "RK_NBK231_P19_2_RECEIPT_COUNT")
    rule["required_facts"].remove("ALL_COUNTING_PATHS_VISIBLE")
    binding = payload["bindings"][rule["rule_id"]]
    del binding["required_fact_states"]["ALL_COUNTING_PATHS_VISIBLE"]
    binding["context_fact_ids"].remove("ALL_COUNTING_PATHS_VISIBLE")
    binding["applicability_fact_ids"].remove("ALL_COUNTING_PATHS_VISIBLE")
    with pytest.raises(ValueError, match="omission requires"):
        RuleCatalog(payload, root=catalog.root)


def test_reviewed_description_handles_absence_without_raw_model_notes():
    catalog = load_catalog()
    rule = catalog.rules["RK_NBK231_P19_2_RECEIPT_COUNT"]
    phrases = [catalog.fact_description(key, state) for key, state in catalog.bindings[rule.rule_id]["required_fact_states"].items()]
    result = catalog.render_description(rule, phrases)
    assert rule.title in result and rule.clause in result and rule.source_url in result
    assert "Возможные признаки" in result
    with pytest.raises(ValueError, match="reviewed phrases"):
        catalog.render_description(rule, ["arbitrary unreviewed model-generated legal claim"])
    with pytest.raises(ValueError, match="unclear"):
        catalog.fact_description("CASH_COUNT_VISIBLE", "unclear")


def test_unreviewed_limitation_or_modified_rule_is_rejected():
    catalog = load_catalog()
    rule = catalog.enabled_rules()[0]
    binding = catalog.bindings[rule.rule_id]
    phrases = [catalog.fact_description(key, state) for key, state in binding["required_fact_states"].items()]
    with pytest.raises(ValueError, match="limitation"):
        catalog.render_description(rule, phrases, ["untrusted arbitrary model notes"])
    with pytest.raises(ValueError, match="differs"):
        catalog.render_description(rule.model_copy(update={"clause": "fabricated clause"}), phrases)


def test_unobservable_authority_identity_and_gesture_gaps_are_explicit():
    catalog = load_catalog()
    unresolved = [item for item in catalog.coverage_gaps if not item["covered_by"]]
    assert len(unresolved) == 5
    assert not any(binding["event_type_id"] == "WORKSPACE_ACCESS_BY_OTHER_PERSON" for binding in catalog.bindings.values())
