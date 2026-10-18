"""Offline, checksum-verified regulatory catalog. Never reads private annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from forte.shared.schemas import FactDefinition, RuleRecord

CATALOG_PATH = Path(__file__).with_name("catalog.json")
OFFICIAL_HOSTS = frozenset({"adilet.zan.kz", "old.adilet.zan.kz", "nationalbank.kz", "zan.gov.kz", "law.gov.kz"})


class RuleCatalog:
    def __init__(self, payload: dict[str, Any], *, root: Path, verify_archives: bool = True):
        if payload.get("schema_version") != "forte.rules.v1":
            raise ValueError("unsupported regulatory catalog schema")
        self.payload = payload
        self.root = root.resolve()
        self.version: str = payload["catalog_version"]
        parsed = [RuleRecord.model_validate(item) for item in payload["rules"]]
        self.rules = {rule.rule_id: rule for rule in parsed}
        if len(parsed) != len(self.rules):
            raise ValueError("duplicate rule ID")
        self.bindings: dict[str, dict[str, Any]] = payload["bindings"]
        self.facts: dict[str, dict[str, str]] = payload["facts"]
        self.event_types: dict[str, dict[str, Any]] = payload["event_types"]
        self.coverage_gaps: list[dict[str, Any]] = payload["coverage_gaps"]
        self.sources: dict[str, dict[str, Any]] = {s["source_id"]: s for s in payload["sources"]}
        if verify_archives:
            for source in self.sources.values():
                self._verify_source(source)
        self._validate_bindings()

    def _verify_source(self, source: dict[str, Any]) -> None:
        if urlparse(source["url"]).hostname not in OFFICIAL_HOSTS:
            raise ValueError("regulatory source must use a reviewed official host")
        for archive in source["archives"]:
            path = (self.root / archive["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("source archive path escapes regulatory catalog")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != archive["sha256"]:
                raise ValueError(f"regulatory source checksum mismatch: {archive['path']}")

    def _validate_bindings(self) -> None:
        for rule_id, rule in self.rules.items():
            binding = self.bindings.get(rule_id)
            if binding is None:
                raise ValueError(f"rule missing binding: {rule_id}")
            source = self.sources[binding["source_id"]]
            if rule.source_sha256 != source["primary_sha256"] or rule.source_url != source["url"]:
                raise ValueError("rule source differs from its reviewed archive")
            if rule.clause != binding["clause"]:
                raise ValueError("rule clause differs from reviewed binding")
            if binding["event_type_id"] not in self.event_types:
                raise ValueError("unknown event type in rule binding")
            states = binding["required_fact_states"]
            if not states or not set(states).issubset(self.facts):
                raise ValueError("unknown or empty required facts in binding")
            if any(state not in {"present", "absent"} for state in states.values()):
                raise ValueError("rule bindings require explicit present/absent states")
            if set(rule.required_facts) != {key for key, state in states.items() if state == "present"}:
                raise ValueError("shared RuleRecord must include every affirmative applicability/violation/context fact")
            for key in binding.get("applicability_fact_ids", []) + binding.get("context_fact_ids", []):
                if states.get(key) != "present":
                    raise ValueError("applicability and context facts must be affirmative")
            text = (self.root / source["text_archive_path"]).read_text(encoding="utf-8")
            if " ".join(binding["source_excerpt"].split()) not in " ".join(text.split()):
                raise ValueError("reviewed clause excerpt missing from archived source")
            if rule.enabled:
                if source["legal_status"] != "current_official_version":
                    raise ValueError("enabled rule requires current verified source")
                if not binding.get("temporal_verified") or not binding.get("exceptions_verified"):
                    raise ValueError("enabled rule requires executable temporal and exception gates")
                if binding.get("is_omission") and not {
                    "PROCEDURE_CONTEXT_COMPLETE", "WORK_AREA_CONTINUOUSLY_VISIBLE", "ALL_COUNTING_PATHS_VISIBLE"
                }.issubset(rule.required_facts):
                    raise ValueError("omission requires complete observable procedure and counting paths")

    def enabled_rules(self) -> list[RuleRecord]:
        return [rule for rule in self.rules.values() if rule.enabled]

    def get_rule(self, rule_id: str, *, require_enabled: bool = True) -> RuleRecord:
        rule = self.rules[rule_id]
        if require_enabled and not rule.enabled:
            raise ValueError("disabled regulatory mapping cannot ground a result")
        return rule

    def fact_definitions(self, *, enabled_only: bool = True, trusted_scene_context: str | None = None) -> list[FactDefinition]:
        ids = set(self.facts)
        if enabled_only:
            ids = {key for rule in self.enabled_rules() for key in self.bindings[rule.rule_id]["required_fact_states"]}
        result = []
        for key in sorted(ids):
            description = f"{self.facts[key]['question_ru']} {self.facts[key]['definition_ru']}"
            if key == "BANK_CASH_WORKSTATION":
                if trusted_scene_context == "bank_cash_workstation":
                    description += (" Доверенная настройка локального оператора подтверждает область банковской кассы."
                                    " Дополнительно установите по кадрам наличие кассового рабочего места;"
                                    " сама настройка не подтверждает видимость или действия конкретного человека.")
                else:
                    description += " Доверенный контекст не задан: состояние этого факта должно быть unclear."
            result.append(FactDefinition(fact_id=key, description=description, required=True))
        return result

    def fact_description(self, fact_id: str, state: str = "present") -> str:
        if state not in {"present", "absent"}:
            raise ValueError("unclear facts have no affirmative evidence phrase")
        key = "description_ru" if state == "present" else "absence_description_ru"
        phrase = self.facts[fact_id].get(key)
        if not phrase:
            raise ValueError("fact has no reviewed description for requested state")
        return phrase

    def event_type_label(self, event_type_id: str) -> str:
        return self.event_types[event_type_id]["name_ru"]

    def render_description(self, rule: RuleRecord | str, observed_facts: Iterable[str], limitations: Iterable[str] | None = None) -> str:
        record = self.get_rule(rule if isinstance(rule, str) else rule.rule_id)
        if isinstance(rule, RuleRecord) and record.model_dump() != rule.model_dump():
            raise ValueError("rule differs from trusted catalog")
        # Accept only deterministic reviewed phrases. Model notes and generated legal
        # conclusions are never interpolated into a public response.
        binding = self.bindings[record.rule_id]
        allowed = {self.fact_description(key, state) for key, state in binding["required_fact_states"].items()}
        phrases = list(dict.fromkeys(observed_facts))
        if not phrases or any(phrase not in allowed for phrase in phrases):
            raise ValueError("descriptions must contain reviewed phrases for this rule")
        approved_limits = set(self.payload["approved_limitations_ru"])
        limits = list(dict.fromkeys(limitations or []))
        if any(limit not in approved_limits for limit in limits):
            raise ValueError("limitation text is not a reviewed catalog phrase")
        public_ids = binding.get("description_fact_ids", list(binding["required_fact_states"]))
        public_phrases = [self.fact_description(key, binding["required_fact_states"][key]) for key in public_ids]
        observation = "; ".join(phrase for phrase in public_phrases if phrase in phrases)
        if not observation:
            raise ValueError("description requires the reviewed event observation")
        text = f"Наблюдаемые признаки: {observation}. {binding['implication_ru']} Источник: {record.title}, {record.clause}. {record.source_url}"
        if limits:
            text += " Ограничения: " + "; ".join(limits) + "."
        return text


def load_catalog(path: str | Path | None = None, *, verify_archives: bool = True) -> RuleCatalog:
    selected = Path(path) if path is not None else CATALOG_PATH
    return RuleCatalog(json.loads(selected.read_text(encoding="utf-8")), root=selected.parent, verify_archives=verify_archives)


def load_rules(path: str | Path | None = None) -> dict[str, RuleRecord]:
    return load_catalog(path).rules


def render_description(rule: RuleRecord | str, observed_facts: Iterable[str], limitations: Iterable[str] | None = None,
                       *, catalog: RuleCatalog | None = None) -> str:
    return (catalog or load_catalog()).render_description(rule, observed_facts, limitations)
