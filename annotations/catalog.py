"""Generic public visual taxonomy. Contains no private annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.v1.json")


def load_taxonomy(path: str | Path = TAXONOMY_PATH) -> dict[str, Any]:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    facts = {fact["fact_id"] for fact in catalog["facts"]}
    if len(facts) != len(catalog["facts"]):
        raise ValueError("duplicate fact ID in taxonomy")
    event_ids = [event["event_type_id"] for event in catalog["event_types"]]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("duplicate event ID in taxonomy")
    for event in catalog["event_types"]:
        if not set(event["required_fact_states"]).issubset(facts):
            raise ValueError("unknown required fact in taxonomy")
        if not set(event.get("required_context_fact_ids", [])).issubset(facts):
            raise ValueError("unknown context fact in taxonomy")
    return catalog


def fact_definitions() -> dict[str, dict[str, Any]]:
    return {fact["fact_id"]: fact for fact in load_taxonomy()["facts"]}


def event_definitions() -> dict[str, dict[str, Any]]:
    return {event["event_type_id"]: event for event in load_taxonomy()["event_types"]}

