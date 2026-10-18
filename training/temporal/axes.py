"""Frozen live nineteen-fact axis, distinct from the larger generic vocabulary."""
from pathlib import Path

from training.guards import file_sha256, load_json, require


def load_axes(config: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    path = Path(config["fact_axis_manifest"]).resolve()
    require(file_sha256(path) == config.get("expected_fact_axis_manifest_sha256"), "weak_fact_axis_pin_mismatch")
    axis = load_json(path)
    taxonomy = Path(config["taxonomy_path"]).resolve()
    require(file_sha256(taxonomy) == config.get("expected_taxonomy_sha256") == axis.get("taxonomy_sha256"),
            "weak_taxonomy_pin_mismatch")
    source = load_json(taxonomy)
    facts, events = tuple(axis["fact_ids"]), tuple(axis["event_type_ids"])
    require(axis.get("schema_version") == "forte.temporal-fact-axis.v1"
            and axis.get("fact_axis_origin") == "enabled_catalog_required_fact_states_sorted"
            and axis.get("state_order") == ["absent", "present", "unclear"]
            and len(facts) == len(set(facts)) == 19 and facts == tuple(sorted(facts))
            and set(facts) <= {fact["fact_id"] for fact in source["facts"]}
            and len(events) == len(set(events)) and set(events) <= {event["event_type_id"] for event in source["event_types"]},
            "weak_full_live_fact_axis_invalid")
    return facts, events
