"""Local JSON size proxies over public fact IDs; no model, tokenizer or media IO.

Run with PYTHONPATH=src from the project root. The output is aggregate metadata
only; character/byte counts are not token counts or latency measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

from forte.decision import DecisionConfig, validate_decisions
from forte.shared.schemas import ClipRequest, FrameRef
from rules.catalog import load_catalog


def measure() -> dict:
    definitions = load_catalog().fact_definitions(enabled_only=True)
    clip = ClipRequest(
        job_id="synthetic-size-only", clip_id="synthetic-size-window",
        frame_refs=[FrameRef(frame_id=f"f{index:02d}", pts_sec=float(index),
                             path=f"/synthetic-only/{index}.jpg", width=32, height=32)
                    for index in range(16)],
        fact_definitions=definitions,
    )
    verbose = {"facts": [{
        "fact_id": definition.fact_id, "state": "unclear", "supporting_frame_ids": [],
        "notes": "Синтетический пример: обзор и контекст недостаточны для уверенного наблюдения.",
    } for definition in definitions], "candidates": []}
    null_notes = {"facts": [{**fact, "notes": None} for fact in verbose["facts"]], "candidates": []}
    optional_notes_omitted = {"facts": [
        {key: value for key, value in fact.items() if key != "notes"} for fact in null_notes["facts"]
    ], "candidates": []}
    outputs = {}
    for name, raw, config in (
        ("synthetic_repeated_visibility_notes", verbose, DecisionConfig()),
        ("concise_null_notes", null_notes, DecisionConfig(concise_observations=True)),
        ("fixed_fact_set_minimum_without_optional_notes", optional_notes_omitted,
         DecisionConfig(concise_observations=True)),
    ):
        batch = validate_decisions(raw, clip, reviewed_event_ids=(), config=config)
        assert batch.disposition == "unclear"
        text = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        outputs[name] = {"characters": len(text), "utf8_bytes": len(text.encode("utf-8")),
                         "facts_validated": len(batch.facts), "candidates_validated": len(batch.candidates)}
    return {
        "schema_version": "forte.synthetic-json-size-proxy.v1",
        "synthetic_only": True, "fact_count": len(definitions), "frame_count": 16,
        "measurement": "compact JSON character and UTF-8 byte counts; proxies only",
        "token_counts_measured": False, "inference_performed": False,
        "scope": "fixed public fact IDs and synthetic unclear decisions; no actual-output assumption",
        "outputs": outputs,
    }


if __name__ == "__main__":
    result = measure()
    Path(__file__).with_name("synthetic-size-proxy.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
