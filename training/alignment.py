"""Private model-bound formatting; canonical reviewed manifests never change."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from forte.decision import (DecisionConfig, build_prompt, compact_frame_request,
                            restore_frame_ids, validate_decisions)
from forte.decision.structured import CONCISE_VISIBILITY_NOTE_MAX_CHARS
from forte.shared.schemas import ClipRequest

from .guards import TrainingGateError, canonical_sha256, require


def _same_decisions(left: Any, right: Any, *, omit_optional_fact_notes: bool = False) -> bool:
    """Compare every canonical semantic field, with an explicit notes-only policy."""
    def fact_values(fact):
        value = fact.model_dump(mode="python")
        if omit_optional_fact_notes:
            value["notes"] = None
        return value

    def event_values(event):
        value = event.model_dump(mode="python")
        value["supporting_facts"] = [fact_values(fact) for fact in event.supporting_facts]
        return value

    return (left.request.model_dump(mode="python") == right.request.model_dump(mode="python")
            and [fact_values(fact) for fact in left.facts] == [fact_values(fact) for fact in right.facts]
            and [event_values(event) for event in left.candidates] == [event_values(event) for event in right.candidates])


def provider_record(canonical: dict[str, Any], *, config: DecisionConfig, media_root: Path) -> dict[str, Any]:
    """Copy one approved canonical window into the selected provider representation.

    With the default config the record is unchanged. Optional modes require the
    frozen full ClipRequest and generic event definitions so the real A08 prompt
    can be rebuilt, rather than editing private strings with replacements.
    The returned copy is transient input for the collator, never a new annotation.
    """
    require(canonical.get("compact_frame_ids", False) is False, "training_manifest_must_use_canonical_ids")
    require(canonical.get("concise_observations", False) is False, "training_manifest_must_preserve_original_notes")
    copied = deepcopy(canonical)
    if not config.compact_frame_ids and not config.concise_observations:
        return copied
    require(isinstance(canonical.get("clip_request"), dict), "compact_training_clip_request_missing")
    require(isinstance(canonical.get("event_definitions"), dict), "compact_training_event_definitions_missing")
    try:
        request_data = deepcopy(canonical["clip_request"])
        request = ClipRequest.model_validate(request_data)
        expected_frames = [(frame["frame_id"], frame["pts_sec"]) for frame in canonical["frames"]]
        require([(frame.frame_id, frame.pts_sec) for frame in request.frame_refs] == expected_frames,
                "compact_training_frame_binding_mismatch")
        require(request.start_sec == canonical["start_sec"] and request.end_sec == canonical["end_sec"],
                "compact_training_interval_mismatch")
        require([fact.fact_id for fact in request.fact_definitions] == canonical["requested_fact_ids"],
                "compact_training_fact_definition_mismatch")
        # A training export may move approved frame files into its own private
        # mount. Rebind only paths in this copy; source IDs, PTS and dimensions stay.
        for index, frame in enumerate(request_data["frame_refs"]):
            frame["path"] = str((media_root / canonical["frames"][index]["path"]).resolve())
            if canonical["frames"][index].get("sha256"):
                frame["jpeg_sha256"] = canonical["frames"][index]["sha256"]
        request = ClipRequest.model_validate(request_data)
        canonical_config = replace(config, compact_frame_ids=False, concise_observations=False)
        bundle = build_prompt(request, reviewed_event_ids=canonical["reviewed_event_ids"],
                              event_definitions=canonical["event_definitions"], config=canonical_config)
        require(bundle.prompt == canonical["prompt"], "compact_training_canonical_prompt_mismatch")
        canonical_batch = validate_decisions(canonical["target"], request,
                                             reviewed_event_ids=canonical["reviewed_event_ids"],
                                             config=canonical_config, prompt_hash=bundle.prompt_hash)
        if config.concise_observations:
            # Omit only optional fact prose in the transient model target. The
            # canonical review retains it. Visibility caveats are never trimmed.
            for fact in copied["target"]["facts"]:
                fact["notes"] = None
            for event in copied["target"]["candidates"]:
                notes = event["visibility_notes"]
                require(len(notes) <= 1 and all(len(note) <= CONCISE_VISIBILITY_NOTE_MAX_CHARS for note in notes),
                        "concise_training_visibility_requires_review")
        aliases = None
        provider_request = request
        if config.compact_frame_ids:
            # Authoritative A08 mapping, exactly once for this window.
            aliases = compact_frame_request(request)
            mapping = aliases.canonical_to_alias
            provider_request = aliases.request
            for frame in copied["frames"]:
                frame["frame_id"] = mapping[frame["frame_id"]]
            for fact in copied["target"]["facts"]:
                fact["supporting_frame_ids"] = [mapping[frame_id] for frame_id in fact["supporting_frame_ids"]]
            for event in copied["target"]["candidates"]:
                event["start_frame_id"] = mapping[event["start_frame_id"]]
                event["end_frame_id"] = mapping[event["end_frame_id"]]
                event["supporting_frame_ids"] = [mapping[frame_id] for frame_id in event["supporting_frame_ids"]]
        provider_bundle = build_prompt(provider_request, reviewed_event_ids=canonical["reviewed_event_ids"],
                                       event_definitions=canonical["event_definitions"], config=config)
        provider_batch = validate_decisions(copied["target"], provider_request,
                                            reviewed_event_ids=canonical["reviewed_event_ids"], config=config,
                                            prompt_hash=provider_bundle.prompt_hash)
        restored = restore_frame_ids(provider_batch, aliases) if aliases is not None else provider_batch
        require(_same_decisions(canonical_batch, restored, omit_optional_fact_notes=config.concise_observations),
                "compact_training_roundtrip_mismatch")
        copied.update(prompt=provider_bundle.prompt, prompt_hash=provider_bundle.prompt_hash,
                      prompt_version=provider_bundle.prompt_version, config_version=provider_bundle.config_version,
                      compact_frame_ids=config.compact_frame_ids, concise_observations=config.concise_observations)
        return copied
    except TrainingGateError:
        raise
    except Exception:
        raise TrainingGateError("compact_training_alignment_invalid") from None


def provider_records(canonical_records: list[dict[str, Any]], *, config: DecisionConfig,
                     media_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return transient copies and an aggregate fingerprint, without exposing maps."""
    aligned = [provider_record(record, config=config, media_root=media_root) for record in canonical_records]
    fingerprint = canonical_sha256({"canonical_records_sha256": canonical_sha256(canonical_records),
                                    "provider_records_sha256": canonical_sha256(aligned),
                                    "compact_frame_ids": config.compact_frame_ids,
                                    "concise_observations": config.concise_observations})
    return aligned, {"compact_frame_ids": config.compact_frame_ids, "concise_observations": config.concise_observations,
                     "aligned_windows": len(aligned),
                     "training_alignment_sha256": fingerprint}
