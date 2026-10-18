"""Optional private request-local frame names, restored before event assembly.

This does not change source media identity, shared contracts or evidence IDs.
It only gives the local inference boundary short, deterministic identifiers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from forte.shared.schemas import CandidateEvent, ClipRequest, FactDecision

from .structured import DecisionBatch, DecisionConfigurationError, DecisionOutputError
from .scopes import candidate_local_transition


@dataclass(frozen=True, slots=True, repr=False)
class FrameAliasMap:
    """Private one-request mapping; never serialize this into a model prompt.

    ``request`` is the copied inference request. ``canonical_request`` is the
    unchanged input used by the returned decision batch and downstream evidence.
    Both mappings are read-only. Snapshots detect accidental provider mutation.
    """

    request: ClipRequest = field(repr=False)
    canonical_request: ClipRequest = field(repr=False)
    alias_to_canonical: Mapping[str, str] = field(repr=False)
    canonical_to_alias: Mapping[str, str] = field(repr=False)
    _canonical_snapshot: dict[str, Any] = field(repr=False, compare=False)
    _inference_snapshot: dict[str, Any] = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return f"FrameAliasMap(frames={len(self.alias_to_canonical)}, mapping=<private>)"


def compact_frame_request(request: ClipRequest) -> FrameAliasMap:
    """Copy a trusted clip and assign ``f00``, ``f01``, ... in supplied order.

    Call exactly once per clip. Input order is retained; the ordinary prompt
    validator still requires chronological order. Paths, checksums, timestamps,
    image dimensions, job/clip IDs and criteria are copied without alteration.
    This helper is also the authoritative input mapping for training targets.
    """
    try:
        canonical_snapshot = request.model_dump(mode="python")
        copied = ClipRequest.model_validate(canonical_snapshot)
    except (ValueError, TypeError):
        raise DecisionConfigurationError("invalid_frame_alias_request") from None
    if not 1 <= len(copied.frame_refs) <= 96:
        raise DecisionConfigurationError("invalid_frame_alias_count")
    alias_to_canonical = {
        f"f{index:02d}": frame.frame_id for index, frame in enumerate(copied.frame_refs)
    }
    inference_snapshot = copied.model_dump(mode="python")
    for index, frame in enumerate(inference_snapshot["frame_refs"]):
        frame["frame_id"] = f"f{index:02d}"
    try:
        inference = ClipRequest.model_validate(inference_snapshot)
    except ValidationError:
        raise DecisionConfigurationError("invalid_frame_alias_request") from None
    # Take separate snapshots: model instances and caller-owned lists remain
    # mutable under the shared contracts, while this boundary must stay bound.
    return FrameAliasMap(
        request=inference,
        canonical_request=request,
        alias_to_canonical=MappingProxyType(alias_to_canonical),
        canonical_to_alias=MappingProxyType({value: key for key, value in alias_to_canonical.items()}),
        _canonical_snapshot=request.model_dump(mode="python"),
        _inference_snapshot=inference.model_dump(mode="python"),
    )


def _validate_alias_binding(batch: DecisionBatch, aliases: FrameAliasMap) -> None:
    if (aliases.canonical_request.model_dump(mode="python") != aliases._canonical_snapshot
            or aliases.request.model_dump(mode="python") != aliases._inference_snapshot
            or batch.request.model_dump(mode="python") != aliases._inference_snapshot):
        raise DecisionOutputError("frame_alias_request_changed")
    expected = {
        f"f{index:02d}": frame.frame_id
        for index, frame in enumerate(aliases.canonical_request.frame_refs)
    }
    if (dict(aliases.alias_to_canonical) != expected
            or dict(aliases.canonical_to_alias) != {value: key for key, value in expected.items()}):
        raise DecisionOutputError("inconsistent_frame_alias_mapping")


def restore_frame_ids(batch: DecisionBatch, aliases: FrameAliasMap) -> DecisionBatch:
    """Return canonical shared decisions after complete alias-bound validation.

    Unknown aliases, mixed canonical/alias supports, changed media metadata and
    inconsistent candidate facts fail. Only structured frame references are
    translated; free-text notes remain untrusted data. No evidence assembler
    receives the aliased request from ``DecisionEngine``.
    """
    _validate_alias_binding(batch, aliases)
    mapping = aliases.alias_to_canonical
    known_facts = {definition.fact_id for definition in aliases.request.fact_definitions}
    fact_ids = [fact.fact_id for fact in batch.facts]
    if len(fact_ids) != len(set(fact_ids)) or set(fact_ids) != known_facts:
        raise DecisionOutputError("frame_alias_fact_set_mismatch")

    def restored_support(ids: list[str]) -> list[str]:
        if len(ids) != len(set(ids)):
            raise DecisionOutputError("duplicate_support_id")
        if not set(ids).issubset(mapping):
            raise DecisionOutputError("unknown_frame_alias")
        return [mapping[frame_id] for frame_id in ids]

    original_facts = {fact.fact_id: fact for fact in batch.facts}
    facts: dict[str, FactDecision] = {}
    for fact in batch.facts:
        values = fact.model_dump(mode="python")
        values["supporting_frame_ids"] = restored_support(fact.supporting_frame_ids)
        try:
            facts[fact.fact_id] = FactDecision.model_validate(values)
        except ValidationError:
            raise DecisionOutputError("invalid_restored_fact") from None
    try:
        aliases.canonical_request.validate_decisions(list(facts.values()))
    except ValueError:
        raise DecisionOutputError("invalid_restored_fact_binding") from None

    frame_times = {frame.frame_id: frame.pts_sec for frame in aliases.request.frame_refs}
    timestamps = set(frame_times.values())
    candidates: list[CandidateEvent] = []
    candidate_ids: set[str] = set()
    for candidate in batch.candidates:
        if candidate.candidate_id in candidate_ids:
            raise DecisionOutputError("duplicate_candidate_id")
        candidate_ids.add(candidate.candidate_id)
        restored_ids = restored_support(candidate.supporting_frame_ids)
        candidate_fact_ids = [fact.fact_id for fact in candidate.supporting_facts]
        if (not candidate_fact_ids or len(candidate_fact_ids) != len(set(candidate_fact_ids))
                or not set(candidate_fact_ids).issubset(facts)):
            raise DecisionOutputError("inconsistent_aliased_candidate_facts")
        restored_candidate_facts = []
        for fact in candidate.supporting_facts:
            values = fact.model_dump(mode="python")
            values["supporting_frame_ids"] = restored_support(fact.supporting_frame_ids)
            original = original_facts[fact.fact_id]
            if (fact.model_dump(mode="python") != original.model_dump(mode="python")
                    and not candidate_local_transition(original, fact,
                        start_sec=candidate.start_sec, end_sec=candidate.end_sec,
                        frame_times=frame_times, candidate_frame_ids=candidate.supporting_frame_ids)):
                raise DecisionOutputError("inconsistent_aliased_candidate_facts")
            try:
                restored_candidate_facts.append(FactDecision.model_validate(values))
            except ValidationError:
                raise DecisionOutputError("invalid_restored_fact") from None
        if (candidate.clip_id != aliases.request.clip_id
                or candidate.start_sec not in timestamps or candidate.end_sec not in timestamps
                or candidate.start_sec > candidate.end_sec
                or any(not candidate.start_sec <= frame_times[frame_id] <= candidate.end_sec
                       for frame_id in candidate.supporting_frame_ids)):
            raise DecisionOutputError("inconsistent_aliased_candidate_interval")
        cited_support = {
            frame_id for fact in candidate.supporting_facts for frame_id in fact.supporting_frame_ids
        }
        if not cited_support.intersection(candidate.supporting_frame_ids):
            raise DecisionOutputError("candidate_without_factual_support")
        values = candidate.model_dump(mode="python")
        values["supporting_frame_ids"] = restored_ids
        values["supporting_facts"] = restored_candidate_facts
        try:
            candidates.append(CandidateEvent.model_validate(values))
        except ValidationError:
            raise DecisionOutputError("invalid_restored_candidate") from None
    return replace(
        batch,
        request=aliases.canonical_request,
        facts=tuple(facts[fact.fact_id] for fact in batch.facts),
        candidates=tuple(candidates),
    )
