"""Turn validated local observations into grounded, distinct source events.

This module performs no model inference and never reads videos, labels or frame
pixels. Supporting IDs must resolve to the caller's original ``FrameRef`` map.
The binding is deliberately explicit: a public legal document alone does not
make a visual event applicable to a particular banking procedure.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Literal

from forte.shared.schemas import CandidateEvent, FactDecision, FrameRef, RuleRecord, VerifiedEvent
from forte.decision.scopes import candidate_local_transition


@dataclass(frozen=True)
class UnresolvedInterval:
    start_sec: float
    end_sec: float
    reason: str
    candidate_id: str | None = None
    event_type_id: str | None = None
    technical: bool = False


@dataclass(frozen=True)
class RefinementRequest:
    start_sec: float
    end_sec: float
    reason: str
    candidate_ids: tuple[str, ...] = ()
    required_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventBuildConfig:
    """Runtime catalog bindings, never model-generated legal references.

    ``rule_bindings`` maps rule IDs to metadata containing ``event_type_id`` and
    ``required_fact_states``. ``applicability_fact_ids`` and ``violation_fact_ids``
    are shorthand for present facts. ``temporal_fact_states`` and
    ``required_context_fact_ids`` encode sequence/context requirements.
    ``exception_fact_ids`` must each be visibly absent. Rules with textual
    exceptions require these IDs or a reviewed ``exceptions_verified`` flag.
    ``is_omission`` additionally requires both complete procedure context and
    continuous visibility. ``min_context_sec`` requests longer refinement when
    the successful clip is too short. ``event_types`` can be the A03 taxonomy's
    event-type mapping or sequence; its requirements are added to the binding.
    """

    rule_bindings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    event_types: Mapping[str, Any] | tuple[Any, ...] | list[Any] = field(default_factory=dict)
    minimum_overlap_fraction: float = 0.5
    refinement_padding_sec: float = 4.0
    timeline_tolerance_sec: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.minimum_overlap_fraction <= 1:
            raise ValueError("minimum_overlap_fraction must lie in (0, 1]")
        if not isfinite(self.refinement_padding_sec) or self.refinement_padding_sec < 0:
            raise ValueError("refinement_padding_sec must be finite and nonnegative")
        if not isfinite(self.timeline_tolerance_sec) or self.timeline_tolerance_sec < 0:
            raise ValueError("timeline_tolerance_sec must be finite and nonnegative")


@dataclass(frozen=True)
class EventBuildResult:
    verified_events: tuple[VerifiedEvent, ...]
    unresolved_intervals: tuple[UnresolvedInterval, ...]
    refinement_requests: tuple[RefinementRequest, ...]
    coverage_complete: bool
    nonblocking_limitations: tuple[str, ...] = ()

    @property
    def events(self) -> tuple[VerifiedEvent, ...]:
        return self.verified_events

    @property
    def can_publish(self) -> bool:
        # A clear positive remains a positive alongside unclear visual context.
        # Technical coverage loss can conceal additional events and blocks any
        # completed result. An uncertain negative can never become no_violation.
        return self.coverage_complete and bool(
            self.verified_events or not self.unresolved_intervals
        )

    @property
    def verdict(self) -> Literal["violation", "no_violation"] | None:
        if not self.can_publish:
            return None
        return "violation" if self.verified_events else "no_violation"

    @property
    def limitations(self) -> tuple[str, ...]:
        return tuple(sorted({item.reason for item in self.unresolved_intervals} | set(self.nonblocking_limitations)))


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    raise TypeError("expected a mapping, to_dict(), or contract model")


def _interval(obj: Any) -> tuple[float, float]:
    start, end = _get(obj, "start_sec"), _get(obj, "end_sec")
    request = _get(obj, "request")
    if start is None and request is not None:
        start, end = _get(request, "start_sec"), _get(request, "end_sec")
    if start is None or end is None:
        raise ValueError("successful decisions require an explicit source interval")
    start, end = float(start), float(end)
    if not isfinite(start) or not isfinite(end) or start < 0 or end < start:
        raise ValueError("invalid decision source interval")
    return start, end


def _covers(intervals: Iterable[tuple[float, float]], duration: float, tolerance: float) -> bool:
    cursor = 0.0
    for start, end in sorted(intervals):
        if start > cursor + tolerance:
            return False
        cursor = max(cursor, end)
    return cursor >= duration - tolerance


def _catalog(rules: Any, config: EventBuildConfig) -> tuple[dict[str, RuleRecord], dict[str, Any]]:
    bindings = dict(_get(rules, "bindings", {}) or {})
    bindings.update(config.rule_bindings)
    records = _get(rules, "rules", rules)
    records = records.values() if isinstance(records, Mapping) else records
    result: dict[str, RuleRecord] = {}
    for raw in records:
        rule = raw if isinstance(raw, RuleRecord) else RuleRecord.model_validate(raw)
        if rule.rule_id in result:
            raise ValueError("duplicate rule IDs")
        result[rule.rule_id] = rule
    return result, bindings


def _taxonomy(config: EventBuildConfig) -> dict[str, Any]:
    raw = config.event_types
    if isinstance(raw, Mapping):
        if "event_types" in raw:
            raw = raw["event_types"]
        else:
            return dict(raw)
    return {_get(value, "event_type_id"): value for value in raw}


def _requirements(rule: RuleRecord, binding: Any, event_type: Any) -> tuple[dict[str, str], bool]:
    required: dict[str, str] = {fact_id: "present" for fact_id in rule.required_facts}
    omission = bool(_get(binding, "is_omission", False) or _get(event_type, "is_omission", False))
    for source in (event_type, binding):
        for name in ("applicability_fact_ids", "violation_fact_ids", "required_context_fact_ids", "context_fact_ids"):
            for fact_id in _get(source, name, ()):
                if fact_id in required and required[fact_id] != "present":
                    raise ValueError("contradictory required fact states")
                required[fact_id] = "present"
        for name in ("required_fact_states", "temporal_fact_states"):
            for fact_id, state in _get(source, name, {}).items():
                if state not in {"present", "absent"}:
                    raise ValueError("required fact states must be present or absent")
                if fact_id in required and required[fact_id] != state:
                    raise ValueError("contradictory required fact states")
                required[fact_id] = state
        for fact_id in _get(source, "exception_fact_ids", ()):
            if fact_id in required and required[fact_id] != "absent":
                raise ValueError("contradictory exception fact state")
            required[fact_id] = "absent"
    if omission:
        required.update(PROCEDURE_CONTEXT_COMPLETE="present", WORK_AREA_CONTINUOUSLY_VISIBLE="present")
    return required, omission


_KNOWLEDGE_GATES = frozenset({
    "WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE", "ALL_COUNTING_PATHS_VISIBLE",
    "PREVIOUSLY_ACCEPTED_CASH_TRACKED", "PREVIOUS_RECEIPT_COMPLETE",
})


@dataclass(frozen=True)
class RuleRelevance:
    enabled_rule_ids: tuple[str, ...]
    refuted_rule_ids: tuple[str, ...]
    undetermined_rule_ids: tuple[str, ...]
    uncertain_fact_ids: tuple[str, ...]
    has_uncertainty: bool

    @property
    def all_branches_refuted(self) -> bool:
        return bool(self.enabled_rule_ids) and not self.undetermined_rule_ids

    @property
    def requires_refinement(self) -> bool:
        return self.has_uncertainty and not self.all_branches_refuted


def analyze_rule_relevance(batch: Any, rules: Any, *, config: EventBuildConfig | None = None,
                           event_type_id: str | None = None) -> RuleRelevance:
    """Distinguish irrelevant unknowns from an undetermined legal branch.

    A refuting condition must have the supported opposite state over the actual
    request frame scope. Missing visibility, procedure boundaries or tracking is
    lack of knowledge, not evidence that a rule is inapplicable. Observed counting
    alone cannot globally refute an omission: it may belong to another procedure.
    """
    config = config or EventBuildConfig()
    records, bindings = _catalog(rules, config)
    taxonomy = _taxonomy(config)
    facts = {_get(fact, "fact_id"): fact for fact in _get(batch, "facts", ())}
    uncertain = tuple(sorted(key for key, fact in facts.items() if _get(fact, "state") == "unclear"))
    has_uncertainty = bool(uncertain or _get(batch, "disposition") == "unclear"
                           or _get(batch, "has_uncertainty", False))
    request = _get(batch, "request")
    frame_ids = {_get(frame, "frame_id") for frame in _get(request, "frame_refs", ())}
    if not frame_ids:
        frame_ids = {key for fact in facts.values() for key in _get(fact, "supporting_frame_ids", ())}
    enabled = []
    refuted = []
    context_gaps: set[str] = set()
    for rule_id, rule in sorted(records.items()):
        binding = bindings.get(rule_id, {})
        branch_type = _get(binding, "event_type_id")
        if not rule.enabled or (event_type_id is not None and branch_type != event_type_id):
            continue
        enabled.append(rule_id)
        try:
            required, _ = _requirements(rule, binding, taxonomy.get(branch_type, {}))
        except ValueError:
            continue
        gates = set(_KNOWLEDGE_GATES)
        for name in ("context_fact_ids", "required_context_fact_ids"):
            gates.update(_get(binding, name, ()))
        gates.update(_get(binding, "temporal_fact_states", {}))
        gates.update(_get(taxonomy.get(branch_type, {}), "required_context_fact_ids", ()))
        for name in ("context_complete_fact_id", "visibility_fact_id"):
            value = _get(binding, name)
            if value:
                gates.add(value)
        for fact_id, required_state in required.items():
            if fact_id in gates or fact_id not in facts:
                continue
            fact = facts[fact_id]
            state = _get(fact, "state")
            support = set(_get(fact, "supporting_frame_ids", ()))
            if state not in {"present", "absent"} or state == required_state or not support:
                continue
            if not frame_ids or not frame_ids <= support:
                continue
            if fact_id == "CASH_COUNT_VISIBLE" and state == "present":
                continue
            refuted.append(rule_id)
            break
        if rule_id not in refuted:
            context_gaps.update(fact_id for fact_id in gates & required.keys()
                                if fact_id not in facts or _get(facts[fact_id], "state") != required[fact_id]
                                or not _get(facts[fact_id], "supporting_frame_ids", ()))
    uncertain = tuple(sorted(set(uncertain) | context_gaps))
    has_uncertainty = bool(has_uncertainty or context_gaps)
    return RuleRelevance(tuple(enabled), tuple(refuted), tuple(key for key in enabled if key not in refuted),
                         uncertain, has_uncertainty)


@dataclass
class _Supported:
    candidate: CandidateEvent
    rule: RuleRecord
    facts: list[FactDecision]
    clip_id: str
    omission: bool
    action_frame_ids: frozenset[str] = frozenset()


@dataclass
class _Group:
    members: list[_Supported]
    common_start: float
    common_end: float


def _support_ids(item: _Supported) -> set[str]:
    ids = set(item.candidate.supporting_frame_ids)
    for fact in item.facts:
        ids.update(fact.supporting_frame_ids)
    return ids


def _same_occurrence(group: _Group, item: _Supported, overlap_fraction: float) -> bool:
    first = group.members[0]
    candidate = item.candidate
    if (first.rule.rule_id, first.candidate.event_type_id) != (item.rule.rule_id, candidate.event_type_id):
        return False
    # A model may identify two same-type actions in one clip; retain that split.
    if any(member.clip_id == item.clip_id for member in group.members):
        return False
    overlap = min(group.common_end, candidate.end_sec) - max(group.common_start, candidate.start_sec)
    if overlap <= 0:
        # Identical point events need the exact same original frame as support.
        return (candidate.start_sec == candidate.end_sec == group.common_start == group.common_end
                and bool(_support_ids(first) & _support_ids(item)))
    shared = set().union(*(_support_ids(member) for member in group.members)) & _support_ids(item)
    smaller = min(group.common_end - group.common_start, candidate.end_sec - candidate.start_sec)
    return bool(shared) or (smaller > 0 and overlap / smaller >= overlap_fraction)


def _event(group: _Group, frames: dict[str, FrameRef]) -> VerifiedEvent | None:
    members = group.members
    start = min(member.candidate.start_sec for member in members)
    end = max(member.candidate.end_sec for member in members)
    ids = set().union(*(_support_ids(member) for member in members))
    available = sorted((frames[key] for key in ids), key=lambda frame: (frame.pts_sec, frame.frame_id))
    action_frame_ids = set().union(*(member.action_frame_ids for member in members))
    anchors = [frame for frame in available if start <= frame.pts_sec <= end
               and (not action_frame_ids or frame.frame_id in action_frame_ids)]
    # Candidate verification below guarantees at least one in-interval anchor.
    target = end if members[0].omission else start
    facts: dict[tuple[str, str], FactDecision] = {}
    for member in members:
        for fact in member.facts:
            key = (fact.fact_id, fact.state)
            if key not in facts:
                facts[key] = fact.model_copy(deep=True)
            else:
                ids = sorted(set(facts[key].supporting_frame_ids) | set(fact.supporting_frame_ids))
                facts[key] = facts[key].model_copy(update={"supporting_frame_ids": ids})
    # Exact bounded set cover: every required affirmative fact is represented in
    # the JPEGs a reviewer receives, rather than only in hidden internal IDs.
    affirmative = [fact for fact in facts.values() if fact.state == "present"]
    full_mask = (1 << len(affirmative)) - 1
    masks = {frame.frame_id: sum(1 << index for index, fact in enumerate(affirmative)
                                if frame.frame_id in fact.supporting_frame_ids)
             for frame in available}
    # Identical support masks have identical set-cover value. Retain the first
    # source frame for each mask and use integer bitsets to locate a partner
    # covering the remaining facts, avoiding a cubic search over all frames.
    representatives: dict[int, FrameRef] = {}
    for frame in available:
        representatives.setdefault(masks[frame.frame_id], frame)
    cover_frames = list(representatives.values())
    fact_coverers = [sum(1 << index for index, frame in enumerate(cover_frames)
                         if masks[frame.frame_id] & (1 << fact_index))
                     for fact_index in range(len(affirmative))]
    selected: set[str] | None = None
    anchor: FrameRef | None = None
    tried_anchor_masks: set[int] = set()
    for proposed_anchor in sorted(anchors, key=lambda frame: (abs(frame.pts_sec - target), frame.pts_sec, frame.frame_id)):
        anchor_mask = masks[proposed_anchor.frame_id]
        if anchor_mask in tried_anchor_masks:
            continue
        tried_anchor_masks.add(anchor_mask)
        if anchor_mask == full_mask:
            selected = {proposed_anchor.frame_id}
        else:
            for before in cover_frames:
                combined = anchor_mask | masks[before.frame_id]
                if combined == full_mask:
                    selected = {proposed_anchor.frame_id, before.frame_id}
                    break
                needed = full_mask & ~combined
                partners = (1 << len(cover_frames)) - 1
                while needed and partners:
                    bit = needed & -needed
                    partners &= fact_coverers[bit.bit_length() - 1]
                    needed ^= bit
                if partners:
                    index = (partners & -partners).bit_length() - 1
                    selected = {proposed_anchor.frame_id, before.frame_id, cover_frames[index].frame_id}
                    break
        if selected is not None:
            anchor = proposed_anchor
            break
    if anchor is None or selected is None:
        return None
    for frame in (available[0], available[-1]):
        if len(selected) < 3:
            selected.add(frame.frame_id)
    evidence = sorted(selected, key=lambda key: (frames[key].pts_sec, key))
    limitations = sorted({note for member in members for note in member.candidate.visibility_notes})
    if members[0].omission:
        limitations.append("omission_anchor:last_supported_frame_at_procedure_end")
    identity = [members[0].candidate.event_type_id, members[0].rule.rule_id, start, end,
                sorted(member.candidate.candidate_id for member in members), evidence]
    event_id = "event_" + sha256(json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()[:24]
    return VerifiedEvent(
        event_id=event_id, event_type_id=members[0].candidate.event_type_id,
        rule_id=members[0].rule.rule_id, start_sec=start, end_sec=end,
        anchor_frame_id=anchor.frame_id, evidence_frame_ids=evidence,
        observed_facts=[facts[key] for key in sorted(facts)], limitations=limitations,
    )


def build_events(
    decisions: Iterable[Any],
    rules: Any,
    frames: Mapping[str, FrameRef] | Iterable[FrameRef],
    coverage: Any,
    *,
    config: EventBuildConfig | None = None,
) -> EventBuildResult:
    """Aggregate A08 batches against an A04 catalog and A06 coverage report.

    Batches expose ``request``, ``facts``, ``candidates``, ``complete`` and
    ``disposition`` (``complete`` or ``unclear``). Mappings with the same fields
    are supported for deterministic fixtures. Technical errors/truncations must
    remain missing coverage, never empty successful batches. ``verdict`` is None
    until publication is justified; it is never a third public verdict value.
    """
    config = config or EventBuildConfig()
    report = _mapping(coverage)
    duration = float(report.get("duration_sec", 0.0))
    if not isfinite(duration) or duration <= 0:
        raise ValueError("coverage requires finite positive duration_sec")
    values = frames.values() if isinstance(frames, Mapping) else frames
    frame_map: dict[str, FrameRef] = {}
    for raw in values:
        frame = raw if isinstance(raw, FrameRef) else FrameRef.model_validate(raw)
        if frame.frame_id in frame_map and frame_map[frame.frame_id] != frame:
            raise ValueError("conflicting source frame IDs")
        if frame.pts_sec > duration + config.timeline_tolerance_sec:
            raise ValueError("source frame lies outside video timeline")
        frame_map[frame.frame_id] = frame
    records, bindings = _catalog(rules, config)
    taxonomy = _taxonomy(config)
    unresolved: list[UnresolvedInterval] = []
    refinements: list[RefinementRequest] = []
    successful_intervals: list[tuple[float, float]] = []
    supported: list[_Supported] = []
    seen_candidates: dict[tuple[str, str], str] = {}
    nonblocking: set[str] = set()

    def defer(candidate: CandidateEvent, reason: str, required: Iterable[str] = ()) -> None:
        unresolved.append(UnresolvedInterval(candidate.start_sec, candidate.end_sec, reason,
                                             candidate.candidate_id, candidate.event_type_id))
        refinements.append(RefinementRequest(
            max(0.0, candidate.start_sec - config.refinement_padding_sec),
            min(duration, candidate.end_sec + config.refinement_padding_sec), reason,
            (candidate.candidate_id,), tuple(sorted(required)),
        ))

    for batch in decisions:
        start, end = _interval(batch)
        if end > duration + config.timeline_tolerance_sec:
            raise ValueError("decision interval lies outside source timeline")
        complete = _get(batch, "complete", False)
        disposition = _get(batch, "disposition", "error")
        if complete is not True or disposition not in {"complete", "unclear"}:
            unresolved.append(UnresolvedInterval(start, end, "incomplete_model_analysis", technical=True))
            continue
        successful_intervals.append((start, end))
        request = _get(batch, "request")
        clip_id = str(_get(batch, "clip_id", _get(request, "clip_id", f"{start}:{end}")))
        request_frames = _get(request, "frame_refs", ())
        allowed_frames = {_get(frame, "frame_id") for frame in request_frames} if request_frames else set(frame_map)
        batch_facts: dict[str, FactDecision] = {}
        duplicate_facts: set[str] = set()
        for raw in _get(batch, "facts", ()):
            fact = raw if isinstance(raw, FactDecision) else FactDecision.model_validate(raw)
            if fact.fact_id in batch_facts:
                duplicate_facts.add(fact.fact_id)
            batch_facts[fact.fact_id] = fact
        batch_support = {key for fact in batch_facts.values() for key in fact.supporting_frame_ids}
        invalid_batch = bool(duplicate_facts or batch_support - frame_map.keys() or batch_support - allowed_frames)
        if request is not None and hasattr(request, "validate_decisions"):
            try:
                request.validate_decisions(list(batch_facts.values()))
            except ValueError:
                invalid_batch = True
        if invalid_batch:
            unresolved.append(UnresolvedInterval(start, end, "invalid_batch_visual_support", technical=True))
            continue
        relevance = analyze_rule_relevance(batch, {"rules": records, "bindings": bindings}, config=config)
        if relevance.requires_refinement:
            unresolved.append(UnresolvedInterval(start, end, "unclear_visual_context"))
            refinements.append(RefinementRequest(start, end, "unclear_visual_context",
                                                 required_fact_ids=relevance.uncertain_fact_ids))
        elif relevance.has_uncertainty:
            nonblocking.add("unclear_visual_context_irrelevant_to_refuted_rules")
        candidates = _get(batch, "candidates", _get(batch, "candidate_events", ()))
        for raw in candidates:
            candidate = raw if isinstance(raw, CandidateEvent) else CandidateEvent.model_validate(raw)
            candidate_clip = candidate.clip_id or clip_id
            identity = (candidate_clip, candidate.candidate_id)
            signature = candidate.model_dump_json()
            if identity in seen_candidates:
                if seen_candidates[identity] != signature:
                    defer(candidate, "conflicting_candidate_id")
                    supported[:] = [item for item in supported
                                    if (item.clip_id, item.candidate.candidate_id) != identity]
                continue
            seen_candidates[identity] = signature
            if candidate.start_sec < start or candidate.end_sec > end:
                defer(candidate, "candidate_outside_analyzed_context")
                continue
            if candidate.clip_id is not None and candidate.clip_id != clip_id:
                defer(candidate, "candidate_clip_mismatch")
                continue
            if taxonomy and candidate.event_type_id not in taxonomy:
                defer(candidate, "unknown_event_type")
                continue
            matches = [(records[rule_id], binding) for rule_id, binding in sorted(bindings.items())
                       if rule_id in records and _get(binding, "event_type_id") == candidate.event_type_id
                       and records[rule_id].enabled and records[rule_id].applicability_verified
                       and records[rule_id].verification_state in {"source_verified", "verified"}]
            if not matches:
                defer(candidate, "no_verified_applicable_rule")
                continue
            facts = dict(batch_facts)
            candidate_fact_ids = [fact.fact_id for fact in candidate.supporting_facts]
            invalid = bool(duplicate_facts or len(candidate_fact_ids) != len(set(candidate_fact_ids)))
            for fact in candidate.supporting_facts:
                if fact.fact_id in facts and facts[fact.fact_id].state != fact.state:
                    if not candidate_local_transition(facts[fact.fact_id], fact,
                            start_sec=candidate.start_sec, end_sec=candidate.end_sec,
                            frame_times={key: frame.pts_sec for key, frame in frame_map.items()},
                            candidate_frame_ids=candidate.supporting_frame_ids):
                        invalid = True
                facts[fact.fact_id] = fact
            all_ids = set(candidate.supporting_frame_ids)
            for fact in facts.values():
                all_ids.update(fact.supporting_frame_ids)
            if invalid or not all_ids <= frame_map.keys() or not all_ids <= allowed_frames:
                defer(candidate, "invalid_or_conflicting_visual_support")
                continue
            # Never use a context-only frame as a fabricated event anchor.
            if not any(candidate.start_sec <= frame_map[key].pts_sec <= candidate.end_sec for key in all_ids):
                defer(candidate, "missing_source_anchor")
                continue
            rejected_bindings: list[tuple[str, Iterable[str]]] = []
            for rule, binding in matches:
                event_type = taxonomy.get(candidate.event_type_id, {})
                try:
                    required, omission = _requirements(rule, binding, event_type)
                except ValueError:
                    rejected_bindings.append(("invalid_rule_fact_binding", ()))
                    continue
                missing = [fact_id for fact_id, state in required.items()
                           if fact_id not in facts or facts[fact_id].state != state
                           or not facts[fact_id].supporting_frame_ids]
                if missing:
                    rejected_bindings.append(("required_visual_facts_not_established", missing))
                    continue
                # Context may legitimately precede/follow the candidate, but a
                # different action elsewhere in the same window cannot ground
                # this occurrence or supply its public timestamp.
                action_support: set[str] = set()
                unsupported_actions = []
                for fact_id in _get(binding, "violation_fact_ids", ()):
                    local_ids = {key for key in facts[fact_id].supporting_frame_ids
                                 if candidate.start_sec <= frame_map[key].pts_sec <= candidate.end_sec}
                    if not local_ids:
                        unsupported_actions.append(fact_id)
                    action_support.update(local_ids)
                if unsupported_actions:
                    rejected_bindings.append(("action_support_outside_candidate", unsupported_actions))
                    continue
                temporal = _get(binding, "temporal_fact_states", {})
                context = _get(binding, "required_context_fact_ids", ()) or _get(binding, "context_fact_ids", ())
                if rule.temporal_conditions and not (temporal or context or omission):
                    rejected_bindings.append(("temporal_conditions_not_encoded", ()))
                    continue
                if rule.exceptions and not (_get(binding, "exception_fact_ids", ())
                                            or _get(binding, "exceptions_verified", False) is True):
                    rejected_bindings.append(("exceptions_not_resolved", ()))
                    continue
                minimum = float(_get(binding, "min_context_sec", 0.0) or 0.0)
                if not isfinite(minimum) or minimum < 0:
                    rejected_bindings.append(("invalid_minimum_context", ()))
                    continue
                if end - start + config.timeline_tolerance_sec < minimum:
                    rejected_bindings.append(("longer_procedure_context_required", required))
                    continue
                selected_facts = [facts[key] for key in sorted(required)]
                if not any(fact.state == "present" for fact in selected_facts):
                    rejected_bindings.append(("missing_affirmative_visual_support", ()))
                    continue
                supported.append(_Supported(candidate, rule, selected_facts, candidate_clip, omission,
                                            frozenset(action_support)))
                # Multiple applicable clauses ground one visual occurrence. The
                # catalog's stable rule-ID order chooses a canonical citation;
                # it must not turn one action into duplicate public events.
                break
            else:
                for reason, required_ids in rejected_bindings:
                    defer(candidate, reason, required_ids)

    coverage_complete = (report.get("complete") is True and not report.get("gaps")
                         and not report.get("missing_window_indices")
                         and _covers(successful_intervals, duration, config.timeline_tolerance_sec)
                         and not any(item.technical for item in unresolved))
    if not coverage_complete:
        unresolved.append(UnresolvedInterval(0.0, duration, "incomplete_analysis_coverage", technical=True))
    groups: list[_Group] = []
    for item in sorted(supported, key=lambda x: (x.candidate.start_sec, x.candidate.end_sec,
                                                x.candidate.event_type_id, x.rule.rule_id,
                                                x.clip_id, x.candidate.candidate_id)):
        group = next((group for group in groups if _same_occurrence(group, item, config.minimum_overlap_fraction)), None)
        if group is None:
            groups.append(_Group([item], item.candidate.start_sec, item.candidate.end_sec))
        else:
            group.members.append(item)
            group.common_start = max(group.common_start, item.candidate.start_sec)
            group.common_end = min(group.common_end, item.candidate.end_sec)
    built_events = []
    for group in groups:
        event = _event(group, frame_map)
        if event is None:
            for member in group.members:
                defer(member.candidate, "evidence_support_exceeds_frame_budget")
        else:
            built_events.append(event)
    events = sorted(built_events,
                    key=lambda event: (frame_map[event.anchor_frame_id].pts_sec, event.event_type_id,
                                       event.rule_id, event.event_id))
    # Preserve uncertainty on each public-facing positive as well as the result.
    limitations = sorted({item.reason for item in unresolved if not item.technical} | nonblocking)
    if limitations:
        events = [event.model_copy(update={"limitations": sorted(set(event.limitations) | set(limitations))})
                  for event in events]
    unresolved = sorted(set(unresolved), key=lambda item: (item.start_sec, item.end_sec, item.reason,
                                                          item.candidate_id or "", item.event_type_id or ""))
    refinements = sorted(set(refinements), key=lambda item: (item.start_sec, item.end_sec, item.reason,
                                                            item.candidate_ids))
    return EventBuildResult(tuple(events), tuple(unresolved), tuple(refinements), coverage_complete, tuple(sorted(nonblocking)))
