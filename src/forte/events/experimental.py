"""Explicitly isolated, versioned temporal verification for experimental rules.

Nothing imports this module from the production pipeline. Inputs are local
observations over original source IDs; opaque cash/body tracks describe visual
continuity, never face identity. A verified experimental event is still a review
proposal. Candidate-only input cannot establish a whole-video negative verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forte.shared.schemas import CandidateEvent, FactDecision, FrameRef, VideoAsset, VerifiedEvent
from forte.events.builder import RefinementRequest, _Group, _Supported, _event, _same_occurrence


class ExperimentalTrack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    track_id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    frame_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_frames(self):
        if len(self.frame_ids) != len(set(self.frame_ids)):
            raise ValueError("experimental track frames must be unique")
        return self


class ExperimentalTemporalContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    context_id: str = Field(min_length=1, max_length=256)
    procedure_id: str = Field(min_length=1, max_length=256)
    prior_procedure_id: str | None = None
    full_context_start_frame_id: str = Field(min_length=1)
    full_context_end_frame_id: str = Field(min_length=1)
    inspected_frame_ids: tuple[str, ...] = Field(min_length=1)
    phase_frame_ids: dict[str, tuple[str, ...]]
    phase_procedure_ids: dict[str, str]
    tracks: dict[Literal["cash", "contributor"], ExperimentalTrack]

    @model_validator(mode="after")
    def unique_scope(self):
        if len(self.inspected_frame_ids) != len(set(self.inspected_frame_ids)):
            raise ValueError("experimental inspected frames must be unique")
        if any(not values or len(values) != len(set(values)) for values in self.phase_frame_ids.values()):
            raise ValueError("experimental phase frames must be nonempty and unique")
        return self


class ExperimentalEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rule_id: str = Field(min_length=1, max_length=256)
    vocabulary_version: str = Field(min_length=1, max_length=256)
    candidate: CandidateEvent
    context: ExperimentalTemporalContext


@dataclass(frozen=True)
class ExperimentalPending:
    rule_id: str
    context_id: str
    candidate_id: str
    reason: str
    missing_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentalSourceReference:
    event_id: str
    rule_id: str
    source_url: str
    clause: str
    source_sha256: str


@dataclass(frozen=True)
class ExperimentalResult:
    catalog_version: str
    vocabulary_version: str
    verified_events: tuple[VerifiedEvent, ...]
    pending: tuple[ExperimentalPending, ...]
    refinement_requests: tuple[RefinementRequest, ...]
    source_references: tuple[ExperimentalSourceReference, ...]
    episodes_checked: int
    runtime_enabled: Literal[False] = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "forte.experimental.temporal_result.v1",
            "catalog_version": self.catalog_version, "vocabulary_version": self.vocabulary_version,
            "verified_events": [event.model_dump(mode="json") for event in self.verified_events],
            "pending": [asdict(item) for item in self.pending],
            "refinement_requests": [asdict(item) for item in self.refinement_requests],
            "source_references": [asdict(item) for item in self.source_references],
            "episodes_checked": self.episodes_checked, "runtime_enabled": False,
        }


def _value(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _coverage_dict(coverage: Any) -> dict:
    if isinstance(coverage, Mapping):
        return dict(coverage)
    if hasattr(coverage, "to_dict"):
        return coverage.to_dict()
    raise ValueError("experimental coverage report is required")


def _covers(intervals, start: float, end: float) -> bool:
    cursor = start
    for left, right in sorted(intervals):
        if right < cursor:
            continue
        if left > cursor + 1e-6:
            return False
        cursor = max(cursor, right)
        if cursor >= end - 1e-6:
            return True
    return False


def _source_frames(asset: VideoAsset, frames: Mapping[str, FrameRef] | Sequence[FrameRef]) -> dict[str, FrameRef]:
    result = {}
    for raw in frames.values() if isinstance(frames, Mapping) else frames:
        frame = FrameRef.model_validate(raw)
        frame.validate_timeline(asset)
        if frame.job_id != asset.job_id or frame.source_pts is None or asset.time_base is None:
            raise ValueError("experimental frames require explicit original source ownership and PTS")
        expected = sha256(f"{asset.content_hash}:{frame.source_pts}:{asset.time_base}".encode("ascii")).hexdigest()
        if frame.frame_id != expected:
            raise ValueError("experimental frame identity differs from source PTS")
        if frame.frame_id in result and result[frame.frame_id] != frame:
            raise ValueError("experimental frame IDs conflict")
        result[frame.frame_id] = frame
    if isinstance(frames, Mapping) and any(key != frame.frame_id for key, frame in frames.items()):
        raise ValueError("experimental frame map keys differ from source IDs")
    return result


def _facts(episode: ExperimentalEpisode):
    result = {}
    for fact in episode.candidate.supporting_facts:
        if fact.fact_id in result:
            return None
        result[fact.fact_id] = fact
    return result


def validate_experimental_vocabulary(catalog: Any, vocabulary_version: str,
                                     feature_axis_order: Sequence[str]) -> None:
    """Reject stale nineteen-axis checkpoints and reordered experimental heads."""
    if (getattr(catalog, "is_experimental", False) is not True
            or vocabulary_version != getattr(catalog, "vocabulary_version", None)
            or list(feature_axis_order) != catalog.vocabulary["feature_axis_order"]):
        raise ValueError("experimental model vocabulary manifest does not match the catalog")


def _temporal_error(episode, binding, facts, frames, report):
    context = episode.context
    policy = binding.get("experimental_temporal", {})
    if policy.get("schema_version") != "forte.experimental.temporal.v1":
        return "experimental_temporal_policy_missing", ()
    if policy.get("max_visibility_gap_sec") is not None:
        return "experimental_visibility_gap_policy_unsupported", ()
    declared = set(context.inspected_frame_ids)
    referenced = declared | {context.full_context_start_frame_id, context.full_context_end_frame_id}
    referenced.update(episode.candidate.supporting_frame_ids)
    for fact in facts.values():
        referenced.update(fact.supporting_frame_ids)
    for values in context.phase_frame_ids.values():
        referenced.update(values)
    for track in context.tracks.values():
        referenced.update(track.frame_ids)
    if not referenced <= frames.keys():
        return "experimental_unknown_source_frame", ()
    start = frames[context.full_context_start_frame_id].pts_sec
    end = frames[context.full_context_end_frame_id].pts_sec
    if end < start or not start <= episode.candidate.start_sec <= episode.candidate.end_sec <= end:
        return "experimental_interval_outside_context", ()
    if any(not start <= frames[key].pts_sec <= end for key in referenced):
        return "experimental_support_outside_context", ()
    source_times = {frames[key].pts_sec for key in referenced}
    if episode.candidate.start_sec not in source_times or episode.candidate.end_sec not in source_times:
        return "experimental_candidate_boundary_not_source_pts", ()
    if not {context.full_context_start_frame_id, context.full_context_end_frame_id} <= declared:
        return "experimental_procedure_boundary_not_inspected", ()
    inspected = report.get("inspected_intervals", ())
    if any(set(_value(item, "frame_ids", ())) - frames.keys() for item in inspected):
        return "experimental_coverage_unknown_source_frame", ()
    intervals = [(float(_value(item, "start_sec")), float(_value(item, "end_sec"))) for item in inspected]
    if not _covers(intervals, start, end):
        return "experimental_context_coverage_gap", ()
    actually_inspected = {key for item in inspected for key in _value(item, "frame_ids", ())
                          if key in frames and start <= frames[key].pts_sec <= end}
    if declared != actually_inspected:
        return "experimental_inspected_frame_scope_mismatch", ()
    if not referenced <= declared:
        return "experimental_support_not_inspected", ()

    phases = policy.get("ordered_phases", ())
    phase_ids = [phase.get("phase_id") for phase in phases]
    if not phases or len(phase_ids) != len(set(phase_ids)) or set(context.phase_frame_ids) != set(phase_ids):
        return "experimental_phase_set_mismatch", ()
    if set(context.phase_procedure_ids) != set(phase_ids):
        return "experimental_phase_procedure_scope_missing", ()
    roles = policy.get("phase_procedure_roles", {})
    distinct = policy.get("distinct_procedure_ids", False)
    if distinct and (not context.prior_procedure_id or context.prior_procedure_id == context.procedure_id):
        return "experimental_distinct_procedures_not_established", ()
    previous_end = None
    for phase in phases:
        phase_id = phase["phase_id"]
        ids = set(context.phase_frame_ids[phase_id])
        first, last = min(frames[key].pts_sec for key in ids), max(frames[key].pts_sec for key in ids)
        if previous_end is not None and (first <= previous_end if policy.get("strict_phase_order", False)
                                         else first < previous_end):
            return "experimental_phase_order_not_established", ()
        previous_end = last
        role = roles.get(phase_id, "current")
        expected = context.prior_procedure_id if role == "prior" else context.procedure_id
        if role not in {"prior", "current"} or context.phase_procedure_ids[phase_id] != expected:
            return "experimental_phase_procedure_mismatch", ()
        for fact_id in phase.get("fact_ids", ()):
            fact = facts.get(fact_id)
            state = binding["required_fact_states"].get(fact_id, "present")
            if fact is None or fact.state != state or not ids.intersection(fact.supporting_frame_ids):
                return "experimental_phase_fact_not_supported", (fact_id,)
    if distinct and set(context.phase_procedure_ids.values()) != {context.procedure_id, context.prior_procedure_id}:
        return "experimental_prior_and_current_procedures_not_observed", ()
    event_phase_id = policy.get("event_phase_id")
    expected_policy = "full_procedure_source_bounds" if binding.get("is_omission", False) else "event_phase_source_bounds"
    if event_phase_id not in phase_ids or policy.get("candidate_interval_policy") != expected_policy:
        return "experimental_event_phase_policy_missing", ()
    for fact_id in policy.get("required_procedure_boundary_fact_ids", ()):
        fact = facts.get(fact_id)
        if fact is None or fact.state != "present" or not fact.supporting_frame_ids:
            return "experimental_procedure_boundary_fact_missing", (fact_id,)

    scope_facts = set(binding.get("context_fact_ids", ())) | set(binding.get("required_context_fact_ids", ()))
    # Anonymous body tracking ends at the observed exit. Requiring the person
    # to remain visible during the absence phase would contradict the rule.
    # Its exact required phases are checked by the track-specific gate below.
    scope_facts.difference_update(policy.get("track_fact_ids", {}).values())
    if policy.get("counting_paths_must_cover_procedure", False):
        scope_facts.update({"ALL_COUNTING_PATHS_VISIBLE", "WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE"})
    for fact_id in scope_facts:
        fact = facts.get(fact_id)
        if fact is None or fact.state != "present" or not declared <= set(fact.supporting_frame_ids):
            return "experimental_complete_context_not_supported", (fact_id,)
    for fact_id, state in binding["required_fact_states"].items():
        if state != "absent":
            continue
        if policy.get("absence_must_cover_procedure", False):
            absence_scope = declared
        else:
            selected_phases = policy.get("absence_phase_ids") or [phase["phase_id"] for phase in phases
                                                                    if fact_id in phase.get("fact_ids", ())]
            absence_scope = {key for phase_id in selected_phases for key in context.phase_frame_ids.get(phase_id, ())}
        fact = facts[fact_id]
        if not absence_scope or not absence_scope <= set(fact.supporting_frame_ids):
            return "experimental_absence_scope_incomplete", (fact_id,)

    for kind in policy.get("required_track_kinds", ()):
        track = context.tracks.get(kind)
        track_phases = policy.get("track_phase_ids", {}).get(kind, phase_ids)
        if not set(track_phases) <= set(phase_ids) or not track_phases:
            return "experimental_track_phase_policy_invalid", ()
        required_frames = {key for phase_id in track_phases for key in context.phase_frame_ids[phase_id]}
        if track is None or not required_frames <= set(track.frame_ids):
            return "experimental_track_continuity_missing", ()
        fact_id = policy.get("track_fact_ids", {}).get(kind)
        fact = facts.get(fact_id)
        if fact_id is None or fact is None or fact.state != "present" or not required_frames <= set(fact.supporting_frame_ids):
            return "experimental_track_fact_not_supported", (fact_id,) if fact_id else ()
    return None


def build_experimental_events(episodes: Sequence[ExperimentalEpisode | Mapping[str, Any]], *,
                              asset: VideoAsset, catalog: Any, frames: Mapping[str, FrameRef] | Sequence[FrameRef],
                              coverage: Any) -> ExperimentalResult:
    """Verify explicit experimental episodes without changing production state.

    The dedicated experimental loader must have verified the source archives.
    These inputs are proposals from local automated analysis. This function
    neither trains a model nor identifies a person nor emits a negative verdict.
    """
    if getattr(catalog, "is_experimental", False) is not True:
        raise ValueError("experimental catalog requires its dedicated explicit loader")
    vocabulary_version = getattr(catalog, "vocabulary_version", None)
    if (not vocabulary_version or not str(vocabulary_version).startswith("forte.experimental.")
            or getattr(catalog, "requires_temporal_wrapper", False) is not True):
        raise ValueError("experimental catalog and vocabulary must be separately versioned")
    frame_map = _source_frames(asset, frames)
    report = _coverage_dict(coverage)
    parsed = [ExperimentalEpisode.model_validate(value) for value in episodes]
    pending = []
    requests = []
    accepted = []
    seen = {}
    context_identities = {}
    for episode in parsed:
        key = (episode.rule_id, episode.context.context_id)
        identity = (episode.context.procedure_id, episode.context.prior_procedure_id,
                    tuple(sorted((kind, track.track_id) for kind, track in episode.context.tracks.items())))
        context_identities.setdefault(key, set()).add(identity)

    def defer(episode, reason, missing=()):
        pending.append(ExperimentalPending(episode.rule_id, episode.context.context_id,
                                           episode.candidate.candidate_id, reason, tuple(sorted(missing))))
        context = episode.context
        first = frame_map.get(context.full_context_start_frame_id)
        last = frame_map.get(context.full_context_end_frame_id)
        if first is not None and last is not None and first.pts_sec <= last.pts_sec:
            requests.append(RefinementRequest(first.pts_sec, last.pts_sec, reason,
                                              (episode.candidate.candidate_id,), tuple(sorted(missing))))

    technical_complete = (report.get("complete") is True and not report.get("gaps")
                          and not report.get("missing_window_indices")
                          and abs(float(report.get("duration_sec", -1)) - asset.duration_sec) <= 1e-6)
    for episode in sorted(parsed, key=lambda value: (value.candidate.start_sec, value.candidate.end_sec,
                                                    value.rule_id, value.context.context_id, value.candidate.candidate_id)):
        identity = (episode.rule_id, episode.context.context_id, episode.candidate.candidate_id)
        signature = json.dumps(episode.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        if identity in seen:
            if seen[identity] != signature:
                accepted[:] = [item for item in accepted if item[0] != identity]
                defer(episode, "experimental_conflicting_episode_identity")
            continue
        seen[identity] = signature
        if len(context_identities[(episode.rule_id, episode.context.context_id)]) > 1:
            defer(episode, "experimental_context_procedure_or_tracks_conflict")
            continue
        if episode.vocabulary_version != vocabulary_version:
            defer(episode, "experimental_episode_vocabulary_mismatch")
            continue
        if not technical_complete:
            defer(episode, "experimental_incomplete_analysis_coverage")
            continue
        rule = catalog.rules.get(episode.rule_id)
        binding = catalog.bindings.get(episode.rule_id)
        if (episode.rule_id not in getattr(catalog, "experimental_rule_ids", ())
                or rule is None or binding is None or not rule.enabled or not rule.applicability_verified
                or rule.verification_state not in {"source_verified", "verified"}
                or not rule.clause or not rule.source_sha256):
            defer(episode, "experimental_rule_source_or_applicability_unverified")
            continue
        if episode.candidate.event_type_id != binding["event_type_id"]:
            defer(episode, "experimental_event_type_binding_mismatch")
            continue
        facts = _facts(episode)
        if facts is None:
            defer(episode, "experimental_duplicate_fact_id")
            continue
        states = binding["required_fact_states"]
        missing = [key for key, state in states.items() if key not in facts or facts[key].state != state
                   or not facts[key].supporting_frame_ids]
        if missing:
            defer(episode, "experimental_required_facts_not_established", missing)
            continue
        error = _temporal_error(episode, binding, facts, frame_map, report)
        if error:
            defer(episode, *error)
            continue
        selected = [facts[key] for key in sorted(states)]
        candidate = episode.candidate
        if binding.get("is_omission", False):
            candidate = candidate.model_copy(update={
                "start_sec": frame_map[episode.context.full_context_start_frame_id].pts_sec,
                "end_sec": frame_map[episode.context.full_context_end_frame_id].pts_sec,
            })
        else:
            event_phase_id = binding["experimental_temporal"]["event_phase_id"]
            phase_frames = episode.context.phase_frame_ids[event_phase_id]
            candidate = candidate.model_copy(update={
                "start_sec": min(frame_map[key].pts_sec for key in phase_frames),
                "end_sec": max(frame_map[key].pts_sec for key in phase_frames),
                "supporting_frame_ids": sorted(set(candidate.supporting_frame_ids) | set(phase_frames)),
            })
        supported = _Supported(candidate, rule, selected, candidate.clip_id or episode.context.context_id,
                               bool(binding.get("is_omission", False)))
        key = (episode.rule_id, episode.context.context_id, episode.context.procedure_id,
               episode.context.prior_procedure_id,
               tuple(sorted((kind, track.track_id) for kind, track in episode.context.tracks.items())))
        accepted.append((identity, key, supported))

    groups = []
    for _, key, supported in accepted:
        found = next((group for group_key, group in groups
                      if group_key == key and _same_occurrence(group, supported, 0.5)), None)
        if found is None:
            groups.append((key, _Group([supported], supported.candidate.start_sec, supported.candidate.end_sec)))
        else:
            found.members.append(supported)
            found.common_start = max(found.common_start, supported.candidate.start_sec)
            found.common_end = min(found.common_end, supported.candidate.end_sec)
    events = []
    references = []
    for key, group in groups:
        event = _event(group, frame_map)
        if event is None:
            for episode in parsed:
                if episode.rule_id == key[0] and episode.context.context_id == key[1]:
                    defer(episode, "experimental_evidence_support_exceeds_frame_budget")
            continue
        rule = catalog.rules[event.rule_id]
        event.validate_context(asset, frame_map, rule)
        event = event.model_copy(update={"limitations": sorted(set(event.limitations) | {"experimental_rule_requires_validation"})})
        events.append(event)
        references.append(ExperimentalSourceReference(event.event_id, rule.rule_id, rule.source_url,
                                                      rule.clause, rule.source_sha256))
    events.sort(key=lambda value: (frame_map[value.anchor_frame_id].pts_sec, value.event_type_id, value.event_id))
    pending = sorted(set(pending), key=lambda value: (value.rule_id, value.context_id, value.candidate_id, value.reason))
    requests = sorted(set(requests), key=lambda value: (value.start_sec, value.end_sec, value.reason, value.candidate_ids))
    references.sort(key=lambda value: value.event_id)
    return ExperimentalResult(catalog.version, vocabulary_version, tuple(events), tuple(pending), tuple(requests),
                              tuple(references), len(parsed))


def main() -> int:
    """Verify a private local episode pack without inference or runtime changes.

    ``python -m forte.events.experimental --input pack.json --output result.json``
    accepts ``forte.experimental.temporal_input.v1`` with asset, frames, coverage
    and episodes. Source IDs must originate from the trusted local decoder.
    Only aggregate counts are printed; facts/tracks remain in the local files.
    """
    import argparse
    import os
    import tempfile
    from rules.experimental_catalog import load_experimental_catalog

    parser = argparse.ArgumentParser(description="Verify isolated experimental temporal episode proofs")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    temporary = None
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "forte.experimental.temporal_input.v1":
            raise ValueError("unsupported experimental input schema")
        asset = VideoAsset.model_validate(payload["asset"])
        frames = [FrameRef.model_validate(value) for value in payload["frames"]]
        result = build_experimental_events(payload["episodes"], asset=asset, frames=frames,
                                           coverage=payload["coverage"], catalog=load_experimental_catalog(args.catalog))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result.to_dict(), stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
        temporary = None
        print(json.dumps({"episodes_checked": result.episodes_checked, "verified_events": len(result.verified_events),
                          "pending": len(result.pending), "runtime_enabled": False}))
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"error": "experimental_input_or_catalog_invalid", "runtime_enabled": False}))
        return 2
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["ExperimentalTrack", "ExperimentalTemporalContext", "ExperimentalEpisode", "ExperimentalPending",
           "ExperimentalSourceReference", "ExperimentalResult", "validate_experimental_vocabulary",
           "build_experimental_events"]


if __name__ == "__main__":
    raise SystemExit(main())
