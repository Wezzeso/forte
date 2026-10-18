"""Optional executable temporal-head decisions; the active Qwen path is unchanged.

The backend must load real pinned weights and source-bound frozen features.
Machine calibration is explicitly provisional. This module contains no model
fallback, label lookup, invented threshold, media download or legal conclusion.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from forte.shared.schemas import ClipRequest
from forte.observability.model_metrics import CLIENT_SECONDS, COUNTS, number

from .head_contract import HeadIdentity, STATE_ORDER, TemporalHeadOutput, validate_head_binding
from .machine_contract import (
    MachineCalibrationManifest, MachineEvidenceVerifier, machine_candidate_readiness,
    machine_manifest_digest,
)
from .structured import DecisionConfig, validate_decisions

DECODER_RECIPE = {
    "schema_version": "forte.temporal-boundary-decoder.v1",
    "boundary_axes": ["start", "end"],
    "peaks": "earliest maximum of each contiguous above-threshold valid-frame run",
    "pairing": "strict chronological nonoverlapping start/end; same-frame pair allowed",
    "unmatched_or_overlapping_boundaries": "fail incomplete, never fabricate an endpoint",
    "timestamps": "original bound source frame IDs only",
    "scores": "independent raw logits or sigmoid scores; no softmax across boundary axes",
}
DECODER_SHA256 = hashlib.sha256(json.dumps(DECODER_RECIPE, sort_keys=True,
    separators=(",", ":")).encode()).hexdigest()

FACT_RUN_RECIPE = {
    "schema_version": "forte.physical-fact-run-proposals.v1",
    "inputs": "calibrated localized physical fact states and source supports only",
    "runs": "maximal contiguous native-frame run satisfying every required state and context",
    "boundaries": "first and last supported original frame; observed bracket, not exact action onset",
    "unknowns": "split runs and remain unresolved; no interpolation or window-label broadcast",
    "legal_effect": "candidate only; applicability/procedure/temporal/evidence guards remain mandatory",
}
FACT_RUN_DECODER_SHA256 = hashlib.sha256(json.dumps(FACT_RUN_RECIPE, sort_keys=True,
    separators=(",", ":")).encode()).hexdigest()


class PhysicalFactRunRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    rule_id: str = Field(min_length=1, max_length=256)
    event_type_id: str = Field(min_length=1, max_length=256)
    # This must include applicability/visibility/procedure facts from the
    # deterministic rule binding, not just a generic visual event label.
    required_fact_states: dict[str, str] = Field(min_length=1)


class PhysicalFactRunPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: str = "forte.physical-fact-run-policy.v1"
    taxonomy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rule_catalog_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    decoder_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    rules: tuple[PhysicalFactRunRule, ...]


def load_physical_fact_run_policy(*, identity: HeadIdentity,
        expected_catalog_sha256: str, catalog_path: Path | None = None) -> PhysicalFactRunPolicy:
    """Compile complete gates from the checksum-verified enabled A04 catalog.

    No rule text is emitted by the model. The ordinary A11 builder retains
    responsibility for legal applicability, temporal relationships and evidence.
    This helper intentionally rejects the experimental37-axis vocabulary.
    """
    from rules.catalog import CATALOG_PATH, load_catalog

    path = Path(catalog_path) if catalog_path is not None else CATALOG_PATH
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_catalog_sha256):
        raise TemporalProviderError("physical_fact_run_catalog_pin_mismatch")
    catalog = load_catalog(path, verify_archives=True)
    rules = []
    for record in catalog.enabled_rules():
        binding = catalog.bindings[record.rule_id]
        required = dict(binding["required_fact_states"])
        for fact_id, state in binding.get("temporal_fact_states", {}).items():
            if fact_id in required and required[fact_id] != state:
                raise TemporalProviderError("physical_fact_run_catalog_state_conflict")
            required[fact_id] = state
        for fact_id in binding.get("required_context_fact_ids", ()) or binding.get("context_fact_ids", ()):
            if fact_id in required and required[fact_id] != "present":
                raise TemporalProviderError("physical_fact_run_catalog_context_conflict")
            required[fact_id] = "present"
        if (not set(required) <= set(identity.fact_ids)
                or binding["event_type_id"] not in identity.event_type_ids):
            raise TemporalProviderError("physical_fact_run_catalog_axis_unsupported")
        rules.append(PhysicalFactRunRule(rule_id=record.rule_id,
            event_type_id=binding["event_type_id"], required_fact_states=required))
    return PhysicalFactRunPolicy(taxonomy_sha256=identity.taxonomy_sha256,
        rule_catalog_sha256=expected_catalog_sha256, decoder_sha256=FACT_RUN_DECODER_SHA256,
        rules=tuple(rules))


def physical_policy_observation_definitions(policy: PhysicalFactRunPolicy):
    """Return shared event gates and absence context from complete rule branches.

    Physical proposals retain the full branch conjunction. Generic event gates
    contain only requirements shared by every same-type branch, avoiding the
    impossible conjunction of mutually exclusive receipt/issue operations.
    Absence requires shared affirmative context plus each proposal's full rule.
    """
    events: dict[str, dict[str, str]] = {}
    absence: dict[str, set[str]] = {}
    for rule in policy.rules:
        required = dict(rule.required_fact_states)
        if rule.event_type_id not in events:
            events[rule.event_type_id] = required
        else:
            events[rule.event_type_id] = {key: state for key, state in events[rule.event_type_id].items()
                if required.get(key) == state}
        positive = {key for key, state in required.items() if state == "present"}
        for key, state in required.items():
            if state == "absent":
                absence[key] = positive if key not in absence else absence[key] & positive
    if any(not states for states in events.values()) or any(not contexts for contexts in absence.values()):
        raise TemporalProviderError("physical_policy_shared_context_missing")
    return ({key: {"required_fact_states": states} for key, states in sorted(events.items())},
            {key: tuple(sorted(contexts)) for key, contexts in sorted(absence.items())})


def propose_physical_fact_runs(
    output: TemporalHeadOutput, manifest: MachineCalibrationManifest,
    policy: PhysicalFactRunPolicy, *, eligible_states: Mapping[str, tuple[str, ...]],
    fact_map: Mapping[str, Mapping[str, Any]], support_candidates: Mapping[str, list[int]],
) -> dict[str, list[tuple[int, int, Mapping[str, str]]]]:
    """Generate provisional WHAT brackets without a trained event boundary head.

    Each source frame in a run independently witnesses every calibrated physical
    state, including required procedure/visibility facts. Video MIL scores and
    organizer anchors are never inputs. Source continuity is mandatory, and the
    candidate still needs the existing deterministic legal/evidence pipeline.
    """
    if (policy.schema_version != "forte.physical-fact-run-policy.v1"
            or policy.taxonomy_sha256 != output.identity.taxonomy_sha256
            or policy.decoder_sha256 != FACT_RUN_DECODER_SHA256):
        raise TemporalProviderError("physical_fact_run_policy_identity_mismatch")
    known_facts, known_events = set(output.identity.fact_ids), set(output.identity.event_type_ids)
    seen_rules = set()
    results: dict[str, list[tuple[int, int, Mapping[str, str]]]] = {}
    for rule in policy.rules:
        if (rule.rule_id in seen_rules or rule.event_type_id not in known_events
                or not set(rule.required_fact_states) <= known_facts
                or any(state not in {"present", "absent"} for state in rule.required_fact_states.values())):
            raise TemporalProviderError("physical_fact_run_rule_axis_invalid")
        seen_rules.add(rule.rule_id)
        if (not output.feature_coverage_complete or not output.native_source_continuity_verified
                or any(frame.source_pts is None for frame in output.frames)
                or any(first.next_source_pts != second.source_pts
                       for first, second in zip(output.frames, output.frames[1:]))):
            continue
        if any(state not in eligible_states[fact_id] or fact_map[fact_id]["state"] != state
               for fact_id, state in rule.required_fact_states.items()):
            continue
        support_sets = {fact_id: set(support_candidates[fact_id]) for fact_id in rule.required_fact_states}
        start = None
        for index in range(len(output.frames) + 1):
            valid = (index < len(output.frames) and output.frame_valid[index]
                and output.global_view_available[index]
                and all(index in supports for supports in support_sets.values()))
            if valid and start is None:
                start = index
            if not valid and start is not None:
                result = (start, index - 1, dict(rule.required_fact_states))
                existing = results.setdefault(rule.event_type_id, [])
                # Identical source brackets from parallel legal branches are a
                # single candidate, retaining the union of their physical facts.
                matched = next((item for item in existing if item[:2] == result[:2]), None)
                if matched is not None:
                    if any(key in matched[2] and matched[2][key] != state
                           for key, state in result[2].items()):
                        raise TemporalProviderError("physical_fact_run_fact_conflict")
                    matched[2].update(result[2])
                else:
                    existing.append(result)
                start = None
    return results


class TemporalProviderError(RuntimeError):
    code = "analysis_incomplete"

    def __init__(self, subcode: str):
        self.subcode = subcode
        super().__init__("optional_temporal_provider_incomplete")


@dataclass(frozen=True, repr=False)
class HeadBackendResult:
    output: TemporalHeadOutput = field(repr=False)
    metrics: Mapping[str, Any] = field(default_factory=dict, repr=False)


class TemporalHeadBackend(Protocol):
    identity: HeadIdentity

    async def check_ready(self, *, deadline: float) -> Any: ...

    async def predict(self, request: ClipRequest, *, deadline: float,
                      cancel_event: Any = None) -> HeadBackendResult: ...


@dataclass(frozen=True, repr=False)
class TemporalModelResponse:
    data: dict[str, Any] = field(repr=False)
    metrics: Mapping[str, Any] = field(repr=False)
    model: str
    finish_reason: str = "stop"


def _checkpoint(deadline: float, cancelled: Any) -> None:
    if cancelled is not None and cancelled.is_set():
        raise asyncio.CancelledError
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError("optional temporal head deadline exceeded")


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def _scores(values: tuple[float, ...], space: str) -> tuple[float, ...]:
    if space == "raw_logit":
        return values
    if space != "softmax_score":
        raise TemporalProviderError("state_score_space_invalid")
    peak = max(values)
    weights = tuple(math.exp(value - peak) for value in values)
    total = sum(weights)
    return tuple(value / total for value in weights)


def _support_score(value: float, space: str) -> float:
    if space == "raw_logit":
        return value
    if space == "sigmoid_score":
        return _sigmoid(value)
    raise TemporalProviderError("support_score_space_invalid")


def _representative(indices: list[int], score: Any, limit: int = 3) -> list[int]:
    """Bound references without inventing source positions or score cutoffs."""
    if not indices:
        return []
    if len(indices) <= limit:
        return sorted(indices)
    peak = max(indices, key=lambda index: (score(index), -index))
    selected = sorted({indices[0], peak, indices[-1]})
    return selected[:limit]


def _peaks(scores: list[float], valid: tuple[bool, ...], minimum: float) -> list[int]:
    result, run = [], []
    for index, value in enumerate(scores):
        if valid[index] and value >= minimum:
            run.append(index)
        elif run:
            result.append(max(run, key=lambda item: (scores[item], -item)))
            run = []
    if run:
        result.append(max(run, key=lambda item: (scores[item], -item)))
    return result


def decode_temporal_brackets(
    output: TemporalHeadOutput, manifest: MachineCalibrationManifest,
    event_types: tuple[str, ...],
) -> dict[str, list[tuple[int, int]]]:
    """Calibrated boundary peaks form actual source-frame brackets only."""
    if (not event_types or output.boundary_logits is None
            or output.trained_components.get("event_boundaries") is not True):
        return {}
    if manifest.decoder_sha256 != DECODER_SHA256:
        raise TemporalProviderError("temporal_decoder_identity_mismatch")
    results = {}
    for event_id in event_types:
        calibration = manifest.event_calibration[event_id]
        if calibration.score_space not in {"raw_logit", "sigmoid_score"}:
            raise TemporalProviderError("boundary_score_space_unsupported")
        axis = output.identity.event_type_ids.index(event_id)
        values = [[frame[axis][side] for frame in output.boundary_logits] for side in (0, 1)]
        if calibration.score_space == "sigmoid_score":
            values = [[_sigmoid(value) for value in side] for side in values]
        starts, ends = (_peaks(side, output.frame_valid, calibration.minimum_decoder_score) for side in values)
        # Close a point event after opening it at the same actual frame.
        actions = sorted([(index, 0) for index in starts] + [(index, 1) for index in ends])
        pending, pairs = None, []
        for index, kind in actions:
            if kind == 0:
                if pending is not None:
                    raise TemporalProviderError("overlapping_or_unpaired_temporal_starts")
                pending = index
            else:
                if pending is None:
                    raise TemporalProviderError("unpaired_temporal_end")
                pairs.append((pending, index))
                pending = None
        if pending is not None:
            raise TemporalProviderError("unpaired_temporal_start")
        results[event_id] = pairs
    return results


def map_temporal_output(
    output: TemporalHeadOutput, request: ClipRequest,
    manifest: MachineCalibrationManifest, *, verifier: MachineEvidenceVerifier,
    event_definitions: Mapping[str, Mapping[str, Any]],
    absence_context_fact_ids: Mapping[str, tuple[str, ...]],
    physical_fact_run_policy: PhysicalFactRunPolicy | None = None,
    trusted_scene_context: str | None = None,
) -> dict[str, Any]:
    """Map calibrated logits to existing observations, never directly to verdicts.

    Missing class/support/context eligibility emits unclear. Absence additionally
    requires complete feature coverage, every global view, and explicit calibrated
    present context facts supplied by the trusted observation policy. A negative
    MIL/video score does not label facts absent or suppress grounded candidates.
    """
    validate_head_binding(output, request, manifest.identity,
                          expected_decoder_sha256=manifest.decoder_sha256)
    return _map_source_bound_output(output, manifest, verifier=verifier,
        event_definitions=event_definitions, absence_context_fact_ids=absence_context_fact_ids,
        physical_fact_run_policy=physical_fact_run_policy, trusted_scene_context=trusted_scene_context)


def _map_source_bound_output(
    output: TemporalHeadOutput, manifest: MachineCalibrationManifest, *, verifier: MachineEvidenceVerifier,
    event_definitions: Mapping[str, Mapping[str, Any]],
    absence_context_fact_ids: Mapping[str, tuple[str, ...]],
    physical_fact_run_policy: PhysicalFactRunPolicy | None = None,
    trusted_scene_context: str | None = None,
) -> dict[str, Any]:
    """Internal mapping only after bounded-request or native-asset binding."""
    readiness = machine_candidate_readiness(manifest, output.identity, verifier=verifier)
    if trusted_scene_context not in {None, "bank_cash_workstation"}:
        raise TemporalProviderError("unsupported_trusted_scene_context")
    if not readiness.evidence_verified:
        raise TemporalProviderError("machine_calibration_not_ready")
    if output.proposals:
        raise TemporalProviderError("external_temporal_proposals_not_supported")
    known_facts = set(output.identity.fact_ids)
    if set(event_definitions) - set(output.identity.event_type_ids):
        raise TemporalProviderError("unsupported_event_vocabulary")
    for definition in event_definitions.values():
        required = definition.get("required_fact_states", {})
        if not required or not set(required).issubset(known_facts) or any(
            state not in {"present", "absent"} for state in required.values()
        ):
            raise TemporalProviderError("unsupported_event_fact_axis")
    for fact_id, context_ids in absence_context_fact_ids.items():
        if fact_id not in known_facts or not set(context_ids).issubset(known_facts):
            raise TemporalProviderError("unsupported_absence_context_axis")
    facts, support_candidates = [], {}
    fact_heads_trained = all(output.trained_components.get(component) is True for component in
        ("window_fact_states", "frame_fact_states", "frame_support"))
    for fact_axis, fact_id in enumerate(output.identity.fact_ids):
        logits = output.window_logits[fact_axis]
        winner = max(range(3), key=lambda axis: logits[axis])
        state = STATE_ORDER[winner]
        record = getattr(manifest.fact_calibration[fact_id], state, None) if state != "unclear" else None
        supports = []
        trusted_scope = fact_id != "BANK_CASH_WORKSTATION" or trusted_scene_context == "bank_cash_workstation"
        if fact_heads_trained and trusted_scope and state in readiness.eligible_fact_states[fact_id] and record is not None:
            threshold = record.threshold
            values = _scores(logits, threshold.score_space)
            alternatives = [value for axis, value in enumerate(values) if axis != winner]
            if values[winner] >= threshold.minimum_score and values[winner] - max(alternatives) >= threshold.minimum_margin:
                supports = []
                for index, valid in enumerate(output.frame_valid):
                    if not valid or not output.global_view_available[index]:
                        continue
                    frame_values = _scores(output.frame_logits[index][fact_axis], threshold.score_space)
                    frame_winner = max(range(3), key=lambda axis: frame_values[axis])
                    other = max(value for axis, value in enumerate(frame_values) if axis != winner)
                    if (frame_winner == winner and frame_values[winner] >= threshold.minimum_score
                            and frame_values[winner] - other >= threshold.minimum_margin
                            and _support_score(output.support_logits[index][fact_axis], threshold.support_score_space)
                            >= threshold.minimum_support_score):
                        supports.append(index)
        if not supports:
            state = "unclear"
        support_candidates[fact_id] = supports
        picked = _representative(supports, lambda index: output.support_logits[index][fact_axis])
        facts.append({"fact_id": fact_id, "state": state,
                      "supporting_frame_ids": [output.frames[index].frame_id for index in picked], "notes": None})
    fact_map = {fact["fact_id"]: fact for fact in facts}
    for fact in facts:
        if fact["state"] != "absent":
            continue
        prerequisites = absence_context_fact_ids.get(fact["fact_id"], ())
        if (not prerequisites or not output.feature_coverage_complete
                or not output.native_source_continuity_verified
                or any(frame.source_pts is None for frame in output.frames)
                or any(first.next_source_pts != second.source_pts
                       for first, second in zip(output.frames, output.frames[1:]))
                or not all(output.frame_valid) or not all(output.global_view_available)
                or len(support_candidates[fact["fact_id"]]) != len(output.frames)
                or any(fact_map[context]["state"] != "present"
                       or len(support_candidates[context]) != len(output.frames) for context in prerequisites)):
            fact.update(state="unclear", supporting_frame_ids=[])
    brackets = {event_id: [(start, end, event_definitions[event_id]["required_fact_states"])
        for start, end in pairs] for event_id, pairs in decode_temporal_brackets(
            output, manifest, tuple(event for event in readiness.candidate_event_types if event in event_definitions)).items()}
    if physical_fact_run_policy is not None:
        # An eligible trained typed decoder owns that axis. Adding a broad
        # physical run around its distinct brackets would invent a third event.
        boundary_owned = (set(readiness.candidate_event_types) if output.boundary_logits is not None
            and output.trained_components.get("event_boundaries") is True else set())
        fallback_policy = physical_fact_run_policy.model_copy(update={"rules": tuple(
            rule for rule in physical_fact_run_policy.rules if rule.event_type_id not in boundary_owned)})
        physical = propose_physical_fact_runs(output, manifest, fallback_policy,
            eligible_states=readiness.eligible_fact_states, fact_map=fact_map,
            support_candidates=support_candidates)
        for event_id, pairs in physical.items():
            if event_id not in event_definitions:
                raise TemporalProviderError("physical_fact_run_event_definition_missing")
            existing = brackets.setdefault(event_id, [])
            for start, end, required in pairs:
                if not set(event_definitions[event_id]["required_fact_states"]) <= set(required):
                    raise TemporalProviderError("physical_fact_run_missing_event_context")
                if any(required[key] != value for key, value in event_definitions[event_id]["required_fact_states"].items()):
                    raise TemporalProviderError("physical_fact_run_event_state_conflict")
                match = next((item for item in existing if item[:2] == (start, end)), None)
                if match is None:
                    existing.append((start, end, required))
                else:
                    if any(key in match[2] and match[2][key] != state for key, state in required.items()):
                        raise TemporalProviderError("physical_fact_run_fact_conflict")
                    match[2].update(required)
    candidates = []
    for event_id, pairs in brackets.items():
        if event_id not in event_definitions:
            continue
        for start, end, required in pairs:
            if any(fact_map[fact_id]["state"] != state for fact_id, state in required.items()):
                continue
            # Each required observation needs its own support in this occurrence.
            # A support for a different fact cannot stand in for a missing one.
            if any(not any(start <= index <= end for index in support_candidates[fact_id])
                   for fact_id in required):
                raise TemporalProviderError("localized_event_fact_support_outside_bracket")
            inside = sorted({index for fact_id in required for index in support_candidates[fact_id]
                             if start <= index <= end})
            if not inside:
                raise TemporalProviderError("localized_event_without_source_support")
            selected = _representative(inside, lambda index: max(
                output.support_logits[index][output.identity.fact_ids.index(fact_id)] for fact_id in required))
            # Retain at least one independently supported source frame per fact.
            selected = sorted(set(selected) | {
                max((index for index in support_candidates[fact_id] if start <= index <= end),
                    key=lambda index: (output.support_logits[index][output.identity.fact_ids.index(fact_id)], -index))
                for fact_id in required
            })
            for fact_id in required:
                extras = [output.frames[index].frame_id for index in selected if index in support_candidates[fact_id]]
                fact_map[fact_id]["supporting_frame_ids"] = list(dict.fromkeys(
                    fact_map[fact_id]["supporting_frame_ids"] + extras))
            candidates.append({
                "candidate_id": f"head-{len(candidates):03d}", "event_type_id": event_id,
                "start_frame_id": output.frames[start].frame_id, "end_frame_id": output.frames[end].frame_id,
                "supporting_fact_ids": [fact_id for fact_id in output.identity.fact_ids if fact_id in required],
                "supporting_frame_ids": [output.frames[index].frame_id for index in selected], "visibility_notes": [],
            })
    return {"facts": facts, "candidates": candidates}


class DiscriminativeDecisionProvider:
    """A07-compatible optional provider. No production factory selects it yet."""
    def __init__(self, backend: TemporalHeadBackend, manifest: MachineCalibrationManifest, *,
                 verifier: MachineEvidenceVerifier,
                 event_definitions: Mapping[str, Mapping[str, Any]],
                 absence_context_fact_ids: Mapping[str, tuple[str, ...]],
                 physical_fact_run_policy: PhysicalFactRunPolicy | None = None,
                 trusted_scene_context: str | None = None):
        self.backend = backend
        self.manifest = MachineCalibrationManifest.model_validate_json(manifest.model_dump_json())
        self._manifest_digest = machine_manifest_digest(self.manifest)
        self.verifier = verifier
        self.event_definitions = event_definitions
        self.absence_context_fact_ids = absence_context_fact_ids
        self.physical_fact_run_policy = physical_fact_run_policy
        self.trusted_scene_context = trusted_scene_context

    async def check_ready(self, *, deadline: float) -> dict[str, Any]:
        _checkpoint(deadline, None)
        if (self.backend.identity != self.manifest.identity
                or machine_manifest_digest(self.manifest) != self._manifest_digest):
            raise TemporalProviderError("checkpoint_calibration_identity_mismatch")
        gate = machine_candidate_readiness(self.manifest, self.backend.identity, verifier=self.verifier)
        if not gate.evidence_verified:
            raise TemporalProviderError("machine_calibration_not_ready")
        await self.backend.check_ready(deadline=deadline)
        return {"ready": True, "provisional": True, "human_fact_event_gold": False,
                "independent_fact_event_accuracy_measured": False,
                "calibrated_physical_fact_states_available": gate.can_emit_provisional_facts,
                "head_sha256": self.backend.identity.head_sha256}

    async def infer(self, clip: ClipRequest, *, prompt: str, response_schema: dict[str, Any],
                    deadline: float, cancel_event: Any = None) -> TemporalModelResponse:
        started = time.monotonic()
        _checkpoint(deadline, cancel_event)
        await self.check_ready(deadline=deadline)
        async with asyncio.timeout_at(deadline):
            result = await self.backend.predict(clip, deadline=deadline, cancel_event=cancel_event)
        _checkpoint(deadline, cancel_event)
        data = map_temporal_output(result.output, clip, self.manifest, verifier=self.verifier,
            event_definitions=self.event_definitions, absence_context_fact_ids=self.absence_context_fact_ids,
            physical_fact_run_policy=self.physical_fact_run_policy, trusted_scene_context=self.trusted_scene_context)
        # Real head outputs still cross the existing full fact/reference contract.
        validate_decisions(data, clip, reviewed_event_ids=tuple(self.event_definitions),
                           config=DecisionConfig(concise_observations=True))
        metrics = {key: cleaned for key in CLIENT_SECONDS + COUNTS
                   if (cleaned := number(result.metrics.get(key))) is not None}
        metrics.update({"elapsed_sec": time.monotonic() - started,
                   "input_frames": len(clip.frame_refs), "generated_tokens": 0,
                   "serving_attestation": {"source": "provisional_machine_temporal_head",
                       "model_revision": self.backend.identity.head_sha256,
                       "processor_revision": self.backend.identity.encoder_revision,
                       "provisional": True, "human_fact_event_gold": False,
                       "independent_fact_event_accuracy_measured": False}})
        return TemporalModelResponse(data, metrics, "forte-temporal-head")

    async def aclose(self) -> None:
        close = getattr(self.backend, "aclose", None)
        if close is not None:
            await close()
