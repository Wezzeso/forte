"""Local visual analysis, deterministic grounding and original-source evidence.

The provider is injectable for synthetic integration tests. Production always
uses the private model service; there is no filename/label or mock fallback.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from forte.config import WorkerSettings
from forte.decision import DecisionConfig, DecisionEngine
from forte.events import EventBuildConfig, RefinementRequest, UnresolvedInterval, analyze_rule_relevance, build_events
from forte.evidence import assemble_result
from forte.model import LocalModelConfig, LocalModelProvider
from forte.model.manifest import ManifestError, load_manifest
from forte.observability import StageTimer, aggregate_inference_metrics
from forte.reproducibility import build_reproducibility_bundle
from forte.shared.schemas import CONTRACT_VERSION
from forte.video.decode import decode_frames
from forte.video.probe import probe_video
from forte.video.sampling import (
    ProcessorBudget, context_window, coverage_report, coverage_windows, make_clip_requests,
    record_inspection, require_complete_coverage, sampling_timestamps,
)
from rules.catalog import load_catalog
from serving.scheduler import FairScheduler, ScheduledProvider

StageCallback = Callable[[str], Awaitable[None]]
SCREENING_FACT_IDS = (
    "CASH_VISIBLE", "CASH_HANDLER_LEAVES_WORKSTATION",
    "CASH_PLACED_IN_CLOTHING_POCKET", "CASH_HANDOVER_VISIBLE",
)


class AnalysisIncomplete(RuntimeError):
    """A safe static diagnostic; private model observations never enter logs."""

    def __init__(self, code: str = "analysis_incomplete") -> None:
        self.code = code
        super().__init__(code)


def atomic_private_json(path: Path, payload: Any) -> None:
    """Write local metadata atomically; callers keep it outside public assets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


@dataclass(frozen=True)
class PipelineResult:
    json_bytes: bytes
    timings: dict[str, Any]
    metadata_path: Path


def _contains(outer: Any, inner: Any, tolerance: float = 1e-6) -> bool:
    return (outer.start_sec <= inner.start_sec + tolerance
            and outer.end_sec >= inner.end_sec - tolerance)


def _candidate_distance(first: Any, second: Any) -> float:
    return max(0.0, first.start_sec - second.end_sec, second.start_sec - first.end_sec)


def _matching_candidate(source: Any, target: Any, spacing: float) -> bool:
    return (source.event_type_id == target.event_type_id
            and _candidate_distance(source, target) <= spacing + 1e-6)


def _explicit_type_refutation(source: Any, event_type_id: str, refined: Any, catalog: Any,
                              *, source_candidate: Any = None) -> bool:
    """An empty candidate list is not a refutation of a prior observation.

    Require a clear pass containing the entire source procedure, reinspecting
    its original positive support, and supported opposite states excluding every
    enabled binding for this exact event type. Wider temporal bounds alone do
    not suffice when capped resampling drops the original action frames.
    """
    if not _contains(refined, source) or any(
        candidate.event_type_id == event_type_id for candidate in refined.candidates
    ):
        return False
    candidates = ((source_candidate,) if source_candidate is not None else tuple(
        candidate for candidate in source.candidates if candidate.event_type_id == event_type_id))
    records = getattr(catalog, "rules", {})
    relevant_bindings = [binding for rule_id, binding in getattr(catalog, "bindings", {}).items()
                         if rule_id in records and records[rule_id].enabled
                         and binding.get("event_type_id") == event_type_id]
    action_ids = {key for binding in relevant_bindings for key in binding.get("violation_fact_ids", ())}
    if not action_ids:
        # Omission branches and generic catalogues have no positive action ID.
        # Retain the original positive procedure evidence instead.
        action_ids = {key for binding in relevant_bindings
                      for key, state in binding.get("required_fact_states", {}).items() if state == "present"}
    prior_support = {key for candidate in candidates for key in candidate.supporting_frame_ids}
    prior_facts = list(getattr(source, "facts", ()))
    prior_facts.extend(fact for candidate in candidates for fact in candidate.supporting_facts)
    prior_support.update(key for fact in prior_facts if fact.fact_id in action_ids and fact.state == "present"
                         for key in fact.supporting_frame_ids)
    reinspected = {frame.frame_id for frame in refined.request.frame_refs}
    if not prior_support or not prior_support <= reinspected:
        return False
    return analyze_rule_relevance(refined, catalog, event_type_id=event_type_id).all_branches_refuted


def _explicit_refutation(source: Any, candidate: Any, refined: Any, catalog: Any) -> bool:
    return _explicit_type_refutation(source, candidate.event_type_id, refined, catalog,
                                     source_candidate=candidate)


def _missing_candidate_types(batch: Any, catalog: Any) -> dict[str, tuple[str, ...]]:
    """Find internally supported rule signals missing an event proposal.

    Inspect original model batches only. Facts retained solely for coverage are
    derived views and must not manufacture this inconsistency after a proposal
    has been explicitly resolved by later context.
    """
    facts = {fact.fact_id: fact for fact in getattr(batch, "facts", ())}
    proposed_types = {candidate.event_type_id for candidate in batch.candidates}
    records = getattr(catalog, "rules", {})
    missing: dict[str, tuple[str, ...]] = {}
    for rule_id, binding in getattr(catalog, "bindings", {}).items():
        event_type_id = binding.get("event_type_id")
        states = binding.get("required_fact_states", {})
        if (rule_id not in records or not records[rule_id].enabled or not states
                or not event_type_id or event_type_id in proposed_types):
            continue
        if all(fact_id in facts and facts[fact_id].state == state
               and facts[fact_id].supporting_frame_ids for fact_id, state in states.items()):
            missing[event_type_id] = tuple(sorted(states))
    return missing


def _reconcile_batches(coarse: list[Any], rounds: list[list[Any]], catalog: Any):
    """Keep confirmed refinements and explicitly account for every proposal.

    One refined candidate may confirm overlap duplicates in different coarse
    clips. It cannot silently resolve two separate candidates in the same clip.
    """
    sources = [(batch, 0) for batch in coarse]
    sources.extend((batch, generation) for generation, batches in enumerate(rounds, 1) for batch in batches)
    final = []
    unresolved = []
    requests = []
    used: set[tuple[str, str, str]] = set()
    resolved_for_clip: dict[str, list[Any]] = {}
    for batch, generation in sources:
        later = [(other, other_generation) for other, other_generation in sources if other_generation > generation]
        remaining = []
        for candidate in sorted(batch.candidates, key=lambda value: (value.start_sec, value.end_sec,
                                                                     value.event_type_id, value.candidate_id)):
            times = sorted(frame.pts_sec for frame in batch.request.frame_refs)
            spacing = min(0.75, max((b - a for a, b in zip(times, times[1:])), default=0.1))
            matches = [(other, proposed) for other, _ in later for proposed in other.candidates
                       if _matching_candidate(candidate, proposed, spacing)
                       and (batch.request.clip_id, other.request.clip_id, proposed.candidate_id) not in used
                       and not any(_matching_candidate(previous, proposed, spacing)
                                   for previous in resolved_for_clip.get(batch.request.clip_id, ()))]
            if matches:
                other, proposed = min(matches, key=lambda pair: (
                    _candidate_distance(candidate, pair[1]),
                    abs(candidate.start_sec - pair[1].start_sec), pair[0].request.clip_id,
                    pair[1].candidate_id,
                ))
                used.add((batch.request.clip_id, other.request.clip_id, proposed.candidate_id))
                resolved_for_clip.setdefault(batch.request.clip_id, []).append(proposed)
                continue
            if any(_explicit_refutation(batch, candidate, other, catalog) for other, _ in later):
                continue
            followed_up = any(_contains(other, candidate) for other, _ in later)
            if generation == 0 or followed_up:
                unresolved.append(UnresolvedInterval(candidate.start_sec, candidate.end_sec,
                    "unresolved_coarse_candidate" if generation == 0 else "unresolved_refined_candidate",
                    candidate.candidate_id, candidate.event_type_id))
                requests.append(RefinementRequest(batch.start_sec, batch.end_sec,
                    "candidate_requires_explicit_confirmation_or_refutation", (candidate.candidate_id,)))
            else:
                remaining.append(candidate)
        for event_type_id, required_fact_ids in sorted(_missing_candidate_types(batch, catalog).items()):
            confirmed = any(
                proposed.event_type_id == event_type_id and _contains(batch, proposed)
                for other, _ in later for proposed in other.candidates
            )
            refuted = any(_explicit_type_refutation(batch, event_type_id, other, catalog)
                          for other, _ in later)
            if not confirmed and not refuted:
                unresolved.append(UnresolvedInterval(batch.start_sec, batch.end_sec,
                    "rule_facts_without_candidate", event_type_id=event_type_id))
                requests.append(RefinementRequest(batch.start_sec, batch.end_sec,
                    "rule_facts_without_candidate", required_fact_ids=required_fact_ids))
        # Only a later clear pass covering the full procedure supersedes an
        # unclear batch. Narrow reinspection cannot erase its visibility limit.
        superseded = batch.has_uncertainty and any(
            not analyze_rule_relevance(other, catalog).requires_refinement and _contains(other, batch)
            for other, _ in later
        )
        if not superseded:
            final.append(replace(batch, candidates=tuple(remaining)))
    return final, tuple(unresolved), tuple(requests)


def _with_pending(built: Any, unresolved: tuple, requests: tuple):
    if not unresolved:
        return built
    limitations = {item.reason for item in unresolved}
    events = tuple(event.model_copy(update={"limitations": sorted(set(event.limitations) | limitations)})
                   for event in built.verified_events)
    return replace(built, verified_events=events,
                   unresolved_intervals=tuple(built.unresolved_intervals) + unresolved,
                   refinement_requests=tuple(built.refinement_requests) + requests)


def _screen_pending(screens: list[Any], full_batches: list[Any], catalog: Any,
                    full_fact_ids: set[str], config: EventBuildConfig):
    """Account for every screen branch without inventing an event timestamp.

    Only complete full-fact passes may resolve a screen. An empty candidate list
    does not do so: a negative needs a supported scoped opposite for each branch.
    """
    unresolved = []
    requests = []
    for screen in screens:
        relevance = analyze_rule_relevance(screen, catalog, config=config)
        for rule_id in relevance.undetermined_rule_ids:
            binding = catalog.bindings[rule_id]
            event_type_id = binding["event_type_id"]
            states = binding["required_fact_states"]
            resolved = False
            for full in full_batches:
                requested = {fact.fact_id for fact in full.request.fact_definitions}
                if not full_fact_ids <= requested or not _contains(full, screen):
                    continue
                refutation = analyze_rule_relevance(full, catalog, config=config)
                same_type = [candidate for candidate in full.candidates if candidate.event_type_id == event_type_id]
                if rule_id in refutation.refuted_rule_ids and not same_type:
                    resolved = True
                    break
                facts = {fact.fact_id: fact for fact in full.facts}
                supported = all(key in facts and facts[key].state == state and facts[key].supporting_frame_ids
                                for key, state in states.items())
                if supported and any(_contains(screen, candidate) for candidate in same_type):
                    resolved = True
                    break
            if not resolved:
                unresolved.append(UnresolvedInterval(screen.start_sec, screen.end_sec,
                    "unresolved_screen_branch", event_type_id=event_type_id))
                requests.append(RefinementRequest(screen.start_sec, screen.end_sec,
                    "unresolved_screen_branch", required_fact_ids=tuple(sorted(states))))
    return tuple(unresolved), tuple(requests)


def _plan_requested_context(requests: Any, sources: list[Any], duration: float,
                            seen: set[tuple[float, float]]) -> list[tuple[float, float]]:
    result: dict[tuple[float, float], None] = {}
    for request in requests:
        if request.candidate_ids:
            parents = [batch for batch in sources if any(
                candidate.candidate_id in request.candidate_ids
                and candidate.start_sec <= request.end_sec and candidate.end_sec >= request.start_sec
                for candidate in batch.candidates)]
        else:
            parents = [batch for batch in sources if abs(batch.start_sec - request.start_sec) < 1e-6
                       and abs(batch.end_sec - request.end_sec) < 1e-6]
        start = max(0.0, min([request.start_sec] + [batch.start_sec for batch in parents]) - 4.0)
        end = min(duration, max([request.end_sec] + [batch.end_sec for batch in parents]) + 4.0)
        if end > start and (start, end) not in seen:
            result[(start, end)] = None
    return list(result)


def _audit_provenance(settings: WorkerSettings) -> dict[str, Any]:
    result = {
        "execution_location": "team_deployed_service", "external_ai_used": False,
        "model_id": settings.model_name, "model_revision": None,
        "processor_version": None, "processor_revision": None,
        "revision_source": None, "actual_snapshot_verified": None,
    }
    if getattr(settings, "model_candidate_launch_config_path", None) is not None:
        from serving.qwen38_identity import load_expected_runtime_identity
        identity = load_expected_runtime_identity(settings.model_candidate_launch_config_path)
        result.update(model_revision=identity["model_revision"],
                      processor_revision=identity["processor_revision"],
                      backend=identity["backend"],
                      adapter_sha256=identity["adapter_sha256"],
                      candidate_runtime_identity=identity,
                      revision_source="configured_candidate_manifest")
        return result
    try:
        manifest = load_manifest(settings.model_manifest_path)
    except (ManifestError, AttributeError):
        return result
    if manifest.get("served_model_name") == settings.model_name:
        result.update(model_revision=manifest["revision"], processor_revision=manifest["processor_revision"],
                      revision_source="configured_manifest")
    return result


def _provider_configuration(settings: WorkerSettings) -> LocalModelConfig:
    values: dict[str, Any] = {
        "media_root": settings.work_dir, "base_url": settings.model_base_url,
        "model": settings.model_name, "api_key": settings.model_api_key,
        "max_frames": settings.max_frames_per_window, "max_tokens": 4096,
    }
    if getattr(settings, "model_candidate_launch_config_path", None) is not None:
        from serving.qwen38_identity import load_expected_runtime_identity, load_json
        path = settings.model_candidate_launch_config_path
        identity = load_expected_runtime_identity(path)
        config = load_json(path)
        if (config["served_model_name"] != settings.model_name
                or settings.max_frames_per_window > config["limits"]["max_images"]):
            raise AnalysisIncomplete("candidate_worker_configuration_mismatch")
        values.update(expected_revision=identity["model_revision"],
                      expected_runtime_identity=tuple(sorted(identity.items())),
                      seed=config["generation"]["seed"],
                      max_tokens=config["generation"]["max_new_tokens"],
                      request_timeout_sec=config["limits"]["request_timeout_seconds"],
                      chat_template_kwargs=(("enable_thinking", False), ("preserve_thinking", False)))
    return LocalModelConfig(**values)


def _private_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(vars(value))


class AnalysisPipeline:
    def __init__(self, settings: WorkerSettings, *, provider: Any = None, catalog: Any = None):
        self.settings = settings
        self.catalog = catalog or load_catalog(settings.rules_path)
        self.rules = self.catalog.enabled_rules()
        if not self.rules:
            raise AnalysisIncomplete("no_enabled_verified_rules")
        self.facts = self.catalog.fact_definitions(trusted_scene_context=settings.trusted_scene_context)
        event_ids = sorted({self.catalog.bindings[rule.rule_id]["event_type_id"] for rule in self.rules})
        self.event_definitions = {key: self.catalog.event_types[key] for key in event_ids}
        raw_provider = provider or LocalModelProvider(_provider_configuration(settings))
        self.provider = ScheduledProvider(raw_provider, FairScheduler(
            concurrency=settings.max_concurrent_clips,
            max_inflight_per_job=settings.max_inflight_per_job,
        ))
        decision_config = DecisionConfig(compact_frame_ids=settings.compact_frame_ids,
                                         concise_observations=settings.concise_observations)
        self.engine = DecisionEngine(self.provider, reviewed_event_ids=event_ids,
                                     event_definitions=self.event_definitions,
                                     config=decision_config)
        self.screening_engine = DecisionEngine(self.provider, reviewed_event_ids=(),
                                                event_definitions={}, config=decision_config)
        definitions = {fact.fact_id: fact for fact in self.facts}
        if settings.staged_screening and not set(SCREENING_FACT_IDS) <= definitions.keys():
            raise AnalysisIncomplete("screening_fact_catalog_incomplete")
        self.screening_facts = [definitions[key] for key in SCREENING_FACT_IDS if key in definitions]
        self.event_config = EventBuildConfig(event_types=self.event_definitions)
        self.audit_provenance = _audit_provenance(settings)
        self.reproducibility_bundle = build_reproducibility_bundle(
            settings, self.catalog, decision_config, getattr(raw_provider, "config", None),
        )

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if close is not None:
            await close()

    @staticmethod
    def _checkpoint(deadline: float, cancelled: threading.Event) -> None:
        if cancelled.is_set():
            raise AnalysisIncomplete("analysis_cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError("analysis_deadline_exceeded")

    async def _infer(self, clip: Any, deadline: float, cancelled: threading.Event) -> Any:
        self._checkpoint(deadline, cancelled)
        return await self.engine.analyze(clip, deadline=deadline, cancel_event=cancelled)

    async def _infer_screen(self, clip: Any, deadline: float, cancelled: threading.Event) -> Any:
        self._checkpoint(deadline, cancelled)
        return await self.screening_engine.analyze(clip, deadline=deadline, cancel_event=cancelled)

    async def analyze(self, job: Any, *, stage: StageCallback, deadline: float,
                      cancel_event: threading.Event) -> PipelineResult:
        timer = StageTimer()
        job_root = self.settings.work_dir / job.job_id
        attempt_root = job_root / f"attempt-{job.attempt}"
        frame_dir = attempt_root / "frames"
        attempt_root.mkdir(parents=True, exist_ok=True)
        self._checkpoint(deadline, cancel_event)
        await stage("decoding")
        with timer.stage("probe_decode"):
            asset = await asyncio.to_thread(probe_video, Path(job.owned_path), job_id=job.job_id,
                                           deadline_monotonic=deadline, cancel=cancel_event)
            if asset.content_hash != job.content_hash:
                raise AnalysisIncomplete("accepted_source_hash_changed")
            windows = coverage_windows(
                asset.duration_sec, window_sec=self.settings.window_seconds,
                stride_sec=self.settings.stride_seconds, sample_fps=self.settings.sample_fps,
                max_frames_per_window=self.settings.max_frames_per_window,
                max_windows=512, max_total_frames=8192,
            )
            frames = await asyncio.to_thread(
                decode_frames, asset, frame_dir, timestamps_sec=sampling_timestamps(windows),
                deadline_monotonic=deadline, cancel=cancel_event,
            )
        with timer.stage("frame_preparation"):
            clips = make_clip_requests(
                job.job_id, frames, windows, fact_definitions=self.facts,
                rule_definitions=self.rules,
                analysis_version=CONTRACT_VERSION,
                budget=ProcessorBudget(max_frames=self.settings.max_frames_per_window),
            )
        metadata_path = attempt_root / "analysis.private.json"
        staged = getattr(self.settings, "staged_screening", False)
        full_clips = clips
        if staged:
            clips = [clip.model_copy(update={
                "clip_id": clip.clip_id + "-screen", "fact_definitions": self.screening_facts,
                "rule_definitions": [],
            }) for clip in full_clips]
        batches: list[Any] = [None] * len(clips)
        screen_batches: list[Any] = []
        primary_batches: list[Any] = []
        primary_required: list[int] = []
        rounds: list[list[Any]] = []
        refined_batches: list[Any] = []
        frame_map = {frame.frame_id: frame for frame in frames}

        def all_model_batches() -> list[Any]:
            seen: set[int] = set()
            result = []
            for batch in screen_batches + primary_batches + batches + refined_batches:
                if batch is not None and id(batch) not in seen:
                    seen.add(id(batch))
                    result.append(batch)
            return result

        def inference_snapshot() -> dict[str, Any]:
            result = timer.snapshot()
            result["model_requests"] = aggregate_inference_metrics([
                getattr(batch, "metrics", {}) for batch in all_model_batches()
            ])
            return result

        async def write_audit(phase: str, built: Any = None, *, error_code: str | None = None,
                              evidence: Any = None) -> None:
            completed = [batch for batch in batches if batch is not None]
            if staged:
                sources = [(batch, 0, "screen") for batch in (screen_batches or completed)]
                sources.extend((batch, 1, "full_primary") for batch in primary_batches)
                sources.extend((batch, generation, "refinement")
                               for generation, group in enumerate(rounds, 2) for batch in group)
            else:
                sources = [(batch, 0, "coarse") for batch in completed]
                sources.extend((batch, generation, "refinement")
                               for generation, group in enumerate(rounds, 1) for batch in group)
            inspected = [record_inspection(window, clip)
                         for window, clip, batch in zip(windows, clips, batches, strict=True) if batch is not None]
            current_coverage = coverage_report(asset.duration_sec, windows, inspected)
            stored_provenance = getattr(self, "audit_provenance", None)
            provenance = dict(stored_provenance if stored_provenance is not None else _audit_provenance(self.settings))
            provenance["local_run_id"] = f"{job.job_id}:{job.attempt}"
            provenance["reproducibility_bundle"] = getattr(self, "reproducibility_bundle", None)
            for batch, _, _ in sources:
                attestation = getattr(batch, "metrics", {}).get("serving_attestation")
                if isinstance(attestation, dict) and attestation.get("model_files_verified") is True:
                    provenance["serving_attestation"] = {key: attestation[key] for key in (
                        "ready", "model_files_verified", "model_revision", "processor_revision",
                        "model_manifest_sha256", "vllm_version",
                        "model_id", "backend", "tokenizer_sha256", "chat_template_sha256",
                        "adapter_sha256", "adapter_config_sha256", "adapter_manifest_sha256",
                        "training_run_sha256", "quantization_sha256", "generation_sha256",
                        "image_sha256", "source_sha256",
                    ) if key in attestation}
                    provenance.update(actual_snapshot_verified=True, revision_source="verified_serving_attestation",
                                      model_revision=attestation.get("model_revision"),
                                      processor_revision=attestation.get("processor_revision"))
            events = list(getattr(built, "verified_events", ()))
            window_decisions = [{
                "clip_id": batch.request.clip_id, "start_sec": batch.start_sec, "end_sec": batch.end_sec,
                "frame_ids": [frame.frame_id for frame in batch.request.frame_refs],
                "requested_fact_ids": [fact.fact_id for fact in batch.request.fact_definitions],
                "requested_rule_ids": [rule.rule_id for rule in batch.request.rule_definitions],
                "phase": batch_phase,
                "prompt_version": batch.prompt_version, "config_version": batch.config_version,
                "prompt_hash": batch.prompt_hash,
                "facts": [_private_dump(fact) for fact in getattr(batch, "facts", ())],
                "candidate_events": [_private_dump(candidate) for candidate in batch.candidates],
                "generation": generation, "complete": getattr(batch, "complete", True),
                "disposition": getattr(batch, "disposition", "unclear" if batch.has_uncertainty else "complete"),
            } for batch, generation, batch_phase in sources]
            payload = {
                "schema_version": "forte.local-proposal.v1", "phase": phase, "error_code": error_code,
                "job_id": job.job_id, "attempt": job.attempt,
                "analysis_version": CONTRACT_VERSION, "catalog_version": self.catalog.version,
                "asset": asset.model_dump(mode="json"),
                "frames": [frame.model_dump(mode="json") for frame in sorted(
                    frame_map.values(), key=lambda value: (value.pts_sec, value.frame_id))],
                "decisions": [fact for window in window_decisions for fact in window["facts"]],
                "candidate_events": [candidate for window in window_decisions for candidate in window["candidate_events"]],
                "verified_events": [_private_dump(event) for event in events],
                "events": [_private_dump(event) for event in events],
                "window_decisions": window_decisions, "provenance": provenance,
                "model_name": self.settings.model_name, "coverage": current_coverage.to_dict(),
                "publication_ready": bool(phase in {"ready", "done"} and built is not None and built.can_publish),
                "limitations": list(getattr(built, "limitations", ())),
                "unresolved_intervals": [asdict(value) for value in getattr(built, "unresolved_intervals", ())],
                "refinement_plan": [asdict(value) for value in getattr(built, "refinement_requests", ())],
                "coarse_requests": len(completed), "refinement_requests": len(refined_batches),
                "staged_screening": staged, "screen_requests": len(screen_batches),
                "primary_required": len(primary_required), "primary_completed": len(primary_batches),
                "pipeline_profile": {
                    "compact_frame_ids": self.settings.compact_frame_ids,
                    "concise_observations": self.settings.concise_observations,
                    "staged_screening": staged,
                    "window_seconds": self.settings.window_seconds,
                    "stride_seconds": self.settings.stride_seconds,
                    "sample_fps": self.settings.sample_fps,
                    "max_frames_per_window": self.settings.max_frames_per_window,
                    "max_refinement_passes": self.settings.max_refinement_passes,
                    "max_concurrent_clips": self.settings.max_concurrent_clips,
                    "max_inflight_per_job": self.settings.max_inflight_per_job,
                },
                "timings": inference_snapshot(), "evidence": asdict(evidence) if evidence is not None else None,
            }
            await asyncio.to_thread(atomic_private_json, metadata_path, payload)

        await stage("screening")
        with timer.stage("coarse_model"):
            # A bounded pool avoids allocating one task per long recording window.
            cursor = iter(enumerate(clips))

            async def consume() -> None:
                for index, clip in cursor:
                    batches[index] = await (self._infer_screen(clip, deadline, cancel_event) if staged
                                            else self._infer(clip, deadline, cancel_event))

            tasks = [asyncio.create_task(consume()) for _ in range(min(len(clips), self.settings.max_concurrent_clips))]
            try:
                await asyncio.gather(*tasks)
            except BaseException as error:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await write_audit("failed", error_code="coarse_model_incomplete")
                try:
                    error.forte_timings = inference_snapshot()
                except (AttributeError, TypeError):
                    pass
                raise
        inspections = [record_inspection(window, clip) for window, clip in zip(windows, clips, strict=True)]
        coverage = coverage_report(asset.duration_sec, windows, inspections)
        require_complete_coverage(coverage)

        if staged:
            screen_batches.extend(batches)
            await write_audit("screen")
            primary_required.extend(index for index, batch in enumerate(screen_batches)
                                    if not analyze_rule_relevance(batch, self.catalog,
                                                                  config=self.event_config).all_branches_refuted)
            # Full-fact resolution is primary analysis, not one of the optional
            # bounded verification passes. Reuse the exact original source refs.
            if primary_required:
                with timer.stage("coarse_model"):
                    primary_cursor = iter(primary_required)

                    async def consume_primary() -> None:
                        for index in primary_cursor:
                            batch = await self._infer(full_clips[index], deadline, cancel_event)
                            primary_batches.append(batch)
                            batches[index] = batch

                    tasks = [asyncio.create_task(consume_primary()) for _ in range(
                        min(len(primary_required), self.settings.max_concurrent_clips))]
                    try:
                        await asyncio.gather(*tasks)
                    except BaseException as error:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        await write_audit("failed", error_code="primary_model_incomplete")
                        try:
                            error.forte_timings = inference_snapshot()
                        except (AttributeError, TypeError):
                            pass
                        raise

        # Ground coarse observations before planning context. Clear candidates
        # receive denser reinspection; omissions and unresolved procedures retain
        # the complete parent window and acquire additional temporal context.
        with timer.stage("rules"):
            provisional = build_events(batches, self.catalog, frame_map, coverage, config=self.event_config)
            if staged:
                screen_pending, screen_requests = _screen_pending(screen_batches, primary_batches,
                    self.catalog, {fact.fact_id for fact in self.facts}, self.event_config)
                provisional = _with_pending(provisional, screen_pending, screen_requests)
        await write_audit("full_primary" if staged else "coarse", provisional)
        specs: dict[tuple[float, float], None] = {}
        requested = tuple(provisional.refinement_requests) + tuple(
            RefinementRequest(batch.start_sec, batch.end_sec, "rule_facts_without_candidate",
                              required_fact_ids=required_fact_ids)
            for batch in batches for required_fact_ids in _missing_candidate_types(batch, self.catalog).values()
        )
        for batch in batches:
            for candidate in batch.candidates:
                already_requested = any(candidate.candidate_id in request.candidate_ids
                                        and candidate.start_sec <= request.end_sec
                                        and candidate.end_sec >= request.start_sec for request in requested)
                if already_requested:
                    continue
                omission = any(binding.get("event_type_id") == candidate.event_type_id
                               and binding.get("is_omission", False)
                               for binding in getattr(self.catalog, "bindings", {}).values())
                if omission:
                    start, end = max(0.0, batch.start_sec - 4.0), min(asset.duration_sec, batch.end_sec + 4.0)
                else:
                    start = max(0.0, candidate.start_sec - 1.0)
                    end = min(asset.duration_sec, candidate.end_sec + 1.0)
                if end > start:
                    specs[(start, end)] = None
            if analyze_rule_relevance(batch, self.catalog, config=self.event_config).requires_refinement:
                specs[(max(0.0, batch.start_sec - 4.0), min(asset.duration_sec, batch.end_sec + 4.0))] = None
        specs.update(dict.fromkeys(_plan_requested_context(requested, batches, asset.duration_sec, set())))
        seen_specs: set[tuple[float, float]] = set()
        built = None
        while specs:
            remaining_budget = self.settings.max_refinement_passes - len(refined_batches)
            if len(specs) > remaining_budget:
                if built is not None and built.can_publish and built.verified_events:
                    break  # Remaining uncertainty is preserved on the result.
                await write_audit("failed", built or provisional, error_code="refinement_budget_insufficient")
                error = AnalysisIncomplete("refinement_budget_insufficient")
                error.forte_timings = inference_snapshot()
                raise error
            await stage("refining")
            with timer.stage("refinement"):
                planned = []
                for offset, (start, end) in enumerate(specs):
                    fps = min(4.0, self.settings.max_frames_per_window / max(end - start, .001))
                    planned.append(context_window(
                        asset.duration_sec, start, end, context_sec=0,
                        sample_fps=fps, max_frames=self.settings.max_frames_per_window,
                        index=len(refined_batches) + offset,
                    ))
                self._checkpoint(deadline, cancel_event)
                # Decode the union once per round; independent forward scans from
                # the origin for every context window waste the video time budget.
                additional = await asyncio.to_thread(
                    decode_frames, asset, attempt_root / f"refinement-round-{len(rounds):02d}",
                    timestamps_sec=sampling_timestamps(planned),
                    deadline_monotonic=deadline, cancel=cancel_event,
                )
                frame_map.update({frame.frame_id: frame for frame in additional})
                requests = make_clip_requests(
                    job.job_id, additional, planned, fact_definitions=self.facts,
                    rule_definitions=self.rules, analysis_version=CONTRACT_VERSION,
                    budget=ProcessorBudget(max_frames=self.settings.max_frames_per_window),
                )
                current_round = []
                rounds.append(current_round)
                for request in requests:
                    try:
                        batch = await self._infer(request, deadline, cancel_event)
                    except BaseException as error:
                        await write_audit("failed", built or provisional, error_code="refinement_model_incomplete")
                        try:
                            error.forte_timings = inference_snapshot()
                        except (AttributeError, TypeError):
                            pass
                        raise
                    current_round.append(batch)
                    refined_batches.append(batch)
                seen_specs.update(specs)
            final_batches, pending, pending_requests = _reconcile_batches(batches, rounds, self.catalog)
            with timer.stage("rules"):
                built = build_events(final_batches, self.catalog, frame_map, coverage, config=self.event_config)
                built = _with_pending(built, pending, pending_requests)
                if staged:
                    screen_pending, screen_requests = _screen_pending(screen_batches,
                        primary_batches + refined_batches, self.catalog,
                        {fact.fact_id for fact in self.facts}, self.event_config)
                    built = _with_pending(built, screen_pending, screen_requests)
            await write_audit("refined", built)
            sources = batches + refined_batches
            next_specs = _plan_requested_context(built.refinement_requests, sources,
                                                  asset.duration_sec, seen_specs)
            specs = dict.fromkeys(next_specs)
            self._checkpoint(deadline, cancel_event)
        if built is None:
            final_batches, pending, pending_requests = _reconcile_batches(batches, rounds, self.catalog)
            with timer.stage("rules"):
                built = build_events(final_batches, self.catalog, frame_map, coverage, config=self.event_config)
                built = _with_pending(built, pending, pending_requests)
                if staged:
                    screen_pending, screen_requests = _screen_pending(screen_batches,
                        primary_batches + refined_batches, self.catalog,
                        {fact.fact_id for fact in self.facts}, self.event_config)
                    built = _with_pending(built, screen_pending, screen_requests)
        self._checkpoint(deadline, cancel_event)
        if not built.can_publish:
            await write_audit("failed", built, error_code="insufficient_visible_evidence")
            error = AnalysisIncomplete("insufficient_visible_evidence")
            error.forte_timings = inference_snapshot()
            raise error
        await write_audit("ready", built)
        await stage("assembling")
        with timer.stage("evidence"):
            try:
                assembled = await asyncio.to_thread(
                    assemble_result, job_id=job.job_id, asset=asset,
                    events=built.verified_events, frames=frame_map, rules=self.catalog,
                    evidence_dir=attempt_root / "evidence", analysis_complete=coverage.complete,
                    deadline=deadline, cancelled=cancel_event.is_set,
                )
            except BaseException as error:
                await write_audit("failed", built, error_code="evidence_incomplete")
                try:
                    error.forte_timings = inference_snapshot()
                except (AttributeError, TypeError):
                    pass
                raise
        self._checkpoint(deadline, cancel_event)
        await write_audit("done", built, evidence=assembled.stats)
        timings = inference_snapshot()
        return PipelineResult(assembled.json_bytes, timings, metadata_path)
