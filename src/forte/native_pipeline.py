"""Optional arbitrary-upload native-head analysis, separate from the live factory.

Only an explicitly supplied, artifact-verified machine calibration can interpret
the private sidecar's logits. Binary video scores never supply visual facts or
event times. All public bytes pass through the ordinary rules/evidence code.
No experimental vocabulary, file-name cache, training or model fallback exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any

from forte.config import WorkerSettings
from forte.decision import DecisionConfig
from forte.decision.machine_contract import (
    MachineCalibrationManifest, MachineEvidenceVerifier, machine_candidate_readiness,
    machine_manifest_digest,
)
from forte.decision.temporal_native import NativeHeadResult
from forte.decision.temporal_http import HttpNativeHeadClient
from forte.decision.temporal_observations import map_native_observations
from forte.decision.temporal_provider import (
    PhysicalFactRunPolicy, load_physical_fact_run_policy, physical_policy_observation_definitions,
)
from forte.events import EventBuildConfig, RefinementRequest, UnresolvedInterval, build_events
from forte.evidence import assemble_result
from forte.observability import StageTimer, aggregate_inference_metrics
from forte.pipeline import AnalysisIncomplete, PipelineResult, StageCallback, atomic_private_json
from forte.reproducibility import build_reproducibility_bundle
from forte.shared.schemas import CONTRACT_VERSION, FrameRef, VideoAsset
from forte.video.evidence_seek import materialize_evidence_frames_seek
from forte.video.probe import probe_video
from rules.catalog import load_catalog


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


class _Cancellation:
    def __init__(self, external: threading.Event):
        self.external = external
        self.local = threading.Event()

    def is_set(self) -> bool:
        return self.local.is_set() or self.external.is_set()


async def _drain(task: asyncio.Task, *, cancel: bool, grace_sec: float = 2.0) -> None:
    """Cancel once and retain ownership until cleanup finishes or is bounded."""
    if cancel and not task.done() and not task.cancelling():
        task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=grace_sec)
    except BaseException:
        # The original cancellation/error is propagated by the caller. Retrieving
        # a late failure prevents an unobserved task exception during shutdown.
        if not task.done():
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())


class NativeAnalysisPipeline:
    """Explicitly optional native route with the existing worker call interface.

    ``client`` implements check_ready/predict_native/aclose. Production callers
    supply HttpNativeHeadClient and ArtifactMachineEvidenceVerifier; injectable
    probe/materializer functions are for synthetic, source-bound integration.
    An absence policy is explicit pinned configuration, never inferred here.
    """

    def __init__(self, settings: WorkerSettings, *, client: Any,
                 manifest: MachineCalibrationManifest, verifier: MachineEvidenceVerifier,
                 catalog: Any = None, physical_fact_run_policy: PhysicalFactRunPolicy | None = None,
                 absence_context_policy: Mapping[str, Any] | Path | None = None,
                 probe: Callable = probe_video,
                 materializer: Callable = materialize_evidence_frames_seek):
        self.settings = settings
        self.client = client
        if isinstance(client, HttpNativeHeadClient) and not all((
            client.expected_execution_manifest_sha256, client.expected_source_code_bundle_sha256,
        )):
            raise AnalysisIncomplete("native_execution_and_code_pins_required")
        self.manifest = MachineCalibrationManifest.model_validate_json(manifest.model_dump_json(), strict=True)
        self.verifier = verifier
        self._manifest_sha256 = machine_manifest_digest(self.manifest)
        self.catalog = catalog if catalog is not None else load_catalog(settings.rules_path)
        if getattr(self.catalog, "is_experimental", False):
            raise AnalysisIncomplete("native_experimental_vocabulary_requires_separate_route")
        self.rules = self.catalog.enabled_rules()
        identity = self.manifest.identity
        definitions = self.catalog.fact_definitions(trusted_scene_context=settings.trusted_scene_context)
        by_id = {fact.fact_id: fact for fact in definitions}
        if not self.rules or len(identity.fact_ids) != 19 or set(identity.fact_ids) != set(by_id):
            raise AnalysisIncomplete("native_baseline_fact_axis_mismatch")
        self.facts = tuple(by_id[key] for key in identity.fact_ids)
        if getattr(client, "identity", None) != identity:
            raise AnalysisIncomplete("native_client_calibration_identity_mismatch")
        if sha256(Path(settings.taxonomy_path).read_bytes()).hexdigest() != identity.taxonomy_sha256:
            raise AnalysisIncomplete("native_taxonomy_pin_mismatch")
        self._catalog_sha256 = sha256(Path(settings.rules_path).read_bytes()).hexdigest()
        compiled = load_physical_fact_run_policy(identity=identity,
            expected_catalog_sha256=self._catalog_sha256, catalog_path=settings.rules_path)
        if physical_fact_run_policy is not None and physical_fact_run_policy != compiled:
            raise AnalysisIncomplete("native_rule_policy_pin_mismatch")
        self.physical_policy = compiled
        self.event_definitions, absence = physical_policy_observation_definitions(compiled)
        self.absence_context_fact_ids = dict(absence)
        self.absence_policy = self._absence_policy(absence_context_policy, identity.fact_ids)
        self.absence_context_fact_ids.update(self.absence_policy.get("context_fact_ids", {}))
        self.event_config = EventBuildConfig(event_types=self.event_definitions)
        self._catalog_payload_sha256 = _digest(self.catalog.payload)
        self.probe = probe
        self.materializer = materializer
        self._closed = False
        self._active: set[asyncio.Task] = set()
        self._scopes: dict[asyncio.Task, _Cancellation] = {}
        self._blocking_tasks: set[asyncio.Task] = set()
        self._native_binding = {
            "schema_version": "forte.native-pipeline-binding.v1",
            "head_identity": identity.model_dump(mode="json"),
            "machine_calibration_sha256": self._manifest_sha256,
            "physical_policy_sha256": _digest(compiled.model_dump(mode="json")),
            "absence_policy_sha256": _digest(self.absence_policy),
            "catalog_file_sha256": self._catalog_sha256,
            "evidence_projection_limit": 96, "evidence_seek_batch_limit": 3,
            "experimental_enabled": False,
            "execution_manifest_sha256": getattr(client, "expected_execution_manifest_sha256", None),
            "source_code_bundle_sha256": getattr(client, "expected_source_code_bundle_sha256", None),
        }
        self.reproducibility_bundle = {
            "base": build_reproducibility_bundle(settings, self.catalog,
                DecisionConfig(concise_observations=True), None),
            "native": self._native_binding,
        }
        self.reproducibility_bundle["bundle_sha256"] = _digest(self.reproducibility_bundle)
        self._observation_config_sha256 = _digest({
            "event_definitions": self.event_definitions,
            "absence_context_fact_ids": self.absence_context_fact_ids,
            "fact_definitions": [fact.model_dump(mode="json") for fact in self.facts],
        })

    def _absence_policy(self, supplied: Mapping[str, Any] | Path | None,
                        fact_ids: Sequence[str]) -> dict[str, Any]:
        if supplied is None:
            return {}
        if isinstance(supplied, Path):
            if supplied.is_symlink() or supplied.stat().st_size > 256 * 1024:
                raise AnalysisIncomplete("native_absence_policy_invalid")
            supplied = json.loads(supplied.read_text(encoding="utf-8"))
        value = dict(supplied)
        allowed = {"schema_version", "taxonomy_sha256", "head_sha256", "context_fact_ids"}
        if (set(value) - allowed or value.get("schema_version") != "forte.fact-absence-context-policy.v1"
                or value.get("taxonomy_sha256") != self.manifest.identity.taxonomy_sha256
                or value.get("head_sha256", self.manifest.identity.head_sha256) != self.manifest.identity.head_sha256
                or not isinstance(value.get("context_fact_ids"), Mapping)):
            raise AnalysisIncomplete("native_absence_policy_invalid")
        known = set(fact_ids)
        contexts = {}
        for fact_id, prerequisites in value["context_fact_ids"].items():
            if (fact_id not in known or not isinstance(prerequisites, (list, tuple)) or not prerequisites
                    or len(prerequisites) != len(set(prerequisites)) or not set(prerequisites) <= known
                    or fact_id in prerequisites
                    or not set(self.absence_context_fact_ids.get(fact_id, ())) <= set(prerequisites)):
                raise AnalysisIncomplete("native_absence_policy_invalid")
            contexts[fact_id] = tuple(sorted(prerequisites))
        return {**value, "context_fact_ids": contexts}

    def _configuration_checkpoint(self) -> None:
        if (machine_manifest_digest(self.manifest) != self._manifest_sha256
                or getattr(self.client, "identity", None) != self.manifest.identity
                or _digest(self.catalog.payload) != self._catalog_payload_sha256
                or _digest(self.physical_policy.model_dump(mode="json")) != self._native_binding["physical_policy_sha256"]
                or _digest(self.absence_policy) != self._native_binding["absence_policy_sha256"]
                or _digest({"event_definitions": self.event_definitions,
                    "absence_context_fact_ids": self.absence_context_fact_ids,
                    "fact_definitions": [fact.model_dump(mode="json") for fact in self.facts],
                }) != self._observation_config_sha256):
            raise AnalysisIncomplete("native_loaded_binding_changed")

    @staticmethod
    def _checkpoint(deadline: float, cancel: _Cancellation) -> None:
        if cancel.is_set():
            raise AnalysisIncomplete("analysis_cancelled")
        if not math.isfinite(deadline) or time.monotonic() >= deadline:
            raise TimeoutError("analysis_deadline_exceeded")

    async def _blocking(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        # asyncio cancellation does not kill a Python thread. Retain it while
        # A05's shared cancellation/deadline hard-stops its codec subprocess.
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._blocking_tasks.add(task)
        task.add_done_callback(self._blocking_tasks.discard)
        try:
            return await asyncio.shield(task)
        except BaseException:
            scope = self._scopes.get(asyncio.current_task())
            if scope is not None:
                scope.local.set()
            await _drain(task, cancel=False)
            raise

    async def analyze(self, job: Any, *, stage: StageCallback, deadline: float,
                      cancel_event: threading.Event) -> PipelineResult:
        if self._closed:
            raise AnalysisIncomplete("native_pipeline_closed")
        scope = _Cancellation(cancel_event)
        operation = asyncio.create_task(self._analyze(job, stage=stage, deadline=deadline, cancel=scope))
        self._active.add(operation)
        self._scopes[operation] = scope
        try:
            while not operation.done():
                self._checkpoint(deadline, scope)
                await asyncio.wait({operation}, timeout=min(.05, max(.001, deadline - time.monotonic())))
            return operation.result()
        except BaseException:
            scope.local.set()
            await _drain(operation, cancel=True)
            raise
        finally:
            def finished(task):
                self._active.discard(task)
                self._scopes.pop(task, None)
            if operation.done():
                finished(operation)
            else:
                operation.add_done_callback(finished)

    async def aclose(self) -> None:
        self._closed = True
        active = tuple(self._active)
        for scope in self._scopes.values():
            scope.local.set()
        for task in active:
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*(_drain(task, cancel=False) for task in active))
        await asyncio.gather(*(_drain(task, cancel=False) for task in tuple(self._blocking_tasks)))
        close = getattr(self.client, "aclose", None)
        if close is not None:
            await close()

    async def _analyze(self, job: Any, *, stage: StageCallback, deadline: float,
                       cancel: _Cancellation) -> PipelineResult:
        timer = StageTimer()
        if (not re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", str(job.job_id))
                or job.job_id in {".", ".."} or type(job.attempt) is not int or job.attempt < 1):
            raise AnalysisIncomplete("native_job_identity_invalid")
        attempt_root = (self.settings.work_dir / job.job_id / f"attempt-{job.attempt}").resolve()
        if not attempt_root.is_relative_to(self.settings.work_dir.resolve()):
            raise AnalysisIncomplete("native_job_work_path_invalid")
        attempt_root.mkdir(parents=True, exist_ok=True)
        metadata_path = attempt_root / "analysis.private.json"
        asset = None
        result = None
        plan = None
        batch = None
        built = None
        readiness = None
        serving_attestation = None
        materialized: list[FrameRef] = []
        seek_metrics: list[dict] = []
        coverage: dict[str, Any] = {"complete": False, "gaps": [], "missing_window_indices": [0],
                                    "planned_intervals": [], "inspected_intervals": []}

        def snapshot():
            metrics = [getattr(batch, "metrics", {})] if batch is not None else []
            value = timer.snapshot()
            value["model_requests"] = aggregate_inference_metrics(metrics)
            # NativeHeadResult itself permits fixed, finite numeric metric keys.
            value["native_model"] = dict(result.metrics) if result is not None else {}
            value["native_evidence_seek"] = {
                "operation_count": len(seek_metrics),
                **{key: sum(float(item.get(key, 0)) for item in seek_metrics) for key in (
                    "source_hash_sec", "decode_jpeg_sec", "total_sec", "decoded_source_frames", "keyframe_groups")},
            }
            value["native_cleanup_pending"] = sum(not task.done() for task in self._blocking_tasks)
            return value

        async def audit(phase: str, error_code: str | None = None, evidence: Any = None):
            windows = []
            if batch is not None:
                windows = [{"clip_id": batch.clip_id, "start_sec": batch.start_sec, "end_sec": batch.end_sec,
                    "frame_ids": [frame.frame_id for frame in batch.request.frame_refs],
                    "requested_fact_ids": [fact.fact_id for fact in batch.request.fact_definitions],
                    "requested_rule_ids": [rule.rule_id for rule in batch.request.rule_definitions],
                    "phase": "native_full", "generation": 0, "complete": batch.complete,
                    "disposition": batch.disposition, "prompt_version": batch.prompt_version,
                    "config_version": batch.config_version, "prompt_hash": batch.prompt_hash,
                    "facts": [fact.model_dump(mode="json") for fact in batch.facts],
                    "candidate_events": [candidate.model_dump(mode="json") for candidate in batch.candidates]}]
            events = [event.model_dump(mode="json") for event in getattr(built, "verified_events", ())]
            provenance = {"execution_location": "team_deployed_service", "external_ai_used": False,
                "local_run_id": f"{job.job_id}:{job.attempt}", "pipeline_kind": "native_temporal",
                "model_id": self.manifest.identity.encoder_id,
                "model_revision": self.manifest.identity.head_sha256,
                "processor_version": None, "processor_revision": self.manifest.identity.encoder_revision,
                "actual_snapshot_verified": False, "revision_source": "calibration_bound_identity",
                "provisional": True, "human_fact_event_gold": False,
                "independent_fact_event_accuracy_measured": False,
                "reproducibility_bundle": self.reproducibility_bundle}
            if serving_attestation is not None:
                provenance["native_serving_attestation"] = serving_attestation
            payload = {"schema_version": "forte.local-proposal.v1", "phase": phase, "error_code": error_code,
                "job_id": job.job_id, "attempt": job.attempt, "analysis_version": CONTRACT_VERSION,
                "catalog_version": self.catalog.version, "pipeline_kind": "native_temporal",
                "asset": asset.model_dump(mode="json") if asset is not None else None,
                "frames": [frame.model_dump(mode="json") for frame in materialized],
                "decisions": [fact for window in windows for fact in window["facts"]],
                "candidate_events": [candidate for window in windows for candidate in window["candidate_events"]],
                "verified_events": events, "events": events, "window_decisions": windows,
                "native_source_context": result.context.model_dump(mode="json") if result is not None else None,
                "native_output_sha256": _digest(result.output.model_dump(mode="json")) if result is not None else None,
                "native_observations": plan.data if plan is not None else None,
                "native_calibration": {"manifest_sha256": self._manifest_sha256,
                    "eligible_fact_states": dict(readiness.eligible_fact_states) if readiness else {},
                    "reason_codes": list(readiness.reason_codes) if readiness else []},
                "provenance": provenance, "coverage": coverage,
                "publication_ready": bool(phase in {"ready", "done"} and built is not None and built.can_publish),
                "limitations": list(getattr(built, "limitations", ())),
                "unresolved_intervals": [asdict(item) for item in getattr(built, "unresolved_intervals", ())],
                "refinement_plan": [asdict(item) for item in getattr(built, "refinement_requests", ())],
                "evidence": asdict(evidence) if evidence is not None else None, "timings": snapshot()}
            await self._blocking(atomic_private_json, metadata_path, payload)

        try:
            self._checkpoint(deadline, cancel)
            self._configuration_checkpoint()
            await stage("decoding")
            with timer.stage("probe_decode"):
                asset = await self._blocking(self.probe, Path(job.owned_path), job_id=job.job_id,
                    deadline_monotonic=deadline, cancel=cancel)
                asset = VideoAsset.model_validate(asset)
                if (asset.job_id != job.job_id or asset.content_hash != job.content_hash
                        or Path(asset.owned_path).resolve() != Path(job.owned_path).resolve()):
                    raise AnalysisIncomplete("accepted_source_hash_or_ownership_changed")
                coverage.update(duration_sec=asset.duration_sec, planned_intervals=[[0.0, asset.duration_sec]],
                                gaps=[[0.0, asset.duration_sec]])
            await audit("probed")
            await stage("screening")
            with timer.stage("coarse_model"):
                readiness = await self._blocking(machine_candidate_readiness, self.manifest,
                    self.manifest.identity, verifier=self.verifier)
                if not readiness.evidence_verified:
                    raise AnalysisIncomplete("native_machine_calibration_not_verified")
                ready = await self.client.check_ready(deadline=deadline)
                if (not isinstance(ready, Mapping) or ready.get("ready") is not True
                        or ready.get("head_sha256") != self.manifest.identity.head_sha256
                        or ready.get("feature_spec_sha256") != self.manifest.identity.feature_spec_sha256):
                    raise AnalysisIncomplete("native_sidecar_not_ready")
                serving_attestation = {key: ready[key] for key in (
                    "ready", "head_sha256", "feature_spec_sha256", "execution_manifest_sha256",
                    "source_code_bundle_sha256", "candidate_calibration_verified",
                ) if key in ready}
                self._checkpoint(deadline, cancel)
                clip_id = f"native-{sha256(f'{job.job_id}:{job.attempt}'.encode()).hexdigest()[:24]}"
                returned = await self.client.predict_native(asset, clip_id=clip_id, deadline=deadline,
                                                            cancel_event=cancel)
                result = NativeHeadResult.model_validate_json(returned.model_dump_json(), strict=True)
                result.context.validate_asset(asset, self.manifest.identity)
                if (result.output.identity != self.manifest.identity or result.output.clip_id != clip_id
                        or result.context.clip_id != clip_id):
                    raise AnalysisIncomplete("native_returned_identity_mismatch")
                coverage.update(complete=True, gaps=[], missing_window_indices=[],
                    policy="complete_native_source_chain", native_frame_count=len(result.context.frames),
                    inspected_intervals=[{"window_index": 0, "start_sec": 0.0, "end_sec": asset.duration_sec,
                        "frame_ids": [], "frame_pts_sec": [], "actual_start_sec": 0.0,
                        "actual_end_sec": result.context.frames[-1].pts_sec}])
            with timer.stage("frame_preparation"):
                # A trained calibrated boundary decoder owns its event axes.
                # Adding a second full fact-run bracket over two decoded
                # occurrences would invent a third, broad duplicate event.
                boundary_axes = (set(readiness.candidate_event_types)
                    if result.output.boundary_logits is not None
                    and result.output.trained_components.get("event_boundaries") is True else set())
                fallback_policy = self.physical_policy.model_copy(update={"rules": tuple(
                    rule for rule in self.physical_policy.rules if rule.event_type_id not in boundary_axes)})
                plan = await self._blocking(map_native_observations, result, asset, self.manifest,
                    verifier=self.verifier, fact_definitions=self.facts, rule_definitions=self.rules,
                    event_definitions=self.event_definitions,
                    absence_context_fact_ids=self.absence_context_fact_ids,
                    physical_fact_run_policy=fallback_policy,
                    trusted_scene_context=self.settings.trusted_scene_context)
            await audit("native_observations")
            self._checkpoint(deadline, cancel)
            if not any(fact["state"] != "unclear" for fact in plan.data["facts"]):
                raise AnalysisIncomplete("native_no_supported_fact_classes")
            with timer.stage("frame_preparation"):
                sources = result.context.source_index()
                source_map = {frame.frame_id: frame for frame in sources}
                selected = [source_map[frame_id] for frame_id in plan.requested_frame_ids]
                for offset in range(0, len(selected), 3):
                    self._checkpoint(deadline, cancel)
                    measurements = {}
                    refs = await self._blocking(self.materializer, asset, selected[offset:offset + 3],
                        attempt_root / "frames", source_index=sources,
                        deadline_monotonic=deadline, cancel=cancel, metrics=measurements)
                    seek_metrics.append(measurements)
                    materialized.extend(FrameRef.model_validate(frame) for frame in refs)
                for frame in materialized:
                    if (frame.job_id != asset.job_id or not Path(frame.path).resolve().is_relative_to(attempt_root)
                            or frame.rotation_deg != asset.rotation_deg):
                        raise AnalysisIncomplete("native_materialized_ownership_or_path_invalid")
                batch = plan.bind_evidence_frames(materialized)
                coverage["inspected_intervals"][0].update(
                    frame_ids=[frame.frame_id for frame in batch.request.frame_refs],
                    frame_pts_sec=[frame.pts_sec for frame in batch.request.frame_refs])
            with timer.stage("rules"):
                frame_map = {frame.frame_id: frame for frame in materialized}
                built = build_events([batch], self.catalog, frame_map, coverage, config=self.event_config)
                pending = []
                requests = []
                # Full-window facts with no localized event do not license a
                # clean result. This is checked on the original batch only.
                proposed = {candidate.event_type_id for candidate in batch.candidates}
                facts = {fact.fact_id: fact for fact in batch.facts}
                for rule in self.rules:
                    binding = self.catalog.bindings[rule.rule_id]
                    states = binding["required_fact_states"]
                    if (binding["event_type_id"] not in proposed
                            and all(key in facts and facts[key].state == state and facts[key].supporting_frame_ids
                                    for key, state in states.items())):
                        pending.append(UnresolvedInterval(0.0, asset.duration_sec, "rule_facts_without_candidate",
                                                          event_type_id=binding["event_type_id"]))
                        requests.append(RefinementRequest(0.0, asset.duration_sec, "rule_facts_without_candidate",
                                                          required_fact_ids=tuple(sorted(states))))
                if any(fact.state == "unclear" for fact in batch.facts):
                    pending.append(UnresolvedInterval(0.0, asset.duration_sec, "native_unknown_fact_classes"))
                if any(not states for states in readiness.eligible_fact_states.values()):
                    pending.append(UnresolvedInterval(0.0, asset.duration_sec, "native_untrained_fact_classes"))
                for index, frame in enumerate(result.context.frames):
                    if not result.output.frame_valid[index] or not result.output.global_view_available[index]:
                        pending.append(UnresolvedInterval(frame.pts_sec,
                            frame.next_pts_sec if frame.next_pts_sec is not None else asset.duration_sec,
                            "native_unusable_feature_context"))
                limits = {item.reason for item in pending}
                if pending:
                    built = replace(built, unresolved_intervals=built.unresolved_intervals + tuple(pending),
                        refinement_requests=built.refinement_requests + tuple(requests),
                        verified_events=tuple(event.model_copy(update={
                            "limitations": sorted(set(event.limitations) | limits)}) for event in built.verified_events))
            await audit("grounded")
            self._checkpoint(deadline, cancel)
            if not built.can_publish:
                raise AnalysisIncomplete("native_insufficient_visible_evidence")
            await audit("ready")
            await stage("assembling")
            with timer.stage("evidence"):
                # Each JPEG already came from A05's hash-checked original seek.
                # The established assembler validates image bytes and every
                # source/support relationship again; no redundant video scan.
                def existing_originals(source, requested, directory):
                    self._checkpoint(deadline, cancel)
                    if source != asset or any(frame_map.get(frame.frame_id) != frame for frame in requested):
                        raise AnalysisIncomplete("native_evidence_reuse_binding_invalid")
                    return list(requested)
                assembled = await self._blocking(assemble_result, job_id=job.job_id, asset=asset,
                    events=built.verified_events, frames=frame_map, rules=self.catalog,
                    evidence_dir=attempt_root / "evidence", analysis_complete=built.can_publish,
                    extractor=existing_originals, deadline=deadline, cancelled=cancel.is_set)
            self._checkpoint(deadline, cancel)
            await audit("done", evidence=assembled.stats)
            return PipelineResult(assembled.json_bytes, snapshot(), metadata_path)
        except BaseException as error:
            cancel.local.set()
            code = error.code if isinstance(error, AnalysisIncomplete) else (
                "analysis_cancelled" if isinstance(error, asyncio.CancelledError) else
                "analysis_deadline_exceeded" if isinstance(error, TimeoutError) else "native_analysis_incomplete")
            try:
                await audit("failed", error_code=code)
            except BaseException:
                pass  # Preserve the original operational error/cancellation.
            try:
                error.forte_timings = snapshot()
            except (AttributeError, TypeError):
                pass
            if isinstance(error, (AnalysisIncomplete, asyncio.CancelledError, TimeoutError)):
                raise
            safe = AnalysisIncomplete("native_analysis_incomplete")
            safe.forte_timings = snapshot()
            raise safe from None


__all__ = ["NativeAnalysisPipeline"]
