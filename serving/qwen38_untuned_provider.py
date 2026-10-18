"""Explicit guarded provider injection for private untuned development runs.

This subclass reuses the production request/media/parser implementation, while
requiring a different base-only diagnostic readiness identity. It is never
selected by the production API/worker configuration or trained candidate gate.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import time

import httpx

from forte.model.adapter import (CancellationSignal, LocalModelConfig, LocalModelProvider,
                                 ModelUnavailable, _checkpoint, strict_json_object)
from serving.qwen38_identity import CandidateError, REVISION, load_json
from serving.qwen38_untuned import expected_untuned_identity, validate_untuned_config, validate_untuned_readiness


class UntunedModelProvider(LocalModelProvider):
    def __init__(self, runtime_config: dict | str | Path, *, media_root: Path,
                 base_url: str = "http://127.0.0.1:8000/v1", api_key: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        launch = load_json(runtime_config) if isinstance(runtime_config, (str, Path)) else runtime_config
        validate_untuned_config(launch)
        self.diagnostic_identity = expected_untuned_identity(launch)
        self.diagnostic_launch_config = launch
        # No fabricated adapter/training identity is sent through the trained
        # LocalModelConfig candidate contract. This class owns strict readiness.
        config = LocalModelConfig(media_root=Path(media_root), base_url=base_url,
            model=launch["served_model_name"], expected_revision=REVISION,
            api_key=api_key, max_frames=launch["limits"]["max_images"],
            max_source_pixels=launch["limits"]["max_source_pixels_per_image"],
            max_tokens=launch["generation"]["max_new_tokens"], seed=20261017,
            request_timeout_sec=launch["limits"]["request_timeout_seconds"],
            max_attempts=1, require_readiness=True, readiness_cache_sec=0,
            chat_template_kwargs=(("enable_thinking", False), ("preserve_thinking", False)))
        super().__init__(config, transport=transport)

    async def check_ready(self, *, deadline: float,
                          cancel_event: CancellationSignal | None = None) -> None:
        acquire = asyncio.create_task(self._readiness_lock.acquire())
        try:
            await self._bounded(acquire, deadline=deadline, cancel_event=cancel_event)
        except BaseException:
            if acquire.done() and not acquire.cancelled() and acquire.exception() is None and acquire.result():
                self._readiness_lock.release()
            raise
        try:
            _checkpoint(deadline, cancel_event)
            self._ready_at = None
            self._serving_attestation = None
            try:
                health_status, _ = await self._bounded(self._request("GET", "/health", deadline=deadline,
                    api_prefix=False), deadline=deadline, cancel_event=cancel_event)
                ready_status, raw = await self._bounded(self._request("GET", "/readyz", deadline=deadline,
                    api_prefix=False), deadline=deadline, cancel_event=cancel_event)
            except httpx.HTTPError:
                raise ModelUnavailable("untuned_endpoint_unavailable") from None
            if health_status != 200 or ready_status != 200:
                raise ModelUnavailable("untuned_not_warm_and_ready")
            ready = strict_json_object(raw)
            try:
                validate_untuned_readiness(ready, self.diagnostic_launch_config)
            except CandidateError:
                raise ModelUnavailable("untuned_readiness_identity_mismatch") from None
            await self.check_available(deadline=deadline, cancel_event=cancel_event)
            self._serving_attestation = {"source": "private_untuned_diagnostic_readyz",
                **self.diagnostic_identity, "ready": True, "verified": True,
                "model_files_verified": True, "model_loaded": True,
                "adapter_loaded_tensor_count": 0, "prewarm_completed": True,
                "schema_smoke_completed": True, "namespace_default_route_absent": True,
                "runtime_load": ready["runtime_load"]}
            self._ready_at = time.monotonic()
        finally:
            self._readiness_lock.release()


def make_untuned_pipeline(settings, runtime_config: dict | str | Path, *,
                         base_url: str = "http://127.0.0.1:8000/v1", api_key: str | None = None,
                         transport: httpx.AsyncBaseTransport | None = None, catalog=None):
    """Inject only the diagnostic model into the existing AnalysisPipeline.

    Sampling, decode, deterministic rules, prompts and original evidence remain
    governed by the caller's frozen WorkerSettings. This is a direct private
    development helper, not a new backend setting or public API route.
    """
    from forte.pipeline import AnalysisPipeline, AnalysisIncomplete

    launch = load_json(runtime_config) if isinstance(runtime_config, (str, Path)) else runtime_config
    validate_untuned_config(launch)
    if settings.max_frames_per_window > launch["limits"]["max_images"]:
        raise AnalysisIncomplete("untuned_worker_frame_budget_mismatch")
    diagnostic_settings = replace(settings, model_base_url=base_url,
        model_name=launch["served_model_name"], model_api_key=api_key,
        model_candidate_launch_config_path=None)
    provider = UntunedModelProvider(launch, media_root=diagnostic_settings.work_dir,
                                   base_url=base_url, api_key=api_key, transport=transport)
    pipeline = AnalysisPipeline(diagnostic_settings, provider=provider, catalog=catalog)
    pipeline.audit_provenance.update(model_id=launch["model_id"], model_revision=REVISION,
        processor_revision=REVISION, adapter_sha256=None, training_run_sha256=None,
        diagnostic_only=True, arm="qwen38_untuned", trained=False,
        revision_source="configured_untuned_diagnostic_pending_actual_readiness",
        untuned_diagnostic_expected_identity=provider.diagnostic_identity)
    return pipeline
