"""Explicit production consumer routing for separately verified native arms.

No model loading occurs here. Baseline/HF routing remains in the original
worker/API branches; importing this module does not import native backends.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import math
from pathlib import Path


def validate_native_selection(path, role, mode, hf_path=None, *, job_timeout=None) -> None:
    selected = (path is not None, role is not None, mode is not None)
    if not any(selected):
        return
    if not all(selected) or hf_path is not None:
        raise ValueError("native selection requires config, role and mode without HF selection")
    if role not in {"base_diagnostic", "fit_candidate"} or mode not in {"serial", "parallel4"}:
        raise ValueError("unsupported native role or mode")
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("native config path is required")
    if job_timeout is not None and (type(job_timeout) not in (int, float) or job_timeout != 285):
        raise ValueError("native production job timeout must remain 285 seconds")


def _contract(settings):
    from serving.qwen38_identity import load_json

    path = settings.model_native_launch_config_path
    role, mode = settings.model_native_role, settings.model_native_mode
    validate_native_selection(path, role, mode, settings.model_candidate_launch_config_path)
    if type(settings.job_timeout_seconds) not in (int, float) or settings.job_timeout_seconds != 285:
        raise ValueError("native production job timeout must remain 285 seconds")
    launch = load_json(path)
    if mode == "serial":
        from serving.qwen38_vllm_identity import validate_native_config, validate_native_readiness
        validate_native_config(launch)
        validator = validate_native_readiness
    else:
        from experimental_parallel4.profile import validate_parallel_config
        from experimental_parallel4.admission import validate_parallel_readiness
        validate_parallel_config(launch)
        validator = validate_parallel_readiness
    return launch, role, mode, validator


class _NativeProductionPipeline:
    """Keep the existing worker/evidence pipeline and its bounded job deadline."""

    def __init__(self, pipeline, settings, launch, role, mode, validator):
        self.pipeline = pipeline
        self.path = Path(settings.model_native_launch_config_path)
        self.file_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.launch, self.role, self.mode, self.validator = launch, role, mode, validator
        self._native_consumer_binding = _worker_binding(launch, role, mode, self.file_sha256)

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    @property
    def native_consumer_binding(self):
        return deepcopy(self._native_consumer_binding)

    async def analyze(self, job, *, stage, deadline, cancel_event):
        from forte.model.adapter import ModelUnavailable
        from serving.qwen38_identity import canonical_sha256

        try:
            unchanged = (self.path.stat().st_size <= 1024 * 1024
                         and hashlib.sha256(self.path.read_bytes()).hexdigest() == self.file_sha256)
        except OSError:
            unchanged = False
        if not unchanged:
            raise ModelUnavailable("native_consumer_configuration_changed")
        # This is GET-only readiness within the original absolute job deadline.
        # Inference still revalidates each request and owns its response cleanup.
        await self.pipeline.provider.delegate.check_ready(deadline=deadline, cancel_event=cancel_event)
        proof = self.pipeline.provider.delegate.serving_attestation
        self.validator(proof, self.launch, self.role)
        self.pipeline.audit_provenance["native_runtime_attestation"] = {
            "mode": self.mode, "native_role": self.role,
            "engine_instance_id": proof["engine_instance_id"],
            "config_file_sha256": self.file_sha256,
            "config_sha256": canonical_sha256(self.launch),
            "runtime_profile_sha256": self.launch["runtime_profile_sha256"],
            "loaded_engine_sha256": canonical_sha256(proof["loaded_engine"]),
            "generated_http_gate": deepcopy(proof["generated_http_gate"]),
            "actual_ready_verified": True, "hf_nf4_equivalent": False,
            "observation_scope": "before_this_job; not proof of drain after a later failure",
        }
        return await self.pipeline.analyze(job, stage=stage, deadline=deadline, cancel_event=cancel_event)

    async def aclose(self):
        await self.pipeline.aclose()


def make_native_worker_pipeline(settings):
    launch, role, mode, validator = _contract(settings)
    if settings.model_name != launch["roles"][role]["served_model_name"]:
        raise ValueError("native worker model alias does not match selected role")
    if settings.max_concurrent_jobs != 1 or settings.max_refinement_passes != 8:
        raise ValueError("native production profile requires one job and eight refinement passes")
    if mode == "serial":
        from serving.qwen38_vllm_provider import make_native_pipeline
        pipeline = make_native_pipeline(settings, launch, role=role,
            base_url=settings.model_base_url, api_key=settings.model_api_key)
    else:
        from experimental_parallel4.source_drain import make_source_draining_parallel4_pipeline
        pipeline = make_source_draining_parallel4_pipeline(settings, launch, role=role,
            base_url=settings.model_base_url, api_key=settings.model_api_key)
    return _NativeProductionPipeline(pipeline, settings, launch, role, mode, validator)


def _worker_binding(launch, role, mode, file_sha256):
    from serving.qwen38_identity import canonical_sha256
    return {"mode": mode, "native_role": role, "config_sha256": canonical_sha256(launch),
            "config_file_sha256": file_sha256,
            "runtime_profile_sha256": launch["runtime_profile_sha256"],
            "source_sha256": launch["source_archive"]["sha256"],
            "image_sha256": launch["runtime_image_sha256"],
            "served_model_name": launch["roles"][role]["served_model_name"]}


def native_worker_binding(settings):
    """Expected current route; a worker emits its immutable loaded snapshot."""
    launch, role, mode, _ = _contract(settings)
    return _worker_binding(launch, role, mode,
        hashlib.sha256(Path(settings.model_native_launch_config_path).read_bytes()).hexdigest())


async def _native_model_alive(settings) -> bool:
    """GET-only API readiness with exact role/engine/arithmetic/gate validation."""
    import httpx
    from forte.model.adapter import strict_json_object, validate_private_endpoint

    try:
        launch, role, _, validator = await asyncio.to_thread(_contract, settings)
        if not isinstance(settings.model_api_key, str) or not settings.model_api_key.strip():
            return False
        validate_private_endpoint(settings.model_base_url, ("model", "model-1", "model-2", "model-3"))
        base = settings.model_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        headers = {"Authorization": "Bearer " + settings.model_api_key,
                   "Accept-Encoding": "identity", "X-Forte-Native-Role": role,
                   "X-Forte-Native-Profile": launch["runtime_profile_sha256"]}
        async with httpx.AsyncClient(timeout=settings.readiness_timeout_seconds,
                trust_env=False, follow_redirects=False, headers=headers) as client:
            async def fetch(endpoint):
                async with client.stream("GET", base + endpoint) as response:
                    if response.status_code != 200 or response.headers.get("content-encoding", "identity") != "identity":
                        return None
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > 1024 * 1024:
                            return None
                    return bytes(raw)
            health, raw = await asyncio.gather(fetch("/health"), fetch("/readyz"))
        if health is None or raw is None:
            return False
        validator(strict_json_object(raw), launch, role)
        return True
    except Exception:
        # Readiness never promotes missing modules, stale proofs or bad configs.
        # Cancellation still propagates (CancelledError derives BaseException).
        return False


async def native_model_alive(settings) -> bool:
    """Bound the entire readiness path, including slow streaming responses."""
    timeout = settings.readiness_timeout_seconds
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        return False
    try:
        async with asyncio.timeout(timeout):
            return await _native_model_alive(settings)
    except TimeoutError:
        return False


__all__ = ["validate_native_selection", "make_native_worker_pipeline", "native_model_alive",
           "native_worker_binding"]
