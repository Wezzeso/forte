"""HTTP adapter for a team-controlled vLLM server, with no external AI path.

Only job-owned JPEGs become inline image data. Paths, original filenames, audio,
labels, and model responses are never logged by this module. The HTTP client
ignores environment proxies, refuses redirects, and accepts private endpoints.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import math
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Awaitable, Protocol, TypeVar
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from PIL import Image, UnidentifiedImageError
from referencing.exceptions import Unresolvable

from forte.shared.schemas import ClipRequest


class ModelError(RuntimeError):
    """Sanitized failure suitable for an internal error code, never raw content."""
    code = "model_error"


class UnsafeMedia(ModelError):
    code = "unsafe_media"


class ModelUnavailable(ModelError):
    code = "model_unavailable"


class InvalidModelOutput(ModelError):
    code = "invalid_model_output"


class ModelDeadlineExceeded(ModelError):
    code = "model_deadline_exceeded"


class ModelCancelled(ModelError):
    code = "model_cancelled"


class CancellationSignal(Protocol):
    """Both asyncio.Event and threading.Event satisfy this nonblocking contract."""
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class LocalModelConfig:
    media_root: Path
    base_url: str = "http://model:8000/v1"
    model: str = "Qwen3-VL-8B-Instruct"
    expected_revision: str = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    expected_vllm_version: str = "0.31.0"
    api_key: str | None = field(default=None, repr=False)
    max_frames: int = 16
    max_image_bytes: int = 10 * 1024 * 1024
    max_clip_bytes: int = 64 * 1024 * 1024
    max_source_pixels: int = 64_000_000
    max_output_bytes: int = 512 * 1024
    max_tokens: int = 4096
    request_timeout_sec: float = 60.0
    connect_timeout_sec: float = 3.0
    max_attempts: int = 2
    retry_delay_sec: float = 0.1
    cleanup_grace_sec: float = 1.0
    seed: int = 20261017
    require_readiness: bool = True
    readiness_cache_sec: float = 2.0
    # Empty keeps the frozen baseline contract. A candidate must bind all of
    # these identities to its separately verified launch receipt, not an alias.
    expected_runtime_identity: tuple[tuple[str, str], ...] = ()
    # Explicit opt-in: send only controls verified against the actual template.
    chat_template_kwargs: tuple[tuple[str, bool], ...] = ()
    # These exact names are team-controlled Docker service names, never suffix matches.
    private_service_hosts: tuple[str, ...] = ("model", "model-1", "model-2", "model-3")

    def __post_init__(self) -> None:
        validate_private_endpoint(self.base_url, self.private_service_hosts)
        if not self.model or len(self.model) > 256:
            raise ValueError("model name must be bounded and nonempty")
        if not re.fullmatch(r"[a-f0-9]{40}", self.expected_revision):
            raise ValueError("model revision must be pinned")
        _validate_runtime_identity(self.expected_runtime_identity, self.expected_revision)
        try:
            controls = dict(self.chat_template_kwargs)
        except (TypeError, ValueError):
            raise ValueError("invalid verified chat template controls") from None
        if (type(self.chat_template_kwargs) is not tuple
                or any(type(pair) is not tuple for pair in self.chat_template_kwargs)
                or len(controls) != len(self.chat_template_kwargs)
                or not set(controls) <= {"enable_thinking", "preserve_thinking"}
                or any(type(value) is not bool for value in controls.values())):
            raise ValueError("invalid verified chat template controls")
        if not 1 <= self.max_frames <= 64 or not 1 <= self.max_attempts <= 3:
            raise ValueError("invalid frame or retry budget")
        if any(not math.isfinite(v) or v <= 0 for v in (
            self.request_timeout_sec, self.connect_timeout_sec, self.cleanup_grace_sec,
        )) or any(not math.isfinite(v) or v < 0 for v in (
            self.retry_delay_sec, self.readiness_cache_sec,
        )):
            raise ValueError("invalid timeout budget")
        if self.cleanup_grace_sec > 5:
            raise ValueError("cleanup grace must remain bounded")
        if any(v <= 0 for v in (self.max_image_bytes, self.max_clip_bytes,
                                self.max_source_pixels, self.max_output_bytes, self.max_tokens)):
            raise ValueError("byte, pixel, and output budgets must be positive")
        if self.expected_runtime_identity:
            if controls != {"enable_thinking": False, "preserve_thinking": False}:
                raise ValueError("candidate requires verified disabled thinking controls")
            generation = {"do_sample": False, "temperature": 0, "seed": self.seed,
                "max_new_tokens": self.max_tokens, **controls, "num_beams": 1}
            digest = hashlib.sha256(json.dumps(generation, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
            if digest != dict(self.expected_runtime_identity)["generation_sha256"]:
                raise ValueError("candidate generation configuration identity mismatch")


_CANDIDATE_HASH_FIELDS = frozenset({
    "tokenizer_sha256", "chat_template_sha256", "model_manifest_sha256",
    "adapter_sha256", "adapter_config_sha256", "training_run_sha256",
    "quantization_sha256", "generation_sha256", "image_sha256", "source_sha256", "config_sha256",
})
_CANDIDATE_IDENTITY_FIELDS = _CANDIDATE_HASH_FIELDS | {
    "model_id", "model_revision", "processor_revision", "backend",
}


def _validate_runtime_identity(pairs: tuple[tuple[str, str], ...], revision: str) -> None:
    if type(pairs) is not tuple:
        raise ValueError("candidate runtime identity must be immutable")
    if not pairs:
        return
    try:
        identity = dict(pairs)
        valid = (type(pairs) is tuple and all(type(pair) is tuple for pair in pairs)
            and len(identity) == len(pairs) and set(identity) == _CANDIDATE_IDENTITY_FIELDS
            and identity["backend"] in {"vllm", "transformers_peft"}
            and isinstance(identity["model_id"], str) and 0 < len(identity["model_id"]) <= 256
            and identity["model_revision"] == revision
            and identity["processor_revision"] == revision
            and all(isinstance(identity[key], str) and re.fullmatch(r"[a-f0-9]{64}", identity[key])
                    for key in _CANDIDATE_HASH_FIELDS))
    except (TypeError, ValueError, KeyError):
        valid = False
    if not valid:
        raise ValueError("candidate runtime identity must be complete and pinned")


@dataclass(frozen=True)
class InferenceMetrics:
    elapsed_sec: float
    http_elapsed_sec: float
    preprocessing_sec: float
    attempts: int
    input_frames: int
    input_dimensions: tuple[tuple[int, int], ...]
    encoded_media_bytes: int
    prompt_tokens: int | None
    generated_tokens: int | None
    total_tokens: int | None
    readiness_sec: float = 0.0
    serving_attestation: dict[str, Any] | None = None
    server_metrics: dict[str, float | None] | None = None
    # The standard vLLM response does not expose actual processor grids. None
    # means unavailable; source dimensions are never mislabeled as processor output.
    processor_output_dimensions: tuple[tuple[int, int], ...] | None = None
    visual_tokens: int | None = None
    score_provenance: str | None = None


@dataclass(frozen=True, repr=False)
class ModelResponse:
    data: dict[str, Any] = field(repr=False)
    metrics: InferenceMetrics
    model: str
    finish_reason: str = "stop"

    def __repr__(self) -> str:
        return f"ModelResponse(model={self.model!r}, metrics={self.metrics!r}, data=<private>)"


def _allowed_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.is_unspecified or address.is_multicast or address.is_link_local:
        return False
    if address.is_loopback:
        return True
    # is_private also admits documentation/reserved ranges; use explicit networks.
    networks = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
    return any(address in ipaddress.ip_network(network) for network in networks
               if address.version == ipaddress.ip_network(network).version)


def validate_private_endpoint(url: str, service_hosts: tuple[str, ...] = ()) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise ValueError("invalid private model endpoint") from None
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") not in {"", "/v1"}
            or (port is not None and not 1 <= port <= 65535)):
        raise ValueError("invalid private model endpoint")
    hostname = parsed.hostname.lower()
    if not (_allowed_ip(hostname) or hostname == "localhost" or hostname in service_hosts):
        raise ValueError("model endpoint must be a private literal or approved service name")
    if hostname in service_hosts and not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", hostname):
        raise ValueError("approved model hosts must be simple internal service names")


def _checkpoint(deadline: float, cancel_event: CancellationSignal | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ModelCancelled("model_cancelled")
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise ModelDeadlineExceeded("model_deadline_exceeded")


T = TypeVar("T")


def _cancel_once(task: asyncio.Future[Any]) -> None:
    """A second Task.cancel can interrupt transport cleanup already in progress."""
    if not task.done() and (not isinstance(task, asyncio.Task) or not task.cancelling()):
        task.cancel()


async def _settle(tasks: tuple[asyncio.Future[Any], ...], *,
                  retained: set[asyncio.Task[Any]], grace_sec: float) -> None:
    """Retain cleanup independently; repeated caller cancellation cannot cancel it.

    An expired grace bounds the caller's drain wait, not the lifetime of cleanup.
    The retained task consumes all child exceptions and removes itself on exit.
    A locally settled stream is not proof that the remote GPU is idle.
    """
    async def drain() -> None:
        await asyncio.gather(*tasks, return_exceptions=True)
    cleanup = asyncio.create_task(drain())
    retained.add(cleanup)
    cleanup.add_done_callback(retained.discard)
    expires = time.monotonic() + grace_sec
    interrupted = False
    while not cleanup.done():
        remaining = expires - time.monotonic()
        if remaining <= 0:
            break
        try:
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=remaining)
        except TimeoutError:
            break
        except asyncio.CancelledError:
            # Preserve cancellation after a bounded chance to close the socket.
            interrupted = True
    if interrupted:
        raise asyncio.CancelledError


async def _bounded(awaitable: Awaitable[T], *, deadline: float,
                   cancel_event: CancellationSignal | None,
                   retained: set[asyncio.Task[Any]], cleanup_grace_sec: float) -> T:
    """Bound work, then give its shielded transport cleanup a separate short grace."""
    task = asyncio.ensure_future(awaitable)
    async def wait_cancel() -> None:
        while cancel_event is not None and not cancel_event.is_set():
            await asyncio.sleep(0.02)
    watcher = asyncio.create_task(wait_cancel()) if cancel_event is not None else None
    try:
        _checkpoint(deadline, cancel_event)
        waiters = [task] if watcher is None else [task, watcher]
        done, _ = await asyncio.wait(waiters, timeout=max(0, deadline - time.monotonic()),
                                     return_when=asyncio.FIRST_COMPLETED)
        if watcher is not None and watcher in done:
            raise ModelCancelled("model_cancelled")
        if task not in done:
            raise ModelDeadlineExceeded("model_deadline_exceeded")
        _checkpoint(deadline, cancel_event)
        return task.result()
    finally:
        _cancel_once(task)
        if watcher is not None:
            _cancel_once(watcher)
        await _settle((task, *([watcher] if watcher is not None else [])),
                      retained=retained, grace_sec=cleanup_grace_sec)


def _reject_constant(_: str) -> None:
    raise ValueError("nonfinite JSON")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("nonfinite JSON")
    return parsed


def _unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def strict_json_object(value: str | bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(value, parse_constant=_reject_constant, parse_float=_finite_json_float,
                            object_pairs_hook=_unique_json_keys)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise InvalidModelOutput("model_json_invalid") from None
    if not isinstance(parsed, dict):
        raise InvalidModelOutput("model_json_object_required")
    return parsed


def _validator(schema: dict[str, Any]) -> Draft202012Validator:
    """Never let JSON Schema fetch a reference from the network."""
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                    not isinstance(item, str) or not item.startswith("#")
                ):
                    raise ValueError("response schema requires local references")
                if key in {"$id", "$schema"} and key == "$id" and (
                    not isinstance(item, str) or not item.startswith("#")
                ):
                    raise ValueError("response schema cannot change reference scope")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    try:
        if len(json.dumps(schema, allow_nan=False)) > 128 * 1024:
            raise ValueError("response schema is too large")
        visit(schema)
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)
    except (SchemaError, TypeError, RecursionError):
        raise ValueError("invalid response schema") from None


def _prepare_media(clip: ClipRequest, config: LocalModelConfig, deadline: float,
                   cancel_event: CancellationSignal | None) -> tuple[list[dict[str, Any]], tuple[tuple[int, int], ...], int]:
    _checkpoint(deadline, cancel_event)
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}", clip.job_id):
        raise UnsafeMedia("media_job_id_invalid")
    root = Path(config.media_root).resolve()
    owned = (root / clip.job_id).resolve()
    if not owned.is_relative_to(root) or owned == root:
        raise UnsafeMedia("media_job_boundary_invalid")
    if not 1 <= len(clip.frame_refs) <= config.max_frames:
        raise UnsafeMedia("media_frame_budget_exceeded")
    blocks: list[dict[str, Any]] = []
    dimensions: list[tuple[int, int]] = []
    total_bytes = 0
    encoded_bytes = 0
    for frame in clip.frame_refs:
        _checkpoint(deadline, cancel_event)
        if frame.job_id is not None and frame.job_id != clip.job_id:
            raise UnsafeMedia("media_job_mismatch")
        # URLs, UNC/network shares and Windows alternate data streams are forbidden.
        path_value = str(frame.path)
        if "://" in path_value or path_value.startswith(("\\\\", "//")):
            raise UnsafeMedia("media_local_path_required")
        if ":" in path_value[2:]:
            raise UnsafeMedia("media_local_path_required")
        try:
            path = Path(path_value).resolve(strict=True)
            if not path.is_relative_to(owned) or not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
                raise UnsafeMedia("media_outside_job_or_not_jpeg")
            with path.open("rb") as handle:
                raw = handle.read(config.max_image_bytes + 1)
        except (OSError, RuntimeError, ValueError):
            raise UnsafeMedia("media_file_unavailable") from None
        if len(raw) > config.max_image_bytes:
            raise UnsafeMedia("media_image_byte_budget_exceeded")
        if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
            raise UnsafeMedia("media_jpeg_invalid")
        if frame.jpeg_sha256 is not None and hashlib.sha256(raw).hexdigest() != frame.jpeg_sha256:
            raise UnsafeMedia("media_integrity_mismatch")
        total_bytes += len(raw)
        if total_bytes > config.max_clip_bytes:
            raise UnsafeMedia("media_clip_byte_budget_exceeded")
        try:
            with Image.open(BytesIO(raw)) as image:
                if image.format != "JPEG" or image.size != (frame.width, frame.height):
                    raise UnsafeMedia("media_dimensions_or_format_mismatch")
                if image.width * image.height > config.max_source_pixels:
                    raise UnsafeMedia("media_pixel_budget_exceeded")
                image.load()
                # Strip EXIF/comments/path metadata. Keep all source pixels and scene context.
                clean = BytesIO()
                image.convert("RGB").save(clean, format="JPEG", quality=95)
                encoded = clean.getvalue()
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            raise UnsafeMedia("media_jpeg_decode_failed") from None
        _checkpoint(deadline, cancel_event)
        encoded_bytes += len(encoded)
        if len(encoded) > config.max_image_bytes or encoded_bytes > config.max_clip_bytes:
            raise UnsafeMedia("media_encoded_byte_budget_exceeded")
        dimensions.append((frame.width, frame.height))
        blocks.append({"type": "text", "text": json.dumps({"frame_id": frame.frame_id,
                       "timestamp_sec": frame.pts_sec}, separators=(",", ":"))})
        blocks.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")}})
    return blocks, tuple(dimensions), encoded_bytes


class LocalModelProvider:
    """Async context manager; caller supplies the absolute monotonic job deadline."""

    def __init__(self, config: LocalModelConfig, *, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        # A supplied transport is for synthetic tests; no inference fallback exists.
        self._client = httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False)
        self._ready_at: float | None = None
        self._serving_attestation: dict[str, Any] | None = None
        self._readiness_lock = asyncio.Lock()
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()

    async def __aenter__(self) -> LocalModelProvider:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._cleanup_tasks:
            await _settle(tuple(self._cleanup_tasks), retained=self._cleanup_tasks,
                          grace_sec=self.config.cleanup_grace_sec)
        await _settle((asyncio.create_task(self._client.aclose()),), retained=self._cleanup_tasks,
                      grace_sec=self.config.cleanup_grace_sec)

    @property
    def pending_cleanup_count(self) -> int:
        """Safe local drain state; never a claim of backend request completion."""
        return sum(not task.done() for task in self._cleanup_tasks)

    async def _bounded(self, awaitable: Awaitable[T], *, deadline: float,
                       cancel_event: CancellationSignal | None) -> T:
        return await _bounded(awaitable, deadline=deadline, cancel_event=cancel_event,
                              retained=self._cleanup_tasks,
                              cleanup_grace_sec=self.config.cleanup_grace_sec)

    @property
    def serving_attestation(self) -> dict[str, Any] | None:
        """Actual private launcher response after successful readiness, never config guesses."""
        return dict(self._serving_attestation) if self._serving_attestation is not None else None

    def _url(self, endpoint: str, *, api_prefix: bool = True) -> str:
        base = self.config.base_url.rstrip("/")
        if not api_prefix:
            parsed = urlsplit(base)
            return f"{parsed.scheme}://{parsed.netloc}" + endpoint
        return (base if base.endswith("/v1") else base + "/v1") + endpoint

    def _headers(self) -> dict[str, str]:
        headers = {"Accept-Encoding": "identity"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def _request(self, method: str, endpoint: str, *, deadline: float,
                       payload: dict[str, Any] | None = None,
                       api_prefix: bool = True) -> tuple[int, bytes]:
        timeout = min(self.config.request_timeout_sec, max(0.001, deadline - time.monotonic()))
        request = self._client.build_request(method, self._url(endpoint, api_prefix=api_prefix),
            json=payload, headers=self._headers(), timeout=httpx.Timeout(timeout,
            connect=min(timeout, self.config.connect_timeout_sec)))
        response: httpx.Response | None = None
        try:
            response = await self._client.send(request, stream=True)
            if response.status_code != 200:
                # Error bodies may echo private prompts. Do not read or include them.
                return response.status_code, b""
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise InvalidModelOutput("model_response_encoding_unsupported")
            raw = bytearray()
            # Own the one response close below. HTTPX's aiter_bytes implicitly
            # closes on EOF before our shield can protect a slow stream close.
            async for chunk in response.stream:
                raw.extend(chunk)
                if len(raw) > self.config.max_output_bytes:
                    raise InvalidModelOutput("model_response_byte_budget_exceeded")
            return response.status_code, bytes(raw)
        finally:
            if response is not None:
                await _settle((asyncio.create_task(response.aclose()),),
                              retained=self._cleanup_tasks, grace_sec=self.config.cleanup_grace_sec)

    async def check_available(self, *, deadline: float,
                              cancel_event: CancellationSignal | None = None) -> None:
        """Checks served model identity only; warm/verified readiness belongs to A15."""
        try:
            status, raw = await self._bounded(self._request("GET", "/models", deadline=deadline),
                                         deadline=deadline, cancel_event=cancel_event)
        except httpx.HTTPError:
            raise ModelUnavailable("model_endpoint_unavailable") from None
        if status != 200:
            raise ModelUnavailable("model_endpoint_unavailable")
        data = strict_json_object(raw)
        entries = data.get("data")
        if not isinstance(entries, list) or not any(
            isinstance(entry, dict) and entry.get("id") == self.config.model for entry in entries
        ):
            raise ModelUnavailable("model_identity_not_served")

    async def check_ready(self, *, deadline: float,
                          cancel_event: CancellationSignal | None = None) -> None:
        """Require A15's hash-verified pinned model, synthetic prewarm, and live engine.

        /models alone identifies an alias and cannot establish model provenance.
        /readyz is private launcher attestation after offline checksum validation.
        """
        acquire = asyncio.create_task(self._readiness_lock.acquire())
        try:
            await self._bounded(acquire, deadline=deadline, cancel_event=cancel_event)
        except BaseException:
            # Cancellation can race with a successful acquire. _bounded has
            # already settled the task; release its resource before propagating.
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
                raise ModelUnavailable("model_endpoint_unavailable") from None
            if health_status != 200 or ready_status != 200:
                raise ModelUnavailable("model_not_warm_and_ready")
            ready = strict_json_object(raw)
            expected_runtime = dict(self.config.expected_runtime_identity)
            if (ready.get("ready") is not True or ready.get("model_files_verified") is not True
                    or ready.get("model_revision") != self.config.expected_revision
                    or ready.get("processor_revision") != self.config.expected_revision
                    or not isinstance(ready.get("model_manifest_sha256"), str)
                    or not re.fullmatch(r"[a-f0-9]{64}", ready["model_manifest_sha256"])
                    or ((not expected_runtime or expected_runtime["backend"] == "vllm")
                        and ready.get("vllm_version") != self.config.expected_vllm_version)
                    or any(ready.get(key) != value for key, value in expected_runtime.items())
                    or (bool(expected_runtime) and (
                        ready.get("model_loaded") is not True or ready.get("prewarm_completed") is not True
                        or type(ready.get("adapter_loaded_tensor_count")) is not int
                        or ready["adapter_loaded_tensor_count"] <= 0))):
                raise ModelUnavailable("model_readiness_identity_mismatch")
            await self.check_available(deadline=deadline, cancel_event=cancel_event)
            self._serving_attestation = {
                "source": "private_launcher_readyz", "model_revision": ready["model_revision"],
                "model_files_verified": True, "vllm_version": ready.get("vllm_version"),
                "processor_revision": ready.get("processor_revision"),
                "model_manifest_sha256": ready.get("model_manifest_sha256"),
                **{key: ready[key] for key in expected_runtime},
            }
            self._ready_at = time.monotonic()
        finally:
            self._readiness_lock.release()

    async def infer(self, clip: ClipRequest, *, prompt: str, response_schema: dict[str, Any],
                    deadline: float, cancel_event: CancellationSignal | None = None) -> ModelResponse:
        started = time.monotonic()
        _checkpoint(deadline, cancel_event)
        readiness_sec = 0.0
        if self.config.require_readiness and (self._ready_at is None
                or started - self._ready_at > self.config.readiness_cache_sec):
            await self.check_ready(deadline=deadline, cancel_event=cancel_event)
            readiness_sec = time.monotonic() - started
        preprocessing_started = time.monotonic()
        request_attestation = self.serving_attestation
        if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > 256 * 1024:
            raise ValueError("prompt must be nonempty and bounded")
        validator = _validator(response_schema)
        blocks, dimensions, media_bytes = await self._bounded(asyncio.to_thread(
            _prepare_media, clip, self.config, deadline, cancel_event),
            deadline=deadline, cancel_event=cancel_event)
        preprocessed = time.monotonic()
        payload = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": blocks}],
            "structured_outputs": {"json": response_schema},
            "temperature": 0.0, "seed": self.config.seed, "max_tokens": self.config.max_tokens,
            "stream": False, "n": 1,
        }
        if self.config.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.config.chat_template_kwargs)
        http_elapsed = 0.0
        for attempt in range(1, self.config.max_attempts + 1):
            _checkpoint(deadline, cancel_event)
            request_started = time.monotonic()
            retryable = False
            try:
                status, raw = await self._bounded(self._request("POST", "/chat/completions",
                    payload=payload, deadline=deadline), deadline=deadline, cancel_event=cancel_event)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                # These failures occur before this POST can be admitted. A read,
                # write or protocol failure has an ambiguous execution outcome.
                status, raw, retryable = 0, b"", True
            except httpx.HTTPError:
                status, raw = 0, b""
            finally:
                http_elapsed += time.monotonic() - request_started
            if status == 200:
                return self._parse(raw, validator, started=started,
                    preprocessing_sec=preprocessed - preprocessing_started,
                    readiness_sec=readiness_sec, http_elapsed_sec=http_elapsed,
                    serving_attestation=request_attestation,
                    attempts=attempt, dimensions=dimensions, media_bytes=media_bytes)
            if not retryable or attempt >= self.config.max_attempts:
                self._ready_at = None
                self._serving_attestation = None
                raise ModelUnavailable(f"model_http_failure_{status}")
            await self._bounded(asyncio.sleep(self.config.retry_delay_sec * attempt),
                           deadline=deadline, cancel_event=cancel_event)
        raise ModelUnavailable("model_attempts_exhausted")

    def _parse(self, raw: bytes, validator: Draft202012Validator, *, started: float,
               preprocessing_sec: float, readiness_sec: float, http_elapsed_sec: float, attempts: int,
               serving_attestation: dict[str, Any] | None,
               dimensions: tuple[tuple[int, int], ...], media_bytes: int) -> ModelResponse:
        envelope = strict_json_object(raw)
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise InvalidModelOutput("model_single_choice_required")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise InvalidModelOutput("model_incomplete_or_truncated")
        if envelope.get("model") != self.config.model:
            raise InvalidModelOutput("model_response_identity_mismatch")
        message = choice.get("message")
        if (not isinstance(message, dict) or message.get("role") != "assistant"
                or message.get("refusal") or message.get("tool_calls") or message.get("function_call")):
            raise InvalidModelOutput("model_message_invalid")
        if (dict(self.config.chat_template_kwargs).get("enable_thinking") is False
                and (message.get("reasoning_content") or message.get("reasoning"))):
            raise InvalidModelOutput("model_thinking_disabled_violation")
        content = message.get("content")
        if not isinstance(content, str):
            raise InvalidModelOutput("model_text_content_required")
        data = strict_json_object(content)
        try:
            validator.validate(data)
        except (ValidationError, Unresolvable, RecursionError):
            raise InvalidModelOutput("model_output_schema_invalid") from None
        usage = envelope.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        def count(key: str) -> int | None:
            value = usage.get(key)
            return value if type(value) is int and value >= 0 else None
        reported_metrics = envelope.get("metrics")
        server_metrics = None
        if isinstance(reported_metrics, dict):
            server_metrics = {}
            for key in ("time_to_first_token_ms", "generation_time_ms", "queue_time_ms",
                        "mean_itl_ms", "tokens_per_second"):
                value = reported_metrics.get(key)
                try:
                    number = float(value) if type(value) in {int, float} else None
                except OverflowError:
                    number = None
                server_metrics[key] = (number if number is not None
                    and math.isfinite(number) and number >= 0 else None)
        metrics = InferenceMetrics(
            elapsed_sec=time.monotonic() - started, http_elapsed_sec=http_elapsed_sec,
            preprocessing_sec=preprocessing_sec, attempts=attempts, input_frames=len(dimensions),
            input_dimensions=dimensions, encoded_media_bytes=media_bytes,
            prompt_tokens=count("prompt_tokens"), generated_tokens=count("completion_tokens"),
            total_tokens=count("total_tokens"),
            readiness_sec=readiness_sec,
            serving_attestation=serving_attestation,
            server_metrics=server_metrics,
        )
        return ModelResponse(data=data, metrics=metrics, model=self.config.model)
