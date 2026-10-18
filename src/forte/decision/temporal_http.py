"""CPU-only private client for the optional native GPU sidecar.

No ML dependency, external API, model fallback or private-content logging.
Returning logits is distinct from calibrated observations or a jury verdict.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from forte.shared.schemas import VideoAsset

from .head_contract import HeadIdentity
from .temporal_native import NativeHeadResult
from .temporal_provider import TemporalProviderError, _checkpoint

MAX_NATIVE_RESPONSE_BYTES = 64 * 1024 * 1024


def private_native_endpoint(endpoint: str, allowed_hostnames: tuple[str, ...] = ()) -> str:
    try:
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"} or parsed.port is None):
            raise ValueError
        host = parsed.hostname.lower()
        if host not in {"localhost", *allowed_hostnames}:
            address = ipaddress.ip_address(host)
            networks = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
            if not address.is_loopback and not any(address in ipaddress.ip_network(net) for net in networks):
                raise ValueError
        return endpoint.rstrip("/")
    except (ValueError, TypeError):
        raise TemporalProviderError("native_endpoint_must_be_explicit_private_service") from None


class HttpNativeHeadClient:
    """Return source-bound native outputs; CPU calibration/legal guards follow.

    The default client ignores proxy environment variables and refuses redirects.
    Named internal hosts require an explicit operator-owned allowlist. No source
    bytes go to a public API. The GPU sidecar independently constrains owned paths
    and recomputes the entire actual uploaded asset SHA before decoding.
    """
    def __init__(self, endpoint: str, *, identity: HeadIdentity,
                 allowed_hostnames: tuple[str, ...] = (), client: httpx.AsyncClient | None = None,
                 expected_execution_manifest_sha256: str | None = None,
                 expected_source_code_bundle_sha256: str | None = None):
        self.endpoint = private_native_endpoint(endpoint, allowed_hostnames)
        self.identity = identity
        for value in (expected_execution_manifest_sha256, expected_source_code_bundle_sha256):
            if value is not None and not re.fullmatch(r"[a-f0-9]{64}", value):
                raise TemporalProviderError("native_execution_pin_invalid")
        self.expected_execution_manifest_sha256 = expected_execution_manifest_sha256
        self.expected_source_code_bundle_sha256 = expected_source_code_bundle_sha256
        self._client = client or httpx.AsyncClient(trust_env=False, follow_redirects=False)
        self._owns_client = client is None

    async def _json(self, method: str, path: str, *, deadline: float,
                    body: dict[str, Any] | None = None, cancel_event: Any = None,
                    limit: int = MAX_NATIVE_RESPONSE_BYTES) -> dict[str, Any]:
        _checkpoint(deadline, cancel_event)
        try:
            async with asyncio.timeout_at(deadline):
                async with self._client.stream(method, self.endpoint + path, json=body,
                        timeout=max(0.001, deadline - time.monotonic()), follow_redirects=False) as response:
                    if response.status_code != 200:
                        raise TemporalProviderError("native_sidecar_non_success_status")
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        _checkpoint(deadline, cancel_event)
                        if len(raw) + len(chunk) > limit:
                            raise TemporalProviderError("native_sidecar_response_budget_exceeded")
                        raw.extend(chunk)
            _checkpoint(deadline, cancel_event)
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError
                    result[key] = value
                return result
            value = json.loads(raw, object_pairs_hook=pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(value, dict):
                raise ValueError
            return value
        except TemporalProviderError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise TimeoutError("optional native sidecar deadline exceeded") from None
        except Exception:
            raise TemporalProviderError("native_sidecar_response_invalid_or_unavailable") from None

    async def check_ready(self, *, deadline: float) -> dict[str, Any]:
        value = await self._json("GET", "/ready", deadline=deadline, limit=64 * 1024)
        if (value.get("ready") is not True or value.get("head_sha256") != self.identity.head_sha256
                or value.get("feature_spec_sha256") != self.identity.feature_spec_sha256):
            raise TemporalProviderError("native_sidecar_identity_or_readiness_failed")
        if (self.expected_execution_manifest_sha256 is not None
                and value.get("execution_manifest_sha256") != self.expected_execution_manifest_sha256
                or self.expected_source_code_bundle_sha256 is not None
                and value.get("source_code_bundle_sha256") != self.expected_source_code_bundle_sha256):
            raise TemporalProviderError("native_sidecar_execution_pin_mismatch")
        result = {"ready": True, "head_sha256": self.identity.head_sha256,
            "feature_spec_sha256": self.identity.feature_spec_sha256, "candidate_calibration_verified": False}
        for key in ("execution_manifest_sha256", "source_code_bundle_sha256"):
            if isinstance(value.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", value[key]):
                result[key] = value[key]
        return result

    async def predict_native(self, asset: VideoAsset, *, clip_id: str,
                             deadline: float, cancel_event: Any = None) -> NativeHeadResult:
        _checkpoint(deadline, cancel_event)
        if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", clip_id):
            raise TemporalProviderError("native_clip_identity_invalid")
        if self.expected_execution_manifest_sha256 is not None or self.expected_source_code_bundle_sha256 is not None:
            await self.check_ready(deadline=deadline)
        # Arbitrary metadata never enters the model or the service request.
        owned_asset = {**asset.model_dump(mode="json"), "metadata": {}}
        raw = await self._json("POST", "/v1/temporal/predict", deadline=deadline,
            cancel_event=cancel_event, body={"asset": owned_asset, "clip_id": clip_id,
                "deadline_budget_sec": min(120.0, max(0.001, deadline - time.monotonic()))})
        try:
            result = NativeHeadResult.model_validate_json(json.dumps(raw, allow_nan=False), strict=True)
            result.context.validate_asset(asset, self.identity)
            if result.output.identity != self.identity or result.output.clip_id != clip_id:
                raise TemporalProviderError("native_sidecar_result_identity_mismatch")
            _checkpoint(deadline, cancel_event)
            return result
        except TemporalProviderError:
            raise
        except TimeoutError:
            raise
        except Exception:
            raise TemporalProviderError("native_sidecar_result_binding_invalid") from None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
