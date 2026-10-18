"""Deny model-side media fetching. Workers send job-owned JPEG data URIs."""

import json
import os
from pathlib import Path


class DataOnlyMiddleware:
    """Private vLLM ASGI guard without importing application/API modules.

    No remote URLs, paths, embeddings or raw video/audio inputs are accepted.
    The model network is internal too; this middleware is defence in depth.
    """

    def __init__(self, app, max_body_bytes: int = 96 * 1024 * 1024):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") == "/readyz":
            # Do not synchronously call /health from inside this ASGI process.
            # The external container check separately validates engine health.
            from serving.readiness import is_ready
            try:
                config_path = Path(os.environ.get("FORTE_SERVING_CONFIG", "/app/configs/serving/baseline.json"))
                state_path = Path(os.environ.get("FORTE_SERVING_STATE", "/run/forte/ready.json"))
                config = json.loads(config_path.read_text(encoding="utf-8"))
                ready = is_ready(config, state_path, check_health=False)
                verified = json.loads(state_path.read_text(encoding="utf-8"))["snapshot_verification"] if ready else {}
            except (OSError, ValueError, KeyError, TypeError):
                ready = False
                verified = {}
            body = json.dumps({"ready": ready,
                               "model_revision": config["model_revision"] if ready else None,
                               "processor_revision": verified.get("processor_revision") if ready else None,
                               "model_manifest_sha256": verified.get("manifest_sha256") if ready else None,
                               "vllm_version": config["vllm_version"] if ready else None,
                               "model_files_verified": ready}).encode()
            await send({"type": "http.response.start", "status": 200 if ready else 503,
                        "headers": [(b"content-type", b"application/json")]})
            return await send({"type": "http.response.body", "body": body})
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        chunks = []
        size = 0
        while True:
            part = await receive()
            if part["type"] == "http.disconnect":
                return
            body = part.get("body", b"")
            size += len(body)
            if size > self.max_body_bytes:
                return await self._reject(send, 413)
            chunks.append(body)
            if not part.get("more_body", False):
                break
        raw = b"".join(chunks)
        try:
            document = json.loads(raw)
            if not isinstance(document, dict) or not self._safe(document):
                return await self._reject(send, 400)
        except (ValueError, RecursionError):
            return await self._reject(send, 400)
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": raw, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)

    @classmethod
    def _safe(cls, item):
        if isinstance(item, list):
            return all(cls._safe(value) for value in item)
        if not isinstance(item, dict):
            return True
        if item.get("type") in {"video_url", "input_audio", "audio_url", "image_embeds", "video_embeds"}:
            return False
        for key, value in item.items():
            if key in {"image_url", "url"}:
                address = value.get("url") if isinstance(value, dict) else value
                if not isinstance(address, str) or not address.startswith("data:image/jpeg;base64,"):
                    return False
            if key in {"video_url", "audio_url", "input_audio", "image_embeds", "video_embeds"}:
                return False
            if not cls._safe(value):
                return False
        return True

    @staticmethod
    async def _reject(send, status):
        # No echo of media or content-dependent input, including in errors.
        body = b'{"error":"Only bounded JPEG data URI model requests are accepted"}'
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
