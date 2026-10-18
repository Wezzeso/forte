"""Authenticate before receiving any protected request body."""

from __future__ import annotations

import hmac

from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings


class BodyLimitExceeded(Exception):
    pass


class ScopedCorsMiddleware:
    """Opt-in browser access to authenticated API routes from explicit origins.

    This wrapper never reads a body. Only a valid preflight may finish before
    the API-key boundary; all actual requests continue through authentication.
    No cookies, ambient credentials, origin regexes or wildcard policy are used.
    """

    _METHODS = frozenset({"GET", "POST"})
    _HEADERS = frozenset({"x-api-key", "content-type", "range"})

    def __init__(self, app: ASGIApp, config: Settings) -> None:
        self.app = app
        self.origins = frozenset(config.cors_allowed_origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        cors_route = path.startswith(("/api/v1/", "/api/ui/")) or path in {"/readyz", "/healthz", "/runtime-config.json"}
        headers = scope.get("headers", [])
        origins = [value for name, value in headers if name.lower() == b"origin"]
        if not cors_route or not origins or not self.origins:
            await self.app(scope, receive, send)
            return
        origin = origins[0].decode("latin-1") if len(origins) == 1 else ""
        if origin not in self.origins:
            await JSONResponse({"detail": "Origin is not allowed"}, status_code=403, headers={"Cache-Control": "no-store", "Vary": "Origin"})(scope, receive, send)
            return

        async def cors_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.extend([
                    (b"access-control-allow-origin", origin.encode("ascii")),
                    (b"access-control-expose-headers", b"Accept-Ranges, Content-Range, Content-Length, Server-Timing, X-Forte-Timing-Origin"),
                ])
                vary_values = [value for name, value in response_headers if name.lower() == b"vary"]
                response_headers = [(name, value) for name, value in response_headers if name.lower() != b"vary"]
                response_headers.append((b"vary", b", ".join(vary_values + [b"Origin"])))
                message = {**message, "headers": response_headers}
            await send(message)

        if scope["method"] == "OPTIONS":
            requested_methods = [value for name, value in headers if name.lower() == b"access-control-request-method"]
            requested_headers = [value for name, value in headers if name.lower() == b"access-control-request-headers"]
            method = requested_methods[0].decode("latin-1") if len(requested_methods) == 1 else ""
            header_names = requested_headers[0].decode("latin-1").lower().split(",") if len(requested_headers) == 1 else []
            valid_headers = len(requested_headers) <= 1 and all(name.strip() in self._HEADERS for name in header_names)
            if method not in self._METHODS or not valid_headers:
                await JSONResponse({"detail": "CORS preflight is not allowed"}, status_code=403, headers={"Cache-Control": "no-store"})(scope, receive, cors_send)
                return
            await Response(status_code=204, headers={
                "Access-Control-Allow-Methods": "GET, POST",
                "Access-Control-Allow-Headers": "X-API-Key, Content-Type, Range",
                "Access-Control-Max-Age": "600",
                "Cache-Control": "no-store",
            })(scope, receive, cors_send)
            return
        if scope["method"] not in self._METHODS:
            await JSONResponse({"detail": "Cross-origin method is not allowed"}, status_code=405, headers={"Cache-Control": "no-store"})(scope, receive, cors_send)
            return
        await self.app(scope, receive, cors_send)


class ApiBoundaryMiddleware:
    def __init__(self, app: ASGIApp, config: Settings) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        protected = path.startswith(("/api/v1/", "/api/ui/"))
        headers = scope.get("headers", [])
        if protected:
            supplied = [value for name, value in headers if name.lower() == b"x-api-key"]
            valid = len(supplied) == 1 and any(
                hmac.compare_digest(supplied[0], expected.encode("utf-8"))
                for expected in self.config.api_keys
            )
            if not valid:
                await JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)(scope, receive, send)
                return
        bounded = scope["method"] == "POST" and path.rstrip("/") == "/api/v1/analyze"
        if bounded and self.config.admission_paused:
            # Authentication above remains first. Reject before multipart,
            # Content-Length handling, persistence or any request-body read.
            await JSONResponse({"detail": "Analysis admission is temporarily paused"},
                status_code=503, headers={"Retry-After": "60", "Cache-Control": "no-store"})(scope, receive, send)
            return
        maximum = self.config.max_upload_bytes + self.config.multipart_overhead_bytes
        if bounded:
            lengths = [value for name, value in headers if name.lower() == b"content-length"]
            try:
                if len(lengths) > 1 or (lengths and int(lengths[0]) < 0):
                    raise ValueError
                if lengths and int(lengths[0]) > maximum:
                    await JSONResponse({"detail": "Upload exceeds configured limit"}, status_code=413)(scope, receive, send)
                    return
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
                return
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if bounded and message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > maximum:
                    raise BodyLimitExceeded
            return message

        async def secure_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                response_headers = list(message.get("headers", []))
                response_headers.extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"same-origin"),
                ])
                if protected:
                    response_headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": response_headers}
            await send(message)

        try:
            await self.app(scope, limited_receive, secure_send)
        except BodyLimitExceeded:
            if started:
                raise
            await JSONResponse({"detail": "Upload exceeds configured limit"}, status_code=413)(scope, receive, secure_send)
