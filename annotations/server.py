"""Loopback-only human workbench; no cloud APIs, telemetry, or request logging."""

from __future__ import annotations

import json
import hashlib
import mimetypes
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .catalog import load_taxonomy
from .store import ReviewError, ReviewStore, RevisionConflict, canonical

STATIC = Path(__file__).with_name("ui")
MAX_JSON_BYTES = 2 * 1024 * 1024


def make_server(store: ReviewStore, port: int = 8765) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalReview/1"

        def log_message(self, *_args: object) -> None:
            pass  # Never disclose video routes, notes or private records in logs.

        def valid_host(self) -> bool:
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def authorized(self, query: dict[str, list[str]]) -> bool:
            expected_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if self.headers.get("Origin") not in expected_origins | {None}:
                return False
            supplied = self.headers.get("X-Review-Token", "") or query.get("token", [""])[0]
            return secrets.compare_digest(token, supplied)

        def headers_for(self, status: int, content_type: str, length: int | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; img-src 'self' blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if length is not None:
                self.send_header("Content-Length", str(length))

        def send_json(self, status: int, value: object) -> None:
            body = canonical(value).encode("utf-8")
            self.headers_for(status, "application/json; charset=utf-8", len(body))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            if not self.valid_host():
                self.send_json(403, {"error": "loopback_host_required"})
                return
            if parsed.path == "/":
                body = (STATIC / "index.html").read_text(encoding="utf-8").replace("__SESSION_TOKEN__", token).encode("utf-8")
                self.headers_for(200, "text/html; charset=utf-8", len(body))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path in {"/app.js", "/style.css"}:
                path = STATIC / parsed.path[1:]
                body = path.read_bytes()
                self.headers_for(200, "text/javascript; charset=utf-8" if path.suffix == ".js" else "text/css; charset=utf-8", len(body))
                self.end_headers()
                self.wfile.write(body)
                return
            if not self.authorized(query):
                self.send_json(403, {"error": "local_session_required"})
                return
            try:
                if parsed.path == "/api/taxonomy":
                    self.send_json(200, load_taxonomy())
                elif parsed.path == "/api/queue":
                    self.send_json(200, {"summary": store.aggregate(), "records": [{"sample_id": row["sample_id"], "review_state": row["review_state"], "revision": row["revision"], "duration_sec": row["duration_sec"], "split": row["split"]} for row in store.list_records()]})
                elif parsed.path.startswith("/api/records/"):
                    sample_id = parsed.path.removeprefix("/api/records/")
                    self.send_json(200, store.get(sample_id))
                elif parsed.path.startswith("/media/"):
                    sample_id = parsed.path.removeprefix("/media/")
                    record = store.get(sample_id)
                    proxy = store.workspace / "proxies" / (sample_id + ".mp4")
                    path = proxy if proxy.is_file() else Path(record["media_paths"][0])
                    self.stream_video(path)
                elif parsed.path.startswith("/frames/"):
                    parts = [unquote(part) for part in parsed.path.removeprefix("/frames/").split("/")]
                    if len(parts) != 2:
                        self.send_json(404, {"error": "not_found"})
                        return
                    record = store.get(parts[0])
                    frames = list(record.get("frames", []))
                    for proposal in record.get("local_model_proposals", []):
                        frames.extend(proposal.get("frames", []))
                    matches = [frame for frame in frames if frame["frame_id"] == parts[1]]
                    if not matches:
                        self.send_json(404, {"error": "unknown_registered_frame"})
                        return
                    frame = matches[0]
                    path = Path(frame["path"])
                    if path.stat().st_size > 16 * 1024 * 1024:
                        raise ReviewError("registered frame exceeds local preview limit")
                    body = path.read_bytes()
                    if frame.get("jpeg_sha256") and hashlib.sha256(body).hexdigest() != frame["jpeg_sha256"]:
                        raise ReviewError("registered source frame changed after import")
                    from PIL import Image
                    from io import BytesIO
                    with Image.open(BytesIO(body)) as image:
                        if image.format != "JPEG" or image.size != (frame["width"], frame["height"]):
                            raise ReviewError("registered frame is not its declared source JPEG")
                        image.verify()
                    self.headers_for(200, "image/jpeg", len(body))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_json(404, {"error": "not_found"})
            except ReviewError as exc:
                self.send_json(400, {"error": str(exc)})
            except (OSError, ValueError, TypeError):
                self.send_json(400, {"error": "local_read_failed"})

        def stream_video(self, path: Path) -> None:
            size = path.stat().st_size
            requested = self.headers.get("Range")
            start, end = 0, size - 1
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
                if not match or not any(match.groups()):
                    self.headers_for(416, "text/plain", 0)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                first, last = match.groups()
                if first:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                else:
                    start = max(0, size - int(last))
                if start > end or start >= size:
                    self.headers_for(416, "text/plain", 0)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            if size <= 0:
                self.send_json(404, {"error": "empty_local_media"})
                return
            self.headers_for(206 if requested else 200, mimetypes.guess_type(path.name)[0] or "video/quicktime", end - start + 1)
            self.send_header("Accept-Ranges", "bytes")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                try:
                    while remaining:
                        chunk = handle.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            if not self.valid_host() or not self.authorized({}):
                self.send_json(403, {"error": "local_session_required"})
                return
            if not parsed.path.startswith("/api/records/"):
                self.send_json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_JSON_BYTES:
                    self.send_json(413, {"error": "bounded_json_body_required"})
                    return
                body = json.loads(self.rfile.read(length))
                updated = store.save(parsed.path.removeprefix("/api/records/"), body["patch"], expected_revision=body["expected_revision"])
                self.send_json(200, updated)
            except RevisionConflict as exc:
                self.send_json(409, {"error": str(exc)})
            except ReviewError as exc:
                self.send_json(422, {"error": str(exc)})
            except (OSError, ValueError, KeyError, TypeError):
                self.send_json(400, {"error": "invalid_local_review_request"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)

