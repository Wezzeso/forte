"""Bounded streaming multipart ingestion into an owned, durable file.

No media decoding, probing, or model imports belong in this module. Multipart
headers are bounded independently of the file, and contents stream directly to
disk rather than collecting a full upload in RAM or copying a spooled upload.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, Request
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool

from .config import Settings


@dataclass(frozen=True)
class StoredUpload:
    path: Path
    sha256: str
    size: int
    original_filename: str


def fsync_directory(path: Path) -> None:
    # Windows does not support opening directory handles with os.open/fsync.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _MultipartUpload:
    def __init__(self, target: BinaryIO, maximum: int) -> None:
        self.target = target
        self.maximum = maximum
        self.size = 0
        self.digest = hashlib.sha256()
        self.parts = 0
        self.finished = False
        self.filename = ""
        self.headers: dict[bytes, bytes] = {}
        self.header_name = bytearray()
        self.header_value = bytearray()
        self.header_bytes = 0

    def on_part_begin(self) -> None:
        self.parts += 1
        if self.parts > 1:
            raise HTTPException(400, "Exactly one multipart file field is required")

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._limit_header(end - start)
        self.header_name.extend(data[start:end])

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._limit_header(end - start)
        self.header_value.extend(data[start:end])

    def _limit_header(self, count: int) -> None:
        self.header_bytes += count
        if self.header_bytes > 16 * 1024:
            raise HTTPException(400, "Multipart headers are too large")

    def on_header_end(self) -> None:
        name = bytes(self.header_name).lower()
        if name in self.headers:
            raise HTTPException(400, "Duplicate multipart header")
        self.headers[name] = bytes(self.header_value)
        self.header_name.clear()
        self.header_value.clear()

    def on_headers_finished(self) -> None:
        disposition, options = parse_options_header(self.headers.get(b"content-disposition", b""))
        if disposition != b"form-data" or options.get(b"name") != b"file" or not options.get(b"filename"):
            raise HTTPException(400, "Multipart field 'file' is required")
        raw_filename = options[b"filename"]
        try:
            filename = raw_filename.decode("utf-8")
        except UnicodeDecodeError:
            filename = raw_filename.decode("latin-1")
        self.filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if len(raw_filename) > 1024 or "\x00" in filename or not self.filename.lower().endswith(".mov"):
            raise HTTPException(415, "A .mov video is required")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        self.size += end - start
        if self.size > self.maximum:
            raise HTTPException(413, "Upload exceeds configured limit")
        piece = memoryview(data)[start:end]
        self.target.write(piece)
        self.digest.update(piece)

    def on_end(self) -> None:
        self.finished = True


async def persist_upload(request: Request, config: Settings, job_id: str) -> StoredUpload:
    media_type, options = parse_options_header(request.headers.get("content-type", ""))
    boundary = options.get(b"boundary", b"")
    if media_type != b"multipart/form-data" or not boundary or len(boundary) > 200:
        raise HTTPException(415, "Expected multipart/form-data with a boundary")
    temporary = config.upload_root / f".{job_id}.part"
    destination = config.upload_root / f"{job_id}.mov"
    committed = False
    handle = await run_in_threadpool(temporary.open, "xb")
    try:
        upload = _MultipartUpload(handle, config.max_upload_bytes)
        callbacks = {name: getattr(upload, name) for name in (
            "on_part_begin", "on_header_field", "on_header_value", "on_header_end",
            "on_headers_finished", "on_part_data", "on_end",
        )}
        parser = MultipartParser(boundary, callbacks)
        try:
            async for chunk in request.stream():
                if chunk:
                    await run_in_threadpool(parser.write, chunk)
            await run_in_threadpool(parser.finalize)
        except MultipartParseError as exc:
            raise HTTPException(400, "Malformed multipart upload") from exc
        if not upload.finished or upload.parts != 1 or not upload.filename or upload.size == 0:
            raise HTTPException(400, "Incomplete or empty multipart file")

        def commit_file() -> None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.replace(temporary, destination)
            fsync_directory(config.upload_root)

        await run_in_threadpool(commit_file)
        committed = True
        return StoredUpload(destination, upload.digest.hexdigest(), upload.size, upload.filename)
    finally:
        if not handle.closed:
            await run_in_threadpool(handle.close)
        if not committed:
            await run_in_threadpool(temporary.unlink, missing_ok=True)
