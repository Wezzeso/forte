"""Optional human operational review, separate from immutable model results.

The shared API key authorizes access. A reviewer's name is a self-reported
label, never a verified identity. Confirmations and rejections append revisions
to the durable review ledger without changing or rerunning model analysis.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool

MAX_REVIEW_BODY_BYTES = 32 * 1024
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$", strict=True)]


class ReviewConflictHTTP(Exception):
    def __init__(self, current_revision: int, current_result_sha256: str) -> None:
        self.current_revision = current_revision
        self.current_result_sha256 = current_result_sha256


async def review_conflict_response(_request: Request, exc: ReviewConflictHTTP) -> JSONResponse:
    return JSONResponse(status_code=409, content={
        "detail": "review_conflict",
        "current_revision": exc.current_revision,
        "current_result_sha256": exc.current_result_sha256,
    })


class _ReviewSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    result_sha256: Digest
    evidence_sha256: Digest
    decision: Literal["confirm", "reject"]
    claimed_reviewer_name: Annotated[str, Field(min_length=1, max_length=120)]
    comment: Annotated[str, Field(max_length=4000)] = ""
    expected_revision: Annotated[int, Field(ge=0, le=2**31 - 1)]

    @field_validator("claimed_reviewer_name")
    @classmethod
    def nonempty_claimed_name(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("A nonempty claimed reviewer name is required")
        return value

    @field_validator("comment")
    @classmethod
    def valid_comment(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            raise ValueError("Invalid review comment")
        return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member")
        result[key] = value
    return result


async def _read_submission(request: Request) -> _ReviewSubmission:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(415, "Review requests require application/json")
    lengths = request.headers.getlist("content-length")
    try:
        if len(lengths) > 1 or (lengths and int(lengths[0]) < 0):
            raise ValueError
        if lengths and int(lengths[0]) > MAX_REVIEW_BODY_BYTES:
            raise HTTPException(413, "Review request exceeds configured limit")
    except ValueError as exc:
        raise HTTPException(400, "Invalid Content-Length") from exc
    size = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REVIEW_BODY_BYTES:
            raise HTTPException(413, "Review request exceeds configured limit")
        chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"), object_pairs_hook=_unique_json_object)
        return _ReviewSubmission.model_validate(payload)
    except (ValueError, TypeError, RecursionError, ValidationError) as exc:
        # Validation never echoes reviewer text, submitted comments or bindings.
        raise HTTPException(422, "Invalid review payload") from exc


def _get(record: Any, key: str, default: Any = None) -> Any:
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def _review_record(record: Any) -> dict[str, Any]:
    updated = _get(record, "updated_at")
    return {
        "job_id": _get(record, "job_id"),
        "result_sha256": _get(record, "result_sha256"),
        "violation_index": _get(record, "violation_index"),
        "evidence_sha256": _get(record, "evidence_digest"),
        "state": _get(record, "state"),
        "revision": _get(record, "revision"),
        "claimed_reviewer_name": _get(record, "actor_claimed"),
        "comment": _get(record, "comment", ""),
        "reviewed_at": updated.isoformat() if hasattr(updated, "isoformat") else updated,
        "reviewer_identity_verified": False,
    }


async def _store_call(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    from forte.jobs import ReviewConflict, ReviewNotAllowed

    try:
        return await run_in_threadpool(method, *args, **kwargs)
    except ReviewConflict as exc:
        raise ReviewConflictHTTP(exc.current_revision, exc.current_result_sha256) from exc
    except ReviewNotAllowed as exc:
        raise HTTPException(409, "review_not_allowed") from exc
    except Exception as exc:
        raise HTTPException(503, "Review service is unavailable") from exc


def create_review_router(find_job: Callable[..., Any]) -> APIRouter:
    router = APIRouter(prefix="/api/ui/jobs", include_in_schema=False)

    async def completed_positive_store(job_id: str) -> tuple[Any, Any]:
        store, record = await find_job(job_id)
        digest = _get(record, "result_sha256")
        if _get(record, "status") != "done" or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise HTTPException(409, "review_not_allowed")
        return store, record

    @router.get("/{job_id}/reviews")
    async def reviews(job_id: str) -> dict[str, Any]:
        store, record = await completed_positive_store(job_id)
        result_sha256 = _get(record, "result_sha256")
        records = await _store_call(store.get_reviews, job_id, result_sha256)
        return {
            "job_id": job_id,
            "result_sha256": result_sha256,
            "source_sha256": _get(record, "input_sha256"),
            "reviewer_identity_verified": False,
            "violations": [{
                "violation_index": _get(review, "violation_index"),
                "evidence_sha256": _get(review, "evidence_digest"),
                "review": _review_record(review),
            } for review in records],
        }

    @router.post("/{job_id}/reviews/{violation_index}")
    async def submit_review(
        job_id: str, request: Request,
        violation_index: Annotated[int, Path(ge=0, le=255)],
    ) -> dict[str, Any]:
        store, _ = await completed_positive_store(job_id)
        submission = await _read_submission(request)
        review = await _store_call(
            store.append_review, job_id, submission.result_sha256, violation_index,
            state="confirmed" if submission.decision == "confirm" else "rejected",
            actor_claimed=submission.claimed_reviewer_name,
            comment=submission.comment,
            expected_revision=submission.expected_revision,
            evidence_digest=submission.evidence_sha256,
        )
        return _review_record(review)

    @router.get("/{job_id}/reviews/{violation_index}/history")
    async def review_history(
        job_id: str,
        violation_index: Annotated[int, Path(ge=0, le=255)],
        result_sha256: Annotated[str, Query(pattern=r"^[a-f0-9]{64}$")],
        after_revision: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
        limit: Annotated[int, Query(ge=1, le=50)] = 50,
    ) -> dict[str, Any]:
        store, _ = await completed_positive_store(job_id)
        records = await _store_call(
            store.get_review_history, job_id, result_sha256, violation_index,
            after_revision=after_revision, limit=limit + 1,
        )
        selected = records[:limit]
        return {
            "job_id": job_id,
            "result_sha256": result_sha256,
            "violation_index": violation_index,
            "entries": [_review_record(review) for review in selected],
            "next_after_revision": _get(selected[-1], "revision") if selected else after_revision,
            "has_more": len(records) > limit,
        }

    return router
