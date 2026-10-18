"""CPU-only authenticated upload/polling service.

Run with ``uvicorn forte.api.main:app``. All inference, media probing and JPEG
assembly run in the independent worker. A 202 is emitted only after the upload
has been fsynced and the durable job transaction has completed.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from forte.observability import TimingMiddleware, mark_durable_acceptance
from forte.shared.schemas import AnalyzeResponse, JobResponse

from .config import Settings
from .metadata import completed_ui_metadata
from .reviews import ReviewConflictHTTP, create_review_router, review_conflict_response
from .security import ApiBoundaryMiddleware, ScopedCorsMiddleware
from .uploads import fsync_directory, persist_upload

_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _value(record: Any, name: str, default: Any = None) -> Any:
    return record.get(name, default) if isinstance(record, dict) else getattr(record, name, default)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _date(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (TypeError, ValueError):
        return None


def _owned_file(value: Any, root: Path) -> Path:
    if not value:
        raise HTTPException(404, "Artifact is unavailable")
    try:
        path = Path(value).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise HTTPException(404, "Artifact is unavailable")
        return path
    except (OSError, ValueError) as exc:
        raise HTTPException(404, "Artifact is unavailable") from exc


def _worker_alive(config: Settings) -> bool:
    try:
        heartbeat = config.worker_heartbeat_file
        if heartbeat is None or heartbeat.stat().st_size > 4096:
            return False
        record = json.loads(heartbeat.read_text(encoding="utf-8"))
        if config.model_native_launch_config_path is not None:
            from forte.native_dispatch import native_worker_binding
            if record.get("native_consumer") != native_worker_binding(config):
                return False
        updated = _date(record["updated_at"])
        if updated is None:
            return False
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        return -5 <= age <= config.worker_heartbeat_max_age_seconds
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError, RecursionError):
        return False


async def _model_alive(config: Settings) -> bool:
    """Require live engine health and a matching, verified prewarm attestation.

    Expected pins come from the same local manifest used by the worker. Engine
    liveness alone does not establish that model files or warmup were verified.
    """
    if os.environ.get("FORTE_ANALYZER", "").strip().lower() == "fast":
        # Fast analyzer: the local vLLM server must answer its model listing.
        base = (os.environ.get("FORTE_FAST_MODEL_BASE_URL") or config.model_base_url).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=config.readiness_timeout_seconds, trust_env=False,
                                         follow_redirects=False) as client:
                response = await client.get(base + "/models")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
    if config.model_native_launch_config_path is not None:
        from forte.native_dispatch import native_model_alive
        return await native_model_alive(config)

    def expected_pins() -> dict[str, Any] | None:
        try:
            if config.model_candidate_launch_config_path is not None:
                from serving.qwen38_identity import CandidateError, load_expected_runtime_identity
                try:
                    return {"model_loaded": True, "prewarm_completed": True,
                            **load_expected_runtime_identity(config.model_candidate_launch_config_path)}
                except (CandidateError, OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
                    return None
            if config.model_manifest_path.stat().st_size > 1024 * 1024:
                return None
            manifest = json.loads(config.model_manifest_path.read_text(encoding="utf-8"))
            revision = manifest["revision"]
            version = manifest["serving"]["version"]
            if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
                return None
            if not isinstance(version, str) or not version.strip():
                return None
            return {"model_revision": revision, "vllm_version": version}
        except (OSError, ValueError, KeyError, TypeError, ImportError, RecursionError):
            return None

    pins = await run_in_threadpool(expected_pins)
    if pins is None:
        return False
    base = config.model_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        from forte.model.adapter import InvalidModelOutput, strict_json_object, validate_private_endpoint
        validate_private_endpoint(config.model_base_url, ("model", "model-1", "model-2", "model-3"))
        headers = {"Authorization": "Bearer " + config.model_api_key} if config.model_api_key else {}
        async with httpx.AsyncClient(timeout=config.readiness_timeout_seconds, trust_env=False,
                                     follow_redirects=False, headers=headers) as client:
            engine_health, attestation_response = await asyncio.gather(
                client.get(base + "/health"),
                client.get(base + "/readyz"),
            )
        if engine_health.status_code != 200 or attestation_response.status_code != 200:
            return False
        if len(attestation_response.content) > 16 * 1024:
            return False
        attestation = strict_json_object(attestation_response.content)
        return (
            isinstance(attestation, dict)
            and attestation.get("ready") is True
            and attestation.get("model_files_verified") is True
            and all(type(attestation.get(key)) is type(value) and attestation.get(key) == value
                    for key, value in pins.items())
            and (config.model_candidate_launch_config_path is None or (
                type(attestation.get("adapter_loaded_tensor_count")) is int
                and attestation["adapter_loaded_tensor_count"] > 0))
        )
    except (httpx.HTTPError, InvalidModelOutput, ValueError, RecursionError):
        return False


def create_app(
    config: Settings | None = None,
    store: Any | None = None,
    readiness: Callable[..., Any] | None = None,
) -> FastAPI:
    """Build an app with injectable durable storage and readiness for tests.

    ``readiness`` is an optional zero-argument callable returning a boolean or
    a mapping of component names to booleans; it may be asynchronous. Production
    defaults check the database, storage, private model health and live worker.
    """
    settings = config or Settings.from_env()
    initialization_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            await run_in_threadpool(settings.upload_root.mkdir, parents=True, exist_ok=True)
            await run_in_threadpool(settings.result_root.mkdir, parents=True, exist_ok=True)
            await run_in_threadpool(fsync_directory, settings.data_dir)
            await ensure_store(application)
        except Exception:
            # Health remains accessible during dependency outages. Never echo a
            # database connection URL or exception containing credentials.
            application.state.initialization_failed = True
        try:
            yield
        finally:
            if store is None and application.state.store is not None:
                engine = getattr(application.state.store, "engine", None)
                if engine is not None:
                    await run_in_threadpool(engine.dispose)

    application = FastAPI(
        title="Forte visual review API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.state.config = settings
    application.state.store = store
    application.state.initialization_failed = False
    application.add_exception_handler(ReviewConflictHTTP, review_conflict_response)

    async def ensure_store(target: FastAPI) -> Any:
        if target.state.store is not None:
            return target.state.store
        async with initialization_lock:
            if target.state.store is None:
                from forte.jobs import JobStore

                target.state.store = await run_in_threadpool(
                    JobStore,
                    settings.database_url,
                    settings.result_root,
                    job_timeout_seconds=settings.job_timeout_seconds,
                )
        return target.state.store

    async def repository() -> Any:
        try:
            return await ensure_store(application)
        except Exception as exc:
            raise HTTPException(503, "Job service is unavailable") from exc

    async def find_job(job_id: str) -> tuple[Any, Any]:
        if not _JOB_ID.fullmatch(job_id):
            raise HTTPException(404, "Job not found")
        job_store = await repository()
        try:
            record = await run_in_threadpool(job_store.get_job, job_id)
        except Exception as exc:
            raise HTTPException(503, "Job service is unavailable") from exc
        if record is None:
            raise HTTPException(404, "Job not found")
        return job_store, record

    @application.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/runtime-config.json", include_in_schema=False)
    async def runtime_config() -> JSONResponse:
        # Operator-selected public origin only. API keys remain runtime secrets
        # entered by the reviewer and are never included in this bootstrap.
        return JSONResponse({"apiBaseUrl": settings.public_api_base_url}, headers={"Cache-Control": "no-store"})

    @application.get("/readyz", include_in_schema=False)
    async def ready() -> JSONResponse:
        if settings.admission_paused:
            # Maintenance is explicit; do not contact the suspended model or
            # claim anything about unqueried dependency health.
            return JSONResponse({"status": "not_ready", "components": {"admission": False}},
                status_code=503, headers={"Cache-Control": "no-store", "Retry-After": "60"})
        if readiness is not None:
            try:
                checks = readiness()
                if inspect.isawaitable(checks):
                    checks = await checks
                components = {str(key): bool(value) for key, value in checks.items()} if isinstance(checks, dict) else {"service": bool(checks)}
            except Exception:
                components = {"service": False}
        else:
            async def database_alive() -> bool:
                try:
                    job_store = await repository()
                    return bool(await run_in_threadpool(job_store.ping))
                except Exception:
                    return False

            db_ok, model_ok = await asyncio.gather(database_alive(), _model_alive(settings))
            components = {
                "authentication": bool(settings.api_keys),
                "database": db_ok,
                "storage": settings.upload_root.is_dir() and os.access(settings.upload_root, os.W_OK),
                "model": model_ok,
                "worker": await run_in_threadpool(_worker_alive, settings),
            }
        healthy = bool(components) and all(components.values())
        return JSONResponse({"status": "ready" if healthy else "not_ready", "components": components}, status_code=200 if healthy else 503, headers={"Cache-Control": "no-store"})

    @application.post("/api/v1/analyze", response_model=AnalyzeResponse, status_code=202)
    async def analyze(request: Request) -> AnalyzeResponse:
        # The upload consumes the same conservative application budget as
        # queueing/inference. The client benchmark measures the earlier true T0.
        deadline_at = datetime.now(timezone.utc) + timedelta(seconds=settings.job_timeout_seconds)
        job_store = await repository()
        job_id = uuid.uuid4().hex
        try:
            upload = await persist_upload(request, settings, job_id)
        except OSError as exc:
            raise HTTPException(503, "Upload storage is unavailable") from exc
        if datetime.now(timezone.utc) >= deadline_at:
            await run_in_threadpool(upload.path.unlink, missing_ok=True)
            await run_in_threadpool(fsync_directory, settings.upload_root)
            raise HTTPException(408, "Upload exceeded the processing deadline")
        try:
            await run_in_threadpool(
                job_store.create_job,
                job_id,
                str(upload.path),
                upload.sha256,
                upload.size,
                upload.original_filename,
                deadline_at=deadline_at,
            )
        except Exception as exc:
            # A timeout can occur after a transaction commits. Never remove a
            # potentially accepted input unless absence is positively verified.
            try:
                existing = await run_in_threadpool(job_store.get_job, job_id)
                if existing is None:
                    await run_in_threadpool(upload.path.unlink, missing_ok=True)
                    await run_in_threadpool(fsync_directory, settings.upload_root)
            except Exception:
                pass
            raise HTTPException(503, "Job could not be durably accepted") from exc
        mark_durable_acceptance(request.scope)
        return AnalyzeResponse(job_id=job_id)

    @application.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str) -> Any:
        job_store, record = await find_job(job_id)
        status = _value(record, "status", _value(record, "public_status", "processing"))
        if status == "done":
            result_path = _value(record, "result_path")
            if result_path:
                try:
                    path = _owned_file(result_path, settings.result_root)
                except HTTPException as exc:
                    raise HTTPException(503, "Completed result is unavailable") from exc
                return FileResponse(path, media_type="application/json")
            try:
                payload = await run_in_threadpool(job_store.get_result_bytes, job_id)
            except Exception as exc:
                raise HTTPException(503, "Completed result is unavailable") from exc
            if payload is None:
                raise HTTPException(503, "Completed result is unavailable")
            return Response(payload, media_type="application/json")
        if status == "failed":
            return {"job_id": job_id, "status": "failed"}
        return {"job_id": job_id, "status": "processing"}

    @application.get("/api/ui/config", include_in_schema=False)
    async def get_ui_config() -> dict[str, Any]:
        # ApiBoundaryMiddleware protects this route and sets no-store. Report
        # configured input limits only, never guesses about codec or duration.
        return {"max_upload_bytes": settings.max_upload_bytes, "accepted_suffixes": [".mov"]}

    @application.get("/api/ui/jobs/{job_id}", include_in_schema=False)
    async def get_ui_job(job_id: str) -> dict[str, Any]:
        _, record = await find_job(job_id)
        status = _value(record, "status", _value(record, "public_status", "processing"))
        accepted = _date(_value(record, "accepted_at", _value(record, "created_at")))
        completed = _date(_value(record, "completed_at"))
        elapsed = max(0.0, ((completed or datetime.now(timezone.utc)) - accepted).total_seconds()) if accepted else None
        enrichment = await run_in_threadpool(completed_ui_metadata, settings, record)
        return {
            "job_id": job_id,
            "status": status,
            "stage": _value(record, "stage", "queued"),
            "created_at": _iso(_value(record, "created_at", _value(record, "accepted_at"))),
            "accepted_at": _iso(_value(record, "accepted_at")),
            "updated_at": _iso(_value(record, "updated_at")),
            "deadline_at": _iso(_value(record, "deadline_at")),
            "completed_at": _iso(_value(record, "completed_at")),
            "elapsed_sec": elapsed,
            **enrichment,
            "result": None,
            "error_code": "analysis_failed" if status == "failed" else None,
        }

    @application.get("/api/ui/jobs/{job_id}/video", include_in_schema=False)
    async def original_video(job_id: str) -> FileResponse:
        _, record = await find_job(job_id)
        path = _owned_file(_value(record, "input_path"), settings.upload_root)
        return FileResponse(path, media_type="video/quicktime", filename="source.mov", content_disposition_type="inline")

    application.include_router(create_review_router(find_job))

    # API routes win over static files; missing API routes never fall through to
    # an SPA catch-all. Assets and scripts are packaged locally by the UI build.
    if (settings.web_dist / "index.html").is_file():
        application.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="web")

    application.add_middleware(ApiBoundaryMiddleware, config=settings)
    application.add_middleware(ScopedCorsMiddleware, config=settings)
    application.add_middleware(TimingMiddleware)
    return application


app = create_app()
