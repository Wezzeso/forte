"""Worker integration: run the fast analyzer on a leased job and build the public result."""

from __future__ import annotations

import asyncio
import base64
import io
import hashlib
import itertools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from forte.fast import catalog as C
from forte.fast.analyzer import DEFAULT_CONFIG, Analysis, analyze
from forte.fast.video import DecodedVideo, decode
from forte.fast.vlm import VLMClient

StageCallback = Callable[[str], Awaitable[None]]


async def _await_owned_thread(task: asyncio.Task, cancel_event: threading.Event):
    """Retain ownership until native code stops, including repeated cancellation."""
    try:
        return await asyncio.shield(task)
    except BaseException:
        cancel_event.set()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()  # Retrieve a possible thread failure; preserve cancellation.
        raise

DISCLAIMER = "Результат является подсказкой для проверки человеком, а не юридическим заключением."
FAST_AUDIT_NAME = "analysis.fast.json"


def _jpeg_b64(image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def describe(type_id: str, observation: str) -> str:
    vt = C.BY_ID[type_id]
    observation = " ".join((observation or "").split())
    if observation and not observation.endswith((".", "!", "?")):
        observation += "."
    seen = observation or "На кадрах зафиксировано действие, соответствующее признаку нарушения."
    return (f"{seen} Это может указывать на нарушение: {vt.name_ru.lower()}. "
            f"Норма: {vt.document}, {vt.clause} — «{vt.rule_ru}». {DISCLAIMER}")


def evidence_frames(video: DecodedVideo, t: float) -> list[dict[str, Any]]:
    picks, seen = [], set()
    for offset in (0.0, -1.0, 1.0):
        frame = video.nearest(min(max(t + offset, 0.0), video.duration))
        if frame.t in seen:
            continue
        seen.add(frame.t)
        picks.append(frame)
    picks.sort(key=lambda f: f.t)
    return [{"timestamp_sec": round(min(f.t, video.duration), 3), "image_base64": _jpeg_b64(f.image)}
            for f in picks[:3]]


def build_result(job_id: str, video: DecodedVideo, analysis: Analysis) -> dict[str, Any]:
    violations = []
    for event in sorted(analysis.events, key=lambda e: e.t):
        t = round(min(max(event.t, 0.0), video.duration), 2)
        violations.append({
            "type": C.BY_ID[event.type_id].name_ru,
            "timecode_sec": float(t),
            "description": describe(event.type_id, event.observation_ru),
            "evidence_frames": evidence_frames(video, t),
        })
    return {"job_id": job_id, "status": "done",
            "verdict": "violation" if violations else "no_violation",
            "violations": violations}


def load_config(data_dir: Path) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    override = data_dir / "fast_config.json"
    if override.exists():
        if not override.is_file() or override.stat().st_size >= 65536:
            raise ValueError("fast_config_invalid")
        data = json.loads(override.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("fast_config_invalid")
        cfg.update(data)
    return cfg


def decode_for_analysis(path: str, cfg: dict[str, Any], runtime: Any = None, *,
                        deadline: float | None = None, cancel_event: threading.Event | None = None):
    """Process native frames once; retain dense original samples for verification."""
    fps = float(cfg.get("decode_fps", 5.0 if runtime is not None else 2.0))
    if runtime is not None and fps < 5.0:
        raise ValueError("specialist_verification_requires_dense_frames")
    kwargs = {"max_frames": int(cfg.get("decode_max_frames", 1800 if runtime is not None else 600)),
              "max_memory_bytes": int(cfg.get("decode_max_memory_bytes", 2 * 1024 ** 3)),
              "deadline": deadline, "cancel_event": cancel_event}
    if runtime is None:
        return decode(path, fps, **kwargs), None
    try:
        video = decode(path, fps, on_frame=runtime.process_frame, on_start=runtime.start_video, **kwargs)
        observations = runtime.finish_video()
        return video, observations
    finally:
        runtime.abort_video()


class FastPipeline:
    """Drop-in replacement for AnalysisPipeline used by forte.worker."""

    rules = tuple(C.TYPES)

    def __init__(self, settings: Any):
        from forte.pipeline import PipelineResult  # local import keeps module light for tools

        self._result_cls = PipelineResult
        self.settings = settings
        base = os.environ.get("FORTE_FAST_MODEL_BASE_URL") or settings.model_base_url
        model = os.environ.get("FORTE_FAST_MODEL_NAME") or settings.model_name
        self.provider = VLMClient(base, model, getattr(settings, "model_api_key", None),
                                  concurrency=int(os.environ.get("FORTE_FAST_CONCURRENCY", "48")))
        self._specialist_lock = asyncio.Lock()
        self._specialist_runtime = None
        self._specialist_identity = None
        self._job_seq = itertools.count()  # claim order; used only when job_priority is on

    def _runtime(self, cfg: dict[str, Any]):
        from forte.fast.specialists import SpecialistRuntime

        spec = cfg.get("specialists", {"enabled": False})
        if not isinstance(spec, dict):
            raise ValueError("fast_specialist_config_invalid")
        identity = json.dumps(spec, sort_keys=True, allow_nan=False)
        if spec.get("enabled"):
            for key in ("checkpoint", "gate_path"):
                stat = Path(spec[key]).stat()
                identity += f"|{stat.st_size}:{stat.st_mtime_ns}"
        if identity != self._specialist_identity:
            self._specialist_runtime = SpecialistRuntime.from_config(spec)
            self._specialist_identity = identity
        return self._specialist_runtime

    async def aclose(self) -> None:
        await self.provider.aclose()

    async def analyze(self, job: Any, *, stage: StageCallback, deadline: float,
                      cancel_event: threading.Event):
        from forte.shared.schemas import CompletedResponse

        timings: dict[str, Any] = {}
        priority = next(self._job_seq)
        root = self.settings.work_dir / job.job_id / f"attempt-{job.attempt}"
        root.mkdir(parents=True, exist_ok=True)
        await stage("decoding")
        t0 = time.monotonic()
        cfg = load_config(self.settings.data_dir)
        remaining = deadline - time.monotonic() - 10.0
        if remaining <= 0:
            raise TimeoutError("analysis_deadline_exceeded")
        # One detector instance owns its per-video trackers. The lock queues
        # native-frame perception; VLM scene/verification calls remain batched.
        async with asyncio.timeout(remaining):
            async with self._specialist_lock:
                loading = asyncio.create_task(asyncio.to_thread(self._runtime, cfg))
                runtime = await _await_owned_thread(loading, cancel_event)
                decoding = asyncio.create_task(asyncio.to_thread(
                    decode_for_analysis, str(job.owned_path), cfg, runtime,
                    deadline=deadline - 10.0, cancel_event=cancel_event))
                video, specialist_analysis = await _await_owned_thread(decoding, cancel_event)
        timings["decode_s"] = round(time.monotonic() - t0, 3)
        if cancel_event.is_set():
            raise RuntimeError("analysis_cancelled")
        await stage("screening")
        remaining = deadline - time.monotonic() - 10.0
        if remaining <= 0:
            raise TimeoutError("analysis_deadline_exceeded")
        analysis = await asyncio.wait_for(analyze(self.provider, video, cfg,
                                                  specialist_analysis=specialist_analysis, priority=priority),
                                          timeout=remaining)
        timings.update({k: round(v, 3) for k, v in analysis.timings.items()})
        await stage("assembling")
        result = build_result(job.job_id, video, analysis)
        validated = CompletedResponse.model_validate(result)
        payload = json.dumps(validated.model_dump(mode="json"), ensure_ascii=False,
                             allow_nan=False, separators=(",", ":")).encode("utf-8")
        audit = {
            "job_id": job.job_id, "attempt": job.attempt, "duration": video.duration,
            "events": [{"type_id": e.type_id, "t": e.t, "confidence": e.confidence, "sources": e.sources}
                       for e in sorted(analysis.events, key=lambda e: e.t)],
            "timings": timings,
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "decisions": analysis.audit,
            "verifications": analysis.verifications,
        }
        audit_path = root / FAST_AUDIT_NAME
        audit_path.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
        return self._result_cls(payload, {"stages_ms": {k: v * 1000 for k, v in timings.items()
                                                        if isinstance(v, (int, float))}}, audit_path)


def ui_metadata(data_dir: Path, job_id: str, attempt: int) -> dict[str, Any] | None:
    """UI enrichment (rule references) for a job analysed by FastPipeline."""
    path = data_dir / "work" / job_id / f"attempt-{attempt}" / FAST_AUDIT_NAME
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if audit.get("job_id") != job_id:
        return None
    events = []
    for e in audit.get("events", []):
        vt = C.BY_ID.get(e.get("type_id"))
        if vt is None:
            return None
        events.append({"rule": {"title": vt.document, "clause": vt.clause, "url": vt.url},
                       "observed_facts": [vt.name_ru], "limitations": []})
    limits = [DISCLAIMER,
              "Анализ строго визуальный: звук, данные учётных систем и проводок не используются.",
              "Лица не распознаются и не идентифицируются."]
    return {"limitations": limits, "events": events}
