"""Real private-model smoke with generated synthetic media and A08 validation.

This command cannot accept an existing media path. It prints aggregate timings
and validation counts, never prompts, model observations, or image payloads.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from forte.decision import build_prompt, validate_decisions
from forte.shared.schemas import ClipRequest, FactDefinition, FrameRef

from .adapter import InvalidModelOutput, LocalModelConfig, LocalModelProvider


async def synthetic_smoke(*, base_url: str, media_root: str | Path,
                          timeout_sec: float = 120, api_key: str | None = None,
                          require_readiness: bool = True,
                          model: str | None = None, expected_revision: str | None = None,
                          seed: int | None = None, max_tokens: int | None = None,
                          chat_template_kwargs: Mapping[str, bool] | None = None,
                          expected_runtime_identity: Mapping[str, str] | None = None) -> dict:
    if not 0 < timeout_sec <= 600:
        raise ValueError("smoke_timeout_invalid")
    root = Path(media_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix="synthetic_smoke_") as temporary:
        owned = Path(temporary)
        frames = []
        for index in range(3):
            path = owned / f"frame_{index}.jpg"
            image = Image.new("RGB", (448, 448), (210, 210, 210))
            draw = ImageDraw.Draw(image)
            draw.rectangle((60 + index*10, 100, 340 + index*10, 330), fill=(220, 20, 20))
            image.save(path, "JPEG", quality=95)
            image.close()
            frames.append(FrameRef(frame_id=f"f_synthetic_{index}", pts_sec=index*0.5, path=path,
                width=448, height=448, job_id=owned.name,
                jpeg_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        clip = ClipRequest(job_id=owned.name, clip_id="c_synthetic", frame_refs=frames,
                           fact_definitions=[FactDefinition(fact_id="synthetic_marker_visible",
                               description="A red rectangular marker is visible in the frame.")])
        event_ids = ("synthetic_marker_event",)
        bundle = build_prompt(clip, reviewed_event_ids=event_ids, event_definitions={
            "synthetic_marker_event": {"definition_ru": "Synthetic test event: a clearly visible red rectangle.",
                "required_fact_states": {"synthetic_marker_visible": "present"}}})
        model_options = {key: value for key, value in {
            "model": model, "expected_revision": expected_revision, "seed": seed, "max_tokens": max_tokens,
        }.items() if value is not None}
        async with LocalModelProvider(LocalModelConfig(media_root=root, base_url=base_url,
            api_key=api_key, require_readiness=require_readiness,
            chat_template_kwargs=tuple((chat_template_kwargs or {}).items()),
            expected_runtime_identity=tuple((expected_runtime_identity or {}).items()),
            **model_options)) as provider:
            response = await provider.infer(clip, prompt=bundle.prompt,
                response_schema=bundle.response_schema, deadline=time.monotonic()+timeout_sec)
        batch = validate_decisions(response.data, clip, reviewed_event_ids=event_ids,
                                   finish_reason=response.finish_reason)
        known_frames = {frame.frame_id for frame in frames}
        if (len(batch.facts) != 1 or batch.facts[0].state != "present"
                or not any(candidate.event_type_id == "synthetic_marker_event"
                    and candidate.supporting_frame_ids
                    and set(candidate.supporting_frame_ids).issubset(known_frames)
                    and any(fact.fact_id == "synthetic_marker_visible" and fact.state == "present"
                            for fact in candidate.supporting_facts)
                    for candidate in batch.candidates)):
            raise InvalidModelOutput("synthetic_marker_visual_sanity_failed")
        return {"synthetic_only": True, "real_local_inference": True,
                "schema_and_semantic_validation_passed": True,
                "synthetic_visual_sanity_passed": True,
                "facts_validated": len(batch.facts), "candidates_validated": len(batch.candidates),
                "metrics": asdict(response.metrics), "model": response.model,
                "event_accuracy_measured": False}
