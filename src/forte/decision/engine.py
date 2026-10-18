"""Small orchestration adapter; inference remains in the private model service."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Collection, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol

from forte.shared.schemas import ClipRequest
from forte.observability import aggregate_inference_metrics, sanitize_model_metrics

from .aliasing import compact_frame_request, restore_frame_ids
from .structured import (
    DECISION_OUTPUT_SUBCODES,
    DecisionBatch, DecisionConfig, DecisionConfigurationError, DecisionOutputError,
    build_prompt, validate_decisions,
)


class DecisionProvider(Protocol):
    async def infer(
        self, clip: ClipRequest, *, prompt: str, response_schema: dict[str, Any],
        deadline: float, cancel_event: asyncio.Event | None = None,
    ) -> Any:
        """Return a completed ModelResponse(data, metrics) or raise an error."""
        ...


class DecisionEngine:
    def __init__(
        self, provider: DecisionProvider, *, reviewed_event_ids: Collection[str],
        event_definitions: Mapping[str, Mapping[str, Any]] | None = None,
        config: DecisionConfig | None = None,
    ) -> None:
        self.provider = provider
        self.reviewed_event_ids = tuple(reviewed_event_ids)
        self.event_definitions = event_definitions
        self.config = config or DecisionConfig()

    async def analyze(
        self, request: ClipRequest, *, deadline: float,
        cancel_event: asyncio.Event | None = None,
    ) -> DecisionBatch:
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise DecisionConfigurationError("invalid_monotonic_deadline")
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        if deadline <= asyncio.get_running_loop().time():
            raise TimeoutError("Local visual decision deadline exceeded")
        aliases = compact_frame_request(request) if self.config.compact_frame_ids else None
        inference_request = aliases.request if aliases is not None else request
        bundle = build_prompt(
            inference_request, reviewed_event_ids=self.reviewed_event_ids,
            event_definitions=self.event_definitions, config=self.config,
        )
        async with asyncio.timeout_at(deadline):
            response = await self.provider.infer(
                inference_request, prompt=bundle.prompt, response_schema=bundle.response_schema,
                deadline=deadline, cancel_event=cancel_event,
            )
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        if deadline <= asyncio.get_running_loop().time():
            raise TimeoutError("Local visual decision deadline exceeded")
        metrics = getattr(response, "metrics", {})
        if is_dataclass(metrics) and not isinstance(metrics, type):
            metrics = asdict(metrics)
        elif hasattr(metrics, "model_dump"):
            metrics = metrics.model_dump()
        if not isinstance(metrics, Mapping):
            metrics = {}
        try:
            batch = validate_decisions(
                response.data, inference_request, reviewed_event_ids=self.reviewed_event_ids,
                config=self.config, prompt_hash=bundle.prompt_hash, metrics=metrics,
                finish_reason=getattr(response, "finish_reason", "stop"),
            )
            return restore_frame_ids(batch, aliases) if aliases is not None else batch
        except DecisionOutputError as error:
            # The provider completed, but its structured observations could not
            # be accepted. Retain only fixed numeric telemetry and a whitelisted
            # diagnostic; never attach the response, prompt, IDs or observations.
            error.forte_decision_subcode = (
                error.code if error.code in DECISION_OUTPUT_SUBCODES
                else "unclassified_decision_output"
            )
            error.forte_model_metrics = sanitize_model_metrics(aggregate_inference_metrics([metrics]))
            raise
