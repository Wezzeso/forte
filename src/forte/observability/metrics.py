"""Bounded aggregate metrics; no request IDs, paths, text, or media labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock
import re
import math


def safe_route(path: str) -> str:
    if path == "/api/v1/analyze":
        return "analyze"
    if re.fullmatch(r"/api/v1/jobs/[^/]+", path):
        return "job"
    return {"/healthz": "health", "/readyz": "readiness", "/metrics": "metrics"}.get(path, "other")


class AggregateMetrics:
    """Constant-memory counters and sum/max timings suitable for local telemetry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counts: Counter = Counter()
        self._sums: dict = defaultdict(float)
        self._maxima: dict = defaultdict(float)
        self._failures: Counter = Counter()

    def observe(self, *, route: str, status: int, duration_ms: float,
                acceptance_ms: float | None = None, failed: bool = False) -> None:
        if route not in {"analyze", "job", "health", "readiness", "metrics", "other"}:
            raise ValueError("Unknown metric route")
        status_class = min(max(status // 100, 1), 5)
        if duration_ms < 0 or (acceptance_ms is not None and acceptance_ms < 0):
            raise ValueError("Negative duration")
        with self._lock:
            self._counts[(route, status_class)] += 1
            self._failures[route] += int(failed)
            for name, value in (("request", duration_ms), ("acceptance", acceptance_ms)):
                if value is not None:
                    self._counts[(route, name)] += 1
                    self._sums[(route, name)] += value
                    self._maxima[(route, name)] = max(self._maxima[(route, name)], value)

    def snapshot(self) -> dict:
        with self._lock:
            return {"failures": [{"route": route, "count": count} for route, count in sorted(self._failures.items()) if count], "requests": [
                {"route": key[0], "status_class": key[1], "count": count}
                for key, count in sorted(self._counts.items(), key=lambda item: str(item[0]))
                if isinstance(key[1], int)
            ], "durations": [
                {"route": key[0], "measurement": key[1], "count": self._counts[key],
                 "sum_ms": value, "max_ms": self._maxima[key]}
                for key, value in sorted(self._sums.items())
            ]}


def discover_vllm_metric_names(payload: str) -> list[str]:
    """Read actual exposed names, discarding all labels and values.

    Metrics differ between vLLM versions. Discovery is evidence of names only,
    never evidence of model readiness or latency. No assumed name is required.
    """
    names = set()
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"(vllm:[A-Za-z0-9_:]+)(?:\{|\s)", line)
        if match:
            names.add(match.group(1))
    return sorted(names)


VLLM_AGGREGATE_METRICS = frozenset(
    {f"vllm:{family}_{suffix}" for family in (
        "request_prefill_time_seconds", "request_decode_time_seconds",
        "request_inference_time_seconds", "request_queue_time_seconds",
        "time_to_first_token_seconds", "e2e_request_latency_seconds",
    ) for suffix in ("sum", "count")}
    | {"vllm:prompt_tokens_total", "vllm:generation_tokens_total", "vllm:request_success_total",
       "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
       "process_start_time_seconds"}
)


def parse_vllm_metric_totals(payload: str) -> dict:
    """Sum only verified numeric metric families, discarding every label."""
    totals = {}
    for line in payload.splitlines():
        match = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{.*\})?\s+(\S+)", line)
        if match is None or match.group(1) not in VLLM_AGGREGATE_METRICS:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        item = totals.setdefault(match.group(1), {"series_count": 0, "sum_value": 0.0})
        item["series_count"] += 1
        item["sum_value"] += value
    return totals

