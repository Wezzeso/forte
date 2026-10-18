"""Monotonic local timings. Never compare these clock values across processes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Callable, Iterator

Clock = Callable[[], int]
STAGES = frozenset({
    "queue", "probe_decode", "frame_preparation", "coarse_model", "prefill",
    "inference", "refinement", "rules", "evidence", "publication", "analysis",
})
BOUNDARIES = frozenset({"T0", "T1", "T2", "T3", "T4", "T5", "request_start",
                        "response_start", "response_end", "lookup_start"})


@dataclass
class StageTimer:
    """Accumulate fixed-name stages without storing video facts or identifiers.

    A timer belongs to one process and one operation. Serialize durations only.
    A queue delay spanning processes must instead use synchronized durable UTC
    timestamps and explicitly report the synchronization uncertainty.
    """

    clock: Clock = field(default=perf_counter_ns, repr=False)
    boundaries_ns: dict[str, int] = field(default_factory=dict, repr=False)
    stages_ns: dict[str, int] = field(default_factory=dict, repr=False)

    def mark(self, name: str) -> int:
        if name not in BOUNDARIES:
            raise ValueError("Unknown timing boundary")
        if name in self.boundaries_ns:
            raise ValueError("Timing boundary already marked")
        value = self.clock()
        if self.boundaries_ns and value < max(self.boundaries_ns.values()):
            raise ValueError("Monotonic clock moved backwards")
        self.boundaries_ns[name] = value
        return value

    def elapsed_ms(self, start: str, end: str) -> float | None:
        if start not in self.boundaries_ns or end not in self.boundaries_ns:
            return None
        value = self.boundaries_ns[end] - self.boundaries_ns[start]
        if value < 0:
            raise ValueError("Timing boundaries out of order")
        return value / 1_000_000

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if name not in STAGES:
            raise ValueError("Unknown stage")
        start = self.clock()
        try:
            yield
        finally:
            elapsed = self.clock() - start
            if elapsed < 0:
                raise ValueError("Monotonic clock moved backwards")
            self.stages_ns[name] = self.stages_ns.get(name, 0) + elapsed

    def snapshot(self) -> dict:
        return {
            "clock": "process_monotonic",
            "stages_ms": {key: value / 1_000_000 for key, value in sorted(self.stages_ns.items())},
            "boundaries_measured": sorted(self.boundaries_ns),
            "acceptance_after_body_ms": self.elapsed_ms("T1", "T2"),
            "analysis_to_publication_ms": self.elapsed_ms("T3", "T4"),
        }


def mark_durable_acceptance(scope: dict) -> None:
    """Call only after both the owned upload and job record are durable."""
    trace = scope.get("forte.timing")
    if isinstance(trace, StageTimer):
        trace.mark("T2")

