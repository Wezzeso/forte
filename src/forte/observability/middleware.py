"""Pure ASGI timing: truthful application boundaries, not proxy/client clocks."""

from __future__ import annotations

from time import perf_counter_ns
from typing import Callable

from .metrics import AggregateMetrics, safe_route
from .timing import StageTimer


class TimingMiddleware:
    def __init__(self, app, metrics: AggregateMetrics | None = None,
                 clock: Callable[[], int] = perf_counter_ns):
        self.app = app
        self.metrics = metrics if metrics is not None else AggregateMetrics()
        self.clock = clock

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        trace = StageTimer(clock=self.clock)
        trace.mark("request_start")
        scope["forte.timing"] = trace
        status = 500
        raised = False
        response_complete = False

        async def timed_receive():
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                if "T1" not in trace.boundaries_ns:
                    trace.mark("T1")
            return message

        async def timed_send(message):
            nonlocal status, response_complete
            if message["type"] == "http.response.start":
                trace.mark("response_start")
                status = message["status"]
                headers = list(message.get("headers", []))
                values = []
                acceptance = trace.elapsed_ms("T1", "T2")
                if acceptance is not None:
                    values.append(f"acceptance;dur={acceptance:.3f}")
                if safe_route(scope.get("path", "")) == "job":
                    lookup = trace.elapsed_ms("request_start", "response_start")
                    values.append(f"lookup;dur={lookup:.3f}")
                if values:
                    headers.append((b"server-timing", ", ".join(values).encode("ascii")))
                headers.append((b"x-forte-timing-origin", b"asgi-local"))
                message = {**message, "headers": headers}
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                if "response_end" not in trace.boundaries_ns:
                    trace.mark("response_end")
                response_complete = True

        try:
            await self.app(scope, timed_receive, timed_send)
        except BaseException:
            raised = True
            raise
        finally:
            if "response_end" not in trace.boundaries_ns:
                trace.mark("response_end")
            self.metrics.observe(
                route=safe_route(scope.get("path", "")), status=status,
                duration_ms=trace.elapsed_ms("request_start", "response_end"),
                acceptance_ms=trace.elapsed_ms("T1", "T2"),
                failed=raised or not response_complete,
            )

