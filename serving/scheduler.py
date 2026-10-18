"""Bounded fair dispatch into a warm provider.

This queue contains short, recoverable clip requests, not durable video jobs.
A14 remains responsible for job leases, fencing, retries and result publication.
Deadlines use the event loop's monotonic clock, never Unix timestamps.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import itertools
import math
from typing import Any, Generic, Protocol, TypeVar

T = TypeVar("T")


class CancellationSignal(Protocol):
    """Nonblocking part shared by asyncio.Event and threading.Event."""

    def is_set(self) -> bool: ...


async def _wait_cancelled(signal: CancellationSignal) -> bool:
    if isinstance(signal, asyncio.Event):
        return await signal.wait()
    while not signal.is_set():
        await asyncio.sleep(0.02)
    return True


class QueueFull(RuntimeError):
    """The bounded transient queue has no capacity; work was not accepted."""


class SchedulerClosed(RuntimeError):
    """No new work is accepted after shutdown."""


@dataclass(slots=True)
class _Request(Generic[T]):
    ticket: int
    job_id: str
    run: Callable[[], Awaitable[T]]
    deadline: float
    future: asyncio.Future[T]
    task: asyncio.Task[T] | None = None


class FairScheduler:
    """Round robin across jobs, FIFO within each job, bounded concurrency.

    ``concurrency`` is the total number of in-flight clip calls across the
    configured replicas. It is a measured deployment setting, not GPU capacity
    discovery. vLLM performs continuous batching within each replica; this layer
    never waits to fill an artificial batch at the expense of a deadline.
    """

    def __init__(
        self,
        *,
        concurrency: int = 2,
        max_pending: int = 64,
        max_pending_per_job: int = 16,
        max_inflight_per_job: int = 1,
    ) -> None:
        for value in (concurrency, max_pending, max_pending_per_job, max_inflight_per_job):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("scheduler limits must be positive integers")
        self.concurrency = concurrency
        self.max_pending = max_pending
        self.max_pending_per_job = max_pending_per_job
        self.max_inflight_per_job = max_inflight_per_job
        self._queues: dict[str, deque[_Request[Any]]] = {}
        self._rotation: deque[str] = deque()
        self._running: Counter[str] = Counter()
        self._requests: dict[int, _Request[Any]] = {}
        self._tickets = itertools.count()
        self._condition = asyncio.Condition()
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False

    async def start(self) -> FairScheduler:
        async with self._condition:
            if self._closed:
                raise SchedulerClosed("scheduler is closed")
            if not self._workers:
                self._workers = [
                    asyncio.create_task(self._worker(), name=f"model-dispatch-{i}")
                    for i in range(self.concurrency)
                ]
        return self

    async def __aenter__(self) -> FairScheduler:
        return await self.start()

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    @property
    def pending(self) -> int:
        return sum(sum(not r.future.done() for r in queue) for queue in self._queues.values())

    @property
    def inflight(self) -> int:
        return sum(self._running.values())

    async def submit(
        self,
        job_id: str,
        run: Callable[[], Awaitable[T]],
        *,
        deadline: float,
        cancel_event: CancellationSignal | None = None,
    ) -> T:
        if not job_id:
            raise ValueError("job_id is required")
        loop = asyncio.get_running_loop()
        if not math.isfinite(deadline) or deadline <= loop.time():
            raise TimeoutError("clip deadline expired before enqueue")
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        await self.start()
        future: asyncio.Future[T] = loop.create_future()
        req = _Request(next(self._tickets), job_id, run, deadline, future)
        async with self._condition:
            if self._closed:
                raise SchedulerClosed("scheduler is closed")
            self._prune()
            queue = self._queues.get(job_id)
            job_pending = sum(not r.future.done() for r in queue) if queue else 0
            if self.pending >= self.max_pending or job_pending >= self.max_pending_per_job:
                raise QueueFull("clip queue capacity reached")
            if queue is None:
                queue = self._queues[job_id] = deque()
                self._rotation.append(job_id)
            queue.append(req)
            self._requests[req.ticket] = req
            self._condition.notify_all()
        cancellation: asyncio.Task[bool] | None = None
        try:
            async with asyncio.timeout_at(deadline):
                if cancel_event is None:
                    return await future
                cancellation = asyncio.create_task(_wait_cancelled(cancel_event))
                await asyncio.wait((future, cancellation), return_when=asyncio.FIRST_COMPLETED)
                if cancellation.done() and cancellation.result():
                    raise asyncio.CancelledError
                return await future
        except (asyncio.CancelledError, TimeoutError):
            future.cancel()
            if req.task is not None:
                req.task.cancel()
            async with self._condition:
                self._condition.notify_all()
            raise
        finally:
            if cancellation is not None:
                cancellation.cancel()
                await asyncio.gather(cancellation, return_exceptions=True)

    async def cancel_job(self, job_id: str) -> int:
        """Cancel queued and in-flight calls after a lease loss or job abort."""
        count = 0
        async with self._condition:
            for req in tuple(self._requests.values()):
                if req.job_id == job_id and not req.future.done():
                    count += 1
                    req.future.cancel()
                    if req.task is not None:
                        req.task.cancel()
            self._condition.notify_all()
        return count

    def _prune(self) -> None:
        """Remove cancelled entries anywhere in a queue, bounding retained RAM."""
        now = asyncio.get_running_loop().time()
        for job_id, queue in tuple(self._queues.items()):
            live = deque()
            for request in queue:
                if request.future.done() or request.deadline <= now:
                    self._requests.pop(request.ticket, None)
                    if not request.future.done():
                        request.future.set_exception(TimeoutError("clip deadline expired in queue"))
                else:
                    live.append(request)
            if live:
                self._queues[job_id] = live
            else:
                del self._queues[job_id]
        self._rotation = deque(job for job in self._rotation if job in self._queues)

    def _pick(self) -> _Request[Any] | None:
        self._prune()
        for _ in range(len(self._rotation)):
            job_id = self._rotation.popleft()
            queue = self._queues[job_id]
            self._rotation.append(job_id)
            if self._running[job_id] >= self.max_inflight_per_job:
                continue
            req = queue.popleft()
            if not queue:
                del self._queues[job_id]
                self._rotation.pop()
            self._running[job_id] += 1
            return req
        return None

    async def _worker(self) -> None:
        while True:
            async with self._condition:
                req = self._pick()
                while req is None and not self._closed:
                    await self._condition.wait()
                    req = self._pick()
                if req is None:
                    return
            try:
                # submit cancellation can run between selection and execution.
                if req.future.done():
                    continue
                req.task = asyncio.create_task(req.run())
                async with asyncio.timeout_at(req.deadline):
                    result = await req.task
                if not req.future.done():
                    req.future.set_result(result)
            except asyncio.CancelledError:
                req.future.cancel()
                if self._closed:
                    return
            except Exception as error:
                # Propagate OOM, provider errors and deadline failures unchanged.
                if not req.future.done():
                    req.future.set_exception(error)
            finally:
                async with self._condition:
                    self._running[req.job_id] -= 1
                    if not self._running[req.job_id]:
                        del self._running[req.job_id]
                    self._requests.pop(req.ticket, None)
                    self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            for req in self._requests.values():
                req.future.cancel()
                if req.task is not None:
                    req.task.cancel()
            for worker in self._workers:
                worker.cancel()
            self._condition.notify_all()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._queues.clear()
        self._rotation.clear()
        self._requests.clear()


class ScheduledProvider:
    """A07-compatible wrapper; instantiate once in the independent worker."""

    def __init__(self, delegate: Any, scheduler: FairScheduler):
        self.delegate = delegate
        self.scheduler = scheduler

    async def check_ready(self, *, deadline: float) -> Any:
        return await self.delegate.check_ready(deadline=deadline)

    async def aclose(self) -> None:
        await self.scheduler.close()
        close = getattr(self.delegate, "aclose", None)
        if close is not None:
            await close()

    async def infer(
        self,
        clip: Any,
        *,
        prompt: str,
        response_schema: dict[str, Any],
        deadline: float,
        cancel_event: CancellationSignal | None = None,
    ) -> Any:
        try:
            return await self.scheduler.submit(
                clip.job_id,
                lambda: self.delegate.infer(
                    clip, prompt=prompt, response_schema=response_schema,
                    deadline=deadline, cancel_event=cancel_event,
                ),
                deadline=deadline,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            if cancel_event is not None and cancel_event.is_set():
                # A lease loss is a provider-level cancellation, distinct from
                # shutting down the entire worker task/event loop.
                from forte.model.adapter import ModelCancelled
                raise ModelCancelled("model_cancelled") from None
            raise

