import logging
import asyncio
from contextlib import suppress
from typing import Any, Protocol


logger = logging.getLogger(__name__)


class PredictionEventSink(Protocol):
    def publish_many(self, events: list[dict[str, Any]]) -> None:
        ...


class PredictionEventPublisher(Protocol):
    def submit_nowait(self, event: dict[str, Any]) -> bool:
        ...


class BufferedPredictionEventPublisher:
    """Publish prediction events through a bounded, best-effort memory queue.

    Inference does not depend on event delivery. Events are dropped when the
    queue rejects them or the sink fails, without retry or durable storage.
    Graceful shutdown drains the queue, but abnormal process termination can
    lose queued events. Delivery failures are logged instead of being
    propagated to inference callers.
    """

    def __init__(
        self,
        sink: PredictionEventSink,
        *,
        queue_max_size: int,
        batch_max_size: int,
        batch_max_wait_seconds: float,
    ) -> None:
        self._sink = sink
        self._batch_max_size = batch_max_size
        self._batch_max_wait_seconds = batch_max_wait_seconds
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=queue_max_size
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._accepting = False

        if self._batch_max_size < 1:
            raise ValueError("PREDICTION_EVENT_BATCH_MAX_SIZE must be >= 1")
        if self._batch_max_wait_seconds < 0:
            raise ValueError("PREDICTION_EVENT_BATCH_MAX_WAIT_MS must be >= 0")
        if queue_max_size < 1:
            raise ValueError("PREDICTION_EVENT_QUEUE_MAX_SIZE must be >= 1")

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._worker_task is not None:
            raise RuntimeError("prediction event publisher already started")

        self._accepting = True
        self._worker_task = asyncio.create_task(self._run())

    def submit_nowait(self, event: dict[str, Any]) -> bool:
        if not self._accepting:
            logger.warning(
                "prediction_event_publisher_not_accepting sample_id=%s prediction_id=%s",
                event.get("sample_id"),
                event.get("prediction_id"),
            )
            return False

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "prediction_event_queue_full sample_id=%s prediction_id=%s",
                event.get("sample_id"),
                event.get("prediction_id"),
            )
            return False

        return True

    async def close(self) -> None:
        self._accepting = False

        if self._worker_task is None:
            return

        await self._queue.join()
        self._worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._worker_task
        self._worker_task = None

    async def _run(self) -> None:
        while True:
            batch: list[dict[str, Any]] = []

            try:
                first = await self._queue.get()
                batch.append(first)

                loop = asyncio.get_running_loop()
                deadline = loop.time() + self._batch_max_wait_seconds

                while len(batch) < self._batch_max_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break

                    try:
                        event = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=remaining,
                        )
                    except asyncio.TimeoutError:
                        break

                    batch.append(event)

                await asyncio.to_thread(self._sink.publish_many, batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "prediction_event_publish_failed event_count=%d",
                    len(batch),
                )
            finally:
                for _ in batch:
                    self._queue.task_done()
