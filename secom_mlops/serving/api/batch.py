import asyncio
import logging
import time

from typing import Any
from contextlib import suppress

from secom_mlops.serving.api.utils import normalize_prediction, build_prediction_event

logger = logging.getLogger(__name__)


from secom_mlops.serving.api.errors import (
    ModelGatewayError,
)
from secom_mlops.serving.api.client import (
    ModelGatewayClient,
)
from secom_mlops.serving.api.model import (
    PredictionEventContext,
    PendingInvocation,
)
from secom_mlops.serving.api.prediction_event_publisher import PredictionEventPublisher



class PredictionBatcher:
    def __init__(
        self,
        client: ModelGatewayClient,
        event_publisher: PredictionEventPublisher,
        max_batch_size: int,
        max_wait_seconds: float,
        queue_max_size: int,
        queue_timeout_seconds: float,
        response_timeout_seconds: float,
    ) -> None:
        self._client = client
        self._event_publisher = event_publisher
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds
        self._queue_timeout_seconds = queue_timeout_seconds
        self._response_timeout_seconds = response_timeout_seconds
        self._queue = asyncio.Queue(maxsize=queue_max_size)
        self._worker_task: asyncio.Task | None = None
        self._running = False

        if self._max_batch_size < 1:
            raise ValueError("MODEL_BATCH_MAX_SIZE must be >= 1")
        if self._max_wait_seconds < 0:
            raise ValueError("MODEL_BATCH_MAX_WAIT_MS must be >= 0")
        if queue_max_size < 1:
            raise ValueError("MODEL_BATCH_QUEUE_MAX_SIZE must be >= 1")
        if self._queue_timeout_seconds <= 0:
            raise ValueError("MODEL_BATCH_QUEUE_TIMEOUT_MS must be > 0")
        if self._response_timeout_seconds <= 0:
            raise ValueError("MODEL_BATCH_RESPONSE_TIMEOUT_SECONDS must be > 0")

    def start(self) -> None:
        if self._worker_task is not None:
            raise RuntimeError("PredictionBatcher already started")
        self._worker_task = asyncio.create_task(self._run())
        self._running = True

    async def close(self) -> None:
        self._running = False
        # TODO: Support graceful shutdown.
        if self._worker_task is not None:
            self._worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker_task

        while True:
            try:
                pending = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if not pending.future.done():
                pending.future.set_exception(ModelGatewayError("model batcher closed"))
            self._queue.task_done()

    async def invoke(self, features: list[float | None], *, event_context: PredictionEventContext) -> dict[str, Any]:
        if not self._running:
            raise RuntimeError("PredictionBatch is not running.")

        pending = self._new_pending(features, event_context)
        return (await self._enqueue_and_wait([pending]))[0]

    async def invoke_many(self, inputs: list[list[float | None]]) -> list[dict[str, Any]]:
        if not self._running:
            raise RuntimeError("PredictionBatch is not running.")

        pending_items = [
            self._new_pending(features, event_context=None)
            for features in inputs
        ]
        return await self._enqueue_and_wait(pending_items)

    @classmethod
    def _new_pending(cls, features: list[float | None], event_context: PredictionEventContext | None) -> PendingInvocation:
        loop = asyncio.get_running_loop()
        return PendingInvocation(
            features=features,
            future=loop.create_future(),
            event_context=event_context,
            started_at=time.perf_counter(),
        )

    async def _enqueue_and_wait(self, pending_items: list[PendingInvocation]) -> list[dict[str, Any]]:
        try:
            for pending in pending_items:
                await asyncio.wait_for(self._queue.put(pending), timeout=self._queue_timeout_seconds)
        except asyncio.TimeoutError as error:
            for pending in pending_items:
                if not pending.future.done():
                    pending.future.set_exception(ModelGatewayError("model batch queue is full"))
            raise ModelGatewayError("model batch queue is full") from error

        try:
            return await asyncio.wait_for(
                asyncio.gather(*(pending.future for pending in pending_items)),
                timeout=self._response_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            for pending in pending_items:
                if not pending.future.done():
                    pending.future.cancel()
            raise ModelGatewayError("model batch response timed out") from error

    async def _run(self) -> None:
        while True:
            batch: list[PendingInvocation] = []
            try:
                first = await self._queue.get()
                batch.append(first)

                loop = asyncio.get_running_loop()
                deadline = loop.time() + self._max_wait_seconds

                while len(batch) < self._max_batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break

                    batch.append(item)

                await self._flush(batch)
            except asyncio.CancelledError:
                for pending in batch:
                    if not pending.future.done():
                        pending.future.set_exception(ModelGatewayError("model batcher stopped"))
                raise
            finally:
                for _pending in batch:
                    self._queue.task_done()


    async def _flush(self, batch: list[PendingInvocation]) -> None:
        active = [pending for pending in batch if not pending.future.done()]

        if not active:
            return

        try:
            predictions = await self._client.invoke_batch(
                [pending.features for pending in active]
            )
        except Exception as error:
            gateway_error = error if isinstance(error, ModelGatewayError) else ModelGatewayError(str(error))
            for pending in active:
                if not pending.future.done():
                    pending.future.set_exception(gateway_error)
        else:
            for pending, prediction in zip(active, predictions):
                if pending.future.done():
                    continue

                try:
                    normalized_prediction = normalize_prediction(prediction, row_index=0)
                except ModelGatewayError as error:
                    pending.future.set_exception(error)
                    continue

                # For real traffic, not debug traffic.
                if pending.event_context is not None:
                    try:
                        event = build_prediction_event(
                            context=pending.event_context,
                            prediction=normalized_prediction,
                            predicted_at=time.time(),
                            latency_ms=(time.perf_counter() - pending.started_at) * 1000,
                        )
                        self._event_publisher.submit_nowait(event)
                    except Exception:
                        logger.exception(
                            "prediction_event_submit_failed prediction_id=%s",
                            pending.event_context.prediction_id,
                        )

                pending.future.set_result(prediction)
