import asyncio
import unittest
from typing import Any

from secom_mlops.serving.api.prediction_event_publisher import (
    BufferedPredictionEventPublisher,
)


class _CapturingSink:
    def __init__(self) -> None:
        self.batches: list[list[dict[str, Any]]] = []

    def publish_many(self, events: list[dict[str, Any]]) -> None:
        self.batches.append(events)


class _FailOnceSink:
    def __init__(self) -> None:
        self.calls = 0
        self.published: list[dict[str, Any]] = []

    def publish_many(self, events: list[dict[str, Any]]) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("broker unavailable")
        self.published.extend(events)


class BufferedPredictionEventPublisherTest(unittest.TestCase):
    def test_publishes_events_in_batches(self) -> None:
        async def scenario() -> None:
            sink = _CapturingSink()
            publisher = BufferedPredictionEventPublisher(
                sink,
                queue_max_size=10,
                batch_max_size=2,
                batch_max_wait_seconds=1.0,
            )
            publisher.start()

            self.assertTrue(publisher.submit_nowait({"prediction_id": "p1"}))
            self.assertTrue(publisher.submit_nowait({"prediction_id": "p2"}))

            await publisher.close()

            self.assertEqual([[{"prediction_id": "p1"}, {"prediction_id": "p2"}]], sink.batches)

        asyncio.run(scenario())

    def test_returns_false_when_queue_is_full(self) -> None:
        async def scenario() -> None:
            sink = _CapturingSink()
            publisher = BufferedPredictionEventPublisher(
                sink,
                queue_max_size=1,
                batch_max_size=1,
                batch_max_wait_seconds=0,
            )
            publisher.start()

            self.assertTrue(publisher.submit_nowait({"prediction_id": "p1"}))
            self.assertFalse(publisher.submit_nowait({"prediction_id": "p2"}))

            await publisher.close()
            self.assertEqual([[{"prediction_id": "p1"}]], sink.batches)

        asyncio.run(scenario())

    def test_publish_failure_does_not_stop_worker(self) -> None:
        async def scenario() -> None:
            sink = _FailOnceSink()
            publisher = BufferedPredictionEventPublisher(
                sink,
                queue_max_size=10,
                batch_max_size=1,
                batch_max_wait_seconds=0,
            )
            publisher.start()

            self.assertTrue(publisher.submit_nowait({"prediction_id": "p1"}))
            self.assertTrue(publisher.submit_nowait({"prediction_id": "p2"}))

            await publisher.close()

            self.assertEqual(2, sink.calls)
            self.assertEqual([{"prediction_id": "p2"}], sink.published)

        asyncio.run(scenario())

    def test_rejects_events_after_close(self) -> None:
        async def scenario() -> None:
            publisher = BufferedPredictionEventPublisher(
                _CapturingSink(),
                queue_max_size=10,
                batch_max_size=1,
                batch_max_wait_seconds=0,
            )
            publisher.start()
            await publisher.close()

            self.assertFalse(publisher.submit_nowait({"prediction_id": "p1"}))

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
