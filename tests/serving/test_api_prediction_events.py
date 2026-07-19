import asyncio
import unittest
from types import SimpleNamespace

from secom_mlops.feature_store.online_snapshot_reader import OnlineFeatureSnapshot
from secom_mlops.serving.api.app import (
    predict,
    predict_by_id,
)
from secom_mlops.serving.api.errors import ModelGatewayError
from secom_mlops.serving.api.metrics import PredictionDestination
from secom_mlops.serving.api.model import PredictionEventContext
from secom_mlops.serving.api.batch import PredictionBatcher
from secom_mlops.serving.api.schemas import (
    BatchPredictRequest,
    PredictByIdRequest,
    PredictRow,
)
from secom_mlops.serving.api.utils import build_snapshot_prediction_event
from secom_mlops_common.schemas.secom import FEATURE_KEYS, NUM_FEATURES

FEATURE_HASH = "sha256:v1:" + "0" * 64


class _FakePredictionService:
    def __init__(self) -> None:
        self.inputs: list[list[list[float | None]]] = []
        self.event_contexts: list[PredictionEventContext | None] = []

    async def predict(
        self,
        features: list[float | None],
        *,
        event_context: PredictionEventContext,
    ) -> dict[str, object]:
        self.inputs.append([features])
        self.event_contexts.append(event_context)
        return _model_prediction()

    async def predict_debug_many(
        self,
        inputs: list[list[float | None]],
    ) -> list[dict[str, object]]:
        self.inputs.append(inputs)
        self.event_contexts.append(None)
        return [_model_prediction() for _ in inputs]


class _FakeModelGatewayClient:
    def __init__(self) -> None:
        self.inputs: list[list[list[float | None]]] = []

    async def invoke_batch(
        self,
        inputs: list[list[float | None]],
    ) -> list[dict[str, object]]:
        self.inputs.append(inputs)
        return [_model_prediction() for _ in inputs]


class _MalformedThenValidModelGatewayClient:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke_batch(
        self,
        inputs: list[list[float | None]],
    ) -> list[object]:
        self.calls += 1
        if self.calls == 1:
            return [None for _ in inputs]
        return [_model_prediction() for _ in inputs]


class _CapturingPredictionEventPublisher:
    def __init__(self, *, accepted: bool = True) -> None:
        self.accepted = accepted
        self.events: list[dict[str, object]] = []

    def submit_nowait(self, event: dict[str, object]) -> bool:
        self.events.append(event)
        return self.accepted


class _FailingPredictionEventPublisher:
    def submit_nowait(self, event: dict[str, object]) -> bool:
        raise RuntimeError(f"publisher stopped: {event['prediction_id']}")


class _CapturingPredictionMetrics:
    def __init__(self) -> None:
        self.dispatches: list[tuple[PredictionDestination, int]] = []

    def record_dispatch(
        self,
        destination: PredictionDestination,
        prediction_count: int,
    ) -> None:
        self.dispatches.append((destination, prediction_count))


class _FailingPredictionMetrics:
    def record_dispatch(
        self,
        destination: PredictionDestination,
        prediction_count: int,
    ) -> None:
        raise RuntimeError(
            f"metric unavailable: {destination} {prediction_count}"
        )


class _FakeOnlineSnapshotStore:
    def __init__(self, snapshot: OnlineFeatureSnapshot) -> None:
        self.snapshot = snapshot
        self.loaded_sample_ids: list[str] = []

    def load(self, sample_id: str) -> OnlineFeatureSnapshot:
        self.loaded_sample_ids.append(sample_id)
        return self.snapshot


class ServingPredictionEventTest(unittest.TestCase):
    def test_snapshot_event_contains_reference_without_features(self) -> None:
        event = build_snapshot_prediction_event(
            prediction_id="prediction-001",
            request_id="request-001",
            sample_id="secom-0000001",
            serving_snapshot_id="state:secom-0000001:1000:3",
            snapshot_version=3,
            feature_hash=FEATURE_HASH,
            prediction=_normalized_prediction(),
            predicted_at=2.0,
            missing_count=1,
            latency_ms=12.5,
        )

        self.assertEqual("state:secom-0000001:1000:3", event["serving_snapshot_id"])
        self.assertEqual(3, event["snapshot_version"])
        self.assertEqual(FEATURE_HASH, event["feature_hash"])
        self.assertNotIn("features", event)
        self.assertEqual(1, event["missing_count"])

    def test_prediction_batcher_submits_event_after_inference(self) -> None:
        async def scenario() -> None:
            client = _FakeModelGatewayClient()
            publisher = _CapturingPredictionEventPublisher()
            prediction_metrics = _CapturingPredictionMetrics()
            batcher = _prediction_batcher(
                client,
                publisher,
                prediction_metrics=prediction_metrics,
            )
            context = PredictionEventContext(
                prediction_id="prediction-001",
                request_id="request-001",
                sample_id="secom-0000001",
                serving_snapshot_id="state:secom-0000001:1000:3",
                snapshot_version=3,
                feature_hash=FEATURE_HASH,
                missing_count=1,
            )
            features = [float(index) for index in range(NUM_FEATURES)]
            batcher.start()

            try:
                prediction = await batcher.invoke(
                    features,
                    event_context=context,
                )
            finally:
                await batcher.close()

            self.assertEqual(-1, prediction["prediction"])
            self.assertEqual([[features]], client.inputs)
            self.assertEqual(1, len(publisher.events))
            event = publisher.events[0]
            self.assertEqual("prediction-001", event["prediction_id"])
            self.assertEqual("request-001", event["request_id"])
            self.assertEqual(context.serving_snapshot_id, event["serving_snapshot_id"])
            self.assertEqual(context.feature_hash, event["feature_hash"])
            self.assertEqual("release", event["runtime_slot"])
            self.assertGreaterEqual(event["latency_ms"], 0)
            self.assertEqual([("release", 1)], prediction_metrics.dispatches)

        asyncio.run(scenario())

    def test_prediction_batcher_invoke_many_does_not_submit_events(self) -> None:
        async def scenario() -> None:
            publisher = _CapturingPredictionEventPublisher()
            batcher = _prediction_batcher(_FakeModelGatewayClient(), publisher)
            batcher.start()

            try:
                predictions = await batcher.invoke_many([
                    [0.0] * NUM_FEATURES,
                    [1.0] * NUM_FEATURES,
                ])
            finally:
                await batcher.close()

            self.assertEqual(2, len(predictions))
            self.assertEqual([], publisher.events)

        asyncio.run(scenario())

    def test_prediction_batcher_does_not_record_debug_dispatch(self) -> None:
        async def scenario() -> None:
            prediction_metrics = _CapturingPredictionMetrics()
            batcher = _prediction_batcher(
                _FakeModelGatewayClient(),
                _CapturingPredictionEventPublisher(),
                prediction_metrics=prediction_metrics,
            )
            batcher.start()

            try:
                await batcher.invoke_many([
                    [0.0] * NUM_FEATURES,
                    [1.0] * NUM_FEATURES,
                ])
            finally:
                await batcher.close()

            self.assertEqual([], prediction_metrics.dispatches)

        asyncio.run(scenario())

    def test_prediction_batcher_survives_metric_record_failure(self) -> None:
        async def scenario() -> None:
            client = _FakeModelGatewayClient()
            batcher = _prediction_batcher(
                client,
                _CapturingPredictionEventPublisher(),
                prediction_metrics=_FailingPredictionMetrics(),
            )
            context = PredictionEventContext(
                prediction_id="prediction-001",
                request_id="request-001",
                sample_id="secom-0000001",
                serving_snapshot_id="state:secom-0000001:1000:3",
                snapshot_version=3,
                feature_hash=FEATURE_HASH,
                missing_count=1,
            )
            batcher.start()

            try:
                with self.assertLogs(
                    "secom_mlops.serving.api.batch",
                    level="ERROR",
                ):
                    first = await batcher.invoke(
                        [0.0] * NUM_FEATURES,
                        event_context=context,
                    )
                    second = await batcher.invoke(
                        [1.0] * NUM_FEATURES,
                        event_context=context,
                    )
            finally:
                await batcher.close()

            self.assertEqual(-1, first["prediction"])
            self.assertEqual(-1, second["prediction"])
            self.assertEqual(2, len(client.inputs))

        asyncio.run(scenario())

    def test_prediction_batcher_survives_malformed_prediction(self) -> None:
        async def scenario() -> None:
            client = _MalformedThenValidModelGatewayClient()
            publisher = _CapturingPredictionEventPublisher()
            batcher = _prediction_batcher(client, publisher)
            batcher.start()

            try:
                with self.assertRaisesRegex(
                    ModelGatewayError,
                    "model gateway prediction must be an object",
                ):
                    await batcher.invoke_many([[0.0] * NUM_FEATURES])

                predictions = await batcher.invoke_many([[1.0] * NUM_FEATURES])
            finally:
                await batcher.close()

            self.assertEqual(2, client.calls)
            self.assertEqual(-1, predictions[0]["prediction"])

        asyncio.run(scenario())

    def test_prediction_batcher_does_not_fail_when_event_is_rejected(self) -> None:
        async def scenario() -> None:
            publisher = _CapturingPredictionEventPublisher(accepted=False)
            batcher = _prediction_batcher(_FakeModelGatewayClient(), publisher)
            context = PredictionEventContext(
                prediction_id="prediction-001",
                request_id="request-001",
                sample_id="secom-0000001",
                serving_snapshot_id="state:secom-0000001:1000:3",
                snapshot_version=3,
                feature_hash=FEATURE_HASH,
                missing_count=1,
            )
            batcher.start()

            try:
                prediction = await batcher.invoke(
                    [0.0] * NUM_FEATURES,
                    event_context=context,
                )
            finally:
                await batcher.close()

            self.assertEqual(-1, prediction["prediction"])
            self.assertEqual(1, len(publisher.events))

        asyncio.run(scenario())

    def test_prediction_batcher_does_not_fail_when_publisher_raises(self) -> None:
        async def scenario() -> None:
            batcher = _prediction_batcher(
                _FakeModelGatewayClient(),
                _FailingPredictionEventPublisher(),
            )
            context = PredictionEventContext(
                prediction_id="prediction-001",
                request_id="request-001",
                sample_id="secom-0000001",
                serving_snapshot_id="state:secom-0000001:1000:3",
                snapshot_version=3,
                feature_hash=FEATURE_HASH,
                missing_count=1,
            )
            batcher.start()

            try:
                with self.assertLogs(
                    "secom_mlops.serving.api.batch",
                    level="ERROR",
                ):
                    prediction = await batcher.invoke(
                        [0.0] * NUM_FEATURES,
                        event_context=context,
                    )
            finally:
                await batcher.close()

            self.assertEqual(-1, prediction["prediction"])

        asyncio.run(scenario())

    def test_predict_returns_inference_without_publishing_event(self) -> None:
        features = [float(index) for index in range(NUM_FEATURES)]
        service = _FakePredictionService()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            prediction_service=service,
        )))

        response = asyncio.run(predict(
            BatchPredictRequest(rows=[
                PredictRow(sample_id="secom-0000001", features=features),
            ]),
            request,
        ))

        self.assertEqual("secom-0000001", response["predictions"][0]["sample_id"])
        self.assertEqual(-1, response["predictions"][0]["prediction"])
        self.assertEqual([[features]], service.inputs)
        self.assertEqual([None], service.event_contexts)

    def test_predict_by_id_submits_loaded_snapshot_context(self) -> None:
        snapshot = _complete_snapshot()
        snapshot_store = _FakeOnlineSnapshotStore(snapshot)
        service = _FakePredictionService()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            online_snapshot_store=snapshot_store,
            prediction_service=service,
        )))

        response = asyncio.run(predict_by_id(
            PredictByIdRequest(sample_id=snapshot.sample_id),
            request,
        ))

        self.assertEqual([snapshot.sample_id], snapshot_store.loaded_sample_ids)
        self.assertEqual([[snapshot.values]], service.inputs)
        self.assertEqual(snapshot.serving_snapshot_id, response["serving_snapshot_id"])
        self.assertEqual(snapshot.snapshot_version, response["snapshot_version"])
        self.assertEqual(snapshot.feature_hash, response["feature_hash"])
        self.assertEqual(1, len(service.event_contexts))
        context = service.event_contexts[0]
        self.assertIsNotNone(context)
        self.assertEqual(snapshot.sample_id, context.sample_id)
        self.assertEqual(snapshot.serving_snapshot_id, context.serving_snapshot_id)
        self.assertEqual(snapshot.snapshot_version, context.snapshot_version)
        self.assertEqual(snapshot.feature_hash, context.feature_hash)
        self.assertEqual(snapshot.missing_count, context.missing_count)


def _complete_snapshot() -> OnlineFeatureSnapshot:
    features = {
        key: None if index == 0 else float(index)
        for index, key in enumerate(FEATURE_KEYS)
    }
    return OnlineFeatureSnapshot(
        serving_snapshot_id="state:secom-0000001:1000:3",
        snapshot_version=3,
        feature_hash=FEATURE_HASH,
        sample_id="secom-0000001",
        snapshot_time=1.0,
        snapshot_status="complete",
        feature_count=NUM_FEATURES,
        missing_count=1,
        is_complete=True,
        features_json=features,
    )


def _prediction_batcher(
    client,
    publisher,
    *,
    prediction_metrics: _CapturingPredictionMetrics | None = None,
) -> PredictionBatcher:
    return PredictionBatcher(
        client=client,
        event_publisher=publisher,
        prediction_metrics=prediction_metrics or _CapturingPredictionMetrics(),
        destination="release",
        max_batch_size=16,
        max_wait_seconds=0,
        queue_max_size=1024,
        queue_timeout_seconds=2.0,
        response_timeout_seconds=30.0,
    )


def _model_prediction() -> dict[str, object]:
    return {
        "fail_probability": 0.2,
        "prediction": -1,
        "label": "pass",
        "threshold": 0.5,
        "model_uri": "models:/secom-fail-detector/1",
        "model_name": "secom-fail-detector",
        "model_version": "1",
        "model_alias": "champion",
        "model_run_id": "run-001",
        "runtime_slot": "release",
    }


def _normalized_prediction() -> dict[str, object]:
    return {
        "row_index": 0,
        **_model_prediction(),
    }


if __name__ == "__main__":
    unittest.main()
