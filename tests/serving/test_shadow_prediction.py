import asyncio
import unittest
from typing import Any
from unittest.mock import patch

from secom_mlops.serving.api.batch import PredictionBatcher
from secom_mlops.serving.api.client import ModelRuntimeClient
from secom_mlops.serving.api.metrics import PredictionDestination
from secom_mlops.serving.api.model import PredictionEventContext
from secom_mlops.serving.api.prediction_service import PredictionService


class _FakePrimaryBatcher:
    def __init__(self) -> None:
        self.features: list[float | None] | None = None
        self.event_context: PredictionEventContext | None = None

    async def invoke(
        self,
        features: list[float | None],
        *,
        event_context: PredictionEventContext,
    ) -> dict[str, Any]:
        self.features = features
        self.event_context = event_context
        return {"prediction": -1}


class _FakeShadowBatcher:
    def __init__(self, *, accepted: bool = True) -> None:
        self.accepted = accepted
        self.features: list[float | None] | None = None
        self.event_context: PredictionEventContext | None = None

    def submit_nowait(
        self,
        features: list[float | None],
        *,
        event_context: PredictionEventContext,
    ) -> bool:
        self.features = features
        self.event_context = event_context
        return self.accepted


class _ShadowModelRuntimeClient:
    async def invoke_batch(
        self,
        inputs: list[list[float | None]],
    ) -> list[dict[str, Any]]:
        return [_model_prediction(runtime_slot="shadow") for _ in inputs]


class _CapturingPredictionEventPublisher:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.submitted = asyncio.Event()

    def submit_nowait(self, event: dict[str, Any]) -> bool:
        self.events.append(event)
        self.submitted.set()
        return True


class _CapturingPredictionMetrics:
    def __init__(self) -> None:
        self.dispatches: list[tuple[PredictionDestination, int]] = []

    def record_dispatch(
        self,
        destination: PredictionDestination,
        prediction_count: int,
    ) -> None:
        self.dispatches.append((destination, prediction_count))


class _FakeHttpResponse:
    status_code = 200
    text = ""

    def json(self) -> dict[str, Any]:
        return {"predictions": [_model_prediction(runtime_slot="shadow")]}


class _CapturingHttpClient:
    def __init__(self) -> None:
        self.path: str | None = None
        self.payload: dict[str, Any] | None = None
        self.closed = False

    async def post(self, path: str, *, json: dict[str, Any]) -> _FakeHttpResponse:
        self.path = path
        self.payload = json
        return _FakeHttpResponse()

    async def aclose(self) -> None:
        self.closed = True


class ShadowPredictionTest(unittest.TestCase):
    def test_service_fans_out_with_distinct_prediction_ids(self) -> None:
        async def scenario() -> None:
            primary = _FakePrimaryBatcher()
            shadow = _FakeShadowBatcher()
            service = PredictionService(primary, shadow)
            context = _event_context(prediction_id="primary-prediction")
            features = [0.0, 1.0]

            prediction = await service.predict(
                features,
                event_context=context,
            )

            self.assertEqual({"prediction": -1}, prediction)
            self.assertEqual(features, primary.features)
            self.assertIs(context, primary.event_context)
            self.assertEqual(features, shadow.features)
            self.assertIsNotNone(shadow.event_context)
            self.assertNotEqual(
                context.prediction_id,
                shadow.event_context.prediction_id,
            )
            self.assertEqual(context.request_id, shadow.event_context.request_id)
            self.assertEqual(context.sample_id, shadow.event_context.sample_id)
            self.assertEqual(
                context.serving_snapshot_id,
                shadow.event_context.serving_snapshot_id,
            )
            self.assertEqual(
                context.snapshot_version,
                shadow.event_context.snapshot_version,
            )
            self.assertEqual(context.feature_hash, shadow.event_context.feature_hash)

        asyncio.run(scenario())

    def test_shadow_rejection_does_not_affect_primary_result(self) -> None:
        async def scenario() -> None:
            service = PredictionService(
                _FakePrimaryBatcher(),
                _FakeShadowBatcher(accepted=False),
            )

            prediction = await service.predict(
                [0.0],
                event_context=_event_context(),
            )

            self.assertEqual({"prediction": -1}, prediction)

        asyncio.run(scenario())

    def test_shadow_batcher_submits_shadow_runtime_event(self) -> None:
        async def scenario() -> None:
            publisher = _CapturingPredictionEventPublisher()
            prediction_metrics = _CapturingPredictionMetrics()
            batcher = PredictionBatcher(
                client=_ShadowModelRuntimeClient(),
                event_publisher=publisher,
                prediction_metrics=prediction_metrics,
                destination="shadow",
                max_batch_size=16,
                max_wait_seconds=0,
                queue_max_size=1024,
                queue_timeout_seconds=2.0,
                response_timeout_seconds=30.0,
            )
            context = _event_context(prediction_id="shadow-prediction")
            batcher.start()

            try:
                self.assertTrue(
                    batcher.submit_nowait([0.0], event_context=context)
                )
                await asyncio.wait_for(publisher.submitted.wait(), timeout=1.0)
            finally:
                await batcher.close()

            self.assertEqual(1, len(publisher.events))
            event = publisher.events[0]
            self.assertEqual("shadow-prediction", event["prediction_id"])
            self.assertEqual(context.request_id, event["request_id"])
            self.assertEqual("shadow", event["runtime_slot"])
            self.assertEqual([("shadow", 1)], prediction_metrics.dispatches)

        asyncio.run(scenario())

    def test_model_runtime_client_uses_shadow_invocation_path(self) -> None:
        async def scenario() -> None:
            http_client = _CapturingHttpClient()

            with patch(
                "secom_mlops.serving.api.client.httpx.AsyncClient",
                return_value=http_client,
            ):
                client = ModelRuntimeClient(
                    base_url="http://model-server-shadow:28093",
                    path="/invocations",
                    timeout_seconds=10.0,
                )

            try:
                predictions = await client.invoke_batch([[0.0]])
            finally:
                await client.close()

            self.assertEqual("/invocations", http_client.path)
            self.assertEqual({"inputs": [[0.0]]}, http_client.payload)
            self.assertEqual("shadow", predictions[0]["runtime_slot"])
            self.assertTrue(http_client.closed)

        asyncio.run(scenario())


def _event_context(
    *,
    prediction_id: str = "prediction-001",
) -> PredictionEventContext:
    return PredictionEventContext(
        prediction_id=prediction_id,
        request_id="request-001",
        sample_id="secom-0000001",
        serving_snapshot_id="state:secom-0000001:1000:3",
        snapshot_version=3,
        feature_hash="sha256:v1:" + "0" * 64,
        missing_count=1,
    )


def _model_prediction(*, runtime_slot: str) -> dict[str, Any]:
    return {
        "fail_probability": 0.2,
        "prediction": -1,
        "label": "pass",
        "threshold": 0.5,
        "model_uri": "models:/secom-fail-detector/1",
        "model_name": "secom-fail-detector",
        "model_version": "1",
        "model_alias": "champion",
        "model_run_id": "run-shadow-001",
        "runtime_slot": runtime_slot,
    }


if __name__ == "__main__":
    unittest.main()
