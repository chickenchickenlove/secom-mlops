import asyncio
import unittest
from types import SimpleNamespace

from secom_mlops.feature_store.online_snapshot_reader import OnlineFeatureSnapshot
from secom_mlops.serving.api import (
    BatchPredictRequest,
    PredictByIdRequest,
    PredictRow,
    _build_snapshot_prediction_event,
    predict,
    predict_by_id,
)
from secom_mlops_common.schemas.secom import FEATURE_KEYS, NUM_FEATURES

FEATURE_HASH = "sha256:v1:" + "0" * 64


class _FakeModelGatewayBatcher:
    def __init__(self) -> None:
        self.inputs: list[list[list[float | None]]] = []

    async def invoke_many(self, inputs: list[list[float | None]]) -> list[dict[str, object]]:
        self.inputs.append(inputs)
        return [_model_prediction() for _ in inputs]


class _CapturingPredictionEventProducer:
    def __init__(self) -> None:
        self.batches: list[list[dict[str, object]]] = []

    def publish_many(self, events: list[dict[str, object]]) -> None:
        self.batches.append(events)


class _FailingPredictionEventProducer:
    def publish_many(self, events: list[dict[str, object]]) -> None:
        raise AssertionError(f"/predict must not publish prediction events: {events}")


class _FakeOnlineSnapshotStore:
    def __init__(self, snapshot: OnlineFeatureSnapshot) -> None:
        self.snapshot = snapshot
        self.loaded_sample_ids: list[str] = []

    def load(self, sample_id: str) -> OnlineFeatureSnapshot:
        self.loaded_sample_ids.append(sample_id)
        return self.snapshot


class ServingPredictionEventTest(unittest.TestCase):
    def test_snapshot_event_contains_reference_without_features(self) -> None:
        event = _build_snapshot_prediction_event(
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

    def test_predict_returns_inference_without_publishing_event(self) -> None:
        features = [float(index) for index in range(NUM_FEATURES)]
        batcher = _FakeModelGatewayBatcher()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            model_gateway_batcher=batcher,
            prediction_event_producer=_FailingPredictionEventProducer(),
        )))

        response = asyncio.run(predict(
            BatchPredictRequest(rows=[
                PredictRow(sample_id="secom-0000001", features=features),
            ]),
            request,
        ))

        self.assertEqual("secom-0000001", response["predictions"][0]["sample_id"])
        self.assertEqual(-1, response["predictions"][0]["prediction"])
        self.assertEqual([[features]], batcher.inputs)

    def test_predict_by_id_publishes_loaded_snapshot_reference(self) -> None:
        snapshot = _complete_snapshot()
        snapshot_store = _FakeOnlineSnapshotStore(snapshot)
        batcher = _FakeModelGatewayBatcher()
        producer = _CapturingPredictionEventProducer()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            online_snapshot_store=snapshot_store,
            model_gateway_batcher=batcher,
            prediction_event_producer=producer,
        )))

        response = asyncio.run(predict_by_id(
            PredictByIdRequest(sample_id=snapshot.sample_id),
            request,
        ))

        self.assertEqual([snapshot.sample_id], snapshot_store.loaded_sample_ids)
        self.assertEqual([[snapshot.values]], batcher.inputs)
        self.assertEqual(snapshot.serving_snapshot_id, response["serving_snapshot_id"])
        self.assertEqual(snapshot.snapshot_version, response["snapshot_version"])
        self.assertEqual(snapshot.feature_hash, response["feature_hash"])
        self.assertEqual(1, len(producer.batches))
        self.assertEqual(1, len(producer.batches[0]))

        event = producer.batches[0][0]
        self.assertEqual(snapshot.serving_snapshot_id, event["serving_snapshot_id"])
        self.assertEqual(snapshot.snapshot_version, event["snapshot_version"])
        self.assertEqual(snapshot.feature_hash, event["feature_hash"])
        self.assertNotIn("features", event)
        self.assertEqual(snapshot.missing_count, event["missing_count"])


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
