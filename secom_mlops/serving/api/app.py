import asyncio
import logging
import math
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from valkey.exceptions import ValkeyError

from secom_mlops.feature_store.online_snapshot_reader import (
    InvalidOnlineFeatureSnapshot,
    OnlineFeatureSnapshotNotFound,
    OnlineFeatureSnapshotStore,
)
from secom_mlops.monitor.prediction_events import PredictionEventProducer
from secom_mlops.serving.api.batch import PredictionBatcher
from secom_mlops.serving.api.client import ModelGatewayClient
from secom_mlops.serving.api.config import ServingApiConfig
from secom_mlops.serving.api.errors import ModelGatewayError
from secom_mlops.serving.api.metrics import prediction_metrics
from secom_mlops.serving.api.model import PredictionEventContext
from secom_mlops.serving.api.prediction_event_publisher import (
    BufferedPredictionEventPublisher,
)
from secom_mlops.serving.api.prediction_service import (
    PredictionService,
)
from secom_mlops.serving.api.schemas import (
    BatchPredictResponse,
    BatchPredictRequest,
    PredictByIdResponse,
    PredictByIdRequest,
)
from secom_mlops.serving.api.utils import normalize_prediction

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    config = ServingApiConfig()
    fast_api_app.state.config = config
    fast_api_app.state.prediction_event_producer = PredictionEventProducer(
        bootstrap_servers=config.kafka_bootstrap_servers,
        topic=config.prediction_events_topic,
        client_id=config.prediction_event_client_id,
        flush_timeout_seconds=config.prediction_event_flush_timeout_seconds,
    )
    fast_api_app.state.prediction_event_publisher = BufferedPredictionEventPublisher(
        fast_api_app.state.prediction_event_producer,
        queue_max_size=config.prediction_event_queue_max_size,
        batch_max_size=config.prediction_event_batch_max_size,
        batch_max_wait_seconds=config.prediction_event_batch_max_wait_seconds,
    )
    fast_api_app.state.shadow_prediction_event_publisher = BufferedPredictionEventPublisher(
        fast_api_app.state.prediction_event_producer,
        queue_max_size=config.prediction_event_queue_max_size,
        batch_max_size=config.prediction_event_batch_max_size,
        batch_max_wait_seconds=config.prediction_event_batch_max_wait_seconds,
    )
    fast_api_app.state.online_snapshot_store = OnlineFeatureSnapshotStore(
        valkey_url=config.valkey_url,
        valkey_host=config.valkey_host,
        valkey_port=config.valkey_port,
        valkey_database=config.valkey_database,
        timeout_seconds=config.valkey_timeout_seconds,
        key_prefix=config.valkey_key_prefix,
    )
    fast_api_app.state.model_gateway_client = ModelGatewayClient(
        base_url=config.model_gateway_url,
        path=config.primary_batch_path,
        timeout_seconds=config.model_gateway_timeout_seconds,
    )
    fast_api_app.state.shadow_model_gateway_client = ModelGatewayClient(
        base_url=config.model_gateway_url,
        path=config.shadow_batch_path,
        timeout_seconds=config.model_gateway_timeout_seconds,
    )

    fast_api_app.state.primary_prediction_batcher = PredictionBatcher(
        client=fast_api_app.state.model_gateway_client,
        event_publisher=fast_api_app.state.prediction_event_publisher,
        prediction_metrics=prediction_metrics,
        destination="release",
        max_batch_size=config.model_batch_max_size,
        max_wait_seconds=config.model_batch_max_wait_seconds,
        queue_max_size=config.model_batch_queue_max_size,
        queue_timeout_seconds=config.model_batch_queue_timeout_seconds,
        response_timeout_seconds=config.model_batch_response_timeout_seconds,
    )
    fast_api_app.state.shadow_prediction_batcher = PredictionBatcher(
        client=fast_api_app.state.shadow_model_gateway_client,
        event_publisher=fast_api_app.state.shadow_prediction_event_publisher,
        prediction_metrics=prediction_metrics,
        destination="shadow",
        max_batch_size=config.model_batch_max_size,
        max_wait_seconds=config.model_batch_max_wait_seconds,
        queue_max_size=config.model_batch_queue_max_size,
        queue_timeout_seconds=config.model_batch_queue_timeout_seconds,
        response_timeout_seconds=config.model_batch_response_timeout_seconds,
    )

    fast_api_app.state.prediction_service = PredictionService(
        fast_api_app.state.primary_prediction_batcher,
        fast_api_app.state.shadow_prediction_batcher,
    )
    fast_api_app.state.prediction_event_publisher.start()
    fast_api_app.state.shadow_prediction_event_publisher.start()
    fast_api_app.state.prediction_service.start()

    try:
        yield
    finally:
        await fast_api_app.state.prediction_service.close()
        await fast_api_app.state.prediction_event_publisher.close()
        await fast_api_app.state.shadow_prediction_event_publisher.close()
        await asyncio.to_thread(fast_api_app.state.prediction_event_producer.close)
        await fast_api_app.state.model_gateway_client.close()
        await fast_api_app.state.shadow_model_gateway_client.close()
        fast_api_app.state.online_snapshot_store.close()


app = FastAPI(title="SECOM Fail Detection API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@app.post("/predict", response_model=BatchPredictResponse)
async def predict(payload: BatchPredictRequest, request: Request):
    request_id = str(uuid4())
    inputs = [row.features for row in payload.rows]

    try:
        raw_predictions = await request.app.state.prediction_service.predict_debug_many(
            inputs
        )
        predictions = [
            normalize_prediction(raw_prediction, row_index=row_index)
            for row_index, raw_prediction in enumerate(raw_predictions)
        ]
    except ModelGatewayError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    response_predictions = []

    for row, prediction in zip(payload.rows, predictions):
        prediction_id = str(uuid4())

        response_predictions.append({
            "prediction_id": prediction_id,
            "request_id": request_id,
            "sample_id": row.sample_id,
            **prediction,
        })

    return {
        "request_id": request_id,
        "predictions": response_predictions,
    }


@app.post("/predict-by-id", response_model=PredictByIdResponse)
async def predict_by_id(payload: PredictByIdRequest, request: Request):
    try:
        snapshot = await asyncio.to_thread(
            request.app.state.online_snapshot_store.load,
            payload.sample_id,
        )
    except OnlineFeatureSnapshotNotFound as error:
        raise HTTPException(
            status_code=404,
            detail=f"online feature snapshot not found: {payload.sample_id}",
        ) from error
    except InvalidOnlineFeatureSnapshot as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValkeyError as error:
        raise HTTPException(
            status_code=503,
            detail="online feature store unavailable",
        ) from error

    if snapshot.snapshot_status == "timed_out":
        return _snapshot_timed_out_response(snapshot)

    if not snapshot.is_complete or snapshot.snapshot_status != "complete":
        return _snapshot_not_ready_response(
            snapshot,
            request.app.state.config.predict_partial_retry_after_ms,
        )

    request_id = str(uuid4())
    prediction_id = str(uuid4())

    try:
        raw_prediction = await request.app.state.prediction_service.predict(
            snapshot.values,
            event_context=PredictionEventContext(
                prediction_id=prediction_id,
                request_id=request_id,
                sample_id=snapshot.sample_id,
                serving_snapshot_id=snapshot.serving_snapshot_id,
                snapshot_version=snapshot.snapshot_version,
                feature_hash=snapshot.feature_hash,
                missing_count=snapshot.missing_count,
            ),
        )
        prediction = normalize_prediction(raw_prediction, row_index=0)
    except ModelGatewayError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "prediction_id": prediction_id,
        "request_id": request_id,
        "sample_id": snapshot.sample_id,
        "serving_snapshot_id": snapshot.serving_snapshot_id,
        "snapshot_version": snapshot.snapshot_version,
        "feature_hash": snapshot.feature_hash,
        "snapshot_time": snapshot.snapshot_time,
        "feature_count": snapshot.feature_count,
        "missing_count": snapshot.missing_count,
        **prediction,
    }


def _snapshot_not_ready_response(snapshot, retry_after_ms: int) -> JSONResponse:
    retry_after_seconds = max(1, math.ceil(retry_after_ms / 1000))

    return JSONResponse(
        status_code=409,
        headers={"Retry-After": str(retry_after_seconds)},
        content={
            "message": "online feature snapshot is not complete",
            "sample_id": snapshot.sample_id,
            "snapshot_status": snapshot.snapshot_status,
            "feature_count": snapshot.feature_count,
            "missing_count": snapshot.missing_count,
            "retryable": True,
            "retry_after_ms": retry_after_ms,
        },
    )


def _snapshot_timed_out_response(snapshot) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "message": "online feature snapshot timed out",
            "sample_id": snapshot.sample_id,
            "snapshot_status": snapshot.snapshot_status,
            "feature_count": snapshot.feature_count,
            "missing_count": snapshot.missing_count,
            "retryable": False,
        },
    )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
