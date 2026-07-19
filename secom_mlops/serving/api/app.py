import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from valkey.exceptions import ValkeyError

from secom_mlops.feature_store.online_snapshot_reader import (
    InvalidOnlineFeatureSnapshot,
    OnlineFeatureSnapshotNotFound,
    OnlineFeatureSnapshotStore,
)
from secom_mlops.monitor.prediction_events import PredictionEventProducer
from secom_mlops.serving.api.prediction_event_publisher import (
    BufferedPredictionEventPublisher,
)
from secom_mlops.serving.api.utils import normalize_prediction
from secom_mlops.serving.api.client import ModelGatewayClient
from secom_mlops.serving.api.batch import PredictionBatcher
from secom_mlops.serving.api.prediction_service import (
    PredictionService,
)
from secom_mlops.serving.api.schemas import (
    BatchPredictResponse,
    BatchPredictRequest,
    PredictByIdResponse,
    PredictByIdRequest,
)
from secom_mlops.serving.api.model import PredictionEventContext
from secom_mlops.serving.api.errors import ModelGatewayError
from secom_mlops_common.config.serving import (
    ENV_MODEL_GATEWAY_TIMEOUT_SECONDS,
    resolve_model_gateway_url,
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    fast_api_app.state.prediction_event_producer = PredictionEventProducer()
    fast_api_app.state.prediction_event_publisher = BufferedPredictionEventPublisher(
        fast_api_app.state.prediction_event_producer,
        queue_max_size=int(os.getenv("PREDICTION_EVENT_QUEUE_MAX_SIZE", "4096")),
        batch_max_size=int(os.getenv("PREDICTION_EVENT_BATCH_MAX_SIZE", "100")),
        batch_max_wait_seconds=(
            float(os.getenv("PREDICTION_EVENT_BATCH_MAX_WAIT_MS", "100")) / 1000
        ),
    )
    fast_api_app.state.shadow_prediction_event_publisher = BufferedPredictionEventPublisher(
        fast_api_app.state.prediction_event_producer,
        queue_max_size=int(os.getenv("PREDICTION_EVENT_QUEUE_MAX_SIZE", "4096")),
        batch_max_size=int(os.getenv("PREDICTION_EVENT_BATCH_MAX_SIZE", "100")),
        batch_max_wait_seconds=(
                float(os.getenv("PREDICTION_EVENT_BATCH_MAX_WAIT_MS", "100")) / 1000
        ),
    )
    fast_api_app.state.online_snapshot_store = OnlineFeatureSnapshotStore()
    fast_api_app.state.model_gateway_client = ModelGatewayClient(
        base_url=resolve_model_gateway_url(),
        path=os.getenv("PRIMARY_BATCH_PATH", "/invocations"),
        timeout_seconds=float(os.getenv(ENV_MODEL_GATEWAY_TIMEOUT_SECONDS, "10.0")),
    )
    fast_api_app.state.shadow_model_gateway_client = ModelGatewayClient(
        base_url=resolve_model_gateway_url(),
        path=os.getenv("SHADOW_BATCH_PATH", "/shadow/invocations"),
        timeout_seconds=float(os.getenv(ENV_MODEL_GATEWAY_TIMEOUT_SECONDS, "10.0")),
    )

    fast_api_app.state.primary_prediction_batcher = PredictionBatcher(
        client=fast_api_app.state.model_gateway_client,
        event_publisher=fast_api_app.state.prediction_event_publisher,
        max_batch_size=int(os.getenv("MODEL_BATCH_MAX_SIZE", "16")),
        max_wait_seconds=(float(os.getenv("MODEL_BATCH_MAX_WAIT_MS", "20")) / 1000),
        queue_max_size=int(os.getenv("MODEL_BATCH_QUEUE_MAX_SIZE", "1024")),
        queue_timeout_seconds=(float(os.getenv("MODEL_BATCH_QUEUE_TIMEOUT_MS", "2000")) / 1000),
        response_timeout_seconds=float(os.getenv("MODEL_BATCH_RESPONSE_TIMEOUT_SECONDS", "30.0")),
    )
    fast_api_app.state.shadow_prediction_batcher = PredictionBatcher(
        client=fast_api_app.state.shadow_model_gateway_client,
        event_publisher=fast_api_app.state.shadow_prediction_event_publisher,
        max_batch_size=int(os.getenv("MODEL_BATCH_MAX_SIZE", "16")),
        max_wait_seconds=(float(os.getenv("MODEL_BATCH_MAX_WAIT_MS", "20")) / 1000),
        queue_max_size=int(os.getenv("MODEL_BATCH_QUEUE_MAX_SIZE", "1024")),
        queue_timeout_seconds=(float(os.getenv("MODEL_BATCH_QUEUE_TIMEOUT_MS", "2000")) / 1000),
        response_timeout_seconds=float(os.getenv("MODEL_BATCH_RESPONSE_TIMEOUT_SECONDS", "30.0")),
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
        return _snapshot_not_ready_response(snapshot)

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


def _snapshot_not_ready_response(snapshot) -> JSONResponse:
    retry_after_ms = int(os.getenv("PREDICT_PARTIAL_RETRY_AFTER_MS", "200"))
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
