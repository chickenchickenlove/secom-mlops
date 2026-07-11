import asyncio
import math
import os
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from valkey.exceptions import ValkeyError

from secom_mlops.feature_store.online_snapshot_reader import (
    InvalidOnlineFeatureSnapshot,
    OnlineFeatureSnapshotNotFound,
    OnlineFeatureSnapshotStore,
)
from secom_mlops.monitor.prediction_events import PredictionEventProducer
from secom_mlops_common.config.serving import (
    ENV_MODEL_GATEWAY_TIMEOUT_SECONDS,
    resolve_model_gateway_url,
)
from secom_mlops_common.schemas.secom import NUM_FEATURES


class PredictRow(BaseModel):
    sample_id: str = Field(..., pattern=r"^secom-\d{7}$")
    features: list[float | None] = Field(..., min_length=NUM_FEATURES, max_length=NUM_FEATURES)


class BatchPredictRequest(BaseModel):
    rows: list[PredictRow] = Field(..., min_length=1)


class PredictByIdRequest(BaseModel):
    sample_id: str = Field(..., pattern=r"^secom-\d{7}$")


class PredictionItem(BaseModel):
    prediction_id: str
    request_id: str
    sample_id: str
    row_index: int
    fail_probability: float
    prediction: int
    label: str
    threshold: float
    model_uri: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    model_run_id: str
    runtime_slot: str | None = None


class BatchPredictResponse(BaseModel):
    request_id: str
    predictions: list[PredictionItem]


class PredictByIdResponse(BaseModel):
    prediction_id: str
    request_id: str
    sample_id: str
    serving_snapshot_id: str
    snapshot_version: int
    snapshot_time: float
    feature_count: int
    missing_count: int
    fail_probability: float
    prediction: int
    label: str
    threshold: float
    model_uri: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    model_run_id: str
    runtime_slot: str | None = None


class ModelGatewayError(Exception):
    pass


@dataclass
class PendingInvocation:
    features: list[float | None]
    future: asyncio.Future[dict[str, Any]]


class ModelGatewayClient:
    def __init__(self) -> None:
        base_url = resolve_model_gateway_url()
        timeout = float(os.getenv(ENV_MODEL_GATEWAY_TIMEOUT_SECONDS, "10.0"))

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 2.0)),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def invoke_batch(self, inputs: list[list[float | None]]) -> list[dict[str, Any]]:
        try:
            response = await self._client.post("/invocations", json={"inputs": inputs})
        except httpx.RequestError as error:
            raise ModelGatewayError("model gateway unavailable") from error

        if response.status_code >= 400:
            raise ModelGatewayError(
                f"model gateway failed: status={response.status_code} body={response.text[:1000]}"
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise ModelGatewayError("model gateway returned invalid JSON") from error

        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(inputs):
            raise ModelGatewayError("invalid model gateway response")

        return predictions


class ModelGatewayBatcher:
    def __init__(self, client: ModelGatewayClient) -> None:
        self._client = client
        self._max_batch_size = int(os.getenv("MODEL_BATCH_MAX_SIZE", "16"))
        self._max_wait_seconds = float(os.getenv("MODEL_BATCH_MAX_WAIT_MS", "20")) / 1000
        self._queue_timeout_seconds = float(os.getenv("MODEL_BATCH_QUEUE_TIMEOUT_MS", "2000")) / 1000
        self._response_timeout_seconds = float(os.getenv("MODEL_BATCH_RESPONSE_TIMEOUT_SECONDS", "30.0"))
        self._queue = asyncio.Queue(
            maxsize=int(os.getenv("MODEL_BATCH_QUEUE_MAX_SIZE", "1024"))
        )
        self._worker_task: asyncio.Task | None = None

        if self._max_batch_size < 1:
            raise ValueError("MODEL_BATCH_MAX_SIZE must be >= 1")

    def start(self) -> None:
        self._worker_task = asyncio.create_task(self._run())

    async def close(self) -> None:
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

    async def invoke_many(self, inputs: list[list[float | None]]) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        pending_items = [
            PendingInvocation(features=features, future=loop.create_future())
            for features in inputs
        ]

        try:
            for pending in pending_items:
                await asyncio.wait_for(
                    self._queue.put(pending),
                    timeout=self._queue_timeout_seconds,
                )
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

                deadline = asyncio.get_running_loop().time() + self._max_wait_seconds

                while len(batch) < self._max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
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

    async def _flush(self, batch: list[PendingInvocation]) -> None:
        active = [pending for pending in batch if not pending.future.done()]

        if not active:
            for pending in batch:
                self._queue.task_done()
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
                if not pending.future.done():
                    pending.future.set_result(prediction)
        finally:
            for pending in batch:
                self._queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_event_producer = PredictionEventProducer()
    app.state.online_snapshot_store = OnlineFeatureSnapshotStore()
    app.state.model_gateway_client = ModelGatewayClient()
    app.state.model_gateway_batcher = ModelGatewayBatcher(app.state.model_gateway_client)
    app.state.model_gateway_batcher.start()

    yield

    app.state.prediction_event_producer.close()
    app.state.online_snapshot_store.close()
    await app.state.model_gateway_batcher.close()
    await app.state.model_gateway_client.close()


app = FastAPI(title="SECOM Fail Detection API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=BatchPredictResponse)
async def predict(payload: BatchPredictRequest, request: Request):
    request_id = str(uuid4())
    inputs = [row.features for row in payload.rows]

    try:
        raw_predictions = await request.app.state.model_gateway_batcher.invoke_many(inputs)
        predictions = [
            _normalize_prediction(raw_prediction, row_index=row_index)
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

    started_at = time.perf_counter()
    try:
        raw_predictions = await request.app.state.model_gateway_batcher.invoke_many([snapshot.values])
        prediction = _normalize_prediction(raw_predictions[0], row_index=0)
    except ModelGatewayError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    predicted_at = time.time()
    latency_ms = (time.perf_counter() - started_at) * 1000

    try:
        await asyncio.to_thread(
            request.app.state.prediction_event_producer.publish_many,
            [_build_snapshot_prediction_event(
                prediction_id=prediction_id,
                request_id=request_id,
                sample_id=snapshot.sample_id,
                serving_snapshot_id=snapshot.serving_snapshot_id,
                snapshot_version=snapshot.snapshot_version,
                prediction=prediction,
                predicted_at=predicted_at,
                missing_count=snapshot.missing_count,
                latency_ms=latency_ms,
            )],
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {
        "prediction_id": prediction_id,
        "request_id": request_id,
        "sample_id": snapshot.sample_id,
        "serving_snapshot_id": snapshot.serving_snapshot_id,
        "snapshot_version": snapshot.snapshot_version,
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


def _normalize_prediction(raw: dict[str, Any], row_index: int) -> dict[str, Any]:
    try:
        model_run_id = raw.get("model_run_id")
        threshold = raw.get("threshold")

        if not model_run_id:
            raise ModelGatewayError("model gateway response missing model_run_id")
        if threshold is None:
            raise ModelGatewayError("model gateway response missing threshold")

        return {
            "row_index": row_index,
            "fail_probability": float(raw["fail_probability"]),
            "prediction": int(raw["prediction"]),
            "label": str(raw["label"]),
            "threshold": float(threshold),
            "model_uri": raw.get("model_uri"),
            "model_name": raw.get("model_name"),
            "model_version": raw.get("model_version"),
            "model_alias": raw.get("model_alias"),
            "model_run_id": str(model_run_id),
            "runtime_slot": raw.get("runtime_slot"),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ModelGatewayError(f"invalid model gateway prediction: {error}") from error


def _build_snapshot_prediction_event(
        prediction_id: str,
        request_id: str,
        sample_id: str,
        serving_snapshot_id: str,
        snapshot_version: int,
        prediction: dict[str, Any],
        predicted_at: float,
        missing_count: int,
        latency_ms: float,
) -> dict[str, Any]:
    return {
        "prediction_id": prediction_id,
        "request_id": request_id,
        "sample_id": sample_id,
        "serving_snapshot_id": serving_snapshot_id,
        "snapshot_version": snapshot_version,
        "model_run_id": prediction["model_run_id"],
        "model_name": prediction.get("model_name"),
        "model_version": prediction.get("model_version"),
        "model_alias": prediction.get("model_alias"),
        "model_uri": prediction.get("model_uri"),
        "runtime_slot": prediction.get("runtime_slot") or "unknown",
        "predicted_at": predicted_at,
        "fail_probability": prediction["fail_probability"],
        "predicted_value": prediction["prediction"],
        "predicted_label": prediction["label"],
        "threshold": prediction["threshold"],
        "missing_count": missing_count,
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
