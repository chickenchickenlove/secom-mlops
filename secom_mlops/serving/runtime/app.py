import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from mlflow.tracking import MlflowClient

from secom_mlops.monitor.deployments import (
    insert_runtime_deployment_state_if_missing,
)
from secom_mlops.serving.runtime.config import ModelRuntimeConfig
from secom_mlops.serving.runtime.model import (
    ModelRuntime,
)
from secom_mlops.serving.runtime.schemas import (
    InvocationRequest,
    ThresholdReloadRequest,
    ModelVersionReloadRequest,
)
from secom_mlops_common.logging import configure_logging, get_logger
from secom_mlops_common.schemas.secom import MODEL_COLUMNS, NUM_FEATURES

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    configure_logging()
    config = ModelRuntimeConfig()
    fast_api_app.state.config = config
    fast_api_app.state.runtime_lock = threading.RLock()
    fast_api_app.state.runtime = load_runtime(config)
    seed_runtime_deployment_state_if_missing(
        fast_api_app.state.runtime,
        config.monitoring_database_url,
    )
    yield


app = FastAPI(
    title="SECOM Model Server",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metadata")
async def metadata(request: Request):
    runtime = _get_runtime(request)
    return _runtime_metadata(runtime)


@app.post("/admin/reload-threshold")
async def reload_threshold(payload: ThresholdReloadRequest, request: Request):
    with request.app.state.runtime_lock:
        current_runtime: ModelRuntime = request.app.state.runtime
        _validate_current_runtime(
            current_runtime,
            expected_model_name=payload.expected_model_name,
            expected_model_version=payload.expected_model_version,
            expected_run_id=payload.expected_run_id,
        )

        next_runtime = replace(
            current_runtime,
            threshold=float(payload.threshold),
            loaded_at_utc=_utc_now(),
            reload_request_id=payload.request_id,
        )
        request.app.state.runtime = next_runtime

    logger.info(
        "model_server_threshold_reloaded "
        "runtime_slot=%s "
        "model_name=%s "
        "model_version=%s "
        "model_run_id=%s "
        "previous_threshold=%s "
        "new_threshold=%s "
        "request_id=%s",
        next_runtime.runtime_slot,
        next_runtime.model_name,
        next_runtime.model_version,
        next_runtime.model_run_id,
        current_runtime.threshold,
        next_runtime.threshold,
        payload.request_id,
    )

    return {
        "status": "ok",
        "reload_type": "threshold",
        "previous": _runtime_metadata(current_runtime),
        "current": _runtime_metadata(next_runtime),
    }


@app.post("/admin/reload-model-version")
async def reload_model_version(payload: ModelVersionReloadRequest, request: Request):
    config: ModelRuntimeConfig = request.app.state.config

    with request.app.state.runtime_lock:
        current_runtime: ModelRuntime = request.app.state.runtime
        _validate_current_runtime(
            current_runtime,
            expected_model_name=payload.model_name,
            expected_model_version=payload.expected_current_model_version,
            expected_run_id=payload.expected_current_run_id,
        )

    model_name = payload.model_name or current_runtime.model_name

    next_runtime = load_runtime_for_model_version(
        config,
        model_name=model_name,
        model_version=payload.model_version,
        threshold_override=payload.threshold,
        expected_run_id=payload.expected_run_id,
        reload_request_id=payload.request_id,
    )

    with request.app.state.runtime_lock:
        request.app.state.runtime = next_runtime

    logger.info(
        "model_server_model_version_reloaded "
        "runtime_slot=%s "
        "model_name=%s "
        "model_version=%s "
        "model_run_id=%s "
        "threshold=%s "
        "request_id=%s",
        next_runtime.runtime_slot,
        next_runtime.model_name,
        next_runtime.model_version,
        next_runtime.model_run_id,
        next_runtime.threshold,
        payload.request_id,
    )

    return {
        "status": "ok",
        "reload_type": "model_version",
        "previous": _runtime_metadata(current_runtime),
        "current": _runtime_metadata(next_runtime),
    }


@app.post("/invocations")
async def invocations(payload: InvocationRequest, request: Request):
    runtime = _get_runtime(request)

    if any(len(row) != NUM_FEATURES for row in payload.inputs):
        raise HTTPException(
            status_code=422,
            detail=f"Expected each input row to have {NUM_FEATURES} features",
        )

    features = pd.DataFrame(
        payload.inputs,
        columns=list(MODEL_COLUMNS),
        dtype="float64",
    )

    started_at = time.perf_counter()
    try:
        prediction_frame = runtime.model.predict(features)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    latency_ms = (time.perf_counter() - started_at) * 1000

    if not isinstance(prediction_frame, pd.DataFrame):
        prediction_frame = pd.DataFrame(prediction_frame)

    predictions = []
    for row in prediction_frame.to_dict(orient="records"):
        prediction = {
            key: _jsonable(value)
            for key, value in row.items()
        }

        prediction = _apply_runtime_threshold(prediction, runtime)

        prediction.update({
            "model_uri": runtime.model_uri,
            "model_name": runtime.model_name,
            "model_version": runtime.model_version,
            "model_alias": runtime.model_alias,
            "model_run_id": runtime.model_run_id,
            "threshold": runtime.threshold,
            "runtime_slot": runtime.runtime_slot,
        })
        predictions.append(prediction)

    return {
        "predictions": predictions,
        "latency_ms": latency_ms,
    }


def load_runtime(config: ModelRuntimeConfig) -> ModelRuntime:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()

    resolved_alias = None
    resolved_version = None
    resolved_run_id = None

    if config.ml_model_uri:
        resolved_model_uri = config.ml_model_uri
        if config.ml_model_version:
            version = client.get_model_version(
                config.ml_model_name,
                config.ml_model_version,
            )
            resolved_version = version.version
            resolved_run_id = version.run_id
        elif config.ml_model_alias:
            version = client.get_model_version_by_alias(
                config.ml_model_name,
                config.ml_model_alias,
            )
            resolved_alias = config.ml_model_alias
            resolved_version = version.version
            resolved_run_id = version.run_id
    elif config.ml_model_version:
        version = client.get_model_version(
            config.ml_model_name,
            config.ml_model_version,
        )
        resolved_model_uri = f"models:/{config.ml_model_name}/{version.version}"
        resolved_version = version.version
        resolved_run_id = version.run_id
    else:
        version = client.get_model_version_by_alias(
            config.ml_model_name,
            config.ml_model_alias,
        )
        resolved_model_uri = (
            f"models:/{config.ml_model_name}@{config.ml_model_alias}"
        )
        resolved_alias = config.ml_model_alias
        resolved_version = version.version
        resolved_run_id = version.run_id

    threshold = _load_threshold(client, resolved_run_id)
    return _load_runtime_from_uri(
        model_uri=resolved_model_uri,
        model_name=config.ml_model_name,
        model_version=resolved_version,
        model_alias=resolved_alias,
        model_run_id=resolved_run_id,
        threshold=threshold,
        runtime_slot=config.model_runtime_slot,
        reload_request_id=None,
    )


def load_runtime_for_model_version(
        config: ModelRuntimeConfig,
        model_name: str,
        model_version: str,
        threshold_override: float | None,
        expected_run_id: str | None,
        reload_request_id: str | None,
) -> ModelRuntime:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()

    version = client.get_model_version(model_name, model_version)
    resolved_version = version.version
    resolved_run_id = version.run_id

    if expected_run_id and resolved_run_id != expected_run_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "Resolved model version run_id did not match expected_run_id: "
                f"model_name={model_name} "
                f"model_version={resolved_version} "
                f"actual_run_id={resolved_run_id} "
                f"expected_run_id={expected_run_id}"
            ),
        )

    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else _load_threshold(client, resolved_run_id)
    )

    return _load_runtime_from_uri(
        model_uri=f"models:/{model_name}/{resolved_version}",
        model_name=model_name,
        model_version=resolved_version,
        model_alias=None,
        model_run_id=resolved_run_id,
        threshold=threshold,
        runtime_slot=config.model_runtime_slot,
        reload_request_id=reload_request_id,
    )


def seed_runtime_deployment_state_if_missing(
        runtime: ModelRuntime,
        database_url: str,
) -> None:
    if (
        runtime.model_version is None
        or runtime.model_run_id is None
        or runtime.threshold is None
    ):
        logger.info(
            "model_server_runtime_state_bootstrap_skipped "
            "runtime_slot=%s "
            "model_name=%s "
            "model_version=%s "
            "model_run_id=%s "
            "threshold=%s "
            "reason=incomplete_runtime_metadata",
            runtime.runtime_slot,
            runtime.model_name,
            runtime.model_version,
            runtime.model_run_id,
            runtime.threshold,
        )
        return

    try:
        runtime_state = insert_runtime_deployment_state_if_missing(
            database_url,
            model_name=runtime.model_name,
            runtime_slot=runtime.runtime_slot,
            target_alias=runtime.model_alias or "champion",
            active_request_id=None,
            active_model_version=runtime.model_version,
            active_model_run_id=runtime.model_run_id,
            active_threshold=runtime.threshold,
            previous_request_id=None,
            previous_model_version=None,
            previous_model_run_id=None,
            previous_threshold=None,
            last_operation="bootstrap",
            last_operation_request_id=None,
        )
    except Exception:
        logger.warning(
            "model_server_runtime_state_bootstrap_failed "
            "runtime_slot=%s "
            "model_name=%s "
            "model_version=%s "
            "model_run_id=%s "
            "threshold=%s",
            runtime.runtime_slot,
            runtime.model_name,
            runtime.model_version,
            runtime.model_run_id,
            runtime.threshold,
            exc_info=True,
        )
        return

    if runtime_state is None:
        logger.info(
            "model_server_runtime_state_bootstrap_skipped "
            "runtime_slot=%s "
            "model_name=%s "
            "reason=state_exists",
            runtime.runtime_slot,
            runtime.model_name,
        )
        return

    logger.info(
        "model_server_runtime_state_bootstrapped "
        "runtime_slot=%s "
        "model_name=%s "
        "active_model_version=%s "
        "active_model_run_id=%s "
        "active_threshold=%s "
        "last_operation=%s",
        runtime_state.get("runtime_slot"),
        runtime_state.get("model_name"),
        runtime_state.get("active_model_version"),
        runtime_state.get("active_model_run_id"),
        runtime_state.get("active_threshold"),
        runtime_state.get("last_operation"),
    )


def _load_runtime_from_uri(
        model_uri: str,
        model_name: str,
        model_version: str | None,
        model_alias: str | None,
        model_run_id: str | None,
        threshold: float | None,
        runtime_slot: str,
        reload_request_id: str | None,
) -> ModelRuntime:
    model = mlflow.pyfunc.load_model(model_uri)
    runtime = ModelRuntime(
        model=model,
        model_uri=model_uri,
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
        model_run_id=model_run_id,
        threshold=threshold,
        runtime_slot=runtime_slot,
        loaded_at_utc=_utc_now(),
        reload_request_id=reload_request_id,
    )

    logger.info(
        "model_server_loaded "
        "runtime_slot=%s "
        "model_uri=%s "
        "model_name=%s "
        "model_version=%s "
        "model_alias=%s "
        "model_run_id=%s "
        "threshold=%s "
        "reload_request_id=%s",
        runtime.runtime_slot,
        runtime.model_uri,
        runtime.model_name,
        runtime.model_version,
        runtime.model_alias,
        runtime.model_run_id,
        runtime.threshold,
        runtime.reload_request_id,
    )

    return runtime


def _load_threshold(client: MlflowClient, run_id: str | None) -> float | None:
    if not run_id:
        return None

    run = client.get_run(run_id)
    value = run.data.params.get("threshold")
    if value is None:
        return None

    return float(value)


def _apply_runtime_threshold(
        prediction: dict[str, Any],
        runtime: ModelRuntime,
) -> dict[str, Any]:
    fail_probability = prediction.get("fail_probability")

    if fail_probability is None or runtime.threshold is None:
        return prediction

    fail_probability = float(fail_probability)
    predicted_value = 1 if fail_probability >= runtime.threshold else -1

    prediction["prediction"] = predicted_value
    prediction["label"] = "fail" if predicted_value == 1 else "pass"
    prediction["threshold"] = runtime.threshold

    return prediction


def _validate_current_runtime(
        runtime: ModelRuntime,
        expected_model_name: str | None,
        expected_model_version: str | None,
        expected_run_id: str | None,
) -> None:
    if expected_model_name and runtime.model_name != expected_model_name:
        raise HTTPException(
            status_code=409,
            detail=(
                "Current model_name did not match expectation: "
                f"actual={runtime.model_name} expected={expected_model_name}"
            ),
        )

    if expected_model_version and runtime.model_version != expected_model_version:
        raise HTTPException(
            status_code=409,
            detail=(
                "Current model_version did not match expectation: "
                f"actual={runtime.model_version} expected={expected_model_version}"
            ),
        )

    if expected_run_id and runtime.model_run_id != expected_run_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "Current model_run_id did not match expectation: "
                f"actual={runtime.model_run_id} expected={expected_run_id}"
            ),
        )


def _get_runtime(request: Request) -> ModelRuntime:
    return request.app.state.runtime


def _runtime_metadata(runtime: ModelRuntime) -> dict[str, Any]:
    return {
        "model_uri": runtime.model_uri,
        "model_name": runtime.model_name,
        "model_version": runtime.model_version,
        "model_alias": runtime.model_alias,
        "model_run_id": runtime.model_run_id,
        "threshold": runtime.threshold,
        "runtime_slot": runtime.runtime_slot,
        "loaded_at_utc": runtime.loaded_at_utc,
        "reload_request_id": runtime.reload_request_id,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        if np.isnan(value):
            return None
        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=28091)
