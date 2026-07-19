from typing import Any

from secom_mlops.serving.api.errors import ModelGatewayError
from secom_mlops.serving.api.model import PredictionEventContext


def normalize_prediction(raw: Any, row_index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ModelGatewayError("model gateway prediction must be an object.")

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


def build_snapshot_prediction_event(
        prediction_id: str,
        request_id: str,
        sample_id: str,
        serving_snapshot_id: str,
        snapshot_version: int,
        feature_hash: str,
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
        "feature_hash": feature_hash,
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


def build_prediction_event(
    context: PredictionEventContext,
    prediction: dict[str, Any],
    predicted_at: float,
    latency_ms: float,
) -> dict[str, Any]:
    return build_snapshot_prediction_event(
        prediction_id=context.prediction_id,
        request_id=context.request_id,
        sample_id=context.sample_id,
        serving_snapshot_id=context.serving_snapshot_id,
        snapshot_version=context.snapshot_version,
        feature_hash=context.feature_hash,
        prediction=prediction,
        predicted_at=predicted_at,
        missing_count=context.missing_count,
        latency_ms=latency_ms,
    )
