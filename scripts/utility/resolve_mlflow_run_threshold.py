import argparse
import sys
from typing import Any

import httpx

from secom_mlops_common.config.mlflow import (
    DEFAULT_CONTAINER_MLFLOW_TRACKING_URI,
    resolve_tracking_uri,
)

DEFAULT_MLFLOW_TRACKING_URI = DEFAULT_CONTAINER_MLFLOW_TRACKING_URI
RUN_GET_PATH = "/api/2.0/mlflow/runs/get"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def get_json(
        client: httpx.Client,
        path: str,
        params: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = client.get(path, params=params)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            f"MLflow request failed: path={path} status={error.response.status_code} "
            f"body={error.response.text[:1000]}"
        ) from error
    except (httpx.HTTPError, ValueError) as error:
        raise RuntimeError(f"MLflow request failed: path={path} error={error}") from error

    if not isinstance(payload, dict):
        raise RuntimeError(f"MLflow response is not a JSON object: path={path}")

    return payload


def params_to_dict(params: Any) -> dict[str, str]:
    if isinstance(params, dict):
        return {str(key): str(value) for key, value in params.items()}

    if isinstance(params, list):
        result: dict[str, str] = {}
        for item in params:
            if isinstance(item, dict) and "key" in item and "value" in item:
                result[str(item["key"])] = str(item["value"])
        return result

    return {}


def resolve_run_threshold(
        mlflow_tracking_uri: str,
        run_id: str,
        timeout_seconds: float,
) -> float:
    with httpx.Client(
            base_url=mlflow_tracking_uri.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
    ) as client:
        payload = get_json(client, RUN_GET_PATH, {"run_id": run_id})

    run = payload.get("run")
    if not isinstance(run, dict):
        raise RuntimeError(f"MLflow run response missing run: run_id={run_id}")

    data = run.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"MLflow run response missing run.data: run_id={run_id}")

    run_params = params_to_dict(data.get("params"))
    threshold = run_params.get("threshold")

    if threshold is None:
        raise RuntimeError(f"MLflow run param threshold not found: run_id={run_id}")

    return float(threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=resolve_tracking_uri(default=DEFAULT_MLFLOW_TRACKING_URI),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold = resolve_run_threshold(
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )

    log(f"resolved_mlflow_run_threshold run_id={args.run_id} threshold={threshold}")
    print(f"{threshold:.12g}", flush=True)


if __name__ == "__main__":
    main()
