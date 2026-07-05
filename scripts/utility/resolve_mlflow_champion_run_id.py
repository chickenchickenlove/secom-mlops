import argparse
import sys
from typing import Any

import httpx

from secom_mlops_common.config.mlflow import (
    DEFAULT_CHAMPION_ALIAS,
    DEFAULT_CONTAINER_MLFLOW_TRACKING_URI,
    DEFAULT_MODEL_NAME as COMMON_DEFAULT_MODEL_NAME,
    resolve_model_alias,
    resolve_model_name,
    resolve_tracking_uri,
)

DEFAULT_MLFLOW_TRACKING_URI = DEFAULT_CONTAINER_MLFLOW_TRACKING_URI
DEFAULT_MODEL_NAME = COMMON_DEFAULT_MODEL_NAME
DEFAULT_MODEL_ALIAS = DEFAULT_CHAMPION_ALIAS
DEFAULT_SKIP_EXIT_CODE = 99

MODEL_VERSION_BY_ALIAS_PATH = "/api/2.0/mlflow/registered-models/alias"
RUN_GET_PATH = "/api/2.0/mlflow/runs/get"


class ResolutionSkipped(Exception):
    pass


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def get_json(
        client: httpx.Client,
        path: str,
        params: dict[str, Any],
        not_found_skip_message: str | None = None,
) -> dict[str, Any]:
    try:
        response = client.get(path, params=params)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 404 and not_found_skip_message is not None:
            raise ResolutionSkipped(not_found_skip_message) from error
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


def load_run_threshold(client: httpx.Client, run_id: str) -> float:
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


def resolve_champion_run_id(
        mlflow_tracking_uri: str,
        model_name: str,
        model_alias: str,
        timeout_seconds: float,
) -> str:
    with httpx.Client(
            base_url=mlflow_tracking_uri.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
    ) as client:
        payload = get_json(
            client,
            MODEL_VERSION_BY_ALIAS_PATH,
            {"name": model_name, "alias": model_alias},
            not_found_skip_message=(
                f"model alias not found in MLflow: model_name={model_name} alias={model_alias}"
            ),
        )

        model_version = payload.get("model_version")
        if not isinstance(model_version, dict):
            raise RuntimeError(
                f"MLflow alias response missing model_version: model_name={model_name} alias={model_alias}"
            )

        run_id = model_version.get("run_id")
        if not run_id:
            raise ResolutionSkipped(
                f"MLflow model version has no run_id: model_name={model_name} alias={model_alias}"
            )

        threshold = load_run_threshold(client, str(run_id))

    log(
        "resolved_mlflow_champion_run "
        f"model_name={model_name} "
        f"model_alias={model_alias} "
        f"model_version={model_version.get('version')} "
        f"run_id={run_id} "
        f"threshold={threshold}"
    )

    return str(run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=resolve_tracking_uri(default=DEFAULT_MLFLOW_TRACKING_URI),
    )
    parser.add_argument(
        "--model-name",
        default=resolve_model_name(default=DEFAULT_MODEL_NAME),
    )
    parser.add_argument(
        "--model-alias",
        default=resolve_model_alias(default=DEFAULT_MODEL_ALIAS),
    )
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--skip-exit-code", type=int, default=DEFAULT_SKIP_EXIT_CODE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        run_id = resolve_champion_run_id(
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            model_name=args.model_name,
            model_alias=args.model_alias,
            timeout_seconds=args.timeout_seconds,
        )
    except ResolutionSkipped as error:
        log(f"mlflow_champion_run_resolution_skipped reason={error}")
        raise SystemExit(args.skip_exit_code)

    # Keep stdout clean for Airflow XCom.
    print(run_id, flush=True)


if __name__ == "__main__":
    main()
