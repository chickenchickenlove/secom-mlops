from __future__ import annotations

import argparse
import time
from typing import Any

import httpx
import mlflow
from mlflow.tracking import MlflowClient

from secom_mlops.monitor.deployments import (
    append_deployment_request_note,
    get_deployment_request,
    insert_runtime_reload_event,
    mark_deployment_request_rollout_status,
    normalize_request_id,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CHAMPION_ALIAS,
    resolve_model_name,
    resolve_tracking_uri,
)
from secom_mlops_common.config.serving import (
    CANARY_METADATA_PATH,
    CANARY_RELOAD_MODEL_VERSION_PATH,
    TRAFFIC_POLICY_PATH,
    model_gateway_admin_endpoint,
    model_gateway_endpoint,
)

ROLLBACK_ELIGIBLE_ROLLOUT_STATUSES = {
    "promoted",
    "canary_reloading",
    "canary_ready",
    "release_reloading",
    "failed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--target-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--request-id", required=True)
    parser.add_argument(
        "--traffic-policy-url",
        default=model_gateway_admin_endpoint(TRAFFIC_POLICY_PATH),
    )
    parser.add_argument(
        "--canary-reload-url",
        default=model_gateway_endpoint(CANARY_RELOAD_MODEL_VERSION_PATH),
    )
    parser.add_argument(
        "--canary-metadata-url",
        default=model_gateway_endpoint(CANARY_METADATA_PATH),
    )
    parser.add_argument("--service-name", default="model-server-canary")
    parser.add_argument("--runtime-slot", default="canary")
    parser.add_argument("--http-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--reload-canary-to-previous", action="store_true")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_deployment_request(args: argparse.Namespace) -> dict[str, Any]:
    request = get_deployment_request(args.request_id)
    if request is None:
        raise RuntimeError(f"deployment request not found: request_id={args.request_id}")
    return request


def validate_deployment_request(
        request: dict[str, Any],
        *,
        model_name: str,
        target_alias: str,
) -> None:
    if request["model_name"] != model_name:
        raise RuntimeError(
            "deployment request model_name mismatch: "
            f"request_id={request['request_id']} "
            f"request_model_name={request['model_name']} "
            f"model_name={model_name}"
        )

    if request["target_alias"] != target_alias:
        raise RuntimeError(
            "deployment request target_alias mismatch: "
            f"request_id={request['request_id']} "
            f"request_target_alias={request['target_alias']} "
            f"target_alias={target_alias}"
        )

    if request["approval_status"] != "approved":
        raise RuntimeError(
            "deployment request approval_status is not approved: "
            f"request_id={request['request_id']} approval_status={request['approval_status']}"
        )

    if request["rollout_status"] not in ROLLBACK_ELIGIBLE_ROLLOUT_STATUSES:
        raise RuntimeError(
            "deployment request is not rollback-eligible: "
            f"request_id={request['request_id']} rollout_status={request['rollout_status']} "
            f"allowed={sorted(ROLLBACK_ELIGIBLE_ROLLOUT_STATUSES)}"
        )


def validate_previous_version(
        client: MlflowClient,
        *,
        model_name: str,
        previous_version: str,
        previous_run_id: str,
) -> None:
    version = client.get_model_version(model_name, previous_version)
    if str(version.run_id) != str(previous_run_id):
        raise RuntimeError(
            "deployment request previous_run_id does not match MLflow model version: "
            f"model_name={model_name} "
            f"previous_version={previous_version} "
            f"request_previous_run_id={previous_run_id} "
            f"mlflow_previous_run_id={version.run_id}"
        )


def load_run_threshold(client: MlflowClient, run_id: str) -> float:
    run = client.get_run(run_id)
    value = run.data.params.get("threshold")
    if value is None:
        raise RuntimeError(f"MLflow run param threshold not found: run_id={run_id}")
    return float(value)


def get_json(url: str, timeout: float) -> dict[str, Any]:
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"HTTP response is not a JSON object: url={url}")
    return payload


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    parsed = response.json()
    if not isinstance(parsed, dict):
        raise RuntimeError(f"HTTP response is not a JSON object: url={url}")
    return parsed


def try_get_metadata(url: str, timeout: float) -> dict[str, Any]:
    try:
        return get_json(url, timeout)
    except Exception as error:
        return {
            "metadata_error": str(error),
        }


def validate_runtime_metadata(
        metadata: dict[str, Any],
        *,
        model_name: str,
        model_version: str,
        model_run_id: str,
        threshold: float,
        runtime_slot: str,
) -> None:
    errors = []

    if metadata.get("model_name") != model_name:
        errors.append(f"model_name actual={metadata.get('model_name')} expected={model_name}")

    if str(metadata.get("model_version")) != str(model_version):
        errors.append(f"model_version actual={metadata.get('model_version')} expected={model_version}")

    if str(metadata.get("model_run_id")) != str(model_run_id):
        errors.append(f"model_run_id actual={metadata.get('model_run_id')} expected={model_run_id}")

    if metadata.get("runtime_slot") != runtime_slot:
        errors.append(f"runtime_slot actual={metadata.get('runtime_slot')} expected={runtime_slot}")

    actual_threshold = metadata.get("threshold")
    if actual_threshold is None or abs(float(actual_threshold) - float(threshold)) > 1e-12:
        errors.append(f"threshold actual={actual_threshold} expected={threshold}")

    if errors:
        raise RuntimeError("canary rollback metadata validation failed: " + "; ".join(errors))


def previous_from_payload(
        reload_payload: dict[str, Any] | None,
        fallback_metadata: dict[str, Any],
) -> dict[str, Any]:
    if reload_payload is not None:
        previous = reload_payload.get("previous")
        if isinstance(previous, dict):
            return previous
    return fallback_metadata


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def record_reload_event(
        *,
        request: dict[str, Any],
        args: argparse.Namespace,
        reload_status: str,
        started_at: float,
        completed_at: float,
        previous_version: str,
        previous_run_id: str,
        previous_threshold: float,
        previous_metadata: dict[str, Any],
        metadata: dict[str, Any],
        error_message: str | None = None,
) -> None:
    insert_runtime_reload_event(
        request_id=request["request_id"],
        service_name=args.service_name,
        runtime_slot=args.runtime_slot,
        model_name=args.model_name,
        previous_model_version=none_or_str(previous_metadata.get("model_version")),
        previous_model_run_id=none_or_str(previous_metadata.get("model_run_id")),
        previous_threshold=float_or_none(previous_metadata.get("threshold")),
        new_model_version=previous_version,
        new_model_run_id=previous_run_id,
        new_threshold=previous_threshold,
        reload_status=reload_status,
        error_message=error_message,
        metadata=metadata,
        started_at=started_at,
        completed_at=completed_at,
    )


def set_canary_traffic_zero(
        *,
        request_id: str,
        traffic_policy_url: str,
        timeout: float,
) -> dict[str, Any]:
    return post_json(
        traffic_policy_url,
        {
            "request_id": request_id,
            "canary_percent": 0,
            "dry_run": False,
        },
        timeout,
    )


def main() -> None:
    args = parse_args()
    args.request_id = normalize_request_id(args.request_id)
    tracking_uri = resolve_tracking_uri(args.tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    request = resolve_deployment_request(args)
    validate_deployment_request(
        request,
        model_name=args.model_name,
        target_alias=args.target_alias,
    )

    previous_version = none_or_str(request.get("previous_version"))
    previous_run_id = none_or_str(request.get("previous_run_id"))
    previous_threshold = None

    if args.reload_canary_to_previous:
        if previous_version is None or previous_run_id is None:
            raise RuntimeError(
                "deployment request previous version is required to reload canary: "
                f"request_id={request['request_id']} "
                f"previous_version={previous_version} previous_run_id={previous_run_id}"
            )

        validate_previous_version(
            client,
            model_name=args.model_name,
            previous_version=previous_version,
            previous_run_id=previous_run_id,
        )
        previous_threshold = load_run_threshold(client, previous_run_id)

    if args.dry_run:
        print(
            "candidate_canary_rollback_dry_run "
            f"request_id={request['request_id']} "
            f"model_name={args.model_name} "
            f"source_version={request['source_version']} "
            f"previous_version={previous_version} "
            f"previous_run_id={previous_run_id} "
            f"previous_threshold={previous_threshold} "
            f"rollout_status={request['rollout_status']} "
            f"traffic_policy_url={args.traffic_policy_url} "
            f"reload_canary_to_previous={args.reload_canary_to_previous} "
            f"canary_reload_url={args.canary_reload_url} "
            f"canary_metadata_url={args.canary_metadata_url}"
        )
        return

    traffic_policy_response = set_canary_traffic_zero(
        request_id=request["request_id"],
        traffic_policy_url=args.traffic_policy_url,
        timeout=args.http_timeout_seconds,
    )

    reload_payload: dict[str, Any] | None = None
    canary_metadata: dict[str, Any] | None = None

    if args.reload_canary_to_previous:
        assert previous_version is not None
        assert previous_run_id is not None
        assert previous_threshold is not None

        started_at = time.time()
        previous_metadata = try_get_metadata(args.canary_metadata_url, args.http_timeout_seconds)

        try:
            reload_payload = post_json(
                args.canary_reload_url,
                {
                    "request_id": request["request_id"],
                    "model_name": args.model_name,
                    "model_version": previous_version,
                    "expected_run_id": previous_run_id,
                    "threshold": previous_threshold,
                },
                args.http_timeout_seconds,
            )

            canary_metadata = get_json(args.canary_metadata_url, args.http_timeout_seconds)
            validate_runtime_metadata(
                canary_metadata,
                model_name=args.model_name,
                model_version=previous_version,
                model_run_id=previous_run_id,
                threshold=previous_threshold,
                runtime_slot=args.runtime_slot,
            )

            completed_at = time.time()
            record_reload_event(
                request=request,
                args=args,
                reload_status="succeeded",
                started_at=started_at,
                completed_at=completed_at,
                previous_version=previous_version,
                previous_run_id=previous_run_id,
                previous_threshold=previous_threshold,
                previous_metadata=previous_from_payload(reload_payload, previous_metadata),
                metadata={
                    "reload_response": reload_payload,
                    "metadata": canary_metadata,
                },
            )
        except Exception as error:
            completed_at = time.time()
            record_reload_event(
                request=request,
                args=args,
                reload_status="failed",
                started_at=started_at,
                completed_at=completed_at,
                previous_version=previous_version,
                previous_run_id=previous_run_id,
                previous_threshold=previous_threshold,
                previous_metadata=previous_from_payload(reload_payload, previous_metadata),
                metadata={
                    "reload_response": reload_payload,
                    "canary_metadata": try_get_metadata(args.canary_metadata_url, args.http_timeout_seconds),
                },
                error_message=str(error),
            )
            raise

    if args.notes:
        append_deployment_request_note(request["request_id"], args.notes)

    updated_request = mark_deployment_request_rollout_status(
        request["request_id"],
        "rolled_back",
    )

    print(
        "candidate_canary_rollback_complete "
        f"request_id={updated_request['request_id']} "
        f"model_name={args.model_name} "
        f"source_version={request['source_version']} "
        f"previous_version={previous_version} "
        f"traffic_canary_percent={traffic_policy_response.get('canary_percent')} "
        f"reload_canary_to_previous={args.reload_canary_to_previous} "
        f"canary_runtime_slot={canary_metadata.get('runtime_slot') if canary_metadata else None} "
        f"rollout_status={updated_request['rollout_status']}"
    )


if __name__ == "__main__":
    main()
