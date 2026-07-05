from __future__ import annotations

import argparse
import time
from typing import Any

import httpx
import mlflow
from mlflow.tracking import MlflowClient

from secom_mlops.monitor.deployments import (
    TERMINAL_ROLLOUT_STATUSES,
    find_next_approved_deployment_request,
    get_deployment_request,
    get_runtime_deployment_state,
    insert_runtime_reload_event,
    mark_deployment_request_rollout_status,
    normalize_request_id,
    upsert_runtime_deployment_state,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CHAMPION_ALIAS,
    resolve_model_name,
    resolve_tracking_uri,
)
from secom_mlops_common.config.serving import (
    CANARY_METADATA_PATH,
    CANARY_RELOAD_MODEL_VERSION_PATH,
    model_gateway_endpoint,
)

CANARY_DEPLOYABLE_ROLLOUT_STATUSES = {
    "not_started",
    "promoted",
    "canary_reloading",
    "canary_ready",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--target-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--request-id", default=None)
    parser.add_argument(
        "--reload-url",
        default=model_gateway_endpoint(CANARY_RELOAD_MODEL_VERSION_PATH),
    )
    parser.add_argument(
        "--metadata-url",
        default=model_gateway_endpoint(CANARY_METADATA_PATH),
    )
    parser.add_argument("--service-name", default="model-server-canary")
    parser.add_argument("--runtime-slot", default="canary")
    parser.add_argument("--http-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_deployment_request(args: argparse.Namespace) -> dict[str, Any]:
    if args.request_id:
        request = get_deployment_request(args.request_id)
        if request is None:
            raise RuntimeError(f"deployment request not found: request_id={args.request_id}")
        return request

    request = find_next_approved_deployment_request(
        model_name=args.model_name,
        target_alias=args.target_alias,
    )
    if request is None:
        raise RuntimeError(
            "approved deployment request not found: "
            f"model_name={args.model_name} target_alias={args.target_alias}"
        )
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

    if request["eval_status"] != "passed":
        raise RuntimeError(
            "deployment request eval_status is not passed: "
            f"request_id={request['request_id']} eval_status={request['eval_status']}"
        )

    if request["approval_status"] != "approved":
        raise RuntimeError(
            "deployment request approval_status is not approved: "
            f"request_id={request['request_id']} approval_status={request['approval_status']}"
        )

    if request["rollout_status"] in TERMINAL_ROLLOUT_STATUSES:
        raise RuntimeError(
            "deployment request is terminal: "
            f"request_id={request['request_id']} rollout_status={request['rollout_status']}"
        )

    if request["rollout_status"] not in CANARY_DEPLOYABLE_ROLLOUT_STATUSES:
        raise RuntimeError(
            "deployment request is not canary deployable: "
            f"request_id={request['request_id']} rollout_status={request['rollout_status']} "
            f"allowed={sorted(CANARY_DEPLOYABLE_ROLLOUT_STATUSES)}"
        )


def validate_mlflow_source(
        client: MlflowClient,
        *,
        model_name: str,
        source_version: str,
        source_run_id: str,
) -> None:
    version = client.get_model_version(model_name, source_version)
    if str(version.run_id) != str(source_run_id):
        raise RuntimeError(
            "deployment request source_run_id does not match MLflow model version: "
            f"model_name={model_name} "
            f"source_version={source_version} "
            f"request_source_run_id={source_run_id} "
            f"mlflow_source_run_id={version.run_id}"
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


def validate_canary_metadata(
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
        raise RuntimeError("canary metadata validation failed: " + "; ".join(errors))


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


def record_reload_event(
        *,
        request: dict[str, Any],
        args: argparse.Namespace,
        reload_status: str,
        started_at: float,
        completed_at: float,
        threshold: float,
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
        new_model_version=str(request["source_version"]),
        new_model_run_id=str(request["source_run_id"]),
        new_threshold=threshold,
        reload_status=reload_status,
        error_message=error_message,
        metadata=metadata,
        started_at=started_at,
        completed_at=completed_at,
    )


def none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def active_request_id_from_state(
        state: dict[str, Any] | None,
        metadata: dict[str, Any],
) -> str | None:
    if state is None or "metadata_error" in metadata:
        return None

    if none_or_str(state.get("active_model_version")) != none_or_str(metadata.get("model_version")):
        return None

    if none_or_str(state.get("active_model_run_id")) != none_or_str(metadata.get("model_run_id")):
        return None

    metadata_threshold = metadata.get("threshold")
    state_threshold = state.get("active_threshold")
    if metadata_threshold is None or state_threshold is None:
        return None

    if abs(float(metadata_threshold) - float(state_threshold)) > 1e-12:
        return None

    return none_or_str(state.get("active_request_id"))


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

    source_version = str(request["source_version"])
    source_run_id = str(request["source_run_id"])

    validate_mlflow_source(
        client,
        model_name=args.model_name,
        source_version=source_version,
        source_run_id=source_run_id,
    )
    threshold = load_run_threshold(client, source_run_id)

    if args.dry_run:
        print(
            "candidate_canary_deploy_dry_run "
            f"request_id={request['request_id']} "
            f"model_name={args.model_name} "
            f"source_version={source_version} "
            f"source_run_id={source_run_id} "
            f"threshold={threshold} "
            f"reload_url={args.reload_url} "
            f"metadata_url={args.metadata_url}"
        )
        return

    started_at = time.time()
    previous_metadata = try_get_metadata(args.metadata_url, args.http_timeout_seconds)
    previous_state = get_runtime_deployment_state(
        model_name=args.model_name,
        runtime_slot=args.runtime_slot,
    )
    reload_payload: dict[str, Any] | None = None

    try:
        mark_deployment_request_rollout_status(
            request["request_id"],
            "canary_reloading",
        )

        reload_payload = post_json(
            args.reload_url,
            {
                "request_id": request["request_id"],
                "model_name": args.model_name,
                "model_version": source_version,
                "expected_run_id": source_run_id,
                "threshold": threshold,
            },
            args.http_timeout_seconds,
        )

        metadata = get_json(args.metadata_url, args.http_timeout_seconds)
        validate_canary_metadata(
            metadata,
            model_name=args.model_name,
            model_version=source_version,
            model_run_id=source_run_id,
            threshold=threshold,
            runtime_slot=args.runtime_slot,
        )

        completed_at = time.time()
        previous_runtime = previous_from_payload(reload_payload, previous_metadata)
        record_reload_event(
            request=request,
            args=args,
            reload_status="succeeded",
            started_at=started_at,
            completed_at=completed_at,
            threshold=threshold,
            previous_metadata=previous_runtime,
            metadata={
                "reload_response": reload_payload,
                "metadata": metadata,
            },
        )

        updated_request = mark_deployment_request_rollout_status(
            request["request_id"],
            "canary_ready",
        )
        runtime_state = upsert_runtime_deployment_state(
            model_name=args.model_name,
            runtime_slot=args.runtime_slot,
            target_alias=args.target_alias,
            active_request_id=updated_request["request_id"],
            active_model_version=source_version,
            active_model_run_id=source_run_id,
            active_threshold=threshold,
            previous_request_id=active_request_id_from_state(previous_state, previous_runtime),
            previous_model_version=none_or_str(previous_runtime.get("model_version")),
            previous_model_run_id=none_or_str(previous_runtime.get("model_run_id")),
            previous_threshold=float_or_none(previous_runtime.get("threshold")),
            last_operation="canary_reload",
            last_operation_request_id=updated_request["request_id"],
        )

        print(
            "candidate_canary_deploy_complete "
            f"request_id={updated_request['request_id']} "
            f"model_name={args.model_name} "
            f"source_version={source_version} "
            f"source_run_id={source_run_id} "
            f"threshold={threshold} "
            f"rollout_status={updated_request['rollout_status']} "
            f"state_active_request_id={runtime_state.get('active_request_id')} "
            f"state_previous_request_id={runtime_state.get('previous_request_id')} "
            f"runtime_slot={metadata.get('runtime_slot')}"
        )
    except Exception as error:
        completed_at = time.time()
        failure_metadata = {
            "reload_response": reload_payload,
            "canary_metadata": try_get_metadata(args.metadata_url, args.http_timeout_seconds),
        }
        record_reload_event(
            request=request,
            args=args,
            reload_status="failed",
            started_at=started_at,
            completed_at=completed_at,
            threshold=threshold,
            previous_metadata=previous_from_payload(reload_payload, previous_metadata),
            metadata=failure_metadata,
            error_message=str(error),
        )
        mark_deployment_request_rollout_status(request["request_id"], "failed")
        raise


if __name__ == "__main__":
    main()
