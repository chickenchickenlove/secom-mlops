from __future__ import annotations

import argparse
import time
from typing import Any

import httpx
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from secom_mlops.monitor.deployments import (
    append_deployment_request_note,
    get_deployment_request,
    get_runtime_deployment_state,
    insert_runtime_reload_event,
    mark_deployment_request_rollout_status,
    upsert_runtime_deployment_state,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CHAMPION_ALIAS,
    resolve_model_name,
    resolve_tracking_uri,
)
from secom_mlops_common.config.serving import (
    RELEASE_METADATA_PATH,
    RELEASE_RELOAD_MODEL_VERSION_PATH,
    TRAFFIC_POLICY_PATH,
    model_gateway_admin_endpoint,
    model_gateway_endpoint,
)

ROLLBACK_ELIGIBLE_ROLLOUT_STATUSES = {
    "deployed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--target-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument(
        "--release-reload-url",
        default=model_gateway_endpoint(RELEASE_RELOAD_MODEL_VERSION_PATH),
    )
    parser.add_argument(
        "--release-metadata-url",
        default=model_gateway_endpoint(RELEASE_METADATA_PATH),
    )
    parser.add_argument(
        "--traffic-policy-url",
        default=model_gateway_admin_endpoint(TRAFFIC_POLICY_PATH),
    )
    parser.add_argument("--service-name", default="model-server-release")
    parser.add_argument("--runtime-slot", default="release")
    parser.add_argument("--http-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--skip-canary-traffic-reset", action="store_true")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_runtime_state(args: argparse.Namespace) -> dict[str, Any]:
    state = get_runtime_deployment_state(
        model_name=args.model_name,
        runtime_slot=args.runtime_slot,
    )
    if state is None:
        raise RuntimeError(
            "runtime deployment state not found: "
            f"model_name={args.model_name} runtime_slot={args.runtime_slot}"
        )
    return state


def resolve_deployment_request(
        args: argparse.Namespace,
        state: dict[str, Any],
) -> dict[str, Any]:
    request_id = none_or_str(state.get("active_request_id"))
    if request_id is None:
        raise RuntimeError(
            "active deployment request_id not found in runtime state: "
            f"model_name={args.model_name} runtime_slot={args.runtime_slot}"
        )

    request = get_deployment_request(request_id)
    if request is None:
        raise RuntimeError(f"deployment request not found: request_id={request_id}")
    return request


def validate_runtime_state_for_rollback(
        state: dict[str, Any],
        *,
        model_name: str,
        runtime_slot: str,
        target_alias: str,
) -> None:
    if state["model_name"] != model_name:
        raise RuntimeError(
            "runtime state model_name mismatch: "
            f"state_model_name={state['model_name']} model_name={model_name}"
        )

    if state["runtime_slot"] != runtime_slot:
        raise RuntimeError(
            "runtime state runtime_slot mismatch: "
            f"state_runtime_slot={state['runtime_slot']} runtime_slot={runtime_slot}"
        )

    if state["target_alias"] != target_alias:
        raise RuntimeError(
            "runtime state target_alias mismatch: "
            f"state_target_alias={state['target_alias']} target_alias={target_alias}"
        )

    if state["last_operation"] == "rollback":
        raise RuntimeError(
            "release rollback is one-step only and the latest operation is already rollback: "
            f"active_model_version={state['active_model_version']} "
            f"previous_model_version={state.get('previous_model_version')}"
        )

    if (
        none_or_str(state.get("previous_model_version")) is None
        or none_or_str(state.get("previous_model_run_id")) is None
        or state.get("previous_threshold") is None
    ):
        raise RuntimeError(
            "no previous runtime state to rollback: "
            f"model_name={model_name} runtime_slot={runtime_slot}"
        )


def validate_deployment_request(
        request: dict[str, Any],
        *,
        model_name: str,
        target_alias: str,
        state: dict[str, Any],
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
            "deployment request is not post-release rollback-eligible: "
            f"request_id={request['request_id']} rollout_status={request['rollout_status']} "
            f"allowed={sorted(ROLLBACK_ELIGIBLE_ROLLOUT_STATUSES)}"
        )

    if str(request["source_version"]) != str(state["active_model_version"]):
        raise RuntimeError(
            "deployment request source_version does not match active runtime state: "
            f"request_id={request['request_id']} "
            f"request_source_version={request['source_version']} "
            f"state_active_model_version={state['active_model_version']}"
        )

    if str(request["source_run_id"]) != str(state["active_model_run_id"]):
        raise RuntimeError(
            "deployment request source_run_id does not match active runtime state: "
            f"request_id={request['request_id']} "
            f"request_source_run_id={request['source_run_id']} "
            f"state_active_model_run_id={state['active_model_run_id']}"
        )


def validate_model_version(
        client: MlflowClient,
        *,
        model_name: str,
        model_version: str,
        expected_run_id: str,
        label: str,
) -> None:
    version = client.get_model_version(model_name, model_version)
    if str(version.run_id) != str(expected_run_id):
        raise RuntimeError(
            f"deployment request {label}_run_id does not match MLflow model version: "
            f"model_name={model_name} "
            f"model_version={model_version} "
            f"request_run_id={expected_run_id} "
            f"mlflow_run_id={version.run_id}"
        )


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


def validate_release_precondition(
        metadata: dict[str, Any],
        *,
        state: dict[str, Any],
) -> None:
    current_version = none_or_str(metadata.get("model_version"))
    active_version = str(state["active_model_version"])
    previous_version = str(state["previous_model_version"])
    allowed_versions = {active_version, previous_version}

    if current_version not in allowed_versions:
        raise RuntimeError(
            "release runtime model version does not match rollback runtime state: "
            f"current_release_version={current_version} "
            f"state_active_model_version={active_version} "
            f"state_previous_model_version={previous_version}"
        )


def validate_alias_precondition(
        client: MlflowClient,
        *,
        model_name: str,
        target_alias: str,
        source_version: str,
        previous_version: str,
) -> None:
    current_alias_version = get_alias_version(client, model_name, target_alias)
    if current_alias_version is None:
        return

    if str(current_alias_version.version) not in {source_version, previous_version}:
        raise RuntimeError(
            "target alias version does not match deployment request lineage: "
            f"model_name={model_name} "
            f"target_alias={target_alias} "
            f"current_alias_version={current_alias_version.version} "
            f"request_source_version={source_version} "
            f"request_previous_version={previous_version}"
        )


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
        raise RuntimeError("release rollback metadata validation failed: " + "; ".join(errors))


def get_alias_version(client: MlflowClient, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def build_reload_payload(
        *,
        request: dict[str, Any],
        model_name: str,
        previous_version: str,
        previous_run_id: str,
        previous_threshold: float,
        previous_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "request_id": request["request_id"],
        "model_name": model_name,
        "model_version": previous_version,
        "expected_run_id": previous_run_id,
        "threshold": previous_threshold,
    }

    current_version = none_or_str(previous_metadata.get("model_version"))
    current_run_id = none_or_str(previous_metadata.get("model_run_id"))
    if current_version is not None:
        payload["expected_current_model_version"] = current_version
    if current_run_id is not None:
        payload["expected_current_run_id"] = current_run_id

    return payload


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


def reset_canary_traffic(
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
    tracking_uri = resolve_tracking_uri(args.tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    state = resolve_runtime_state(args)
    validate_runtime_state_for_rollback(
        state,
        model_name=args.model_name,
        runtime_slot=args.runtime_slot,
        target_alias=args.target_alias,
    )

    request = resolve_deployment_request(args, state)
    validate_deployment_request(
        request,
        model_name=args.model_name,
        target_alias=args.target_alias,
        state=state,
    )

    source_version = str(state["active_model_version"])
    source_run_id = str(state["active_model_run_id"])
    previous_version = str(state["previous_model_version"])
    previous_run_id = str(state["previous_model_run_id"])
    previous_threshold = float(state["previous_threshold"])

    validate_model_version(
        client,
        model_name=args.model_name,
        model_version=source_version,
        expected_run_id=source_run_id,
        label="source",
    )
    validate_model_version(
        client,
        model_name=args.model_name,
        model_version=previous_version,
        expected_run_id=previous_run_id,
        label="previous",
    )

    if args.dry_run:
        print(
            "release_deployment_rollback_dry_run "
            f"request_id={request['request_id']} "
            f"model_name={args.model_name} "
            f"source_version={source_version} "
            f"source_run_id={source_run_id} "
            f"previous_version={previous_version} "
            f"previous_run_id={previous_run_id} "
            f"previous_threshold={previous_threshold} "
            f"previous_request_id={state.get('previous_request_id')} "
            f"rollout_status={request['rollout_status']} "
            f"release_reload_url={args.release_reload_url} "
            f"release_metadata_url={args.release_metadata_url} "
            f"traffic_policy_url={args.traffic_policy_url} "
            f"reset_canary_traffic={not args.skip_canary_traffic_reset}"
        )
        return

    validate_alias_precondition(
        client,
        model_name=args.model_name,
        target_alias=args.target_alias,
        source_version=source_version,
        previous_version=previous_version,
    )

    traffic_policy_response = None
    if not args.skip_canary_traffic_reset:
        traffic_policy_response = reset_canary_traffic(
            request_id=request["request_id"],
            traffic_policy_url=args.traffic_policy_url,
            timeout=args.http_timeout_seconds,
        )

    started_at = time.time()
    previous_metadata = get_json(args.release_metadata_url, args.http_timeout_seconds)
    validate_release_precondition(previous_metadata, state=state)
    reload_payload: dict[str, Any] | None = None
    reload_event_recorded = False

    try:
        reload_payload = post_json(
            args.release_reload_url,
            build_reload_payload(
                request=request,
                model_name=args.model_name,
                previous_version=previous_version,
                previous_run_id=previous_run_id,
                previous_threshold=previous_threshold,
                previous_metadata=previous_metadata,
            ),
            args.http_timeout_seconds,
        )

        metadata = get_json(args.release_metadata_url, args.http_timeout_seconds)
        validate_runtime_metadata(
            metadata,
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
                "metadata": metadata,
                "traffic_policy_response": traffic_policy_response,
            },
        )
        reload_event_recorded = True

        client.set_registered_model_alias(
            args.model_name,
            args.target_alias,
            previous_version,
        )

        if args.notes:
            append_deployment_request_note(request["request_id"], args.notes)

        updated_request = mark_deployment_request_rollout_status(
            request["request_id"],
            "rolled_back",
        )
        runtime_state = upsert_runtime_deployment_state(
            model_name=args.model_name,
            runtime_slot=args.runtime_slot,
            target_alias=args.target_alias,
            active_request_id=none_or_str(state.get("previous_request_id")),
            active_model_version=previous_version,
            active_model_run_id=previous_run_id,
            active_threshold=previous_threshold,
            previous_request_id=request["request_id"],
            previous_model_version=source_version,
            previous_model_run_id=source_run_id,
            previous_threshold=float(state["active_threshold"]),
            last_operation="rollback",
            last_operation_request_id=request["request_id"],
        )

        print(
            "release_deployment_rollback_complete "
            f"request_id={updated_request['request_id']} "
            f"model_name={args.model_name} "
            f"source_version={source_version} "
            f"previous_version={previous_version} "
            f"previous_run_id={previous_run_id} "
            f"previous_threshold={previous_threshold} "
            f"target_alias={args.target_alias} "
            f"traffic_canary_percent={traffic_policy_response.get('canary_percent') if traffic_policy_response else None} "
            f"rollout_status={updated_request['rollout_status']} "
            f"state_active_request_id={runtime_state.get('active_request_id')} "
            f"state_previous_request_id={runtime_state.get('previous_request_id')} "
            f"runtime_slot={metadata.get('runtime_slot')}"
        )
    except Exception as error:
        completed_at = time.time()
        if not reload_event_recorded:
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
                    "release_metadata": try_get_metadata(args.release_metadata_url, args.http_timeout_seconds),
                    "traffic_policy_response": traffic_policy_response,
                },
                error_message=str(error),
            )
        raise


if __name__ == "__main__":
    main()
