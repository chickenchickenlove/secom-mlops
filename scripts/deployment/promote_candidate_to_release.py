from __future__ import annotations

import argparse
import time
from typing import Any

import httpx
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from secom_mlops.monitor.deployments import (
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
    RELEASE_METADATA_PATH,
    RELEASE_RELOAD_MODEL_VERSION_PATH,
    TRAFFIC_POLICY_PATH,
    model_gateway_admin_endpoint,
    model_gateway_endpoint,
)

RELEASE_PROMOTABLE_ROLLOUT_STATUSES = {
    "canary_ready",
    "release_reloading",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--target-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--request-id", required=True)
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
    parser.add_argument("--keep-source-alias", action="store_true")
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

    if request["rollout_status"] not in RELEASE_PROMOTABLE_ROLLOUT_STATUSES:
        raise RuntimeError(
            "deployment request is not release-promotable: "
            f"request_id={request['request_id']} rollout_status={request['rollout_status']} "
            f"allowed={sorted(RELEASE_PROMOTABLE_ROLLOUT_STATUSES)}"
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


def validate_release_precondition(
        metadata: dict[str, Any],
        request: dict[str, Any],
) -> None:
    if "metadata_error" in metadata:
        return

    current_version = none_or_str(metadata.get("model_version"))
    if current_version is None:
        return

    source_version = str(request["source_version"])
    previous_version = none_or_str(request.get("previous_version"))
    allowed_versions = {source_version}
    if previous_version is not None:
        allowed_versions.add(previous_version)

    if current_version not in allowed_versions:
        raise RuntimeError(
            "release runtime model version does not match deployment request lineage: "
            f"request_id={request['request_id']} "
            f"current_release_version={current_version} "
            f"request_previous_version={previous_version} "
            f"request_source_version={source_version}"
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
        raise RuntimeError("release metadata validation failed: " + "; ".join(errors))


def get_alias_version(client: MlflowClient, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def clear_source_alias_if_current(
        client: MlflowClient,
        *,
        model_name: str,
        source_alias: str | None,
        source_version: str,
        target_alias: str,
        keep_source_alias: bool,
) -> bool:
    if keep_source_alias or source_alias is None or source_alias == target_alias:
        return False

    current_source = get_alias_version(client, model_name, source_alias)
    if current_source is None or str(current_source.version) != str(source_version):
        return False

    client.delete_registered_model_alias(model_name, source_alias)
    return True


def build_reload_payload(
        *,
        request: dict[str, Any],
        model_name: str,
        source_version: str,
        source_run_id: str,
        threshold: float,
        previous_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "request_id": request["request_id"],
        "model_name": model_name,
        "model_version": source_version,
        "expected_run_id": source_run_id,
        "threshold": threshold,
    }

    if "metadata_error" not in previous_metadata:
        previous_version = none_or_str(previous_metadata.get("model_version"))
        previous_run_id = none_or_str(previous_metadata.get("model_run_id"))
        if previous_version is not None:
            payload["expected_current_model_version"] = previous_version
        if previous_run_id is not None:
            payload["expected_current_run_id"] = previous_run_id

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
            "candidate_release_promote_dry_run "
            f"request_id={request['request_id']} "
            f"model_name={args.model_name} "
            f"source_version={source_version} "
            f"source_run_id={source_run_id} "
            f"threshold={threshold} "
            f"release_reload_url={args.release_reload_url} "
            f"release_metadata_url={args.release_metadata_url} "
            f"traffic_policy_url={args.traffic_policy_url} "
            f"reset_canary_traffic={not args.skip_canary_traffic_reset} "
            f"keep_source_alias={args.keep_source_alias}"
        )
        return

    started_at = time.time()
    previous_metadata = try_get_metadata(args.release_metadata_url, args.http_timeout_seconds)
    previous_state = get_runtime_deployment_state(
        model_name=args.model_name,
        runtime_slot=args.runtime_slot,
    )
    reload_payload: dict[str, Any] | None = None
    reload_event_recorded = False

    try:
        validate_release_precondition(previous_metadata, request)

        mark_deployment_request_rollout_status(
            request["request_id"],
            "release_reloading",
        )

        reload_payload = post_json(
            args.release_reload_url,
            build_reload_payload(
                request=request,
                model_name=args.model_name,
                source_version=source_version,
                source_run_id=source_run_id,
                threshold=threshold,
                previous_metadata=previous_metadata,
            ),
            args.http_timeout_seconds,
        )

        metadata = get_json(args.release_metadata_url, args.http_timeout_seconds)
        validate_runtime_metadata(
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
        reload_event_recorded = True

        client.set_registered_model_alias(
            args.model_name,
            args.target_alias,
            source_version,
        )
        source_alias_cleared = clear_source_alias_if_current(
            client,
            model_name=args.model_name,
            source_alias=request.get("source_alias"),
            source_version=source_version,
            target_alias=args.target_alias,
            keep_source_alias=args.keep_source_alias,
        )

        traffic_policy_response = None
        if not args.skip_canary_traffic_reset:
            traffic_policy_response = reset_canary_traffic(
                request_id=request["request_id"],
                traffic_policy_url=args.traffic_policy_url,
                timeout=args.http_timeout_seconds,
            )

        updated_request = mark_deployment_request_rollout_status(
            request["request_id"],
            "deployed",
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
            last_operation="promote",
            last_operation_request_id=updated_request["request_id"],
        )

        print(
            "candidate_release_promote_complete "
            f"request_id={updated_request['request_id']} "
            f"model_name={args.model_name} "
            f"source_version={source_version} "
            f"source_run_id={source_run_id} "
            f"threshold={threshold} "
            f"target_alias={args.target_alias} "
            f"source_alias={request.get('source_alias')} "
            f"source_alias_cleared={source_alias_cleared} "
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
                threshold=threshold,
                previous_metadata=previous_from_payload(reload_payload, previous_metadata),
                metadata={
                    "reload_response": reload_payload,
                    "release_metadata": try_get_metadata(args.release_metadata_url, args.http_timeout_seconds),
                },
                error_message=str(error),
            )
        mark_deployment_request_rollout_status(request["request_id"], "failed")
        raise


if __name__ == "__main__":
    main()
