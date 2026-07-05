from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import httpx
from psycopg.rows import dict_row

from secom_mlops.monitor.db import connect
from secom_mlops_common.config.mlflow import resolve_model_name
from secom_mlops_common.config.serving import (
    CANARY_METADATA_PATH,
    PRODUCTION_METADATA_PATH,
    RELEASE_METADATA_PATH,
    TRAFFIC_POLICY_PATH,
    model_gateway_admin_endpoint,
    model_gateway_endpoint,
)

INSPECT_DEPLOYMENT_REQUESTS_SQL = """
SELECT
  request_id,
  model_name,
  target_alias,
  source_alias,
  source_version,
  source_run_id,
  previous_version,
  previous_run_id,
  eval_type,
  eval_status,
  approval_status,
  rollout_status,
  runtime_target,
  notes,
  requested_by,
  approved_by,
  requested_at,
  approved_at,
  promoted_at,
  canary_reload_started_at,
  canary_ready_at,
  release_reload_started_at,
  deployed_at,
  failed_at,
  rolled_back_at,
  updated_at
FROM model_deployment_requests
WHERE model_name = %(model_name)s
  AND target_alias = %(target_alias)s
ORDER BY requested_at DESC
LIMIT %(limit)s;
"""

INSPECT_RELOAD_EVENTS_SQL = """
SELECT
  event_id,
  request_id,
  service_name,
  runtime_slot,
  model_name,
  previous_model_version,
  previous_model_run_id,
  previous_threshold,
  new_model_version,
  new_model_run_id,
  new_threshold,
  reload_status,
  error_message,
  started_at,
  completed_at,
  created_at
FROM model_runtime_reload_events
WHERE model_name = %(model_name)s
ORDER BY started_at DESC
LIMIT %(limit)s;
"""

INSPECT_RUNTIME_DEPLOYMENT_STATE_SQL = """
SELECT
  model_name,
  runtime_slot,
  target_alias,
  active_request_id,
  active_model_version,
  active_model_run_id,
  active_threshold,
  previous_request_id,
  previous_model_version,
  previous_model_run_id,
  previous_threshold,
  last_operation,
  last_operation_request_id,
  created_at,
  updated_at
FROM model_runtime_deployment_state
WHERE model_name = %(model_name)s
ORDER BY runtime_slot;
"""

TIMESTAMP_FIELDS = {
    "requested_at",
    "approved_at",
    "promoted_at",
    "canary_reload_started_at",
    "canary_ready_at",
    "release_reload_started_at",
    "deployed_at",
    "failed_at",
    "rolled_back_at",
    "updated_at",
    "started_at",
    "completed_at",
    "created_at",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--target-alias", default="champion")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--reload-event-limit", type=int, default=20)
    parser.add_argument("--include-runtime-metadata", action="store_true")
    parser.add_argument("--release-metadata-url", default=model_gateway_endpoint(RELEASE_METADATA_PATH))
    parser.add_argument("--canary-metadata-url", default=model_gateway_endpoint(CANARY_METADATA_PATH))
    parser.add_argument("--production-metadata-url", default=model_gateway_endpoint(PRODUCTION_METADATA_PATH))
    parser.add_argument("--traffic-policy-url", default=model_gateway_admin_endpoint(TRAFFIC_POLICY_PATH))
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def fetch_rows(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    return [dict(row) for row in rows]


def get_json_or_error(url: str, timeout: float) -> dict[str, Any]:
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {
                "status": "error",
                "url": url,
                "error": "response is not a JSON object",
                "payload": payload,
            }
        return payload
    except Exception as error:
        return {
            "status": "error",
            "url": url,
            "error": str(error),
        }


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for field in TIMESTAMP_FIELDS:
        value = normalized.get(field)
        normalized[f"{field}_utc"] = epoch_to_iso(value)
    return normalized


def epoch_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), UTC).isoformat()


def print_request_summary(requests: list[dict[str, Any]]) -> None:
    print("deployment_request_summary")
    for request in requests:
        print(
            "request "
            f"request_id={request['request_id']} "
            f"source_version={request['source_version']} "
            f"previous_version={request['previous_version']} "
            f"eval_status={request['eval_status']} "
            f"approval_status={request['approval_status']} "
            f"rollout_status={request['rollout_status']} "
            f"requested_at={request['requested_at_utc']} "
            f"canary_ready_at={request['canary_ready_at_utc']} "
            f"deployed_at={request['deployed_at_utc']} "
            f"failed_at={request['failed_at_utc']} "
            f"rolled_back_at={request['rolled_back_at_utc']}"
        )


def print_runtime_state_summary(states: list[dict[str, Any]]) -> None:
    print("runtime_deployment_state_summary")
    if not states:
        print("runtime_state none")
        return

    for state in states:
        print(
            "runtime_state "
            f"runtime_slot={state['runtime_slot']} "
            f"active_request_id={state['active_request_id']} "
            f"active_model_version={state['active_model_version']} "
            f"active_threshold={state['active_threshold']} "
            f"previous_request_id={state['previous_request_id']} "
            f"previous_model_version={state['previous_model_version']} "
            f"previous_threshold={state['previous_threshold']} "
            f"last_operation={state['last_operation']} "
            f"last_operation_request_id={state['last_operation_request_id']} "
            f"updated_at={state['updated_at_utc']}"
        )


def main() -> None:
    args = parse_args()
    limit = max(1, args.limit)
    reload_event_limit = max(1, args.reload_event_limit)

    requests = [
        normalize_row(row)
        for row in fetch_rows(
            INSPECT_DEPLOYMENT_REQUESTS_SQL,
            {
                "model_name": args.model_name,
                "target_alias": args.target_alias,
                "limit": limit,
            },
        )
    ]
    reload_events = [
        normalize_row(row)
        for row in fetch_rows(
            INSPECT_RELOAD_EVENTS_SQL,
            {
                "model_name": args.model_name,
                "limit": reload_event_limit,
            },
        )
    ]
    runtime_states = [
        normalize_row(row)
        for row in fetch_rows(
            INSPECT_RUNTIME_DEPLOYMENT_STATE_SQL,
            {
                "model_name": args.model_name,
            },
        )
    ]

    runtime_metadata: dict[str, Any] | None = None
    if args.include_runtime_metadata:
        runtime_metadata = {
            "release": get_json_or_error(args.release_metadata_url, args.http_timeout_seconds),
            "canary": get_json_or_error(args.canary_metadata_url, args.http_timeout_seconds),
            "production": get_json_or_error(args.production_metadata_url, args.http_timeout_seconds),
            "traffic_policy": get_json_or_error(args.traffic_policy_url, args.http_timeout_seconds),
        }

    print_request_summary(requests)
    print_runtime_state_summary(runtime_states)
    print("deployment_request_inspection_json")
    print(
        json.dumps(
            {
                "model_name": args.model_name,
                "target_alias": args.target_alias,
                "requests": requests,
                "runtime_deployment_state": runtime_states,
                "reload_events": reload_events,
                "runtime_metadata": runtime_metadata,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
