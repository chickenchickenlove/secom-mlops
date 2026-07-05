import time
from typing import Any
from uuid import uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from secom_mlops.monitor.db import connect

VALID_EVAL_STATUSES = {"passed", "failed", "insufficient_data", "skipped", "unknown"}
VALID_APPROVAL_STATUSES = {"pending", "approved", "rejected"}
VALID_ROLLOUT_STATUSES = {
    "not_started",
    "promoted",
    "canary_reloading",
    "canary_ready",
    "release_reloading",
    "deployed",
    "failed",
    "rolled_back",
    "superseded",
}

TERMINAL_ROLLOUT_STATUSES = {"deployed", "failed", "rolled_back", "superseded"}


def normalize_request_id(request_id: str | None) -> str | None:
    if request_id is None:
        return None

    value = str(request_id).strip()
    if value.startswith("request_id="):
        value = value.split("=", 1)[1].strip()

    return value


INSERT_DEPLOYMENT_REQUEST_SQL = """
INSERT INTO model_deployment_requests (
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
  eval_summary_json,
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
  created_at,
  updated_at
)
VALUES (
  %(request_id)s,
  %(model_name)s,
  %(target_alias)s,
  %(source_alias)s,
  %(source_version)s,
  %(source_run_id)s,
  %(previous_version)s,
  %(previous_run_id)s,
  %(eval_type)s,
  %(eval_status)s,
  %(eval_summary_json)s,
  %(approval_status)s,
  %(rollout_status)s,
  %(runtime_target)s,
  %(notes)s,
  %(requested_by)s,
  %(approved_by)s,
  %(requested_at)s,
  %(approved_at)s,
  %(promoted_at)s,
  %(canary_reload_started_at)s,
  %(canary_ready_at)s,
  %(release_reload_started_at)s,
  %(deployed_at)s,
  %(failed_at)s,
  %(rolled_back_at)s,
  %(created_at)s,
  %(updated_at)s
);
"""

SELECT_REQUEST_SQL = """
SELECT *
FROM model_deployment_requests
WHERE request_id = %s;
"""

APPROVED_DEPLOYMENT_REQUEST_SQL = """
SELECT *
FROM model_deployment_requests
WHERE model_name = %s
  AND source_version = %s
  AND target_alias = %s
  AND eval_status = 'passed'
  AND approval_status = 'approved'
  AND rollout_status NOT IN ('deployed', 'failed', 'rolled_back', 'superseded')
ORDER BY requested_at DESC
LIMIT 1;
"""

MARK_PROMOTED_SQL = """
UPDATE model_deployment_requests
SET
  rollout_status = 'promoted',
  promoted_at = COALESCE(promoted_at, %s),
  updated_at = %s
WHERE request_id = %s
RETURNING *;
"""

NEXT_APPROVED_DEPLOYMENT_REQUEST_SQL = """
SELECT *
FROM model_deployment_requests
WHERE model_name = %s
  AND target_alias = %s
  AND eval_status = 'passed'
  AND approval_status = 'approved'
  AND rollout_status IN ('not_started', 'promoted', 'canary_reloading', 'canary_ready')
ORDER BY requested_at DESC
LIMIT 1;
"""

NEXT_RELEASE_PROMOTABLE_DEPLOYMENT_REQUEST_SQL = """
SELECT *
FROM model_deployment_requests
WHERE model_name = %s
  AND target_alias = %s
  AND eval_status = 'passed'
  AND approval_status = 'approved'
  AND rollout_status IN ('canary_ready', 'release_reloading')
ORDER BY requested_at DESC
LIMIT 1;
"""

NEXT_ROLLBACK_ELIGIBLE_DEPLOYMENT_REQUEST_SQL = """
SELECT *
FROM model_deployment_requests
WHERE model_name = %s
  AND target_alias = %s
  AND approval_status = 'approved'
  AND rollout_status IN (
    'promoted',
    'canary_reloading',
    'canary_ready',
    'release_reloading',
    'failed'
  )
ORDER BY requested_at DESC
LIMIT 1;
"""

MARK_ROLLOUT_STATUS_SQL = """
UPDATE model_deployment_requests
SET
  rollout_status = %(rollout_status)s,
  canary_reload_started_at = CASE
    WHEN %(rollout_status)s = 'canary_reloading' THEN COALESCE(canary_reload_started_at, %(timestamp)s)
    ELSE canary_reload_started_at
  END,
  canary_ready_at = CASE
    WHEN %(rollout_status)s = 'canary_ready' THEN COALESCE(canary_ready_at, %(timestamp)s)
    ELSE canary_ready_at
  END,
  release_reload_started_at = CASE
    WHEN %(rollout_status)s = 'release_reloading' THEN COALESCE(release_reload_started_at, %(timestamp)s)
    ELSE release_reload_started_at
  END,
  deployed_at = CASE
    WHEN %(rollout_status)s = 'deployed' THEN COALESCE(deployed_at, %(timestamp)s)
    ELSE deployed_at
  END,
  failed_at = CASE
    WHEN %(rollout_status)s = 'failed' THEN COALESCE(failed_at, %(timestamp)s)
    ELSE failed_at
  END,
  rolled_back_at = CASE
    WHEN %(rollout_status)s = 'rolled_back' THEN COALESCE(rolled_back_at, %(timestamp)s)
    ELSE rolled_back_at
  END,
  updated_at = %(timestamp)s
WHERE request_id = %(request_id)s
RETURNING *;
"""

APPEND_DEPLOYMENT_REQUEST_NOTE_SQL = """
UPDATE model_deployment_requests
SET
  notes = CASE
    WHEN CAST(%(note)s AS text) IS NULL OR CAST(%(note)s AS text) = '' THEN notes
    WHEN notes IS NULL OR notes = '' THEN CAST(%(note)s AS text)
    ELSE notes || E'\\n' || CAST(%(note)s AS text)
  END,
  updated_at = %(timestamp)s
WHERE request_id = %(request_id)s
RETURNING *;
"""

INSERT_RUNTIME_RELOAD_EVENT_SQL = """
INSERT INTO model_runtime_reload_events (
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
  metadata_json,
  started_at,
  completed_at,
  created_at
)
VALUES (
  %(event_id)s,
  %(request_id)s,
  %(service_name)s,
  %(runtime_slot)s,
  %(model_name)s,
  %(previous_model_version)s,
  %(previous_model_run_id)s,
  %(previous_threshold)s,
  %(new_model_version)s,
  %(new_model_run_id)s,
  %(new_threshold)s,
  %(reload_status)s,
  %(error_message)s,
  %(metadata_json)s,
  %(started_at)s,
  %(completed_at)s,
  %(created_at)s
);
"""

SELECT_RUNTIME_DEPLOYMENT_STATE_SQL = """
SELECT *
FROM model_runtime_deployment_state
WHERE model_name = %s
  AND runtime_slot = %s;
"""

UPSERT_RUNTIME_DEPLOYMENT_STATE_SQL = """
INSERT INTO model_runtime_deployment_state (
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
)
VALUES (
  %(model_name)s,
  %(runtime_slot)s,
  %(target_alias)s,
  %(active_request_id)s,
  %(active_model_version)s,
  %(active_model_run_id)s,
  %(active_threshold)s,
  %(previous_request_id)s,
  %(previous_model_version)s,
  %(previous_model_run_id)s,
  %(previous_threshold)s,
  %(last_operation)s,
  %(last_operation_request_id)s,
  %(timestamp)s,
  %(timestamp)s
)
ON CONFLICT (model_name, runtime_slot)
DO UPDATE SET
  target_alias = EXCLUDED.target_alias,
  active_request_id = EXCLUDED.active_request_id,
  active_model_version = EXCLUDED.active_model_version,
  active_model_run_id = EXCLUDED.active_model_run_id,
  active_threshold = EXCLUDED.active_threshold,
  previous_request_id = EXCLUDED.previous_request_id,
  previous_model_version = EXCLUDED.previous_model_version,
  previous_model_run_id = EXCLUDED.previous_model_run_id,
  previous_threshold = EXCLUDED.previous_threshold,
  last_operation = EXCLUDED.last_operation,
  last_operation_request_id = EXCLUDED.last_operation_request_id,
  updated_at = EXCLUDED.updated_at
RETURNING *;
"""

INSERT_RUNTIME_DEPLOYMENT_STATE_IF_MISSING_SQL = """
INSERT INTO model_runtime_deployment_state (
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
)
VALUES (
  %(model_name)s,
  %(runtime_slot)s,
  %(target_alias)s,
  %(active_request_id)s,
  %(active_model_version)s,
  %(active_model_run_id)s,
  %(active_threshold)s,
  %(previous_request_id)s,
  %(previous_model_version)s,
  %(previous_model_run_id)s,
  %(previous_threshold)s,
  %(last_operation)s,
  %(last_operation_request_id)s,
  %(timestamp)s,
  %(timestamp)s
)
ON CONFLICT (model_name, runtime_slot) DO NOTHING
RETURNING *;
"""


def resolve_eval_status(eval_status: str | None = None, gate_status: str | None = None) -> str:
    return eval_status or gate_status or "unknown"


def validate_deployment_statuses(
        *,
        eval_status: str | None = None,
        gate_status: str | None = None,
        approval_status: str,
        rollout_status: str = "not_started",
) -> None:
    resolved_eval_status = resolve_eval_status(eval_status, gate_status)

    if resolved_eval_status not in VALID_EVAL_STATUSES:
        raise ValueError(f"eval_status must be one of {sorted(VALID_EVAL_STATUSES)}")

    if approval_status not in VALID_APPROVAL_STATUSES:
        raise ValueError(f"approval_status must be one of {sorted(VALID_APPROVAL_STATUSES)}")

    if rollout_status not in VALID_ROLLOUT_STATUSES:
        raise ValueError(f"rollout_status must be one of {sorted(VALID_ROLLOUT_STATUSES)}")


def build_deployment_request_row(
        *,
        model_name: str,
        source_alias: str | None,
        source_version: str,
        source_run_id: str,
        target_alias: str,
        previous_version: str | None,
        previous_run_id: str | None,
        approval_status: str,
        eval_status: str | None = None,
        gate_status: str | None = None,
        rollout_status: str = "not_started",
        runtime_target: str = "release",
        eval_type: str = "unknown",
        eval_summary: dict[str, Any] | None = None,
        metric_summary: dict[str, Any] | None = None,
        notes: str | None = None,
        requested_by: str | None = None,
        approved_by: str | None = None,
        request_id: str | None = None,
        now: float | None = None,
) -> dict[str, Any]:
    resolved_eval_status = resolve_eval_status(eval_status, gate_status)

    validate_deployment_statuses(
        eval_status=resolved_eval_status,
        approval_status=approval_status,
        rollout_status=rollout_status,
    )

    timestamp = time.time() if now is None else now
    summary = eval_summary if eval_summary is not None else metric_summary
    if summary is None:
        summary = {}

    return {
        "request_id": request_id or str(uuid4()),
        "model_name": model_name,
        "target_alias": target_alias,
        "source_alias": source_alias,
        "source_version": str(source_version),
        "source_run_id": source_run_id,
        "previous_version": None if previous_version is None else str(previous_version),
        "previous_run_id": previous_run_id,
        "eval_type": eval_type,
        "eval_status": resolved_eval_status,
        "eval_summary_json": Jsonb(summary),
        "approval_status": approval_status,
        "rollout_status": rollout_status,
        "runtime_target": runtime_target,
        "notes": notes,
        "requested_by": requested_by,
        "approved_by": approved_by,
        "requested_at": timestamp,
        "approved_at": timestamp if approval_status == "approved" else None,
        "promoted_at": timestamp if rollout_status == "promoted" else None,
        "canary_reload_started_at": timestamp if rollout_status == "canary_reloading" else None,
        "canary_ready_at": timestamp if rollout_status == "canary_ready" else None,
        "release_reload_started_at": timestamp if rollout_status == "release_reloading" else None,
        "deployed_at": timestamp if rollout_status == "deployed" else None,
        "failed_at": timestamp if rollout_status == "failed" else None,
        "rolled_back_at": timestamp if rollout_status == "rolled_back" else None,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def insert_deployment_request(row: dict[str, Any]) -> None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(INSERT_DEPLOYMENT_REQUEST_SQL, row)


def get_deployment_request(request_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(SELECT_REQUEST_SQL, [request_id])
            row = cursor.fetchone()

    return None if row is None else dict(row)


def find_approved_deployment_request(
        *,
        model_name: str,
        source_version: str,
        target_alias: str,
) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                APPROVED_DEPLOYMENT_REQUEST_SQL,
                [model_name, str(source_version), target_alias],
            )
            row = cursor.fetchone()

    return None if row is None else dict(row)


def find_next_approved_deployment_request(
        *,
        model_name: str,
        target_alias: str,
) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                NEXT_APPROVED_DEPLOYMENT_REQUEST_SQL,
                [model_name, target_alias],
            )
            row = cursor.fetchone()

    return None if row is None else dict(row)


def find_next_release_promotable_deployment_request(
        *,
        model_name: str,
        target_alias: str,
) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                NEXT_RELEASE_PROMOTABLE_DEPLOYMENT_REQUEST_SQL,
                [model_name, target_alias],
            )
            row = cursor.fetchone()

    return None if row is None else dict(row)


def find_next_rollback_eligible_deployment_request(
        *,
        model_name: str,
        target_alias: str,
) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                NEXT_ROLLBACK_ELIGIBLE_DEPLOYMENT_REQUEST_SQL,
                [model_name, target_alias],
            )
            row = cursor.fetchone()

    return None if row is None else dict(row)


def mark_deployment_request_promoted(request_id: str) -> dict[str, Any]:
    now = time.time()

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(MARK_PROMOTED_SQL, [now, now, request_id])
            row = cursor.fetchone()

    if row is None:
        raise RuntimeError(f"deployment request not found: request_id={request_id}")

    return dict(row)


def mark_deployment_request_rollout_status(
        request_id: str,
        rollout_status: str,
        now: float | None = None,
) -> dict[str, Any]:
    if rollout_status not in VALID_ROLLOUT_STATUSES:
        raise ValueError(f"rollout_status must be one of {sorted(VALID_ROLLOUT_STATUSES)}")

    timestamp = time.time() if now is None else now

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                MARK_ROLLOUT_STATUS_SQL,
                {
                    "request_id": request_id,
                    "rollout_status": rollout_status,
                    "timestamp": timestamp,
                },
            )
            row = cursor.fetchone()

    if row is None:
        raise RuntimeError(f"deployment request not found: request_id={request_id}")

    return dict(row)


def append_deployment_request_note(
        request_id: str,
        note: str | None,
        now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else now

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                APPEND_DEPLOYMENT_REQUEST_NOTE_SQL,
                {
                    "request_id": request_id,
                    "note": note,
                    "timestamp": timestamp,
                },
            )
            row = cursor.fetchone()

    if row is None:
        raise RuntimeError(f"deployment request not found: request_id={request_id}")

    return dict(row)


def insert_runtime_reload_event(
        *,
        request_id: str | None,
        service_name: str,
        runtime_slot: str,
        model_name: str,
        new_model_version: str,
        new_model_run_id: str,
        new_threshold: float,
        reload_status: str,
        started_at: float,
        completed_at: float | None = None,
        previous_model_version: str | None = None,
        previous_model_run_id: str | None = None,
        previous_threshold: float | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
) -> str:
    if reload_status not in {"started", "succeeded", "failed"}:
        raise ValueError("reload_status must be one of ['started', 'succeeded', 'failed']")

    timestamp = time.time()
    resolved_event_id = event_id or str(uuid4())

    row = {
        "event_id": resolved_event_id,
        "request_id": request_id,
        "service_name": service_name,
        "runtime_slot": runtime_slot,
        "model_name": model_name,
        "previous_model_version": previous_model_version,
        "previous_model_run_id": previous_model_run_id,
        "previous_threshold": previous_threshold,
        "new_model_version": str(new_model_version),
        "new_model_run_id": str(new_model_run_id),
        "new_threshold": float(new_threshold),
        "reload_status": reload_status,
        "error_message": error_message,
        "metadata_json": Jsonb(metadata or {}),
        "started_at": started_at,
        "completed_at": completed_at,
        "created_at": timestamp,
    }

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(INSERT_RUNTIME_RELOAD_EVENT_SQL, row)

    return resolved_event_id


def get_runtime_deployment_state(
        *,
        model_name: str,
        runtime_slot: str,
) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(SELECT_RUNTIME_DEPLOYMENT_STATE_SQL, [model_name, runtime_slot])
            row = cursor.fetchone()

    return None if row is None else dict(row)


def upsert_runtime_deployment_state(
        *,
        model_name: str,
        runtime_slot: str,
        target_alias: str,
        active_request_id: str | None,
        active_model_version: str,
        active_model_run_id: str,
        active_threshold: float,
        previous_request_id: str | None,
        previous_model_version: str | None,
        previous_model_run_id: str | None,
        previous_threshold: float | None,
        last_operation: str,
        last_operation_request_id: str | None,
        now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else now

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                UPSERT_RUNTIME_DEPLOYMENT_STATE_SQL,
                {
                    "model_name": model_name,
                    "runtime_slot": runtime_slot,
                    "target_alias": target_alias,
                    "active_request_id": active_request_id,
                    "active_model_version": str(active_model_version),
                    "active_model_run_id": active_model_run_id,
                    "active_threshold": float(active_threshold),
                    "previous_request_id": previous_request_id,
                    "previous_model_version": (
                        None
                        if previous_model_version is None
                        else str(previous_model_version)
                    ),
                    "previous_model_run_id": previous_model_run_id,
                    "previous_threshold": (
                        None
                        if previous_threshold is None
                        else float(previous_threshold)
                    ),
                    "last_operation": last_operation,
                    "last_operation_request_id": last_operation_request_id,
                    "timestamp": timestamp,
                },
            )
            row = cursor.fetchone()

    if row is None:
        raise RuntimeError(
            "runtime deployment state upsert failed: "
            f"model_name={model_name} runtime_slot={runtime_slot}"
        )

    return dict(row)


def insert_runtime_deployment_state_if_missing(
        *,
        model_name: str,
        runtime_slot: str,
        target_alias: str,
        active_request_id: str | None,
        active_model_version: str,
        active_model_run_id: str,
        active_threshold: float,
        previous_request_id: str | None,
        previous_model_version: str | None,
        previous_model_run_id: str | None,
        previous_threshold: float | None,
        last_operation: str,
        last_operation_request_id: str | None,
        now: float | None = None,
) -> dict[str, Any] | None:
    timestamp = time.time() if now is None else now

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                INSERT_RUNTIME_DEPLOYMENT_STATE_IF_MISSING_SQL,
                {
                    "model_name": model_name,
                    "runtime_slot": runtime_slot,
                    "target_alias": target_alias,
                    "active_request_id": active_request_id,
                    "active_model_version": str(active_model_version),
                    "active_model_run_id": active_model_run_id,
                    "active_threshold": float(active_threshold),
                    "previous_request_id": previous_request_id,
                    "previous_model_version": (
                        None
                        if previous_model_version is None
                        else str(previous_model_version)
                    ),
                    "previous_model_run_id": previous_model_run_id,
                    "previous_threshold": (
                        None
                        if previous_threshold is None
                        else float(previous_threshold)
                    ),
                    "last_operation": last_operation,
                    "last_operation_request_id": last_operation_request_id,
                    "timestamp": timestamp,
                },
            )
            row = cursor.fetchone()

    return None if row is None else dict(row)
