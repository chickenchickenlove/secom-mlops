"""PostgreSQL access for immutable serving-gate datasets."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row

from secom_mlops.datasets.serving_gate_dataset import (
    DATASET_SCHEMA_VERSION,
    DATASET_TYPE,
    SELECTOR_VERSION,
    ServingGateDatasetConfig,
    ServingGateDatasetIdentity,
    ServingGateDatasetMember,
    member_from_row,
)
from secom_mlops.monitor.db import connect


def build_cohort_query(
        config: ServingGateDatasetConfig,
        *,
        include_features: bool,
) -> tuple[str, dict[str, Any]]:
    features_column = ",\n      s.features_json" if include_features else ""
    sql = f"""
    WITH ranked_release_decisions AS (
      SELECT
        p.prediction_id,
        p.request_id,
        p.sample_id,
        p.serving_snapshot_id,
        p.snapshot_version,
        p.feature_hash,
        p.model_run_id AS source_model_run_id,
        p.runtime_slot,
        p.threshold AS source_threshold,
        p.predicted_at,
        ROW_NUMBER() OVER (
          PARTITION BY p.sample_id, p.snapshot_version
          ORDER BY p.predicted_at ASC, p.prediction_id ASC
        ) AS decision_rank,
        EXISTS (
          SELECT 1
          FROM prediction_logs conflicting
          WHERE conflicting.runtime_slot = 'release'
            AND conflicting.sample_id = p.sample_id
            AND conflicting.snapshot_version = p.snapshot_version
            AND (
              conflicting.serving_snapshot_id <> p.serving_snapshot_id
              OR conflicting.feature_hash <> p.feature_hash
            )
        ) AS has_conflicting_snapshot_identity
      FROM prediction_logs p
      WHERE p.runtime_slot = 'release'
    ),
    decision_cohort AS (
      SELECT *
      FROM ranked_release_decisions
      WHERE decision_rank = 1
        AND predicted_at >= %(cohort_start_time)s
        AND predicted_at < %(cohort_end_time)s
    ),
    cohort_samples AS (
      SELECT DISTINCT sample_id
      FROM decision_cohort
    ),
    ranked_labels AS (
      SELECT
        le.*,
        ROW_NUMBER() OVER (
          PARTITION BY le.sample_id
          ORDER BY
            le.label_revision DESC,
            le.available_at DESC,
            le.label_event_id ASC
        ) AS label_rank
      FROM label_events le
      JOIN cohort_samples c
        ON c.sample_id = le.sample_id
      WHERE le.available_at <= %(cutoff_time)s
    ),
    labels_at_cutoff AS (
      SELECT *
      FROM ranked_labels
      WHERE label_rank = 1
    )
    SELECT
      p.prediction_id,
      p.request_id,
      p.sample_id,
      p.serving_snapshot_id,
      p.snapshot_version,
      p.feature_hash,
      p.source_model_run_id,
      p.runtime_slot,
      p.source_threshold,
      p.predicted_at,
      p.has_conflicting_snapshot_identity,
      s.serving_snapshot_id AS stored_serving_snapshot_id,
      s.sample_id AS stored_sample_id,
      s.snapshot_version AS stored_snapshot_version,
      s.feature_hash AS snapshot_feature_hash,
      s.snapshot_status,
      s.is_complete,
      s.feature_count,
      s.missing_count AS serving_missing_count,
      s.snapshot_time,
      s.available_at AS snapshot_available_at,
      s.window_start,
      s.window_end,
      s.simulation_run_id,
      s.drift_segment{features_column},
      l.label_event_id,
      l.label_revision,
      l.measured_at AS label_measured_at,
      l.available_at AS label_available_at,
      l.actual_value,
      l.actual_label
    FROM decision_cohort p
    LEFT JOIN serving_feature_snapshots s
      ON s.serving_snapshot_id = p.serving_snapshot_id
     AND s.sample_id = p.sample_id
     AND s.snapshot_version = p.snapshot_version
    LEFT JOIN labels_at_cutoff l
      ON l.sample_id = p.sample_id
    ORDER BY p.predicted_at ASC, p.prediction_id ASC;
    """
    return sql, {
        "cohort_start_time": config.cohort_start_time,
        "cohort_end_time": config.cohort_end_time,
        "cutoff_time": config.cutoff_time,
    }


def fetch_members(
        cursor: Any,
        config: ServingGateDatasetConfig,
        *,
        include_features: bool,
) -> list[ServingGateDatasetMember]:
    sql, params = build_cohort_query(config, include_features=include_features)
    cursor.execute(sql, params)
    return [member_from_row(dict(row)) for row in cursor.fetchall()]


def find_ready_dataset_by_manifest(
        cursor: Any,
        manifest_hash: str,
) -> dict[str, Any] | None:
    cursor.execute(
        """
        SELECT *
        FROM dataset_builds
        WHERE dataset_type = %s
          AND manifest_hash = %s
          AND status = 'READY';
        """,
        [DATASET_TYPE, manifest_hash],
    )
    row = cursor.fetchone()
    return dict(row) if row is not None else None


def get_dataset_build(dataset_id: str) -> dict[str, Any]:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT *
                FROM dataset_builds
                WHERE dataset_id = %s;
                """,
                [dataset_id],
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"dataset catalog row not found: dataset_id={dataset_id}")
    return dict(row)


def get_ready_dataset_by_manifest(manifest_hash: str) -> dict[str, Any] | None:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            return find_ready_dataset_by_manifest(cursor, manifest_hash)


def claim_dataset_build(
        identity: ServingGateDatasetIdentity,
        config: ServingGateDatasetConfig,
        stats: dict[str, Any],
) -> bool:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO dataset_builds (
                  dataset_id,
                  dataset_type,
                  dataset_schema_version,
                  selector_version,
                  status,
                  cohort_start_time,
                  cutoff_time,
                  label_maturity_seconds,
                  manifest_hash,
                  eligible_sample_count,
                  labeled_sample_count,
                  unlabeled_sample_count,
                  label_coverage,
                  fail_count,
                  pass_count
                )
                VALUES (
                  %s, %s, %s, %s, 'BUILDING',
                  %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (dataset_type, manifest_hash)
                DO UPDATE SET
                  status = 'BUILDING',
                  cutoff_time = EXCLUDED.cutoff_time,
                  label_maturity_seconds = EXCLUDED.label_maturity_seconds,
                  eligible_sample_count = EXCLUDED.eligible_sample_count,
                  labeled_sample_count = EXCLUDED.labeled_sample_count,
                  unlabeled_sample_count = EXCLUDED.unlabeled_sample_count,
                  label_coverage = EXCLUDED.label_coverage,
                  fail_count = EXCLUDED.fail_count,
                  pass_count = EXCLUDED.pass_count,
                  mlflow_run_id = NULL,
                  artifact_uri = NULL,
                  artifact_sha256 = NULL,
                  ready_at = NULL,
                  error_message = NULL,
                  updated_at = EXTRACT(EPOCH FROM clock_timestamp())
                WHERE dataset_builds.status <> 'READY'
                RETURNING dataset_id;
                """,
                [
                    identity.dataset_id,
                    DATASET_TYPE,
                    DATASET_SCHEMA_VERSION,
                    SELECTOR_VERSION,
                    config.cohort_start_time,
                    config.cutoff_time,
                    config.label_maturity_seconds,
                    identity.manifest_hash,
                    stats["decision_count"],
                    stats["labeled_decision_count"],
                    stats["unlabeled_decision_count"],
                    stats["label_coverage"],
                    stats["fail_count"],
                    stats["pass_count"],
                ],
            )
            return cursor.fetchone() is not None


def mark_dataset_ready(
        dataset_id: str,
        *,
        mlflow_run_id: str,
        artifact_uri: str,
        artifact_sha256: str,
) -> None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE dataset_builds
                SET status = 'READY',
                    mlflow_run_id = %s,
                    artifact_uri = %s,
                    artifact_sha256 = %s,
                    ready_at = EXTRACT(EPOCH FROM clock_timestamp()),
                    updated_at = EXTRACT(EPOCH FROM clock_timestamp()),
                    error_message = NULL
                WHERE dataset_id = %s;
                """,
                [mlflow_run_id, artifact_uri, artifact_sha256, dataset_id],
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"dataset catalog row not found: dataset_id={dataset_id}"
                )


def mark_dataset_failed(dataset_id: str, error_message: str) -> None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE dataset_builds
                SET status = 'FAILED',
                    error_message = %s,
                    updated_at = EXTRACT(EPOCH FROM clock_timestamp())
                WHERE dataset_id = %s;
                """,
                [error_message[:4000], dataset_id],
            )
