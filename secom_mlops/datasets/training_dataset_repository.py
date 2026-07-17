"""PostgreSQL access for point-in-time dataset builds."""

from __future__ import annotations

from typing import Any

from secom_mlops.datasets.training_dataset import (
    DATASET_SCHEMA_VERSION,
    DATASET_TYPE,
    SELECTOR_VERSION,
    DatasetBuildConfig,
    DatasetIdentity,
    DatasetMember,
    member_from_row,
)
from secom_mlops.monitor.db import connect


def build_cohort_query(
        config: DatasetBuildConfig,
        *,
        include_features: bool,
) -> tuple[str, list[Any]]:
    snapshot_filters = [
        "s.is_complete = TRUE",
        "s.snapshot_status = 'complete'",
    ]
    params: list[Any] = []

    if config.simulation_run_id is not None:
        snapshot_filters.append("s.simulation_run_id = %s")
        params.append(config.simulation_run_id)
    if config.drift_segment is not None:
        snapshot_filters.append("s.drift_segment = %s")
        params.append(config.drift_segment)

    ranked_features_column = ",\n        s.features_json" if include_features else ""
    selected_features_column = ",\n      s.features_json" if include_features else ""
    params.extend([
        config.cohort_start_time,
        config.cohort_end_time,
        config.cutoff_time,
    ])

    sql = f"""
    WITH ranked_complete_snapshots AS (
      SELECT
        s.serving_snapshot_id,
        s.sample_id,
        s.snapshot_version,
        s.feature_hash,
        s.snapshot_time,
        s.available_at AS snapshot_available_at,
        s.window_start,
        s.window_end,
        s.feature_count,
        s.missing_count AS serving_missing_count,
        s.simulation_run_id,
        s.drift_segment{ranked_features_column},
        ROW_NUMBER() OVER (
          PARTITION BY s.sample_id
          ORDER BY
            s.available_at ASC,
            s.snapshot_version ASC,
            s.serving_snapshot_id ASC
        ) AS snapshot_rank
      FROM serving_feature_snapshots s
      WHERE {' AND '.join(snapshot_filters)}
    ),
    first_complete_cohort AS (
      SELECT
        *
      FROM ranked_complete_snapshots
      WHERE snapshot_rank = 1
        AND snapshot_available_at >= %s
        AND snapshot_available_at < %s
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
      JOIN first_complete_cohort s
        ON s.sample_id = le.sample_id
      WHERE le.available_at <= %s
    ),
    labels_at_cutoff AS (
      SELECT
        *
      FROM ranked_labels
      WHERE label_rank = 1
    )
    SELECT
      s.serving_snapshot_id,
      s.sample_id,
      s.snapshot_version,
      s.feature_hash,
      s.snapshot_time,
      s.snapshot_available_at,
      s.window_start,
      s.window_end,
      s.feature_count,
      s.serving_missing_count,
      s.simulation_run_id,
      s.drift_segment{selected_features_column},
      l.label_event_id,
      l.label_revision,
      l.measured_at AS label_measured_at,
      l.available_at AS label_available_at,
      l.actual_value,
      l.actual_label
    FROM first_complete_cohort s
    LEFT JOIN labels_at_cutoff l
      ON l.sample_id = s.sample_id
    ORDER BY
      s.snapshot_available_at ASC,
      s.sample_id ASC,
      s.serving_snapshot_id ASC;
    """
    return sql, params


def fetch_members(
        cursor: Any,
        config: DatasetBuildConfig,
        *,
        include_features: bool,
) -> list[DatasetMember]:
    sql, params = build_cohort_query(config, include_features=include_features)
    cursor.execute(sql, params)
    return [member_from_row(dict(row)) for row in cursor.fetchall()]


def ready_dataset_exists(cursor: Any, manifest_hash: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM dataset_builds
        WHERE dataset_type = %s
          AND manifest_hash = %s
          AND status = 'READY';
        """,
        [DATASET_TYPE, manifest_hash],
    )
    return cursor.fetchone() is not None


def claim_dataset_build(
        identity: DatasetIdentity,
        config: DatasetBuildConfig,
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
                    stats["eligible_sample_count"],
                    stats["labeled_sample_count"],
                    stats["unlabeled_sample_count"],
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
                    f"dataset build catalog row not found: dataset_id={dataset_id}"
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
