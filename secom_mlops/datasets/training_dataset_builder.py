"""Orchestrate immutable training-source dataset persistence."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg.rows import dict_row

from secom_mlops.datasets.training_dataset import (
    DATASET_SCHEMA_VERSION,
    DATASET_TYPE,
    SELECTOR_VERSION,
    DatasetBuildConfig,
    DatasetIdentity,
    build_dataset_frame,
    build_dataset_identity,
    build_manifest,
    evaluate_readiness,
)
from secom_mlops.datasets.training_dataset_repository import (
    claim_dataset_build,
    fetch_members,
    mark_dataset_failed,
    mark_dataset_ready,
    ready_dataset_exists,
)
from secom_mlops.monitor.db import connect

DEFAULT_EXPERIMENT_NAME = "secom-training-datasets"
MLFLOW_DATASET_CONTEXT = "training_source"
MLFLOW_DATASET_DIGEST_LENGTH = 36


@dataclass(frozen=True)
class PersistedDataset:
    dataset_id: str
    manifest_hash: str
    artifact_sha256: str
    mlflow_run_id: str
    artifact_uri: str
    stats: dict[str, Any]


class DatasetBuildSkipped(RuntimeError):
    """The scheduled check completed without requiring a new dataset."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:v1:{digest.hexdigest()}"


def write_artifacts(
        output_dir: Path,
        frame: pd.DataFrame,
        manifest: dict[str, Any],
        stats: dict[str, Any],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.parquet"
    manifest_path = output_dir / "manifest.json"
    stats_path = output_dir / "stats.json"

    frame.to_parquet(dataset_path, index=False, engine="pyarrow")
    persisted_frame = pd.read_parquet(dataset_path, engine="pyarrow")
    if list(persisted_frame.columns) != list(frame.columns):
        raise RuntimeError("persisted dataset schema does not match the source frame")
    if len(persisted_frame) != len(frame):
        raise RuntimeError("persisted dataset row count does not match the source frame")

    artifact_sha256 = sha256_file(dataset_path)
    persisted_manifest = {
        **manifest,
        "artifact": {
            "path": "data/dataset.parquet",
            "sha256": artifact_sha256,
            "row_count": len(frame),
            "columns": list(frame.columns),
        },
    }
    manifest_path.write_text(
        json.dumps(
            persisted_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_sha256


def build_mlflow_dataset(
        frame: pd.DataFrame,
        identity: DatasetIdentity,
) -> Any:
    """Build MLflow Dataset metadata without copying the persisted artifact."""
    import mlflow

    manifest_digest = identity.manifest_hash.removeprefix("sha256:v1:")
    return mlflow.data.from_pandas(
        frame,
        name=identity.dataset_id,
        targets="actual_value",
        digest=manifest_digest[:MLFLOW_DATASET_DIGEST_LENGTH],
    )


def persist_artifacts_to_mlflow(
        artifact_dir: Path,
        frame: pd.DataFrame,
        identity: DatasetIdentity,
        config: DatasetBuildConfig,
        stats: dict[str, Any],
        artifact_sha256: str,
        *,
        tracking_uri: str,
        experiment_name: str,
) -> tuple[str, str]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=identity.dataset_id,
            tags={
                "dataset_id": identity.dataset_id,
                "dataset_type": DATASET_TYPE,
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "selector_version": SELECTOR_VERSION,
                "manifest_hash": identity.manifest_hash,
                "decision_time_column": "snapshot_available_at",
            },
    ) as run:
        mlflow.log_params({
            "cohort_start_time": config.cohort_start_time,
            "cutoff_time": config.cutoff_time,
            "cohort_end_time": config.cohort_end_time,
            "label_maturity_seconds": config.label_maturity_seconds,
            "min_labeled_samples": config.min_labeled_samples,
            "min_label_coverage": config.min_label_coverage,
            "min_fail_samples": config.min_fail_samples,
            "min_pass_samples": config.min_pass_samples,
            "simulation_run_id": config.simulation_run_id or "",
            "drift_segment": config.drift_segment or "",
        })
        mlflow.log_metrics({
            "eligible_sample_count": float(stats["eligible_sample_count"]),
            "labeled_sample_count": float(stats["labeled_sample_count"]),
            "unlabeled_sample_count": float(stats["unlabeled_sample_count"]),
            "label_coverage": float(stats["label_coverage"]),
            "fail_count": float(stats["fail_count"]),
            "pass_count": float(stats["pass_count"]),
        })
        mlflow.log_artifacts(str(artifact_dir), artifact_path="data")
        artifact_uri = mlflow.get_artifact_uri("data/dataset.parquet")
        tracked_dataset = build_mlflow_dataset(frame, identity)
        mlflow.log_input(
            tracked_dataset,
            context=MLFLOW_DATASET_CONTEXT,
            tags={
                "dataset_id": identity.dataset_id,
                "manifest_hash": identity.manifest_hash,
                "artifact_uri": artifact_uri,
                "artifact_sha256": artifact_sha256,
            },
        )
        return run.info.run_id, artifact_uri


def build_training_dataset(
        config: DatasetBuildConfig,
        *,
        tracking_uri: str,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> PersistedDataset:
    config.validate()

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
            )
            metadata_members = fetch_members(cursor, config, include_features=False)
            readiness = evaluate_readiness(config, metadata_members)
            if not readiness.ready:
                raise DatasetBuildSkipped("; ".join(readiness.reasons))

            identity = build_dataset_identity(config, metadata_members)
            if ready_dataset_exists(cursor, identity.manifest_hash):
                raise DatasetBuildSkipped(
                    "dataset membership is already READY: "
                    f"dataset_id={identity.dataset_id}"
                )

            full_members = fetch_members(cursor, config, include_features=True)
            metadata_identity = [member.identity_record() for member in metadata_members]
            full_identity = [member.identity_record() for member in full_members]
            if metadata_identity != full_identity:
                raise RuntimeError("dataset membership changed within repeatable-read build")

    if not claim_dataset_build(identity, config, readiness.stats):
        raise DatasetBuildSkipped(
            "dataset membership became READY concurrently: "
            f"dataset_id={identity.dataset_id}"
        )

    try:
        frame = build_dataset_frame(identity.dataset_id, full_members)
        manifest = build_manifest(config, identity, readiness.stats)
        with tempfile.TemporaryDirectory(prefix=f"{identity.dataset_id}_") as temp_dir:
            artifact_dir = Path(temp_dir)
            artifact_sha256 = write_artifacts(
                artifact_dir,
                frame,
                manifest,
                readiness.stats,
            )
            mlflow_run_id, artifact_uri = persist_artifacts_to_mlflow(
                artifact_dir,
                frame,
                identity,
                config,
                readiness.stats,
                artifact_sha256,
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
            )

        mark_dataset_ready(
            identity.dataset_id,
            mlflow_run_id=mlflow_run_id,
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_sha256,
        )
        return PersistedDataset(
            dataset_id=identity.dataset_id,
            manifest_hash=identity.manifest_hash,
            artifact_sha256=artifact_sha256,
            mlflow_run_id=mlflow_run_id,
            artifact_uri=artifact_uri,
            stats=readiness.stats,
        )
    except Exception as exc:
        try:
            mark_dataset_failed(identity.dataset_id, str(exc))
        except Exception as catalog_exc:
            print(
                "failed to record dataset build failure: "
                f"dataset_id={identity.dataset_id} error={catalog_exc}",
                file=sys.stderr,
            )
        raise
