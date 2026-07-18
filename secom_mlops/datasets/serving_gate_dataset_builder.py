"""Materialize an immutable serving-gate dataset before model evaluation."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg.rows import dict_row

from secom_mlops.datasets.dataset_artifacts import (
    MLFLOW_DATASET_DIGEST_LENGTH,
    write_artifacts,
)
from secom_mlops.datasets.serving_gate_dataset import (
    DATASET_SCHEMA_VERSION,
    DATASET_TYPE,
    SELECTOR_VERSION,
    ServingGateDatasetConfig,
    ServingGateDatasetIdentity,
    build_dataset_frame,
    build_dataset_identity,
    build_manifest,
    evaluate_readiness,
)
from secom_mlops.datasets.serving_gate_dataset_repository import (
    claim_dataset_build,
    fetch_members,
    find_ready_dataset_by_manifest,
    get_ready_dataset_by_manifest,
    mark_dataset_failed,
    mark_dataset_ready,
)
from secom_mlops.monitor.db import connect

DEFAULT_EXPERIMENT_NAME = "secom-serving-gate-datasets"
MLFLOW_DATASET_CONTEXT = "serving_gate"


@dataclass(frozen=True)
class PersistedServingGateDataset:
    dataset_id: str
    manifest_hash: str
    artifact_sha256: str
    mlflow_run_id: str
    artifact_uri: str
    stats: dict[str, Any]
    reused: bool = False


class InsufficientServingGateData(RuntimeError):
    """The requested cohort does not satisfy the materialization contract."""


def _persisted_from_catalog(
        row: dict[str, Any],
        *,
        reused: bool,
) -> PersistedServingGateDataset:
    required_fields = ("artifact_sha256", "mlflow_run_id", "artifact_uri")
    if missing := [name for name in required_fields if not row.get(name)]:
        raise RuntimeError(
            "READY dataset catalog row is incomplete: "
            f"dataset_id={row.get('dataset_id')} fields={missing}"
        )
    return PersistedServingGateDataset(
        dataset_id=str(row["dataset_id"]),
        manifest_hash=str(row["manifest_hash"]),
        artifact_sha256=str(row["artifact_sha256"]),
        mlflow_run_id=str(row["mlflow_run_id"]),
        artifact_uri=str(row["artifact_uri"]),
        stats={
            "decision_count": int(row["eligible_sample_count"]),
            "labeled_decision_count": int(row["labeled_sample_count"]),
            "unlabeled_decision_count": int(row["unlabeled_sample_count"]),
            "label_coverage": float(row["label_coverage"]),
            "fail_count": int(row["fail_count"]),
            "pass_count": int(row["pass_count"]),
        },
        reused=reused,
    )


def build_mlflow_dataset(
        frame: pd.DataFrame,
        identity: ServingGateDatasetIdentity,
) -> Any:
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
        identity: ServingGateDatasetIdentity,
        config: ServingGateDatasetConfig,
        stats: dict[str, Any],
        artifact_sha256: str,
        *,
        tracking_uri: str,
        experiment_name: str,
) -> tuple[str, str]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        mlflow.create_experiment(experiment_name)
        if experiment is None
        else experiment.experiment_id
    )

    with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=identity.dataset_id,
            tags={
                "dataset_id": identity.dataset_id,
                "dataset_type": DATASET_TYPE,
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "selector_version": SELECTOR_VERSION,
                "manifest_hash": identity.manifest_hash,
                "decision_time_column": "predicted_at",
                "runtime_slot": "release",
            },
    ) as run:
        mlflow.log_params({
            "cohort_start_time": config.cohort_start_time,
            "cohort_end_time": config.cohort_end_time,
            "cutoff_time": config.cutoff_time,
            "label_maturity_seconds": config.label_maturity_seconds,
            "min_decisions": config.min_decisions,
            "min_labeled_decisions": config.min_labeled_decisions,
            "min_label_coverage": config.min_label_coverage,
            "min_fail_samples": config.min_fail_samples,
            "min_pass_samples": config.min_pass_samples,
        })
        mlflow.log_metrics({
            "decision_count": float(stats["decision_count"]),
            "labeled_decision_count": float(stats["labeled_decision_count"]),
            "unlabeled_decision_count": float(stats["unlabeled_decision_count"]),
            "unique_sample_count": float(stats["unique_sample_count"]),
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


def build_serving_gate_dataset(
        config: ServingGateDatasetConfig,
        *,
        tracking_uri: str,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> PersistedServingGateDataset:
    config.validate()

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
            )
            metadata_members = fetch_members(cursor, config, include_features=False)
            readiness = evaluate_readiness(config, metadata_members)
            if not readiness.ready:
                raise InsufficientServingGateData("; ".join(readiness.reasons))

            identity = build_dataset_identity(config, metadata_members)
            existing = find_ready_dataset_by_manifest(cursor, identity.manifest_hash)
            if existing is not None:
                return _persisted_from_catalog(existing, reused=True)

            full_members = fetch_members(cursor, config, include_features=True)
            metadata_identity = [member.identity_record() for member in metadata_members]
            full_identity = [member.identity_record() for member in full_members]
            if metadata_identity != full_identity:
                raise RuntimeError(
                    "serving-gate dataset membership changed within repeatable-read build"
                )

    if not claim_dataset_build(identity, config, readiness.stats):
        existing = get_ready_dataset_by_manifest(identity.manifest_hash)
        if existing is None:
            raise RuntimeError(
                "serving-gate dataset claim failed without a READY catalog row: "
                f"dataset_id={identity.dataset_id}"
            )
        return _persisted_from_catalog(existing, reused=True)

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
        return PersistedServingGateDataset(
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
                "failed to record serving-gate dataset build failure: "
                f"dataset_id={identity.dataset_id} error={catalog_exc}",
                file=sys.stderr,
            )
        raise
