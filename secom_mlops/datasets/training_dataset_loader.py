"""Download and verify a persisted training-source dataset."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mlflow

from secom_mlops.datasets.dataset_artifacts import sha256_file
from secom_mlops.datasets.training_dataset import (
    DATASET_COLUMNS,
    DATASET_SCHEMA_VERSION,
    DATASET_TYPE,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    SELECTOR_VERSION,
)
from secom_mlops.datasets.training_dataset_repository import get_dataset_build
from secom_mlops_common.schemas.secom import MODEL_COLUMNS


@dataclass(frozen=True)
class LoadedTrainingDataset:
    dataset_id: str
    manifest_hash: str
    artifact_sha256: str
    mlflow_run_id: str
    artifact_uri: str
    frame: pd.DataFrame
    manifest: dict[str, Any]


def _identity_manifest_hash(identity: dict[str, Any]) -> str:
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:v1:{hashlib.sha256(canonical).hexdigest()}"


def _require_equal(*, dataset_id: str, field: str, expected: Any, actual: Any) -> None:
    if actual != expected:
        raise RuntimeError(
            "training dataset contract mismatch: "
            f"dataset_id={dataset_id} field={field} "
            f"expected={expected} actual={actual}"
        )


def verify_downloaded_artifacts(
        artifact_dir: Path,
        catalog: dict[str, Any], #
) -> LoadedTrainingDataset:
    dataset_id = str(catalog["dataset_id"])
    manifest_path = artifact_dir / "manifest.json"
    dataset_path = artifact_dir / "dataset.parquet"
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise RuntimeError(f"training artifact is incomplete: dataset_id={dataset_id}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = str(catalog["manifest_hash"])
    _require_equal(
        dataset_id=dataset_id,
        field="manifest.dataset_id",
        expected=dataset_id,
        actual=manifest.get("dataset_id"),
    )
    _require_equal(
        dataset_id=dataset_id,
        field="manifest.manifest_hash",
        expected=manifest_hash,
        actual=manifest.get("manifest_hash"),
    )

    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"training artifact identity is missing: dataset_id={dataset_id}")
    _require_equal(
        dataset_id=dataset_id,
        field="identity_hash",
        expected=manifest_hash,
        actual=_identity_manifest_hash(identity),
    )
    for name, expected in {
        "dataset_type": DATASET_TYPE,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
    }.items():
        _require_equal(
            dataset_id=dataset_id,
            field=f"identity.{name}",
            expected=expected,
            actual=identity.get(name),
        )
    expected_dataset_id = f"training_{manifest_hash.removeprefix('sha256:v1:')[:16]}"
    _require_equal(
        dataset_id=dataset_id,
        field="dataset_id_from_manifest_hash",
        expected=expected_dataset_id,
        actual=dataset_id,
    )

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError(f"training artifact metadata is missing: dataset_id={dataset_id}")
    artifact_sha256 = sha256_file(dataset_path)
    _require_equal(
        dataset_id=dataset_id,
        field="catalog.artifact_sha256", # from DB
        expected=str(catalog["artifact_sha256"]),
        actual=artifact_sha256,
    )
    _require_equal(
        dataset_id=dataset_id,
        field="manifest.artifact.sha256", # from MLflow
        expected=artifact.get("sha256"),
        actual=artifact_sha256,
    )
    _require_equal(
        dataset_id=dataset_id,
        field="manifest.artifact.path",
        expected="data/dataset.parquet",
        actual=artifact.get("path"),
    )
    _require_equal(
        dataset_id=dataset_id,
        field="manifest.artifact.columns",
        expected=DATASET_COLUMNS,
        actual=artifact.get("columns"),
    )

    frame = pd.read_parquet(dataset_path, engine="pyarrow")
    _require_equal(
        dataset_id=dataset_id,
        field="parquet.columns",
        expected=DATASET_COLUMNS,
        actual=list(frame.columns),
    )
    _require_equal(
        dataset_id=dataset_id,
        field="parquet.row_count",
        expected=int(artifact.get("row_count", -1)),
        actual=len(frame),
    )
    if frame.empty or not frame["dataset_id"].eq(dataset_id).all():
        raise RuntimeError(f"persisted training dataset_id mismatch: dataset_id={dataset_id}")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError(f"training dataset contains duplicate samples: dataset_id={dataset_id}")

    labeled = frame["label_event_id"].notna()
    label_columns = [
        "label_revision",
        "label_measured_at",
        "label_available_at",
        "actual_value",
        "actual_label",
    ]
    if frame.loc[labeled, label_columns].isna().any().any():
        raise RuntimeError(f"labeled training row is incomplete: dataset_id={dataset_id}")
    if frame.loc[~labeled, label_columns].notna().any().any():
        raise RuntimeError(f"unlabeled training row has label data: dataset_id={dataset_id}")

    actual_labels = frame.loc[labeled, "actual_value"].astype("int64")
    if not actual_labels.isin([NEGATIVE_CLASS, POSITIVE_CLASS]).all():
        raise RuntimeError(f"training dataset contains an invalid target: dataset_id={dataset_id}")
    expected_names = actual_labels.map({NEGATIVE_CLASS: "pass", POSITIVE_CLASS: "fail"})
    actual_names = frame.loc[labeled, "actual_label"].astype("string")
    if not actual_names.reset_index(drop=True).equals(
            expected_names.astype("string").reset_index(drop=True)
    ):
        raise RuntimeError(f"training target and label disagree: dataset_id={dataset_id}")

    computed_missing = frame[list(MODEL_COLUMNS)].isna().sum(axis=1).astype("int64")
    stored_missing = frame["serving_missing_count"].astype("int64")
    if not computed_missing.equals(stored_missing):
        raise RuntimeError(f"training feature missing_count mismatch: dataset_id={dataset_id}")
    feature_values = frame[list(MODEL_COLUMNS)].to_numpy(dtype="float64")
    if not (np.isnan(feature_values) | np.isfinite(feature_values)).all():
        raise RuntimeError(f"training feature contains a non-finite value: dataset_id={dataset_id}")

    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        raise RuntimeError(f"training artifact stats are missing: dataset_id={dataset_id}")
    labeled_count = int(labeled.sum())
    actual_stats = {
        "eligible_sample_count": len(frame),
        "labeled_sample_count": labeled_count,
        "unlabeled_sample_count": len(frame) - labeled_count,
        "fail_count": int((actual_labels == POSITIVE_CLASS).sum()),
        "pass_count": int((actual_labels == NEGATIVE_CLASS).sum()),
    }
    for name, expected in actual_stats.items():
        _require_equal(
            dataset_id=dataset_id,
            field=f"manifest.stats.{name}",
            expected=expected,
            actual=int(stats.get(name, -1)),
        )
        _require_equal(
            dataset_id=dataset_id,
            field=f"catalog.{name}",
            expected=expected,
            actual=int(catalog[name]),
        )
    coverage = labeled_count / len(frame)
    if not math.isclose(float(stats.get("label_coverage", -1.0)), coverage):
        raise RuntimeError(f"training manifest label coverage mismatch: dataset_id={dataset_id}")
    if not math.isclose(float(catalog["label_coverage"]), coverage):
        raise RuntimeError(f"training catalog label coverage mismatch: dataset_id={dataset_id}")

    build_context = manifest.get("build_context")
    if not isinstance(build_context, dict):
        raise RuntimeError(f"training build context is missing: dataset_id={dataset_id}")
    catalog_times = {
        "cohort_start_time": identity.get("cohort_start_time"),
        "cutoff_time": build_context.get("cutoff_time"),
        "label_maturity_seconds": identity.get("label_maturity_seconds"),
    }
    for name, expected in catalog_times.items():
        if expected is None or not math.isclose(float(catalog[name]), float(expected)):
            raise RuntimeError(
                f"training catalog time mismatch: dataset_id={dataset_id} field={name}"
            )

    return LoadedTrainingDataset(
        dataset_id=dataset_id,
        manifest_hash=manifest_hash,
        artifact_sha256=artifact_sha256,
        mlflow_run_id=str(catalog["mlflow_run_id"]),
        artifact_uri=str(catalog["artifact_uri"]),
        frame=frame,
        manifest=manifest,
    )


def load_training_dataset(dataset_id: str, *, tracking_uri: str) -> LoadedTrainingDataset:
    catalog = get_dataset_build(dataset_id)
    # Validation
    for name, expected in {
        "dataset_type": DATASET_TYPE,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "status": "READY",
    }.items():
        _require_equal(
            dataset_id=dataset_id,
            field=f"catalog.{name}",
            expected=expected,
            actual=catalog.get(name),
        )

    for name in ("mlflow_run_id", "artifact_uri", "artifact_sha256", "manifest_hash"):
        if not catalog.get(name):
            raise RuntimeError(
                f"READY training dataset field is missing: dataset_id={dataset_id} field={name}"
            )

    # download artifact
    mlflow.set_tracking_uri(tracking_uri)
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=str(catalog["mlflow_run_id"]),
        artifact_path="data",
        tracking_uri=tracking_uri,
    )
    #
    return verify_downloaded_artifacts(Path(downloaded), catalog)
