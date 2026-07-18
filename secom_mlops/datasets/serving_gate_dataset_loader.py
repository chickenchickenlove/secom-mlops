"""Download and verify a persisted serving-gate dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from secom_mlops.datasets.dataset_artifacts import sha256_file
from secom_mlops.datasets.serving_gate_dataset import (
    DATASET_COLUMNS,
    DATASET_SCHEMA_VERSION,
    DATASET_TYPE,
    SELECTOR_VERSION,
)
from secom_mlops.datasets.serving_gate_dataset_repository import get_dataset_build


@dataclass(frozen=True)
class LoadedServingGateDataset:
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


def verify_downloaded_artifacts(
        artifact_dir: Path,
        catalog: dict[str, Any],
) -> LoadedServingGateDataset:
    dataset_id = str(catalog["dataset_id"])
    manifest_path = artifact_dir / "manifest.json"
    dataset_path = artifact_dir / "dataset.parquet"
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise RuntimeError(
            f"serving-gate artifact is incomplete: dataset_id={dataset_id}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = str(catalog["manifest_hash"])
    if manifest.get("dataset_id") != dataset_id:
        raise RuntimeError(f"artifact dataset_id mismatch: dataset_id={dataset_id}")
    if manifest.get("manifest_hash") != manifest_hash:
        raise RuntimeError(f"artifact manifest_hash mismatch: dataset_id={dataset_id}")

    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"artifact identity is missing: dataset_id={dataset_id}")
    if _identity_manifest_hash(identity) != manifest_hash:
        raise RuntimeError(f"artifact identity hash mismatch: dataset_id={dataset_id}")
    if identity.get("dataset_type") != DATASET_TYPE:
        raise RuntimeError(f"artifact dataset_type mismatch: dataset_id={dataset_id}")
    if identity.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError(f"artifact schema version mismatch: dataset_id={dataset_id}")
    if identity.get("selector_version") != SELECTOR_VERSION:
        raise RuntimeError(f"artifact selector version mismatch: dataset_id={dataset_id}")
    expected_dataset_id = f"serving_gate_{manifest_hash.removeprefix('sha256:v1:')[:16]}"
    if dataset_id != expected_dataset_id:
        raise RuntimeError(f"dataset_id does not match manifest hash: dataset_id={dataset_id}")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise RuntimeError(f"artifact metadata is missing: dataset_id={dataset_id}")
    artifact_sha256 = sha256_file(dataset_path)
    if artifact_sha256 != str(catalog["artifact_sha256"]):
        raise RuntimeError(f"catalog artifact hash mismatch: dataset_id={dataset_id}")
    if artifact_sha256 != artifact.get("sha256"):
        raise RuntimeError(f"manifest artifact hash mismatch: dataset_id={dataset_id}")
    if artifact.get("path") != "data/dataset.parquet":
        raise RuntimeError(f"unexpected artifact path: dataset_id={dataset_id}")
    if artifact.get("columns") != DATASET_COLUMNS:
        raise RuntimeError(f"manifest dataset columns mismatch: dataset_id={dataset_id}")

    frame = pd.read_parquet(dataset_path, engine="pyarrow")
    if list(frame.columns) != DATASET_COLUMNS:
        raise RuntimeError(f"persisted dataset schema mismatch: dataset_id={dataset_id}")
    if len(frame) != int(artifact.get("row_count", -1)):
        raise RuntimeError(f"persisted dataset row count mismatch: dataset_id={dataset_id}")
    if frame.empty or not frame["dataset_id"].eq(dataset_id).all():
        raise RuntimeError(f"persisted dataset_id column mismatch: dataset_id={dataset_id}")
    if not frame["runtime_slot"].eq("release").all():
        raise RuntimeError(f"persisted dataset contains a non-release row: dataset_id={dataset_id}")
    if frame.duplicated(["sample_id", "snapshot_version"]).any():
        raise RuntimeError(
            f"persisted dataset contains duplicate sample snapshots: dataset_id={dataset_id}"
        )

    labeled = frame["label_event_id"].notna()
    if frame.loc[labeled, "actual_value"].isna().any():
        raise RuntimeError(f"labeled dataset row has no target: dataset_id={dataset_id}")
    if frame.loc[~labeled, "actual_value"].notna().any():
        raise RuntimeError(f"unlabeled dataset row has a target: dataset_id={dataset_id}")

    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        raise RuntimeError(f"artifact stats are missing: dataset_id={dataset_id}")
    actual_labeled_count = int(labeled.sum())
    expected_counts = {
        "decision_count": len(frame),
        "labeled_decision_count": actual_labeled_count,
        "unlabeled_decision_count": len(frame) - actual_labeled_count,
    }
    for name, actual in expected_counts.items():
        if int(stats.get(name, -1)) != actual:
            raise RuntimeError(
                f"artifact stats mismatch: dataset_id={dataset_id} field={name}"
            )
    if int(catalog["eligible_sample_count"]) != len(frame):
        raise RuntimeError(f"catalog decision count mismatch: dataset_id={dataset_id}")
    if int(catalog["labeled_sample_count"]) != actual_labeled_count:
        raise RuntimeError(f"catalog labeled count mismatch: dataset_id={dataset_id}")

    return LoadedServingGateDataset(
        dataset_id=dataset_id,
        manifest_hash=manifest_hash,
        artifact_sha256=artifact_sha256,
        mlflow_run_id=str(catalog["mlflow_run_id"]),
        artifact_uri=str(catalog["artifact_uri"]),
        frame=frame,
        manifest=manifest,
    )


def load_serving_gate_dataset(
        dataset_id: str,
        *,
        tracking_uri: str,
) -> LoadedServingGateDataset:
    import mlflow

    catalog = get_dataset_build(dataset_id)
    expected_catalog_values = {
        "dataset_type": DATASET_TYPE,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "status": "READY",
    }
    # For Validate
    for name, expected in expected_catalog_values.items():
        if catalog.get(name) != expected:
            raise RuntimeError(
                "dataset catalog contract mismatch: "
                f"dataset_id={dataset_id} field={name} "
                f"expected={expected} actual={catalog.get(name)}"
            )
    for name in ("mlflow_run_id", "artifact_uri", "artifact_sha256", "manifest_hash"):
        if not catalog.get(name):
            raise RuntimeError(
                f"READY dataset catalog field is missing: dataset_id={dataset_id} field={name}"
            )

    mlflow.set_tracking_uri(tracking_uri)
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=str(catalog["mlflow_run_id"]),
        artifact_path="data",
        tracking_uri=tracking_uri,
    )
    return verify_downloaded_artifacts(Path(downloaded), catalog)
