import hashlib
import json
from typing import Any
from dataclasses import dataclass

import pandas as pd

from secom_mlops.datasets.training_dataset_loader import (
    LoadedTrainingDataset,
)
from secom_mlops_common.schemas.secom import MODEL_COLUMNS

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1

DEFAULT_N_ESTIMATORS = "100,300"
DEFAULT_MIN_SAMPLES_LEAF = "1,3"
DEFAULT_THRESHOLDS = "0.1,0.2,0.3,0.4,0.5"
MAX_DEVELOPMENT_SAMPLES = 1000
VALIDATION_SIZE = 0.2
DEFAULT_MIN_LABEL_COVERAGE = 0.95
DEVELOPMENT_SAMPLE_SELECTION = "latest_labeled_snapshot_available_at"
TRAIN_SOURCE = "versioned_training_dataset"
TRAINING_SPINE = "serving_feature_snapshots"
TRAINING_DECISION_TIME = "snapshot_available_at"
SNAPSHOT_SELECTION = "first_complete"
LABEL_SELECTION = "available_at_lte_cutoff_then_max_revision"
GATE_SOURCE = "serving_feature_snapshots"



@dataclass(frozen=True)
class PreparedTrainingData:
    features: pd.DataFrame
    targets: pd.Series
    sample_ids: list[str]
    metadata: dict[str, Any]
    selected_rows: pd.DataFrame

# Select Labeled Data
# Latest 1,000 limit.
# Validation


def prepare_training_dataset(loaded_dataset: LoadedTrainingDataset) -> PreparedTrainingData:
    selected = select_latest_labeled_training_rows(loaded_dataset.frame, dataset_id=loaded_dataset.dataset_id)
    frame = selected[list(MODEL_COLUMNS)].astype("float64").copy()
    target = selected["actual_value"].astype("int64").copy()
    sample_ids = selected["sample_id"].astype(str).tolist()

    manifest = loaded_dataset.manifest
    identity = manifest["identity"]
    build_context = manifest["build_context"]
    full_labeled = loaded_dataset.frame["label_event_id"].notna()
    selected_snapshot_times = selected["snapshot_time"].astype(float)
    selected_decision_times = selected["snapshot_available_at"].astype(float)
    selected_missing_counts = selected["serving_missing_count"].astype(int)
    selected_label_revisions = selected["label_revision"].astype(int)
    selected_label_measured_times = selected["label_measured_at"].astype(float)
    selected_label_available_times = selected["label_available_at"].astype(float)

    metadata = {
        "train_source": TRAIN_SOURCE,
        "training_spine": TRAINING_SPINE,
        "training_decision_time": TRAINING_DECISION_TIME,
        "snapshot_selection": SNAPSHOT_SELECTION,
        "label_selection": LABEL_SELECTION,
        "dataset_id": loaded_dataset.dataset_id,
        "dataset_manifest_hash": loaded_dataset.manifest_hash,
        "dataset_artifact_sha256": loaded_dataset.artifact_sha256,
        "dataset_mlflow_run_id": loaded_dataset.mlflow_run_id,
        "dataset_artifact_uri": loaded_dataset.artifact_uri,
        "dataset_selection_hash": _selection_hash(loaded_dataset.dataset_id, selected),
        "cohort_start_time": float(identity["cohort_start_time"]),
        "cohort_end_time": float(build_context["cohort_end_time"]),
        "cutoff_time": float(build_context["cutoff_time"]),
        "label_maturity_seconds": float(identity["label_maturity_seconds"]),
        "simulation_run_id": identity.get("simulation_run_id"),
        "drift_segment": identity.get("drift_segment"),
        "development_sample_limit": MAX_DEVELOPMENT_SAMPLES,
        "development_sample_selection": DEVELOPMENT_SAMPLE_SELECTION,
        "eligible_cohort_count": len(loaded_dataset.frame),
        "labeled_cohort_count": int(full_labeled.sum()),
        "unlabeled_cohort_count": int((~full_labeled).sum()),
        "label_coverage": float(full_labeled.mean()),
        "sample_count": len(selected),
        "fail_count": int((target == POSITIVE_CLASS).sum()),
        "pass_count": int((target == NEGATIVE_CLASS).sum()),
        "eligible_decision_time_min": float(loaded_dataset.frame["snapshot_available_at"].min()),
        "eligible_decision_time_max": float(loaded_dataset.frame["snapshot_available_at"].max()),
        "decision_time_min": float(selected_decision_times.min()),
        "decision_time_max": float(selected_decision_times.max()),
        "snapshot_time_min": float(selected_snapshot_times.min()),
        "snapshot_time_max": float(selected_snapshot_times.max()),
        "snapshot_version_min": int(selected["snapshot_version"].min()),
        "snapshot_version_max": int(selected["snapshot_version"].max()),
        "window_start_min": float(selected["window_start"].min()),
        "window_end_max": float(selected["window_end"].max()),
        "serving_missing_count_avg": float(selected_missing_counts.mean()),
        "serving_missing_count_max": int(selected_missing_counts.max()),
        "label_revision_min": int(selected_label_revisions.min()),
        "label_revision_max": int(selected_label_revisions.max()),
        "label_measured_at_min": float(selected_label_measured_times.min()),
        "label_measured_at_max": float(selected_label_measured_times.max()),
        "label_available_at_min": float(selected_label_available_times.min()),
        "label_available_at_max": float(selected_label_available_times.max()),
        "first_eligible_sample_id": str(loaded_dataset.frame.iloc[0]["sample_id"]),
        "last_eligible_sample_id": str(loaded_dataset.frame.iloc[-1]["sample_id"]),
        "first_sample_id": sample_ids[0],
        "last_sample_id": sample_ids[-1],
        "first_snapshot_id": str(selected.iloc[0]["serving_snapshot_id"]),
        "last_snapshot_id": str(selected.iloc[-1]["serving_snapshot_id"]),
        "first_label_event_id": str(selected.iloc[0]["label_event_id"]),
        "last_label_event_id": str(selected.iloc[-1]["label_event_id"]),
    }

    return PreparedTrainingData(
        features=frame,
        targets=target,
        sample_ids=sample_ids,
        metadata=metadata,
        selected_rows=selected
    )


def validate_training_data(
        metadata: dict[str, Any],
        min_samples: int,
        min_label_coverage: float,
        min_fail_samples: int,
        min_pass_samples: int,
) -> None:
    actual_coverage = metadata["label_coverage"]
    if actual_coverage is None or actual_coverage < min_label_coverage:
        raise ValueError(
            "point-in-time label coverage below training minimum: "
            f"required={min_label_coverage:.6f} "
            f"actual={0.0 if actual_coverage is None else actual_coverage:.6f} "
            f"eligible={metadata['eligible_cohort_count']} "
            f"labeled={metadata['labeled_cohort_count']}"
        )

    if metadata["sample_count"] < min_samples:
        raise ValueError(
            "not enough labeled point-in-time offline feature rows for training: "
            f"required={min_samples} actual={metadata['sample_count']}"
        )

    if metadata["fail_count"] < min_fail_samples:
        raise ValueError(
            "not enough fail samples for training: "
            f"required={min_fail_samples} actual={metadata['fail_count']}"
        )

    if metadata["pass_count"] < min_pass_samples:
        raise ValueError(
            "not enough pass samples for training: "
            f"required={min_pass_samples} actual={metadata['pass_count']}"
        )


def select_latest_labeled_training_rows(
        frame: pd.DataFrame,
        *,
        dataset_id: str
) -> pd.DataFrame:
    labeled = frame.loc[frame["label_event_id"].notna()].copy()
    if len(labeled) < MAX_DEVELOPMENT_SAMPLES:
        raise ValueError(
            "not enough labeled rows in training dataset: "
            f"dataset_id={dataset_id} "
            f"required={MAX_DEVELOPMENT_SAMPLES} actual={len(labeled)}"
        )

    order_columns = [
        "snapshot_available_at",
        "sample_id",
        "serving_snapshot_id",
    ]
    selected = (
        labeled.sort_values(
            order_columns,
            ascending=[False, False, False],
            kind="mergesort",
        )
        .head(MAX_DEVELOPMENT_SAMPLES)
        .sort_values(
            order_columns,
            ascending=[True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return selected


def _selection_hash(dataset_id: str, selected: pd.DataFrame) -> str:
    members = selected[
        ["sample_id", "serving_snapshot_id", "snapshot_version"]
    ].to_dict(orient="records")
    canonical = json.dumps(
        {
            "dataset_id": dataset_id,
            "selection": DEVELOPMENT_SAMPLE_SELECTION,
            "limit": MAX_DEVELOPMENT_SAMPLES,
            "members": members,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:v1:{hashlib.sha256(canonical).hexdigest()}"