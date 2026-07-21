"""Point-in-time training-source dataset contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from secom_mlops_common.schemas.secom import (
    FEATURE_KEYS,
    FEATURE_KEY_SET,
    MODEL_COLUMNS,
    NUM_FEATURES,
    normalize_feature_value,
    parse_feature_object,
)

DATASET_TYPE = "training"
DATASET_SCHEMA_VERSION = "training_source_dataset.v1"
SELECTOR_VERSION = "sample_first_complete_point_in_time.v1"
DEFAULT_LABEL_MATURITY_SECONDS = 120.0
DEFAULT_MIN_LABELED_SAMPLES = 1000
DEFAULT_MIN_LABEL_COVERAGE = 0.95
DEFAULT_MIN_FAIL_SAMPLES = 20
DEFAULT_MIN_PASS_SAMPLES = 20
POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1

METADATA_COLUMNS = [
    "dataset_id",
    "sample_id",
    "serving_snapshot_id",
    "snapshot_version",
    "feature_hash",
    "simulation_run_id",
    "drift_segment",
    "snapshot_time",
    "snapshot_available_at",
    "window_start",
    "window_end",
    "serving_missing_count",
    "label_event_id",
    "label_revision",
    "label_measured_at",
    "label_available_at",
    "actual_value",
    "actual_label",
]
DATASET_COLUMNS = METADATA_COLUMNS + list(MODEL_COLUMNS)


@dataclass(frozen=True)
class DatasetBuildConfig:
    cohort_start_time: float
    cutoff_time: float
    label_maturity_seconds: float = DEFAULT_LABEL_MATURITY_SECONDS
    min_labeled_samples: int = DEFAULT_MIN_LABELED_SAMPLES
    min_label_coverage: float = DEFAULT_MIN_LABEL_COVERAGE
    min_fail_samples: int = DEFAULT_MIN_FAIL_SAMPLES
    min_pass_samples: int = DEFAULT_MIN_PASS_SAMPLES
    simulation_run_id: str | None = None
    drift_segment: str | None = None

    @property
    def cohort_end_time(self) -> float:
        return self.cutoff_time - self.label_maturity_seconds

    def validate(self) -> None:
        finite_values = {
            "cohort_start_time": self.cohort_start_time,
            "cutoff_time": self.cutoff_time,
            "label_maturity_seconds": self.label_maturity_seconds,
            "min_label_coverage": self.min_label_coverage,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.cohort_start_time < 0.0:
            raise ValueError("cohort_start_time must be >= 0")
        if self.cutoff_time < 0.0:
            raise ValueError("cutoff_time must be >= 0")
        if self.label_maturity_seconds < 0.0:
            raise ValueError("label_maturity_seconds must be >= 0")
        if self.cohort_end_time < self.cohort_start_time:
            raise ValueError(
                "cohort_start_time must be <= cutoff_time - label_maturity_seconds"
            )
        if self.min_labeled_samples < 1:
            raise ValueError("min_labeled_samples must be >= 1")
        if not 0.0 <= self.min_label_coverage <= 1.0:
            raise ValueError("min_label_coverage must be between 0 and 1")
        if self.min_fail_samples < 1:
            raise ValueError("min_fail_samples must be >= 1")
        if self.min_pass_samples < 1:
            raise ValueError("min_pass_samples must be >= 1")


@dataclass(frozen=True)
class DatasetMember:
    sample_id: str
    serving_snapshot_id: str
    snapshot_version: int
    feature_hash: str
    snapshot_time: float
    snapshot_available_at: float
    window_start: float
    window_end: float
    feature_count: int
    serving_missing_count: int
    simulation_run_id: str | None
    drift_segment: str | None
    label_event_id: str | None
    label_revision: int | None
    label_measured_at: float | None
    label_available_at: float | None
    actual_value: int | None
    actual_label: str | None
    features_json: Any | None = None

    @property
    def is_labeled(self) -> bool:
        return self.label_event_id is not None

    def identity_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "serving_snapshot_id": self.serving_snapshot_id,
            "snapshot_version": self.snapshot_version,
            "feature_hash": self.feature_hash,
            "snapshot_time": self.snapshot_time,
            "snapshot_available_at": self.snapshot_available_at,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "feature_count": self.feature_count,
            "serving_missing_count": self.serving_missing_count,
            "simulation_run_id": self.simulation_run_id,
            "drift_segment": self.drift_segment,
            "label_event_id": self.label_event_id,
            "label_revision": self.label_revision,
            "label_measured_at": self.label_measured_at,
            "label_available_at": self.label_available_at,
            "actual_value": self.actual_value,
            "actual_label": self.actual_label,
        }


@dataclass(frozen=True)
class DatasetReadiness:
    ready: bool
    reasons: tuple[str, ...]
    stats: dict[str, Any]


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    manifest_hash: str
    identity_payload: dict[str, Any]


def member_from_row(row: dict[str, Any]) -> DatasetMember:
    label_event_id = row.get("label_event_id")
    label_values = {
        "label_revision": row.get("label_revision"),
        "label_measured_at": row.get("label_measured_at"),
        "label_available_at": row.get("label_available_at"),
        "actual_value": row.get("actual_value"),
        "actual_label": row.get("actual_label"),
    }
    if label_event_id is None and any(value is not None for value in label_values.values()):
        raise ValueError(
            f"unlabeled row has partial label data: sample_id={row['sample_id']}"
        )
    if label_event_id is not None and any(value is None for value in label_values.values()):
        raise ValueError(
            f"labeled row is missing label data: sample_id={row['sample_id']}"
        )

    actual_value = int(label_values["actual_value"]) if label_event_id is not None else None
    actual_label = str(label_values["actual_label"]) if label_event_id is not None else None
    expected_label = {
        NEGATIVE_CLASS: "pass",
        POSITIVE_CLASS: "fail",
    }.get(actual_value)
    if label_event_id is not None and expected_label is None:
        raise ValueError(
            f"unexpected label value: sample_id={row['sample_id']} actual_value={actual_value}"
        )
    if expected_label is not None and actual_label != expected_label:
        raise ValueError(
            "label value and name do not match: "
            f"sample_id={row['sample_id']} actual_value={actual_value} actual_label={actual_label}"
        )

    return DatasetMember(
        sample_id=str(row["sample_id"]),
        serving_snapshot_id=str(row["serving_snapshot_id"]),
        snapshot_version=int(row["snapshot_version"]),
        feature_hash=str(row["feature_hash"]),
        snapshot_time=float(row["snapshot_time"]),
        snapshot_available_at=float(row["snapshot_available_at"]),
        window_start=float(row["window_start"]),
        window_end=float(row["window_end"]),
        feature_count=int(row["feature_count"]),
        serving_missing_count=int(row["serving_missing_count"]),
        simulation_run_id=(
            str(row["simulation_run_id"])
            if row.get("simulation_run_id") is not None
            else None
        ),
        drift_segment=(
            str(row["drift_segment"])
            if row.get("drift_segment") is not None
            else None
        ),
        label_event_id=str(label_event_id) if label_event_id is not None else None,
        label_revision=(
            int(label_values["label_revision"])
            if label_event_id is not None
            else None
        ),
        label_measured_at=(
            float(label_values["label_measured_at"])
            if label_event_id is not None
            else None
        ),
        label_available_at=(
            float(label_values["label_available_at"])
            if label_event_id is not None
            else None
        ),
        actual_value=actual_value,
        actual_label=actual_label,
        features_json=row.get("features_json"),
    )


def summarize_members(members: Iterable[DatasetMember]) -> dict[str, Any]:
    member_list = list(members)
    labeled = [member for member in member_list if member.is_labeled]
    fail_count = sum(member.actual_value == POSITIVE_CLASS for member in labeled)
    pass_count = sum(member.actual_value == NEGATIVE_CLASS for member in labeled)
    if fail_count + pass_count != len(labeled):
        raise ValueError("every labeled dataset row must be either pass or fail")

    snapshot_times = [member.snapshot_time for member in member_list]
    snapshot_available_times = [member.snapshot_available_at for member in member_list]
    label_available_times = [
        member.label_available_at
        for member in labeled
        if member.label_available_at is not None
    ]
    eligible_count = len(member_list)
    labeled_count = len(labeled)

    return {
        "eligible_sample_count": eligible_count,
        "labeled_sample_count": labeled_count,
        "unlabeled_sample_count": eligible_count - labeled_count,
        "label_coverage": labeled_count / eligible_count if eligible_count else 0.0,
        "fail_count": fail_count,
        "pass_count": pass_count,
        "snapshot_time_min": min(snapshot_times) if snapshot_times else None,
        "snapshot_time_max": max(snapshot_times) if snapshot_times else None,
        "snapshot_available_at_min": (
            min(snapshot_available_times) if snapshot_available_times else None
        ),
        "snapshot_available_at_max": (
            max(snapshot_available_times) if snapshot_available_times else None
        ),
        "label_available_at_min": (
            min(label_available_times) if label_available_times else None
        ),
        "label_available_at_max": (
            max(label_available_times) if label_available_times else None
        ),
    }


def evaluate_readiness(
        config: DatasetBuildConfig,
        members: Iterable[DatasetMember],
) -> DatasetReadiness:
    stats = summarize_members(members)
    reasons: list[str] = []

    if stats["labeled_sample_count"] < config.min_labeled_samples:
        reasons.append(
            "not enough labeled samples: "
            f"required={config.min_labeled_samples} "
            f"actual={stats['labeled_sample_count']}"
        )
    if stats["label_coverage"] < config.min_label_coverage:
        reasons.append(
            "label coverage below minimum: "
            f"required={config.min_label_coverage:.6f} "
            f"actual={stats['label_coverage']:.6f}"
        )
    if stats["fail_count"] < config.min_fail_samples:
        reasons.append(
            "not enough fail samples: "
            f"required={config.min_fail_samples} actual={stats['fail_count']}"
        )
    if stats["pass_count"] < config.min_pass_samples:
        reasons.append(
            "not enough pass samples: "
            f"required={config.min_pass_samples} actual={stats['pass_count']}"
        )

    return DatasetReadiness(
        ready=not reasons,
        reasons=tuple(reasons),
        stats=stats,
    )


def build_dataset_identity(
        config: DatasetBuildConfig,
        members: Iterable[DatasetMember],
) -> DatasetIdentity:
    ordered_members = sorted(
        (member.identity_record() for member in members),
        key=lambda member: (
            member["snapshot_available_at"],
            member["sample_id"],
            member["serving_snapshot_id"],
        ),
    )
    identity_payload = {
        "dataset_type": DATASET_TYPE,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "temporal_spine": "serving_feature_snapshots",
        "decision_time_column": "snapshot_available_at",
        "snapshot_selection": "first_complete_per_sample",
        "label_selection": "max_revision_available_at_cutoff",
        "cohort_start_time": config.cohort_start_time,
        "label_maturity_seconds": config.label_maturity_seconds,
        "simulation_run_id": config.simulation_run_id,
        "drift_segment": config.drift_segment,
        "members": ordered_members,
    }
    canonical = json.dumps(
        identity_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return DatasetIdentity(
        dataset_id=f"training_{digest[:16]}",
        manifest_hash=f"sha256:v1:{digest}",
        identity_payload=identity_payload,
    )


def build_manifest(
        config: DatasetBuildConfig,
        identity: DatasetIdentity,
        stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": identity.dataset_id,
        "manifest_hash": identity.manifest_hash,
        "identity": identity.identity_payload,
        "build_context": {
            "cutoff_time": config.cutoff_time,
            "cohort_end_time": config.cohort_end_time,
            "min_labeled_samples": config.min_labeled_samples,
            "min_label_coverage": config.min_label_coverage,
            "min_fail_samples": config.min_fail_samples,
            "min_pass_samples": config.min_pass_samples,
        },
        "stats": stats,
    }


def build_dataset_frame(
        dataset_id: str,
        members: Iterable[DatasetMember],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for member in members:
        if member.features_json is None:
            raise ValueError(
                f"dataset row has no feature payload: sample_id={member.sample_id}"
            )
        if member.feature_count != NUM_FEATURES:
            raise ValueError(
                "complete snapshot must contain all feature keys: "
                f"sample_id={member.sample_id} feature_count={member.feature_count}"
            )

        raw_features = parse_feature_object(
            member.features_json,
            sample_id=member.sample_id,
        )
        actual_keys = set(raw_features)
        unexpected_keys = sorted(actual_keys - FEATURE_KEY_SET)
        missing_keys = sorted(FEATURE_KEY_SET - actual_keys)
        if unexpected_keys:
            raise ValueError(
                "unexpected feature keys: "
                f"sample_id={member.sample_id} keys={unexpected_keys[:5]}"
            )
        if missing_keys:
            raise ValueError(
                "missing feature keys: "
                f"sample_id={member.sample_id} keys={missing_keys[:5]}"
            )

        normalized_features = [
            normalize_feature_value(
                raw_features[key],
                sample_id=member.sample_id,
                feature_key=key,
            )
            for key in FEATURE_KEYS
        ]
        computed_missing_count = sum(value is None for value in normalized_features)
        if computed_missing_count != member.serving_missing_count:
            raise ValueError(
                "snapshot missing_count does not match feature payload: "
                f"sample_id={member.sample_id} "
                f"stored={member.serving_missing_count} "
                f"computed={computed_missing_count}"
            )

        row = {
            "dataset_id": dataset_id,
            "sample_id": member.sample_id,
            "serving_snapshot_id": member.serving_snapshot_id,
            "snapshot_version": member.snapshot_version,
            "feature_hash": member.feature_hash,
            "simulation_run_id": member.simulation_run_id,
            "drift_segment": member.drift_segment,
            "snapshot_time": member.snapshot_time,
            "snapshot_available_at": member.snapshot_available_at,
            "window_start": member.window_start,
            "window_end": member.window_end,
            "serving_missing_count": member.serving_missing_count,
            "label_event_id": member.label_event_id,
            "label_revision": member.label_revision,
            "label_measured_at": member.label_measured_at,
            "label_available_at": member.label_available_at,
            "actual_value": member.actual_value,
            "actual_label": member.actual_label,
        }
        row.update(dict(zip(MODEL_COLUMNS, normalized_features)))
        rows.append(row)

    frame = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    if frame.empty:
        return frame

    for column in ("dataset_id", "sample_id", "serving_snapshot_id", "feature_hash"):
        frame[column] = frame[column].astype("string")
    for column in ("simulation_run_id", "drift_segment", "label_event_id", "actual_label"):
        frame[column] = frame[column].astype("string")
    for column in ("snapshot_version", "serving_missing_count"):
        frame[column] = frame[column].astype("int64")
    for column in ("label_revision", "actual_value"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    for column in (
        "snapshot_time",
        "snapshot_available_at",
        "window_start",
        "window_end",
        "label_measured_at",
        "label_available_at",
    ):
        frame[column] = pd.array(frame[column], dtype="Float64")
    for column in MODEL_COLUMNS:
        frame[column] = frame[column].astype("float64")
    return frame
