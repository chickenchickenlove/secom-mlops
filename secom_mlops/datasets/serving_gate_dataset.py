"""Immutable serving-gate evaluation dataset contract."""

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

DATASET_TYPE = "serving_gate"
DATASET_SCHEMA_VERSION = "serving_gate_dataset.v1"
SELECTOR_VERSION = "first_release_decision_per_sample_snapshot.v1"
DECISION_SELECTION = "first_release_decision_per_sample_snapshot"
DEFAULT_LABEL_MATURITY_SECONDS = 120.0
DEFAULT_MIN_DECISIONS = 1000
DEFAULT_MIN_LABELED_DECISIONS = 1000
DEFAULT_MIN_LABEL_COVERAGE = 0.95
DEFAULT_MIN_FAIL_SAMPLES = 20
DEFAULT_MIN_PASS_SAMPLES = 20
POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1


@dataclass(frozen=True)
class ServingGateDatasetConfig:
    cohort_start_time: float
    cutoff_time: float
    label_maturity_seconds: float = DEFAULT_LABEL_MATURITY_SECONDS
    min_decisions: int = DEFAULT_MIN_DECISIONS
    min_labeled_decisions: int = DEFAULT_MIN_LABELED_DECISIONS
    min_label_coverage: float = DEFAULT_MIN_LABEL_COVERAGE
    min_fail_samples: int = DEFAULT_MIN_FAIL_SAMPLES
    min_pass_samples: int = DEFAULT_MIN_PASS_SAMPLES

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
        if self.cohort_end_time <= self.cohort_start_time:
            raise ValueError(
                "cohort_start_time must be less than "
                "cutoff_time - label_maturity_seconds"
            )
        if self.min_decisions < DEFAULT_MIN_DECISIONS:
            raise ValueError(
                f"min_decisions must be >= {DEFAULT_MIN_DECISIONS}"
            )
        if self.min_labeled_decisions < DEFAULT_MIN_LABELED_DECISIONS:
            raise ValueError(
                "min_labeled_decisions must be >= "
                f"{DEFAULT_MIN_LABELED_DECISIONS}"
            )
        if not 0.0 <= self.min_label_coverage <= 1.0:
            raise ValueError("min_label_coverage must be between 0 and 1")
        if self.min_fail_samples < 1:
            raise ValueError("min_fail_samples must be >= 1")
        if self.min_pass_samples < 1:
            raise ValueError("min_pass_samples must be >= 1")


@dataclass(frozen=True)
class ServingGateDatasetMember:
    prediction_id: str
    request_id: str
    sample_id: str
    serving_snapshot_id: str
    snapshot_version: int
    feature_hash: str
    source_model_run_id: str
    runtime_slot: str
    source_threshold: float
    predicted_at: float
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
            "prediction_id": self.prediction_id,
            "request_id": self.request_id,
            "sample_id": self.sample_id,
            "serving_snapshot_id": self.serving_snapshot_id,
            "snapshot_version": self.snapshot_version,
            "feature_hash": self.feature_hash,
            "source_model_run_id": self.source_model_run_id,
            "runtime_slot": self.runtime_slot,
            "source_threshold": self.source_threshold,
            "predicted_at": self.predicted_at,
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
class ServingGateDatasetReadiness:
    ready: bool
    reasons: tuple[str, ...]
    stats: dict[str, Any]


@dataclass(frozen=True)
class ServingGateDatasetIdentity:
    dataset_id: str
    manifest_hash: str
    identity_payload: dict[str, Any]


def member_from_row(row: dict[str, Any]) -> ServingGateDatasetMember:
    prediction_id = str(row["prediction_id"])
    sample_id = str(row["sample_id"])
    snapshot_version = int(row["snapshot_version"])

    if row.get("has_conflicting_snapshot_identity") is True:
        raise RuntimeError(
            "repeated release decisions have conflicting snapshot identity: "
            f"prediction_id={prediction_id} sample_id={sample_id} "
            f"snapshot_version={snapshot_version}"
        )
    if row.get("stored_serving_snapshot_id") is None:
        raise RuntimeError(
            "release decision has no exact serving snapshot: "
            f"prediction_id={prediction_id} sample_id={sample_id} "
            f"snapshot_version={snapshot_version}"
        )
    if str(row["stored_serving_snapshot_id"]) != str(row["serving_snapshot_id"]):
        raise RuntimeError(
            f"serving snapshot identity mismatch: prediction_id={prediction_id}"
        )
    if str(row["stored_sample_id"]) != sample_id:
        raise RuntimeError(f"serving snapshot sample mismatch: prediction_id={prediction_id}")
    if int(row["stored_snapshot_version"]) != snapshot_version:
        raise RuntimeError(f"serving snapshot version mismatch: prediction_id={prediction_id}")
    if str(row["snapshot_feature_hash"]) != str(row["feature_hash"]):
        raise RuntimeError(
            f"prediction and snapshot feature_hash mismatch: prediction_id={prediction_id}"
        )
    if row["snapshot_status"] != "complete" or row["is_complete"] is not True:
        raise RuntimeError(
            f"release decision must reference a complete snapshot: prediction_id={prediction_id}"
        )
    if int(row["feature_count"]) != NUM_FEATURES:
        raise RuntimeError(
            "complete serving snapshot must contain all feature keys: "
            f"prediction_id={prediction_id} feature_count={row['feature_count']}"
        )
    if str(row["runtime_slot"]) != "release":
        raise RuntimeError(f"unexpected runtime slot: prediction_id={prediction_id}")

    source_threshold = float(row["source_threshold"])
    if not math.isfinite(source_threshold) or not 0.0 <= source_threshold <= 1.0:
        raise RuntimeError(f"invalid source threshold: prediction_id={prediction_id}")

    predicted_at = float(row["predicted_at"])
    snapshot_available_at = float(row["snapshot_available_at"])
    if snapshot_available_at > predicted_at:
        raise RuntimeError(
            "snapshot became available after its release decision: "
            f"prediction_id={prediction_id}"
        )

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
            f"unlabeled decision has partial label data: prediction_id={prediction_id}"
        )
    if label_event_id is not None and any(value is None for value in label_values.values()):
        raise ValueError(
            f"labeled decision is missing label data: prediction_id={prediction_id}"
        )

    actual_value = int(label_values["actual_value"]) if label_event_id is not None else None
    actual_label = str(label_values["actual_label"]) if label_event_id is not None else None
    expected_label = {NEGATIVE_CLASS: "pass", POSITIVE_CLASS: "fail"}.get(actual_value)
    if label_event_id is not None and expected_label is None:
        raise ValueError(
            f"unexpected label value: prediction_id={prediction_id} actual_value={actual_value}"
        )
    if expected_label is not None and actual_label != expected_label:
        raise ValueError(
            "label value and name do not match: "
            f"prediction_id={prediction_id} actual_value={actual_value} "
            f"actual_label={actual_label}"
        )

    return ServingGateDatasetMember(
        prediction_id=prediction_id,
        request_id=str(row["request_id"]),
        sample_id=sample_id,
        serving_snapshot_id=str(row["serving_snapshot_id"]),
        snapshot_version=snapshot_version,
        feature_hash=str(row["feature_hash"]),
        source_model_run_id=str(row["source_model_run_id"]),
        runtime_slot=str(row["runtime_slot"]),
        source_threshold=source_threshold,
        predicted_at=predicted_at,
        snapshot_time=float(row["snapshot_time"]),
        snapshot_available_at=snapshot_available_at,
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
        label_revision=(int(label_values["label_revision"]) if label_event_id is not None else None),
        label_measured_at=(float(label_values["label_measured_at"]) if label_event_id is not None else None),
        label_available_at=(float(label_values["label_available_at"]) if label_event_id is not None else None),
        actual_value=actual_value,
        actual_label=actual_label,
        features_json=row.get("features_json"),
    )


def summarize_members(members: Iterable[ServingGateDatasetMember]) -> dict[str, Any]:
    member_list = list(members)
    labeled = [member for member in member_list if member.is_labeled]
    fail_count = sum(member.actual_value == POSITIVE_CLASS for member in labeled)
    pass_count = sum(member.actual_value == NEGATIVE_CLASS for member in labeled)
    decision_count = len(member_list)
    labeled_count = len(labeled)
    decision_times = [member.predicted_at for member in member_list]
    snapshot_available_times = [member.snapshot_available_at for member in member_list]
    label_available_times = [
        member.label_available_at
        for member in labeled
        if member.label_available_at is not None
    ]

    return {
        "decision_count": decision_count,
        "labeled_decision_count": labeled_count,
        "unlabeled_decision_count": decision_count - labeled_count,
        "unique_sample_count": len({member.sample_id for member in member_list}),
        "label_coverage": labeled_count / decision_count if decision_count else 0.0,
        "fail_count": fail_count,
        "pass_count": pass_count,
        "decision_time_min": min(decision_times) if decision_times else None,
        "decision_time_max": max(decision_times) if decision_times else None,
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
        "source_model_run_ids": sorted(
            {member.source_model_run_id for member in member_list}
        ),
        "source_thresholds": sorted(
            {member.source_threshold for member in member_list}
        ),
    }


def evaluate_readiness(
        config: ServingGateDatasetConfig,
        members: Iterable[ServingGateDatasetMember],
) -> ServingGateDatasetReadiness:
    stats = summarize_members(members)
    reasons: list[str] = []
    if stats["decision_count"] < config.min_decisions:
        reasons.append(
            "not enough release decisions: "
            f"required={config.min_decisions} actual={stats['decision_count']}"
        )
    if stats["labeled_decision_count"] < config.min_labeled_decisions:
        reasons.append(
            "not enough labeled release decisions: "
            f"required={config.min_labeled_decisions} "
            f"actual={stats['labeled_decision_count']}"
        )
    if stats["label_coverage"] < config.min_label_coverage:
        reasons.append(
            "label coverage below minimum: "
            f"required={config.min_label_coverage:.6f} "
            f"actual={stats['label_coverage']:.6f}"
        )
    if stats["fail_count"] < config.min_fail_samples:
        reasons.append(
            f"not enough fail samples: required={config.min_fail_samples} "
            f"actual={stats['fail_count']}"
        )
    if stats["pass_count"] < config.min_pass_samples:
        reasons.append(
            f"not enough pass samples: required={config.min_pass_samples} "
            f"actual={stats['pass_count']}"
        )
    return ServingGateDatasetReadiness(
        ready=not reasons,
        reasons=tuple(reasons),
        stats=stats,
    )


def build_dataset_identity(
        config: ServingGateDatasetConfig,
        members: Iterable[ServingGateDatasetMember],
) -> ServingGateDatasetIdentity:
    ordered_members = sorted(
        (member.identity_record() for member in members),
        key=lambda member: (member["predicted_at"], member["prediction_id"]),
    )
    identity_payload = {
        "dataset_type": DATASET_TYPE,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "temporal_spine": "prediction_logs",
        "decision_time_column": "predicted_at",
        "runtime_slot": "release",
        "decision_selection": DECISION_SELECTION,
        "deduplication_key": ["sample_id", "snapshot_version"],
        "deduplication_order": ["predicted_at ASC", "prediction_id ASC"],
        "label_selection": "max_revision_available_at_cutoff",
        "cohort_start_time": config.cohort_start_time,
        "cohort_end_time": config.cohort_end_time,
        "label_maturity_seconds": config.label_maturity_seconds,
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
    return ServingGateDatasetIdentity(
        dataset_id=f"serving_gate_{digest[:16]}",
        manifest_hash=f"sha256:v1:{digest}",
        identity_payload=identity_payload,
    )


def build_manifest(
        config: ServingGateDatasetConfig,
        identity: ServingGateDatasetIdentity,
        stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": identity.dataset_id,
        "manifest_hash": identity.manifest_hash,
        "identity": identity.identity_payload,
        "build_context": {
            "cutoff_time": config.cutoff_time,
            "min_decisions": config.min_decisions,
            "min_labeled_decisions": config.min_labeled_decisions,
            "min_label_coverage": config.min_label_coverage,
            "min_fail_samples": config.min_fail_samples,
            "min_pass_samples": config.min_pass_samples,
        },
        "stats": stats,
    }


METADATA_COLUMNS = [
    "dataset_id",
    "prediction_id",
    "request_id",
    "sample_id",
    "serving_snapshot_id",
    "snapshot_version",
    "feature_hash",
    "source_model_run_id",
    "runtime_slot",
    "source_threshold",
    "predicted_at",
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


def build_dataset_frame(
        dataset_id: str,
        members: Iterable[ServingGateDatasetMember],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_snapshot_versions: set[tuple[str, int]] = set()
    for member in members:
        decision_key = (member.sample_id, member.snapshot_version)
        if decision_key in seen_snapshot_versions:
            raise ValueError(
                "serving-gate dataset contains duplicate sample snapshot: "
                f"sample_id={member.sample_id} snapshot_version={member.snapshot_version}"
            )
        seen_snapshot_versions.add(decision_key)
        if member.features_json is None:
            raise ValueError(
                f"dataset row has no feature payload: prediction_id={member.prediction_id}"
            )

        raw_features = parse_feature_object(member.features_json, sample_id=member.sample_id)
        actual_keys = set(raw_features)

        if unexpected_keys := sorted(actual_keys - FEATURE_KEY_SET):
            raise ValueError(
                f"unexpected feature keys: prediction_id={member.prediction_id} "
                f"keys={unexpected_keys[:5]}"
            )
        if missing_keys := sorted(FEATURE_KEY_SET - actual_keys):
            raise ValueError(
                f"missing feature keys: prediction_id={member.prediction_id} "
                f"keys={missing_keys[:5]}"
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
                f"prediction_id={member.prediction_id} "
                f"stored={member.serving_missing_count} computed={computed_missing_count}"
            )

        row = {
            "dataset_id": dataset_id,
            **member.identity_record(),
        }
        row.pop("feature_count")
        row.update(dict(zip(MODEL_COLUMNS, normalized_features)))
        rows.append(row)

    frame = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    if frame.empty:
        return frame

    for column in (
            "dataset_id", "prediction_id", "request_id", "sample_id",
            "serving_snapshot_id", "feature_hash", "source_model_run_id", "runtime_slot",
    ):
        frame[column] = frame[column].astype("string")
    for column in ("simulation_run_id", "drift_segment", "label_event_id", "actual_label"):
        frame[column] = frame[column].astype("string")
    for column in ("snapshot_version", "serving_missing_count"):
        frame[column] = frame[column].astype("int64")
    for column in ("label_revision", "actual_value"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    for column in (
            "source_threshold", "predicted_at", "snapshot_time", "snapshot_available_at",
            "window_start", "window_end", "label_measured_at", "label_available_at",
    ):
        frame[column] = pd.array(frame[column], dtype="Float64")
    for column in MODEL_COLUMNS:
        frame[column] = frame[column].astype("float64")
    return frame
