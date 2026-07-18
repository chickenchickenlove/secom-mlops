"""Train a candidate model from first-complete serving feature snapshots."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from secom_mlops.monitor.db import connect
from secom_mlops.models.secom_pyfunc_model import SECOMFailDetectionPyfunc
from secom_mlops_common.cli.validators import (
    non_negative_float,
    positive_int,
    positive_int_list,
    probability_list,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CANDIDATE_ALIAS,
    ENV_ML_CANDIDATE_GROUP,
    ENV_ML_TRAINING_JOB_ID,
    MODEL_ROLE_CANDIDATE,
    MODEL_ROLES,
    get_env_value,
    resolve_model_alias,
    resolve_model_name,
    resolve_model_role,
    resolve_tracking_uri,
)
from secom_mlops_common.schemas.secom import (
    FEATURE_KEYS,
    FEATURE_KEY_SET,
    MODEL_COLUMNS,
    NUM_FEATURES,
    normalize_feature_value,
    parse_feature_object,
)

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1

DEFAULT_N_ESTIMATORS = "100,300"
DEFAULT_MIN_SAMPLES_LEAF = "1,3"
DEFAULT_THRESHOLDS = "0.1,0.2,0.3,0.4,0.5"
MAX_DEVELOPMENT_SAMPLES = 1000
VALIDATION_SIZE = 0.2
DEFAULT_MIN_LABEL_COVERAGE = 0.95
DEVELOPMENT_SAMPLE_SELECTION = "latest_eligible_available_at"
TRAIN_SOURCE = "serving_feature_snapshot_history"
TRAINING_SPINE = "serving_feature_snapshots"
TRAINING_DECISION_TIME = "serving_feature_snapshots.available_at"
SNAPSHOT_SELECTION = "first_complete"
LABEL_SELECTION = "available_at_lte_cutoff_then_max_revision"
GATE_SOURCE = "serving_feature_snapshots"


def coverage_float(raw_value: str) -> float:
    value = float(raw_value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and between 0 and 1"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument(
        "--model-alias",
        default=resolve_model_alias(default=DEFAULT_CANDIDATE_ALIAS),
    )
    parser.add_argument("--model-role", default=resolve_model_role())
    parser.add_argument("--candidate-group", default=get_env_value(ENV_ML_CANDIDATE_GROUP))
    parser.add_argument("--training-job-id", default=get_env_value(ENV_ML_TRAINING_JOB_ID))

    parser.add_argument("--cohort-start-time", type=non_negative_float, required=True)
    parser.add_argument("--cutoff-time", type=non_negative_float, required=True)
    parser.add_argument("--label-maturity-seconds", type=non_negative_float, required=True)
    parser.add_argument("--simulation-run-id", default=None)
    parser.add_argument("--drift-segment", default=None)

    parser.add_argument("--min-samples", type=positive_int, default=500)
    parser.add_argument(
        "--min-label-coverage",
        type=coverage_float,
        default=DEFAULT_MIN_LABEL_COVERAGE,
    )
    parser.add_argument("--min-fail-samples", type=positive_int, default=20)
    parser.add_argument("--min-pass-samples", type=positive_int, default=20)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--n-estimators", default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--min-samples-leaf", default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    cohort_end_time = args.cutoff_time - args.label_maturity_seconds
    if cohort_end_time < args.cohort_start_time:
        raise ValueError(
            "cohort_start_time must be <= cutoff_time - label_maturity_seconds"
        )

    if args.model_role not in MODEL_ROLES:
        raise ValueError("model_role must be one of: candidate, champion")

    if args.min_samples > MAX_DEVELOPMENT_SAMPLES:
        raise ValueError(
            "min_samples must be <= development sample limit: "
            f"min_samples={args.min_samples} "
            f"development_sample_limit={MAX_DEVELOPMENT_SAMPLES}"
        )

    if args.model_role == MODEL_ROLE_CANDIDATE:
        missing = []
        if not args.candidate_group:
            missing.append("--candidate-group or ML_CANDIDATE_GROUP")
        if not args.training_job_id:
            missing.append("--training-job-id or ML_TRAINING_JOB_ID")
        if missing:
            raise ValueError("candidate training requires " + ", ".join(missing))


def load_labeled_point_in_time_features(
        args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    snapshot_filters = [
        "s.is_complete = TRUE",
        "s.snapshot_status = 'complete'",
    ]
    params: list[Any] = []

    if args.simulation_run_id is not None:
        snapshot_filters.append("s.simulation_run_id = %s")
        params.append(args.simulation_run_id)

    if args.drift_segment is not None:
        snapshot_filters.append("s.drift_segment = %s")
        params.append(args.drift_segment)

    cohort_end_time = args.cutoff_time - args.label_maturity_seconds
    params.extend([
        args.cohort_start_time,
        cohort_end_time,
        args.cutoff_time,
    ])

    params.append(MAX_DEVELOPMENT_SAMPLES)

    sql = f"""
    WITH ranked_complete_snapshots AS (
      SELECT
        s.serving_snapshot_id,
        s.sample_id,
        s.snapshot_version,
        s.snapshot_time,
        s.window_start,
        s.window_end,
        s.available_at,
        s.feature_count,
        s.missing_count,
        s.features_json,
        s.simulation_run_id,
        s.drift_segment,
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
        AND available_at >= %s
        AND available_at <= %s
    ),
    ranked_labels AS (
      SELECT
        le.*,
        ROW_NUMBER() OVER (
          PARTITION BY le.sample_id
          ORDER BY le.label_revision DESC
        ) AS label_rank
      FROM label_events le
      WHERE le.available_at <= %s
    ),
    labels_at_cutoff AS (
      SELECT
        *
      FROM ranked_labels
      WHERE label_rank = 1
    ),
    latest_eligible_cohort AS (
      SELECT
        *
      FROM first_complete_cohort
      ORDER BY
        available_at DESC,
        sample_id DESC
      LIMIT %s
    ),
    point_in_time_development_cohort AS (
      SELECT
        s.serving_snapshot_id,
        s.sample_id,
        s.snapshot_version,
        s.snapshot_time,
        s.window_start,
        s.window_end,
        s.available_at AS decision_time,
        s.feature_count,
        s.missing_count,
        s.features_json,
        s.simulation_run_id,
        s.drift_segment,
        a.label_event_id,
        a.label_revision,
        a.measured_at AS label_measured_at,
        a.available_at AS label_available_at,
        a.actual_value,
        a.actual_label
      FROM latest_eligible_cohort s
      LEFT JOIN labels_at_cutoff a
        ON a.sample_id = s.sample_id
    )
    SELECT
      *
    FROM point_in_time_development_cohort
    ORDER BY
      decision_time ASC,
      sample_id ASC;
    """

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(
            "no first-complete serving snapshots found for training cohort: "
            f"cohort_start_time={args.cohort_start_time} "
            f"cohort_end_time={cohort_end_time} "
            f"cutoff_time={args.cutoff_time} "
            f"label_maturity_seconds={args.label_maturity_seconds}"
        )

    eligible_sample_ids: list[str] = []
    eligible_decision_times: list[float] = []
    sample_ids: list[str] = []
    snapshot_ids: list[str] = []
    snapshot_versions: list[int] = []
    feature_rows: list[list[float | None]] = []
    labels: list[int] = []
    snapshot_times: list[float] = []
    decision_times: list[float] = []
    window_starts: list[float] = []
    window_ends: list[float] = []
    serving_missing_counts: list[int] = []
    label_event_ids: list[str] = []
    label_revisions: list[int] = []
    label_measured_times: list[float] = []
    label_available_times: list[float] = []

    for row in rows:
        serving_snapshot_id = str(row[0])
        sample_id = str(row[1])
        snapshot_version = int(row[2])
        snapshot_time = float(row[3])
        window_start = float(row[4])
        window_end = float(row[5])
        decision_time = float(row[6])
        feature_count = int(row[7])
        stored_missing_count = int(row[8])
        raw_features = parse_feature_object(row[9], sample_id=sample_id)

        eligible_sample_ids.append(sample_id)
        eligible_decision_times.append(decision_time)

        if feature_count != NUM_FEATURES:
            raise ValueError(
                "complete serving snapshot must contain all feature keys: "
                f"sample_id={sample_id} feature_count={feature_count}"
            )

        actual_feature_keys = set(raw_features)
        unexpected_feature_keys = sorted(actual_feature_keys - FEATURE_KEY_SET)
        missing_feature_keys = sorted(FEATURE_KEY_SET - actual_feature_keys)
        if unexpected_feature_keys:
            raise ValueError(
                "unexpected feature keys in serving snapshot: "
                f"sample_id={sample_id} keys={unexpected_feature_keys[:5]}"
            )
        if missing_feature_keys:
            raise ValueError(
                "missing feature keys in serving snapshot: "
                f"sample_id={sample_id} keys={missing_feature_keys[:5]}"
            )

        normalized_features = [
            normalize_feature_value(
                raw_features[key],
                sample_id=sample_id,
                feature_key=key,
            )
            for key in FEATURE_KEYS
        ]
        computed_missing_count = sum(value is None for value in normalized_features)
        if computed_missing_count != stored_missing_count:
            raise ValueError(
                "serving snapshot missing_count mismatch: "
                f"sample_id={sample_id} "
                f"stored={stored_missing_count} computed={computed_missing_count}"
            )

        label_event_id = row[12]
        if label_event_id is None:
            continue

        snapshot_ids.append(serving_snapshot_id)
        sample_ids.append(sample_id)
        snapshot_versions.append(snapshot_version)
        snapshot_times.append(snapshot_time)
        decision_times.append(decision_time)
        window_starts.append(window_start)
        window_ends.append(window_end)
        serving_missing_counts.append(stored_missing_count)
        feature_rows.append(normalized_features)
        label_event_ids.append(str(label_event_id))
        label_revisions.append(int(row[13]))
        label_measured_times.append(float(row[14]))
        label_available_times.append(float(row[15]))
        labels.append(int(row[16]))

    frame = pd.DataFrame(feature_rows, columns=list(MODEL_COLUMNS), dtype="float64")
    target = pd.Series(labels, dtype="int64")
    eligible_cohort_count = len(rows)
    labeled_cohort_count = len(sample_ids)
    unlabeled_cohort_count = eligible_cohort_count - labeled_cohort_count

    metadata = {
        "train_source": TRAIN_SOURCE,
        "training_spine": TRAINING_SPINE,
        "training_decision_time": TRAINING_DECISION_TIME,
        "snapshot_selection": SNAPSHOT_SELECTION,
        "label_selection": LABEL_SELECTION,
        "cohort_start_time": args.cohort_start_time,
        "cohort_end_time": cohort_end_time,
        "cutoff_time": args.cutoff_time,
        "label_maturity_seconds": args.label_maturity_seconds,
        "development_sample_limit": MAX_DEVELOPMENT_SAMPLES,
        "development_sample_selection": DEVELOPMENT_SAMPLE_SELECTION,
        "eligible_cohort_count": eligible_cohort_count,
        "labeled_cohort_count": labeled_cohort_count,
        "unlabeled_cohort_count": unlabeled_cohort_count,
        "label_coverage": (
            labeled_cohort_count / eligible_cohort_count
            if eligible_cohort_count > 0
            else None
        ),
        "sample_count": len(sample_ids),
        "fail_count": int((target == POSITIVE_CLASS).sum()),
        "pass_count": int((target == NEGATIVE_CLASS).sum()),
        "eligible_decision_time_min": min(eligible_decision_times),
        "eligible_decision_time_max": max(eligible_decision_times),
        "decision_time_min": min(decision_times) if decision_times else None,
        "decision_time_max": max(decision_times) if decision_times else None,
        "snapshot_time_min": min(snapshot_times) if snapshot_times else None,
        "snapshot_time_max": max(snapshot_times) if snapshot_times else None,
        "snapshot_version_min": min(snapshot_versions) if snapshot_versions else None,
        "snapshot_version_max": max(snapshot_versions) if snapshot_versions else None,
        "window_start_min": min(window_starts) if window_starts else None,
        "window_end_max": max(window_ends) if window_ends else None,
        "serving_missing_count_avg": float(np.mean(serving_missing_counts)) if serving_missing_counts else None,
        "serving_missing_count_max": int(max(serving_missing_counts)) if serving_missing_counts else None,
        "label_revision_min": min(label_revisions) if label_revisions else None,
        "label_revision_max": max(label_revisions) if label_revisions else None,
        "label_measured_at_min": min(label_measured_times) if label_measured_times else None,
        "label_measured_at_max": max(label_measured_times) if label_measured_times else None,
        "label_available_at_min": min(label_available_times) if label_available_times else None,
        "label_available_at_max": max(label_available_times) if label_available_times else None,
        "first_eligible_sample_id": eligible_sample_ids[0],
        "last_eligible_sample_id": eligible_sample_ids[-1],
        "first_sample_id": sample_ids[0] if sample_ids else None,
        "last_sample_id": sample_ids[-1] if sample_ids else None,
        "first_snapshot_id": snapshot_ids[0] if snapshot_ids else None,
        "last_snapshot_id": snapshot_ids[-1] if snapshot_ids else None,
        "first_label_event_id": label_event_ids[0] if label_event_ids else None,
        "last_label_event_id": label_event_ids[-1] if label_event_ids else None,
    }

    return frame, target, sample_ids, metadata


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


def split_indices(
        y: pd.Series,
        validation_size: float,
        random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=y,
    )
    return train_indices, validation_indices


def build_model(n_estimators: int, min_samples_leaf: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_predictions(
        y_true: pd.Series,
        fail_probability: np.ndarray,
        threshold: float,
) -> tuple[dict[str, float], dict[str, Any], np.ndarray]:
    y_pred = np.where(fail_probability >= threshold, POSITIVE_CLASS, NEGATIVE_CLASS)

    report = classification_report(
        y_true,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
    )
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_1": float(report["1"]["precision"]),
        "recall_1": float(report["1"]["recall"]),
        "f1_1": float(report["1"]["f1-score"]),
        "pr_auc": float(
            average_precision_score(
                y_true,
                fail_probability,
                pos_label=POSITIVE_CLASS,
            )
        ),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }

    return metrics, report, matrix


def select_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame,
        y_validation: pd.Series,
        n_estimators_values: list[int],
        min_samples_leaf_values: list[int],
        threshold_values: list[float],
        random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], np.ndarray]:
    rows: list[dict[str, Any]] = []
    reports: dict[tuple[int, int, float], dict[str, Any]] = {}
    matrices: dict[tuple[int, int, float], np.ndarray] = {}

    for n_estimators in n_estimators_values:
        for min_samples_leaf in min_samples_leaf_values:
            model = build_model(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            model.fit(x_train, y_train)

            class_order = list(model.named_steps["model"].classes_)
            positive_index = class_order.index(POSITIVE_CLASS)
            fail_probability = model.predict_proba(x_validation)[:, positive_index]

            for threshold in threshold_values:
                metrics, report, matrix = evaluate_predictions(
                    y_true=y_validation,
                    fail_probability=fail_probability,
                    threshold=threshold,
                )
                key = (n_estimators, min_samples_leaf, threshold)
                reports[key] = report
                matrices[key] = matrix
                rows.append({
                    "n_estimators": n_estimators,
                    "min_samples_leaf": min_samples_leaf,
                    "threshold": threshold,
                    **metrics,
                })

    results = pd.DataFrame(rows)
    best_row = results.sort_values(
        by=["f1_1", "recall_1", "precision_1", "pr_auc", "balanced_accuracy"],
        ascending=False,
    ).iloc[0].to_dict()

    best_key = (
        int(best_row["n_estimators"]),
        int(best_row["min_samples_leaf"]),
        float(best_row["threshold"]),
    )
    return results, best_row, reports[best_key], matrices[best_key]


def log_artifacts(
        result_frame: pd.DataFrame,
        report: dict[str, Any],
        matrix: np.ndarray,
        training_summary: dict[str, Any],
        development_sample_ids: list[str],
        train_sample_ids: list[str],
        validation_sample_ids: list[str],
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        result_path = tmp_path / "offline_feature_training_threshold_results.csv"
        report_path = tmp_path / "validation_classification_report.json"
        matrix_path = tmp_path / "validation_confusion_matrix.json"
        summary_path = tmp_path / "offline_feature_training_summary.json"
        development_samples_path = tmp_path / "development_sample_ids.txt"
        train_samples_path = tmp_path / "train_sample_ids.txt"
        validation_samples_path = tmp_path / "validation_sample_ids.txt"

        result_frame.to_csv(result_path, index=False)
        report_path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
        matrix_path.write_text(
            json.dumps(
                {"labels": [NEGATIVE_CLASS, POSITIVE_CLASS], "matrix": matrix.tolist()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        summary_path.write_text(json.dumps(json_safe(training_summary), indent=2, ensure_ascii=False), encoding="utf-8")
        development_samples_path.write_text("\n".join(development_sample_ids), encoding="utf-8")
        train_samples_path.write_text("\n".join(train_sample_ids), encoding="utf-8")
        validation_samples_path.write_text("\n".join(validation_sample_ids), encoding="utf-8")

        mlflow.log_artifact(str(result_path), artifact_path="results")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(matrix_path), artifact_path="reports")
        mlflow.log_artifact(str(summary_path), artifact_path="reports")
        mlflow.log_artifact(str(development_samples_path), artifact_path="data")
        mlflow.log_artifact(str(train_samples_path), artifact_path="data")
        mlflow.log_artifact(str(validation_samples_path), artifact_path="data")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def find_registered_model_version(
        client: MlflowClient,
        model_name: str,
        run_id: str,
):
    versions = [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if version.run_id == run_id
    ]

    if not versions:
        raise RuntimeError(f"registered model version not found: model_name={model_name} run_id={run_id}")

    return max(versions, key=lambda version: int(version.version))


def train_and_register(args: argparse.Namespace) -> None:
    x, y, sample_ids, metadata = load_labeled_point_in_time_features(args)
    validate_training_data(
        metadata=metadata,
        min_samples=args.min_samples,
        min_label_coverage=args.min_label_coverage,
        min_fail_samples=args.min_fail_samples,
        min_pass_samples=args.min_pass_samples,
    )

    train_indices, validation_indices = split_indices(
        y=y,
        validation_size=VALIDATION_SIZE,
        random_state=args.random_state,
    )
    x_train = x.iloc[train_indices].copy()
    y_train = y.iloc[train_indices].copy()
    x_validation = x.iloc[validation_indices].copy()
    y_validation = y.iloc[validation_indices].copy()

    train_sample_ids = [sample_ids[index] for index in train_indices]
    validation_sample_ids = [sample_ids[index] for index in validation_indices]

    n_estimators_values = positive_int_list(args.n_estimators, "--n-estimators")
    min_samples_leaf_values = positive_int_list(args.min_samples_leaf, "--min-samples-leaf")
    threshold_values = probability_list(args.thresholds, "--thresholds")

    result_frame, best_row, best_report, best_matrix = select_model(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        n_estimators_values=n_estimators_values,
        min_samples_leaf_values=min_samples_leaf_values,
        threshold_values=threshold_values,
        random_state=args.random_state,
    )

    if args.dry_run:
        print(
            "serving_snapshot_candidate_training_dry_run "
            f"sample_count={metadata['sample_count']} "
            f"fail_count={metadata['fail_count']} "
            f"pass_count={metadata['pass_count']} "
            f"eligible_cohort_count={metadata['eligible_cohort_count']} "
            f"unlabeled_cohort_count={metadata['unlabeled_cohort_count']} "
            f"label_coverage={metadata['label_coverage']} "
            f"min_label_coverage={args.min_label_coverage} "
            f"development_sample_limit={MAX_DEVELOPMENT_SAMPLES} "
            f"development_sample_selection={DEVELOPMENT_SAMPLE_SELECTION} "
            f"first_sample_id={metadata['first_sample_id']} "
            f"last_sample_id={metadata['last_sample_id']} "
            f"decision_time_min={metadata['decision_time_min']} "
            f"decision_time_max={metadata['decision_time_max']} "
            f"training_sample_count={len(train_indices)} "
            f"validation_sample_count={len(validation_indices)} "
            f"best_f1_1={best_row['f1_1']} "
            f"best_recall_1={best_row['recall_1']} "
            f"best_precision_1={best_row['precision_1']} "
            f"best_threshold={best_row['threshold']}"
        )
        return

    # Hyperparameters and the threshold are selected on validation data. The
    # registered candidate is then fitted on the complete development cohort;
    # its untouched final evaluation is the serving prediction decision gate.
    fit_x = x
    fit_y = y

    selected_model = build_model(
        n_estimators=int(best_row["n_estimators"]),
        min_samples_leaf=int(best_row["min_samples_leaf"]),
        random_state=args.random_state,
    )
    selected_model.fit(fit_x, fit_y)

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("secom-fail-detection")

    run_name = (
        "serving_snapshot_candidate"
        f"_{args.cohort_start_time:.0f}_{args.cutoff_time:.0f}"
        f"_{args.training_job_id}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        selected_params = {
            "model_name": "RandomForestClassifier",
            "registered_model_name": args.model_name,
            "registered_model_alias": args.model_alias,
            "n_estimators": int(best_row["n_estimators"]),
            "min_samples_leaf": int(best_row["min_samples_leaf"]),
            "class_weight": "balanced",
            "random_state": args.random_state,
            "development_sample_limit": MAX_DEVELOPMENT_SAMPLES,
            "development_sample_selection": DEVELOPMENT_SAMPLE_SELECTION,
            "min_label_coverage": args.min_label_coverage,
            "validation_size": VALIDATION_SIZE,
            "stratify": True,
            "imputer_strategy": "median",
            "threshold": float(best_row["threshold"]),
            "positive_class": POSITIVE_CLASS,
            "train_source": TRAIN_SOURCE,
            "training_spine": TRAINING_SPINE,
            "training_decision_time": TRAINING_DECISION_TIME,
            "snapshot_selection": SNAPSHOT_SELECTION,
            "label_selection": LABEL_SELECTION,
            "gate_source": GATE_SOURCE,
            "cohort_start_time": args.cohort_start_time,
            "cohort_end_time": metadata["cohort_end_time"],
            "cutoff_time": args.cutoff_time,
            "label_maturity_seconds": args.label_maturity_seconds,
            "simulation_run_id": args.simulation_run_id,
            "drift_segment": args.drift_segment,
            "final_fit_scope": "complete_development_cohort",
            "final_evaluation_source": "serving_prediction_decision_gate",
        }
        selected_metrics = {
            # Keep the unprefixed validation metrics for compatibility with the
            # legacy offline comparator. The explicit validation_* metrics are
            # the authoritative names for this training contract.
            "accuracy": float(best_row["accuracy"]),
            "balanced_accuracy": float(best_row["balanced_accuracy"]),
            "precision_1": float(best_row["precision_1"]),
            "recall_1": float(best_row["recall_1"]),
            "f1_1": float(best_row["f1_1"]),
            "pr_auc": float(best_row["pr_auc"]),
            "tn": float(best_row["tn"]),
            "fp": float(best_row["fp"]),
            "fn": float(best_row["fn"]),
            "tp": float(best_row["tp"]),
            "validation_accuracy": float(best_row["accuracy"]),
            "validation_balanced_accuracy": float(best_row["balanced_accuracy"]),
            "validation_precision_1": float(best_row["precision_1"]),
            "validation_recall_1": float(best_row["recall_1"]),
            "validation_f1_1": float(best_row["f1_1"]),
            "validation_pr_auc": float(best_row["pr_auc"]),
            "label_coverage": float(metadata["label_coverage"]),
            "training_sample_count": float(len(train_indices)),
            "validation_sample_count": float(len(validation_indices)),
            "final_fit_sample_count": float(len(y)),
            "training_fail_count": float((y_train == POSITIVE_CLASS).sum()),
            "validation_fail_count": float((y_validation == POSITIVE_CLASS).sum()),
            "final_fit_fail_count": float((y == POSITIVE_CLASS).sum()),
        }

        mlflow.set_tag("project", "secom-fail-detection")
        mlflow.set_tag("stage", "serving-snapshot-candidate")
        mlflow.set_tag("purpose", "candidate_training_from_first_complete_serving_snapshots")
        mlflow.set_tag("role", args.model_role)
        mlflow.set_tag("candidate_group", args.candidate_group)
        mlflow.set_tag("training_job_id", args.training_job_id)
        mlflow.set_tag("train_source", TRAIN_SOURCE)
        mlflow.set_tag("training_spine", TRAINING_SPINE)
        mlflow.set_tag("training_decision_time", TRAINING_DECISION_TIME)
        mlflow.set_tag("snapshot_selection", SNAPSHOT_SELECTION)
        mlflow.set_tag("label_selection", LABEL_SELECTION)
        mlflow.set_tag("gate_source", GATE_SOURCE)
        mlflow.set_tag("cohort_start_time", str(args.cohort_start_time))
        mlflow.set_tag("cohort_end_time", str(metadata["cohort_end_time"]))
        mlflow.set_tag("cutoff_time", str(args.cutoff_time))
        mlflow.set_tag("label_maturity_seconds", str(args.label_maturity_seconds))
        mlflow.set_tag("development_sample_selection", DEVELOPMENT_SAMPLE_SELECTION)
        mlflow.set_tag("min_label_coverage", str(args.min_label_coverage))
        mlflow.set_tag("offline_metric_scope", "validation_selection")
        mlflow.set_tag("final_evaluation_source", "serving_prediction_decision_gate")

        mlflow.log_params(selected_params)
        mlflow.log_metrics(selected_metrics)

        training_summary = {
            "tracking_uri": tracking_uri,
            "model_name": args.model_name,
            "model_alias": args.model_alias,
            "model_role": args.model_role,
            "candidate_group": args.candidate_group,
            "training_job_id": args.training_job_id,
            "metadata": metadata,
            "train_sample_count": len(train_indices),
            "validation_sample_count": len(validation_indices),
            "final_fit_sample_count": len(y),
            "best_row": best_row,
        }
        log_artifacts(
            result_frame=result_frame,
            report=best_report,
            matrix=best_matrix,
            training_summary=training_summary,
            development_sample_ids=sample_ids,
            train_sample_ids=train_sample_ids,
            validation_sample_ids=validation_sample_ids,
        )

        pyfunc_model = SECOMFailDetectionPyfunc(
            model=selected_model,
            threshold=float(best_row["threshold"]),
            model_name=args.model_name,
            model_run_id=run.info.run_id,
            positive_class=POSITIVE_CLASS,
        )

        input_example = fit_x.head(5).copy()
        output_example = pyfunc_model.predict(None, input_example)
        signature = infer_signature(input_example, output_example)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc_model,
            signature=signature,
            input_example=input_example,
            registered_model_name=args.model_name,
            await_registration_for=300,
            pip_requirements=[
                "mlflow==3.14.0",
                "numpy==1.26.0",
                "pandas>=2.2,<3",
                "scikit-learn==1.9.0",
            ],
        )

        client = MlflowClient()
        model_version = find_registered_model_version(
            client=client,
            model_name=args.model_name,
            run_id=run.info.run_id,
        )

        version_tags = {
            "registered_model_alias": args.model_alias,
            "source_run_id": run.info.run_id,
            "role": args.model_role,
            "candidate_group": args.candidate_group,
            "training_job_id": args.training_job_id,
            "train_source": TRAIN_SOURCE,
            "training_spine": TRAINING_SPINE,
            "training_decision_time": TRAINING_DECISION_TIME,
            "snapshot_selection": SNAPSHOT_SELECTION,
            "label_selection": LABEL_SELECTION,
            "gate_source": GATE_SOURCE,
            "development_sample_limit": MAX_DEVELOPMENT_SAMPLES,
            "development_sample_selection": DEVELOPMENT_SAMPLE_SELECTION,
            "min_label_coverage": args.min_label_coverage,
            "validation_size": VALIDATION_SIZE,
            "final_fit_scope": "complete_development_cohort",
            "final_evaluation_source": "serving_prediction_decision_gate",
            "cohort_start_time": args.cohort_start_time,
            "cohort_end_time": metadata["cohort_end_time"],
            "cutoff_time": args.cutoff_time,
            "label_maturity_seconds": args.label_maturity_seconds,
        }

        for key, value in version_tags.items():
            if value is not None:
                client.set_model_version_tag(args.model_name, model_version.version, key, str(value))

        client.set_registered_model_alias(
            args.model_name,
            args.model_alias,
            model_version.version,
        )

        print(
            "serving_snapshot_candidate_registered "
            f"tracking_uri={tracking_uri} "
            f"model_name={args.model_name} "
            f"alias={args.model_alias} "
            f"version={model_version.version} "
            f"run_id={run.info.run_id} "
            f"sample_count={metadata['sample_count']} "
            f"fail_count={metadata['fail_count']} "
            f"pass_count={metadata['pass_count']} "
            f"eligible_cohort_count={metadata['eligible_cohort_count']} "
            f"unlabeled_cohort_count={metadata['unlabeled_cohort_count']} "
            f"label_coverage={metadata['label_coverage']} "
            f"min_label_coverage={args.min_label_coverage} "
            f"cohort_start_time={args.cohort_start_time} "
            f"cohort_end_time={metadata['cohort_end_time']} "
            f"cutoff_time={args.cutoff_time} "
            f"label_maturity_seconds={args.label_maturity_seconds} "
            f"development_sample_limit={MAX_DEVELOPMENT_SAMPLES} "
            f"development_sample_selection={DEVELOPMENT_SAMPLE_SELECTION} "
            f"first_sample_id={metadata['first_sample_id']} "
            f"last_sample_id={metadata['last_sample_id']} "
            f"decision_time_min={metadata['decision_time_min']} "
            f"decision_time_max={metadata['decision_time_max']} "
            f"training_sample_count={len(train_indices)} "
            f"validation_sample_count={len(validation_indices)} "
            f"final_fit_sample_count={len(y)} "
            f"best_f1_1={best_row['f1_1']} "
            f"best_recall_1={best_row['recall_1']} "
            f"best_precision_1={best_row['precision_1']} "
            f"best_threshold={best_row['threshold']}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    train_and_register(args)


if __name__ == "__main__":
    main()
