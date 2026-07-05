"""Train a candidate model from point-in-time features reconstructed from feature_events."""

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
    non_negative_int,
    positive_float,
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
TRAIN_SOURCE = "offline_feature_store_point_in_time"
TRAINING_SPINE = "serving_feature_snapshots"
TRAINING_POINT_TIME = "serving_feature_snapshots.snapshot_time"
GATE_SOURCE = "serving_feature_snapshots"


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

    parser.add_argument("--point-time-start", type=non_negative_float, required=True)
    parser.add_argument("--point-time", type=non_negative_float, required=True)
    parser.add_argument("--simulation-run-id", default=None)
    parser.add_argument("--drift-segment", default=None)
    parser.add_argument("--limit", type=non_negative_int, default=0)

    parser.add_argument("--min-samples", type=positive_int, default=500)
    parser.add_argument("--min-fail-samples", type=positive_int, default=20)
    parser.add_argument("--min-pass-samples", type=positive_int, default=20)
    parser.add_argument("--test-size", type=positive_float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--n-estimators", default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--min-samples-leaf", default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--refit-on-all-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.point_time <= args.point_time_start:
        raise ValueError("point_time must be > point_time_start")

    if args.test_size >= 1.0:
        raise ValueError("test_size must be < 1")

    if args.model_role not in MODEL_ROLES:
        raise ValueError("model_role must be one of: candidate, champion")

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
    filters = [
        "s.is_complete = TRUE",
        "s.snapshot_status = 'complete'",
        "s.snapshot_time >= %s",
        "s.snapshot_time <= %s",
    ]
    params: list[Any] = [args.point_time_start, args.point_time]

    if args.simulation_run_id is not None:
        filters.append("s.simulation_run_id = %s")
        params.append(args.simulation_run_id)

    if args.drift_segment is not None:
        filters.append("s.drift_segment = %s")
        params.append(args.drift_segment)

    limit_sql = ""
    spine_params = [*params]
    if args.limit > 0:
        limit_sql = "LIMIT %s"
        spine_params.append(args.limit)

    event_join_filters = [
        "e.sample_id = spine.sample_id",
        "e.event_time <= spine.snapshot_time",
        "(spine.simulation_run_id IS NULL OR e.simulation_run_id = spine.simulation_run_id)",
        "(spine.drift_segment IS NULL OR e.drift_segment = spine.drift_segment)",
    ]

    sql = f"""
    WITH ranked_spine AS (
      SELECT
        s.serving_snapshot_id,
        s.sample_id,
        s.snapshot_time,
        s.window_start,
        s.window_end,
        s.feature_count,
        s.missing_count,
        s.features_json,
        s.simulation_run_id,
        s.drift_segment,
        s.created_at,
        ROW_NUMBER() OVER (
          PARTITION BY s.sample_id
          ORDER BY
            s.snapshot_time DESC,
            s.created_at DESC,
            s.serving_snapshot_id DESC
        ) AS rn
      FROM serving_feature_snapshots s
      WHERE {' AND '.join(filters)}
    ),
    spine AS (
      SELECT
        s.serving_snapshot_id,
        s.sample_id,
        s.snapshot_time,
        s.window_start,
        s.window_end,
        s.feature_count AS serving_feature_count,
        s.missing_count AS serving_missing_count,
        s.simulation_run_id,
        s.drift_segment,
        a.actual_value,
        a.actual_label,
        a.labeled_at
      FROM ranked_spine s
      JOIN actual_labels a
        ON a.sample_id = s.sample_id
      WHERE s.rn = 1
        AND s.snapshot_time <= a.labeled_at
      ORDER BY s.snapshot_time, s.sample_id
      {limit_sql}
    )
    SELECT
      spine.serving_snapshot_id,
      spine.sample_id,
      spine.snapshot_time,
      spine.window_start,
      spine.window_end,
      spine.serving_feature_count,
      spine.serving_missing_count,
      spine.simulation_run_id,
      spine.drift_segment,
      spine.actual_value,
      spine.actual_label,
      spine.labeled_at,
      e.event_id,
      e.event_time,
      e.feature_group,
      e.features_json,
      e.created_at AS event_created_at
    FROM spine
    LEFT JOIN feature_events e
      ON {' AND '.join(event_join_filters)}
    ORDER BY
      spine.snapshot_time,
      spine.sample_id,
      e.event_time,
      e.created_at,
      e.event_id;
    """

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, spine_params)
            rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(
            "no labeled serving snapshot spine rows found for point-in-time training: "
            f"point_time_start={args.point_time_start} "
            f"point_time={args.point_time}"
        )

    spines: dict[str, dict[str, Any]] = {}
    spine_order: list[str] = []

    for row in rows:
        serving_snapshot_id = str(row[0])
        sample_id = str(row[1])

        if serving_snapshot_id not in spines:
            spine_order.append(serving_snapshot_id)
            spines[serving_snapshot_id] = {
                "serving_snapshot_id": serving_snapshot_id,
                "sample_id": sample_id,
                "snapshot_time": float(row[2]),
                "window_start": float(row[3]),
                "window_end": float(row[4]),
                "serving_feature_count": int(row[5]),
                "serving_missing_count": int(row[6]),
                "simulation_run_id": row[7],
                "drift_segment": row[8],
                "actual_value": int(row[9]),
                "actual_label": str(row[10]),
                "labeled_at": float(row[11]),
                "feature_state": {},
                "source_event_count": 0,
                "max_event_time": None,
            }

        event_id = row[12]
        if event_id is None:
            continue

        event_time = float(row[13])
        features = parse_feature_object(row[15], sample_id=sample_id)
        state = spines[serving_snapshot_id]["feature_state"]
        spines[serving_snapshot_id]["source_event_count"] += 1
        spines[serving_snapshot_id]["max_event_time"] = event_time

        for key, value in features.items():
            if key not in FEATURE_KEY_SET:
                raise ValueError(f"unexpected feature key: sample_id={sample_id} key={key}")
            state[key] = normalize_feature_value(value, sample_id=sample_id, feature_key=key)

    sample_ids: list[str] = []
    snapshot_ids: list[str] = []
    feature_rows: list[list[float | None]] = []
    labels: list[int] = []
    snapshot_times: list[float] = []
    point_times: list[float] = []
    window_starts: list[float] = []
    window_ends: list[float] = []
    serving_missing_counts: list[int] = []
    missing_counts: list[int] = []
    labeled_times: list[float] = []
    source_event_counts: list[int] = []
    max_event_times: list[float] = []
    incomplete_snapshot_ids: list[str] = []

    for serving_snapshot_id in spine_order:
        spine = spines[serving_snapshot_id]
        sample_id = spine["sample_id"]
        feature_state = spine["feature_state"]
        feature_count = len(feature_state)
        features = {
            key: feature_state.get(key)
            for key in FEATURE_KEYS
        }
        missing_count = sum(value is None for value in features.values())

        if feature_count != NUM_FEATURES:
            incomplete_snapshot_ids.append(serving_snapshot_id)
            continue

        snapshot_ids.append(serving_snapshot_id)
        sample_ids.append(sample_id)
        snapshot_times.append(spine["snapshot_time"])
        point_times.append(spine["snapshot_time"])
        window_starts.append(spine["window_start"])
        window_ends.append(spine["window_end"])
        serving_missing_counts.append(spine["serving_missing_count"])
        missing_counts.append(missing_count)
        source_event_counts.append(spine["source_event_count"])
        if spine["max_event_time"] is not None:
            max_event_times.append(float(spine["max_event_time"]))
        feature_rows.append([features[key] for key in FEATURE_KEYS])
        labels.append(spine["actual_value"])
        labeled_times.append(spine["labeled_at"])

    frame = pd.DataFrame(feature_rows, columns=list(MODEL_COLUMNS), dtype="float64")
    target = pd.Series(labels, dtype="int64")

    metadata = {
        "train_source": TRAIN_SOURCE,
        "training_spine": TRAINING_SPINE,
        "training_point_time": TRAINING_POINT_TIME,
        "spine_row_count": len(spine_order),
        "offline_complete_count": len(sample_ids),
        "offline_incomplete_count": len(incomplete_snapshot_ids),
        "sample_count": len(sample_ids),
        "fail_count": int((target == POSITIVE_CLASS).sum()),
        "pass_count": int((target == NEGATIVE_CLASS).sum()),
        "snapshot_time_min": min(snapshot_times) if snapshot_times else None,
        "snapshot_time_max": max(snapshot_times) if snapshot_times else None,
        "point_time_min": min(point_times) if point_times else None,
        "point_time_max": max(point_times) if point_times else None,
        "window_start_min": min(window_starts) if window_starts else None,
        "window_end_max": max(window_ends) if window_ends else None,
        "offline_missing_count_avg": float(np.mean(missing_counts)) if missing_counts else None,
        "offline_missing_count_max": int(max(missing_counts)) if missing_counts else None,
        "serving_missing_count_avg": float(np.mean(serving_missing_counts)) if serving_missing_counts else None,
        "serving_missing_count_max": int(max(serving_missing_counts)) if serving_missing_counts else None,
        "source_event_count_avg": float(np.mean(source_event_counts)) if source_event_counts else None,
        "source_event_count_max": int(max(source_event_counts)) if source_event_counts else None,
        "max_event_time_min": min(max_event_times) if max_event_times else None,
        "max_event_time_max": max(max_event_times) if max_event_times else None,
        "labeled_at_min": min(labeled_times) if labeled_times else None,
        "labeled_at_max": max(labeled_times) if labeled_times else None,
        "first_sample_id": sample_ids[0] if sample_ids else None,
        "last_sample_id": sample_ids[-1] if sample_ids else None,
        "first_snapshot_id": snapshot_ids[0] if snapshot_ids else None,
        "last_snapshot_id": snapshot_ids[-1] if snapshot_ids else None,
        "first_incomplete_snapshot_id": incomplete_snapshot_ids[0] if incomplete_snapshot_ids else None,
        "last_incomplete_snapshot_id": incomplete_snapshot_ids[-1] if incomplete_snapshot_ids else None,
    }

    return frame, target, sample_ids, metadata


def validate_training_data(
        metadata: dict[str, Any],
        min_samples: int,
        min_fail_samples: int,
        min_pass_samples: int,
) -> None:
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
        test_size: float,
        random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=test_size,
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
        train_sample_ids: list[str],
        validation_sample_ids: list[str],
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        result_path = tmp_path / "offline_feature_training_threshold_results.csv"
        report_path = tmp_path / "classification_report.json"
        matrix_path = tmp_path / "confusion_matrix.json"
        summary_path = tmp_path / "offline_feature_training_summary.json"
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
        train_samples_path.write_text("\n".join(train_sample_ids), encoding="utf-8")
        validation_samples_path.write_text("\n".join(validation_sample_ids), encoding="utf-8")

        mlflow.log_artifact(str(result_path), artifact_path="results")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(matrix_path), artifact_path="reports")
        mlflow.log_artifact(str(summary_path), artifact_path="reports")
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
        min_fail_samples=args.min_fail_samples,
        min_pass_samples=args.min_pass_samples,
    )

    train_indices, validation_indices = split_indices(
        y=y,
        test_size=args.test_size,
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
            "offline_feature_candidate_training_dry_run "
            f"sample_count={metadata['sample_count']} "
            f"fail_count={metadata['fail_count']} "
            f"pass_count={metadata['pass_count']} "
            f"offline_incomplete_count={metadata['offline_incomplete_count']} "
            f"best_f1_1={best_row['f1_1']} "
            f"best_recall_1={best_row['recall_1']} "
            f"best_precision_1={best_row['precision_1']} "
            f"best_threshold={best_row['threshold']}"
        )
        return

    fit_x = x if args.refit_on_all_data else x_train
    fit_y = y if args.refit_on_all_data else y_train

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
        "offline_feature_candidate"
        f"_{args.point_time_start:.0f}_{args.point_time:.0f}"
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
            "test_size": args.test_size,
            "stratify": True,
            "imputer_strategy": "median",
            "threshold": float(best_row["threshold"]),
            "positive_class": POSITIVE_CLASS,
            "train_source": TRAIN_SOURCE,
            "training_spine": TRAINING_SPINE,
            "training_point_time": TRAINING_POINT_TIME,
            "gate_source": GATE_SOURCE,
            "point_time_start": args.point_time_start,
            "point_time": args.point_time,
            "simulation_run_id": args.simulation_run_id,
            "drift_segment": args.drift_segment,
            "refit_on_all_data": args.refit_on_all_data,
        }
        selected_metrics = {
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
            "training_sample_count": float(len(train_indices)),
            "validation_sample_count": float(len(validation_indices)),
            "training_fail_count": float((y_train == POSITIVE_CLASS).sum()),
            "validation_fail_count": float((y_validation == POSITIVE_CLASS).sum()),
        }

        mlflow.set_tag("project", "secom-fail-detection")
        mlflow.set_tag("stage", "offline-feature-candidate")
        mlflow.set_tag("purpose", "candidate_training_from_offline_feature_store_point_in_time")
        mlflow.set_tag("role", args.model_role)
        mlflow.set_tag("candidate_group", args.candidate_group)
        mlflow.set_tag("training_job_id", args.training_job_id)
        mlflow.set_tag("train_source", TRAIN_SOURCE)
        mlflow.set_tag("training_spine", TRAINING_SPINE)
        mlflow.set_tag("training_point_time", TRAINING_POINT_TIME)
        mlflow.set_tag("gate_source", GATE_SOURCE)
        mlflow.set_tag("point_time_start", str(args.point_time_start))
        mlflow.set_tag("point_time", str(args.point_time))

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
            "best_row": best_row,
        }
        log_artifacts(
            result_frame=result_frame,
            report=best_report,
            matrix=best_matrix,
            training_summary=training_summary,
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
            "training_point_time": TRAINING_POINT_TIME,
            "gate_source": GATE_SOURCE,
            "point_time_start": args.point_time_start,
            "point_time": args.point_time,
            "gate_status": "pending" if args.model_role == "candidate" else "not_required",
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
            "offline_feature_candidate_registered "
            f"tracking_uri={tracking_uri} "
            f"model_name={args.model_name} "
            f"alias={args.model_alias} "
            f"version={model_version.version} "
            f"run_id={run.info.run_id} "
            f"sample_count={metadata['sample_count']} "
            f"fail_count={metadata['fail_count']} "
            f"pass_count={metadata['pass_count']} "
            f"offline_incomplete_count={metadata['offline_incomplete_count']} "
            f"point_time_start={args.point_time_start} "
            f"point_time={args.point_time} "
            f"best_f1_1={best_row['f1_1']} "
            f"best_recall_1={best_row['recall_1']} "
            f"best_precision_1={best_row['precision_1']} "
            f"best_threshold={best_row['threshold']} "
            f"gate_status={version_tags['gate_status']}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    train_and_register(args)


if __name__ == "__main__":
    main()
