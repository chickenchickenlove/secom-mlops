import argparse
import os
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from secom_mlops.monitor.db import connect
from secom_mlops.monitor.deployments import (
    build_deployment_request_row,
    insert_deployment_request,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_CHAMPION_ALIAS,
    resolve_model_name,
    resolve_tracking_uri,
)
from secom_mlops_common.schemas.secom import (
    FEATURE_KEY_SET,
    FEATURE_KEYS,
    MODEL_COLUMNS,
    NUM_FEATURES,
    normalize_feature_value,
    parse_feature_object,
)

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1
DEFAULT_MAX_DECISIONS = 1000
DECISION_SELECTION = "latest_champion_decisions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--candidate-alias", default=DEFAULT_CANDIDATE_ALIAS)
    parser.add_argument("--champion-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--candidate-version", default=None)
    parser.add_argument("--champion-version", default=None)

    parser.add_argument("--cohort-start-time", type=float, required=True)
    parser.add_argument("--cutoff-time", type=float, required=True)
    parser.add_argument("--label-maturity-seconds", type=float, required=True)
    parser.add_argument(
        "--max-decisions",
        "--limit",
        dest="max_decisions",
        type=int,
        default=DEFAULT_MAX_DECISIONS,
    )

    parser.add_argument("--min-decisions", type=int, default=500)
    parser.add_argument("--min-label-coverage", type=float, default=0.95)
    parser.add_argument("--min-fail-samples", type=int, default=20)
    parser.add_argument("--min-pass-samples", type=int, default=20)

    parser.add_argument("--primary-metric", default="fail_f1")
    parser.add_argument("--min-primary-delta", type=float, default=0.0)
    parser.add_argument("--min-recall-delta", type=float, default=-0.02)
    parser.add_argument("--min-precision-delta", type=float, default=-0.05)
    parser.add_argument("--set-tags", action="store_true")
    parser.add_argument("--fail-on-gate-failure", action="store_true")
    parser.add_argument("--record-deployment-request", action="store_true")
    parser.add_argument("--deployment-approval-status", default="approved")
    parser.add_argument("--deployment-notes", default=None)
    parser.add_argument("--deployment-requested-by", default=os.getenv("USER"))
    parser.add_argument("--deployment-approved-by", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    numeric_times = {
        "cohort_start_time": args.cohort_start_time,
        "cutoff_time": args.cutoff_time,
        "label_maturity_seconds": args.label_maturity_seconds,
    }
    for name, value in numeric_times.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")

    cohort_end_time = args.cutoff_time - args.label_maturity_seconds
    if cohort_end_time <= args.cohort_start_time:
        raise ValueError(
            "cutoff_time - label_maturity_seconds must be greater than "
            "cohort_start_time"
        )

    if args.max_decisions <= 0:
        raise ValueError("max_decisions must be > 0")
    if args.min_decisions <= 0:
        raise ValueError("min_decisions must be > 0")
    if args.min_decisions > args.max_decisions:
        raise ValueError("min_decisions must be <= max_decisions")
    if not 0.0 <= args.min_label_coverage <= 1.0:
        raise ValueError("min_label_coverage must be between 0 and 1")
    if args.min_fail_samples <= 0:
        raise ValueError("min_fail_samples must be > 0")
    if args.min_pass_samples <= 0:
        raise ValueError("min_pass_samples must be > 0")


def resolve_model(client: MlflowClient, model_name: str, alias: str, version: str | None) -> dict[str, Any]:
    if version:
        model_version = client.get_model_version(model_name, version)
        model_uri = f"models:/{model_name}/{model_version.version}"
        resolved_alias = None
    else:
        model_version = client.get_model_version_by_alias(model_name, alias)
        model_uri = f"models:/{model_name}@{alias}"
        resolved_alias = alias

    return {
        "model_uri": model_uri,
        "model": mlflow.pyfunc.load_model(model_uri),
        "model_version": str(model_version.version),
        "model_alias": resolved_alias,
        "model_run_id": str(model_version.run_id),
    }


def load_labeled_serving_decisions(
        args: argparse.Namespace,
        champion_model_run_id: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    cohort_end_time = args.cutoff_time - args.label_maturity_seconds
    params: dict[str, Any] = {
        "champion_model_run_id": champion_model_run_id,
        "cohort_start_time": args.cohort_start_time,
        "cohort_end_time": cohort_end_time,
        "cutoff_time": args.cutoff_time,
        "max_decisions": args.max_decisions,
    }

    sql = f"""
    WITH canonical_predictions AS (
      SELECT
        p.prediction_id,
        p.request_id,
        p.sample_id,
        p.serving_snapshot_id,
        p.snapshot_version,
        p.feature_hash,
        p.model_run_id,
        p.runtime_slot,
        p.threshold,
        p.predicted_at,
        EXISTS (
          SELECT 1
          FROM prediction_logs conflicting
          WHERE conflicting.model_run_id = p.model_run_id
            AND conflicting.threshold = p.threshold
            AND conflicting.sample_id = p.sample_id
            AND conflicting.serving_snapshot_id = p.serving_snapshot_id
            AND conflicting.snapshot_version = p.snapshot_version
            AND conflicting.feature_hash <> p.feature_hash
        ) AS has_conflicting_feature_hash
      FROM prediction_logs p
      WHERE p.model_run_id = %(champion_model_run_id)s
        AND NOT EXISTS (
          SELECT 1
          FROM prediction_logs earlier
          WHERE earlier.model_run_id = p.model_run_id
            AND earlier.threshold = p.threshold
            AND earlier.sample_id = p.sample_id
            AND earlier.serving_snapshot_id = p.serving_snapshot_id
            AND earlier.snapshot_version = p.snapshot_version
            AND (
              earlier.predicted_at < p.predicted_at
              OR (
                earlier.predicted_at = p.predicted_at
                AND earlier.prediction_id < p.prediction_id
              )
            )
        )
    ),
    latest_prediction_cohort AS (
      SELECT *
      FROM canonical_predictions
      WHERE predicted_at >= %(cohort_start_time)s
        AND predicted_at < %(cohort_end_time)s
      ORDER BY predicted_at DESC, prediction_id DESC
      LIMIT %(max_decisions)s
    ),
    prediction_cohort AS (
      SELECT *
      FROM latest_prediction_cohort
      ORDER BY predicted_at ASC, prediction_id ASC
    ),
    cohort_samples AS (
      SELECT DISTINCT sample_id
      FROM prediction_cohort
    ),
    ranked_labels AS (
      SELECT
        le.label_event_id,
        le.sample_id,
        le.label_revision,
        le.measured_at,
        le.available_at,
        le.actual_value,
        le.actual_label,
        ROW_NUMBER() OVER (
          PARTITION BY le.sample_id
          ORDER BY
            le.label_revision DESC,
            le.available_at DESC,
            le.label_event_id DESC
        ) AS label_rank
      FROM label_events le
      JOIN cohort_samples c
        ON c.sample_id = le.sample_id
      WHERE le.available_at <= %(cutoff_time)s
    ),
    labels_at_cutoff AS (
      SELECT
        label_event_id,
        sample_id,
        label_revision,
        measured_at,
        available_at,
        actual_value,
        actual_label
      FROM ranked_labels
      WHERE label_rank = 1
    )
    SELECT
      p.prediction_id,
      p.request_id,
      p.sample_id,
      p.serving_snapshot_id,
      p.snapshot_version,
      p.feature_hash AS prediction_feature_hash,
      p.model_run_id,
      p.runtime_slot,
      p.threshold AS source_threshold,
      p.predicted_at,
      p.has_conflicting_feature_hash,
      s.serving_snapshot_id AS stored_serving_snapshot_id,
      s.sample_id AS stored_sample_id,
      s.snapshot_version AS stored_snapshot_version,
      s.feature_hash AS snapshot_feature_hash,
      s.snapshot_status,
      s.is_complete,
      s.feature_count,
      s.missing_count,
      s.features_json,
      s.available_at AS snapshot_available_at,
      l.label_event_id,
      l.label_revision,
      l.measured_at AS label_measured_at,
      l.available_at AS label_available_at,
      l.actual_value,
      l.actual_label
    FROM prediction_cohort p
    LEFT JOIN serving_feature_snapshots s
      ON s.serving_snapshot_id = p.serving_snapshot_id
     AND s.sample_id = p.sample_id
     AND s.snapshot_version = p.snapshot_version
    LEFT JOIN labels_at_cutoff l
      ON l.sample_id = p.sample_id
    ORDER BY p.predicted_at, p.prediction_id;
    """

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            columns = [description.name for description in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    labeled_prediction_ids: list[str] = []
    feature_rows: list[list[float | None]] = []
    labels: list[int] = []
    decision_times: list[float] = []
    snapshot_available_times: list[float] = []
    missing_counts: list[int] = []
    label_revisions: list[int] = []
    label_available_times: list[float] = []

    for row in rows:
        prediction_id = str(row["prediction_id"])
        sample_id = str(row["sample_id"])
        serving_snapshot_id = str(row["serving_snapshot_id"])
        snapshot_version = int(row["snapshot_version"])
        decision_times.append(float(row["predicted_at"]))

        if row["has_conflicting_feature_hash"]:
            raise RuntimeError(
                "repeated prediction decision has conflicting feature_hash values: "
                f"prediction_id={prediction_id} "
                f"sample_id={sample_id} "
                f"serving_snapshot_id={serving_snapshot_id} "
                f"snapshot_version={snapshot_version}"
            )

        if row["stored_serving_snapshot_id"] is None:
            raise RuntimeError(
                "prediction decision has no exact serving snapshot: "
                f"prediction_id={prediction_id} "
                f"sample_id={sample_id} "
                f"serving_snapshot_id={serving_snapshot_id} "
                f"snapshot_version={snapshot_version}"
            )

        prediction_feature_hash = str(row["prediction_feature_hash"])
        snapshot_feature_hash = str(row["snapshot_feature_hash"])
        if prediction_feature_hash != snapshot_feature_hash:
            raise RuntimeError(
                "prediction and serving snapshot feature_hash mismatch: "
                f"prediction_id={prediction_id} "
                f"serving_snapshot_id={serving_snapshot_id} "
                f"prediction_feature_hash={prediction_feature_hash} "
                f"snapshot_feature_hash={snapshot_feature_hash}"
            )

        if row["snapshot_status"] != "complete" or row["is_complete"] is not True:
            raise RuntimeError(
                "prediction decision must reference a complete serving snapshot: "
                f"prediction_id={prediction_id} "
                f"serving_snapshot_id={serving_snapshot_id} "
                f"snapshot_status={row['snapshot_status']} "
                f"is_complete={row['is_complete']}"
            )

        feature_count = int(row["feature_count"])
        if feature_count != NUM_FEATURES:
            raise RuntimeError(
                "complete serving snapshot must contain all feature keys: "
                f"prediction_id={prediction_id} "
                f"serving_snapshot_id={serving_snapshot_id} "
                f"feature_count={feature_count}"
            )

        raw_features = parse_feature_object(row["features_json"], sample_id=sample_id)
        actual_feature_keys = set(raw_features)
        unexpected_feature_keys = sorted(actual_feature_keys - FEATURE_KEY_SET)
        missing_feature_keys = sorted(FEATURE_KEY_SET - actual_feature_keys)
        if unexpected_feature_keys:
            raise RuntimeError(
                "unexpected feature keys in serving snapshot: "
                f"prediction_id={prediction_id} "
                f"keys={unexpected_feature_keys[:5]}"
            )
        if missing_feature_keys:
            raise RuntimeError(
                "missing feature keys in serving snapshot: "
                f"prediction_id={prediction_id} "
                f"keys={missing_feature_keys[:5]}"
            )

        normalized_features = [
            normalize_feature_value(
                raw_features[key],
                sample_id=sample_id,
                feature_key=key,
            )
            for key in FEATURE_KEYS
        ]
        stored_missing_count = int(row["missing_count"])
        computed_missing_count = sum(value is None for value in normalized_features)
        if stored_missing_count != computed_missing_count:
            raise RuntimeError(
                "serving snapshot missing_count mismatch: "
                f"prediction_id={prediction_id} "
                f"serving_snapshot_id={serving_snapshot_id} "
                f"stored={stored_missing_count} "
                f"computed={computed_missing_count}"
            )

        snapshot_available_times.append(float(row["snapshot_available_at"]))
        missing_counts.append(stored_missing_count)

        if row["label_event_id"] is None:
            continue

        labeled_prediction_ids.append(prediction_id)
        feature_rows.append(normalized_features)
        labels.append(int(row["actual_value"]))
        label_revisions.append(int(row["label_revision"]))
        label_available_times.append(float(row["label_available_at"]))

    x = pd.DataFrame(feature_rows, columns=list(MODEL_COLUMNS), dtype="float64")
    y = pd.Series(labels, dtype="int64")

    decision_count = len(rows)
    labeled_decision_count = len(labeled_prediction_ids)
    metadata = {
        "cohort_start_time": args.cohort_start_time,
        "cohort_end_time": cohort_end_time,
        "cutoff_time": args.cutoff_time,
        "label_maturity_seconds": args.label_maturity_seconds,
        "decision_selection": DECISION_SELECTION,
        "max_decisions": args.max_decisions,
        "champion_source_model_run_id": champion_model_run_id,
        "decision_count": decision_count,
        "labeled_decision_count": labeled_decision_count,
        "unlabeled_decision_count": decision_count - labeled_decision_count,
        "label_coverage": (
            0.0 if decision_count == 0 else labeled_decision_count / decision_count
        ),
        "fail_count": int((y == POSITIVE_CLASS).sum()) if len(y) else 0,
        "pass_count": int((y == NEGATIVE_CLASS).sum()) if len(y) else 0,
        "decision_time_min": min(decision_times) if decision_times else None,
        "decision_time_max": max(decision_times) if decision_times else None,
        "snapshot_available_at_min": (
            min(snapshot_available_times) if snapshot_available_times else None
        ),
        "snapshot_available_at_max": (
            max(snapshot_available_times) if snapshot_available_times else None
        ),
        "missing_count_avg": float(np.mean(missing_counts)) if missing_counts else None,
        "label_revision_min": min(label_revisions) if label_revisions else None,
        "label_revision_max": max(label_revisions) if label_revisions else None,
        "label_available_at_min": (
            min(label_available_times) if label_available_times else None
        ),
        "label_available_at_max": (
            max(label_available_times) if label_available_times else None
        ),
        "runtime_slots": sorted({str(row["runtime_slot"]) for row in rows}),
        "source_thresholds": sorted({float(row["source_threshold"]) for row in rows}),
        "first_sample_id": str(rows[0]["sample_id"]) if rows else None,
        "last_sample_id": str(rows[-1]["sample_id"]) if rows else None,
    }
    return x, y, labeled_prediction_ids, metadata


def predict_model(model_bundle: dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    predictions = model_bundle["model"].predict(features.copy())
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)

    if "fail_probability" not in predictions.columns:
        raise ValueError("model output missing fail_probability")

    if "prediction" in predictions.columns:
        predicted_value = predictions["prediction"]
    elif "predicted_value" in predictions.columns:
        predicted_value = predictions["predicted_value"]
    else:
        raise ValueError("model output missing prediction")

    return pd.DataFrame({
        "fail_probability": predictions["fail_probability"].astype(float),
        "predicted_value": predicted_value.astype(int),
    })


def evaluate_predictions(y_true: pd.Series, prediction_frame: pd.DataFrame) -> dict[str, float | int | None]:
    y_pred = prediction_frame["predicted_value"].astype(int)
    fail_probability = prediction_frame["fail_probability"].astype(float)

    matrix = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_CLASS, POSITIVE_CLASS])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    n_fail_samples = int((y_true == POSITIVE_CLASS).sum())

    pr_auc = None
    if n_fail_samples > 0:
        pr_auc = float(average_precision_score(y_true, fail_probability, pos_label=POSITIVE_CLASS))

    return {
        "n_samples": int(len(y_true)),
        "n_fail_samples": n_fail_samples,
        "n_pass_samples": int((y_true == NEGATIVE_CLASS).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fail_precision": float(precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "fail_recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "fail_f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)),
        "pr_auc": pr_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def metric_names() -> list[str]:
    return [
        "fail_f1",
        "fail_recall",
        "fail_precision",
        "pr_auc",
        "balanced_accuracy",
        "accuracy",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]


def metric_delta(candidate_metrics: dict[str, Any], champion_metrics: dict[str, Any], metric_name: str) -> float | None:
    candidate_value = candidate_metrics.get(metric_name)
    champion_value = champion_metrics.get(metric_name)
    if candidate_value is None or champion_value is None:
        return None
    return float(candidate_value) - float(champion_value)


def evaluate_gate(
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
        primary_metric: str,
        min_primary_delta: float,
        min_recall_delta: float,
        min_precision_delta: float,
) -> tuple[str, list[str]]:
    reasons = []

    primary_delta = metric_delta(candidate_metrics, champion_metrics, primary_metric)
    if primary_delta is None:
        reasons.append(f"primary metric unavailable: {primary_metric}")
    elif primary_delta < min_primary_delta:
        reasons.append(
            f"{primary_metric} delta below gate: delta={primary_delta:.6f} required>={min_primary_delta:.6f}")

    recall_delta = metric_delta(candidate_metrics, champion_metrics, "fail_recall")
    if recall_delta is not None and recall_delta < min_recall_delta:
        reasons.append(f"fail_recall regression too large: delta={recall_delta:.6f} required>={min_recall_delta:.6f}")

    precision_delta = metric_delta(candidate_metrics, champion_metrics, "fail_precision")
    if precision_delta is not None and precision_delta < min_precision_delta:
        reasons.append(
            f"fail_precision regression too large: delta={precision_delta:.6f} required>={min_precision_delta:.6f}")

    return ("failed" if reasons else "passed"), reasons


def insufficient_data_reasons(metadata: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons = []
    if metadata["decision_count"] < args.min_decisions:
        reasons.append(
            "not enough champion prediction decisions: "
            f"required={args.min_decisions} "
            f"actual={metadata['decision_count']}"
        )
    if metadata["label_coverage"] < args.min_label_coverage:
        reasons.append(
            "label coverage below gate: "
            f"required={args.min_label_coverage:.6f} "
            f"actual={metadata['label_coverage']:.6f}"
        )
    if metadata["fail_count"] < args.min_fail_samples:
        reasons.append(f"not enough fail samples: required={args.min_fail_samples} actual={metadata['fail_count']}")
    if metadata["pass_count"] < args.min_pass_samples:
        reasons.append(f"not enough pass samples: required={args.min_pass_samples} actual={metadata['pass_count']}")
    return reasons


def format_metric(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def set_candidate_tags(
        client: MlflowClient,
        model_name: str,
        candidate_version: str,
        status: str,
        reasons: list[str],
        args: argparse.Namespace,
        metadata: dict[str, Any],
        candidate_metrics: dict[str, Any] | None = None,
        champion_metrics: dict[str, Any] | None = None,
) -> None:
    tags = {
        "candidate_serving_snapshot_eval_status": status,
        "candidate_serving_snapshot_eval_reason": " | ".join(reasons) if reasons else "ok",
        "candidate_serving_snapshot_eval_cohort_start_time": str(args.cohort_start_time),
        "candidate_serving_snapshot_eval_cohort_end_time": str(metadata["cohort_end_time"]),
        "candidate_serving_snapshot_eval_cutoff_time": str(args.cutoff_time),
        "candidate_serving_snapshot_eval_label_maturity_seconds": str(args.label_maturity_seconds),
        "candidate_serving_snapshot_eval_decision_selection": metadata["decision_selection"],
        "candidate_serving_snapshot_eval_max_decisions": str(metadata["max_decisions"]),
        "candidate_serving_snapshot_eval_decisions": str(metadata["decision_count"]),
        "candidate_serving_snapshot_eval_labeled_decisions": str(metadata["labeled_decision_count"]),
        "candidate_serving_snapshot_eval_label_coverage": str(metadata["label_coverage"]),
        "candidate_serving_snapshot_eval_fail_samples": str(metadata["fail_count"]),
        "candidate_serving_snapshot_eval_pass_samples": str(metadata["pass_count"]),
    }

    if candidate_metrics is not None and champion_metrics is not None:
        for name in metric_names():
            tags[f"candidate_serving_snapshot_candidate_{name}"] = format_metric(candidate_metrics.get(name))
            tags[f"candidate_serving_snapshot_champion_{name}"] = format_metric(champion_metrics.get(name))
            tags[f"candidate_serving_snapshot_delta_{name}"] = format_metric(
                metric_delta(candidate_metrics, champion_metrics, name)
            )

    for key, value in tags.items():
        client.set_model_version_tag(model_name, candidate_version, key, value)


def build_eval_summary(
        *,
        args: argparse.Namespace,
        metadata: dict[str, Any],
        candidate: dict[str, Any],
        champion: dict[str, Any],
        status: str,
        reasons: list[str],
        candidate_metrics: dict[str, Any] | None,
        champion_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    metric_names_for_summary = metric_names()

    return {
        "comparison_type": "serving_prediction_decision_candidate_vs_champion",
        "cohort_start_time": args.cohort_start_time,
        "cohort_end_time": metadata["cohort_end_time"],
        "cutoff_time": args.cutoff_time,
        "label_maturity_seconds": args.label_maturity_seconds,
        "decision_selection": metadata["decision_selection"],
        "max_decisions": args.max_decisions,
        "primary_metric": args.primary_metric,
        "eval_status": status,
        "eval_reasons": reasons,
        "metadata": metadata,
        "candidate": {
            "model_version": candidate["model_version"],
            "model_run_id": candidate["model_run_id"],
            "model_uri": candidate["model_uri"],
            "metrics": {
                name: None if candidate_metrics is None else candidate_metrics.get(name)
                for name in metric_names_for_summary
            },
        },
        "champion": {
            "model_version": champion["model_version"],
            "model_run_id": champion["model_run_id"],
            "model_uri": champion["model_uri"],
            "metrics": {
                name: None if champion_metrics is None else champion_metrics.get(name)
                for name in metric_names_for_summary
            },
        },
        "delta": {
            name: (
                None
                if candidate_metrics is None or champion_metrics is None
                else metric_delta(candidate_metrics, champion_metrics, name)
            )
            for name in metric_names_for_summary
        },
    }


def record_deployment_request(
        args: argparse.Namespace,
        candidate: dict[str, Any],
        champion: dict[str, Any],
        eval_summary: dict[str, Any],
) -> str:
    row = build_deployment_request_row(
        model_name=args.model_name,
        source_alias=candidate["model_alias"],
        source_version=candidate["model_version"],
        source_run_id=candidate["model_run_id"],
        target_alias=args.champion_alias,
        previous_version=champion["model_version"],
        previous_run_id=champion["model_run_id"],
        eval_type="serving_snapshot",
        eval_status="passed",
        approval_status=args.deployment_approval_status,
        rollout_status="not_started",
        runtime_target="release",
        eval_summary=eval_summary,
        notes=args.deployment_notes,
        requested_by=args.deployment_requested_by,
        approved_by=args.deployment_approved_by,
    )

    insert_deployment_request(row)
    return str(row["request_id"])


def print_result(
        tracking_uri: str,
        args: argparse.Namespace,
        metadata: dict[str, Any],
        candidate: dict[str, Any],
        champion: dict[str, Any],
        status: str,
        reasons: list[str],
        candidate_metrics: dict[str, Any] | None = None,
        champion_metrics: dict[str, Any] | None = None,
        deployment_request_id: str | None = None,
) -> None:
    print("serving_prediction_decision_candidate_vs_champion_comparison")
    print(f"tracking_uri={tracking_uri}")
    print(f"model_name={args.model_name}")
    print(f"cohort_start_time={args.cohort_start_time}")
    print(f"cohort_end_time={metadata['cohort_end_time']}")
    print(f"cutoff_time={args.cutoff_time}")
    print(f"label_maturity_seconds={args.label_maturity_seconds}")
    print(f"decision_selection={metadata['decision_selection']}")
    print(f"max_decisions={metadata['max_decisions']}")
    print(f"n_decisions={metadata['decision_count']}")
    print(f"n_labeled_decisions={metadata['labeled_decision_count']}")
    print(f"n_unlabeled_decisions={metadata['unlabeled_decision_count']}")
    print(f"label_coverage={metadata['label_coverage']:.6f}")
    print(f"n_fail_samples={metadata['fail_count']}")
    print(f"n_pass_samples={metadata['pass_count']}")
    print(f"first_sample_id={metadata['first_sample_id']}")
    print(f"last_sample_id={metadata['last_sample_id']}")
    print(
        f"candidate version={candidate['model_version']} alias={candidate['model_alias']} run_id={candidate['model_run_id']}")
    print(
        f"champion version={champion['model_version']} alias={champion['model_alias']} run_id={champion['model_run_id']}")

    if candidate_metrics is not None and champion_metrics is not None:
        for name in metric_names():
            print(
                f"metric={name} "
                f"candidate={format_metric(candidate_metrics.get(name))} "
                f"champion={format_metric(champion_metrics.get(name))} "
                f"delta={format_metric(metric_delta(candidate_metrics, champion_metrics, name))}"
            )

    print(f"candidate_serving_snapshot_eval_status={status}")
    for reason in reasons:
        print(f"candidate_serving_snapshot_eval_reason={reason}")

    if deployment_request_id is not None:
        print(f"deployment_request_id={deployment_request_id}")

    if status == "passed":
        print(
            "promote_command="
            f"python scripts/deployment/promote_model_alias.py "
            f"--source-alias {args.candidate_alias} "
            f"--target-alias {args.champion_alias}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    candidate = resolve_model(client, args.model_name, args.candidate_alias, args.candidate_version)
    champion = resolve_model(client, args.model_name, args.champion_alias, args.champion_version)

    features, y_true, _, metadata = load_labeled_serving_decisions(
        args,
        champion_model_run_id=champion["model_run_id"],
    )
    reasons = insufficient_data_reasons(metadata, args)

    if reasons:
        status = "insufficient_data"
        if args.set_tags:
            set_candidate_tags(client, args.model_name, candidate["model_version"], status, reasons, args, metadata)
        print_result(tracking_uri, args, metadata, candidate, champion, status, reasons)
        if args.record_deployment_request:
            print("deployment_request_skipped reason=insufficient_data")
        if args.fail_on_gate_failure:
            raise SystemExit(1)
        return

    candidate_metrics = evaluate_predictions(y_true, predict_model(candidate, features))
    champion_metrics = evaluate_predictions(y_true, predict_model(champion, features))
    status, reasons = evaluate_gate(
        candidate_metrics,
        champion_metrics,
        args.primary_metric,
        args.min_primary_delta,
        args.min_recall_delta,
        args.min_precision_delta,
    )

    if args.set_tags:
        set_candidate_tags(
            client,
            args.model_name,
            candidate["model_version"],
            status,
            reasons,
            args,
            metadata,
            candidate_metrics,
            champion_metrics,
        )

    eval_summary = build_eval_summary(
        args=args,
        metadata=metadata,
        candidate=candidate,
        champion=champion,
        status=status,
        reasons=reasons,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
    )

    deployment_request_id = None
    if args.record_deployment_request:
        if status == "passed":
            deployment_request_id = record_deployment_request(
                args=args,
                candidate=candidate,
                champion=champion,
                eval_summary=eval_summary,
            )
        else:
            print("deployment_request_skipped reason=gate_failed")

    print_result(
        tracking_uri,
        args,
        metadata,
        candidate,
        champion,
        status,
        reasons,
        candidate_metrics,
        champion_metrics,
        deployment_request_id,
    )

    if args.fail_on_gate_failure and status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
