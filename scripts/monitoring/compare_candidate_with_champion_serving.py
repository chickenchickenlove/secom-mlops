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
    FEATURE_KEYS,
    MODEL_COLUMNS,
    NUM_FEATURES,
    normalize_feature_value,
    parse_feature_object,
)

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--candidate-alias", default=DEFAULT_CANDIDATE_ALIAS)
    parser.add_argument("--champion-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--candidate-version", default=None)
    parser.add_argument("--champion-version", default=None)

    parser.add_argument("--point-time-start", type=float, required=True)
    parser.add_argument("--point-time", type=float, required=True)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--min-samples", type=int, default=500)
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


def load_labeled_serving_snapshots(args: argparse.Namespace) -> tuple[
    pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    params: list[Any] = [args.point_time_start, args.point_time]
    limit_sql = ""
    if args.limit > 0:
        limit_sql = "LIMIT %s"
        params.append(args.limit)

    sql = f"""
    WITH ranked_snapshots AS (
      SELECT
        s.*,
        ROW_NUMBER() OVER (
          PARTITION BY s.sample_id
          ORDER BY s.snapshot_time DESC, s.created_at DESC, s.serving_snapshot_id DESC
        ) AS rn
      FROM serving_feature_snapshots s
      WHERE s.is_complete = TRUE
        AND s.snapshot_status = 'complete'
        AND s.snapshot_time >= %s
        AND s.snapshot_time <= %s
    )
    SELECT
      s.sample_id,
      s.snapshot_time,
      s.feature_count,
      s.missing_count,
      s.features_json,
      a.actual_value,
      a.labeled_at
    FROM ranked_snapshots s
    JOIN actual_labels a ON a.sample_id = s.sample_id
    WHERE s.rn = 1
      AND s.snapshot_time <= a.labeled_at
    ORDER BY s.snapshot_time, s.sample_id
    {limit_sql};
    """

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    sample_ids: list[str] = []
    feature_rows: list[list[float | None]] = []
    labels: list[int] = []
    snapshot_times: list[float] = []
    missing_counts: list[int] = []
    labeled_times: list[float] = []

    for sample_id, snapshot_time, feature_count, missing_count, raw_features, actual_value, labeled_at in rows:
        if int(feature_count) != NUM_FEATURES:
            raise ValueError(f"complete snapshot must have {NUM_FEATURES} features: sample_id={sample_id}")

        features = parse_feature_object(raw_features, sample_id=str(sample_id))
        sample_ids.append(str(sample_id))
        snapshot_times.append(float(snapshot_time))
        missing_counts.append(int(missing_count))
        labeled_times.append(float(labeled_at))
        feature_rows.append([
            normalize_feature_value(features.get(key), sample_id=str(sample_id), feature_key=key)
            for key in FEATURE_KEYS
        ])
        labels.append(int(actual_value))

    x = pd.DataFrame(feature_rows, columns=list(MODEL_COLUMNS), dtype="float64")
    y = pd.Series(labels, dtype="int64")

    metadata = {
        "sample_count": len(sample_ids),
        "fail_count": int((y == POSITIVE_CLASS).sum()) if len(y) else 0,
        "pass_count": int((y == NEGATIVE_CLASS).sum()) if len(y) else 0,
        "snapshot_time_min": min(snapshot_times) if snapshot_times else None,
        "snapshot_time_max": max(snapshot_times) if snapshot_times else None,
        "missing_count_avg": float(np.mean(missing_counts)) if missing_counts else None,
        "labeled_at_min": min(labeled_times) if labeled_times else None,
        "labeled_at_max": max(labeled_times) if labeled_times else None,
        "first_sample_id": sample_ids[0] if sample_ids else None,
        "last_sample_id": sample_ids[-1] if sample_ids else None,
    }
    return x, y, sample_ids, metadata


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
    if metadata["sample_count"] < args.min_samples:
        reasons.append(
            f"not enough labeled serving snapshots: required={args.min_samples} actual={metadata['sample_count']}")
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
        "candidate_serving_snapshot_eval_point_time_start": str(args.point_time_start),
        "candidate_serving_snapshot_eval_point_time": str(args.point_time),
        "candidate_serving_snapshot_eval_samples": str(metadata["sample_count"]),
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
        "comparison_type": "serving_snapshot_candidate_vs_champion",
        "point_time_start": args.point_time_start,
        "point_time": args.point_time,
        "limit": None if args.limit <= 0 else args.limit,
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
    print("serving_candidate_vs_champion_comparison")
    print(f"tracking_uri={tracking_uri}")
    print(f"model_name={args.model_name}")
    print(f"point_time_start={args.point_time_start}")
    print(f"point_time={args.point_time}")
    print(f"n_samples={metadata['sample_count']}")
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
    if args.point_time <= args.point_time_start:
        raise ValueError("point_time must be greater than point_time_start")

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    candidate = resolve_model(client, args.model_name, args.candidate_alias, args.candidate_version)
    champion = resolve_model(client, args.model_name, args.champion_alias, args.champion_version)

    features, y_true, _, metadata = load_labeled_serving_snapshots(args)
    reasons = insufficient_data_reasons(metadata, args)

    if reasons:
        status = "insufficient_data"
        if args.set_tags:
            set_candidate_tags(client, args.model_name, candidate["model_version"], status, reasons, args, metadata)
        print_result(tracking_uri, args, metadata, candidate, champion, status, reasons)
        if args.record_deployment_request:
            print("deployment_request_skipped reason=insufficient_data")
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
