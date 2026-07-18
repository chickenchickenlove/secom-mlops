import argparse
import os
from typing import Any

import mlflow
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
    parser.add_argument("--candidate-version", default=None)
    parser.add_argument("--champion-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--champion-version", default=None)

    parser.add_argument("--build-cutoff-time", type=float, required=True)
    parser.add_argument("--limit", type=int, default=10000)

    parser.add_argument("--primary-metric", default="fail_f1")
    parser.add_argument("--min-primary-delta", type=float, default=0.0)
    parser.add_argument("--min-recall-delta", type=float, default=-0.02)
    parser.add_argument("--min-precision-delta", type=float, default=-0.05)

    parser.add_argument("--record-deployment-request", action="store_true")
    parser.add_argument("--deployment-approval-status", default="approved")
    parser.add_argument("--deployment-notes", default=None)
    parser.add_argument("--deployment-requested-by", default=os.getenv("USER"))
    parser.add_argument("--deployment-approved-by", default=None)
    return parser.parse_args()


def resolve_model(
        client: MlflowClient,
        model_name: str,
        alias: str,
        version: str | None,
) -> dict[str, Any]:
    if version is not None:
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
        "model_name": model_name,
        "model_version": str(model_version.version),
        "model_alias": resolved_alias,
        "model_run_id": str(model_version.run_id),
        "version_tags": dict(model_version.tags or {}),
    }


def load_labeled_snapshots(
        build_cutoff_time: float,
        limit: int | None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    params: list[Any] = [build_cutoff_time]
    limit_sql = ""

    if limit is not None and limit > 0:
        limit_sql = "LIMIT %s"
        params.append(limit)

    sql = f"""
    SELECT
      s.sample_id,
      s.features_json,
      a.actual_value
    FROM offline_feature_snapshots s
    JOIN actual_labels a
      ON a.sample_id = s.sample_id
    WHERE s.build_cutoff_time = %s
      AND s.is_complete = TRUE
    ORDER BY s.sample_id
    {limit_sql};
    """

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(
            f"no labeled offline snapshots found: build_cutoff_time={build_cutoff_time}"
        )

    sample_ids = []
    feature_rows = []
    labels = []

    for sample_id, raw_features, actual_value in rows:
        features = parse_feature_object(raw_features)
        sample_ids.append(str(sample_id))
        feature_rows.append([normalize_feature_value(features.get(key)) for key in FEATURE_KEYS])
        labels.append(int(actual_value))

    return (
        pd.DataFrame(feature_rows, columns=list(MODEL_COLUMNS), dtype="float64"),
        pd.Series(labels, dtype="int64"),
        sample_ids,
    )


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


def evaluate_predictions(
        y_true: pd.Series,
        prediction_frame: pd.DataFrame,
) -> dict[str, float | int | None]:
    y_pred = prediction_frame["predicted_value"].astype(int)
    fail_probability = prediction_frame["fail_probability"].astype(float)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
    )
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    n_fail_samples = int((y_true == POSITIVE_CLASS).sum())

    pr_auc = None
    if n_fail_samples > 0:
        pr_auc = float(
            average_precision_score(
                y_true,
                fail_probability,
                pos_label=POSITIVE_CLASS,
            )
        )

    return {
        "n_samples": int(len(y_true)),
        "n_fail_samples": n_fail_samples,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fail_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "fail_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "fail_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_CLASS,
                zero_division=0,
            )
        ),
        "pr_auc": pr_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def metric_delta(
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
        metric_name: str,
) -> float | None:
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
) -> tuple[bool, list[str]]:
    reasons = []

    primary_delta = metric_delta(candidate_metrics, champion_metrics, primary_metric)
    if primary_delta is None:
        reasons.append(f"primary metric unavailable: {primary_metric}")
    elif primary_delta < min_primary_delta:
        reasons.append(
            f"{primary_metric} delta below gate: "
            f"delta={primary_delta:.6f} required>={min_primary_delta:.6f}"
        )

    recall_delta = metric_delta(candidate_metrics, champion_metrics, "fail_recall")
    if recall_delta is not None and recall_delta < min_recall_delta:
        reasons.append(
            f"fail_recall regression too large: "
            f"delta={recall_delta:.6f} required>={min_recall_delta:.6f}"
        )

    precision_delta = metric_delta(candidate_metrics, champion_metrics, "fail_precision")
    if precision_delta is not None and precision_delta < min_precision_delta:
        reasons.append(
            f"fail_precision regression too large: "
            f"delta={precision_delta:.6f} required>={min_precision_delta:.6f}"
        )

    return len(reasons) == 0, reasons


def format_metric(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_metric_summary(
        build_cutoff_time: float,
        limit: int | None,
        candidate: dict[str, Any],
        champion: dict[str, Any],
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
        primary_metric: str,
        passed: bool,
        reasons: list[str],
) -> dict[str, Any]:
    metric_names = [
        "n_samples",
        "n_fail_samples",
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

    return {
        "comparison_type": "offline_candidate_vs_champion",
        "build_cutoff_time": build_cutoff_time,
        "limit": limit,
        "primary_metric": primary_metric,
        "eval_status": "passed" if passed else "failed",
        "eval_reasons": reasons,
        "candidate": {
            "model_version": candidate["model_version"],
            "model_run_id": candidate["model_run_id"],
            "model_uri": candidate["model_uri"],
            "metrics": {
                name: candidate_metrics.get(name)
                for name in metric_names
            },
        },
        "champion": {
            "model_version": champion["model_version"],
            "model_run_id": champion["model_run_id"],
            "model_uri": champion["model_uri"],
            "metrics": {
                name: champion_metrics.get(name)
                for name in metric_names
            },
        },
        "delta": {
            name: metric_delta(candidate_metrics, champion_metrics, name)
            for name in metric_names
        },
    }


def record_deployment_request(
        args: argparse.Namespace,
        candidate: dict[str, Any],
        champion: dict[str, Any],
        metric_summary: dict[str, Any],
) -> str:
    row = build_deployment_request_row(
        model_name=args.model_name,
        source_alias=candidate["model_alias"],
        source_version=candidate["model_version"],
        source_run_id=candidate["model_run_id"],
        target_alias=args.champion_alias,
        previous_version=champion["model_version"],
        previous_run_id=champion["model_run_id"],
        eval_type="offline_candidate_vs_champion",
        eval_status="passed",
        approval_status=args.deployment_approval_status,
        metric_summary=metric_summary,
        notes=args.deployment_notes,
        requested_by=args.deployment_requested_by,
        approved_by=args.deployment_approved_by,
    )

    insert_deployment_request(row)
    return str(row["request_id"])


def main() -> None:
    args = parse_args()
    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    limit = None if args.limit is not None and args.limit <= 0 else args.limit

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    candidate = resolve_model(
        client=client,
        model_name=args.model_name,
        alias=args.candidate_alias,
        version=args.candidate_version,
    )
    champion = resolve_model(
        client=client,
        model_name=args.model_name,
        alias=args.champion_alias,
        version=args.champion_version,
    )

    features, y_true, sample_ids = load_labeled_snapshots(
        build_cutoff_time=args.build_cutoff_time,
        limit=limit,
    )

    candidate_predictions = predict_model(candidate, features)
    champion_predictions = predict_model(champion, features)

    candidate_metrics = evaluate_predictions(y_true, candidate_predictions)
    champion_metrics = evaluate_predictions(y_true, champion_predictions)

    passed, reasons = evaluate_gate(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        primary_metric=args.primary_metric,
        min_primary_delta=args.min_primary_delta,
        min_recall_delta=args.min_recall_delta,
        min_precision_delta=args.min_precision_delta,
    )

    metric_summary = build_metric_summary(
        build_cutoff_time=args.build_cutoff_time,
        limit=limit,
        candidate=candidate,
        champion=champion,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        primary_metric=args.primary_metric,
        passed=passed,
        reasons=reasons,
    )

    deployment_request_id = None
    if args.record_deployment_request:
        if passed:
            deployment_request_id = record_deployment_request(
                args=args,
                candidate=candidate,
                champion=champion,
                metric_summary=metric_summary,
            )
        else:
            print("deployment_request_skipped reason=gate_failed")

    print("offline_candidate_vs_champion_comparison")
    print(f"tracking_uri={tracking_uri}")
    print(f"model_name={args.model_name}")
    print(f"build_cutoff_time={args.build_cutoff_time}")
    print(f"limit={limit}")
    print(f"n_samples={len(y_true)}")
    print(f"n_fail_samples={int((y_true == POSITIVE_CLASS).sum())}")
    print(f"first_sample_id={sample_ids[0]}")
    print(f"last_sample_id={sample_ids[-1]}")

    print(
        f"candidate version={candidate['model_version']} "
        f"alias={candidate['model_alias']} "
        f"run_id={candidate['model_run_id']} "
        f"uri={candidate['model_uri']}"
    )
    print(
        f"champion version={champion['model_version']} "
        f"alias={champion['model_alias']} "
        f"run_id={champion['model_run_id']} "
        f"uri={champion['model_uri']}"
    )

    for metric_name in [
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
    ]:
        print(
            f"metric={metric_name} "
            f"candidate={format_metric(candidate_metrics.get(metric_name))} "
            f"champion={format_metric(champion_metrics.get(metric_name))} "
            f"delta={format_metric(metric_delta(candidate_metrics, champion_metrics, metric_name))}"
        )

    print(f"eval_status={'passed' if passed else 'failed'}")
    for reason in reasons:
        print(f"eval_reason={reason}")

    if deployment_request_id is not None:
        print(f"deployment_request_id={deployment_request_id}")

    if passed:
        print(
            "promote_command="
            f"python scripts/deployment/promote_model_alias.py "
            f"--source-alias {args.candidate_alias} "
            f"--target-alias {args.champion_alias}"
        )


if __name__ == "__main__":
    main()
