import argparse
import time
from uuid import uuid4

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from secom_mlops.monitor.db import connect
from secom_mlops.monitor.metrics import ModelMetricStore

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cutoff-time", type=float, required=True)
    parser.add_argument("--model-run-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-fail-count", type=int, default=5)
    return parser.parse_args()


def load_labeled_offline_predictions(
        build_cutoff_time: float,
        model_run_id: str | None,
        limit: int | None,
) -> pd.DataFrame:
    params = [build_cutoff_time]
    model_filter = ""

    if model_run_id is not None:
        model_filter = "AND p.model_run_id = %s"
        params.append(model_run_id)

    query = f"""
    SELECT
      p.offline_prediction_id,
      p.offline_snapshot_id,
      p.sample_id,
      p.build_cutoff_time,
      p.model_run_id,
      p.threshold,
      p.predicted_at,
      p.fail_probability,
      p.predicted_value,
      p.predicted_label,
      p.missing_count,
      a.actual_value,
      a.actual_label
    FROM offline_prediction_logs p
    JOIN actual_labels a
      ON p.sample_id = a.sample_id
    WHERE p.build_cutoff_time = %s
      {model_filter}
    ORDER BY p.predicted_at DESC;
    """

    if limit is not None:
        query = query.rstrip().rstrip(";") + "\nLIMIT %s;"
        params.append(limit)

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [desc.name for desc in cursor.description]

    return pd.DataFrame(rows, columns=columns)


def evaluate(df: pd.DataFrame) -> dict:
    y_true = df["actual_value"]
    y_pred = df["predicted_value"]
    fail_probability = df["fail_probability"]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
    )
    tn, fp, fn, tp = matrix.ravel()
    actual_fail_count = int((y_true == POSITIVE_CLASS).sum())

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
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
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "pr_auc": None,
    }

    if actual_fail_count > 0:
        metrics["pr_auc"] = float(
            average_precision_score(
                y_true,
                fail_probability,
                pos_label=POSITIVE_CLASS,
            )
        )

    return metrics


def build_metric_rows(
        metrics: dict,
        df: pd.DataFrame,
        build_cutoff_time: float,
        model_run_id: str,
        threshold: float | None,
) -> list[dict]:
    now = time.time()
    evaluation_id = str(uuid4())

    return [
        {
            "evaluation_id": evaluation_id,
            "computed_at": now,
            "model_run_id": model_run_id,
            "threshold": threshold,
            "window_type": "offline_snapshot_cutoff",
            "window_size": len(df),
            "window_start": build_cutoff_time,
            "window_end": build_cutoff_time,
            "metric_name": name,
            "metric_value": value,
            "n_samples": len(df),
            "n_fail_samples": int((df["actual_value"] == POSITIVE_CLASS).sum()),
            "positive_class": POSITIVE_CLASS,
            "created_at": now,
        }
        for name, value in metrics.items()
    ]


def main() -> None:
    args = parse_args()

    df = load_labeled_offline_predictions(
        build_cutoff_time=args.build_cutoff_time,
        model_run_id=args.model_run_id,
        limit=args.limit,
    )

    if df.empty:
        print("No labeled offline predictions found.")
        return

    store = ModelMetricStore()
    saved_rows = 0

    group_keys = ["build_cutoff_time", "model_run_id", "threshold"]
    for (build_cutoff_time, model_run_id, threshold), group_df in df.groupby(group_keys, dropna=False):
        threshold_value = None if pd.isna(threshold) else float(threshold)
        metrics = evaluate(group_df)

        rows = build_metric_rows(
            metrics=metrics,
            df=group_df,
            build_cutoff_time=float(build_cutoff_time),
            model_run_id=model_run_id,
            threshold=threshold_value,
        )

        store.append_many(rows)
        saved_rows += len(rows)

        actual_fail_count = int((group_df["actual_value"] == POSITIVE_CLASS).sum())

        print(f"build_cutoff_time: {float(build_cutoff_time)}")
        print(f"model_run_id: {model_run_id}")
        print(f"threshold: {threshold_value}")
        print(f"labeled_offline_predictions: {len(group_df)}")
        print(f"actual_fail_count: {actual_fail_count}")

        for key, value in metrics.items():
            print(f"{key}: {value}")

        if actual_fail_count < args.min_fail_count:
            print(f"warning: actual_fail_count is below {args.min_fail_count}")

    print(f"saved_model_metric_rows: {saved_rows}")


if __name__ == "__main__":
    main()
