import argparse
import time
from pathlib import Path
from secom_mlops.monitor.db import connect


import pandas as pd
from uuid import uuid4
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from secom_mlops.monitor.metrics import ModelMetricStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1

def load_labeled_predictions(limit: int | None,
                             model_run_id: str | None
                             ) -> pd.DataFrame:
    params = []
    where_sql = ""

    if model_run_id is not None:
        where_sql = "WHERE p.model_run_id = %s"
        params.append(model_run_id)

    query = f"""
SELECT p.prediction_id, p.sample_id, p.model_run_id, p.threshold, p.predicted_at, p.fail_probability, \
       p.predicted_value, p.predicted_label, a.actual_value, a.actual_label
FROM prediction_logs p
JOIN actual_labels a
  ON p.sample_id = a.sample_id
{where_sql}
ORDER BY p.predicted_at DESC \
"""

    if limit is not None:
        query += "\nLIMIT %s"
        params.append(limit)

    with connect() as conn:
        with conn.cursor()as cursor:
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


def build_metric_rows(metrics: dict,
                      df: pd.DataFrame,
                      limit: int | None,
                      model_run_id: str,
                      threshold: float | None,
                      ) -> list[dict]:
    now = time.time()
    evaluation_id = str(uuid4())
    predicted_at = pd.to_numeric(df["predicted_at"], errors="coerce")

    return [
        {
            "evaluation_id": evaluation_id,
            "computed_at": now,
            "model_run_id": model_run_id,
            "threshold": threshold,
            "window_type": "recent_n_labeled_predictions",
            "window_size": limit,
            "window_start": float(predicted_at.min()),
            "window_end": float(predicted_at.max()),
            "metric_name": name,
            "metric_value": value,
            "n_samples": len(df),
            "n_fail_samples": int((df["actual_value"] == POSITIVE_CLASS).sum()),
            "positive_class": POSITIVE_CLASS,
            "created_at": now,
        }
        for name, value in metrics.items()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    # SECOM 데이터 셋 전체에서 발생하는 Fail의 값이 몇 개 없기 때문
    parser.add_argument("--min-fail-count", type=int, default=5)
    parser.add_argument("--model-run-id", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_labeled_predictions(args.limit, args.model_run_id)

    if df.empty:
        print("No labeled predictions found.")
        return

    store = ModelMetricStore()
    saved_rows = 0

    for (model_run_id, threshold), group_df in df.groupby(["model_run_id", "threshold"], dropna=False):
        threshold_value = None if pd.isna(threshold) else float(threshold)
        metrics = evaluate(group_df)
        rows = build_metric_rows(metrics, group_df, args.limit, model_run_id, threshold_value)

        store.append_many(rows)
        saved_rows += len(rows)

        print(f"model_run_id: {model_run_id}")
        print(f"threshold: {threshold_value}")
        print(f"labeled_predictions: {len(group_df)}")
        print(f"actual_fail_count: {int((group_df['actual_value'] == POSITIVE_CLASS).sum())}")

        for key, value in metrics.items():
            print(f"{key}: {value}")

        if int((group_df["actual_value"] == POSITIVE_CLASS).sum()) < args.min_fail_count:
            print(f"warning: actual_fail_count is below {args.min_fail_count}")
    print(f"saved_model_metric_rows: {saved_rows}")


if __name__ == "__main__":
    main()
