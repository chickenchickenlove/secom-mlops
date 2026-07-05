import argparse
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd

from secom_mlops.monitor.prediction_metrics import PredictionWindowMetricStore
from secom_mlops.monitor.db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1

# missing count : 한 row 안에서 feature 값이 비어있는 갯수.
#                 missing count가 갑자기 늘어나는 게 중요한 이상 신호일 수도 있음.


def load_predictions(limit: int | None, model_run_id: str | None) -> pd.DataFrame:
    params = []
    where_sql = ""

    if model_run_id is not None:
        where_sql = "WHERE model_run_id = %s"
        params.append(model_run_id)

    query = f"""
    SELECT
      prediction_id,
      request_id,
      sample_id,
      model_run_id,
      threshold,
      predicted_at,
      fail_probability,
      predicted_value,
      predicted_label,
      missing_count,
      latency_ms
    FROM prediction_logs
    {where_sql}
    ORDER BY predicted_at DESC, prediction_id DESC
    """

    if limit is not None:
        query += "\nLIMIT %s"
        params.append(limit)


    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [desc.name for desc in cursor.description]

            return pd.DataFrame(rows, columns=columns)

def evaluate(df: pd.DataFrame) -> dict:
    fail_probability = df["fail_probability"]
    missing_count = df["missing_count"]
    latency_ms = df["latency_ms"]

    predicted_fail_count = int((df["predicted_value"] == POSITIVE_CLASS).sum())
    predicted_pass_count = int((df["predicted_value"] == NEGATIVE_CLASS).sum())
    prediction_count = int(len(df))

    return {
        "prediction_count": prediction_count,
        "request_count": int(df["request_id"].nunique()),
        "predicted_fail_count": predicted_fail_count,
        "predicted_pass_count": predicted_pass_count,
        "predicted_fail_ratio": predicted_fail_count / prediction_count,
        "fail_probability_mean": float(fail_probability.mean()),
        "fail_probability_p50": float(fail_probability.quantile(0.50)),
        "fail_probability_p95": float(fail_probability.quantile(0.95)),
        "fail_probability_min": float(fail_probability.min()),
        "fail_probability_max": float(fail_probability.max()),
        "missing_count_mean": float(missing_count.mean()),
        "missing_count_p50": float(missing_count.quantile(0.50)),
        "missing_count_p95": float(missing_count.quantile(0.95)),
        "missing_count_max": float(missing_count.max()),
        "latency_ms_mean": float(latency_ms.mean()),
        "latency_ms_p95": float(latency_ms.quantile(0.95)),
    }


def build_metric_rows(
        metrics: dict,
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
            "window_type": "recent_n_predictions",
            "window_size": limit,
            "window_start": float(predicted_at.min()),
            "window_end": float(predicted_at.max()),
            "metric_name": name,
            "metric_value": value,
            "n_predictions": len(df),
            "created_at": now,
        }
        for name, value in metrics.items()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model-run-id", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_predictions(args.limit, args.model_run_id)

    if df.empty:
        print("No predictions found.")
        return

    store = PredictionWindowMetricStore()
    saved_rows = 0

    if args.limit is not None and len(df) < args.limit:
        print(
            "Not enough labeled predictions for a full window: "
            f"required={args.limit}, actual={len(df)}. "
            "Metric calculation skipped."
        )
        return

    for (model_run_id, threshold), group_df in df.groupby(
            ["model_run_id", "threshold"],
            dropna=False,
    ):
        threshold_value = None if pd.isna(threshold) else float(threshold)
        metrics = evaluate(group_df)
        rows = build_metric_rows(
            metrics,
            group_df,
            args.limit,
            model_run_id,
            threshold_value,
        )

        store.append_many(rows)
        saved_rows += len(rows)

        print(f"model_run_id: {model_run_id}")
        print(f"threshold: {threshold_value}")
        print(f"prediction_window_size: {len(group_df)}")

        for key, value in metrics.items():
            print(f"{key}: {value}")

    print(f"saved_prediction_window_metric_rows: {saved_rows}")


if __name__ == "__main__":
    main()
