"""Evaluate label-backed model quality over a live sliding decision window."""

import argparse
import math
import time
from typing import Any
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
from secom_mlops.monitor.live_model_quality import LiveModelQualityEvaluationStore


POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1
WINDOW_TYPE = "sliding_time"


def non_negative_float(raw_value: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and >= 0")
    return value


def positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and > 0")
    return value


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def coverage_float(raw_value: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("value must be finite and between 0 and 1")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-maturity-seconds",
        type=non_negative_float,
        required=True,
    )
    parser.add_argument(
        "--monitoring-window-seconds",
        type=positive_float,
        required=True,
    )
    parser.add_argument("--min-decisions", type=positive_int, required=True)
    parser.add_argument(
        "--min-label-coverage",
        type=coverage_float,
        required=True,
    )
    parser.add_argument("--min-fail-samples", type=positive_int, required=True)
    parser.add_argument("--min-pass-samples", type=positive_int, required=True)
    parser.add_argument("--model-run-id", type=str, default=None)
    return parser.parse_args()


def load_decision_cohort(
        label_maturity_seconds: float,
        monitoring_window_seconds: float,
        model_run_id: str | None,
) -> tuple[float, float, float, pd.DataFrame]:
    model_filter_sql = ""
    params: dict[str, Any] = {}

    if model_run_id is not None:
        model_filter_sql = "AND p.model_run_id = %(model_run_id)s"
        params["model_run_id"] = model_run_id

    query = f"""
WITH prediction_cohort AS (
  SELECT
    p.prediction_id,
    p.request_id,
    p.sample_id,
    p.serving_snapshot_id,
    p.snapshot_version,
    p.feature_hash,
    p.model_run_id,
    p.threshold,
    p.predicted_at,
    p.fail_probability,
    p.predicted_value,
    p.predicted_label
  FROM prediction_logs p
  WHERE p.predicted_at >= %(window_start)s
    AND p.predicted_at < %(window_end)s
    {model_filter_sql}
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
  p.feature_hash,
  p.model_run_id,
  p.threshold,
  p.predicted_at,
  p.fail_probability,
  p.predicted_value,
  p.predicted_label,
  l.label_event_id,
  l.label_revision,
  l.measured_at AS label_measured_at,
  l.available_at AS label_available_at,
  l.actual_value,
  l.actual_label
FROM prediction_cohort p
LEFT JOIN labels_at_cutoff l
  ON l.sample_id = p.sample_id
ORDER BY
  p.model_run_id,
  p.threshold,
  p.predicted_at,
  p.prediction_id;
"""

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT EXTRACT(EPOCH FROM clock_timestamp())::DOUBLE PRECISION;"
            )
            cutoff_time = float(cursor.fetchone()[0])
            window_end = cutoff_time - label_maturity_seconds
            window_start = window_end - monitoring_window_seconds

            if window_start < 0.0:
                raise ValueError(
                    "cutoff_time must be >= "
                    "label_maturity_seconds + monitoring_window_seconds"
                )

            params.update({
                "cutoff_time": cutoff_time,
                "window_start": window_start,
                "window_end": window_end,
            })
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]

    return cutoff_time, window_start, window_end, pd.DataFrame(rows, columns=columns)


def evaluation_status(
        n_decisions: int,
        label_coverage: float,
        n_fail_samples: int,
        n_pass_samples: int,
        min_decisions: int,
        min_label_coverage: float,
        min_fail_samples: int,
        min_pass_samples: int,
) -> str:
    if n_decisions < min_decisions:
        return "insufficient_decisions"
    if label_coverage < min_label_coverage:
        return "insufficient_label_coverage"
    if n_fail_samples < min_fail_samples:
        return "insufficient_fail_labels"
    if n_pass_samples < min_pass_samples:
        return "insufficient_pass_labels"
    return "ok"


def confusion_counts(labeled_df: pd.DataFrame) -> dict[str, int]:
    if labeled_df.empty:
        return {
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 0,
        }

    matrix = confusion_matrix(
        labeled_df["actual_value"],
        labeled_df["predicted_value"],
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
    )
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    return {
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def scalar_metrics(
        labeled_df: pd.DataFrame,
        status: str,
) -> dict[str, float | None]:
    if status != "ok":
        return {
            "accuracy": None,
            "fail_precision": None,
            "fail_recall": None,
            "fail_f1": None,
            "fail_average_precision": None,
        }

    y_true = labeled_df["actual_value"]
    y_pred = labeled_df["predicted_value"]
    fail_probability = labeled_df["fail_probability"]

    return {
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
        "fail_average_precision": float(
            average_precision_score(
                y_true,
                fail_probability,
                pos_label=POSITIVE_CLASS,
            )
        ),
    }


def build_evaluation(
        group_df: pd.DataFrame,
        model_run_id: str,
        threshold: float,
        cutoff_time: float,
        window_start: float,
        window_end: float,
        computed_at: float,
        args: argparse.Namespace,
) -> dict[str, Any]:
    labeled_df = group_df[group_df["actual_value"].notna()].copy()
    if not labeled_df.empty:
        labeled_df["actual_value"] = pd.to_numeric(
            labeled_df["actual_value"], errors="raise"
        ).astype(int)
        labeled_df["predicted_value"] = pd.to_numeric(
            labeled_df["predicted_value"], errors="raise"
        ).astype(int)
        labeled_df["fail_probability"] = pd.to_numeric(
            labeled_df["fail_probability"], errors="raise"
        ).astype(float)

    n_decisions = len(group_df)
    n_samples = len(labeled_df)
    n_fail_samples = int((labeled_df["actual_value"] == POSITIVE_CLASS).sum())
    n_pass_samples = int((labeled_df["actual_value"] == NEGATIVE_CLASS).sum())
    label_coverage = 0.0 if n_decisions == 0 else n_samples / n_decisions

    status = evaluation_status(
        n_decisions=n_decisions,
        label_coverage=label_coverage,
        n_fail_samples=n_fail_samples,
        n_pass_samples=n_pass_samples,
        min_decisions=args.min_decisions,
        min_label_coverage=args.min_label_coverage,
        min_fail_samples=args.min_fail_samples,
        min_pass_samples=args.min_pass_samples,
    )

    evaluation = {
        "evaluation_id": str(uuid4()),
        "computed_at": computed_at,
        "model_run_id": model_run_id,
        "threshold": threshold,
        "window_type": WINDOW_TYPE,
        "cutoff_time": cutoff_time,
        "label_maturity_seconds": args.label_maturity_seconds,
        "monitoring_window_seconds": args.monitoring_window_seconds,
        "window_start": window_start,
        "window_end": window_end,
        "n_decisions": n_decisions,
        "n_samples": n_samples,
        "n_pass_samples": n_pass_samples,
        "n_fail_samples": n_fail_samples,
        "label_coverage": label_coverage,
        "min_decisions": args.min_decisions,
        "min_label_coverage": args.min_label_coverage,
        "min_pass_samples": args.min_pass_samples,
        "min_fail_samples": args.min_fail_samples,
        "evaluation_status": status,
    }
    evaluation.update(confusion_counts(labeled_df))
    evaluation.update(scalar_metrics(labeled_df, status))
    return evaluation


def main() -> None:
    args = parse_args()
    cutoff_time, window_start, window_end, cohort_df = load_decision_cohort(
        label_maturity_seconds=args.label_maturity_seconds,
        monitoring_window_seconds=args.monitoring_window_seconds,
        model_run_id=args.model_run_id,
    )

    print(
        f"cutoff_time={cutoff_time} "
        f"label_maturity_seconds={args.label_maturity_seconds} "
        f"monitoring_window_seconds={args.monitoring_window_seconds} "
        f"window_start={window_start} "
        f"window_end={window_end}"
    )

    if cohort_df.empty:
        print("No prediction decisions found in the monitoring window.")
        return

    computed_at = time.time()
    evaluations = []

    group_keys = ["model_run_id", "threshold"]
    for (model_run_id, threshold), group_df in cohort_df.groupby(
            group_keys,
            dropna=False,
    ):
        evaluation = build_evaluation(
            group_df=group_df,
            model_run_id=str(model_run_id),
            threshold=float(threshold),
            cutoff_time=cutoff_time,
            window_start=window_start,
            window_end=window_end,
            computed_at=computed_at,
            args=args,
        )
        evaluations.append(evaluation)

        print(
            f"model_run_id={evaluation['model_run_id']} "
            f"threshold={evaluation['threshold']} "
            f"evaluation_status={evaluation['evaluation_status']} "
            f"n_decisions={evaluation['n_decisions']} "
            f"n_samples={evaluation['n_samples']} "
            f"n_fail_samples={evaluation['n_fail_samples']} "
            f"n_pass_samples={evaluation['n_pass_samples']} "
            f"label_coverage={evaluation['label_coverage']}"
        )

    LiveModelQualityEvaluationStore().append_many(evaluations)
    print(f"saved_live_model_quality_evaluations={len(evaluations)}")


if __name__ == "__main__":
    main()
