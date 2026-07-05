import argparse
import time
from collections import Counter
from typing import Any

from psycopg.rows import dict_row
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from secom_mlops.monitor.db import connect
from secom_mlops_common.cli.validators import non_negative_int, positive_int
from secom_mlops_common.logging import configure_logging, get_logger

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1
WINDOW_TYPE = "non_overlapping_labeled_predictions"
logger = get_logger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_quality_windows (
  id BIGSERIAL PRIMARY KEY,
  window_type TEXT NOT NULL DEFAULT 'non_overlapping_labeled_predictions',
  window_size INTEGER NOT NULL,
  window_id INTEGER NOT NULL,
  window_start DOUBLE PRECISION NOT NULL,
  window_end DOUBLE PRECISION NOT NULL,
  computed_at DOUBLE PRECISION NOT NULL,

  model_name TEXT,
  model_version TEXT,
  model_alias TEXT,
  model_run_id TEXT NOT NULL,
  threshold DOUBLE PRECISION NOT NULL,

  n_samples INTEGER NOT NULL,
  n_fail_samples INTEGER NOT NULL,
  evaluation_status TEXT NOT NULL CHECK (
    evaluation_status IN ('ok', 'insufficient_samples', 'insufficient_fail_labels')
  ),

  accuracy DOUBLE PRECISION,
  fail_precision DOUBLE PRECISION,
  fail_recall DOUBLE PRECISION,
  fail_f1 DOUBLE PRECISION,
  pr_auc DOUBLE PRECISION,

  true_negative INTEGER NOT NULL,
  false_positive INTEGER NOT NULL,
  false_negative INTEGER NOT NULL,
  true_positive INTEGER NOT NULL,

  created_at DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),
  updated_at DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW()),

  UNIQUE (model_run_id, threshold, window_type, window_size, window_id)
);

CREATE INDEX IF NOT EXISTS idx_model_quality_windows_window_end
  ON model_quality_windows (window_end);

CREATE INDEX IF NOT EXISTS idx_model_quality_windows_status
  ON model_quality_windows (evaluation_status, window_end);
"""

FETCH_SQL = """
SELECT
  p.prediction_id,
  p.sample_id,
  p.model_name,
  p.model_version,
  p.model_alias,
  p.model_run_id,
  p.threshold,
  p.predicted_at,
  p.fail_probability,
  p.predicted_value,
  p.predicted_label,
  a.actual_value,
  a.actual_label
FROM prediction_logs p
JOIN actual_labels a
  ON a.sample_id = p.sample_id
{where_sql}
ORDER BY
  p.model_run_id,
  p.threshold,
  p.predicted_at,
  p.prediction_id;
"""

UPSERT_SQL = """
INSERT INTO model_quality_windows (
  window_type,
  window_size,
  window_id,
  window_start,
  window_end,
  computed_at,
  model_name,
  model_version,
  model_alias,
  model_run_id,
  threshold,
  n_samples,
  n_fail_samples,
  evaluation_status,
  accuracy,
  fail_precision,
  fail_recall,
  fail_f1,
  pr_auc,
  true_negative,
  false_positive,
  false_negative,
  true_positive,
  created_at,
  updated_at
)
VALUES (
  %(window_type)s,
  %(window_size)s,
  %(window_id)s,
  %(window_start)s,
  %(window_end)s,
  %(computed_at)s,
  %(model_name)s,
  %(model_version)s,
  %(model_alias)s,
  %(model_run_id)s,
  %(threshold)s,
  %(n_samples)s,
  %(n_fail_samples)s,
  %(evaluation_status)s,
  %(accuracy)s,
  %(fail_precision)s,
  %(fail_recall)s,
  %(fail_f1)s,
  %(pr_auc)s,
  %(true_negative)s,
  %(false_positive)s,
  %(false_negative)s,
  %(true_positive)s,
  %(created_at)s,
  %(updated_at)s
)
ON CONFLICT (model_run_id, threshold, window_type, window_size, window_id)
DO UPDATE SET
  window_start = EXCLUDED.window_start,
  window_end = EXCLUDED.window_end,
  computed_at = EXCLUDED.computed_at,
  model_name = EXCLUDED.model_name,
  model_version = EXCLUDED.model_version,
  model_alias = EXCLUDED.model_alias,
  n_samples = EXCLUDED.n_samples,
  n_fail_samples = EXCLUDED.n_fail_samples,
  evaluation_status = EXCLUDED.evaluation_status,
  accuracy = EXCLUDED.accuracy,
  fail_precision = EXCLUDED.fail_precision,
  fail_recall = EXCLUDED.fail_recall,
  fail_f1 = EXCLUDED.fail_f1,
  pr_auc = EXCLUDED.pr_auc,
  true_negative = EXCLUDED.true_negative,
  false_positive = EXCLUDED.false_positive,
  false_negative = EXCLUDED.false_negative,
  true_positive = EXCLUDED.true_positive,
  updated_at = EXCLUDED.updated_at;
"""


def ensure_table() -> None:
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_TABLE_SQL)


def load_labeled_predictions(model_run_id: str | None) -> list[dict[str, Any]]:
    params: list[Any] = []
    where_sql = ""

    if model_run_id:
        where_sql = "WHERE p.model_run_id = %s"
        params.append(model_run_id)

    query = FETCH_SQL.format(where_sql=where_sql)

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return list(cursor.fetchall())


def first_present(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def evaluation_status(
        n_samples: int,
        n_fail_samples: int,
        window_size: int,
        min_fail_samples: int,
) -> str:
    if n_samples < window_size:
        return "insufficient_samples"

    if n_fail_samples < min_fail_samples:
        return "insufficient_fail_labels"

    return "ok"


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["actual_value"]) for row in rows]
    y_pred = [int(row["predicted_value"]) for row in rows]
    fail_probability = [float(row["fail_probability"]) for row in rows]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
    )

    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    n_fail_samples = sum(1 for value in y_true if value == POSITIVE_CLASS)

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
        "pr_auc": pr_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def build_quality_windows(
        rows: list[dict[str, Any]],
        window_size: int,
        min_fail_samples: int,
) -> list[dict[str, Any]]:
    computed_at = time.time()
    windows: list[dict[str, Any]] = []

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model_run_id"]), float(row["threshold"]))
        grouped.setdefault(key, []).append(row)

    for (model_run_id, threshold), group_rows in grouped.items():
        group_rows.sort(key=lambda row: (float(row["predicted_at"]), str(row["prediction_id"])))

        for window_id, start in enumerate(range(0, len(group_rows), window_size)):
            window_rows = group_rows[start: start + window_size]
            n_samples = len(window_rows)
            n_fail_samples = sum(
                1 for row in window_rows if int(row["actual_value"]) == POSITIVE_CLASS
            )
            metrics = compute_metrics(window_rows)

            window = {
                "window_type": WINDOW_TYPE,
                "window_size": window_size,
                "window_id": window_id,
                "window_start": float(window_rows[0]["predicted_at"]),
                "window_end": float(window_rows[-1]["predicted_at"]),
                "computed_at": computed_at,
                "model_name": first_present(window_rows, "model_name"),
                "model_version": first_present(window_rows, "model_version"),
                "model_alias": first_present(window_rows, "model_alias"),
                "model_run_id": model_run_id,
                "threshold": threshold,
                "n_samples": n_samples,
                "n_fail_samples": n_fail_samples,
                "evaluation_status": evaluation_status(
                    n_samples=n_samples,
                    n_fail_samples=n_fail_samples,
                    window_size=window_size,
                    min_fail_samples=min_fail_samples,
                ),
                "created_at": computed_at,
                "updated_at": computed_at,
            }
            window.update(metrics)
            windows.append(window)

    return windows


def save_quality_windows(windows: list[dict[str, Any]]) -> None:
    if not windows:
        return

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(UPSERT_SQL, windows)


def evaluate_once(args: argparse.Namespace) -> None:
    ensure_table()

    rows = load_labeled_predictions(args.model_run_id)
    if not rows:
        logger.info("quality_window_evaluation_skipped reason=no_labeled_predictions")
        return

    windows = build_quality_windows(
        rows=rows,
        window_size=args.window_size,
        min_fail_samples=args.min_fail_samples,
    )

    if not args.dry_run:
        save_quality_windows(windows)

    status_counts = Counter(window["evaluation_status"] for window in windows)

    logger.info(
        "quality_window_evaluation_finished "
        "dry_run=%s "
        "labeled_predictions=%s "
        "windows=%s "
        "window_size=%s "
        "min_fail_samples=%s "
        "ok=%s "
        "insufficient_samples=%s "
        "insufficient_fail_labels=%s",
        args.dry_run,
        len(rows),
        len(windows),
        args.window_size,
        args.min_fail_samples,
        status_counts.get("ok", 0),
        status_counts.get("insufficient_samples", 0),
        status_counts.get("insufficient_fail_labels", 0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interval-seconds", type=positive_int, default=60)
    parser.add_argument("--window-size", type=positive_int, default=500)
    parser.add_argument("--min-fail-samples", type=non_negative_int, default=10)
    parser.add_argument("--model-run-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.loop and args.once:
        raise ValueError("Use either --loop or --once, not both.")

    if not args.loop:
        evaluate_once(args)
        return

    logger.info(
        "quality_window_evaluator_loop_started "
        "interval_seconds=%s "
        "window_size=%s "
        "min_fail_samples=%s",
        args.interval_seconds,
        args.window_size,
        args.min_fail_samples,
    )

    while True:
        started_at = time.monotonic()
        evaluate_once(args)
        elapsed = time.monotonic() - started_at
        sleep_seconds = max(1.0, args.interval_seconds - elapsed)

        logger.info(
            "quality_window_evaluator_sleep "
            "elapsed_seconds=%.2f "
            "sleep_seconds=%.2f",
            elapsed,
            sleep_seconds,
        )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
