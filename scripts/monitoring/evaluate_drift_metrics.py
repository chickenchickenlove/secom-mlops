import argparse
import time
from collections import Counter
from typing import Any
from uuid import uuid4

from psycopg.rows import dict_row

from secom_mlops.monitor.db import connect
from secom_mlops_common.cli.validators import positive_int
from secom_mlops_common.logging import configure_logging, get_logger
from secom_mlops_common.metrics.stats import (
    as_metric_value,
    delta,
    empty_feature_stats,
    feature_mean,
    feature_std,
    feature_vector,
    first_present,
    mean,
    percentile,
    ratio,
    update_feature_stats,
)
from secom_mlops_common.schemas.secom import NUM_FEATURES, feature_key

WINDOW_TYPE = "recent_vs_previous_time_window"
POSITIVE_CLASS = 1
logger = get_logger(__name__)

FETCH_SQL = """
WITH anchors AS (
  SELECT
    model_run_id,
    threshold,
    MAX(predicted_at) AS current_end
  FROM prediction_logs
  {anchor_where}
  GROUP BY model_run_id, threshold
)
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
  p.features_json,
  p.missing_count,
  a.current_end
FROM prediction_logs p
JOIN anchors a
  ON a.model_run_id = p.model_run_id
 AND a.threshold = p.threshold
WHERE p.predicted_at > a.current_end - %s
  AND p.predicted_at <= a.current_end
  {prediction_where}
ORDER BY
  p.model_run_id,
  p.threshold,
  p.predicted_at,
  p.prediction_id;
"""

INSERT_SQL = """
INSERT INTO drift_metrics (
  evaluation_id,
  computed_at,
  window_type,
  window_minutes,
  baseline_start,
  baseline_end,
  current_start,
  current_end,
  model_name,
  model_version,
  model_alias,
  model_run_id,
  threshold,
  metric_family,
  metric_name,
  feature_name,
  metric_value,
  baseline_value,
  current_value,
  delta_value,
  baseline_samples,
  current_samples,
  created_at
)
VALUES (
  %(evaluation_id)s,
  %(computed_at)s,
  %(window_type)s,
  %(window_minutes)s,
  %(baseline_start)s,
  %(baseline_end)s,
  %(current_start)s,
  %(current_end)s,
  %(model_name)s,
  %(model_version)s,
  %(model_alias)s,
  %(model_run_id)s,
  %(threshold)s,
  %(metric_family)s,
  %(metric_name)s,
  %(feature_name)s,
  %(metric_value)s,
  %(baseline_value)s,
  %(current_value)s,
  %(delta_value)s,
  %(baseline_samples)s,
  %(current_samples)s,
  %(created_at)s
)
ON CONFLICT DO NOTHING;
"""


def load_prediction_rows(window_minutes: int, model_run_id: str | None) -> list[dict[str, Any]]:
    lookback_seconds = float(window_minutes * 60 * 2)
    anchor_where = ""
    prediction_where = ""
    params: list[Any] = []

    if model_run_id:
        anchor_where = "WHERE model_run_id = %s"
        prediction_where = "AND p.model_run_id = %s"
        params.append(model_run_id)

    params.append(lookback_seconds)

    if model_run_id:
        params.append(model_run_id)

    query = FETCH_SQL.format(
        anchor_where=anchor_where,
        prediction_where=prediction_where,
    )

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return list(cursor.fetchall())


def split_windows(
        rows: list[dict[str, Any]],
        current_end: float,
        window_minutes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    window_seconds = float(window_minutes * 60)

    current_start = current_end - window_seconds
    baseline_end = current_start
    baseline_start = current_end - (2.0 * window_seconds)

    baseline_rows = [
        row for row in rows
        if float(row["predicted_at"]) > baseline_start
           and float(row["predicted_at"]) <= baseline_end
    ]
    current_rows = [
        row for row in rows
        if float(row["predicted_at"]) > current_start
           and float(row["predicted_at"]) <= current_end
    ]

    return baseline_rows, current_rows, {
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "current_start": current_start,
        "current_end": current_end,
    }


def add_metric(
        metrics: list[dict[str, Any]],
        evaluation_id: str,
        computed_at: float,
        window_minutes: int,
        bounds: dict[str, float],
        group_rows: list[dict[str, Any]],
        metric_family: str,
        metric_name: str,
        feature_name: str | None,
        baseline_value: float | None,
        current_value: float | None,
        baseline_samples: int,
        current_samples: int,
        metric_value: float | None = None,
) -> None:
    if metric_value is None:
        metric_value = current_value

    metrics.append({
        "evaluation_id": evaluation_id,
        "computed_at": computed_at,
        "window_type": WINDOW_TYPE,
        "window_minutes": window_minutes,
        "baseline_start": bounds["baseline_start"],
        "baseline_end": bounds["baseline_end"],
        "current_start": bounds["current_start"],
        "current_end": bounds["current_end"],
        "model_name": first_present(group_rows, "model_name"),
        "model_version": first_present(group_rows, "model_version"),
        "model_alias": first_present(group_rows, "model_alias"),
        "model_run_id": str(group_rows[0]["model_run_id"]),
        "threshold": float(group_rows[0]["threshold"]),
        "metric_family": metric_family,
        "metric_name": metric_name,
        "feature_name": feature_name,
        "metric_value": metric_value,
        "baseline_value": baseline_value,
        "current_value": current_value,
        "delta_value": delta(current_value, baseline_value),
        "baseline_samples": baseline_samples,
        "current_samples": current_samples,
        "created_at": computed_at,
    })


def build_output_metrics(
        metrics: list[dict[str, Any]],
        evaluation_id: str,
        computed_at: float,
        window_minutes: int,
        bounds: dict[str, float],
        group_rows: list[dict[str, Any]],
        baseline_rows: list[dict[str, Any]],
        current_rows: list[dict[str, Any]],
) -> None:
    baseline_probabilities = [float(row["fail_probability"]) for row in baseline_rows]
    current_probabilities = [float(row["fail_probability"]) for row in current_rows]

    baseline_fail_count = sum(1 for row in baseline_rows if int(row["predicted_value"]) == POSITIVE_CLASS)
    current_fail_count = sum(1 for row in current_rows if int(row["predicted_value"]) == POSITIVE_CLASS)

    specs = [
        ("prediction_count", float(len(baseline_rows)), float(len(current_rows))),
        ("predicted_fail_ratio", ratio(baseline_fail_count, len(baseline_rows)),
         ratio(current_fail_count, len(current_rows))),
        ("fail_probability_avg", mean(baseline_probabilities), mean(current_probabilities)),
        ("fail_probability_p50", percentile(baseline_probabilities, 0.50), percentile(current_probabilities, 0.50)),
        ("fail_probability_p95", percentile(baseline_probabilities, 0.95), percentile(current_probabilities, 0.95)),
    ]

    for metric_name, baseline_value, current_value in specs:
        add_metric(
            metrics, evaluation_id, computed_at, window_minutes, bounds,
            group_rows, "output", metric_name, None,
            baseline_value, current_value, len(baseline_rows), len(current_rows),
        )


def build_input_metrics(
        metrics: list[dict[str, Any]],
        evaluation_id: str,
        computed_at: float,
        window_minutes: int,
        bounds: dict[str, float],
        group_rows: list[dict[str, Any]],
        baseline_rows: list[dict[str, Any]],
        current_rows: list[dict[str, Any]],
) -> None:
    baseline_missing = [float(row["missing_count"]) for row in baseline_rows]
    current_missing = [float(row["missing_count"]) for row in current_rows]

    specs = [
        ("missing_count_avg", mean(baseline_missing), mean(current_missing)),
        ("missing_count_p95", percentile(baseline_missing, 0.95), percentile(current_missing, 0.95)),
        ("missing_count_max", max(baseline_missing) if baseline_missing else None,
         max(current_missing) if current_missing else
         None),
    ]

    for metric_name, baseline_value, current_value in specs:
        add_metric(
            metrics, evaluation_id, computed_at, window_minutes, bounds,
            group_rows, "input", metric_name, None,
            baseline_value, current_value, len(baseline_rows), len(current_rows),
        )


def build_feature_metrics(
        metrics: list[dict[str, Any]],
        evaluation_id: str,
        computed_at: float,
        window_minutes: int,
        bounds: dict[str, float],
        group_rows: list[dict[str, Any]],
        baseline_rows: list[dict[str, Any]],
        current_rows: list[dict[str, Any]],
        feature_count: int,
        top_n_features: int,
        min_feature_non_null: int,
        min_feature_samples: int,
) -> None:
    baseline_stats = empty_feature_stats(feature_count)
    current_stats = empty_feature_stats(feature_count)

    update_feature_stats(baseline_stats, baseline_rows, feature_count)
    update_feature_stats(current_stats, current_rows, feature_count)

    mean_scores: list[tuple[float, int, float, float]] = []
    missing_scores: list[tuple[float, int, float, float]] = []

    for index in range(feature_count):
        baseline = baseline_stats[index]
        current = current_stats[index]

        baseline_samples = int(baseline["samples"])
        current_samples = int(current["samples"])
        baseline_non_null = int(baseline["non_null"])
        current_non_null = int(current["non_null"])

        if baseline_samples >= min_feature_samples and current_samples >= min_feature_samples:
            baseline_null_ratio = ratio(int(baseline["null_count"]), baseline_samples)
            current_null_ratio = ratio(int(current["null_count"]), current_samples)

            if baseline_null_ratio is not None and current_null_ratio is not None:
                missing_scores.append((
                    abs(current_null_ratio - baseline_null_ratio),
                    index,
                    baseline_null_ratio,
                    current_null_ratio,
                ))

        if baseline_non_null >= min_feature_non_null and current_non_null >= min_feature_non_null:
            baseline_mean = feature_mean(baseline)
            current_mean = feature_mean(current)

            if baseline_mean is None or current_mean is None:
                continue

            baseline_std = feature_std(baseline, insufficient_value=0.0)
            current_std = feature_std(current, insufficient_value=0.0)
            denominator = max(abs(baseline_std), abs(current_std), 1e-9)
            standardized_delta = abs(current_mean - baseline_mean) / denominator

            mean_scores.append((float(standardized_delta), index, baseline_mean, current_mean))

    mean_scores.sort(key=lambda item: (-item[0], item[1]))
    missing_scores.sort(key=lambda item: (-item[0], item[1]))

    for score, index, baseline_value, current_value in mean_scores[:top_n_features]:
        add_metric(
            metrics, evaluation_id, computed_at, window_minutes, bounds,
            group_rows, "feature", "feature_mean_standardized_delta", feature_key(index),
            baseline_value, current_value,
            int(baseline_stats[index]["non_null"]),
            int(current_stats[index]["non_null"]),
            metric_value=score,
        )

    for score, index, baseline_value, current_value in missing_scores[:top_n_features]:
        add_metric(
            metrics, evaluation_id, computed_at, window_minutes, bounds,
            group_rows, "feature", "feature_missing_ratio_abs_delta", feature_key(index),
            baseline_value, current_value,
            int(baseline_stats[index]["samples"]),
            int(current_stats[index]["samples"]),
            metric_value=score,
        )


def build_metrics(
        rows: list[dict[str, Any]],
        window_minutes: int,
        feature_count: int,
        top_n_features: int,
        min_feature_non_null: int,
        min_feature_samples: int,
) -> list[dict[str, Any]]:
    evaluation_id = str(uuid4())
    computed_at = time.time()
    metrics: list[dict[str, Any]] = []

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["model_run_id"]), float(row["threshold"]))
        grouped.setdefault(key, []).append(row)

    for group_rows in grouped.values():
        current_end = max(float(row["current_end"]) for row in group_rows)
        baseline_rows, current_rows, bounds = split_windows(group_rows, current_end, window_minutes)

        if not current_rows:
            continue

        build_output_metrics(metrics, evaluation_id, computed_at, window_minutes, bounds, group_rows, baseline_rows,
                             current_rows)
        build_input_metrics(metrics, evaluation_id, computed_at, window_minutes, bounds, group_rows, baseline_rows,
                            current_rows)
        build_feature_metrics(
            metrics, evaluation_id, computed_at, window_minutes, bounds,
            group_rows, baseline_rows, current_rows,
            feature_count, top_n_features, min_feature_non_null, min_feature_samples,
        )

    return metrics


def save_metrics(metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        return

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(INSERT_SQL, metrics)


def evaluate_once(args: argparse.Namespace) -> None:
    rows = load_prediction_rows(args.window_minutes, args.model_run_id)

    if not rows:
        logger.info("drift_metrics_evaluation_skipped reason=no_prediction_rows")
        return

    metrics = build_metrics(
        rows=rows,
        window_minutes=args.window_minutes,
        feature_count=args.feature_count,
        top_n_features=args.top_n_features,
        min_feature_non_null=args.min_feature_non_null,
        min_feature_samples=args.min_feature_samples,
    )

    if not args.dry_run:
        save_metrics(metrics)

    counts = Counter(row["metric_family"] for row in metrics)

    logger.info(
        "drift_metrics_evaluation_finished "
        "dry_run=%s "
        "prediction_rows=%s "
        "metric_rows=%s "
        "output_rows=%s "
        "input_rows=%s "
        "feature_rows=%s "
        "window_minutes=%s",
        args.dry_run,
        len(rows),
        len(metrics),
        counts.get("output", 0),
        counts.get("input", 0),
        counts.get("feature", 0),
        args.window_minutes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interval-seconds", type=positive_int, default=30)
    parser.add_argument("--window-minutes", type=positive_int, default=5)
    parser.add_argument("--feature-count", type=positive_int, default=NUM_FEATURES)
    parser.add_argument("--top-n-features", type=positive_int, default=30)
    parser.add_argument("--min-feature-non-null", type=positive_int, default=30)
    parser.add_argument("--min-feature-samples", type=positive_int, default=30)
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
        "drift_metrics_evaluator_loop_started "
        "interval_seconds=%s "
        "window_minutes=%s "
        "top_n_features=%s",
        args.interval_seconds,
        args.window_minutes,
        args.top_n_features,
    )

    while True:
        started_at = time.monotonic()
        evaluate_once(args)
        elapsed = time.monotonic() - started_at
        sleep_seconds = max(1.0, args.interval_seconds - elapsed)

        logger.info(
            "drift_metrics_evaluator_sleep "
            "elapsed_seconds=%.2f "
            "sleep_seconds=%.2f",
            elapsed,
            sleep_seconds,
        )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
