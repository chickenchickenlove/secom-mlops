import argparse
import json
import math
import time
from bisect import bisect_left
from collections import Counter
from typing import Any
from uuid import uuid4

from psycopg.rows import dict_row

from secom_mlops.monitor.db import connect
from secom_mlops_common.cli.validators import positive_float, positive_int
from secom_mlops_common.metrics.stats import (
    as_metric_value,
    collect_feature_values,
    delta,
    empty_feature_stats,
    feature_mean,
    feature_std,
    feature_vector,
    first_present,
    mean,
    percentile,
    ratio,
    ratios_from_counts,
    update_feature_stats,
)
from secom_mlops_common.schemas.secom import NUM_FEATURES, feature_key

WINDOW_TYPE = "fixed_reference_time_window"
POSITIVE_CLASS = 1

BASELINE_SQL = """
SELECT
  baseline_id,
  baseline_name,
  source_start,
  source_end,
  model_name,
  model_version,
  model_alias,
  model_run_id,
  threshold,
  sample_count
FROM drift_reference_baselines
WHERE baseline_id = %s
  AND status = 'active';
"""

STATS_SQL = """
SELECT
  metric_family,
  metric_name,
  feature_name,
  metric_value,
  sample_count,
  non_null_count,
  null_count,
  metadata_json
FROM drift_reference_stats
WHERE baseline_id = %s;
"""

CURRENT_ROWS_SQL = """
WITH resolved_predictions AS NOT MATERIALIZED (
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
    s.features_json AS features_json,
    p.missing_count
  FROM prediction_logs p
  JOIN serving_feature_snapshots s
    ON s.serving_snapshot_id = p.serving_snapshot_id
   AND s.sample_id = p.sample_id
   AND s.snapshot_version = p.snapshot_version
   AND s.feature_hash = p.feature_hash
),
anchor AS (
  SELECT MAX(predicted_at) AS current_end
  FROM resolved_predictions
  WHERE model_run_id = %s
    AND threshold = %s
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
FROM resolved_predictions p
CROSS JOIN anchor a
WHERE a.current_end IS NOT NULL
  AND p.model_run_id = %s
  AND p.threshold = %s
  AND p.predicted_at > a.current_end - %s
  AND p.predicted_at <= a.current_end
ORDER BY p.predicted_at, p.prediction_id;
"""

INSERT_SQL = """
INSERT INTO drift_metrics (
  evaluation_id,
  reference_baseline_id,
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
  %(reference_baseline_id)s,
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


def load_baseline(baseline_id: str) -> dict[str, Any]:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(BASELINE_SQL, [baseline_id])
            row = cursor.fetchone()

    if row is None:
        raise ValueError(f"active drift reference baseline not found: {baseline_id}")

    return dict(row)


def load_reference_stats(baseline_id: str) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(STATS_SQL, [baseline_id])
            rows = [dict(row) for row in cursor.fetchall()]

    if not rows:
        raise ValueError(f"drift reference stats not found: {baseline_id}")

    return {
        (row["metric_family"], row["metric_name"], row["feature_name"]): row
        for row in rows
    }


def load_current_rows(baseline: dict[str, Any], window_minutes: int) -> list[dict[str, Any]]:
    lookback_seconds = float(window_minutes * 60)
    model_run_id = str(baseline["model_run_id"])
    threshold = float(baseline["threshold"])

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                CURRENT_ROWS_SQL,
                [model_run_id, threshold, model_run_id, threshold, lookback_seconds],
            )
            return [dict(row) for row in cursor.fetchall()]


def stat(
        reference_stats: dict[tuple[str, str, str | None], dict[str, Any]],
        metric_family: str,
        metric_name: str,
        feature_name: str | None = None,
) -> dict[str, Any] | None:
    return reference_stats.get((metric_family, metric_name, feature_name))


def stat_value(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    return as_metric_value(row["metric_value"])


def stat_sample_count(row: dict[str, Any] | None) -> int:
    if row is None:
        return 0
    return int(row["sample_count"])


def stat_non_null_count(row: dict[str, Any] | None) -> int:
    if row is None:
        return 0
    value = row.get("non_null_count")
    if value is None:
        return int(row["sample_count"])
    return int(value)


def stat_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}

    raw_value = row.get("metadata_json") or {}
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    if isinstance(raw_value, dict):
        return raw_value
    return dict(raw_value)


def numeric_list(raw_values: Any) -> list[float]:
    if not isinstance(raw_values, list):
        return []

    values: list[float] = []
    for raw_value in raw_values:
        value = as_metric_value(raw_value)
        if value is not None:
            values.append(value)
    return values


def bucket_counts(values: list[float], bin_edges: list[float], expected_bin_count: int) -> list[int]:
    counts = [0 for _ in range(expected_bin_count)]

    for value in values:
        bucket = bisect_left(bin_edges, value)
        bucket = max(0, min(bucket, expected_bin_count - 1))
        counts[bucket] += 1

    return counts


def population_stability_index(
        baseline_ratios: list[float],
        current_ratios: list[float],
        epsilon: float,
) -> float | None:
    if len(baseline_ratios) != len(current_ratios) or not baseline_ratios:
        return None

    score = 0.0
    for baseline_ratio, current_ratio in zip(baseline_ratios, current_ratios):
        expected = max(float(baseline_ratio), epsilon)
        actual = max(float(current_ratio), epsilon)
        score += (actual - expected) * math.log(actual / expected)

    return float(score)


def feature_psi_score(
        baseline_distribution_stat: dict[str, Any],
        current_values: list[float],
        epsilon: float,
) -> float | None:
    metadata = stat_metadata(baseline_distribution_stat)
    bin_edges = numeric_list(metadata.get("bin_edges"))
    baseline_ratios = numeric_list(metadata.get("baseline_bin_ratios"))

    if len(baseline_ratios) != len(bin_edges) + 1:
        return None

    current_counts = bucket_counts(current_values, bin_edges, len(baseline_ratios))
    current_ratios = ratios_from_counts(current_counts)
    return population_stability_index(baseline_ratios, current_ratios, epsilon)


def add_metric(
        metrics: list[dict[str, Any]],
        evaluation_id: str,
        baseline_id: str,
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
        "reference_baseline_id": baseline_id,
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
        reference_stats: dict[tuple[str, str, str | None], dict[str, Any]],
        evaluation_id: str,
        baseline: dict[str, Any],
        computed_at: float,
        window_minutes: int,
        bounds: dict[str, float],
        current_rows: list[dict[str, Any]],
) -> None:
    probabilities = [float(row["fail_probability"]) for row in current_rows]
    predicted_fail_count = sum(
        1 for row in current_rows
        if int(row["predicted_value"]) == POSITIVE_CLASS
    )

    specs = [
        ("prediction_count", float(len(current_rows))),
        ("predicted_fail_ratio", ratio(predicted_fail_count, len(current_rows))),
        ("fail_probability_avg", mean(probabilities)),
        ("fail_probability_p50", percentile(probabilities, 0.50)),
        ("fail_probability_p95", percentile(probabilities, 0.95)),
    ]

    for metric_name, current_value in specs:
        baseline_stat = stat(reference_stats, "output", metric_name)
        add_metric(
            metrics, evaluation_id, baseline["baseline_id"], computed_at, window_minutes,
            bounds, current_rows, "output", metric_name, None,
            stat_value(baseline_stat), current_value,
            stat_sample_count(baseline_stat), len(current_rows),
        )


def build_input_metrics(
        metrics: list[dict[str, Any]],
        reference_stats: dict[tuple[str, str, str | None], dict[str, Any]],
        evaluation_id: str,
        baseline: dict[str, Any],
        computed_at: float,
        window_minutes: int,
        bounds: dict[str, float],
        current_rows: list[dict[str, Any]],
) -> None:
    missing_counts = [float(row["missing_count"]) for row in current_rows]

    specs = [
        ("missing_count_avg", mean(missing_counts)),
        ("missing_count_p95", percentile(missing_counts, 0.95)),
        ("missing_count_max", max(missing_counts) if missing_counts else None),
    ]

    for metric_name, current_value in specs:
        baseline_stat = stat(reference_stats, "input", metric_name)
        add_metric(
            metrics, evaluation_id, baseline["baseline_id"], computed_at, window_minutes,
            bounds, current_rows, "input", metric_name, None,
            stat_value(baseline_stat), current_value,
            stat_sample_count(baseline_stat), len(current_rows),
        )


def build_feature_metrics(
        metrics: list[dict[str, Any]],
        reference_stats: dict[tuple[str, str, str | None], dict[str, Any]],
        evaluation_id: str,
        baseline: dict[str, Any],
        computed_at: float,
        window_minutes: int,
        bounds: dict[str, float],
        current_rows: list[dict[str, Any]],
        feature_count: int,
        top_n_features: int,
        min_feature_non_null: int,
        min_feature_samples: int,
        min_psi_feature_non_null: int,
        psi_epsilon: float,
) -> None:
    current_stats = empty_feature_stats(feature_count)
    update_feature_stats(current_stats, current_rows, feature_count)
    current_values = collect_feature_values(current_rows, feature_count)

    mean_scores: list[tuple[float, int, float, float, dict[str, Any], int]] = []
    missing_scores: list[tuple[float, int, float, float, dict[str, Any], int]] = []
    psi_scores: list[tuple[float, int, dict[str, Any], int]] = []

    for index, current in enumerate(current_stats):
        feature_name = feature_key(index)
        current_samples = int(current["samples"])
        current_non_null = int(current["non_null"])

        missing_stat = stat(reference_stats, "feature", "feature_missing_ratio", feature_name)
        baseline_missing_ratio = stat_value(missing_stat)
        current_missing_ratio = ratio(int(current["null_count"]), current_samples)

        if (
                missing_stat is not None
                and current_samples >= min_feature_samples
                and baseline_missing_ratio is not None
                and current_missing_ratio is not None
        ):
            missing_scores.append((
                abs(current_missing_ratio - baseline_missing_ratio),
                index,
                baseline_missing_ratio,
                current_missing_ratio,
                missing_stat,
                current_samples,
            ))

        mean_stat = stat(reference_stats, "feature", "feature_mean", feature_name)
        std_stat = stat(reference_stats, "feature", "feature_std", feature_name)
        baseline_mean = stat_value(mean_stat)
        baseline_std = stat_value(std_stat) or 0.0
        current_mean = feature_mean(current)
        current_std = feature_std(current) or 0.0

        if (
                mean_stat is not None
                and current_non_null >= min_feature_non_null
                and baseline_mean is not None
                and current_mean is not None
        ):
            denominator = max(abs(baseline_std), abs(current_std), 1e-9)
            score = abs(current_mean - baseline_mean) / denominator
            mean_scores.append((
                float(score),
                index,
                baseline_mean,
                current_mean,
                mean_stat,
                current_non_null,
            ))

        psi_stat = stat(reference_stats, "feature", "feature_distribution_bins", feature_name)
        psi_current_values = current_values[index]

        if (
                psi_stat is not None
                and stat_non_null_count(psi_stat) >= min_psi_feature_non_null
                and len(psi_current_values) >= min_psi_feature_non_null
        ):
            psi_score = feature_psi_score(psi_stat, psi_current_values, psi_epsilon)
            if psi_score is not None:
                psi_scores.append((
                    psi_score,
                    index,
                    psi_stat,
                    len(psi_current_values),
                ))

    mean_scores.sort(key=lambda item: (-item[0], item[1]))
    missing_scores.sort(key=lambda item: (-item[0], item[1]))
    psi_scores.sort(key=lambda item: (-item[0], item[1]))

    for score, index, baseline_value, current_value, baseline_stat, current_samples in mean_scores[:top_n_features]:
        add_metric(
            metrics, evaluation_id, baseline["baseline_id"], computed_at, window_minutes,
            bounds, current_rows, "feature", "feature_mean_standardized_delta", feature_key(index),
            baseline_value, current_value,
            stat_non_null_count(baseline_stat), current_samples,
            metric_value=score,
        )

    for score, index, baseline_value, current_value, baseline_stat, current_samples in missing_scores[:top_n_features]:
        add_metric(
            metrics, evaluation_id, baseline["baseline_id"], computed_at, window_minutes,
            bounds, current_rows, "feature", "feature_missing_ratio_abs_delta", feature_key(index),
            baseline_value, current_value,
            stat_sample_count(baseline_stat), current_samples,
            metric_value=score,
        )

    for score, index, baseline_stat, current_samples in psi_scores[:top_n_features]:
        add_metric(
            metrics, evaluation_id, baseline["baseline_id"], computed_at, window_minutes,
            bounds, current_rows, "feature", "feature_psi", feature_key(index),
            0.0, score,
            stat_non_null_count(baseline_stat), current_samples,
            metric_value=score,
        )


def build_metrics(
        baseline: dict[str, Any],
        reference_stats: dict[tuple[str, str, str | None], dict[str, Any]],
        current_rows: list[dict[str, Any]],
        window_minutes: int,
        feature_count: int,
        top_n_features: int,
        min_feature_non_null: int,
        min_feature_samples: int,
        min_psi_feature_non_null: int,
        psi_epsilon: float,
) -> list[dict[str, Any]]:
    evaluation_id = str(uuid4())
    computed_at = time.time()
    current_end = max(float(row["current_end"]) for row in current_rows)
    current_start = current_end - float(window_minutes * 60)

    bounds = {
        "baseline_start": float(baseline["source_start"]),
        "baseline_end": float(baseline["source_end"]),
        "current_start": current_start,
        "current_end": current_end,
    }

    metrics: list[dict[str, Any]] = []
    build_output_metrics(metrics, reference_stats, evaluation_id, baseline, computed_at, window_minutes, bounds,
                         current_rows)
    build_input_metrics(metrics, reference_stats, evaluation_id, baseline, computed_at, window_minutes, bounds,
                        current_rows)
    build_feature_metrics(
        metrics, reference_stats, evaluation_id, baseline, computed_at, window_minutes, bounds,
        current_rows, feature_count, top_n_features, min_feature_non_null, min_feature_samples,
        min_psi_feature_non_null, psi_epsilon,
    )

    return metrics


def save_metrics(metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        return

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(INSERT_SQL, metrics)


def evaluate_once(args: argparse.Namespace) -> None:
    baseline = load_baseline(args.reference_baseline_id)
    reference_stats = load_reference_stats(args.reference_baseline_id)
    current_rows = load_current_rows(baseline, args.window_minutes)

    if not current_rows:
        print(
            "fixed_reference_drift_evaluation_skipped "
            f"reason=no_current_prediction_rows "
            f"reference_baseline_id={args.reference_baseline_id}",
            flush=True,
        )
        return

    metrics = build_metrics(
        baseline=baseline,
        reference_stats=reference_stats,
        current_rows=current_rows,
        window_minutes=args.window_minutes,
        feature_count=args.feature_count,
        top_n_features=args.top_n_features,
        min_feature_non_null=args.min_feature_non_null,
        min_feature_samples=args.min_feature_samples,
        min_psi_feature_non_null=args.min_psi_feature_non_null,
        psi_epsilon=args.psi_epsilon,
    )

    if not args.dry_run:
        save_metrics(metrics)

    counts = Counter(row["metric_family"] for row in metrics)
    print(
        "fixed_reference_drift_evaluation_finished "
        f"dry_run={args.dry_run} "
        f"reference_baseline_id={args.reference_baseline_id} "
        f"current_rows={len(current_rows)} "
        f"metric_rows={len(metrics)} "
        f"output_rows={counts.get('output', 0)} "
        f"input_rows={counts.get('input', 0)} "
        f"feature_rows={counts.get('feature', 0)} "
        f"window_minutes={args.window_minutes}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-baseline-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window-minutes", type=positive_int, default=3)
    parser.add_argument("--feature-count", type=positive_int, default=NUM_FEATURES)
    parser.add_argument("--top-n-features", type=positive_int, default=30)
    parser.add_argument("--min-feature-non-null", type=positive_int, default=30)
    parser.add_argument("--min-feature-samples", type=positive_int, default=30)
    parser.add_argument("--min-psi-feature-non-null", type=positive_int, default=30)
    parser.add_argument("--psi-epsilon", type=positive_float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_once(args)


if __name__ == "__main__":
    main()
