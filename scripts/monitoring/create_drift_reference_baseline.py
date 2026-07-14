import argparse
import math
import os
import time
from bisect import bisect_left
from typing import Any
from uuid import uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from secom_mlops.monitor.db import connect
from secom_mlops_common.cli.validators import non_negative_float, positive_int
from secom_mlops_common.metrics.stats import (
    collect_feature_values,
    empty_feature_stats,
    feature_mean,
    feature_std,
    first_present,
    mean,
    percentile,
    quantile_from_sorted,
    ratio,
    ratios_from_counts,
    update_feature_stats,
)
from secom_mlops_common.schemas.secom import NUM_FEATURES, feature_key

POSITIVE_CLASS = 1

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
  s.features_json AS features_json,
  p.missing_count
FROM prediction_logs p
JOIN serving_feature_snapshots s
  ON s.serving_snapshot_id = p.serving_snapshot_id
 AND s.sample_id = p.sample_id
 AND s.snapshot_version = p.snapshot_version
 AND s.feature_hash = p.feature_hash
WHERE p.model_run_id = %s
  AND p.threshold = %s
  AND p.predicted_at > %s
  AND p.predicted_at <= %s
ORDER BY p.predicted_at, p.prediction_id;
"""

INSERT_BASELINE_SQL = """
INSERT INTO drift_reference_baselines (
  baseline_id,
  baseline_name,
  baseline_type,
  source_table,
  source_start,
  source_end,
  source_filter_json,
  model_name,
  model_version,
  model_alias,
  model_run_id,
  threshold,
  feature_count,
  sample_count,
  status,
  notes,
  created_by,
  created_at
)
VALUES (
  %(baseline_id)s,
  %(baseline_name)s,
  %(baseline_type)s,
  %(source_table)s,
  %(source_start)s,
  %(source_end)s,
  %(source_filter_json)s,
  %(model_name)s,
  %(model_version)s,
  %(model_alias)s,
  %(model_run_id)s,
  %(threshold)s,
  %(feature_count)s,
  %(sample_count)s,
  %(status)s,
  %(notes)s,
  %(created_by)s,
  %(created_at)s
);
"""

INSERT_STAT_SQL = """
INSERT INTO drift_reference_stats (
  baseline_id,
  metric_family,
  metric_name,
  feature_name,
  metric_value,
  sample_count,
  non_null_count,
  null_count,
  metadata_json,
  created_at
)
VALUES (
  %(baseline_id)s,
  %(metric_family)s,
  %(metric_name)s,
  %(feature_name)s,
  %(metric_value)s,
  %(sample_count)s,
  %(non_null_count)s,
  %(null_count)s,
  %(metadata_json)s,
  %(created_at)s
);
"""

RETIRE_EXISTING_ACTIVE_BASELINES_SQL = """
UPDATE drift_reference_baselines
SET status = 'retired'
WHERE baseline_type = 'fixed_reference'
  AND status = 'active'
  AND model_run_id = %s;
"""


def quantile_bin_edges(sorted_values: list[float], requested_bin_count: int) -> list[float]:
    edges: list[float] = []

    for bin_index in range(1, requested_bin_count):
        edge = quantile_from_sorted(sorted_values, bin_index / requested_bin_count)
        if edge is None or not math.isfinite(edge):
            continue
        if not edges or edge > edges[-1]:
            edges.append(float(edge))

    return edges


def constant_baseline_edges(value: float) -> list[float]:
    return [float(math.nextafter(value, -math.inf)), float(value)]


def bucket_counts(values: list[float], bin_edges: list[float]) -> list[int]:
    counts = [0 for _ in range(len(bin_edges) + 1)]

    for value in values:
        bucket = bisect_left(bin_edges, value)
        counts[bucket] += 1

    return counts


def build_feature_distribution_metadata(
        values: list[float],
        requested_bin_count: int,
) -> dict[str, Any] | None:
    if not values:
        return None

    sorted_values = sorted(values)
    min_value = float(sorted_values[0])
    max_value = float(sorted_values[-1])

    if min_value == max_value:
        bin_edges = constant_baseline_edges(min_value)
        binning = "constant_baseline_anchor"
    else:
        bin_edges = quantile_bin_edges(sorted_values, requested_bin_count)
        if not bin_edges:
            bin_edges = [float((min_value + max_value) / 2.0)]
        binning = "baseline_quantile"

    counts = bucket_counts(sorted_values, bin_edges)

    return {
        "distribution": "non_null_feature_values",
        "null_handling": "excluded_from_psi_missing_ratio_tracked_separately",
        "binning": binning,
        "bin_count_requested": requested_bin_count,
        "bin_edges": bin_edges,
        "bin_count": len(counts),
        "baseline_bin_counts": counts,
        "baseline_bin_ratios": ratios_from_counts(counts),
        "baseline_non_null_count": len(sorted_values),
        "baseline_min": min_value,
        "baseline_max": max_value,
    }


def add_stat(
        stats: list[dict[str, Any]],
        baseline_id: str,
        created_at: float,
        metric_family: str,
        metric_name: str,
        feature_name: str | None,
        metric_value: float | None,
        sample_count: int,
        non_null_count: int | None = None,
        null_count: int | None = None,
        metadata: dict[str, Any] | None = None,
) -> None:
    stats.append({
        "baseline_id": baseline_id,
        "metric_family": metric_family,
        "metric_name": metric_name,
        "feature_name": feature_name,
        "metric_value": metric_value,
        "sample_count": sample_count,
        "non_null_count": non_null_count,
        "null_count": null_count,
        "metadata_json": Jsonb(metadata or {}),
        "created_at": created_at,
    })


def load_prediction_rows(
        model_run_id: str,
        threshold: float,
        source_start: float,
        source_end: float,
) -> list[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                FETCH_SQL,
                [model_run_id, threshold, source_start, source_end],
            )
            return list(cursor.fetchall())


def build_baseline_row(
        args: argparse.Namespace,
        rows: list[dict[str, Any]],
        baseline_id: str,
        created_at: float,
) -> dict[str, Any]:
    source_filter = {
        "model_run_id": args.model_run_id,
        "threshold": args.threshold,
        "predicted_at": {
            "gt": args.source_start,
            "lte": args.source_end,
        },
    }

    return {
        "baseline_id": baseline_id,
        "baseline_name": args.baseline_name,
        "baseline_type": "fixed_reference",
        "source_table": "prediction_logs",
        "source_start": args.source_start,
        "source_end": args.source_end,
        "source_filter_json": Jsonb(source_filter),
        "model_name": first_present(rows, "model_name"),
        "model_version": first_present(rows, "model_version"),
        "model_alias": first_present(rows, "model_alias"),
        "model_run_id": args.model_run_id,
        "threshold": args.threshold,
        "feature_count": args.feature_count,
        "sample_count": len(rows),
        "status": args.status,
        "notes": args.notes,
        "created_by": args.created_by,
        "created_at": created_at,
    }


def build_reference_stats(
        rows: list[dict[str, Any]],
        baseline_id: str,
        created_at: float,
        feature_count: int,
        min_feature_non_null: int,
        min_feature_samples: int,
        psi_bin_count: int,
        min_psi_feature_non_null: int,
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    sample_count = len(rows)

    probabilities = [float(row["fail_probability"]) for row in rows]
    missing_counts = [float(row["missing_count"]) for row in rows]
    predicted_fail_count = sum(
        1 for row in rows
        if int(row["predicted_value"]) == POSITIVE_CLASS
    )

    add_stat(stats, baseline_id, created_at, "output", "prediction_count", None,
             float(sample_count), sample_count)
    add_stat(stats, baseline_id, created_at, "output", "predicted_fail_ratio", None,
             ratio(predicted_fail_count, sample_count), sample_count)
    add_stat(stats, baseline_id, created_at, "output", "fail_probability_avg", None,
             mean(probabilities), sample_count, non_null_count=sample_count)
    add_stat(stats, baseline_id, created_at, "output", "fail_probability_p50", None,
             percentile(probabilities, 0.50), sample_count, non_null_count=sample_count)
    add_stat(stats, baseline_id, created_at, "output", "fail_probability_p95", None,
             percentile(probabilities, 0.95), sample_count, non_null_count=sample_count)

    add_stat(stats, baseline_id, created_at, "input", "missing_count_avg", None,
             mean(missing_counts), sample_count, non_null_count=sample_count)
    add_stat(stats, baseline_id, created_at, "input", "missing_count_p95", None,
             percentile(missing_counts, 0.95), sample_count, non_null_count=sample_count)
    add_stat(stats, baseline_id, created_at, "input", "missing_count_max", None,
             max(missing_counts) if missing_counts else None, sample_count, non_null_count=sample_count)

    feature_stats = empty_feature_stats(feature_count)
    update_feature_stats(feature_stats, rows, feature_count)
    feature_values = collect_feature_values(rows, feature_count)

    for index, item in enumerate(feature_stats):
        feature_name = feature_key(index)
        samples = int(item["samples"])
        null_count = int(item["null_count"])
        non_null = int(item["non_null"])

        metadata = {"feature_index": index}

        if samples >= min_feature_samples:
            add_stat(
                stats,
                baseline_id,
                created_at,
                "feature",
                "feature_missing_ratio",
                feature_name,
                ratio(null_count, samples),
                samples,
                non_null_count=non_null,
                null_count=null_count,
                metadata=metadata,
            )

        if non_null >= min_feature_non_null:
            add_stat(
                stats,
                baseline_id,
                created_at,
                "feature",
                "feature_mean",
                feature_name,
                feature_mean(item),
                samples,
                non_null_count=non_null,
                null_count=null_count,
                metadata=metadata,
            )
            add_stat(
                stats,
                baseline_id,
                created_at,
                "feature",
                "feature_std",
                feature_name,
                feature_std(item),
                samples,
                non_null_count=non_null,
                null_count=null_count,
                metadata=metadata,
            )

        if non_null >= min_psi_feature_non_null:
            distribution_metadata = build_feature_distribution_metadata(
                feature_values[index],
                psi_bin_count,
            )
            if distribution_metadata is not None:
                distribution_metadata["feature_index"] = index
                add_stat(
                    stats,
                    baseline_id,
                    created_at,
                    "feature",
                    "feature_distribution_bins",
                    feature_name,
                    None,
                    samples,
                    non_null_count=non_null,
                    null_count=null_count,
                    metadata=distribution_metadata,
                )

    return stats


def save_baseline(
        baseline_row: dict[str, Any],
        stats: list[dict[str, Any]],
        retire_existing_active: bool,
) -> None:
    with connect() as conn:
        with conn.cursor() as cursor:
            if retire_existing_active:
                cursor.execute(
                    RETIRE_EXISTING_ACTIVE_BASELINES_SQL,
                    [baseline_row["model_run_id"]],
                )
            cursor.execute(INSERT_BASELINE_SQL, baseline_row)
            cursor.executemany(INSERT_STAT_SQL, stats)


def print_summary(
        args: argparse.Namespace,
        baseline_row: dict[str, Any],
        stats: list[dict[str, Any]],
) -> None:
    family_counts: dict[str, int] = {}
    for row in stats:
        family = row["metric_family"]
        family_counts[family] = family_counts.get(family, 0) + 1

    print(
        "drift_reference_baseline_built "
        f"dry_run={args.dry_run} "
        f"baseline_id={baseline_row['baseline_id']} "
        f"baseline_name={baseline_row['baseline_name']} "
        f"model_run_id={baseline_row['model_run_id']} "
        f"threshold={baseline_row['threshold']} "
        f"source_start={baseline_row['source_start']} "
        f"source_end={baseline_row['source_end']} "
        f"sample_count={baseline_row['sample_count']} "
        f"stats_rows={len(stats)} "
        f"output_stats={family_counts.get('output', 0)} "
        f"input_stats={family_counts.get('input', 0)} "
        f"feature_stats={family_counts.get('feature', 0)}"
    )

    preview_names = {
        ("output", "predicted_fail_ratio", None),
        ("output", "fail_probability_avg", None),
        ("output", "fail_probability_p95", None),
        ("input", "missing_count_avg", None),
        ("input", "missing_count_p95", None),
    }

    for row in stats:
        key = (row["metric_family"], row["metric_name"], row["feature_name"])
        if key in preview_names:
            print(
                "baseline_stat "
                f"family={row['metric_family']} "
                f"name={row['metric_name']} "
                f"feature={row['feature_name']} "
                f"value={row['metric_value']} "
                f"sample_count={row['sample_count']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-id", default=None)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--source-start", type=non_negative_float, required=True)
    parser.add_argument("--source-end", type=non_negative_float, required=True)
    parser.add_argument("--feature-count", type=positive_int, default=NUM_FEATURES)
    parser.add_argument("--min-samples", type=positive_int, default=500)
    parser.add_argument("--min-feature-non-null", type=positive_int, default=30)
    parser.add_argument("--min-feature-samples", type=positive_int, default=30)
    parser.add_argument("--psi-bin-count", type=positive_int, default=10)
    parser.add_argument("--min-psi-feature-non-null", type=positive_int, default=30)
    parser.add_argument("--status", choices=["active", "retired"], default="active")
    parser.add_argument(
        "--retire-existing-active",
        action="store_true",
        help=(
            "Retire existing active fixed-reference baselines for the same "
            "model_run_id before saving this baseline."
        ),
    )
    parser.add_argument("--notes", default=None)
    parser.add_argument("--created-by", default=os.getenv("USER"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    if args.source_end < args.source_start:
        raise ValueError("source_end must be >= source_start")

    rows = load_prediction_rows(
        model_run_id=args.model_run_id,
        threshold=args.threshold,
        source_start=args.source_start,
        source_end=args.source_end,
    )

    if len(rows) < args.min_samples:
        raise ValueError(
            "not enough prediction rows for baseline: "
            f"required={args.min_samples} actual={len(rows)}"
        )

    baseline_id = args.baseline_id or str(uuid4())
    created_at = time.time()

    baseline_row = build_baseline_row(
        args=args,
        rows=rows,
        baseline_id=baseline_id,
        created_at=created_at,
    )
    stats = build_reference_stats(
        rows=rows,
        baseline_id=baseline_id,
        created_at=created_at,
        feature_count=args.feature_count,
        min_feature_non_null=args.min_feature_non_null,
        min_feature_samples=args.min_feature_samples,
        psi_bin_count=args.psi_bin_count,
        min_psi_feature_non_null=args.min_psi_feature_non_null,
    )

    print_summary(args, baseline_row, stats)

    if args.dry_run:
        print("drift_reference_baseline_write_skipped dry_run=true")
        return

    save_baseline(
        baseline_row,
        stats,
        retire_existing_active=args.retire_existing_active,
    )
    print(
        "drift_reference_baseline_saved "
        f"baseline_id={baseline_id} "
        f"retire_existing_active={args.retire_existing_active} "
        f"stats_rows={len(stats)}"
    )


if __name__ == "__main__":
    main()
