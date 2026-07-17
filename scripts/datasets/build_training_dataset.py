"""Airflow-internal entry point for a scheduled point-in-time dataset build.

Direct or concurrent invocation is outside the supported single-writer execution
model. Airflow serializes calls through the dataset DAG's ``max_active_runs=1``.
"""

from __future__ import annotations

import argparse
import json

from secom_mlops.datasets.training_dataset import (
    DEFAULT_LABEL_MATURITY_SECONDS,
    DEFAULT_MIN_FAIL_SAMPLES,
    DEFAULT_MIN_LABELED_SAMPLES,
    DEFAULT_MIN_LABEL_COVERAGE,
    DEFAULT_MIN_PASS_SAMPLES,
    DatasetBuildConfig,
)
from secom_mlops.datasets.training_dataset_builder import (
    DEFAULT_EXPERIMENT_NAME,
    DatasetBuildSkipped,
    build_training_dataset,
)
from secom_mlops_common.cli.validators import non_negative_float, positive_int
from secom_mlops_common.config.mlflow import (
    DEFAULT_CONTAINER_MLFLOW_TRACKING_URI,
    resolve_tracking_uri,
)

SKIP_EXIT_CODE = 99


def coverage_float(raw_value: str) -> float:
    value = float(raw_value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def optional_text(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if value in {"", "None", "none", "null", "NULL"}:
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and persist an immutable point-in-time training-source dataset "
            "when readiness requirements are met."
        )
    )
    parser.add_argument("--cohort-start-time", type=non_negative_float, required=True)
    parser.add_argument("--cutoff-time", type=non_negative_float, required=True)
    parser.add_argument(
        "--label-maturity-seconds",
        type=non_negative_float,
        default=DEFAULT_LABEL_MATURITY_SECONDS,
    )
    parser.add_argument(
        "--min-labeled-samples",
        type=positive_int,
        default=DEFAULT_MIN_LABELED_SAMPLES,
    )
    parser.add_argument(
        "--min-label-coverage",
        type=coverage_float,
        default=DEFAULT_MIN_LABEL_COVERAGE,
    )
    parser.add_argument(
        "--min-fail-samples",
        type=positive_int,
        default=DEFAULT_MIN_FAIL_SAMPLES,
    )
    parser.add_argument(
        "--min-pass-samples",
        type=positive_int,
        default=DEFAULT_MIN_PASS_SAMPLES,
    )
    parser.add_argument("--simulation-run-id", default=None)
    parser.add_argument("--drift-segment", default=None)
    parser.add_argument(
        "--tracking-uri",
        default=resolve_tracking_uri(default=DEFAULT_CONTAINER_MLFLOW_TRACKING_URI),
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DatasetBuildConfig(
        cohort_start_time=args.cohort_start_time,
        cutoff_time=args.cutoff_time,
        label_maturity_seconds=args.label_maturity_seconds,
        min_labeled_samples=args.min_labeled_samples,
        min_label_coverage=args.min_label_coverage,
        min_fail_samples=args.min_fail_samples,
        min_pass_samples=args.min_pass_samples,
        simulation_run_id=optional_text(args.simulation_run_id),
        drift_segment=optional_text(args.drift_segment),
    )

    try:
        result = build_training_dataset(
            config,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
        )
    except DatasetBuildSkipped as exc:
        print(json.dumps({"status": "SKIPPED", "reason": str(exc)}, sort_keys=True))
        return SKIP_EXIT_CODE

    print(json.dumps({
        "status": "READY",
        "dataset_id": result.dataset_id,
        "manifest_hash": result.manifest_hash,
        "artifact_sha256": result.artifact_sha256,
        "mlflow_run_id": result.mlflow_run_id,
        "artifact_uri": result.artifact_uri,
        "stats": result.stats,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
