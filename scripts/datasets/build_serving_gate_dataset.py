"""Airflow-internal entry point for on-demand serving-gate materialization.

Direct or concurrent invocation is outside the supported single-writer execution
model. The Gate DAG serializes calls through ``max_active_runs=1``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from secom_mlops.datasets.serving_gate_dataset import (
    DEFAULT_LABEL_MATURITY_SECONDS,
    DEFAULT_MIN_DECISIONS,
    DEFAULT_MIN_FAIL_SAMPLES,
    DEFAULT_MIN_LABELED_DECISIONS,
    DEFAULT_MIN_LABEL_COVERAGE,
    DEFAULT_MIN_PASS_SAMPLES,
    ServingGateDatasetConfig,
)
from secom_mlops.datasets.serving_gate_dataset_builder import (
    DEFAULT_EXPERIMENT_NAME,
    build_serving_gate_dataset,
)
from secom_mlops_common.cli.validators import non_negative_float, positive_int
from secom_mlops_common.config.mlflow import (
    DEFAULT_CONTAINER_MLFLOW_TRACKING_URI,
    resolve_tracking_uri,
)


def epoch_or_iso_datetime(raw_value: str) -> float:
    value = raw_value.strip()
    try:
        return non_negative_float(value)
    except (argparse.ArgumentTypeError, ValueError):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "must be epoch seconds or an ISO datetime"
            ) from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=ZoneInfo(os.getenv("AIRFLOW_INPUT_TIMEZONE", "Asia/Seoul"))
            )
        return parsed.timestamp()


def coverage_float(raw_value: str) -> float:
    value = float(raw_value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize an immutable release-decision dataset before the "
            "serving gate evaluates models."
        )
    )
    parser.add_argument("--cohort-start-time", type=epoch_or_iso_datetime, required=True)
    parser.add_argument("--cutoff-time", type=epoch_or_iso_datetime, required=True)
    parser.add_argument(
        "--label-maturity-seconds",
        type=non_negative_float,
        default=DEFAULT_LABEL_MATURITY_SECONDS,
    )
    parser.add_argument(
        "--min-decisions",
        type=positive_int,
        default=DEFAULT_MIN_DECISIONS,
    )
    parser.add_argument(
        "--min-labeled-decisions",
        type=positive_int,
        default=DEFAULT_MIN_LABELED_DECISIONS,
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
    parser.add_argument(
        "--tracking-uri",
        default=resolve_tracking_uri(default=DEFAULT_CONTAINER_MLFLOW_TRACKING_URI),
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_serving_gate_dataset(
        ServingGateDatasetConfig(
            cohort_start_time=args.cohort_start_time,
            cutoff_time=args.cutoff_time,
            label_maturity_seconds=args.label_maturity_seconds,
            min_decisions=args.min_decisions,
            min_labeled_decisions=args.min_labeled_decisions,
            min_label_coverage=args.min_label_coverage,
            min_fail_samples=args.min_fail_samples,
            min_pass_samples=args.min_pass_samples,
        ),
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    print(json.dumps({
        "status": "READY",
        "dataset_id": result.dataset_id,
        "manifest_hash": result.manifest_hash,
        "artifact_sha256": result.artifact_sha256,
        "mlflow_run_id": result.mlflow_run_id,
        "artifact_uri": result.artifact_uri,
        "stats": result.stats,
        "reused": result.reused,
    }, sort_keys=True), file=sys.stderr)
    # BashOperator pushes the last stdout line to XCom. Keep stdout dataset-id only.
    print(result.dataset_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
