from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator


with DAG(
        dag_id="build_periodic_training_dataset",
        description=(
            "Check training dataset readiness every five minutes and persist an "
            "immutable dataset for the maturity-shifted Airflow data interval."
        ),
        start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
        schedule="*/5 * * * *",
        catchup=False,
        # Single-writer contract: Airflow is the only supported caller of the
        # dataset builder, so DAG runs must not overlap.
        max_active_runs=1,
        default_args={
            "owner": "mlops",
            "retries": 1,
            "retry_delay": timedelta(seconds=30),
        },
        params={
            "label_maturity_seconds": Param(120, type="number", minimum=0),
            "min_labeled_samples": Param(1000, type="integer", minimum=1),
            "min_label_coverage": Param(
                0.95,
                type="number",
                minimum=0,
                maximum=1,
            ),
            "min_fail_samples": Param(20, type="integer", minimum=1),
            "min_pass_samples": Param(20, type="integer", minimum=1),
        },
        tags=["ml", "dataset", "training"],
) as dag:
    build_training_dataset_if_ready = BashOperator(
        task_id="build_training_dataset_if_ready",
        execution_timeout=timedelta(minutes=30),
        skip_on_exit_code=99,
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

python scripts/datasets/build_training_dataset.py \
  --cohort-start-time "{{ data_interval_start.int_timestamp - params.label_maturity_seconds }}" \
  --cutoff-time "{{ data_interval_end.int_timestamp }}" \
  --label-maturity-seconds "{{ params.label_maturity_seconds }}" \
  --min-labeled-samples "{{ params.min_labeled_samples }}" \
  --min-label-coverage "{{ params.min_label_coverage }}" \
  --min-fail-samples "{{ params.min_fail_samples }}" \
  --min-pass-samples "{{ params.min_pass_samples }}"
""",
    )
