from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

with DAG(
        dag_id="train_candidate_from_offline_point_in_time_features",
        description="Train and register an MLflow candidate from a READY training Dataset.",
        start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
        schedule=None,
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "mlops",
            "retries": 0,
        },
        params={
            "dataset_id": Param(
                type="string",
                minLength=1,
                pattern=r"\S",
                title="Training Dataset ID",
                description=(
                        "Required READY training dataset_builds.dataset_id. The trainer "
                        "loads and verifies its MLflow Parquet artifact."
                ),
            ),
            "min_label_coverage": Param(
                0.95,
                type="number",
                minimum=0,
                maximum=1,
            ),
            "min_fail_samples": Param(20, type="integer", minimum=1),
            "min_pass_samples": Param(20, type="integer", minimum=1),
            "dry_run": Param(False, type="boolean"),
        },
        tags=["ml", "candidate"],
) as dag:
    train_candidate_from_training_dataset = BashOperator(
        task_id="train_candidate_from_training_dataset",
        execution_timeout=timedelta(minutes=30),
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

bash scripts/wrapper/train_candidate_from_offline_point_in_time_features.sh \
  --dataset-id "{{ params.dataset_id }}" \
  --candidate-group "airflow_{{ run_id }}" \
  --training-job-id "airflow_train_{{ run_id }}" \
  --min-label-coverage "{{ params.min_label_coverage }}" \
  --min-fail-samples "{{ params.min_fail_samples }}" \
  --min-pass-samples "{{ params.min_pass_samples }}" \
  --dry-run "{{ params.dry_run }}"
""",
    )
