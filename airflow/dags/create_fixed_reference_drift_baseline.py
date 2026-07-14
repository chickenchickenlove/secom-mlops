from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

with DAG(
        dag_id="create_fixed_reference_drift_baseline",
        description=(
                "Create an active fixed-reference drift baseline for the current "
                "MLflow champion model from a selected prediction log window."
        ),
        start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
        schedule=None,
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "mlops",
            "retries": 0,
        },
        params={
            "baseline_start": Param(
                None,
                type=["null", "string"],
                format="date-time",
            ),
            "baseline_end": Param(
                None,
                type=["null", "string"],
                format="date-time",
            ),
            "baseline_name": Param(
                "fixed-reference-champion-manual",
                type="string",
            ),
            "min_samples": Param(500, type="integer", minimum=1),
            "retire_existing_active": Param(True, type="boolean"),
            "dry_run": Param(False, type="boolean"),
        },
        tags=["monitoring", "drift", "fixed-reference"],
) as dag:
    create_fixed_reference_drift_baseline = BashOperator(
        task_id="create_fixed_reference_drift_baseline",
        execution_timeout=timedelta(minutes=5),
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

bash scripts/wrapper/create_fixed_reference_drift_baseline.sh \
  --baseline-start "{{ params.baseline_start }}" \
  --baseline-end "{{ params.baseline_end }}" \
  --baseline-name "{{ params.baseline_name }}" \
  --min-samples "{{ params.min_samples }}" \
  --retire-existing-active "{{ params.retire_existing_active }}" \
  --dry-run "{{ params.dry_run }}"
""",
    )
