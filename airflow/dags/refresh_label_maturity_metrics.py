from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
        dag_id="refresh_label_maturity_metrics",
        description="Refresh fixed-cohort label maturity metrics for Grafana.",
        start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
        schedule="*/1 * * * *",
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "mlops",
            "retries": 0,
        },
        tags=["monitoring", "labels", "maturity"],
) as dag:
    refresh_label_maturity_metrics = BashOperator(
        task_id="refresh_label_maturity_metrics",
        execution_timeout=timedelta(minutes=1),
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

python scripts/monitoring/refresh_label_maturity_metrics.py
""",
    )
