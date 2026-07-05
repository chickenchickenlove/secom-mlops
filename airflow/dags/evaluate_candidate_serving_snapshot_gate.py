from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

with DAG(
        dag_id="evaluate_candidate_serving_snapshot_gate",
        description="Evaluate the MLflow candidate against champion on a serving snapshot window.",
        start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
        schedule=None,
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "mlops",
            "retries": 0,
        },
        params={
            "point_time": Param(
                None,
                type=["null", "string"],
                format="date-time",
            ),
            "recent_minutes": Param(
                10,
                type="integer",
                minimum=1,
            ),
            "candidate_version": Param(
                None,
                type=["null", "string"],
            ),
            "min_samples": Param(500, type="integer", minimum=1),
            "min_fail_samples": Param(20, type="integer", minimum=1),
            "min_pass_samples": Param(20, type="integer", minimum=1),
            "dry_run": Param(False, type="boolean"),
        },
        tags=["ml", "candidate"],
) as dag:
    resolve_point_time = BashOperator(
        task_id="resolve_point_time",
        execution_timeout=timedelta(minutes=1),
        do_xcom_push=True,
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

bash scripts/wrapper/resolve_candidate_retraining_point_time.sh \
  --point-time "{{ params.point_time }}"
""",
    )

    resolve_point_time_start = BashOperator(
        task_id="resolve_point_time_start",
        execution_timeout=timedelta(minutes=1),
        do_xcom_push=True,
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

bash scripts/wrapper/resolve_candidate_retraining_point_time_start.sh \
  --point-time "{{ ti.xcom_pull(task_ids='resolve_point_time') }}" \
  --recent-minutes "{{ params.recent_minutes }}"
""",
    )

    evaluate_candidate_against_champion = BashOperator(
        task_id="evaluate_candidate_against_champion",
        execution_timeout=timedelta(minutes=10),
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

bash scripts/wrapper/compare_candidate_with_champion_serving.sh \
  --point-time-start "{{ ti.xcom_pull(task_ids='resolve_point_time_start') }}" \
  --point-time "{{ ti.xcom_pull(task_ids='resolve_point_time') }}" \
  --candidate-version "{{ params.candidate_version }}" \
  --min-samples "{{ params.min_samples }}" \
  --min-fail-samples "{{ params.min_fail_samples }}" \
  --min-pass-samples "{{ params.min_pass_samples }}" \
  --dry-run "{{ params.dry_run }}"
""",
    )

    resolve_point_time >> resolve_point_time_start >> evaluate_candidate_against_champion
