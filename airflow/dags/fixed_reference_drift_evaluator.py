from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
        dag_id="fixed_reference_drift_evaluator",
        description="Evaluate latest prediction window against the active fixed drift reference baseline.",
        start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
        schedule="*/3 * * * *",
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "mlops",
            "retries": 1,
            "retry_delay": timedelta(seconds=30),
        },
        tags=["monitoring", "drift", "fixed-reference"],
) as dag:
    resolve_champion_run_id = BashOperator(
        task_id="resolve_champion_run_id",
        execution_timeout=timedelta(minutes=1),
        skip_on_exit_code=99,
        do_xcom_push=True,
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

python scripts/utility/resolve_mlflow_champion_run_id.py
""",
    )

    resolve_reference_baseline_id = BashOperator(
        task_id="resolve_reference_baseline_id",
        execution_timeout=timedelta(minutes=1),
        skip_on_exit_code=99,
        do_xcom_push=True,
        bash_command=r"""
set -euo pipefail

CHAMPION_RUN_ID="{{ ti.xcom_pull(task_ids='resolve_champion_run_id') }}"

if [ -z "${CHAMPION_RUN_ID}" ] || [ "${CHAMPION_RUN_ID}" = "None" ]; then
  echo "champion run id was not resolved"
  exit 1
fi

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

python scripts/utility/resolve_active_fixed_reference_baseline.py \
  --model-run-id "${CHAMPION_RUN_ID}"
""",
    )

    evaluate_fixed_reference_drift = BashOperator(
        task_id="evaluate_fixed_reference_drift",
        execution_timeout=timedelta(minutes=5),
        bash_command=r"""
set -euo pipefail

REFERENCE_BASELINE_ID="{{ ti.xcom_pull(task_ids='resolve_reference_baseline_id') }}"

if [ -z "${REFERENCE_BASELINE_ID}" ] || [ "${REFERENCE_BASELINE_ID}" = "None" ]; then
  echo "reference baseline id was not resolved"
  exit 1
fi

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

python scripts/monitoring/evaluate_fixed_reference_drift_metrics.py \
  --reference-baseline-id "${REFERENCE_BASELINE_ID}" \
  --window-minutes 3 \
  --top-n-features 30 \
  --min-feature-non-null 30 \
  --min-feature-samples 30
""",
    )

    resolve_champion_run_id >> resolve_reference_baseline_id >> evaluate_fixed_reference_drift
