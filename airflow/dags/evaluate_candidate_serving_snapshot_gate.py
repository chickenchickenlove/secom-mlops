from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

with DAG(
        dag_id="evaluate_candidate_serving_snapshot_gate",
        description=(
                "Evaluate the MLflow candidate against champion on a fixed serving "
                "prediction decision cohort."
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
            "cohort_start_time": Param(
                None,
                type=["null", "string"],
                format="date-time",
            ),
            "cutoff_time": Param(
                None,
                type=["null", "string"],
                format="date-time",
            ),
            "label_maturity_seconds": Param(
                60,
                type="number",
                minimum=0,
            ),
            "candidate_version": Param(
                None,
                type=["string"],
            ),
            "max_decisions": Param(1000, type="integer", minimum=1),
            "min_decisions": Param(500, type="integer", minimum=1),
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
        tags=["ml", "candidate", "serving", "gate"],
) as dag:
    evaluate_candidate_against_champion = BashOperator(
        task_id="evaluate_candidate_against_champion",
        execution_timeout=timedelta(minutes=10),
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

bash scripts/wrapper/compare_candidate_with_champion_serving.sh \
  --cohort-start-time "{{ params.cohort_start_time }}" \
  --cutoff-time "{{ params.cutoff_time }}" \
  --label-maturity-seconds "{{ params.label_maturity_seconds }}" \
  --candidate-version "{{ params.candidate_version }}" \
  --max-decisions "{{ params.max_decisions }}" \
  --min-decisions "{{ params.min_decisions }}" \
  --min-label-coverage "{{ params.min_label_coverage }}" \
  --min-fail-samples "{{ params.min_fail_samples }}" \
  --min-pass-samples "{{ params.min_pass_samples }}" \
  --fail-on-gate-failure true \
  --dry-run "{{ params.dry_run }}"
""",
    )
