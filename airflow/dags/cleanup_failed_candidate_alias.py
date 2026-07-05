from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

from secom_mlops_common.config.mlflow import (
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_CONTAINER_MLFLOW_TRACKING_URI,
    DEFAULT_MODEL_NAME,
)

with DAG(
        dag_id="cleanup_failed_candidate_alias",
        description=(
                "Clear the MLflow candidate alias when the candidate serving snapshot "
                "evaluation rejected the model version."
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
            "tracking_uri": Param(DEFAULT_CONTAINER_MLFLOW_TRACKING_URI, type="string"),
            "model_name": Param(DEFAULT_MODEL_NAME, type="string"),
            "alias": Param(DEFAULT_CANDIDATE_ALIAS, type="string"),
            "cleanup_policy": Param(
                "serving_snapshot_eval_rejected",
                enum=["serving_snapshot_eval_rejected"],
                type="string",
            ),
            "dry_run": Param(True, type="boolean"),
        },
        tags=["ml", "candidate", "cleanup", "mlflow"],
) as dag:
    cleanup_failed_candidate_alias = BashOperator(
        task_id="cleanup_failed_candidate_alias",
        execution_timeout=timedelta(minutes=2),
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

ARGS=(
  python scripts/deployment/clear_model_alias.py
  --tracking-uri "{{ params.tracking_uri }}"
  --model-name "{{ params.model_name }}"
  --alias "{{ params.alias }}"
  --cleanup-policy "{{ params.cleanup_policy }}"
)

case "{{ params.dry_run }}" in
  "1"|"true"|"True"|"TRUE"|"yes"|"Yes"|"YES")
    ARGS+=(--dry-run)
    ;;
esac

echo "candidate_alias_cleanup_command model_name={{ params.model_name }} alias={{ params.alias }}
cleanup_policy={{ params.cleanup_policy }} dry_run={{ params.dry_run }}"

"${ARGS[@]}"
""",
    )
