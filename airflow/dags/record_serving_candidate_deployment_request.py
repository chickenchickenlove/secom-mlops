from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

from secom_mlops_common.config.mlflow import (
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_CHAMPION_ALIAS,
    DEFAULT_CONTAINER_MLFLOW_TRACKING_URI,
    DEFAULT_MODEL_NAME,
)

with DAG(
        dag_id="record_serving_candidate_deployment_request",
        description=(
                "Create a deployment request from a completed, passed serving-gate "
                "MLflow evaluation run."
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
            "candidate_alias": Param(DEFAULT_CANDIDATE_ALIAS, type="string"),
            "candidate_version": Param(
                None,
                type=["null", "string"],
                title="Candidate version",
                description=(
                        "Optional concrete model version. MLflow UI style v2 and API "
                        "style 2 are both accepted. Leave empty to use candidate_alias."
                ),
            ),
            "champion_alias": Param(DEFAULT_CHAMPION_ALIAS, type="string"),
            "evaluation_run_id": Param(
                "",
                type="string",
                title="Serving Gate evaluation run ID",
                description=(
                        "Required MLflow run ID emitted by "
                        "evaluate_candidate_serving_snapshot_gate."
                ),
            ),
            "approval_status": Param(
                "approved",
                enum=["pending", "approved", "rejected"],
                type="string",
                title="Approval status",
            ),
            "notes": Param(
                "serving snapshot eval passed; deployment request created",
                type="string",
                title="Notes",
            ),
            "requested_by": Param("airflow", type="string"),
            "approved_by": Param("", type="string"),
            "dry_run": Param(False, type="boolean"),
        },
        tags=["ml", "serving", "candidate", "deployment"],
) as dag:
    record_serving_candidate_deployment_request = BashOperator(
        task_id="record_serving_candidate_deployment_request",
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

ARGS=(
  python scripts/deployment/record_serving_candidate_deployment_request.py
  --tracking-uri "{{ params.tracking_uri }}"
  --model-name "{{ params.model_name }}"
  --candidate-alias "{{ params.candidate_alias }}"
  --champion-alias "{{ params.champion_alias }}"
  --approval-status "{{ params.approval_status }}"
  --notes "{{ params.notes }}"
  --requested-by "{{ params.requested_by }}"
)

EVALUATION_RUN_ID="{{ params.evaluation_run_id }}"
EVALUATION_RUN_ID="${EVALUATION_RUN_ID#evaluation_run_id=}"
if [ -z "${EVALUATION_RUN_ID//[[:space:]]/}" ] || [ "${EVALUATION_RUN_ID}" = "None" ] || [ "${EVALUATION_RUN_ID}" = "null" ]; then
  echo "evaluation_run_id_required dag_id=record_serving_candidate_deployment_request"
  exit 2
fi
ARGS+=(--evaluation-run-id "${EVALUATION_RUN_ID}")

CANDIDATE_VERSION="{{ params.candidate_version }}"
if [ -n "${CANDIDATE_VERSION}" ] && [ "${CANDIDATE_VERSION}" != "None" ] && [ "${CANDIDATE_VERSION}" != "null" ]; then
  ARGS+=(--candidate-version "${CANDIDATE_VERSION}")
fi

APPROVED_BY="{{ params.approved_by }}"
if [ -n "${APPROVED_BY}" ]; then
  ARGS+=(--approved-by "${APPROVED_BY}")
fi

case "{{ params.dry_run }}" in
  "1"|"true"|"True"|"TRUE"|"yes"|"Yes"|"YES")
    ARGS+=(--dry-run)
    ;;
esac

echo "record_serving_candidate_deployment_request_command evaluation_run_id=${EVALUATION_RUN_ID} model_name={{ params.model_name }} candidate_alias={{ params.candidate_alias }} approval_status={{ params.approval_status }} dry_run={{ params.dry_run }}"

"${ARGS[@]}"
""",
    )
