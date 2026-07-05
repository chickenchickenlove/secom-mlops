from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

from secom_mlops_common.config.mlflow import (
    DEFAULT_CHAMPION_ALIAS,
    DEFAULT_CONTAINER_MLFLOW_TRACKING_URI,
    DEFAULT_MODEL_NAME,
)
from secom_mlops_common.config.serving import (
    CANARY_METADATA_PATH,
    CANARY_RELOAD_MODEL_VERSION_PATH,
    DEFAULT_CONTAINER_MODEL_GATEWAY_URL,
    build_url,
)

with DAG(
        dag_id="deploy_candidate_to_canary",
        description=(
                "Reload an approved candidate model version into the canary model "
                "server slot and mark the deployment request canary_ready."
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
            "target_alias": Param(DEFAULT_CHAMPION_ALIAS, type="string"),
            "request_id": Param(
                None,
                type=["string"],
                title="Deployment request ID",
                description=(
                        "Optional model_deployment_requests.request_id. Leave empty "
                        "to deploy the latest approved request."
                ),
            ),
            "reload_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_URL, CANARY_RELOAD_MODEL_VERSION_PATH),
                type="string",
            ),
            "metadata_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_URL, CANARY_METADATA_PATH),
                type="string",
            ),
            "http_timeout_seconds": Param(180, type="integer"),
            "dry_run": Param(True, type="boolean"),
        },
        tags=["ml", "serving", "canary", "deployment"],
) as dag:
    deploy_candidate_to_canary = BashOperator(
        task_id="deploy_candidate_to_canary",
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

ARGS=(
  python scripts/deployment/deploy_candidate_to_canary.py
  --tracking-uri "{{ params.tracking_uri }}"
  --model-name "{{ params.model_name }}"
  --target-alias "{{ params.target_alias }}"
  --reload-url "{{ params.reload_url }}"
  --metadata-url "{{ params.metadata_url }}"
  --http-timeout-seconds "{{ params.http_timeout_seconds }}"
)

REQUEST_ID="{{ params.request_id }}"
REQUEST_ID="${REQUEST_ID#request_id=}"
if [ -n "${REQUEST_ID}" ] && [ "${REQUEST_ID}" != "None" ] && [ "${REQUEST_ID}" != "null" ]; then
  ARGS+=(--request-id "${REQUEST_ID}")
fi

case "{{ params.dry_run }}" in
  "1"|"true"|"True"|"TRUE"|"yes"|"Yes"|"YES")
    ARGS+=(--dry-run)
    ;;
esac

echo "deploy_candidate_to_canary_command model_name={{ params.model_name }} target_alias={{ params.target_alias }}"
echo "request_id=${REQUEST_ID:-auto} dry_run={{ params.dry_run }}"

"${ARGS[@]}"
""",
    )
