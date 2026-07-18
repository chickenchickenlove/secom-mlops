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
    DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL,
    DEFAULT_CONTAINER_MODEL_GATEWAY_URL,
    TRAFFIC_POLICY_PATH,
    build_url,
)

with DAG(
        dag_id="rollback_candidate_canary",
        description=(
                "Reset gateway canary traffic to 0%, optionally reload the canary "
                "slot to the previous champion model version, and mark the request rolled_back."
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
                "",
                type="string",
                title="Deployment request ID",
                description=(
                        "Required model_deployment_requests.request_id. Run "
                        "inspect_deployment_requests first and paste the selected request_id."
                ),
            ),
            "traffic_policy_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL, TRAFFIC_POLICY_PATH),
                type="string",
            ),
            "canary_reload_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_URL, CANARY_RELOAD_MODEL_VERSION_PATH),
                type="string",
            ),
            "canary_metadata_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_URL, CANARY_METADATA_PATH),
                type="string",
            ),
            "reload_canary_to_previous": Param(True, type="boolean"),
            "notes": Param("candidate canary rollback requested", type="string"),
            "http_timeout_seconds": Param(180, type="integer"),
            "dry_run": Param(False, type="boolean"),
        },
        tags=["ml", "serving", "canary", "rollback", "deployment"],
) as dag:
    rollback_candidate_canary = BashOperator(
        task_id="rollback_candidate_canary",
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

ARGS=(
  python scripts/deployment/rollback_candidate_canary.py
  --tracking-uri "{{ params.tracking_uri }}"
  --model-name "{{ params.model_name }}"
  --target-alias "{{ params.target_alias }}"
  --traffic-policy-url "{{ params.traffic_policy_url }}"
  --canary-reload-url "{{ params.canary_reload_url }}"
  --canary-metadata-url "{{ params.canary_metadata_url }}"
  --http-timeout-seconds "{{ params.http_timeout_seconds }}"
)

REQUEST_ID="{{ params.request_id }}"
REQUEST_ID="${REQUEST_ID#request_id=}"
if [ -z "${REQUEST_ID//[[:space:]]/}" ] || [ "${REQUEST_ID}" = "None" ] || [ "${REQUEST_ID}" = "null" ]; then
  echo "request_id_required dag_id=rollback_candidate_canary"
  echo "Run inspect_deployment_requests, choose the intended request_id, then rerun this DAG."
  exit 2
fi

ARGS+=(--request-id "${REQUEST_ID}")

case "{{ params.reload_canary_to_previous }}" in
  "1"|"true"|"True"|"TRUE"|"yes"|"Yes"|"YES")
    ARGS+=(--reload-canary-to-previous)
    ;;
esac

NOTES="{{ params.notes }}"
if [ -n "${NOTES}" ]; then
  ARGS+=(--notes "${NOTES}")
fi

case "{{ params.dry_run }}" in
  "1"|"true"|"True"|"TRUE"|"yes"|"Yes"|"YES")
    ARGS+=(--dry-run)
    ;;
esac

echo "rollback_candidate_canary_command model_name={{ params.model_name }} target_alias={{ params.target_alias }}"
echo "request_id=${REQUEST_ID} reload_canary_to_previous={{ params.reload_canary_to_previous }} dry_run={{ params.dry_run }}"

"${ARGS[@]}"
""",
    )
