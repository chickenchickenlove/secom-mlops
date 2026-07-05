from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

from secom_mlops_common.config.mlflow import DEFAULT_CHAMPION_ALIAS, DEFAULT_MODEL_NAME
from secom_mlops_common.config.serving import (
    CANARY_METADATA_PATH,
    DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL,
    DEFAULT_CONTAINER_MODEL_GATEWAY_URL,
    PRODUCTION_METADATA_PATH,
    RELEASE_METADATA_PATH,
    TRAFFIC_POLICY_PATH,
    build_url,
)

with DAG(
        dag_id="inspect_deployment_requests",
        description=(
                "Inspect recent model deployment requests, reload events, and "
                "optional runtime metadata before choosing a request_id."
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
            "model_name": Param(DEFAULT_MODEL_NAME, type="string"),
            "target_alias": Param(DEFAULT_CHAMPION_ALIAS, type="string"),
            "limit": Param(20, type="integer", minimum=1),
            "reload_event_limit": Param(20, type="integer", minimum=1),
            "include_runtime_metadata": Param(True, type="boolean"),
            "release_metadata_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_URL, RELEASE_METADATA_PATH),
                type="string",
            ),
            "canary_metadata_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_URL, CANARY_METADATA_PATH),
                type="string",
            ),
            "production_metadata_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_URL, PRODUCTION_METADATA_PATH),
                type="string",
            ),
            "traffic_policy_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL, TRAFFIC_POLICY_PATH),
                type="string",
            ),
            "http_timeout_seconds": Param(30, type="integer", minimum=1),
        },
        tags=["ml", "serving", "deployment", "inspect"],
) as dag:
    inspect_deployment_requests = BashOperator(
        task_id="inspect_deployment_requests",
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

ARGS=(
  python scripts/utility/inspect_deployment_requests.py
  --model-name "{{ params.model_name }}"
  --target-alias "{{ params.target_alias }}"
  --limit "{{ params.limit }}"
  --reload-event-limit "{{ params.reload_event_limit }}"
  --release-metadata-url "{{ params.release_metadata_url }}"
  --canary-metadata-url "{{ params.canary_metadata_url }}"
  --production-metadata-url "{{ params.production_metadata_url }}"
  --traffic-policy-url "{{ params.traffic_policy_url }}"
  --http-timeout-seconds "{{ params.http_timeout_seconds }}"
)

case "{{ params.include_runtime_metadata }}" in
  "1"|"true"|"True"|"TRUE"|"yes"|"Yes"|"YES")
    ARGS+=(--include-runtime-metadata)
    ;;
esac

echo "inspect_deployment_requests_command model_name={{ params.model_name }} target_alias={{ params.target_alias }} limit={{ params.limit }}"

"${ARGS[@]}"
""",
    )
