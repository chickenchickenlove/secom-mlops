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
    DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL,
    DEFAULT_CONTAINER_MODEL_GATEWAY_URL,
    RELEASE_METADATA_PATH,
    RELEASE_RELOAD_MODEL_VERSION_PATH,
    TRAFFIC_POLICY_PATH,
    build_url,
)

with DAG(
        dag_id="promote_candidate_to_release",
        description=(
                "Reload a canary-ready candidate into the release model server slot, "
                "move the champion alias, reset canary traffic, and mark it deployed."
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
            "release_reload_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_URL, RELEASE_RELOAD_MODEL_VERSION_PATH),
                type="string",
            ),
            "release_metadata_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_URL, RELEASE_METADATA_PATH),
                type="string",
            ),
            "traffic_policy_url": Param(
                build_url(DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL, TRAFFIC_POLICY_PATH),
                type="string",
            ),
            "reset_canary_traffic": Param(True, type="boolean"),
            "keep_source_alias": Param(False, type="boolean"),
            "http_timeout_seconds": Param(180, type="integer"),
            "dry_run": Param(False, type="boolean"),
        },
        tags=["ml", "serving", "release", "deployment"],
) as dag:
    promote_candidate_to_release = BashOperator(
        task_id="promote_candidate_to_release",
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

ARGS=(
  python scripts/deployment/promote_candidate_to_release.py
  --tracking-uri "{{ params.tracking_uri }}"
  --model-name "{{ params.model_name }}"
  --target-alias "{{ params.target_alias }}"
  --release-reload-url "{{ params.release_reload_url }}"
  --release-metadata-url "{{ params.release_metadata_url }}"
  --traffic-policy-url "{{ params.traffic_policy_url }}"
  --http-timeout-seconds "{{ params.http_timeout_seconds }}"
)

REQUEST_ID="{{ params.request_id }}"
REQUEST_ID="${REQUEST_ID#request_id=}"
if [ -z "${REQUEST_ID//[[:space:]]/}" ] || [ "${REQUEST_ID}" = "None" ] || [ "${REQUEST_ID}" = "null" ]; then
  echo "request_id_required dag_id=promote_candidate_to_release"
  echo "Run inspect_deployment_requests, choose the intended request_id, then rerun this DAG."
  exit 2
fi

ARGS+=(--request-id "${REQUEST_ID}")

case "{{ params.reset_canary_traffic }}" in
  "0"|"false"|"False"|"FALSE"|"no"|"No"|"NO")
    ARGS+=(--skip-canary-traffic-reset)
    ;;
esac

case "{{ params.keep_source_alias }}" in
  "1"|"true"|"True"|"TRUE"|"yes"|"Yes"|"YES")
    ARGS+=(--keep-source-alias)
    ;;
esac

case "{{ params.dry_run }}" in
  "1"|"true"|"True"|"TRUE"|"yes"|"Yes"|"YES")
    ARGS+=(--dry-run)
    ;;
esac

echo "promote_candidate_to_release_command model_name={{ params.model_name }} target_alias={{ params.target_alias }}"
echo "request_id=${REQUEST_ID} reset_canary_traffic={{ params.reset_canary_traffic }} keep_source_alias={{ params.keep_source_alias }} dry_run={{ params.dry_run }}"

"${ARGS[@]}"
""",
    )
