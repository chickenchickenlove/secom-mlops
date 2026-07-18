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
        dag_id="rollback_release_deployment",
        description=(
                "Rollback a deployed release to the previous runtime deployment "
                "state, move the champion alias back, reset canary traffic, and "
                "mark the active request rolled_back."
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
            "notes": Param("post-release rollback requested", type="string"),
            "http_timeout_seconds": Param(180, type="integer"),
            "dry_run": Param(False, type="boolean"),
        },
        tags=["ml", "serving", "release", "rollback", "deployment"],
) as dag:
    rollback_release_deployment = BashOperator(
        task_id="rollback_release_deployment",
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

ARGS=(
  python scripts/deployment/rollback_release_deployment.py
  --tracking-uri "{{ params.tracking_uri }}"
  --model-name "{{ params.model_name }}"
  --target-alias "{{ params.target_alias }}"
  --release-reload-url "{{ params.release_reload_url }}"
  --release-metadata-url "{{ params.release_metadata_url }}"
  --traffic-policy-url "{{ params.traffic_policy_url }}"
  --http-timeout-seconds "{{ params.http_timeout_seconds }}"
)

case "{{ params.reset_canary_traffic }}" in
  "0"|"false"|"False"|"FALSE"|"no"|"No"|"NO")
    ARGS+=(--skip-canary-traffic-reset)
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

echo "rollback_release_deployment_command model_name={{ params.model_name }} target_alias={{ params.target_alias }}"
echo "rollback_source=release_runtime_state reset_canary_traffic={{ params.reset_canary_traffic }} dry_run={{ params.dry_run }}"

"${ARGS[@]}"
""",
    )
