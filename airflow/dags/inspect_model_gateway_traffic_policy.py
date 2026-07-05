from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

from secom_mlops_common.config.serving import DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL

with DAG(
        dag_id="inspect_model_gateway_traffic_policy",
        description="Inspect the active model gateway release/canary traffic policy.",
        start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
        schedule=None,
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "mlops",
            "retries": 0,
        },
        tags=["ml", "serving", "gateway", "canary", "inspect"],
) as dag:
    inspect_traffic_policy = BashOperator(
        task_id="inspect_traffic_policy",
        bash_command=r"""
set -euo pipefail

MODEL_GATEWAY_ADMIN_URL="${MODEL_GATEWAY_ADMIN_URL:-__DEFAULT_MODEL_GATEWAY_ADMIN_URL__}"

echo "model_gateway_traffic_policy_inspect"
echo "admin_url=${MODEL_GATEWAY_ADMIN_URL}"

export MODEL_GATEWAY_ADMIN_URL

python -c "import json, os, httpx
url = os.environ['MODEL_GATEWAY_ADMIN_URL'].rstrip('/') + '/admin/traffic-policy'
response = httpx.get(url)
try:
    print(json.dumps(response.json(), sort_keys=True))
except ValueError:
    print(response.text)
response.raise_for_status()
"

echo "model_gateway_traffic_policy_inspect_complete admin_url=${MODEL_GATEWAY_ADMIN_URL}"
""".replace("__DEFAULT_MODEL_GATEWAY_ADMIN_URL__", DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL),
    )
