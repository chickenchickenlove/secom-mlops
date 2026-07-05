from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

from secom_mlops_common.config.serving import DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL

with DAG(
        dag_id="set_model_gateway_canary_traffic",
        description=(
                "Switch the model gateway production upstream policy to a prepared "
                "release/canary traffic split."
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
            "canary_percent": Param(
                0,
                enum=[0, 1, 5, 10, 50, 100],
                type="integer",
            ),
            "dry_run": Param(True, type="boolean"),
        },
        tags=["ml", "serving", "gateway", "canary"],
) as dag:
    set_canary_traffic_policy = BashOperator(
        task_id="set_canary_traffic_policy",
        bash_command=r"""
set -euo pipefail

CANARY_PERCENT="{{ params.canary_percent }}"
DRY_RUN="{{ params.dry_run }}"

case "${CANARY_PERCENT}" in
  "0"|"1"|"5"|"10"|"50"|"100")
    ;;
  *)
    echo "invalid_canary_percent value=${CANARY_PERCENT}"
    exit 2
    ;;
esac

TRAFFIC_POLICY_REQUEST_ID="airflow_{{ run_id }}_${CANARY_PERCENT}"
MODEL_GATEWAY_ADMIN_URL="${MODEL_GATEWAY_ADMIN_URL:-__DEFAULT_MODEL_GATEWAY_ADMIN_URL__}"

echo "model_gateway_canary_traffic_policy"
echo "canary_percent=${CANARY_PERCENT}"
echo "dry_run=${DRY_RUN}"
echo "admin_url=${MODEL_GATEWAY_ADMIN_URL}"

export MODEL_GATEWAY_ADMIN_URL
export TRAFFIC_POLICY_REQUEST_ID
export CANARY_PERCENT
export DRY_RUN

python -c "import json, os, httpx
dry_run = os.environ['DRY_RUN'] in {'1', 'true', 'True', 'TRUE', 'yes', 'Yes', 'YES'}
url = os.environ['MODEL_GATEWAY_ADMIN_URL'].rstrip('/') + '/admin/traffic-policy'
payload = {
    'request_id': os.environ['TRAFFIC_POLICY_REQUEST_ID'],
    'canary_percent': int(os.environ['CANARY_PERCENT']),
    'dry_run': dry_run,
}

def print_response(label, response):
    print(label)
    try:
        print(json.dumps(response.json(), sort_keys=True))
    except ValueError:
        print(response.text)

before_response = httpx.get(url)
print_response('model_gateway_traffic_policy_before', before_response)
before_response.raise_for_status()

apply_response = httpx.post(url, json=payload)
print_response('model_gateway_traffic_policy_apply_response', apply_response)
apply_response.raise_for_status()

after_response = httpx.get(url)
print_response('model_gateway_traffic_policy_after', after_response)
after_response.raise_for_status()
"

echo "model_gateway_canary_traffic_policy_confirmed request_id=${TRAFFIC_POLICY_REQUEST_ID} admin_url=${MODEL_GATEWAY_ADMIN_URL}"
""".replace("__DEFAULT_MODEL_GATEWAY_ADMIN_URL__", DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL),
    )
