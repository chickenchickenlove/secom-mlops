from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

with DAG(
        dag_id="evaluate_candidate_serving_snapshot_gate",
        description=(
                "Persist a fixed release-decision dataset, then evaluate the MLflow "
                "candidate against champion from that artifact."
        ),
        start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
        schedule=None,
        catchup=False,
        # Airflow is the only supported dataset writer. Serializing DAG runs keeps
        # materialization and its immediate evaluation within one Gate execution.
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
                120,
                type="number",
                minimum=0,
            ),
            "candidate_version": Param(
                None,
                type=["null", "string"],
            ),
            "min_decisions": Param(1000, type="integer", minimum=1000),
            "min_labeled_decisions": Param(1000, type="integer", minimum=1000),
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
    materialize_serving_gate_dataset = BashOperator(
        task_id="materialize_serving_gate_dataset",
        execution_timeout=timedelta(minutes=30),
        do_xcom_push=True,
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

# Dry-run does not create a catalog row or MLflow artifact. The downstream
# wrapper exits before consuming this sentinel.
case "{{ params.dry_run }}" in
  1|true|True|TRUE|yes|Yes|YES) printf '%s\n' '__DRY_RUN__'; exit 0 ;;
esac

# stdout is intentionally dataset-id only so BashOperator can pass exactly one
# internal value through XCom to the evaluator.
python scripts/datasets/build_serving_gate_dataset.py \
  --cohort-start-time "{{ params.cohort_start_time }}" \
  --cutoff-time "{{ params.cutoff_time }}" \
  --label-maturity-seconds "{{ params.label_maturity_seconds }}" \
  --min-decisions "{{ params.min_decisions }}" \
  --min-labeled-decisions "{{ params.min_labeled_decisions }}" \
  --min-label-coverage "{{ params.min_label_coverage }}" \
  --min-fail-samples "{{ params.min_fail_samples }}" \
  --min-pass-samples "{{ params.min_pass_samples }}"
""",
    )

    evaluate_candidate_against_champion = BashOperator(
        task_id="evaluate_candidate_against_champion",
        execution_timeout=timedelta(minutes=10),
        do_xcom_push=True,
        bash_command=r"""
set -euo pipefail

cd "${ML_PROJECT_DIR:-/opt/airflow/mlops}"

bash scripts/wrapper/compare_candidate_with_champion_serving.sh \
  --dataset-id "{{ ti.xcom_pull(task_ids='materialize_serving_gate_dataset') }}" \
  --candidate-version "{{ params.candidate_version }}" \
  --fail-on-gate-failure true \
  --dry-run "{{ params.dry_run }}"
""",
    )

    materialize_serving_gate_dataset >> evaluate_candidate_against_champion
