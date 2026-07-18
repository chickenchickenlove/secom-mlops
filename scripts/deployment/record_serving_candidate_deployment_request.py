from __future__ import annotations

import argparse
import os
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from secom_mlops.monitor.deployments import (
    SERVING_GATE_EVAL_TYPE,
    build_deployment_request_row,
    insert_deployment_request,
)
from secom_mlops.monitor.serving_gate_evaluations import (
    LATEST_EVALUATION_RUN_ID_TAG,
    ServingGateEvaluationRecord,
    load_evaluation_run,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_CHAMPION_ALIAS,
    resolve_model_name,
    resolve_tracking_uri,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--candidate-alias", default=DEFAULT_CANDIDATE_ALIAS)
    parser.add_argument("--candidate-version", default=None)
    parser.add_argument("--champion-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--approval-status", default="pending")
    parser.add_argument("--rollout-status", default="not_started")
    parser.add_argument("--runtime-target", default="release")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--requested-by", default=os.getenv("USER") or "airflow")
    parser.add_argument("--approved-by", default=None)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_model_version(value: str) -> str:
    normalized = value.strip()
    if normalized.lower().startswith("v") and normalized[1:].isdigit():
        return normalized[1:]
    return normalized


def resolve_source_alias_if_matching(
        *,
        client: MlflowClient,
        model_name: str,
        candidate_alias: str,
        version: str,
) -> str | None:
    try:
        alias_version = client.get_model_version_by_alias(model_name, candidate_alias)
    except MlflowException:
        return None
    return candidate_alias if str(alias_version.version) == str(version) else None


def resolve_candidate(
        client: MlflowClient,
        model_name: str,
        candidate_alias: str,
        candidate_version: str | None,
) -> dict[str, Any]:
    if candidate_version is not None:
        version = client.get_model_version(
            model_name,
            normalize_model_version(candidate_version),
        )
        source_alias = resolve_source_alias_if_matching(
            client=client,
            model_name=model_name,
            candidate_alias=candidate_alias,
            version=str(version.version),
        )
    else:
        version = client.get_model_version_by_alias(model_name, candidate_alias)
        source_alias = candidate_alias
    tags = dict(getattr(version, "tags", {}) or {})
    return {
        "source_alias": source_alias,
        "version": str(version.version),
        "run_id": str(version.run_id),
        "latest_evaluation_run_id": tags.get(LATEST_EVALUATION_RUN_ID_TAG),
    }


def resolve_current_champion(
        client: MlflowClient,
        model_name: str,
        champion_alias: str,
) -> dict[str, str]:
    version = client.get_model_version_by_alias(model_name, champion_alias)
    return {
        "version": str(version.version),
        "run_id": str(version.run_id),
    }


def validate_evaluation_for_deployment(
        evaluation: ServingGateEvaluationRecord,
        *,
        model_name: str,
        candidate: dict[str, Any],
        current_champion: dict[str, str],
) -> None:
    if evaluation.evaluation_status != "passed":
        raise RuntimeError(
            "serving-gate evaluation did not pass: "
            f"evaluation_run_id={evaluation.evaluation_run_id} "
            f"status={evaluation.evaluation_status}"
        )
    evaluated_candidate = (
        evaluation.model_name,
        evaluation.candidate_model_version,
        evaluation.candidate_model_run_id,
    )
    requested_candidate = (
        model_name,
        candidate["version"],
        candidate["run_id"],
    )
    if evaluated_candidate != requested_candidate:
        raise RuntimeError(
            "candidate does not match serving-gate evaluation: "
            f"evaluation_run_id={evaluation.evaluation_run_id} "
            f"evaluated={evaluated_candidate} "
            f"requested={requested_candidate}"
        )

    # For the corner case
    latest_evaluation_run_id = candidate.get("latest_evaluation_run_id")
    if evaluation.evaluation_run_id != latest_evaluation_run_id:
        raise RuntimeError(
            "serving-gate evaluation is not the Candidate's latest published evaluation: "
            f"requested_evaluation_run_id={evaluation.evaluation_run_id} "
            f"latest_evaluation_run_id={latest_evaluation_run_id}"
        )

    if (
            evaluation.champion_model_version != current_champion["version"]
            or evaluation.champion_model_run_id != current_champion["run_id"]
    ):
        raise RuntimeError(
            "champion changed after serving-gate evaluation; reevaluation is required: "
            f"evaluation_run_id={evaluation.evaluation_run_id} "
            f"evaluated_version={evaluation.champion_model_version} "
            f"evaluated_run_id={evaluation.champion_model_run_id} "
            f"current_version={current_champion['version']} "
            f"current_run_id={current_champion['run_id']}"
        )


def main() -> None:
    args = parse_args()
    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    evaluation = load_evaluation_run(client, args.evaluation_run_id)
    candidate = resolve_candidate(
        client=client,
        model_name=args.model_name,
        candidate_alias=args.candidate_alias,
        candidate_version=args.candidate_version,
    )
    current_champion = resolve_current_champion(
        client=client,
        model_name=args.model_name,
        champion_alias=args.champion_alias,
    )
    validate_evaluation_for_deployment(
        evaluation,
        model_name=args.model_name,
        candidate=candidate,
        current_champion=current_champion,
    )

    row = build_deployment_request_row(
        request_id=args.request_id,
        model_name=args.model_name,
        source_alias=candidate["source_alias"],
        source_version=candidate["version"],
        source_run_id=candidate["run_id"],
        target_alias=args.champion_alias,
        previous_version=evaluation.champion_model_version,
        previous_run_id=evaluation.champion_model_run_id,
        eval_type=SERVING_GATE_EVAL_TYPE,
        eval_status=evaluation.evaluation_status, # passed
        approval_status=args.approval_status,
        rollout_status=args.rollout_status,
        runtime_target=args.runtime_target,
        eval_summary=evaluation.summary,
        notes=args.notes,
        requested_by=args.requested_by,
        approved_by=args.approved_by,
    )

    if args.dry_run:
        action = "serving_candidate_deployment_request_dry_run"
    else:
        insert_deployment_request(row)
        action = "serving_candidate_deployment_request_recorded"

    print(
        f"{action} "
        f"request_id={row['request_id']} "
        f"evaluation_run_id={evaluation.evaluation_run_id} "
        f"tracking_uri={tracking_uri} "
        f"model_name={row['model_name']} "
        f"source_alias={row['source_alias']} "
        f"source_version={row['source_version']} "
        f"source_run_id={row['source_run_id']} "
        f"target_alias={row['target_alias']} "
        f"previous_version={row['previous_version']} "
        f"previous_run_id={row['previous_run_id']} "
        f"eval_status={row['eval_status']} "
        f"approval_status={row['approval_status']} "
        f"rollout_status={row['rollout_status']} "
        f"runtime_target={row['runtime_target']}"
    )


if __name__ == "__main__":
    main()
