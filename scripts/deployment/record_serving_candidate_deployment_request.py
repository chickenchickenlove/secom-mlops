from __future__ import annotations

import argparse
import os
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from secom_mlops.monitor.deployments import (
    build_deployment_request_row,
    insert_deployment_request,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_CHAMPION_ALIAS,
    resolve_model_name,
    resolve_tracking_uri,
)

EVAL_TAG_PREFIX = "candidate_serving_snapshot_"
EVAL_STATUS_TAG = "candidate_serving_snapshot_eval_status"
EVAL_REASON_TAG = "candidate_serving_snapshot_eval_reason"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def resolve_candidate(
        client: MlflowClient,
        model_name: str,
        candidate_alias: str,
        candidate_version: str | None,
) -> dict[str, Any]:
    if candidate_version is not None:
        version = client.get_model_version(model_name, normalize_model_version(candidate_version))
        source_alias = resolve_source_alias_if_matching(
            client=client,
            model_name=model_name,
            candidate_alias=candidate_alias,
            version=str(version.version),
        )
    else:
        version = client.get_model_version_by_alias(model_name, candidate_alias)
        source_alias = candidate_alias

    return {
        "source_alias": source_alias,
        "version": str(version.version),
        "run_id": str(version.run_id),
        "tags": dict(version.tags),
    }


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

    if str(alias_version.version) == str(version):
        return candidate_alias

    return None


def normalize_model_version(value: str) -> str:
    normalized = value.strip()
    if normalized.lower().startswith("v") and normalized[1:].isdigit():
        return normalized[1:]
    return normalized


def resolve_champion(
        client: MlflowClient,
        model_name: str,
        champion_alias: str,
) -> dict[str, Any]:
    try:
        version = client.get_model_version_by_alias(model_name, champion_alias)
    except MlflowException:
        return {
            "version": None,
            "run_id": None,
        }

    return {
        "version": str(version.version),
        "run_id": str(version.run_id),
    }


def validate_candidate_eval(candidate: dict[str, Any]) -> None:
    status = candidate["tags"].get(EVAL_STATUS_TAG)
    if status != "passed":
        raise RuntimeError(
            "candidate serving snapshot eval is not passed: "
            f"candidate_version={candidate['version']} "
            f"status={status} "
            f"reason={candidate['tags'].get(EVAL_REASON_TAG)}"
        )


def build_eval_summary(candidate: dict[str, Any], champion: dict[str, Any]) -> dict[str, Any]:
    tags = {
        key: value
        for key, value in candidate["tags"].items()
        if key.startswith(EVAL_TAG_PREFIX)
    }

    return {
        "comparison_type": "serving_snapshot_candidate_vs_champion",
        "eval_status": candidate["tags"].get(EVAL_STATUS_TAG),
        "eval_reason": candidate["tags"].get(EVAL_REASON_TAG),
        "candidate": {
            "model_version": candidate["version"],
            "model_run_id": candidate["run_id"],
        },
        "champion": {
            "model_version": champion["version"],
            "model_run_id": champion["run_id"],
        },
        "tags": tags,
    }


def main() -> None:
    args = parse_args()
    tracking_uri = resolve_tracking_uri(args.tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    candidate = resolve_candidate(
        client=client,
        model_name=args.model_name,
        candidate_alias=args.candidate_alias,
        candidate_version=args.candidate_version,
    )
    champion = resolve_champion(
        client=client,
        model_name=args.model_name,
        champion_alias=args.champion_alias,
    )

    validate_candidate_eval(candidate)
    eval_summary = build_eval_summary(candidate, champion)

    row = build_deployment_request_row(
        request_id=args.request_id,
        model_name=args.model_name,
        source_alias=candidate["source_alias"],
        source_version=candidate["version"],
        source_run_id=candidate["run_id"],
        target_alias=args.champion_alias,
        previous_version=champion["version"],
        previous_run_id=champion["run_id"],
        eval_type="serving_snapshot",
        eval_status="passed",
        approval_status=args.approval_status,
        rollout_status=args.rollout_status,
        runtime_target=args.runtime_target,
        eval_summary=eval_summary,
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
