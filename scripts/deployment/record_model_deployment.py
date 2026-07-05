import argparse
import json
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from secom_mlops.monitor.deployments import (
    build_deployment_request_row,
    insert_deployment_request,
    validate_deployment_statuses,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CHAMPION_ALIAS,
    resolve_model_name,
    resolve_tracking_uri,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())

    parser.add_argument("--source-alias", default=None)
    parser.add_argument("--source-version", default=None)
    parser.add_argument("--target-alias", default=DEFAULT_CHAMPION_ALIAS)

    parser.add_argument("--eval-type", default="unknown")
    parser.add_argument("--eval-status", default=None)
    parser.add_argument("--gate-status", default=None)
    parser.add_argument("--approval-status", default="pending")
    parser.add_argument("--rollout-status", default="not_started")
    parser.add_argument("--runtime-target", default="release")

    parser.add_argument("--eval-summary-json", default=None)
    parser.add_argument("--metric-summary-json", default=None)

    parser.add_argument("--notes", default=None)
    parser.add_argument("--requested-by", default=os.getenv("USER"))
    parser.add_argument("--approved-by", default=None)

    parser.add_argument("--request-id", default=None)
    return parser.parse_args()


def resolved_eval_status(args: argparse.Namespace) -> str:
    return args.eval_status or args.gate_status or "unknown"


def validate_args(args: argparse.Namespace) -> None:
    if bool(args.source_alias) == bool(args.source_version):
        raise ValueError("Specify exactly one of --source-alias or --source-version")

    validate_deployment_statuses(
        eval_status=resolved_eval_status(args),
        approval_status=args.approval_status,
        rollout_status=args.rollout_status,
    )


def resolve_source(
        client: MlflowClient,
        model_name: str,
        source_alias: str | None,
        source_version: str | None,
) -> dict:
    if source_version is not None:
        version = client.get_model_version(model_name, source_version)
        resolved_alias = None
    else:
        version = client.get_model_version_by_alias(model_name, source_alias)
        resolved_alias = source_alias

    return {
        "alias": resolved_alias,
        "version": str(version.version),
        "run_id": str(version.run_id),
    }


def resolve_previous(
        client: MlflowClient,
        model_name: str,
        target_alias: str,
) -> dict:
    try:
        version = client.get_model_version_by_alias(model_name, target_alias)
    except MlflowException:
        return {
            "version": None,
            "run_id": None,
        }

    return {
        "version": str(version.version),
        "run_id": str(version.run_id),
    }


def parse_json_object(raw: str | None) -> dict:
    if raw is None:
        return {}

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("summary JSON must be a JSON object")
    return parsed


def main() -> None:
    args = parse_args()
    validate_args(args)

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    summary_raw = args.eval_summary_json
    if summary_raw is None:
        summary_raw = args.metric_summary_json
    eval_summary = parse_json_object(summary_raw)

    source = resolve_source(
        client=client,
        model_name=args.model_name,
        source_alias=args.source_alias,
        source_version=args.source_version,
    )
    previous = resolve_previous(
        client=client,
        model_name=args.model_name,
        target_alias=args.target_alias,
    )

    row = build_deployment_request_row(
        request_id=args.request_id,
        model_name=args.model_name,
        source_alias=source["alias"],
        source_version=source["version"],
        source_run_id=source["run_id"],
        target_alias=args.target_alias,
        previous_version=previous["version"],
        previous_run_id=previous["run_id"],
        eval_type=args.eval_type,
        eval_status=resolved_eval_status(args),
        approval_status=args.approval_status,
        rollout_status=args.rollout_status,
        runtime_target=args.runtime_target,
        eval_summary=eval_summary,
        notes=args.notes,
        requested_by=args.requested_by,
        approved_by=args.approved_by,
    )

    insert_deployment_request(row)

    print(
        "model_deployment_request_recorded "
        f"request_id={row['request_id']} "
        f"model_name={row['model_name']} "
        f"source_alias={row['source_alias']} "
        f"source_version={row['source_version']} "
        f"source_run_id={row['source_run_id']} "
        f"target_alias={row['target_alias']} "
        f"previous_version={row['previous_version']} "
        f"previous_run_id={row['previous_run_id']} "
        f"eval_type={row['eval_type']} "
        f"eval_status={row['eval_status']} "
        f"approval_status={row['approval_status']} "
        f"rollout_status={row['rollout_status']} "
        f"runtime_target={row['runtime_target']}"
    )


if __name__ == "__main__":
    main()
