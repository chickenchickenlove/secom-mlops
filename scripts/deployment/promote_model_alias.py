import argparse

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from secom_mlops.monitor.deployments import (
    find_approved_deployment_request,
    get_deployment_request,
    mark_deployment_request_promoted,
    validate_serving_gate_eval_type,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CHAMPION_ALIAS,
    ENV_ML_TARGET_MODEL_ALIAS,
    resolve_model_alias,
    resolve_model_name,
    resolve_tracking_uri,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--source-alias", default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument(
        "--target-alias",
        default=resolve_model_alias(
            env_name=ENV_ML_TARGET_MODEL_ALIAS,
            default=DEFAULT_CHAMPION_ALIAS,
        ),
    )
    parser.add_argument("--keep-source-alias", action="store_true")
    parser.add_argument("--require-approved-request", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_alias_version(client: MlflowClient, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def resolve_source_version(
        client: MlflowClient,
        model_name: str,
        source_alias: str | None,
        model_version: str | None,
):
    if bool(source_alias) == bool(model_version):
        raise ValueError("Specify exactly one of --source-alias or --model-version")

    if model_version is not None:
        return client.get_model_version(model_name, model_version)

    return client.get_model_version_by_alias(model_name, source_alias)


def validate_request(request: dict, source, target_alias: str) -> None:
    validate_serving_gate_eval_type(request)

    if request["eval_status"] != "passed":
        raise RuntimeError(f"deployment request eval_status is not passed: {request['eval_status']}")

    if request["approval_status"] != "approved":
        raise RuntimeError(f"deployment request approval_status is not approved: {request['approval_status']}")

    if request["rollout_status"] in {"deployed", "failed", "rolled_back", "superseded"}:
        raise RuntimeError(f"deployment request is terminal: rollout_status={request['rollout_status']}")

    if str(request["source_version"]) != str(source.version):
        raise RuntimeError(
            "deployment request source_version mismatch: "
            f"request_id={request['request_id']} "
            f"request_source_version={request['source_version']} "
            f"resolved_source_version={source.version}"
        )

    if str(request["source_run_id"]) != str(source.run_id):
        raise RuntimeError(
            "deployment request source_run_id mismatch: "
            f"request_id={request['request_id']} "
            f"request_source_run_id={request['source_run_id']} "
            f"resolved_source_run_id={source.run_id}"
        )

    if str(request["target_alias"]) != str(target_alias):
        raise RuntimeError(
            "deployment request target_alias mismatch: "
            f"request_id={request['request_id']} "
            f"request_target_alias={request['target_alias']} "
            f"target_alias={target_alias}"
        )


def main() -> None:
    args = parse_args()

    if args.request_id is None and bool(args.source_alias) == bool(args.model_version):
        raise ValueError("Specify --request-id, or exactly one of --source-alias/--model-version")

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    deployment_request = None
    source_alias = args.source_alias
    model_version = args.model_version

    if args.request_id is not None:
        deployment_request = get_deployment_request(args.request_id)
        if deployment_request is None:
            raise RuntimeError(f"deployment request not found: request_id={args.request_id}")

        if model_version is None and source_alias is None:
            model_version = str(deployment_request["source_version"])
            source_alias = deployment_request["source_alias"]

    source = resolve_source_version(
        client=client,
        model_name=args.model_name,
        source_alias=source_alias if model_version is None else None,
        model_version=model_version,
    )

    previous = get_alias_version(
        client=client,
        model_name=args.model_name,
        alias=args.target_alias,
    )

    if args.require_approved_request and deployment_request is None:
        deployment_request = find_approved_deployment_request(
            model_name=args.model_name,
            source_version=str(source.version),
            target_alias=args.target_alias,
        )

        if deployment_request is None:
            raise RuntimeError(
                "approved deployment request not found: "
                f"model_name={args.model_name} "
                f"source_version={source.version} "
                f"target_alias={args.target_alias}"
            )

    if deployment_request is not None:
        validate_request(deployment_request, source, args.target_alias)

    source_alias_to_clear = None
    if source_alias is not None and source_alias != args.target_alias and not args.keep_source_alias:
        source_alias_to_clear = source_alias

    if args.dry_run:
        action = "model_alias_promotion_dry_run"
        source_alias_cleared = False
        deployment_request_marked_promoted = False
    else:
        client.set_registered_model_alias(
            args.model_name,
            args.target_alias,
            source.version,
        )

        source_alias_cleared = False
        if source_alias_to_clear is not None:
            current_source = get_alias_version(
                client=client,
                model_name=args.model_name,
                alias=source_alias_to_clear,
            )
            if current_source is not None and str(current_source.version) == str(source.version):
                client.delete_registered_model_alias(args.model_name, source_alias_to_clear)
                source_alias_cleared = True

        deployment_request_marked_promoted = False
        if deployment_request is not None:
            deployment_request = mark_deployment_request_promoted(deployment_request["request_id"])
            deployment_request_marked_promoted = True

        action = "model_alias_promotion_complete"

    print(
        f"{action} "
        f"tracking_uri={tracking_uri} "
        f"model_name={args.model_name} "
        f"target_alias={args.target_alias} "
        f"previous_version={previous.version if previous else None} "
        f"previous_run_id={previous.run_id if previous else None} "
        f"new_version={source.version} "
        f"new_run_id={source.run_id} "
        f"source_alias={source_alias} "
        f"clear_source_alias={source_alias_to_clear is not None} "
        f"source_alias_cleared={source_alias_cleared} "
        f"require_approved_request={args.require_approved_request} "
        f"deployment_request_id={deployment_request['request_id'] if deployment_request else None} "
        f"deployment_request_marked_promoted={deployment_request_marked_promoted} "
        f"rollout_status={deployment_request['rollout_status'] if deployment_request else None}"
    )


if __name__ == "__main__":
    main()
