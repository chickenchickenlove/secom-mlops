import argparse
from datetime import datetime, timezone
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from secom_mlops_common.config.mlflow import resolve_model_name, resolve_tracking_uri

CLEANUP_POLICIES: dict[str, dict[str, Any]] = {
    "serving_snapshot_eval_rejected": {
        "required_tag": "candidate_serving_snapshot_eval_status",
        "allowed_tag_values": ("failed", "insufficient_data"),
        "review_status": "rejected",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--alias", required=True)
    parser.add_argument(
        "--cleanup-policy",
        choices=sorted(CLEANUP_POLICIES),
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_alias_version(client: MlflowClient, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def model_version_tags(model_version) -> dict[str, str]:
    return dict(getattr(model_version, "tags", {}) or {})


def main() -> None:
    args = parse_args()
    policy = CLEANUP_POLICIES[args.cleanup_policy]
    required_tag = str(policy["required_tag"])
    allowed_tag_values = tuple(policy["allowed_tag_values"])
    review_status = str(policy["review_status"])

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_version = get_alias_version(client, args.model_name, args.alias)
    if model_version is None:
        print(
            "model_alias_clear_skipped "
            f"tracking_uri={tracking_uri} "
            f"model_name={args.model_name} "
            f"alias={args.alias} "
            "reason=alias_not_found"
        )
        return

    tags = model_version_tags(model_version)
    actual_tag_value = tags.get(required_tag)

    if actual_tag_value not in set(allowed_tag_values):
        print(
            "model_alias_clear_skipped "
            f"tracking_uri={tracking_uri} "
            f"model_name={args.model_name} "
            f"alias={args.alias} "
            f"version={model_version.version} "
            f"run_id={model_version.run_id} "
            "reason=required_tag_not_matched "
            f"cleanup_policy={args.cleanup_policy} "
            f"required_tag={required_tag} "
            f"actual_tag_value={actual_tag_value} "
            f"allowed_tag_values={','.join(allowed_tag_values)}"
        )
        return

    if args.dry_run:
        print(
            "model_alias_clear_dry_run "
            f"tracking_uri={tracking_uri} "
            f"model_name={args.model_name} "
            f"alias={args.alias} "
            f"version={model_version.version} "
            f"run_id={model_version.run_id} "
            f"cleanup_policy={args.cleanup_policy} "
            f"required_tag={required_tag} "
            f"actual_tag_value={actual_tag_value} "
            f"candidate_review_status={review_status}"
        )
        return

    current = get_alias_version(client, args.model_name, args.alias)
    if current is None or str(current.version) != str(model_version.version):
        print(
            "model_alias_clear_skipped "
            f"tracking_uri={tracking_uri} "
            f"model_name={args.model_name} "
            f"alias={args.alias} "
            f"expected_version={model_version.version} "
            f"actual_version={current.version if current else None} "
            "reason=alias_changed_before_delete"
        )
        return

    now = datetime.now(timezone.utc).isoformat()
    review_reason = f"{required_tag}={actual_tag_value}"

    client.set_model_version_tag(args.model_name, model_version.version, "candidate_review_status", review_status)
    client.set_model_version_tag(args.model_name, model_version.version, "candidate_review_policy", args.cleanup_policy)
    client.set_model_version_tag(args.model_name, model_version.version, "candidate_review_reason", review_reason)
    client.set_model_version_tag(args.model_name, model_version.version, "candidate_reviewed_at_utc", now)

    client.delete_registered_model_alias(args.model_name, args.alias)

    print(
        "model_alias_clear_complete "
        f"tracking_uri={tracking_uri} "
        f"model_name={args.model_name} "
        f"alias={args.alias} "
        f"version={model_version.version} "
        f"run_id={model_version.run_id} "
        f"cleanup_policy={args.cleanup_policy} "
        f"required_tag={required_tag} "
        f"actual_tag_value={actual_tag_value} "
        f"candidate_review_status={review_status}"
    )


if __name__ == "__main__":
    main()
