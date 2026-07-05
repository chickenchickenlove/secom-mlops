import argparse

import mlflow
from mlflow.tracking import MlflowClient

from secom_mlops_common.config.mlflow import (
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_CHAMPION_ALIAS,
    resolve_model_name,
    resolve_tracking_uri,
)

DEFAULT_METRICS = [
    "f1_1",
    "recall_1",
    "precision_1",
    "pr_auc",
    "balanced_accuracy",
    "accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--candidate-alias", default=DEFAULT_CANDIDATE_ALIAS)
    parser.add_argument("--champion-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--primary-metric", default="f1_1")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--min-primary-delta", type=float, default=0.0)
    parser.add_argument("--min-recall-delta", type=float, default=-0.02)
    parser.add_argument("--min-precision-delta", type=float, default=-0.05)
    parser.add_argument("--set-tags", action="store_true")
    return parser.parse_args()


def metric(run, name: str) -> float | None:
    value = run.data.metrics.get(name)
    if value is None:
        return None
    return float(value)


def delta(candidate_value: float | None, champion_value: float | None) -> float | None:
    if candidate_value is None or champion_value is None:
        return None
    return candidate_value - champion_value


def format_value(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.6f}"


def get_alias_bundle(
        client: MlflowClient,
        model_name: str,
        alias: str,
) -> dict:
    version = client.get_model_version_by_alias(model_name, alias)
    run = client.get_run(version.run_id)

    return {
        "alias": alias,
        "version": str(version.version),
        "run_id": str(version.run_id),
        "run": run,
        "version_tags": dict(version.tags or {}),
        "run_tags": dict(run.data.tags or {}),
    }


def evaluate_gate(
        candidate: dict,
        champion: dict,
        primary_metric: str,
        min_primary_delta: float,
        min_recall_delta: float,
        min_precision_delta: float,
) -> tuple[bool, list[str]]:
    reasons = []

    candidate_primary = metric(candidate["run"], primary_metric)
    champion_primary = metric(champion["run"], primary_metric)
    primary_delta = delta(candidate_primary, champion_primary)

    if candidate_primary is None:
        reasons.append(f"candidate missing primary metric: {primary_metric}")
    elif champion_primary is None:
        reasons.append(f"champion missing primary metric: {primary_metric}")
    elif primary_delta is not None and primary_delta < min_primary_delta:
        reasons.append(
            f"{primary_metric} delta below gate: "
            f"delta={primary_delta:.6f} required>={min_primary_delta:.6f}"
        )

    recall_delta = delta(metric(candidate["run"], "recall_1"), metric(champion["run"], "recall_1"))
    if recall_delta is not None and recall_delta < min_recall_delta:
        reasons.append(
            f"recall_1 regression too large: "
            f"delta={recall_delta:.6f} required>={min_recall_delta:.6f}"
        )

    precision_delta = delta(metric(candidate["run"], "precision_1"), metric(champion["run"], "precision_1"))
    if precision_delta is not None and precision_delta < min_precision_delta:
        reasons.append(
            f"precision_1 regression too large: "
            f"delta={precision_delta:.6f} required>={min_precision_delta:.6f}"
        )

    return len(reasons) == 0, reasons


def set_gate_tags(
        client: MlflowClient,
        model_name: str,
        candidate_version: str,
        passed: bool,
        reasons: list[str],
) -> None:
    client.set_model_version_tag(
        model_name,
        candidate_version,
        "gate_status",
        "passed" if passed else "failed",
    )
    client.set_model_version_tag(
        model_name,
        candidate_version,
        "gate_reason",
        " | ".join(reasons) if reasons else "ok",
    )


def main() -> None:
    args = parse_args()
    tracking_uri = resolve_tracking_uri(args.tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    metrics = [
        item.strip()
        for item in args.metrics.split(",")
        if item.strip()
    ]

    candidate = get_alias_bundle(
        client=client,
        model_name=args.model_name,
        alias=args.candidate_alias,
    )
    champion = get_alias_bundle(
        client=client,
        model_name=args.model_name,
        alias=args.champion_alias,
    )

    passed, reasons = evaluate_gate(
        candidate=candidate,
        champion=champion,
        primary_metric=args.primary_metric,
        min_primary_delta=args.min_primary_delta,
        min_recall_delta=args.min_recall_delta,
        min_precision_delta=args.min_precision_delta,
    )

    if args.set_tags:
        set_gate_tags(
            client=client,
            model_name=args.model_name,
            candidate_version=candidate["version"],
            passed=passed,
            reasons=reasons,
        )

    print("candidate_vs_champion_comparison")
    print(f"tracking_uri={tracking_uri}")
    print(f"model_name={args.model_name}")

    print(
        f"candidate alias={candidate['alias']} "
        f"version={candidate['version']} "
        f"run_id={candidate['run_id']} "
        f"version_tags={candidate['version_tags']}"
    )
    print(
        f"champion alias={champion['alias']} "
        f"version={champion['version']} "
        f"run_id={champion['run_id']} "
        f"version_tags={champion['version_tags']}"
    )

    for name in metrics:
        candidate_value = metric(candidate["run"], name)
        champion_value = metric(champion["run"], name)
        metric_delta = delta(candidate_value, champion_value)

        print(
            f"metric={name} "
            f"candidate={format_value(candidate_value)} "
            f"champion={format_value(champion_value)} "
            f"delta={format_value(metric_delta)}"
        )

    print(f"gate_status={'passed' if passed else 'failed'}")

    for reason in reasons:
        print(f"gate_reason={reason}")

    if passed:
        print(
            "promote_command="
            f"python scripts/deployment/promote_model_alias.py "
            f"--model-version {candidate['version']} "
            f"--target-alias {args.champion_alias}"
        )


if __name__ == "__main__":
    main()
