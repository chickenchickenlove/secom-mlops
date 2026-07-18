import argparse
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from secom_mlops.datasets.serving_gate_dataset import (
    DECISION_SELECTION,
    DEFAULT_MIN_LABELED_DECISIONS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
)
from secom_mlops.datasets.serving_gate_dataset_loader import (
    load_serving_gate_dataset,
)
from secom_mlops.monitor.serving_gate_evaluations import (
    COMPARISON_TYPE,
    DEFAULT_EXPERIMENT_NAME,
    EVALUATION_SCHEMA_VERSION,
    LATEST_EVALUATION_RUN_ID_TAG,
    evaluation_reason_json,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CANDIDATE_ALIAS,
    DEFAULT_CHAMPION_ALIAS,
    resolve_model_name,
    resolve_tracking_uri,
)
from secom_mlops_common.schemas.secom import MODEL_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--model-name", default=resolve_model_name())
    parser.add_argument("--candidate-alias", default=DEFAULT_CANDIDATE_ALIAS)
    parser.add_argument("--champion-alias", default=DEFAULT_CHAMPION_ALIAS)
    parser.add_argument("--candidate-version", default=None)
    parser.add_argument("--champion-version", default=None)
    parser.add_argument(
        "--evaluation-experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
    )
    parser.add_argument("--primary-metric", default="fail_f1")
    parser.add_argument("--min-primary-delta", type=float, default=0.0)
    parser.add_argument("--min-recall-delta", type=float, default=-0.02)
    parser.add_argument("--min-precision-delta", type=float, default=-0.05)
    parser.add_argument("--set-tags", action="store_true")
    parser.add_argument("--fail-on-gate-failure", action="store_true")
    return parser.parse_args()


def resolve_model(
        client: MlflowClient,
        model_name: str,
        alias: str,
        version: str | None,
) -> dict[str, Any]:
    if version:
        model_version = client.get_model_version(model_name, version)
        resolved_alias = None
    else:
        model_version = client.get_model_version_by_alias(model_name, alias)
        resolved_alias = alias

    # Pin the resolved version. An alias may move while the Gate is running.
    model_uri = f"models:/{model_name}/{model_version.version}"
    return {
        "model_uri": model_uri,
        "model": mlflow.pyfunc.load_model(model_uri),
        "model_version": str(model_version.version),
        "model_alias": resolved_alias,
        "model_run_id": str(model_version.run_id),
    }


def validate_distinct_models(
        candidate: dict[str, Any],
        champion: dict[str, Any],
) -> None:
    if candidate["model_run_id"] == champion["model_run_id"]:
        raise RuntimeError(
            "candidate and champion must be different models: "
            f"model_run_id={candidate['model_run_id']}"
        )


def load_labeled_serving_decisions(
        dataset_id: str,
        *,
        tracking_uri: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    loaded = load_serving_gate_dataset(dataset_id, tracking_uri=tracking_uri)
    frame = loaded.frame
    labeled = frame["label_event_id"].notna()
    labeled_frame = frame.loc[labeled].copy()
    labeled_count = len(labeled_frame)
    if labeled_count < DEFAULT_MIN_LABELED_DECISIONS:
        raise RuntimeError(
            "persisted serving-gate dataset has too few labeled decisions: "
            f"required={DEFAULT_MIN_LABELED_DECISIONS} actual={labeled_count}"
        )

    y_true = labeled_frame["actual_value"].astype("int64")
    actual_values = set(y_true.unique())
    if not actual_values.issubset({NEGATIVE_CLASS, POSITIVE_CLASS}):
        raise RuntimeError(
            f"persisted serving-gate dataset has invalid targets: {sorted(actual_values)}"
        )
    features = labeled_frame.loc[:, list(MODEL_COLUMNS)].astype("float64")
    prediction_ids = labeled_frame["prediction_id"].astype(str).tolist()

    identity = loaded.manifest["identity"]
    build_context = loaded.manifest["build_context"]
    stats = loaded.manifest["stats"]
    metadata = {
        "dataset_id": loaded.dataset_id,
        "manifest_hash": loaded.manifest_hash,
        "artifact_sha256": loaded.artifact_sha256,
        "dataset_mlflow_run_id": loaded.mlflow_run_id,
        "artifact_uri": loaded.artifact_uri,
        "cohort_start_time": identity["cohort_start_time"],
        "cohort_end_time": identity["cohort_end_time"],
        "cutoff_time": build_context["cutoff_time"],
        "label_maturity_seconds": identity["label_maturity_seconds"],
        "decision_selection": identity["decision_selection"],
        "decision_count": int(stats["decision_count"]),
        "labeled_decision_count": int(stats["labeled_decision_count"]),
        "unlabeled_decision_count": int(stats["unlabeled_decision_count"]),
        "unique_sample_count": int(stats["unique_sample_count"]),
        "label_coverage": float(stats["label_coverage"]),
        "fail_count": int(stats["fail_count"]),
        "pass_count": int(stats["pass_count"]),
        "decision_time_min": stats["decision_time_min"],
        "decision_time_max": stats["decision_time_max"],
        "snapshot_available_at_min": stats["snapshot_available_at_min"],
        "snapshot_available_at_max": stats["snapshot_available_at_max"],
        "label_available_at_min": stats["label_available_at_min"],
        "label_available_at_max": stats["label_available_at_max"],
        "source_model_run_ids": stats["source_model_run_ids"],
        "source_thresholds": stats["source_thresholds"],
        "first_sample_id": str(frame.iloc[0]["sample_id"]),
        "last_sample_id": str(frame.iloc[-1]["sample_id"]),
    }
    return features, y_true, prediction_ids, metadata


def predict_model(model_bundle: dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    predictions = model_bundle["model"].predict(features.copy())
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)

    if "fail_probability" not in predictions.columns:
        raise ValueError("model output missing fail_probability")
    if "prediction" in predictions.columns:
        predicted_value = predictions["prediction"]
    elif "predicted_value" in predictions.columns:
        predicted_value = predictions["predicted_value"]
    else:
        raise ValueError("model output missing prediction")

    return pd.DataFrame({
        "fail_probability": predictions["fail_probability"].astype(float),
        "predicted_value": predicted_value.astype(int),
    })


def evaluate_predictions(
        y_true: pd.Series,
        prediction_frame: pd.DataFrame,
) -> dict[str, float | int | None]:
    y_pred = prediction_frame["predicted_value"].astype(int)
    fail_probability = prediction_frame["fail_probability"].astype(float)
    matrix = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_CLASS, POSITIVE_CLASS])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    n_fail_samples = int((y_true == POSITIVE_CLASS).sum())
    pr_auc = (
        float(average_precision_score(y_true, fail_probability, pos_label=POSITIVE_CLASS))
        if n_fail_samples > 0
        else None
    )
    return {
        "n_samples": int(len(y_true)),
        "n_fail_samples": n_fail_samples,
        "n_pass_samples": int((y_true == NEGATIVE_CLASS).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fail_precision": float(precision_score(
            y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        )),
        "fail_recall": float(recall_score(
            y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        )),
        "fail_f1": float(f1_score(
            y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        )),
        "pr_auc": pr_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def metric_names() -> list[str]:
    return [
        "fail_f1",
        "fail_recall",
        "fail_precision",
        "pr_auc",
        "balanced_accuracy",
        "accuracy",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]


def metric_delta(
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
        metric_name: str,
) -> float | None:
    candidate_value = candidate_metrics.get(metric_name)
    champion_value = champion_metrics.get(metric_name)
    if candidate_value is None or champion_value is None:
        return None
    return float(candidate_value) - float(champion_value)


def evaluate_gate(
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
        primary_metric: str,
        min_primary_delta: float,
        min_recall_delta: float,
        min_precision_delta: float,
) -> tuple[str, list[str]]:
    reasons = []
    primary_delta = metric_delta(candidate_metrics, champion_metrics, primary_metric)
    if primary_delta is None:
        reasons.append(f"primary metric unavailable: {primary_metric}")
    elif primary_delta < min_primary_delta:
        reasons.append(
            f"{primary_metric} delta below gate: delta={primary_delta:.6f} "
            f"required>={min_primary_delta:.6f}"
        )

    recall_delta = metric_delta(candidate_metrics, champion_metrics, "fail_recall")
    if recall_delta is not None and recall_delta < min_recall_delta:
        reasons.append(
            "fail_recall regression too large: "
            f"delta={recall_delta:.6f} required>={min_recall_delta:.6f}"
        )
    precision_delta = metric_delta(candidate_metrics, champion_metrics, "fail_precision")
    if precision_delta is not None and precision_delta < min_precision_delta:
        reasons.append(
            "fail_precision regression too large: "
            f"delta={precision_delta:.6f} required>={min_precision_delta:.6f}"
        )
    return ("failed" if reasons else "passed"), reasons


def format_metric(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def log_evaluation_run(
        *,
        experiment_name: str,
        status: str,
        reasons: list[str],
        args: argparse.Namespace,
        metadata: dict[str, Any],
        candidate: dict[str, Any],
        champion: dict[str, Any],
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        mlflow.create_experiment(experiment_name)
        if experiment is None
        else experiment.experiment_id
    )

    with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=(
                f"candidate_v{candidate['model_version']}_"
                f"{metadata['dataset_id']}"
            ),
            tags={
                "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
                "comparison_type": COMPARISON_TYPE,
                "dataset_id": metadata["dataset_id"],
                "candidate_model_version": candidate["model_version"],
                "champion_model_version": champion["model_version"],
            },
    ) as run:
        evaluation_run_id = str(run.info.run_id)
        summary = build_eval_summary(
            args=args,
            metadata=metadata,
            candidate=candidate,
            champion=champion,
            status=status,
            reasons=reasons,
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
        )
        summary["evaluation_run_id"] = evaluation_run_id
        mlflow.log_params({
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "comparison_type": COMPARISON_TYPE,
            "evaluation_status": status,
            "evaluation_reasons": evaluation_reason_json(reasons),
            "model_name": args.model_name,
            "dataset_id": metadata["dataset_id"],
            "dataset_manifest_hash": metadata["manifest_hash"],
            "dataset_artifact_sha256": metadata["artifact_sha256"],
            "dataset_mlflow_run_id": metadata["dataset_mlflow_run_id"],
            "candidate_model_version": candidate["model_version"],
            "candidate_model_run_id": candidate["model_run_id"],
            "champion_model_version": champion["model_version"],
            "champion_model_run_id": champion["model_run_id"],
            "primary_metric": args.primary_metric,
            "min_primary_delta": args.min_primary_delta,
            "min_recall_delta": args.min_recall_delta,
            "min_precision_delta": args.min_precision_delta,
            "cohort_start_time": metadata["cohort_start_time"],
            "cohort_end_time": metadata["cohort_end_time"],
            "cutoff_time": metadata["cutoff_time"],
            "label_maturity_seconds": metadata["label_maturity_seconds"],
            "decision_selection": metadata["decision_selection"],
        })

        evaluation_metrics: dict[str, float] = {
            "dataset_decision_count": float(metadata["decision_count"]),
            "dataset_labeled_decision_count": float(
                metadata["labeled_decision_count"]
            ),
            "dataset_label_coverage": float(metadata["label_coverage"]),
            "dataset_fail_count": float(metadata["fail_count"]),
            "dataset_pass_count": float(metadata["pass_count"]),
        }
        for name in metric_names():
            candidate_value = candidate_metrics.get(name)
            champion_value = champion_metrics.get(name)
            delta_value = metric_delta(candidate_metrics, champion_metrics, name)
            if candidate_value is not None:
                evaluation_metrics[f"candidate_{name}"] = float(candidate_value)
            if champion_value is not None:
                evaluation_metrics[f"champion_{name}"] = float(champion_value)
            if delta_value is not None:
                evaluation_metrics[f"delta_{name}"] = float(delta_value)
        mlflow.log_metrics(evaluation_metrics)
        mlflow.log_dict(summary, "evaluation/evaluation.json")
        return evaluation_run_id, summary


def set_candidate_evaluation_pointer(
        client: MlflowClient,
        model_name: str,
        candidate_version: str,
        evaluation_run_id: str,
) -> None:
    client.set_model_version_tag(
        model_name,
        candidate_version,
        LATEST_EVALUATION_RUN_ID_TAG,
        evaluation_run_id,
    )


def build_eval_summary(
        *,
        args: argparse.Namespace,
        metadata: dict[str, Any],
        candidate: dict[str, Any],
        champion: dict[str, Any],
        status: str,
        reasons: list[str],
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
) -> dict[str, Any]:
    names = metric_names()
    return {
        "comparison_type": COMPARISON_TYPE,
        "dataset": {
            "dataset_id": metadata["dataset_id"],
            "manifest_hash": metadata["manifest_hash"],
            "artifact_sha256": metadata["artifact_sha256"],
            "mlflow_run_id": metadata["dataset_mlflow_run_id"],
            "artifact_uri": metadata["artifact_uri"],
        },
        "cohort_start_time": metadata["cohort_start_time"],
        "cohort_end_time": metadata["cohort_end_time"],
        "cutoff_time": metadata["cutoff_time"],
        "label_maturity_seconds": metadata["label_maturity_seconds"],
        "decision_selection": metadata["decision_selection"],
        "gate_policy": {
            "primary_metric": args.primary_metric,
            "min_primary_delta": args.min_primary_delta,
            "min_recall_delta": args.min_recall_delta,
            "min_precision_delta": args.min_precision_delta,
        },
        "eval_status": status,
        "eval_reasons": reasons,
        "metadata": metadata,
        "candidate": {
            "model_version": candidate["model_version"],
            "model_run_id": candidate["model_run_id"],
            "model_uri": candidate["model_uri"],
            "metrics": {name: candidate_metrics.get(name) for name in names},
        },
        "champion": {
            "model_version": champion["model_version"],
            "model_run_id": champion["model_run_id"],
            "model_uri": champion["model_uri"],
            "metrics": {name: champion_metrics.get(name) for name in names},
        },
        "delta": {
            name: metric_delta(candidate_metrics, champion_metrics, name)
            for name in names
        },
    }


def print_result(
        tracking_uri: str,
        args: argparse.Namespace,
        metadata: dict[str, Any],
        candidate: dict[str, Any],
        champion: dict[str, Any],
        status: str,
        reasons: list[str],
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
        evaluation_run_id: str,
) -> None:
    print("serving_gate_dataset_candidate_vs_champion_comparison")
    print(f"tracking_uri={tracking_uri}")
    print(f"model_name={args.model_name}")
    print(f"dataset_id={metadata['dataset_id']}")
    print(f"manifest_hash={metadata['manifest_hash']}")
    print(f"artifact_sha256={metadata['artifact_sha256']}")
    print(f"cohort_start_time={metadata['cohort_start_time']}")
    print(f"cohort_end_time={metadata['cohort_end_time']}")
    print(f"cutoff_time={metadata['cutoff_time']}")
    print(f"label_maturity_seconds={metadata['label_maturity_seconds']}")
    print(f"decision_selection={metadata['decision_selection']}")
    print(f"n_decisions={metadata['decision_count']}")
    print(f"n_labeled_decisions={metadata['labeled_decision_count']}")
    print(f"n_unlabeled_decisions={metadata['unlabeled_decision_count']}")
    print(f"label_coverage={metadata['label_coverage']:.6f}")
    print(f"n_fail_samples={metadata['fail_count']}")
    print(f"n_pass_samples={metadata['pass_count']}")
    print(f"source_model_run_ids={metadata['source_model_run_ids']}")
    print(f"source_thresholds={metadata['source_thresholds']}")
    print(
        f"candidate version={candidate['model_version']} alias={candidate['model_alias']} "
        f"run_id={candidate['model_run_id']}"
    )
    print(
        f"champion version={champion['model_version']} alias={champion['model_alias']} "
        f"run_id={champion['model_run_id']}"
    )
    for name in metric_names():
        print(
            f"metric={name} "
            f"candidate={format_metric(candidate_metrics.get(name))} "
            f"champion={format_metric(champion_metrics.get(name))} "
            f"delta={format_metric(metric_delta(candidate_metrics, champion_metrics, name))}"
        )
    print(f"candidate_serving_snapshot_eval_status={status}")
    for reason in reasons:
        print(f"candidate_serving_snapshot_eval_reason={reason}")
    # Keep the evaluation run ID as the final stdout line for Airflow XCom.
    print(f"evaluation_run_id={evaluation_run_id}")


def main() -> None:
    args = parse_args()
    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    candidate = resolve_model(
        client, args.model_name, args.candidate_alias, args.candidate_version
    )
    champion = resolve_model(
        client, args.model_name, args.champion_alias, args.champion_version
    )
    validate_distinct_models(candidate, champion)
    features, y_true, _, metadata = load_labeled_serving_decisions(
        args.dataset_id,
        tracking_uri=tracking_uri,
    )

    candidate_metrics = evaluate_predictions(y_true, predict_model(candidate, features))
    champion_metrics = evaluate_predictions(y_true, predict_model(champion, features))
    status, reasons = evaluate_gate(
        candidate_metrics,
        champion_metrics,
        args.primary_metric,
        args.min_primary_delta,
        args.min_recall_delta,
        args.min_precision_delta,
    )

    evaluation_run_id, _ = log_evaluation_run(
        experiment_name=args.evaluation_experiment_name,
        status=status,
        reasons=reasons,
        args=args,
        metadata=metadata,
        candidate=candidate,
        champion=champion,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
    )
    if args.set_tags:
        set_candidate_evaluation_pointer(
            client,
            args.model_name,
            candidate["model_version"],
            evaluation_run_id,
        )

    print_result(
        tracking_uri,
        args,
        metadata,
        candidate,
        champion,
        status,
        reasons,
        candidate_metrics,
        champion_metrics,
        evaluation_run_id,
    )
    if args.fail_on_gate_failure and status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
