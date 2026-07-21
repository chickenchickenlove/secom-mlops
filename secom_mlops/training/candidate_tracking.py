"""Build the MLflow tracking contract for Dataset-based candidate training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FINAL_FIT_SCOPE = "complete_development_cohort"
FINAL_EVALUATION_SOURCE = "serving_prediction_decision_gate"
OFFLINE_METRIC_SCOPE = "validation_selection"
PROJECT = "secom-fail-detection"
RUN_STAGE = "training-dataset-candidate"
RUN_PURPOSE = "candidate_training_from_versioned_training_dataset"
_REGISTERED_ONLY_EVENT_FIELDS = {
    "cohort_start_time",
    "cohort_end_time",
    "cutoff_time",
    "label_maturity_seconds",
    "final_fit_sample_count",
}


@dataclass(frozen=True)
class CandidateTrackingContext:
    tracking_uri: str
    model_name: str
    model_alias: str
    model_role: str
    candidate_group: str | None
    training_job_id: str | None
    random_state: int
    min_label_coverage: float
    validation_size: float
    gate_source: str


@dataclass(frozen=True)
class CandidateSplitStats:
    training_sample_count: int
    validation_sample_count: int
    training_fail_count: int
    validation_fail_count: int


@dataclass(frozen=True)
class CandidateTrackingRecord:
    context: CandidateTrackingContext
    params: dict[str, Any]
    metrics: dict[str, float]
    run_tags: dict[str, str]
    dataset_input_tags: dict[str, str]
    summary: dict[str, Any]
    version_tag_values: dict[str, Any]
    event_fields: dict[str, Any]

    def model_version_tags(self, run_id: str) -> dict[str, Any]:
        return {
            "registered_model_alias": self.context.model_alias,
            "source_run_id": run_id,
            **self.version_tag_values,
        }

    def dry_run_message(self) -> str:
        return format_event(
            "training_dataset_candidate_training_dry_run",
            {
                key: value
                for key, value in self.event_fields.items()
                if key not in _REGISTERED_ONLY_EVENT_FIELDS
            },
        )

    def registered_message(self, *, model_version: str, run_id: str) -> str:
        return format_event(
            "training_dataset_candidate_registered",
            {
                "tracking_uri": self.context.tracking_uri,
                "dataset_id": self.event_fields["dataset_id"],
                "dataset_selection_hash": self.event_fields[
                    "dataset_selection_hash"
                ],
                "model_name": self.context.model_name,
                "alias": self.context.model_alias,
                "version": model_version,
                "run_id": run_id,
                **{
                    key: value
                    for key, value in self.event_fields.items()
                    if key not in {"dataset_id", "dataset_selection_hash"}
                },
            },
        )


def _string_values(values: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in values.items()}


def format_event(name: str, fields: dict[str, Any]) -> str:
    return " ".join([name, *(f"{key}={value}" for key, value in fields.items())])


def build_candidate_tracking_record(
        *,
        context: CandidateTrackingContext,
        metadata: dict[str, Any],
        best_row: dict[str, Any],
        split_stats: CandidateSplitStats,
) -> CandidateTrackingRecord:
    training_lineage = {
        "train_source": metadata["train_source"],
        "training_spine": metadata["training_spine"],
        "training_decision_time": metadata["training_decision_time"],
        "snapshot_selection": metadata["snapshot_selection"],
        "label_selection": metadata["label_selection"],
        "gate_source": context.gate_source,
    }
    dataset_lineage = {
        "training_dataset_id": metadata["dataset_id"],
        "training_dataset_manifest_hash": metadata["dataset_manifest_hash"],
        "training_dataset_artifact_sha256": metadata["dataset_artifact_sha256"],
        "training_dataset_mlflow_run_id": metadata["dataset_mlflow_run_id"],
        "training_dataset_selection_hash": metadata["dataset_selection_hash"],
    }
    cohort_lineage = {
        "cohort_start_time": metadata["cohort_start_time"],
        "cohort_end_time": metadata["cohort_end_time"],
        "cutoff_time": metadata["cutoff_time"],
        "label_maturity_seconds": metadata["label_maturity_seconds"],
    }
    development_contract = {
        "development_sample_limit": metadata["development_sample_limit"],
        "development_sample_selection": metadata[
            "development_sample_selection"
        ],
        "min_label_coverage": context.min_label_coverage,
        "validation_size": context.validation_size,
        "final_fit_scope": FINAL_FIT_SCOPE,
        "final_evaluation_source": FINAL_EVALUATION_SOURCE,
    }
    run_identity = {
        "role": context.model_role,
        "candidate_group": context.candidate_group,
        "training_job_id": context.training_job_id,
    }

    params = {
        "model_name": "RandomForestClassifier",
        "registered_model_name": context.model_name,
        "registered_model_alias": context.model_alias,
        "n_estimators": int(best_row["n_estimators"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "class_weight": "balanced",
        "random_state": context.random_state,
        "stratify": True,
        "imputer_strategy": "median",
        "threshold": float(best_row["threshold"]),
        "positive_class": 1,
        **development_contract,
        **training_lineage,
        **dataset_lineage,
        **cohort_lineage,
        "simulation_run_id": metadata["simulation_run_id"],
        "drift_segment": metadata["drift_segment"],
    }

    validation_metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision_1",
        "recall_1",
        "f1_1",
        "pr_auc",
    )
    metrics = {
        **{
            name: float(best_row[name])
            for name in (*validation_metric_names, "tn", "fp", "fn", "tp")
        },
        **{
            f"validation_{name}": float(best_row[name])
            for name in validation_metric_names
        },
        "label_coverage": float(metadata["label_coverage"]),
        "training_sample_count": float(split_stats.training_sample_count),
        "validation_sample_count": float(split_stats.validation_sample_count),
        "final_fit_sample_count": float(metadata["sample_count"]),
        "training_fail_count": float(split_stats.training_fail_count),
        "validation_fail_count": float(split_stats.validation_fail_count),
        "final_fit_fail_count": float(metadata["fail_count"]),
    }

    run_tags = _string_values({
        "project": PROJECT,
        "stage": RUN_STAGE,
        "purpose": RUN_PURPOSE,
        **run_identity,
        **training_lineage,
        **dataset_lineage,
        **cohort_lineage,
        "development_sample_selection": metadata[
            "development_sample_selection"
        ],
        "min_label_coverage": context.min_label_coverage,
        "offline_metric_scope": OFFLINE_METRIC_SCOPE,
        "final_evaluation_source": FINAL_EVALUATION_SOURCE,
    })

    dataset_input_tags = _string_values({
        "source_manifest_hash": metadata["dataset_manifest_hash"],
        "source_artifact_uri": metadata["dataset_artifact_uri"],
        "source_artifact_sha256": metadata["dataset_artifact_sha256"],
        "source_mlflow_run_id": metadata["dataset_mlflow_run_id"],
        "selection": metadata["development_sample_selection"],
        "selection_limit": metadata["development_sample_limit"],
        "selection_hash": metadata["dataset_selection_hash"],
    })

    summary = {
        "tracking_uri": context.tracking_uri,
        "model_name": context.model_name,
        "model_alias": context.model_alias,
        "model_role": context.model_role,
        "candidate_group": context.candidate_group,
        "training_job_id": context.training_job_id,
        "metadata": metadata,
        "train_sample_count": split_stats.training_sample_count,
        "validation_sample_count": split_stats.validation_sample_count,
        "final_fit_sample_count": metadata["sample_count"],
        "best_row": best_row,
    }

    version_tag_values = {
        **run_identity,
        **training_lineage,
        **dataset_lineage,
        **development_contract,
        **cohort_lineage,
    }

    event_fields = {
        "dataset_id": metadata["dataset_id"],
        "dataset_selection_hash": metadata["dataset_selection_hash"],
        "sample_count": metadata["sample_count"],
        "fail_count": metadata["fail_count"],
        "pass_count": metadata["pass_count"],
        "eligible_cohort_count": metadata["eligible_cohort_count"],
        "unlabeled_cohort_count": metadata["unlabeled_cohort_count"],
        "label_coverage": metadata["label_coverage"],
        "min_label_coverage": context.min_label_coverage,
        **cohort_lineage,
        "development_sample_limit": metadata["development_sample_limit"],
        "development_sample_selection": metadata[
            "development_sample_selection"
        ],
        "first_sample_id": metadata["first_sample_id"],
        "last_sample_id": metadata["last_sample_id"],
        "decision_time_min": metadata["decision_time_min"],
        "decision_time_max": metadata["decision_time_max"],
        "training_sample_count": split_stats.training_sample_count,
        "validation_sample_count": split_stats.validation_sample_count,
        "final_fit_sample_count": metadata["sample_count"],
        "best_f1_1": best_row["f1_1"],
        "best_recall_1": best_row["recall_1"],
        "best_precision_1": best_row["precision_1"],
        "best_threshold": best_row["threshold"],
    }

    return CandidateTrackingRecord(
        context=context,
        params=params,
        metrics=metrics,
        run_tags=run_tags,
        dataset_input_tags=dataset_input_tags,
        summary=summary,
        version_tag_values=version_tag_values,
        event_fields=event_fields,
    )
