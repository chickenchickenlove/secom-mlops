import json
import tempfile
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from secom_mlops.datasets.dataset_artifacts import (
    to_mlflow_dataset_digest,
)
from secom_mlops.training.training_data_preparation import (
    PreparedTrainingData
)
from secom_mlops.training.random_forest_training import (
    RandomForestTrainingResult,
)
from secom_mlops.models.secom_pyfunc_model import SECOMFailDetectionPyfunc
from secom_mlops.training.candidate_tracking import (
    CandidateTrackingRecord,
)

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1


@dataclass(frozen=True)
class CandidateRegistrationConfig:
    tracking_uri: str
    experiment_name: str
    model_name: str
    model_alias: str
    await_registration_for: int = 300


@dataclass(frozen=True)
class RegisteredCandidate:
    run_id: str
    model_version: str


def register_candidate(
        *,
        training_data: PreparedTrainingData,
        training_result: RandomForestTrainingResult,
        tracking: CandidateTrackingRecord,
        config: CandidateRegistrationConfig
) -> RegisteredCandidate:

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    metadata = training_data.metadata
    context = tracking.context

    run_name = (
        "training_dataset_candidate"
        f"_{metadata['dataset_id']}"
        f"_{context.training_job_id}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        _log_tracking_record(tracking)
        _log_training_dataset(training_data, tracking)
        _log_artifacts(
            training_data=training_data,
            training_result=training_result,
            tracking=tracking,
        )

        model_version = _log_and_register_model(
            run_id=run.info.run_id,
            training_data=training_data,
            training_result=training_result,
            tracking=tracking,
            config=config,
        )

        client = MlflowClient()
        _set_candidate_alias(
            client=client,
            model_name=config.model_name,
            model_alias=config.model_alias,
            model_version=model_version,
        )

        return RegisteredCandidate(
            run_id=run.info.run_id,
            model_version=str(model_version),
        )


def _log_and_register_model(
        *,
        run_id: str,
        training_data: PreparedTrainingData,
        training_result: RandomForestTrainingResult,
        tracking: CandidateTrackingRecord,
        config: CandidateRegistrationConfig,
) -> str:
    pyfunc_model = SECOMFailDetectionPyfunc(
        model=training_result.model,
        threshold=float(
            training_result.best_row["threshold"]
        ),
        model_name=config.model_name,
        model_run_id=run_id,
        positive_class=POSITIVE_CLASS
    )

    # Get the input / output interface such as
    #   ModelSignature(
    #       inputs=Schema([
    #           ColSpec("double", "feature_0"),
    #           ColSpec("double", "feature_1"),
    #           # ...
    #       ]),
    #       outputs=Schema([
    #           ColSpec("long", "row_index"),
    #           ColSpec("double", "fail_probability"),
    #           ColSpec("long", "prediction"),
    #           ColSpec("string", "label"),
    #           # ...
    #       ]),
    #   )
    input_example = training_data.features.head(5).copy()
    output_example = pyfunc_model.predict(
        None,
        input_example,
    )
    signature = infer_signature(
        input_example,
        output_example,
    )

    # Persist model binary and metadata to MLflow server.
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=pyfunc_model,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow==3.14.0",
            "numpy==1.26.0",
            "pandas>=2.2,<3",
            "scikit-learn==1.9.0",
        ],
    )

    version_tags = {
        key: str(value)
        for key, value in tracking.model_version_tags(run_id).items()
        if value is not None
    }

    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=config.model_name,
        await_registration_for=config.await_registration_for,
        tags=version_tags,
    )

    return str(registered_model.version)


def _log_training_dataset(
        training_data: PreparedTrainingData,
        tracking: CandidateTrackingRecord,
) -> None:
    metadata = training_data.metadata

    tracked_dataset = mlflow.data.from_pandas(
        training_data.selected_rows,
        name=metadata["dataset_id"],
        targets="actual_value",
        digest=to_mlflow_dataset_digest(metadata["dataset_selection_hash"])
    )

    # Run abc123
    #   └─ Inputs
    #      └─ Dataset
    #         ├─ name: training_abc
    #         ├─ digest: 1234abcd...
    #         ├─ context: training
    #         ├─ target: actual_value
    #         ├─ schema: ...
    #         └─ lineage tags: ...
    mlflow.log_input(
        tracked_dataset,
        context="training",
        tags=tracking.dataset_input_tags,
    )


def _log_tracking_record(
        tracking: CandidateTrackingRecord,
) -> None:
    # if 'with mlflow.start_run(run_name=run_name) as run:' is called
    # all 'mlflow.xxx' knows the run id.

    # tag for searching.
    #   {
    #       "project": "secom-fail-detection",
    #       "stage": "training-dataset-candidate",
    #       "role": "candidate",
    #       "training_dataset_id": "dataset-123",
    #       "training_job_id": "job-456",
    #   }
    mlflow.set_tags(tracking.run_tags)

    # Record hyper parameters and training parameters.
    #   {
    #       "model_name": "RandomForestClassifier",
    #       "n_estimators": 300,
    #       "min_samples_leaf": 3,
    #       "random_state": 42,
    #       "threshold": 0.2,
    #       "imputer_strategy": "median",
    #       "validation_size": 0.2,
    #   }
    mlflow.log_params(tracking.params)

    # {
    #       "validation_f1_1": 0.65,
    #       "validation_recall_1": 0.6,
    #       "validation_precision_1": 0.7,
    #       "validation_pr_auc": 0.72,
    #       "training_sample_count": 800.0,
    #  }
    mlflow.log_metrics(tracking.metrics)

def _log_artifacts(
        *,
        training_data: PreparedTrainingData,
        training_result: RandomForestTrainingResult,
        tracking: CandidateTrackingRecord,
) -> None:
    #   Experiment
    #   └── Run abc123
    #       └── Artifacts
    #           ├── reports/
    #           ├── results/
    #           └── data/
    train_sample_ids = [
        training_data.sample_ids[int(index)]
        for index in training_result.train_indices
    ]
    validation_sample_ids = [
        training_data.sample_ids[int(index)]
        for index in training_result.validation_indices
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        encoding = 'utf-8'

        result_path = tmp_path / "offline_feature_training_threshold_results.csv"
        report_path = tmp_path / "validation_classification_report.json"
        matrix_path = tmp_path / "validation_confusion_matrix.json"
        summary_path = tmp_path / "offline_feature_training_summary.json"
        development_samples_path = tmp_path / "development_sample_ids.txt"
        train_samples_path = tmp_path / "train_sample_ids.txt"
        validation_samples_path = tmp_path / "validation_sample_ids.txt"

        training_result.search_results.to_csv(result_path, index=False)

        report_json = json.dumps(_json_safe(training_result.report), indent=2, ensure_ascii=False)
        report_path.write_text(report_json, encoding=encoding)

        matrix_json = json.dumps(
                {
                    "labels": [
                        NEGATIVE_CLASS,
                        POSITIVE_CLASS,
                    ],
                    "matrix": (
                        training_result
                        .confusion_matrix
                        .tolist()
                    ),
                }, indent=2, ensure_ascii=False,
            )
        matrix_path.write_text(matrix_json, encoding=encoding)

        summary_json = json.dumps(_json_safe(tracking.summary), indent=2, ensure_ascii=False)
        summary_path.write_text(summary_json, encoding=encoding)

        development_samples_path.write_text("\n".join(training_data.sample_ids), encoding=encoding)
        train_samples_path.write_text("\n".join(train_sample_ids), encoding=encoding)
        validation_samples_path.write_text("\n".join(validation_sample_ids), encoding=encoding)

        # results/...
        mlflow.log_artifact(str(result_path), artifact_path="results")

        # reports/...
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(matrix_path), artifact_path="reports")
        mlflow.log_artifact(str(summary_path), artifact_path="reports")

        # data/...
        mlflow.log_artifact(str(development_samples_path), artifact_path="data")
        mlflow.log_artifact(str(train_samples_path), artifact_path="data")
        mlflow.log_artifact(str(validation_samples_path), artifact_path="data")


def _set_candidate_alias(
        *,
        client: MlflowClient,
        model_name: str,
        model_alias: str,
        model_version: str,
) -> None:
    client.set_registered_model_alias(
        name=model_name,
        alias=model_alias,
        version=model_version,
    )

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value
