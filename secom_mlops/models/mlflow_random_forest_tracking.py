import json
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from secom_mlops.models.secom_pyfunc_model import SECOMFailDetectionPyfunc
from secom_mlops_common.config.mlflow import (
    DEFAULT_CHAMPION_ALIAS,
    ENV_ML_CANDIDATE_GROUP,
    ENV_ML_MODEL_ROLE,
    ENV_ML_TRAINING_JOB_ID,
    MODEL_ROLE_CANDIDATE,
    MODEL_ROLES,
    get_env_value,
    resolve_model_alias,
    resolve_model_name,
    resolve_tracking_uri,
)
from secom_mlops_common.logging import configure_logging, get_logger
from secom_mlops_common.schemas.secom import MODEL_COLUMNS, NUM_FEATURES

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RANDOM_STATE = 42
MAX_DEVELOPMENT_SAMPLES = 1000
VALIDATION_SIZE = 0.2
DEVELOPMENT_SAMPLE_SELECTION = "first_raw_rows_in_source_order"
POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1

MODEL_REGISTRY_NAME = resolve_model_name()
MODEL_REGISTRY_ALIAS = resolve_model_alias(default=DEFAULT_CHAMPION_ALIAS)
MODEL_VERSION_ROLE = get_env_value(ENV_ML_MODEL_ROLE)
MODEL_CANDIDATE_GROUP = get_env_value(ENV_ML_CANDIDATE_GROUP)
MODEL_TRAINING_JOB_ID = get_env_value(ENV_ML_TRAINING_JOB_ID)
MODEL_TRAIN_SOURCE = os.getenv("ML_TRAIN_SOURCE", "raw_secom")

def load_data():
    features = pd.read_csv(
        RAW_DATA_DIR / "secom.data",
        sep=r"\s+",
        header=None,
        na_values="NaN",
    )

    labels = pd.read_csv(
        RAW_DATA_DIR / "secom_labels.data",
        sep=r"\s+",
        header=None,
        na_values="NaN",
    )

    if features.shape[1] != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, got {features.shape[1]}")
    features.columns = MODEL_COLUMNS
    labels.columns = ["success_label", "date"]

    X = features
    y = labels["success_label"]

    return X, y


def build_model(n_estimators, min_samples_leaf):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate(y_true, fail_probability, threshold):
    y_pred = np.where(fail_probability >= threshold, POSITIVE_CLASS, NEGATIVE_CLASS)

    report = classification_report(
        y_true,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_1": float(report["1"]["precision"]),
        "recall_1": float(report["1"]["recall"]),
        "f1_1": float(report["1"]["f1-score"]),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }, report, matrix


def log_artifacts(report, matrix):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        report_path = tmp_path / "classification_report.json"
        matrix_path = tmp_path / "confusion_matrix.json"

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        matrix_path.write_text(
            json.dumps(
                {
                    "labels": [NEGATIVE_CLASS, POSITIVE_CLASS],
                    "matrix": matrix.tolist(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(matrix_path), artifact_path="reports")


def find_registered_model_version(
        client: MlflowClient,
        model_name: str,
        run_id: str,
):
    versions = [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if version.run_id == run_id
    ]

    if not versions:
        raise RuntimeError(
            f"registered model version not found: model_name={model_name} run_id={run_id}"
        )

    return max(versions, key=lambda version: int(version.version))


def validate_model_metadata() -> None:
    if MODEL_VERSION_ROLE is None:
        return

    if MODEL_VERSION_ROLE not in MODEL_ROLES:
        raise ValueError(
            "ML_MODEL_ROLE must be one of: candidate, champion "
            f"got={MODEL_VERSION_ROLE}"
        )

    if MODEL_VERSION_ROLE == MODEL_ROLE_CANDIDATE:
        missing = [
            name
            for name, value in [
                ("ML_CANDIDATE_GROUP", MODEL_CANDIDATE_GROUP),
                ("ML_TRAINING_JOB_ID", MODEL_TRAINING_JOB_ID),
            ]
            if value is None or value.strip() == ""
        ]

        if missing:
            raise ValueError(
                "candidate training requires "
                + ", ".join(missing)
            )


def main():
    configure_logging()
    validate_model_metadata()
    mlflow.set_tracking_uri(resolve_tracking_uri())
    mlflow.set_experiment("secom-fail-detection")

    X, y = load_data()

    if len(X) < MAX_DEVELOPMENT_SAMPLES:
        raise ValueError(
            "not enough raw rows for champion development cohort: "
            f"required={MAX_DEVELOPMENT_SAMPLES} actual={len(X)}"
        )

    # The raw SECOM rows are time-ordered. Keep the first 1,000 rows as the
    # fixed Champion development cohort so the remaining rows stay outside it.
    X_development = X.iloc[:MAX_DEVELOPMENT_SAMPLES].copy()
    y_development = y.iloc[:MAX_DEVELOPMENT_SAMPLES].copy()

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        stratify=y_development,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    rows = []

    # for n_estimators in [100, 300, 500, 700]:
    #     for min_samples_leaf in [1, 3, 5, 10]:
    for n_estimators in [100, 300]:
        for min_samples_leaf in [1, 3]:
            model = build_model(n_estimators, min_samples_leaf)
            model.fit(X_train, y_train)

            class_order = model.named_steps["model"].classes_
            positive_index = list(class_order).index(POSITIVE_CLASS)
            fail_probability = model.predict_proba(X_validation)[
                :, positive_index
            ]

            pr_auc = float(
                average_precision_score(
                    y_validation,
                    fail_probability,
                    pos_label=POSITIVE_CLASS,
                )
            )

            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                metrics, report, matrix = evaluate(
                    y_true=y_validation,
                    fail_probability=fail_probability,
                    threshold=threshold,
                )
                metrics["pr_auc"] = pr_auc

                params = {
                    "model_name": "RandomForestClassifier",
                    "n_estimators": n_estimators,
                    "min_samples_leaf": min_samples_leaf,
                    "class_weight": "balanced",
                    "random_state": RANDOM_STATE,
                    "development_sample_limit": MAX_DEVELOPMENT_SAMPLES,
                    "development_sample_selection": DEVELOPMENT_SAMPLE_SELECTION,
                    "validation_size": VALIDATION_SIZE,
                    "stratify": True,
                    "imputer_strategy": "median",
                    "threshold": threshold,
                    "positive_class": POSITIVE_CLASS,
                }

                run_name = (
                    f"rf_n{n_estimators}"
                    f"_leaf{min_samples_leaf}"
                    f"_th{threshold}"
                )

                with mlflow.start_run(run_name=run_name):
                    mlflow.set_tag("project", "secom-fail-detection")
                    mlflow.set_tag("stage", "baseline")
                    mlflow.set_tag("purpose", "random_forest_threshold_comparison")

                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    log_artifacts(report, matrix)

                rows.append({**params, **metrics})

    result_df = pd.DataFrame(rows)
    best_row = result_df.sort_values(
        by=["f1_1", "recall_1", "precision_1"],
        ascending=False,
    ).iloc[0]

    best_model = build_model(
        n_estimators=int(best_row["n_estimators"]),
        min_samples_leaf=int(best_row["min_samples_leaf"]),
    )
    best_model.fit(X_development, y_development)

    with mlflow.start_run(run_name="selected_random_forest_baseline") as run:
        mlflow.set_tag("project", "secom-fail-detection")
        mlflow.set_tag("stage", "selected-baseline")
        mlflow.set_tag(
            "selection_reason",
            "best fail-class f1 under unknown false-positive and false-negative costs",
        )

        selected_params = {
            "model_name": "RandomForestClassifier",
            "registered_model_name": MODEL_REGISTRY_NAME,
            "registered_model_alias": MODEL_REGISTRY_ALIAS,
            "n_estimators": int(best_row["n_estimators"]),
            "min_samples_leaf": int(best_row["min_samples_leaf"]),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "development_sample_limit": MAX_DEVELOPMENT_SAMPLES,
            "development_sample_selection": DEVELOPMENT_SAMPLE_SELECTION,
            "validation_size": VALIDATION_SIZE,
            "stratify": True,
            "imputer_strategy": "median",
            "threshold": float(best_row["threshold"]),
            "positive_class": POSITIVE_CLASS,
            "final_fit_scope": "complete_development_cohort",
            "final_evaluation_source": "serving_prediction_decision_gate",
        }

        selected_metrics = {
            "accuracy": float(best_row["accuracy"]),
            "balanced_accuracy": float(best_row["balanced_accuracy"]),
            "precision_1": float(best_row["precision_1"]),
            "recall_1": float(best_row["recall_1"]),
            "f1_1": float(best_row["f1_1"]),
            "pr_auc": float(best_row["pr_auc"]),
            "tn": float(best_row["tn"]),
            "fp": float(best_row["fp"]),
            "fn": float(best_row["fn"]),
            "tp": float(best_row["tp"]),
            "validation_accuracy": float(best_row["accuracy"]),
            "validation_balanced_accuracy": float(best_row["balanced_accuracy"]),
            "validation_precision_1": float(best_row["precision_1"]),
            "validation_recall_1": float(best_row["recall_1"]),
            "validation_f1_1": float(best_row["f1_1"]),
            "validation_pr_auc": float(best_row["pr_auc"]),
            "training_sample_count": float(len(y_train)),
            "validation_sample_count": float(len(y_validation)),
            "final_fit_sample_count": float(len(y_development)),
            "training_fail_count": float((y_train == POSITIVE_CLASS).sum()),
            "validation_fail_count": float((y_validation == POSITIVE_CLASS).sum()),
            "final_fit_fail_count": float(
                (y_development == POSITIVE_CLASS).sum()
            ),
        }

        mlflow.log_params(selected_params)
        mlflow.log_metrics(selected_metrics)

        pyfunc_model = SECOMFailDetectionPyfunc(
            model=best_model,
            threshold=float(best_row["threshold"]),
            model_name=MODEL_REGISTRY_NAME,
            model_run_id=run.info.run_id,
            positive_class=POSITIVE_CLASS,
        )

        input_example = X_development.head(5).copy()
        output_example = pyfunc_model.predict(None, input_example)
        signature = infer_signature(input_example, output_example)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc_model,
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_REGISTRY_NAME,
            await_registration_for=300,
            pip_requirements=[
                "mlflow==3.14.0",
                "numpy==1.26.0",
                "pandas>=2.2,<3",
                "scikit-learn==1.9.0",
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = Path(tmp_dir) / "threshold_results.csv"
            result_df.to_csv(result_path, index=False)
            mlflow.log_artifact(str(result_path), artifact_path="results")

        client = MlflowClient()
        model_version = find_registered_model_version(
            client=client,
            model_name=MODEL_REGISTRY_NAME,
            run_id=run.info.run_id,
        )

        version_tags = {
            "registered_model_alias": MODEL_REGISTRY_ALIAS,
            "source_run_id": run.info.run_id,
            "train_source": MODEL_TRAIN_SOURCE,
            "development_sample_selection": DEVELOPMENT_SAMPLE_SELECTION,
            "development_sample_count": str(len(y_development)),
            "final_fit_sample_count": str(len(y_development)),
        }

        if MODEL_VERSION_ROLE:
            version_tags["role"] = MODEL_VERSION_ROLE
        if MODEL_CANDIDATE_GROUP:
            version_tags["candidate_group"] = MODEL_CANDIDATE_GROUP
        if MODEL_TRAINING_JOB_ID:
            version_tags["training_job_id"] = MODEL_TRAINING_JOB_ID

        for key, value in version_tags.items():
            client.set_model_version_tag(
                MODEL_REGISTRY_NAME,
                model_version.version,
                key,
                str(value),
            )

        client.set_registered_model_alias(
            MODEL_REGISTRY_NAME,
            MODEL_REGISTRY_ALIAS,
            model_version.version,
        )

        logger.info(
            "registered_selected_model "
            "name=%s "
            "alias=%s "
            "version=%s "
            "run_id=%s",
            MODEL_REGISTRY_NAME,
            MODEL_REGISTRY_ALIAS,
            model_version.version,
            run.info.run_id,
        )

    logger.info("MLflow tracking complete.")
    logger.info("Best baseline:\n%s", best_row.to_string())


if __name__ == "__main__":
    main()
