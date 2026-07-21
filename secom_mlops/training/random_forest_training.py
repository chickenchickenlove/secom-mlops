from typing import Any, Sequence
from dataclasses import dataclass


import numpy as np
import pandas as pd
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

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1


@dataclass(frozen=True)
class RandomForestTrainingConfig:
    n_estimators: Sequence[int]
    min_samples_leaf: Sequence[int]
    thresholds: Sequence[float]
    validation_size: float
    random_state: int


@dataclass
class RandomForestTrainingResult:
    model: Pipeline
    best_row: dict[str, Any]
    search_results: pd.DataFrame
    report: dict[str, Any]
    confusion_matrix: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray


def build_model(n_estimators: int, min_samples_leaf: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def validate_inputs_or_throws(features, targets):
    if len(features) != len(targets):
        raise ValueError(
          "features and targets must have the same number of rows: "
          f"features={len(features)} targets={len(targets)}"
        )

def train_random_forest(
        features: pd.DataFrame,
        targets: pd.Series,
        config: RandomForestTrainingConfig
) -> RandomForestTrainingResult:
    validate_inputs_or_throws(features, targets)

    train_indices, validation_indices = split_indices(
        targets,
        validation_size=config.validation_size,
        random_state=config.random_state)

    x_train = features.iloc[train_indices].copy()
    y_train = targets.iloc[train_indices].copy()
    x_validation = features.iloc[validation_indices].copy()
    y_validation = targets.iloc[validation_indices].copy()

    # Select hyperparameters and a threshold using the validation split.
    # To select model, we split dataset into train and validation.
    # During this sequence, we can find best model.
    search_results, best_row, best_report, best_matrix = select_model(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        n_estimators_values=config.n_estimators,
        min_samples_leaf_values=config.min_samples_leaf,
        threshold_values=config.thresholds,
        random_state=config.random_state,
    )

    # Refit a fresh model on the complete development dataset using the selected hyperparameters.
    # When best model is selected, Then we can build a model, and train it.
    # In this case, we use all samples for training.
    final_model = build_model(
        n_estimators=int(best_row["n_estimators"]),
        min_samples_leaf=int(best_row["min_samples_leaf"]),
        random_state=config.random_state,
    )
    final_model.fit(features, targets)

    return RandomForestTrainingResult(
        model=final_model,
        best_row=best_row,
        search_results=search_results,
        report=best_report,
        confusion_matrix=best_matrix,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def select_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame,
        y_validation: pd.Series,
        n_estimators_values: Sequence[int],
        min_samples_leaf_values: Sequence[int],
        threshold_values: Sequence[float],
        random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], np.ndarray]:
    rows: list[dict[str, Any]] = []
    reports: dict[tuple[int, int, float], dict[str, Any]] = {}
    matrices: dict[tuple[int, int, float], np.ndarray] = {}

    for n_estimators in n_estimators_values:
        for min_samples_leaf in min_samples_leaf_values:
            model = build_model(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
            model.fit(x_train, y_train)

            class_order = list(model.named_steps["model"].classes_)
            positive_index = class_order.index(POSITIVE_CLASS)
            fail_probability = model.predict_proba(x_validation)[:, positive_index]

            for threshold in threshold_values:
                metrics, report, matrix = evaluate_predictions(
                    y_true=y_validation,
                    fail_probability=fail_probability,
                    threshold=threshold,
                )
                key = (n_estimators, min_samples_leaf, threshold)
                reports[key] = report
                matrices[key] = matrix
                rows.append({
                    "n_estimators": n_estimators,
                    "min_samples_leaf": min_samples_leaf,
                    "threshold": threshold,
                    **metrics,
                })

    results = pd.DataFrame(rows)
    best_row = results.sort_values(
        by=["f1_1", "recall_1", "precision_1", "pr_auc", "balanced_accuracy"],
        ascending=False,
    ).iloc[0].to_dict()

    best_key = (
        int(best_row["n_estimators"]),
        int(best_row["min_samples_leaf"]),
        float(best_row["threshold"]),
    )
    return results, best_row, reports[best_key], matrices[best_key]


def evaluate_predictions(
        y_true: pd.Series,
        fail_probability: np.ndarray,
        threshold: float,
) -> tuple[dict[str, float], dict[str, Any], np.ndarray]:
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

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_1": float(report["1"]["precision"]),
        "recall_1": float(report["1"]["recall"]),
        "f1_1": float(report["1"]["f1-score"]),
        "pr_auc": float(
            average_precision_score(
                y_true,
                fail_probability,
                pos_label=POSITIVE_CLASS,
            )
        ),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }

    return metrics, report, matrix

def split_indices(
        y: pd.Series,
        validation_size: float,
        random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Use indices function instead of train_test_split(features, target, ...)
    # Because we should track which samples are used for train, test.
    indices = np.arange(len(y))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=y,
    )
    return train_indices, validation_indices