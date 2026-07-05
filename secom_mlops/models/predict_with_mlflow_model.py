import mlflow
import numpy as np
import pandas as pd
from pathlib import Path

from secom_mlops_common.config.mlflow import ENV_ML_RUN_ID, get_env_value, resolve_tracking_uri
from secom_mlops_common.schemas.secom import MODEL_COLUMNS, NUM_FEATURES

DEFAULT_ML_RUN_ID = "7b0905cb6c45453dbf9d08281f7bdb2e"


def validate_features(df):
    if df.shape[1] != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, got {df.shape[1]}")


def with_column_names(df):
    df.columns = MODEL_COLUMNS
    return df

def predict(df, model, threshold: float):
    validate_features(df)
    df = with_column_names(df)

    class_order = list(model.named_steps["model"].classes_)
    fail_index = class_order.index(1)

    y_proba = model.predict_proba(df)
    fail_proba = y_proba[:, fail_index]
    prediction = np.where(fail_proba >= threshold, 1, -1)
    return as_response(fail_proba, prediction)

def as_response(fail_proba, prediction):
    return pd.DataFrame({
        "fail_probability": fail_proba,
        "prediction": prediction,
        "label": np.where(prediction == 1, "fail", "pass")
    })


def load_model(run_id):
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")

def load_threshold(run_id):
    run = mlflow.get_run(run_id)
    return float(run.data.params["threshold"])


def bootstrap_model():
    ml_flow_url = resolve_tracking_uri()
    ml_flow_run_id = get_env_value(ENV_ML_RUN_ID) or DEFAULT_ML_RUN_ID

    mlflow.set_tracking_uri(ml_flow_url)
    model = load_model(ml_flow_run_id)
    threshold = load_threshold(ml_flow_run_id)

    return RandomForestSECOMPredictor(model, threshold, ml_flow_run_id)


class RandomForestSECOMPredictor:

    def __init__(self, model, threshold, run_id):
        self.model = model
        self.threshold = threshold
        self.run_id = run_id

        class_order = list(self.model.named_steps["model"].classes_)
        self.fail_index = class_order.index(1)

    def predict(self, df):
        self._validate_features(df)
        df = self._with_column_names(df)

        y_proba = self.model.predict_proba(df)
        fail_proba = y_proba[:, self.fail_index]
        prediction = np.where(fail_proba >= self.threshold, 1, -1)
        return self._as_batch_response(fail_proba, prediction)

    def _validate_features(self, df):
        if df.shape[1] != NUM_FEATURES:
            raise ValueError(f"Expected {NUM_FEATURES} features, got {df.shape[1]}")

    def _with_column_names(self, df):
        df.columns = MODEL_COLUMNS
        return df

    def _as_response(self, fail_proba, prediction):
        return {
            "fail_probability": fail_proba,
            "prediction": prediction,
            "label": np.where(prediction == 1, "fail", "pass")
        }

    def _as_batch_response(self, fail_proba, prediction):
        return [
            {
                    "row_index": idx,
                    "fail_probability": float(prob),
                    "prediction": int(pred),
                    "label": "fail" if int(pred) == 1 else "pass",
                    "threshold": self.threshold
            } for idx, (prob, pred) in enumerate(zip(fail_proba, prediction))
        ]

def main():
    mlflow.set_tracking_uri(resolve_tracking_uri())
    project_root = Path(__file__).resolve().parents[2]
    run_id = get_env_value(ENV_ML_RUN_ID) or DEFAULT_ML_RUN_ID

    model = load_model(run_id)
    threshold = load_threshold(run_id)

    sample = pd.read_csv(
        project_root / "data" / "raw" / "secom.data",
        sep=r"\s+",
        header=None,
        na_values="NaN",
    ).head(10)
    print(predict(sample, model, threshold))

# if __name__ == "__main__":
#     main()
