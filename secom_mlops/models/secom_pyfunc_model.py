from __future__ import annotations

from mlflow.pyfunc.utils import pyfunc

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from secom_mlops_common.schemas.secom import (
    MODEL_COLUMNS,
    NUM_FEATURES,
    SNAPSHOT_FEATURE_KEYS,
)


class SECOMFailDetectionPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(
            self,
            model: Any,
            threshold: float,
            model_name: str,
            model_run_id: str,
            positive_class: int = 1,
    ) -> None:
        self.model = model
        self.threshold = float(threshold)
        self.model_name = model_name
        self.model_run_id = model_run_id
        self.positive_class = int(positive_class)

    @pyfunc
    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        features = _to_model_frame(model_input)

        class_order = list(self.model.named_steps["model"].classes_)
        fail_index = class_order.index(self.positive_class)

        fail_probability = self.model.predict_proba(features)[:, fail_index]
        predicted_value = np.where(
            fail_probability >= self.threshold,
            1,
            -1,
        ).astype(int)

        return pd.DataFrame({
            "row_index": list(range(len(features))),
            "fail_probability": fail_probability.astype(float),
            "prediction": predicted_value,
            "label": np.where(predicted_value == 1, "fail", "pass"),
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_run_id": self.model_run_id,
        })


def _to_model_frame(model_input: Any) -> pd.DataFrame:
    frame = model_input.copy() if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)

    if set(MODEL_COLUMNS).issubset(frame.columns):
        frame = frame.loc[:, list(MODEL_COLUMNS)]
    elif set(SNAPSHOT_FEATURE_KEYS).issubset(frame.columns):
        frame = frame.loc[:, list(SNAPSHOT_FEATURE_KEYS)]
        frame.columns = list(MODEL_COLUMNS)
    elif frame.shape[1] == NUM_FEATURES:
        frame.columns = list(MODEL_COLUMNS)
    else:
        raise ValueError(f"Expected {NUM_FEATURES} features, got shape={frame.shape}")

    return frame.astype("float64")
