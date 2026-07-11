from typing import Literal

from pydantic import BaseModel, Field

from uuid import UUID

from secom_mlops_common.schemas.secom import NUM_FEATURES


class PredictionLog(BaseModel):
    prediction_id: UUID
    request_id: UUID
    sample_id: str = Field(..., pattern=r"^secom-\d{7}$")
    serving_snapshot_id: str = Field(min_length=1, pattern=r"^\S+$")
    snapshot_version: int = Field(ge=1)
    model_run_id: str
    runtime_slot: str | None = None
    predicted_at: float = Field(ge=0) # unix time

    fail_probability: float = Field(ge=0.0, le=1.0)
    predicted_value: Literal[-1, 1]
    predicted_label: Literal["pass", "fail"]
    threshold: float = Field(ge=0.0, le=1.0)

    missing_count: int = Field(ge=0, le=NUM_FEATURES)
    latency_ms: float | None = None


class ActualLabel(BaseModel):
    sample_id: str = Field(..., pattern=r"^secom-\d{7}$")
    actual_value: Literal[-1, 1]
    actual_label: Literal["pass", "fail"]
    labeled_at: float = Field(ge=0) # unix time
