from pydantic import BaseModel, Field
from secom_mlops_common.schemas.secom import NUM_FEATURES

SAMPLE_ID_PATTERN = r"^secom-\d{7}$"

class PredictRow(BaseModel):
    sample_id: str = Field(..., pattern=SAMPLE_ID_PATTERN)
    features: list[float | None] = Field(..., min_length=NUM_FEATURES, max_length=NUM_FEATURES)


class BatchPredictRequest(BaseModel):
    rows: list[PredictRow] = Field(..., min_length=1)


class PredictByIdRequest(BaseModel):
    sample_id: str = Field(..., pattern=SAMPLE_ID_PATTERN)


class PredictionItem(BaseModel):
    prediction_id: str
    request_id: str
    sample_id: str
    row_index: int
    fail_probability: float
    prediction: int
    label: str
    threshold: float
    model_uri: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    model_run_id: str
    runtime_slot: str | None = None


class BatchPredictResponse(BaseModel):
    request_id: str
    predictions: list[PredictionItem]


class PredictByIdResponse(BaseModel):
    prediction_id: str
    request_id: str
    sample_id: str
    serving_snapshot_id: str
    snapshot_version: int
    feature_hash: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    snapshot_time: float
    feature_count: int
    missing_count: int
    fail_probability: float
    prediction: int
    label: str
    threshold: float
    model_uri: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_alias: str | None = None
    model_run_id: str
    runtime_slot: str | None = None