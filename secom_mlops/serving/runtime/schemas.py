from pydantic import BaseModel, Field

class InvocationRequest(BaseModel):
    inputs: list[list[float | None]] = Field(..., min_length=1)


class ThresholdReloadRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)
    request_id: str | None = None
    expected_model_name: str | None = None
    expected_model_version: str | None = None
    expected_run_id: str | None = None


class ModelVersionReloadRequest(BaseModel):
    model_version: str
    model_name: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    request_id: str | None = None
    expected_run_id: str | None = None
    expected_current_model_version: str | None = None
    expected_current_run_id: str | None = None