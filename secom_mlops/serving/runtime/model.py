import mlflow

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRuntime:
    model: mlflow.pyfunc.PyFuncModel
    model_uri: str
    model_name: str
    model_version: str | None
    model_alias: str | None
    model_run_id: str | None
    threshold: float | None
    runtime_slot: str
    loaded_at_utc: str
    reload_request_id: str | None