from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRuntimeConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5100",
        min_length=1,
        validation_alias=AliasChoices("MLFLOW_TRACKING_URI", "ML_URL"),
    )
    ml_model_name: str = Field(
        default="secom-fail-detector",
        min_length=1,
    )
    ml_model_alias: str = Field(default="champion", min_length=1)
    ml_model_version: str | None = None
    ml_model_uri: str | None = None
    model_runtime_slot: Literal["release", "canary", "shadow"] = "release"
    monitoring_database_url: str = Field(
        default="postgresql://mlops:mlops@localhost:5432/monitoring",
        min_length=1,
    )
