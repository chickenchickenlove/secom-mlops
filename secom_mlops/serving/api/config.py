from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingApiConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    kafka_bootstrap_servers: str = Field(
        default="127.0.0.1:9092",
        min_length=1,
    )
    prediction_events_topic: str = Field(
        default="secom-prediction-events",
        min_length=1,
    )
    prediction_event_client_id: str = Field(
        default="serving-api-prediction-event-producer",
        min_length=1,
    )
    prediction_event_flush_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
    )

    valkey_url: str | None = None
    valkey_host: str = Field(default="127.0.0.1", min_length=1)
    valkey_port: int = Field(default=6379, ge=1, le=65535)
    valkey_database: int = Field(default=0, ge=0)
    valkey_timeout_seconds: float = Field(default=2.0, gt=0)
    valkey_key_prefix: str = Field(
        default="online_feature_snapshot",
        min_length=1,
    )

    predictor_slot: Literal["release", "canary", "shadow"]

    model_runtime_url: str = Field(min_length=1)
    model_runtime_path: str = "/invocations"
    shadow_model_runtime_url: str | None = Field(default=None, min_length=1)
    shadow_model_runtime_path: str = "/invocations"
    model_runtime_timeout_seconds: float = Field(default=10.0, gt=0)

    model_batch_max_size: int = Field(default=16, ge=1)
    model_batch_max_wait_ms: float = Field(default=20.0, ge=0)
    model_batch_queue_max_size: int = Field(default=1024, ge=1)
    model_batch_queue_timeout_ms: float = Field(default=2000.0, gt=0)
    model_batch_response_timeout_seconds: float = Field(default=30.0, gt=0)

    prediction_event_queue_max_size: int = Field(default=4096, ge=1)
    prediction_event_batch_max_size: int = Field(default=100, ge=1)
    prediction_event_batch_max_wait_ms: float = Field(default=100.0, ge=0)
    predict_partial_retry_after_ms: int = Field(default=200, ge=1)

    @field_validator("model_runtime_path", "shadow_model_runtime_path")
    @classmethod
    def validate_invocation_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("invocation path must start with '/'")
        return value

    @property
    def model_batch_max_wait_seconds(self) -> float:
        return self.model_batch_max_wait_ms / 1000

    @property
    def model_batch_queue_timeout_seconds(self) -> float:
        return self.model_batch_queue_timeout_ms / 1000

    @property
    def prediction_event_batch_max_wait_seconds(self) -> float:
        return self.prediction_event_batch_max_wait_ms / 1000
