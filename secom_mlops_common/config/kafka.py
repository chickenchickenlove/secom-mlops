"""Kafka configuration helpers."""

import os
from collections.abc import Mapping

ENV_KAFKA_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"
ENV_FEATURE_PATCHES_TOPIC = "FEATURE_PATCHES_TOPIC"
ENV_FEATURE_STATE_UPDATES_TOPIC = "FEATURE_STATE_UPDATES_TOPIC"
ENV_LABEL_EVENTS_TOPIC = "LABEL_EVENTS_TOPIC"
ENV_PREDICTION_EVENTS_TOPIC = "PREDICTION_EVENTS_TOPIC"
ENV_PREDICTION_EVENT_CLIENT_ID = "PREDICTION_EVENT_CLIENT_ID"
ENV_PREDICTION_EVENT_FLUSH_TIMEOUT_SECONDS = "PREDICTION_EVENT_FLUSH_TIMEOUT_SECONDS"

DEFAULT_LOCAL_KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
DEFAULT_CONTAINER_KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"

DEFAULT_FEATURE_PATCHES_TOPIC = "secom-feature-patches"
DEFAULT_FEATURE_STATE_UPDATES_TOPIC = "secom-feature-state-updates"
DEFAULT_LABEL_EVENTS_TOPIC = "secom-label-events"
DEFAULT_PREDICTION_EVENTS_TOPIC = "secom-prediction-events"

DEFAULT_PREDICTION_EVENT_CLIENT_ID = "serving-api-prediction-event-producer"
DEFAULT_PREDICTION_EVENT_FLUSH_TIMEOUT_SECONDS = 10.0


def resolve_kafka_bootstrap_servers(
        *,
        default: str = DEFAULT_LOCAL_KAFKA_BOOTSTRAP_SERVERS,
        environ: Mapping[str, str] | None = None,
) -> str:
    return _get_env_value(ENV_KAFKA_BOOTSTRAP_SERVERS, default, environ=environ)


def resolve_feature_patches_topic(
        *,
        default: str = DEFAULT_FEATURE_PATCHES_TOPIC,
        environ: Mapping[str, str] | None = None,
) -> str:
    return _get_env_value(ENV_FEATURE_PATCHES_TOPIC, default, environ=environ)


def resolve_feature_state_updates_topic(
        *,
        default: str = DEFAULT_FEATURE_STATE_UPDATES_TOPIC,
        environ: Mapping[str, str] | None = None,
) -> str:
    return _get_env_value(ENV_FEATURE_STATE_UPDATES_TOPIC, default, environ=environ)


def resolve_label_events_topic(
        *,
        default: str = DEFAULT_LABEL_EVENTS_TOPIC,
        environ: Mapping[str, str] | None = None,
) -> str:
    return _get_env_value(ENV_LABEL_EVENTS_TOPIC, default, environ=environ)


def resolve_prediction_events_topic(
        *,
        default: str = DEFAULT_PREDICTION_EVENTS_TOPIC,
        environ: Mapping[str, str] | None = None,
) -> str:
    return _get_env_value(ENV_PREDICTION_EVENTS_TOPIC, default, environ=environ)


def resolve_prediction_event_client_id(
        *,
        default: str = DEFAULT_PREDICTION_EVENT_CLIENT_ID,
        environ: Mapping[str, str] | None = None,
) -> str:
    return _get_env_value(ENV_PREDICTION_EVENT_CLIENT_ID, default, environ=environ)


def resolve_prediction_event_flush_timeout_seconds(
        *,
        default: float = DEFAULT_PREDICTION_EVENT_FLUSH_TIMEOUT_SECONDS,
        environ: Mapping[str, str] | None = None,
) -> float:
    raw_value = _get_env_value(
        ENV_PREDICTION_EVENT_FLUSH_TIMEOUT_SECONDS,
        str(default),
        environ=environ,
    )
    return float(raw_value)


def _get_env_value(
        name: str,
        default: str,
        *,
        environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    value = env.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()
