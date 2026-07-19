"""Kafka configuration helpers."""

import os
from collections.abc import Mapping

ENV_KAFKA_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"
ENV_FEATURE_PATCHES_TOPIC = "FEATURE_PATCHES_TOPIC"
ENV_LABEL_EVENTS_TOPIC = "LABEL_EVENTS_TOPIC"

DEFAULT_LOCAL_KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
DEFAULT_CONTAINER_KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"

DEFAULT_FEATURE_PATCHES_TOPIC = "secom-feature-patches"
DEFAULT_LABEL_EVENTS_TOPIC = "secom-label-events"


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


def resolve_label_events_topic(
        *,
        default: str = DEFAULT_LABEL_EVENTS_TOPIC,
        environ: Mapping[str, str] | None = None,
) -> str:
    return _get_env_value(ENV_LABEL_EVENTS_TOPIC, default, environ=environ)


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
