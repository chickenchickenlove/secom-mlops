"""Database configuration helpers."""

import os

ENV_MONITORING_DATABASE_URL = "MONITORING_DATABASE_URL"

DEFAULT_LOCAL_MONITORING_DATABASE_URL = "postgresql://mlops:mlops@localhost:5432/monitoring"
DEFAULT_CONTAINER_MONITORING_DATABASE_URL = "postgresql://mlops:mlops@postgres:5432/monitoring"


def resolve_monitoring_database_url(
        *,
        default: str = DEFAULT_LOCAL_MONITORING_DATABASE_URL,
) -> str:
    value = os.getenv(ENV_MONITORING_DATABASE_URL)
    if value is None or value.strip() == "":
        return default
    return value.strip()
