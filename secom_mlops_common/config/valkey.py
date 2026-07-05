"""Valkey configuration helpers."""

import os
from collections.abc import Mapping

ENV_VALKEY_URL = "VALKEY_URL"
ENV_VALKEY_HOST = "VALKEY_HOST"
ENV_VALKEY_PORT = "VALKEY_PORT"
ENV_VALKEY_DATABASE = "VALKEY_DATABASE"
ENV_VALKEY_TIMEOUT_SECONDS = "VALKEY_TIMEOUT_SECONDS"
ENV_VALKEY_KEY_PREFIX = "VALKEY_KEY_PREFIX"

DEFAULT_LOCAL_VALKEY_HOST = "127.0.0.1"
DEFAULT_CONTAINER_VALKEY_HOST = "valkey"
DEFAULT_VALKEY_PORT = 6379
DEFAULT_VALKEY_DATABASE = 0
DEFAULT_VALKEY_TIMEOUT_SECONDS = 2.0
DEFAULT_VALKEY_KEY_PREFIX = "online_feature_snapshot"


def resolve_valkey_url(
        *,
        default: str | None = None,
        environ: Mapping[str, str] | None = None,
) -> str | None:
    return _get_optional_env_value(ENV_VALKEY_URL, default, environ=environ)


def resolve_valkey_host(
        *,
        default: str = DEFAULT_LOCAL_VALKEY_HOST,
        environ: Mapping[str, str] | None = None,
) -> str:
    return _get_env_value(ENV_VALKEY_HOST, default, environ=environ)


def resolve_valkey_port(
        *,
        default: int = DEFAULT_VALKEY_PORT,
        environ: Mapping[str, str] | None = None,
) -> int:
    return int(_get_env_value(ENV_VALKEY_PORT, str(default), environ=environ))


def resolve_valkey_database(
        *,
        default: int = DEFAULT_VALKEY_DATABASE,
        environ: Mapping[str, str] | None = None,
) -> int:
    return int(_get_env_value(ENV_VALKEY_DATABASE, str(default), environ=environ))


def resolve_valkey_timeout_seconds(
        *,
        default: float = DEFAULT_VALKEY_TIMEOUT_SECONDS,
        environ: Mapping[str, str] | None = None,
) -> float:
    return float(_get_env_value(ENV_VALKEY_TIMEOUT_SECONDS, str(default), environ=environ))


def resolve_valkey_key_prefix(
        *,
        default: str = DEFAULT_VALKEY_KEY_PREFIX,
        environ: Mapping[str, str] | None = None,
) -> str:
    return _get_env_value(ENV_VALKEY_KEY_PREFIX, default, environ=environ)


def _get_env_value(
        name: str,
        default: str,
        *,
        environ: Mapping[str, str] | None = None,
) -> str:
    value = _get_optional_env_value(name, None, environ=environ)
    if value is None:
        return default
    return value


def _get_optional_env_value(
        name: str,
        default: str | None,
        *,
        environ: Mapping[str, str] | None = None,
) -> str | None:
    env = os.environ if environ is None else environ
    value = env.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()
