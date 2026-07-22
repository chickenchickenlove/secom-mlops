"""Serving and model gateway configuration helpers."""

import os

ENV_MODEL_GATEWAY_URL = "MODEL_GATEWAY_URL"
ENV_MODEL_GATEWAY_ADMIN_URL = "MODEL_GATEWAY_ADMIN_URL"
ENV_SERVING_API_URL = "SERVING_API_URL"

DEFAULT_LOCAL_SERVING_API_URL = "http://127.0.0.1:8080"
DEFAULT_LOCAL_MODEL_GATEWAY_URL = "http://127.0.0.1:8080"
DEFAULT_LOCAL_MODEL_GATEWAY_ADMIN_URL = "http://127.0.0.1:18080"

DEFAULT_CONTAINER_MODEL_GATEWAY_URL = "http://model-gateway:8080"
DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL = "http://model-gateway:18080"

RELEASE_RELOAD_MODEL_VERSION_PATH = "/release/admin/reload-model-version"
RELEASE_METADATA_PATH = "/release/metadata"
CANARY_RELOAD_MODEL_VERSION_PATH = "/canary/admin/reload-model-version"
CANARY_METADATA_PATH = "/canary/metadata"
PRODUCTION_METADATA_PATH = "/metadata"
TRAFFIC_POLICY_PATH = "/admin/traffic-policy"


def build_url(base_url: str, path: str = "") -> str:
    if not path:
        return base_url.rstrip("/")
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def resolve_model_gateway_url(
        *,
        default: str = DEFAULT_LOCAL_MODEL_GATEWAY_URL,
) -> str:
    return _get_env_url(ENV_MODEL_GATEWAY_URL, default)


def resolve_model_gateway_admin_url(
        *,
        default: str = DEFAULT_LOCAL_MODEL_GATEWAY_ADMIN_URL,
) -> str:
    return _get_env_url(ENV_MODEL_GATEWAY_ADMIN_URL, default)


def resolve_serving_api_url(
        *,
        default: str = DEFAULT_LOCAL_SERVING_API_URL,
) -> str:
    return _get_env_url(ENV_SERVING_API_URL, default)


def model_gateway_endpoint(
        path: str,
        *,
        default: str = DEFAULT_CONTAINER_MODEL_GATEWAY_URL,
) -> str:
    return build_url(resolve_model_gateway_url(default=default), path)


def model_gateway_admin_endpoint(
        path: str,
        *,
        default: str = DEFAULT_CONTAINER_MODEL_GATEWAY_ADMIN_URL,
) -> str:
    return build_url(resolve_model_gateway_admin_url(default=default), path)


def _get_env_url(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()
