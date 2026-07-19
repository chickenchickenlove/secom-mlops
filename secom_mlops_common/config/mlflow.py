"""Common MLflow and model registry configuration defaults."""

import os
from collections.abc import Mapping

ENV_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_ML_URL = "ML_URL"
ENV_ML_MODEL_NAME = "ML_MODEL_NAME"
ENV_ML_MODEL_ALIAS = "ML_MODEL_ALIAS"
ENV_ML_RUN_ID = "ML_RUN_ID"
ENV_ML_TARGET_MODEL_ALIAS = "ML_TARGET_MODEL_ALIAS"
ENV_ML_MODEL_ROLE = "ML_MODEL_ROLE"
ENV_ML_CANDIDATE_GROUP = "ML_CANDIDATE_GROUP"
ENV_ML_TRAINING_JOB_ID = "ML_TRAINING_JOB_ID"

DEFAULT_LOCAL_MLFLOW_TRACKING_URI = "http://localhost:5100"
DEFAULT_CONTAINER_MLFLOW_TRACKING_URI = "http://mlflow:5100"
DEFAULT_MODEL_NAME = "secom-fail-detector"
DEFAULT_CHAMPION_ALIAS = "champion"
DEFAULT_CANDIDATE_ALIAS = "candidate"

MODEL_ROLE_CANDIDATE = "candidate"
MODEL_ROLE_CHAMPION = "champion"
MODEL_ROLES = frozenset({MODEL_ROLE_CANDIDATE, MODEL_ROLE_CHAMPION})


def resolve_tracking_uri(
        argument_value: str | None = None,
        *,
        default: str = DEFAULT_LOCAL_MLFLOW_TRACKING_URI,
        environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    return _first_non_empty(
        argument_value,
        env.get(ENV_MLFLOW_TRACKING_URI),
        env.get(ENV_ML_URL),
        default,
    )


def resolve_model_name(argument_value: str | None = None,
                       *,
                       default: str = DEFAULT_MODEL_NAME,
                       environ: Mapping[str, str] | None = None,
                       ) -> str:
    env = os.environ if environ is None else environ
    return _first_non_empty(
        argument_value,
        env.get(ENV_ML_MODEL_NAME),
        default
    )


def resolve_model_alias(argument_value: str | None = None,
                        *,
                        env_name: str = ENV_ML_MODEL_ALIAS,
                        default: str = DEFAULT_CHAMPION_ALIAS,
                        environ: Mapping[str, str] | None = None,
                        ) -> str:
    env = os.environ if environ is None else environ
    return _first_non_empty(argument_value, env.get(env_name), default)


def resolve_model_role(argument_value: str | None = None,
                       *,
                       default: str = MODEL_ROLE_CANDIDATE,
                       environ: Mapping[str, str] | None = None,
                       ) -> str:
    env = os.environ if environ is None else environ
    return _first_non_empty(argument_value, env.get(ENV_ML_MODEL_ROLE), default)


def get_env_value(
        env_name: str,
        *,
        environ: Mapping[str, str] | None = None,
) -> str | None:
    env = os.environ if environ is None else environ
    return env.get(env_name)


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value:
            return value

    raise ValueError("at least one non-empty value is required")
