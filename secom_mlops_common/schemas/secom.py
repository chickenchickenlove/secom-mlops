"""Canonical SECOM feature schema."""

import json
import math
from collections.abc import Mapping
from typing import Any, Literal, TypeVar

NUM_FEATURES = 590
FEATURE_KEYS = tuple(f"f{index:03d}" for index in range(NUM_FEATURES))
FEATURE_KEY_SET = set(FEATURE_KEYS)
MODEL_COLUMNS = tuple(f"feature_{index:03d}" for index in range(NUM_FEATURES))

SNAPSHOT_FEATURE_KEYS = FEATURE_KEYS

FeatureValue = TypeVar("FeatureValue")


def ordered_feature_values(features_json: Mapping[str, FeatureValue]) -> list[FeatureValue]:
    return [features_json[key] for key in FEATURE_KEYS]


def missing_feature_keys(features_json: Mapping[str, object]) -> list[str]:
    return [key for key in FEATURE_KEYS if key not in features_json]


def feature_key(index: int) -> str:
    return FEATURE_KEYS[index]


def model_column(index: int) -> str:
    return MODEL_COLUMNS[index]


def feature_lookup_keys(index: int) -> tuple[str, str, str]:
    return feature_key(index), model_column(index), str(index)


def parse_feature_object(raw_features: Any, *, sample_id: str | None = None) -> dict[str, Any]:
    if isinstance(raw_features, dict):
        return raw_features

    if isinstance(raw_features, bytes):
        raw_features = raw_features.decode("utf-8")

    if isinstance(raw_features, str):
        parsed = json.loads(raw_features)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"features_json must be an object{_feature_error_context(sample_id=sample_id)}")


def normalize_feature_value(
        value: Any,
        *,
        sample_id: str | None = None,
        feature_key: str | None = None,
        non_finite: Literal["none", "allow", "error"] = "none",
) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(_numeric_feature_error(sample_id=sample_id, feature_key=feature_key))

    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
        if non_finite == "none":
            return None
        if non_finite == "allow":
            return number
        if non_finite == "error":
            raise ValueError(_finite_feature_error(sample_id=sample_id, feature_key=feature_key))
        raise ValueError(f"unknown non_finite policy: {non_finite}")

    raise ValueError(_numeric_feature_error(sample_id=sample_id, feature_key=feature_key))


def _numeric_feature_error(*, sample_id: str | None, feature_key: str | None) -> str:
    return (
        "feature value must be numeric or null"
        f"{_feature_error_context(sample_id=sample_id, feature_key=feature_key)}"
    )


def _finite_feature_error(*, sample_id: str | None, feature_key: str | None) -> str:
    return (
        "feature value must be finite"
        f"{_feature_error_context(sample_id=sample_id, feature_key=feature_key)}"
    )


def _feature_error_context(
        *,
        sample_id: str | None = None,
        feature_key: str | None = None,
) -> str:
    parts = []
    if sample_id is not None:
        parts.append(f"sample_id={sample_id}")
    if feature_key is not None:
        parts.append(f"key={feature_key}")
    if not parts:
        return ""
    return f": {' '.join(parts)}"
