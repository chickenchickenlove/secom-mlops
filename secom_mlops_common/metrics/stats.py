"""Statistics helpers shared by monitoring and drift scripts."""

import json
import math
from typing import Any

from secom_mlops_common.schemas.secom import feature_lookup_keys


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    return quantile_from_sorted(sorted_values, q)


def indexed_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * q)
    return float(sorted_values[index])


def quantile_from_sorted(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(sorted_values[int(position)])

    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return float(current - baseline)


def as_metric_value(value: Any) -> float | None:
    if value is None:
        return None

    value = float(value)
    if not math.isfinite(value):
        return None

    return value


def parse_features(raw_features: Any) -> Any:
    if isinstance(raw_features, (dict, list)):
        return raw_features

    if isinstance(raw_features, bytes):
        raw_features = raw_features.decode("utf-8")

    if isinstance(raw_features, str):
        return json.loads(raw_features)

    return raw_features


def feature_value_from_object(features: dict[str, Any], index: int) -> Any:
    for key in feature_lookup_keys(index):
        if key in features:
            return features[key]
    return None


def feature_vector(raw_features: Any, feature_count: int) -> list[float | None]:
    features = parse_features(raw_features)
    values: list[float | None] = []

    for index in range(feature_count):
        raw_value = None

        if isinstance(features, list):
            if index < len(features):
                raw_value = features[index]
        elif isinstance(features, dict):
            raw_value = feature_value_from_object(features, index)

        values.append(as_metric_value(raw_value))

    return values


def first_present(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def empty_feature_stats(feature_count: int) -> list[dict[str, float]]:
    return [
        {"samples": 0, "null_count": 0, "non_null": 0, "sum": 0.0, "sum_sq": 0.0}
        for _ in range(feature_count)
    ]


def update_feature_stats(
        stats: list[dict[str, float]],
        rows: list[dict[str, Any]],
        feature_count: int,
) -> None:
    for row in rows:
        values = feature_vector(row["features_json"], feature_count)

        for index, value in enumerate(values):
            item = stats[index]
            item["samples"] += 1

            if value is None:
                item["null_count"] += 1
                continue

            item["non_null"] += 1
            item["sum"] += value
            item["sum_sq"] += value * value


def feature_mean(stats: dict[str, float]) -> float | None:
    non_null = int(stats["non_null"])
    if non_null == 0:
        return None
    return float(stats["sum"] / non_null)


def feature_std(
        stats: dict[str, float],
        *,
        insufficient_value: float | None = None,
) -> float | None:
    non_null = int(stats["non_null"])
    if non_null < 2:
        return insufficient_value

    variance = (stats["sum_sq"] - ((stats["sum"] * stats["sum"]) / non_null)) / (non_null - 1)
    return float(math.sqrt(max(variance, 0.0)))


def collect_feature_values(
        rows: list[dict[str, Any]],
        feature_count: int,
) -> list[list[float]]:
    values_by_feature: list[list[float]] = [[] for _ in range(feature_count)]

    for row in rows:
        values = feature_vector(row["features_json"], feature_count)
        for index, value in enumerate(values):
            if value is not None:
                values_by_feature[index].append(value)

    return values_by_feature


def ratios_from_counts(counts: list[int]) -> list[float]:
    total = sum(counts)
    if total <= 0:
        return []
    return [float(count / total) for count in counts]
