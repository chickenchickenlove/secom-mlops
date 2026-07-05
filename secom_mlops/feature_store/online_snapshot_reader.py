import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import valkey

from secom_mlops_common.config.valkey import (
    resolve_valkey_database,
    resolve_valkey_host,
    resolve_valkey_key_prefix,
    resolve_valkey_port,
    resolve_valkey_timeout_seconds,
    resolve_valkey_url,
)
from secom_mlops_common.schemas.secom import FEATURE_KEYS, NUM_FEATURES, normalize_feature_value


class OnlineFeatureSnapshotNotFound(Exception):
    pass


class InvalidOnlineFeatureSnapshot(ValueError):
    pass


@dataclass(frozen=True)
class OnlineFeatureSnapshot:
    serving_snapshot_id: str
    sample_id: str
    snapshot_time: float
    snapshot_status: str
    feature_count: int
    missing_count: int
    is_complete: bool
    features_json: dict[str, float | None]

    @property
    def values(self) -> list[float | None]:
        return [self.features_json[key] for key in FEATURE_KEYS]


class OnlineFeatureSnapshotStore:
    def __init__(
            self,
            client: valkey.Valkey | None = None,
            key_prefix: str | None = None,
    ) -> None:
        self._client = client or _build_default_client()
        self._key_prefix = key_prefix or resolve_valkey_key_prefix()

    def load(self, sample_id: str) -> OnlineFeatureSnapshot:
        raw = self._client.get(f"{self._key_prefix}:{sample_id}")
        if raw is None:
            raise OnlineFeatureSnapshotNotFound(sample_id)

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        return _parse_snapshot(str(raw), expected_sample_id=sample_id)

    def close(self) -> None:
        self._client.close()


def online_snapshot_to_dataframe(snapshot: OnlineFeatureSnapshot) -> pd.DataFrame:
    if not snapshot.is_complete:
        raise ValueError(f"incomplete snapshot cannot be used for model input: {snapshot.sample_id}")

    return pd.DataFrame(
        [snapshot.values],
        columns=list(FEATURE_KEYS),
        dtype="float64",
    )


def _build_default_client() -> valkey.Valkey:
    timeout = resolve_valkey_timeout_seconds()

    valkey_url = resolve_valkey_url()
    if valkey_url:
        return valkey.Valkey.from_url(
            valkey_url,
            decode_responses=True,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
        )

    return valkey.Valkey(
        host=resolve_valkey_host(),
        port=resolve_valkey_port(),
        db=resolve_valkey_database(),
        decode_responses=True,
        socket_timeout=timeout,
        socket_connect_timeout=timeout,
    )


def _parse_snapshot(raw: str, expected_sample_id: str) -> OnlineFeatureSnapshot:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise InvalidOnlineFeatureSnapshot("snapshot value is not valid JSON") from error

    if not isinstance(payload, dict):
        raise InvalidOnlineFeatureSnapshot("snapshot value must be a JSON object")

    sample_id = _required_str(payload, "sample_id")
    if sample_id != expected_sample_id:
        raise InvalidOnlineFeatureSnapshot(
            f"snapshot sample_id mismatch: key={expected_sample_id} payload={sample_id}"
        )

    serving_snapshot_id = _required_str(payload, "serving_snapshot_id")
    snapshot_time = _required_float(payload, "snapshot_time")
    snapshot_status = _required_str(payload, "snapshot_status")
    feature_count = _required_int(payload, "feature_count")
    missing_count = _required_int(payload, "missing_count")
    is_complete = _required_bool(payload, "is_complete")

    if snapshot_status not in {"partial", "complete"}:
        raise InvalidOnlineFeatureSnapshot(f"invalid snapshot_status: {snapshot_status}")

    if feature_count < 0 or feature_count > NUM_FEATURES:
        raise InvalidOnlineFeatureSnapshot(f"feature_count out of range: {feature_count}")

    if missing_count < 0 or missing_count > NUM_FEATURES:
        raise InvalidOnlineFeatureSnapshot(f"missing_count out of range: {missing_count}")

    if is_complete != (feature_count == NUM_FEATURES):
        raise InvalidOnlineFeatureSnapshot(
            f"is_complete must match feature_count == {NUM_FEATURES}"
        )

    features = payload.get("features")
    if not isinstance(features, dict):
        raise InvalidOnlineFeatureSnapshot("features must be a JSON object")

    if len(features) != NUM_FEATURES:
        raise InvalidOnlineFeatureSnapshot(
            f"features must contain {NUM_FEATURES} keys: {len(features)}"
        )

    missing_keys = [key for key in FEATURE_KEYS if key not in features]
    if missing_keys:
        raise InvalidOnlineFeatureSnapshot(
            f"features is missing canonical keys: missing_count={len(missing_keys)}"
        )

    normalized_features = {
        key: _normalize_feature_value(features[key], sample_id, key)
        for key in FEATURE_KEYS
    }

    if sum(value is None for value in normalized_features.values()) != missing_count:
        raise InvalidOnlineFeatureSnapshot("missing_count does not match null feature count")

    return OnlineFeatureSnapshot(
        serving_snapshot_id=serving_snapshot_id,
        sample_id=sample_id,
        snapshot_time=snapshot_time,
        snapshot_status=snapshot_status,
        feature_count=feature_count,
        missing_count=missing_count,
        is_complete=is_complete,
        features_json=normalized_features,
    )


def _normalize_feature_value(
        value: Any,
        sample_id: str,
        feature_key: str,
) -> float | None:
    try:
        return normalize_feature_value(
            value,
            sample_id=sample_id,
            feature_key=feature_key,
            non_finite="error",
        )
    except ValueError as error:
        raise InvalidOnlineFeatureSnapshot(str(error)) from error


def _required_str(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise InvalidOnlineFeatureSnapshot(f"{field} must be a non-empty string")
    return value


def _required_float(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidOnlineFeatureSnapshot(f"{field} must be numeric")
    return float(value)


def _required_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if type(value) is not int:
        raise InvalidOnlineFeatureSnapshot(f"{field} must be an integer")
    return value


def _required_bool(payload: dict[str, Any], field: str) -> bool:
    value = payload.get(field)
    if type(value) is not bool:
        raise InvalidOnlineFeatureSnapshot(f"{field} must be boolean")
    return value
