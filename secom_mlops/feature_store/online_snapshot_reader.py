import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import valkey

from secom_mlops_common.schemas.secom import FEATURE_KEYS, NUM_FEATURES, normalize_feature_value


class OnlineFeatureSnapshotNotFound(Exception):
    pass


class InvalidOnlineFeatureSnapshot(ValueError):
    pass


FEATURE_HASH_PATTERN = re.compile(r"^sha256:v1:[0-9a-f]{64}$")


@dataclass(frozen=True)
class OnlineFeatureSnapshot:
    serving_snapshot_id: str
    snapshot_version: int
    feature_hash: str
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
        valkey_url: str | None,
        valkey_host: str,
        valkey_port: int,
        valkey_database: int,
        timeout_seconds: float,
        key_prefix: str,
    ) -> None:
        self._client = _build_client(
            valkey_url=valkey_url,
            valkey_host=valkey_host,
            valkey_port=valkey_port,
            valkey_database=valkey_database,
            timeout_seconds=timeout_seconds,
        )
        self._key_prefix = key_prefix

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


def _build_client(
    valkey_url: str | None,
    valkey_host: str,
    valkey_port: int,
    valkey_database: int,
    timeout_seconds: float,
) -> valkey.Valkey:
    if valkey_url:
        return valkey.Valkey.from_url(
            valkey_url,
            decode_responses=True,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
        )

    return valkey.Valkey(
        host=valkey_host,
        port=valkey_port,
        db=valkey_database,
        decode_responses=True,
        socket_timeout=timeout_seconds,
        socket_connect_timeout=timeout_seconds,
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
    source_event_count = _required_int(payload, "source_event_count")
    snapshot_version = _required_int(payload, "snapshot_version")
    feature_hash = _required_str(payload, "feature_hash")
    snapshot_time = _required_float(payload, "snapshot_time")
    snapshot_status = _required_str(payload, "snapshot_status")
    feature_count = _required_int(payload, "feature_count")
    missing_count = _required_int(payload, "missing_count")
    is_complete = _required_bool(payload, "is_complete")

    if source_event_count < 1:
        raise InvalidOnlineFeatureSnapshot("source_event_count must be >= 1")

    if snapshot_version < 1:
        raise InvalidOnlineFeatureSnapshot("snapshot_version must be >= 1")

    if snapshot_version != source_event_count:
        raise InvalidOnlineFeatureSnapshot(
            "snapshot_version must match source_event_count: "
            f"snapshot_version={snapshot_version} source_event_count={source_event_count}"
        )

    if not FEATURE_HASH_PATTERN.fullmatch(feature_hash):
        raise InvalidOnlineFeatureSnapshot(f"invalid feature_hash: {feature_hash}")

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
        snapshot_version=snapshot_version,
        feature_hash=feature_hash,
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
