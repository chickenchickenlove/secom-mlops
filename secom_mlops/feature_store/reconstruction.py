from dataclasses import dataclass
from typing import Any

from secom_mlops.monitor.db import connect
from secom_mlops_common.schemas.secom import (
    FEATURE_KEYS,
    FEATURE_KEY_SET,
    NUM_FEATURES,
    normalize_feature_value,
)


@dataclass(frozen=True)
class ReconstructedFeatureRow:
    sample_id: str
    build_cutoff_time: float
    features_json: dict[str, float | None]
    observed_feature_count: int
    missing_count: int
    patch_complete: bool
    has_no_missing_values: bool
    source_event_count: int
    max_event_time: float | None

    @property
    def values(self) -> list[float | None]:
        return [self.features_json[key] for key in FEATURE_KEYS]


def reconstruct_feature_rows(
        build_cutoff_time: float,
) -> list[ReconstructedFeatureRow]:
    if build_cutoff_time < 0:
        raise ValueError("build_cutoff_time must be >= 0")

    events = _fetch_events(build_cutoff_time)

    states: dict[str, dict[str, float | None]] = {}
    source_event_counts: dict[str, int] = {}
    max_event_times: dict[str, float] = {}

    for sample_id, event_time, features_json in events:
        if not isinstance(features_json, dict):
            raise ValueError(f"features_json must be an object: sample_id={sample_id}")

        state = states.setdefault(sample_id, {})
        source_event_counts[sample_id] = source_event_counts.get(sample_id, 0) + 1
        max_event_times[sample_id] = event_time

        for key, value in features_json.items():
            if key not in FEATURE_KEY_SET:
                raise ValueError(f"unexpected feature key: sample_id={sample_id} key={key}")

            state[key] = normalize_feature_value(
                value,
                sample_id=sample_id,
                feature_key=key,
                non_finite="allow",
            )

    reconstructed = []

    for sample_id in sorted(states):
        state = states[sample_id]
        features = {
            key: state.get(key)
            for key in FEATURE_KEYS
        }

        observed_feature_count = len(state)
        missing_count = sum(value is None for value in features.values())

        reconstructed.append(ReconstructedFeatureRow(
            sample_id=sample_id,
            build_cutoff_time=build_cutoff_time,
            features_json=features,
            observed_feature_count=observed_feature_count,
            missing_count=missing_count,
            patch_complete=observed_feature_count == NUM_FEATURES,
            has_no_missing_values=missing_count == 0,
            source_event_count=source_event_counts[sample_id],
            max_event_time=max_event_times.get(sample_id),
        ))

    return reconstructed


def _fetch_events(
        build_cutoff_time: float,
) -> list[tuple[str, float, dict[str, Any]]]:
    sql = """
          SELECT sample_id, event_time, features_json
          FROM feature_events
          WHERE event_time <= %s
          ORDER BY sample_id, event_time, created_at, event_id; \
          """

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, [build_cutoff_time])
            return cursor.fetchall()

