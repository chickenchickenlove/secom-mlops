from dataclasses import dataclass
import pandas as pd

from secom_mlops.monitor.db import connect
from secom_mlops_common.schemas.secom import FEATURE_KEYS, normalize_feature_value


@dataclass(frozen=True)
class OfflineFeatureSnapshot:
    offline_snapshot_id: str
    sample_id: str
    build_cutoff_time: float
    feature_count: int
    missing_count: int
    is_complete: bool
    features_json: dict[str, float | None]
    source_event_count: int
    max_event_time: float | None

    @property
    def values(self) -> list[float | None]:
        return [self.features_json[key] for key in FEATURE_KEYS]


def load_offline_feature_snapshots(
        build_cutoff_time: float,
        only_complete: bool = True,
) -> list[OfflineFeatureSnapshot]:
    if build_cutoff_time < 0:
        raise ValueError("build_cutoff_time must be >= 0")

    complete_filter = "AND is_complete = TRUE" if only_complete else ""

    sql = f"""
    SELECT
      offline_snapshot_id,
      sample_id,
      build_cutoff_time,
      feature_count,
      missing_count,
      is_complete,
      features_json,
      source_event_count,
      max_event_time
    FROM offline_feature_snapshots
    WHERE build_cutoff_time = %s
      {complete_filter}
    ORDER BY sample_id;
    """

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, [build_cutoff_time])
            rows = cursor.fetchall()

    return [_to_snapshot(row) for row in rows]


def offline_snapshots_to_dataframe(
        snapshots: list[OfflineFeatureSnapshot],
) -> pd.DataFrame:
    incomplete = [
        snapshot.sample_id
        for snapshot in snapshots
        if not snapshot.is_complete
    ]
    if incomplete:
        raise ValueError(f"incomplete snapshots cannot be used for model input: {incomplete}")

    return pd.DataFrame(
        [snapshot.values for snapshot in snapshots],
        columns=list(FEATURE_KEYS),
        dtype="float64",
    )


def _to_snapshot(row: tuple) -> OfflineFeatureSnapshot:
    features_json = row[6]
    if not isinstance(features_json, dict):
        raise ValueError(f"features_json must be an object: sample_id={row[1]}")

    missing_keys = set(FEATURE_KEYS) - features_json.keys()
    if missing_keys:
        raise ValueError(
            f"offline snapshot is missing feature keys: "
            f"sample_id={row[1]} missing_count={len(missing_keys)}"
        )

    return OfflineFeatureSnapshot(
        offline_snapshot_id=str(row[0]),
        sample_id=str(row[1]),
        build_cutoff_time=float(row[2]),
        feature_count=int(row[3]),
        missing_count=int(row[4]),
        is_complete=bool(row[5]),
        features_json={
            key: normalize_feature_value(
                features_json[key],
                sample_id=str(row[1]),
                feature_key=key,
                non_finite="allow",
            )
            for key in FEATURE_KEYS
        },
        source_event_count=int(row[7]),
        max_event_time=None if row[8] is None else float(row[8]),
    )
