import time
from dataclasses import dataclass

from psycopg.types.json import Jsonb

from secom_mlops.feature_store.reconstruction import ReconstructedFeatureRow
from secom_mlops.monitor.db import connect

UPSERT_SQL = """
  INSERT INTO offline_feature_snapshots (
    offline_snapshot_id,
    sample_id,
    build_cutoff_time,
    feature_count,
    missing_count,
    is_complete,
    features_json,
    source_event_count,
    max_event_time,
    simulation_run_id,
    drift_segment,
    created_at
  )
  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
  ON CONFLICT (offline_snapshot_id) DO UPDATE SET
    feature_count = EXCLUDED.feature_count,
    missing_count = EXCLUDED.missing_count,
    is_complete = EXCLUDED.is_complete,
    features_json = EXCLUDED.features_json,
    source_event_count = EXCLUDED.source_event_count,
    max_event_time = EXCLUDED.max_event_time,
    simulation_run_id = EXCLUDED.simulation_run_id,
    drift_segment = EXCLUDED.drift_segment,
    created_at = EXCLUDED.created_at;
"""


@dataclass(frozen=True)
class OfflineSnapshotWriteResult:
    attempted: int
    saved: int


class OfflineFeatureSnapshotStore:

    def save_many(
            self,
            rows: list[ReconstructedFeatureRow],
            saved_at: float | None = None,
    ) -> OfflineSnapshotWriteResult:
        if not rows:
            return OfflineSnapshotWriteResult(attempted=0, saved=0)

        created_at = time.time() if saved_at is None else saved_at
        insert_rows = [
            self._to_insert_row(row, created_at)
            for row in rows
        ]

        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(UPSERT_SQL, insert_rows)
                saved = max(cursor.rowcount, 0)

        return OfflineSnapshotWriteResult(
            attempted=len(rows),
            saved=saved,
        )

    @staticmethod
    def _to_insert_row(
            row: ReconstructedFeatureRow,
            created_at: float,
    ) -> tuple:
        return (
            build_offline_snapshot_id(row.sample_id, row.build_cutoff_time),
            row.sample_id,
            row.build_cutoff_time,
            row.observed_feature_count,
            row.missing_count,
            row.patch_complete,
            Jsonb(row.features_json),
            row.source_event_count,
            row.max_event_time,
            None,
            None,
            created_at,
        )


def build_offline_snapshot_id(
        sample_id: str,
        build_cutoff_time: float,
) -> str:
    return f"offline:{sample_id}:{build_cutoff_time:.6f}"
