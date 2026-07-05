import time
from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Jsonb

from secom_mlops.monitor.db import connect

INSERT_SQL = """
INSERT INTO feature_events (
  event_id,
  sample_id,
  event_time,
  feature_group,
  features_json,
  simulation_run_id,
  drift_segment,
  created_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO NOTHING;
"""

REQUIRED_FIELDS = {
    "event_id",
    "sample_id",
    "event_time",
    "feature_group",
    "features",
}


@dataclass(frozen=True)
class FeatureEventAppendResult:
    received: int
    inserted: int
    duplicates: int


class FeatureEventStore:

    def append_many(self,
                    events: list[dict[str, Any]],
                    archived_at: float | None = None
    ) -> FeatureEventAppendResult:
        if not events:
            return FeatureEventAppendResult(received=0, inserted=0, duplicates=0)

        created_at = time.time() if archived_at is None else archived_at
        rows = [self._to_insert_row(event, created_at) for event in events]

        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(INSERT_SQL, rows)
                inserted = max(cursor.rowcount, 0)

        return FeatureEventAppendResult(
            received=len(events),
            inserted=inserted,
            duplicates=len(events) - inserted,
        )

    @staticmethod
    def _to_insert_row(event: dict[str, Any], created_at: float) -> tuple:
        missing_fields = REQUIRED_FIELDS - event.keys()
        if missing_fields:
            raise ValueError(f"missing feature event fields: {sorted(missing_fields)}")

        features = event["features"]
        if not isinstance(features, dict):
            raise ValueError("feature event features must be a JSON object")

        feature_group = str(event["feature_group"])
        if not feature_group:
            raise ValueError("feature_group must not be empty")

        return (
            str(event["event_id"]),
            str(event["sample_id"]),
            float(event["event_time"]),
            feature_group,
            Jsonb(features),
            event.get("simulation_run_id"),
            event.get("drift_segment"),
            float(created_at),
        )
