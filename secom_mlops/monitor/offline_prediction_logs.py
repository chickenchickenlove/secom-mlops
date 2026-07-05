import time
from dataclasses import dataclass
from typing import Any

from secom_mlops.monitor.db import connect

UPSERT_SQL = """
  INSERT INTO offline_prediction_logs (
    offline_prediction_id,
    offline_snapshot_id,
    sample_id,
    build_cutoff_time,
    model_run_id,
    predicted_at,
    fail_probability,
    predicted_value,
    predicted_label,
    threshold,
    missing_count,
    latency_ms,
    created_at
  )
  VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
  ON CONFLICT (offline_snapshot_id, model_run_id, threshold) DO UPDATE SET
    predicted_at = EXCLUDED.predicted_at,
    fail_probability = EXCLUDED.fail_probability,
    predicted_value = EXCLUDED.predicted_value,
    predicted_label = EXCLUDED.predicted_label,
    missing_count = EXCLUDED.missing_count,
    latency_ms = EXCLUDED.latency_ms,
    created_at = EXCLUDED.created_at;
"""


@dataclass(frozen=True)
class OfflinePredictionLogWriteResult:
    attempted: int
    saved: int


class OfflinePredictionLogStore:

    def save_many(
            self,
            logs: list[dict[str, Any]],
            saved_at: float | None = None,
    ) -> OfflinePredictionLogWriteResult:
        if not logs:
            return OfflinePredictionLogWriteResult(attempted=0, saved=0)

        created_at = time.time() if saved_at is None else saved_at
        rows = [self._to_insert_row(log, created_at) for log in logs]

        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(UPSERT_SQL, rows)
                saved = max(cursor.rowcount, 0)

        return OfflinePredictionLogWriteResult(
            attempted=len(logs),
            saved=saved,
        )

    @staticmethod
    def _to_insert_row(log: dict[str, Any], created_at: float) -> tuple:
        return (
            log["offline_prediction_id"],
            log["offline_snapshot_id"],
            log["sample_id"],
            float(log["build_cutoff_time"]),
            log["model_run_id"],
            float(log["predicted_at"]),
            float(log["fail_probability"]),
            int(log["predicted_value"]),
            log["predicted_label"],
            float(log["threshold"]),
            int(log["missing_count"]),
            float(log["latency_ms"]),
            float(created_at),
        )
