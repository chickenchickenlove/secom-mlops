from typing import Any

from secom_mlops.monitor.db import connect

INSERT_SQL = """
INSERT INTO prediction_logs (
  prediction_id,
  request_id,
  sample_id,
  serving_snapshot_id,
  snapshot_version,
  model_run_id,
  model_name,
  model_version,
  model_alias,
  model_uri,
  runtime_slot,
  predicted_at,
  fail_probability,
  predicted_value,
  predicted_label,
  threshold,
  missing_count,
  latency_ms
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""


class PredictionLogStore:

    def save_many(self, logs: list[dict[str, Any]]) -> None:
        if not logs:
            return

        rows = [self._to_insert_row(log) for log in logs]

        with connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(INSERT_SQL, rows)

    def _to_insert_row(self, log: dict[str, Any]) -> tuple:
        return (
            log.get("prediction_id"),
            log.get("request_id"),
            log.get("sample_id"),
            log.get("serving_snapshot_id"),
            int(log.get("snapshot_version")),
            log.get("model_run_id"),
            log.get("model_name"),
            log.get("model_version"),
            log.get("model_alias"),
            log.get("model_uri"),
            log.get("runtime_slot", "unknown"),
            float(log.get("predicted_at")),
            float(log.get("fail_probability")),
            int(log.get("predicted_value")),
            log.get("predicted_label"),
            float(log.get("threshold")),
            int(log.get("missing_count")),
            float(log.get("latency_ms")),
        )
