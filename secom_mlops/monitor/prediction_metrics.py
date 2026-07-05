from typing import Any
from secom_mlops.monitor.db import connect

INSERT_SQL = """
INSERT INTO prediction_window_metrics (
  evaluation_id,
  computed_at,
  model_run_id,
  threshold,
  window_type,
  window_size,
  window_start,
  window_end,
  metric_name,
  metric_value,
  n_predictions,
  created_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

class PredictionWindowMetricStore:

    def append_many(self, metrics: list[dict[str, Any]]) -> None:
        if not metrics:
            return

        rows = [self._to_insert_row(metric) for metric in metrics]

        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(INSERT_SQL, rows)

    def _to_insert_row(self, metric: dict[str, Any]) -> tuple:
        metric_value = metric["metric_value"]

        return (
            metric["evaluation_id"],
            float(metric["computed_at"]),
            metric["model_run_id"],
            None if metric["threshold"] is None else float(metric["threshold"]),
            metric["window_type"],
            metric["window_size"],
            metric["window_start"],
            metric["window_end"],
            metric["metric_name"],
            None if metric_value is None else float(metric_value),
            int(metric["n_predictions"]),
            float(metric["created_at"]),
        )
