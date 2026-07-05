from typing import Any
from secom_mlops.monitor.db import connect

UPSERT_SQL = """
INSERT INTO actual_labels (
  sample_id,
  actual_value,
  actual_label,
  labeled_at
)
VALUES (%s, %s, %s, %s)
ON CONFLICT(sample_id) DO UPDATE SET
  actual_value = excluded.actual_value,
  actual_label = excluded.actual_label,
  labeled_at = excluded.labeled_at;  
"""


class ActualLabelStore:

    def save_many(self, labels: list[dict[str, Any]]) -> None:
        if not labels:
            return

        rows = [self._to_upsert_row(label) for label in labels]

        with connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(UPSERT_SQL, rows)

    def _to_upsert_row(self, label: dict[str, Any]) -> tuple:
        return (
            label["sample_id"],
            int(label["actual_value"]),
            label["actual_label"],
            float(label["labeled_at"]),
        )
