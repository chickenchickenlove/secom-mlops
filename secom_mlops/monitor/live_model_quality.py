from typing import Any

from secom_mlops.monitor.db import connect


INSERT_SQL = """
INSERT INTO live_model_quality_evaluations (
  evaluation_id,
  computed_at,
  model_run_id,
  threshold,
  window_type,
  cutoff_time,
  label_maturity_seconds,
  monitoring_window_seconds,
  window_start,
  window_end,
  n_decisions,
  n_samples,
  n_pass_samples,
  n_fail_samples,
  label_coverage,
  min_decisions,
  min_label_coverage,
  min_pass_samples,
  min_fail_samples,
  evaluation_status,
  accuracy,
  fail_precision,
  fail_recall,
  fail_f1,
  fail_average_precision,
  true_negative,
  false_positive,
  false_negative,
  true_positive
)
VALUES (
  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
  %s, %s, %s, %s, %s, %s, %s, %s, %s
);
"""


def _nullable_float(value: Any) -> float | None:
    return None if value is None else float(value)


class LiveModelQualityEvaluationStore:

    def append_many(self, evaluations: list[dict[str, Any]]) -> None:
        if not evaluations:
            return

        rows = [self._to_insert_row(evaluation) for evaluation in evaluations]

        with connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(INSERT_SQL, rows)

    def _to_insert_row(self, evaluation: dict[str, Any]) -> tuple:
        return (
            evaluation["evaluation_id"],
            float(evaluation["computed_at"]),
            evaluation["model_run_id"],
            float(evaluation["threshold"]),
            evaluation["window_type"],
            float(evaluation["cutoff_time"]),
            float(evaluation["label_maturity_seconds"]),
            float(evaluation["monitoring_window_seconds"]),
            float(evaluation["window_start"]),
            float(evaluation["window_end"]),
            int(evaluation["n_decisions"]),
            int(evaluation["n_samples"]),
            int(evaluation["n_pass_samples"]),
            int(evaluation["n_fail_samples"]),
            float(evaluation["label_coverage"]),
            int(evaluation["min_decisions"]),
            float(evaluation["min_label_coverage"]),
            int(evaluation["min_pass_samples"]),
            int(evaluation["min_fail_samples"]),
            evaluation["evaluation_status"],
            _nullable_float(evaluation["accuracy"]),
            _nullable_float(evaluation["fail_precision"]),
            _nullable_float(evaluation["fail_recall"]),
            _nullable_float(evaluation["fail_f1"]),
            _nullable_float(evaluation["fail_average_precision"]),
            int(evaluation["true_negative"]),
            int(evaluation["false_positive"]),
            int(evaluation["false_negative"]),
            int(evaluation["true_positive"]),
        )
