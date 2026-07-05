import argparse
import sys

from psycopg.rows import dict_row

from secom_mlops.monitor.db import connect

DEFAULT_SKIP_EXIT_CODE = 99

ACTIVE_BASELINE_SQL = """
SELECT
  baseline_id,
  baseline_name,
  model_run_id,
  model_version,
  threshold
FROM drift_reference_baselines
WHERE baseline_type = 'fixed_reference'
  AND status = 'active'
  AND model_run_id = %s
ORDER BY created_at DESC, baseline_id DESC;
"""


class ResolutionSkipped(Exception):
    pass


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def resolve_active_baseline_id(model_run_id: str) -> str:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(ACTIVE_BASELINE_SQL, [model_run_id])
            rows = [dict(row) for row in cursor.fetchall()]

    if not rows:
        raise ResolutionSkipped(
            f"active fixed-reference baseline not found for model_run_id={model_run_id}"
        )

    if len(rows) > 1:
        baseline_ids = ", ".join(str(row["baseline_id"]) for row in rows)
        raise RuntimeError(
            f"multiple active fixed-reference baselines found for model_run_id={model_run_id}: {baseline_ids}"
        )

    baseline = rows[0]

    log(
        "resolved_active_fixed_reference_baseline "
        f"baseline_id={baseline['baseline_id']} "
        f"baseline_name={baseline['baseline_name']} "
        f"model_run_id={baseline['model_run_id']} "
        f"model_version={baseline['model_version']} "
        f"threshold={baseline['threshold']}"
    )

    return str(baseline["baseline_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--skip-exit-code", type=int, default=DEFAULT_SKIP_EXIT_CODE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        baseline_id = resolve_active_baseline_id(args.model_run_id)
    except ResolutionSkipped as error:
        log(f"active_fixed_reference_baseline_resolution_skipped reason={error}")
        raise SystemExit(args.skip_exit_code)

    # Keep stdout clean for Airflow XCom.
    print(baseline_id, flush=True)


if __name__ == "__main__":
    main()