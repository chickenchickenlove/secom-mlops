import argparse
import time

from secom_mlops.monitor.db import connect
from secom_mlops_common.logging import configure_logging, get_logger


logger = get_logger(__name__)

MATERIALIZED_VIEW = "label_maturity_cohort_age_metrics"
REFRESH_SQL = f"REFRESH MATERIALIZED VIEW {MATERIALIZED_VIEW};"
SUMMARY_SQL = f"""
SELECT
  COUNT(*) AS metric_rows,
  MAX(computed_at) AS computed_at
FROM {MATERIALIZED_VIEW};
"""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def refresh_once() -> tuple[int, float | None]:
    started_at = time.monotonic()

    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(REFRESH_SQL)
            cursor.execute(SUMMARY_SQL)
            metric_rows, computed_at = cursor.fetchone()

    elapsed_seconds = time.monotonic() - started_at
    logger.info(
        "label_maturity_materialized_view_refreshed "
        "view=%s metric_rows=%s computed_at=%s elapsed_seconds=%.3f",
        MATERIALIZED_VIEW,
        metric_rows,
        computed_at,
        elapsed_seconds,
    )
    return int(metric_rows), None if computed_at is None else float(computed_at)


def run_loop(interval_seconds: int) -> None:
    while True:
        started_at = time.monotonic()

        try:
            refresh_once()
        except Exception:
            logger.exception(
                "label_maturity_materialized_view_refresh_failed view=%s",
                MATERIALIZED_VIEW,
            )

        elapsed_seconds = time.monotonic() - started_at
        sleep_seconds = max(1.0, interval_seconds - elapsed_seconds)
        time.sleep(sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=positive_int, default=60)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.loop:
        logger.info(
            "label_maturity_materialized_view_refresh_loop_started "
            "view=%s interval_seconds=%s",
            MATERIALIZED_VIEW,
            args.interval_seconds,
        )
        run_loop(args.interval_seconds)
        return

    refresh_once()


if __name__ == "__main__":
    main()
