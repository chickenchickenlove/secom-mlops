import os
import subprocess
import sys
import time

from secom_mlops_common.logging import configure_logging, get_logger

logger = get_logger(__name__)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def run_command(command: list[str]) -> None:
    logger.info("+ %s", " ".join(command))
    subprocess.run(command, check=True)


def run_once() -> None:
    prediction_window_limit = env_int("PREDICTION_WINDOW_LIMIT", 500)
    model_metric_limit = env_int("MODEL_METRIC_LIMIT", 500)
    min_fail_count = env_int("MIN_FAIL_COUNT", 1)

    try:
        run_command([
            sys.executable,
            "scripts/monitoring/evaluate_prediction_window_metrics.py",
            "--limit",
            str(prediction_window_limit),
        ])
    except subprocess.CalledProcessError as error:
        logger.warning(
            "prediction_window_evaluation_failed "
            "returncode=%s",
            error.returncode,
        )

    try:
        run_command([
            sys.executable,
            "scripts/monitoring/evaluate_model_metrics.py",
            "--limit",
            str(model_metric_limit),
            "--min-fail-count",
            str(min_fail_count),
        ])
    except subprocess.CalledProcessError as error:
        logger.warning(
            "model_metric_evaluation_failed "
            "returncode=%s",
            error.returncode,
        )


def main() -> None:
    configure_logging()
    interval_seconds = env_int("EVALUATION_INTERVAL_SECONDS", 60)

    if interval_seconds < 1:
        raise ValueError("EVALUATION_INTERVAL_SECONDS must be >= 1")

    logger.info(
        "metrics_evaluator_loop_started "
        "interval_seconds=%s "
        "prediction_window_limit=%s "
        "model_metric_limit=%s "
        "min_fail_count=%s",
        interval_seconds,
        env_int("PREDICTION_WINDOW_LIMIT", 500),
        env_int("MODEL_METRIC_LIMIT", 500),
        env_int("MIN_FAIL_COUNT", 1),
    )

    while True:
        started_at = time.monotonic()
        run_once()

        elapsed = time.monotonic() - started_at
        sleep_seconds = max(1.0, interval_seconds - elapsed)

        logger.info(
            "metrics_evaluator_loop_sleep "
            "elapsed_seconds=%.2f "
            "sleep_seconds=%.2f",
            elapsed,
            sleep_seconds,
        )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
