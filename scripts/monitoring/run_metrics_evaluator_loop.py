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


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def run_command(command: list[str]) -> None:
    logger.info("+ %s", " ".join(command))
    subprocess.run(command, check=True)


def run_once() -> None:
    prediction_window_limit = env_int("PREDICTION_WINDOW_LIMIT", 500)
    label_maturity_seconds = env_float("LABEL_MATURITY_SECONDS", 0.0)
    monitoring_window_seconds = env_float("MONITORING_WINDOW_SECONDS", 600.0)
    min_decisions = env_int("MIN_DECISIONS", 500)
    min_label_coverage = env_float("MIN_LABEL_COVERAGE", 0.95)
    min_fail_samples = env_int("MIN_FAIL_SAMPLES", 20)
    min_pass_samples = env_int("MIN_PASS_SAMPLES", 20)

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
            "scripts/monitoring/evaluate_live_model_quality.py",
            "--label-maturity-seconds",
            str(label_maturity_seconds),
            "--monitoring-window-seconds",
            str(monitoring_window_seconds),
            "--min-decisions",
            str(min_decisions),
            "--min-label-coverage",
            str(min_label_coverage),
            "--min-fail-samples",
            str(min_fail_samples),
            "--min-pass-samples",
            str(min_pass_samples),
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
        "label_maturity_seconds=%s "
        "monitoring_window_seconds=%s "
        "min_decisions=%s "
        "min_label_coverage=%s "
        "min_fail_samples=%s "
        "min_pass_samples=%s",
        interval_seconds,
        env_int("PREDICTION_WINDOW_LIMIT", 500),
        env_float("LABEL_MATURITY_SECONDS", 0.0),
        env_float("MONITORING_WINDOW_SECONDS", 600.0),
        env_int("MIN_DECISIONS", 500),
        env_float("MIN_LABEL_COVERAGE", 0.95),
        env_int("MIN_FAIL_SAMPLES", 20),
        env_int("MIN_PASS_SAMPLES", 20),
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
