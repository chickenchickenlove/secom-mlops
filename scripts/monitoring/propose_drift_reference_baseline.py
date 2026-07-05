import argparse
import math
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from psycopg.rows import dict_row

from secom_mlops.monitor.db import connect

UTILITY_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "utility"
if str(UTILITY_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_SCRIPT_DIR))

from resolve_mlflow_champion_run_id import (
    MODEL_VERSION_BY_ALIAS_PATH,
    ResolutionSkipped,
    get_json,
    load_run_threshold,
)
from secom_mlops_common.cli.validators import (
    non_negative_float,
    positive_float,
    positive_int,
)
from secom_mlops_common.config.mlflow import (
    DEFAULT_CHAMPION_ALIAS,
    DEFAULT_CONTAINER_MLFLOW_TRACKING_URI,
    DEFAULT_MODEL_NAME,
    resolve_model_alias,
    resolve_model_name,
    resolve_tracking_uri,
)
from secom_mlops_common.metrics.stats import mean, percentile, ratio

POSITIVE_CLASS = 1

PREDICTION_RANGE_SQL = """
SELECT
  COUNT(*) AS sample_count,
  MIN(predicted_at) AS first_predicted_at,
  MAX(predicted_at) AS last_predicted_at
FROM prediction_logs
WHERE model_run_id = %s
  AND threshold = %s;
"""

FETCH_PREDICTIONS_SQL = """
SELECT
  prediction_id,
  sample_id,
  model_name,
  model_version,
  model_alias,
  model_run_id,
  threshold,
  predicted_at,
  fail_probability,
  predicted_value,
  predicted_label,
  missing_count
FROM prediction_logs
WHERE model_run_id = %s
  AND threshold = %s
  AND predicted_at > %s
  AND predicted_at <= %s
ORDER BY predicted_at, prediction_id;
"""


@dataclass(frozen=True)
class ResolvedChampion:
    model_name: str
    model_alias: str
    model_version: str | None
    model_run_id: str
    threshold: float


@dataclass(frozen=True)
class ExclusionWindow:
    source_start: float
    source_end: float


@dataclass(frozen=True)
class CandidateWindow:
    source_start: float
    source_end: float
    sample_count: int
    first_predicted_at: float | None
    last_predicted_at: float | None
    predicted_fail_ratio: float | None
    fail_probability_avg: float | None
    fail_probability_p50: float | None
    fail_probability_p95: float | None
    missing_count_avg: float | None
    missing_count_p50: float | None
    missing_count_p95: float | None
    missing_count_max: float | None


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def epoch_to_utc(value: float | None) -> str:
    if value is None:
        return "None"
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def resolve_champion(
        mlflow_tracking_uri: str,
        model_name: str,
        model_alias: str,
        timeout_seconds: float,
) -> ResolvedChampion:
    with httpx.Client(
            base_url=mlflow_tracking_uri.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
    ) as client:
        payload = get_json(
            client,
            MODEL_VERSION_BY_ALIAS_PATH,
            {"name": model_name, "alias": model_alias},
            not_found_skip_message=(
                f"model alias not found in MLflow: model_name={model_name} alias={model_alias}"
            ),
        )

        model_version = payload.get("model_version")
        if not isinstance(model_version, dict):
            raise RuntimeError(
                f"MLflow alias response missing model_version: model_name={model_name} alias={model_alias}"
            )

        run_id = model_version.get("run_id")
        if not run_id:
            raise ResolutionSkipped(
                f"MLflow model version has no run_id: model_name={model_name} alias={model_alias}"
            )

        threshold = load_run_threshold(client, str(run_id))

    return ResolvedChampion(
        model_name=model_name,
        model_alias=model_alias,
        model_version=str(model_version.get("version")) if model_version.get("version") is not None else None,
        model_run_id=str(run_id),
        threshold=float(threshold),
    )


def load_prediction_range(model_run_id: str, threshold: float) -> dict[str, Any]:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(PREDICTION_RANGE_SQL, [model_run_id, threshold])
            row = cursor.fetchone()

    if row is None:
        return {"sample_count": 0, "first_predicted_at": None, "last_predicted_at": None}

    return dict(row)


def load_prediction_rows(
        model_run_id: str,
        threshold: float,
        source_start: float,
        source_end: float,
) -> list[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                FETCH_PREDICTIONS_SQL,
                [model_run_id, threshold, source_start, source_end],
            )
            return [dict(row) for row in cursor.fetchall()]


def parse_exclusion_window(value: str) -> ExclusionWindow:
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("must use START_EPOCH:END_EPOCH")

    source_start = float(parts[0])
    source_end = float(parts[1])

    if source_start < 0.0 or source_end < 0.0:
        raise argparse.ArgumentTypeError("exclusion epochs must be >= 0")

    if source_end <= source_start:
        raise argparse.ArgumentTypeError("exclusion end must be greater than start")

    return ExclusionWindow(source_start=source_start, source_end=source_end)


def overlaps_exclusion(
        source_start: float,
        source_end: float,
        exclusions: list[ExclusionWindow],
) -> bool:
    for exclusion in exclusions:
        if source_start < exclusion.source_end and source_end > exclusion.source_start:
            return True
    return False


def build_windows(
        lookback_start: float,
        lookback_end: float,
        window_seconds: float,
        step_seconds: float,
        exclusions: list[ExclusionWindow],
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    source_start = lookback_start

    while source_start + window_seconds <= lookback_end + 0.000001:
        source_end = source_start + window_seconds

        if not overlaps_exclusion(source_start, source_end, exclusions):
            windows.append((source_start, source_end))

        source_start += step_seconds

    return windows


def summarize_window(
        rows: list[dict[str, Any]],
        source_start: float,
        source_end: float,
) -> CandidateWindow:
    sample_count = len(rows)

    probabilities = [
        float(row["fail_probability"])
        for row in rows
        if row.get("fail_probability") is not None
    ]
    missing_counts = [
        float(row["missing_count"])
        for row in rows
        if row.get("missing_count") is not None
    ]
    predicted_fail_count = sum(
        1 for row in rows
        if row.get("predicted_value") is not None and int(row["predicted_value"]) == POSITIVE_CLASS
    )

    predicted_times = [
        float(row["predicted_at"])
        for row in rows
        if row.get("predicted_at") is not None
    ]

    return CandidateWindow(
        source_start=source_start,
        source_end=source_end,
        sample_count=sample_count,
        first_predicted_at=min(predicted_times) if predicted_times else None,
        last_predicted_at=max(predicted_times) if predicted_times else None,
        predicted_fail_ratio=ratio(predicted_fail_count, sample_count),
        fail_probability_avg=mean(probabilities),
        fail_probability_p50=percentile(probabilities, 0.50),
        fail_probability_p95=percentile(probabilities, 0.95),
        missing_count_avg=mean(missing_counts),
        missing_count_p50=percentile(missing_counts, 0.50),
        missing_count_p95=percentile(missing_counts, 0.95),
        missing_count_max=max(missing_counts) if missing_counts else None,
    )


def propose_candidates(
        model_run_id: str,
        threshold: float,
        lookback_start: float,
        lookback_end: float,
        window_seconds: float,
        step_seconds: float,
        min_samples: int,
        exclusions: list[ExclusionWindow],
) -> list[CandidateWindow]:
    windows = build_windows(
        lookback_start=lookback_start,
        lookback_end=lookback_end,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        exclusions=exclusions,
    )

    candidates: list[CandidateWindow] = []

    for source_start, source_end in windows:
        rows = load_prediction_rows(
            model_run_id=model_run_id,
            threshold=threshold,
            source_start=source_start,
            source_end=source_end,
        )

        if len(rows) < min_samples:
            continue

        candidates.append(
            summarize_window(
                rows=rows,
                source_start=source_start,
                source_end=source_end,
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (candidate.sample_count, candidate.source_end),
        reverse=True,
    )


def default_baseline_name(champion: ResolvedChampion, candidate: CandidateWindow) -> str:
    start = datetime.fromtimestamp(candidate.source_start, tz=timezone.utc).strftime("%Y%m%d-%H%M")
    end = datetime.fromtimestamp(candidate.source_end, tz=timezone.utc).strftime("%Y%m%d-%H%M")
    return f"{champion.model_alias}-online-stable-{start}-{end}"


def build_create_command(
        champion: ResolvedChampion,
        candidate: CandidateWindow,
        baseline_name: str,
        min_samples: int,
        notes: str,
        dry_run: bool,
) -> list[str]:
    command = [
        "python",
        "scripts/monitoring/create_drift_reference_baseline.py",
        "--baseline-name",
        baseline_name,
        "--model-run-id",
        champion.model_run_id,
        "--threshold",
        str(champion.threshold),
        "--source-start",
        str(candidate.source_start),
        "--source-end",
        str(candidate.source_end),
        "--min-samples",
        str(min_samples),
        "--status",
        "active",
        "--notes",
        notes,
    ]

    if dry_run:
        command.append("--dry-run")

    return command


def print_candidate(rank: int, candidate: CandidateWindow) -> None:
    print(f"candidate_rank={rank}")
    print(f"  source_start={candidate.source_start}")
    print(f"  source_end={candidate.source_end}")
    print(f"  source_start_utc={epoch_to_utc(candidate.source_start)}")
    print(f"  source_end_utc={epoch_to_utc(candidate.source_end)}")
    print(f"  first_predicted_at_utc={epoch_to_utc(candidate.first_predicted_at)}")
    print(f"  last_predicted_at_utc={epoch_to_utc(candidate.last_predicted_at)}")
    print(f"  sample_count={candidate.sample_count}")
    print("  output_summary:")
    print(f"    predicted_fail_ratio={format_float(candidate.predicted_fail_ratio)}")
    print(f"    fail_probability_avg={format_float(candidate.fail_probability_avg)}")
    print(f"    fail_probability_p50={format_float(candidate.fail_probability_p50)}")
    print(f"    fail_probability_p95={format_float(candidate.fail_probability_p95)}")
    print("  missing_count_summary:")
    print(f"    missing_count_avg={format_float(candidate.missing_count_avg)}")
    print(f"    missing_count_p50={format_float(candidate.missing_count_p50)}")
    print(f"    missing_count_p95={format_float(candidate.missing_count_p95)}")
    print(f"    missing_count_max={format_float(candidate.missing_count_max)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend fixed-reference drift baseline candidate windows from prediction_logs. "
            "This script does not create or approve a baseline."
        )
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=resolve_tracking_uri(default=DEFAULT_CONTAINER_MLFLOW_TRACKING_URI),
    )
    parser.add_argument(
        "--model-name",
        default=resolve_model_name(default=DEFAULT_MODEL_NAME),
    )
    parser.add_argument(
        "--model-alias",
        default=resolve_model_alias(default=DEFAULT_CHAMPION_ALIAS),
    )
    parser.add_argument("--lookback-hours", type=positive_float, default=6.0)
    parser.add_argument("--window-minutes", type=positive_float, default=60.0)
    parser.add_argument(
        "--step-minutes",
        type=positive_float,
        default=None,
        help="Defaults to --window-minutes for non-overlapping candidate windows.",
    )
    parser.add_argument("--min-samples", type=positive_int, default=500)
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument(
        "--end-at",
        type=non_negative_float,
        default=None,
        help="Epoch seconds for the lookback end. Defaults to latest prediction for the champion run.",
    )
    parser.add_argument(
        "--exclude-window",
        type=parse_exclusion_window,
        action="append",
        default=[],
        help="Exclude candidate windows overlapping START_EPOCH:END_EPOCH. Repeatable.",
    )
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument(
        "--notes",
        default="operator-approved stable online window proposed by propose_drift_reference_baseline.py",
    )
    parser.add_argument("--timeout-seconds", type=positive_float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    champion = resolve_champion(
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        model_name=args.model_name,
        model_alias=args.model_alias,
        timeout_seconds=args.timeout_seconds,
    )

    prediction_range = load_prediction_range(
        model_run_id=champion.model_run_id,
        threshold=champion.threshold,
    )

    total_samples = int(prediction_range["sample_count"] or 0)
    last_predicted_at = prediction_range["last_predicted_at"]

    if total_samples == 0 or last_predicted_at is None:
        raise ValueError(
            "no prediction_logs rows found for champion model: "
            f"model_run_id={champion.model_run_id} threshold={champion.threshold}"
        )

    lookback_end = float(args.end_at) if args.end_at is not None else float(math.ceil(float(last_predicted_at)))
    lookback_start = lookback_end - (args.lookback_hours * 60.0 * 60.0)
    window_seconds = args.window_minutes * 60.0
    step_seconds = (args.step_minutes if args.step_minutes is not None else args.window_minutes) * 60.0

    if window_seconds > (lookback_end - lookback_start):
        raise ValueError("--window-minutes must be less than or equal to --lookback-hours")

    candidates = propose_candidates(
        model_run_id=champion.model_run_id,
        threshold=champion.threshold,
        lookback_start=lookback_start,
        lookback_end=lookback_end,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        min_samples=args.min_samples,
        exclusions=args.exclude_window,
    )

    print(
        "resolved_champion_model "
        f"model_name={champion.model_name} "
        f"model_alias={champion.model_alias} "
        f"model_version={champion.model_version} "
        f"model_run_id={champion.model_run_id} "
        f"threshold={champion.threshold}"
    )
    print(
        "baseline_candidate_search "
        f"lookback_start={lookback_start} "
        f"lookback_end={lookback_end} "
        f"lookback_start_utc={epoch_to_utc(lookback_start)} "
        f"lookback_end_utc={epoch_to_utc(lookback_end)} "
        f"lookback_hours={args.lookback_hours} "
        f"window_minutes={args.window_minutes} "
        f"step_minutes={args.step_minutes if args.step_minutes is not None else args.window_minutes} "
        f"min_samples={args.min_samples} "
        f"excluded_windows={len(args.exclude_window)}"
    )

    if not candidates:
        raise ValueError(
            "no eligible baseline candidate windows found: "
            f"lookback_hours={args.lookback_hours} "
            f"window_minutes={args.window_minutes} "
            f"min_samples={args.min_samples}"
        )

    for rank, candidate in enumerate(candidates[:args.top_k], start=1):
        print()
        print_candidate(rank, candidate)

    selected = candidates[0]
    baseline_name = args.baseline_name or default_baseline_name(champion, selected)

    dry_run_command = build_create_command(
        champion=champion,
        candidate=selected,
        baseline_name=baseline_name,
        min_samples=args.min_samples,
        notes=args.notes,
        dry_run=True,
    )
    create_command = build_create_command(
        champion=champion,
        candidate=selected,
        baseline_name=baseline_name,
        min_samples=args.min_samples,
        notes=args.notes,
        dry_run=False,
    )

    print()
    print("copy_paste_dry_run_command:")
    print(shell_join(dry_run_command))
    print()
    print("copy_paste_create_command_after_operator_approval:")
    print(shell_join(create_command))


if __name__ == "__main__":
    try:
        main()
    except ResolutionSkipped as error:
        log(f"baseline_proposal_skipped reason={error}")
        raise SystemExit(99)
