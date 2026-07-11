import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from confluent_kafka import Producer

from secom_mlops_common.config.kafka import (
    resolve_feature_patches_topic,
    resolve_kafka_bootstrap_servers,
)
from secom_mlops_common.schemas.secom import FEATURE_KEYS, NUM_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = PROJECT_ROOT / "runtime"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "secom.data"
DEFAULT_DRIFT_SEGMENT = "on_time"

FEATURE_GROUPS = {
    "early": range(0, 197),
    "middle": range(197, 394),
    "late": range(394, NUM_FEATURES),
}


@dataclass
class PublishStats:
    attempted: int = 0
    delivered: int = 0
    failed: int = 0


@dataclass(frozen=True)
class FeatureOffsetAction:
    feature_index: int
    direction: str
    offset: float
    frequency: int

    @property
    def delta(self) -> float:
        if self.direction == "+":
            return self.offset
        return -self.offset

    @property
    def direction_name(self) -> str:
        if self.direction == "+":
            return "up"
        return "down"


def load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "next_feature_index": 0,
        }

    state = json.loads(path.read_text(encoding="utf-8"))
    return {
        "next_feature_index": int(state.get("next_feature_index", 0)),
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**state, "updated_at": time.time()}

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def normalize_value(value: Any) -> float | None:
    if pd.isna(value):
        return None

    number = float(value)
    if not math.isfinite(number):
        return None

    return number


def load_features(data_path: str) -> list[list[float | None]]:
    frame = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        na_values="NaN",
    )

    return [
        [normalize_value(value) for value in raw_row]
        for raw_row in frame.values.tolist()
    ]


def resolve_state_path(raw_state_path: str | None, feature_group: str) -> Path:
    if raw_state_path is not None:
        return Path(raw_state_path)

    return DEFAULT_STATE_DIR / f"online_workload_next_feature_{feature_group}_state.json"


def selected_feature_groups(feature_group: str) -> dict[str, range]:
    if feature_group == "all":
        return dict(FEATURE_GROUPS)

    return {
        feature_group: FEATURE_GROUPS[feature_group],
    }


def apply_feature_offset(
        row: list[float | None],
        direction: str,
        ratio: float,
) -> list[float | None]:
    if direction == "none" or ratio == 0:
        return row

    if direction == "up":
        multiplier = 1.0 + ratio
    elif direction == "down":
        multiplier = 1.0 - ratio
    else:
        raise ValueError(f"unknown feature_offset_direction={direction}")

    return [
        None if value is None else value * multiplier
        for value in row
    ]


def normalize_feature_offset_actions(raw_actions: Any) -> list[FeatureOffsetAction]:
    if raw_actions is None:
        return []
    if isinstance(raw_actions, FeatureOffsetAction):
        return [raw_actions]
    return list(raw_actions)


def apply_feature_offset_actions(
        row: list[float | None],
        actions: list[FeatureOffsetAction],
        trial_number: int,
) -> list[float | None]:
    updated_row = row

    for action in actions:
        if trial_number % action.frequency != 0:
            continue

        value = updated_row[action.feature_index]
        if value is None:
            continue

        if updated_row is row:
            updated_row = list(row)

        updated_row[action.feature_index] = value + action.delta

    return updated_row


def build_feature_patch(row: list[float | None], indices: range) -> dict[str, float | None]:
    return {
        FEATURE_KEYS[idx]: row[idx]
        for idx in indices
    }


def format_float_for_name(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def format_ratio_for_name(ratio: float) -> str:
    return format_float_for_name(ratio)


def format_feature_offset_action_for_name(action: FeatureOffsetAction) -> str:
    offset_name = format_float_for_name(action.offset)
    return (
        f"feature_offset_action_"
        f"f{action.feature_index:03d}_"
        f"{action.direction_name}_"
        f"{offset_name}_"
        f"every_{action.frequency}"
    )


def resolve_drift_segment(args: argparse.Namespace) -> str:
    if args.drift_segment is not None:
        return args.drift_segment

    segment_parts = []
    if args.feature_offset_direction != "none":
        ratio_name = format_ratio_for_name(args.feature_offset_ratio)
        segment_parts.append(f"feature_offset_{args.feature_offset_direction}_{ratio_name}")

    for action in normalize_feature_offset_actions(args.feature_offset_action):
        segment_parts.append(format_feature_offset_action_for_name(action))

    if not segment_parts:
        return DEFAULT_DRIFT_SEGMENT

    return "_".join(segment_parts)


def build_feature_events(
        rows: list[list[float | None]],
        args: argparse.Namespace,
        start_index: int,
        batch_size: int,
        simulation_run_id: str,
        action_start_index: int = 0,
) -> list[dict[str, Any]]:
    drift_segment = resolve_drift_segment(args)
    created_at = time.time()
    event_time = created_at
    groups = selected_feature_groups(args.feature_group)
    offset_actions = normalize_feature_offset_actions(args.feature_offset_action)
    events = []

    for item_index in range(batch_size):
        global_sample_index = start_index + item_index
        source_row_index = global_sample_index % len(rows)
        sample_id = f"secom-{global_sample_index:07d}"
        row = apply_feature_offset(
            rows[source_row_index],
            args.feature_offset_direction,
            args.feature_offset_ratio,
        )
        row = apply_feature_offset_actions(
            row,
            offset_actions,
            action_start_index + item_index + 1,
        )

        for feature_group, indices in groups.items():
            events.append({
                "event_id": (
                    f"{simulation_run_id}:"
                    f"{sample_id}:"
                    f"{feature_group}:"
                    f"{int(event_time * 1000)}"
                ),
                "sample_id": sample_id,
                "source_row_index": source_row_index,
                "event_time": event_time,
                "scheduled_publish_time": event_time,
                "feature_group": feature_group,
                "features": build_feature_patch(row, indices),
                "simulation_run_id": simulation_run_id,
                "drift_segment": drift_segment,
                "feature_offset_direction": args.feature_offset_direction,
                "feature_offset_ratio": args.feature_offset_ratio,
                "created_at": created_at,
            })

    return sorted_for_publish(events)


def sorted_for_publish(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_events = list(enumerate(events))
    indexed_events.sort(
        key=lambda item: (
            float(item[1]["scheduled_publish_time"]),
            item[0],
        )
    )
    return [event for _, event in indexed_events]


def delivery_callback(stats: PublishStats):
    def callback(error, message) -> None:
        if error is not None:
            stats.failed += 1
            print(f"delivery_failed key={message.key()!r} error={error}", file=sys.stderr)
            return

        stats.delivered += 1

    return callback


def publish_feature_events(args: argparse.Namespace, events: list[dict[str, Any]]) -> PublishStats:
    producer = Producer({
        "bootstrap.servers": args.bootstrap_servers,
        "client.id": "secom-feature-event-cursor-producer",
        "broker.address.family": "v4",
    })

    stats = PublishStats()
    callback = delivery_callback(stats)

    for event in events:
        value = json.dumps(event, separators=(",", ":"), allow_nan=False)

        while True:
            try:
                producer.produce(
                    topic=args.topic,
                    key=event["sample_id"].encode("utf-8"),
                    value=value.encode("utf-8"),
                    on_delivery=callback,
                )
                stats.attempted += 1
                break
            except BufferError:
                producer.poll(1.0)

        producer.poll(0)

    remaining = producer.flush(30.0)
    if remaining > 0:
        raise RuntimeError(f"producer_flush_timeout remaining_messages={remaining}")

    return stats


def summarize(events: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = {}

    for event in events:
        key = (
            event["drift_segment"],
            event["feature_group"],
        )
        counts[key] = counts.get(key, 0) + 1

    for key in sorted(counts):
        drift_segment, feature_group = key
        print(
            f"drift_segment={drift_segment} "
            f"feature_group={feature_group} "
            f"count={counts[key]}"
        )


def parse_feature_index(raw_value: str) -> int:
    value = raw_value.strip()
    if value.startswith("feature_"):
        value = value.removeprefix("feature_")
    elif value.startswith("f") and len(value) > 1:
        value = value[1:]

    try:
        feature_index = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"feature index must be an integer, fNNN, or feature_NNN: {raw_value}"
        ) from error

    if feature_index < 0 or feature_index >= NUM_FEATURES:
        raise argparse.ArgumentTypeError(
            f"feature index must be between 0 and {NUM_FEATURES - 1}: {raw_value}"
        )

    return feature_index


def parse_feature_offset_direction(raw_value: str) -> str:
    value = raw_value.strip().lower()
    if value in {"+", "positive", "up"}:
        return "+"
    if value in {"-", "negative", "down"}:
        return "-"

    raise argparse.ArgumentTypeError(
        f"feature offset direction must be one of +, -, positive, negative, up, down: {raw_value}"
    )


def parse_feature_offset_action(raw_value: str) -> FeatureOffsetAction:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 4 or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(
            "feature-offset-action must use FEATURE,SIGN,OFFSET,FREQUENCY, "
            "for example: 480,+,5.9,2"
        )

    feature_index = parse_feature_index(parts[0])
    direction = parse_feature_offset_direction(parts[1])

    try:
        offset = float(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"feature offset must be a number: {parts[2]}"
        ) from error

    if not math.isfinite(offset) or offset <= 0:
        raise argparse.ArgumentTypeError("feature offset must be a finite number > 0")

    try:
        frequency = int(parts[3])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"feature offset frequency must be an integer: {parts[3]}"
        ) from error

    if frequency < 1:
        raise argparse.ArgumentTypeError("feature offset frequency must be >= 1")

    return FeatureOffsetAction(
        feature_index=feature_index,
        direction=direction,
        offset=offset,
        frequency=frequency,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--bootstrap-servers", default=resolve_kafka_bootstrap_servers())
    parser.add_argument("--topic", default=resolve_feature_patches_topic())
    parser.add_argument("--simulation-run-id-prefix", default="online_feature_workload")
    parser.add_argument(
        "--feature-group",
        choices=["all", *FEATURE_GROUPS.keys()],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--feature-offset-action",
        type=parse_feature_offset_action,
        action="append",
        default=None,
        metavar="FEATURE,SIGN,OFFSET,FREQUENCY",
        help=(
            "Apply an absolute offset to one feature every N samples, e.g. 480,+,5.9,2. "
            "May be specified multiple times."
        ),
    )
    parser.add_argument(
        "--feature-offset-direction",
        choices=["none", "up", "down"],
        default="none",
    )
    parser.add_argument("--feature-offset-ratio", type=float, default=0.0)
    parser.add_argument("--drift-segment", default=None)
    args = parser.parse_args()
    args.feature_offset_action = normalize_feature_offset_actions(args.feature_offset_action)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 1:
        raise ValueError("max-samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    if args.sleep_seconds < 0:
        raise ValueError("sleep-seconds must be >= 0")
    if args.feature_offset_ratio < 0:
        raise ValueError("feature-offset-ratio must be >= 0")
    if args.feature_offset_direction == "none" and args.feature_offset_ratio != 0:
        raise ValueError("feature-offset-ratio must be 0 when feature-offset-direction is none")
    if args.feature_offset_direction != "none" and args.feature_offset_ratio <= 0:
        raise ValueError("feature-offset-ratio must be > 0 when feature-offset-direction is up/down")
    if args.feature_offset_direction == "down" and args.feature_offset_ratio >= 1:
        raise ValueError("feature-offset-ratio must be < 1 when feature-offset-direction is down")


def main() -> None:
    args = parse_args()
    validate_args(args)

    rows = load_features(args.data_path)
    if not rows:
        raise ValueError(f"no feature rows loaded: {args.data_path}")
    if any(len(row) != NUM_FEATURES for row in rows):
        raise ValueError(f"expected each feature row to have {NUM_FEATURES} values")

    state_path = resolve_state_path(args.state_path, args.feature_group)
    state = load_state(state_path)

    remaining = args.max_samples
    total_sent = 0

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        start_index = state["next_feature_index"]
        drift_segment = resolve_drift_segment(args)
        simulation_run_id = (
            f"{args.simulation_run_id_prefix}_"
            f"{start_index}_{start_index + batch_size - 1}_{drift_segment}"
        )

        events = build_feature_events(
            rows=rows,
            args=args,
            start_index=start_index,
            batch_size=batch_size,
            simulation_run_id=simulation_run_id,
            action_start_index=total_sent,
        )

        summarize(events)

        if args.dry_run:
            for event in events[:5]:
                print(json.dumps(event, indent=2, allow_nan=False))
            print(
                f"dry_run_feature_events={len(events)} "
                f"feature_group={args.feature_group} "
                f"start_index={start_index} "
                f"batch_size={batch_size} "
                f"state_path={state_path} "
                f"next_feature_index_would_be={start_index + batch_size}"
            )
        else:
            stats = publish_feature_events(args, events)
            print(
                "publish_complete "
                f"topic={args.topic} "
                f"feature_group={args.feature_group} "
                f"simulation_run_id={simulation_run_id} "
                f"attempted={stats.attempted} "
                f"delivered={stats.delivered} "
                f"failed={stats.failed}"
            )

            if stats.failed > 0:
                raise SystemExit(1)

            state["next_feature_index"] = start_index + batch_size
            save_state(state_path, state)

        total_sent += batch_size
        remaining -= batch_size

        print(
            "feature_batch_done "
            f"feature_group={args.feature_group} "
            f"start_index={start_index} "
            f"batch_size={batch_size} "
            f"total_processed={total_sent} "
            f"state_path={state_path} "
            f"next_feature_index={state['next_feature_index'] if not args.dry_run else start_index}"
        )

        if remaining > 0 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
