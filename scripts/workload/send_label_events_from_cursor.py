import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from confluent_kafka import Producer

from secom_mlops_common.config.kafka import (
    resolve_kafka_bootstrap_servers,
    resolve_label_events_topic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = PROJECT_ROOT / "runtime" / "online_workload_next_label_state.json"
DEFAULT_LABEL_PATH = PROJECT_ROOT / "data" / "raw" / "secom_labels.data"


@dataclass
class PublishStats:
    attempted: int = 0
    delivered: int = 0
    failed: int = 0


def load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "next_label_index": 0,
        }

    state = json.loads(path.read_text(encoding="utf-8"))
    return {
        "next_label_index": int(state.get("next_label_index", 0)),
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**state, "updated_at": time.time()}

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def to_actual_label(actual_value: int) -> str:
    if actual_value == -1:
        return "pass"
    if actual_value == 1:
        return "fail"
    raise ValueError(f"unexpected actual_value={actual_value}")


def load_labels(label_path: str) -> list[tuple[int, str]]:
    frame = pd.read_csv(
        label_path,
        sep=r"\s+",
        header=None,
        names=["actual_value", "timestamp"],
    )

    labels = []
    for value in frame["actual_value"].tolist():
        actual_value = int(value)
        labels.append((actual_value, to_actual_label(actual_value)))

    if not labels:
        raise ValueError(f"label file is empty: {label_path}")

    return labels


def resolve_base_event_time(
        start_index: int,
        sample_interval_seconds: float,
        base_event_time: float | None,
) -> float:
    if base_event_time is not None:
        return base_event_time

    return time.time() - start_index * sample_interval_seconds


def format_ratio_for_name(ratio: float) -> str:
    return f"{ratio:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def resolve_drift_segment(args: argparse.Namespace) -> str | None:
    if args.drift_segment is not None:
        return args.drift_segment

    if args.feature_offset_direction == "none":
        return None

    ratio_name = format_ratio_for_name(args.feature_offset_ratio)
    return f"feature_offset_{args.feature_offset_direction}_{ratio_name}"


def build_label_events(
        labels: list[tuple[int, str]],
        args: argparse.Namespace,
        start_index: int,
        batch_size: int,
        simulation_run_id: str,
) -> list[dict[str, Any]]:
    base_event_time = resolve_base_event_time(
        start_index=start_index,
        sample_interval_seconds=args.sample_interval_seconds,
        base_event_time=args.base_event_time,
    )
    drift_segment = resolve_drift_segment(args)
    created_at = time.time()
    events = []

    for item_index in range(batch_size):
        global_sample_index = start_index + item_index
        source_row_index = global_sample_index % len(labels)
        sample_id = f"secom-{global_sample_index:07d}"
        actual_value, actual_label = labels[source_row_index]

        sample_base_time = (
                base_event_time
                + global_sample_index * args.sample_interval_seconds
        )
        label_available_time = sample_base_time + args.label_delay_seconds

        event = {
            "label_event_id": (
                f"{simulation_run_id}:"
                f"label:"
                f"{sample_id}:"
                f"{int(label_available_time * 1000)}"
            ),
            "sample_id": sample_id,
            "source_row_index": source_row_index,
            "event_time": label_available_time,
            "label_available_time": label_available_time,
            "actual_value": actual_value,
            "actual_label": actual_label,
            "simulation_run_id": simulation_run_id,
            "created_at": created_at,
        }

        if drift_segment is not None:
            event["drift_segment"] = drift_segment

        if args.feature_offset_direction != "none":
            event["feature_offset_direction"] = args.feature_offset_direction
            event["feature_offset_ratio"] = args.feature_offset_ratio

        events.append(event)

    return sorted_for_publish(events)


def sorted_for_publish(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_events = list(enumerate(events))
    indexed_events.sort(
        key=lambda item: (
            float(item[1]["label_available_time"]),
            item[0],
        )
    )
    return [event for _, event in indexed_events]


def delivery_callback(stats: PublishStats):
    def callback(error, message) -> None:
        if error is not None:
            stats.failed += 1
            print(f"label_delivery_failed key={message.key()!r} error={error}", file=sys.stderr)
            return

        stats.delivered += 1

    return callback


def publish_label_events(args: argparse.Namespace, events: list[dict[str, Any]]) -> PublishStats:
    producer = Producer({
        "bootstrap.servers": args.bootstrap_servers,
        "client.id": "secom-label-event-cursor-producer",
        "broker.address.family": "v4",
    })

    stats = PublishStats()
    callback = delivery_callback(stats)
    first_publish_time = float(events[0]["label_available_time"])
    started_at = time.monotonic()

    for event in events:
        if args.realtime:
            target_elapsed = (
                                     float(event["label_available_time"]) - first_publish_time
                             ) / args.time_scale
            sleep_seconds = target_elapsed - (time.monotonic() - started_at)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

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
    fail_count = sum(1 for event in events if event["actual_label"] == "fail")
    pass_count = sum(1 for event in events if event["actual_label"] == "pass")
    drift_segments = sorted({
        str(event.get("drift_segment"))
        for event in events
        if event.get("drift_segment") is not None
    })

    print(
        "label_events_built "
        f"count={len(events)} "
        f"pass={pass_count} "
        f"fail={fail_count} "
        f"first_sample_id={events[0]['sample_id']} "
        f"last_sample_id={events[-1]['sample_id']} "
        f"drift_segments={','.join(drift_segments) if drift_segments else 'none'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--label-path", default=str(DEFAULT_LABEL_PATH))
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--bootstrap-servers", default=resolve_kafka_bootstrap_servers())
    parser.add_argument("--topic", default=resolve_label_events_topic())
    parser.add_argument("--simulation-run-id-prefix", default="online_label_workload")
    parser.add_argument("--base-event-time", type=float, default=None)
    parser.add_argument("--sample-interval-seconds", type=float, default=30.0)
    parser.add_argument("--label-delay-seconds", type=float, default=300.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--feature-offset-direction",
        choices=["none", "up", "down"],
        default="none",
    )
    parser.add_argument("--feature-offset-ratio", type=float, default=0.0)
    parser.add_argument("--drift-segment", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 1:
        raise ValueError("max-samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    if args.sleep_seconds < 0:
        raise ValueError("sleep-seconds must be >= 0")
    if args.sample_interval_seconds <= 0:
        raise ValueError("sample-interval-seconds must be > 0")
    if args.label_delay_seconds < 0:
        raise ValueError("label-delay-seconds must be >= 0")
    if args.time_scale <= 0:
        raise ValueError("time-scale must be > 0")
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

    labels = load_labels(args.label_path)

    state_path = Path(args.state_path)
    state = load_state(state_path)

    remaining = args.max_samples
    total_sent = 0

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        start_index = state["next_label_index"]
        simulation_run_id = (
            f"{args.simulation_run_id_prefix}_"
            f"{start_index}_{start_index + batch_size - 1}_{int(time.time())}"
        )

        events = build_label_events(
            labels=labels,
            args=args,
            start_index=start_index,
            batch_size=batch_size,
            simulation_run_id=simulation_run_id,
        )

        summarize(events)

        if args.dry_run:
            for event in events[:5]:
                print(json.dumps(event, indent=2, allow_nan=False))
            print(
                f"dry_run_label_events={len(events)} "
                f"start_index={start_index} "
                f"batch_size={batch_size} "
                f"next_label_index_would_be={start_index + batch_size}"
            )
        else:
            stats = publish_label_events(args, events)
            print(
                "label_publish_complete "
                f"topic={args.topic} "
                f"simulation_run_id={simulation_run_id} "
                f"attempted={stats.attempted} "
                f"delivered={stats.delivered} "
                f"failed={stats.failed}"
            )

            if stats.failed > 0:
                raise SystemExit(1)

            state["next_label_index"] = start_index + batch_size
            save_state(state_path, state)

        total_sent += batch_size
        remaining -= batch_size

        print(
            "label_batch_done "
            f"start_index={start_index} "
            f"batch_size={batch_size} "
            f"total_processed={total_sent} "
            f"next_label_index={state['next_label_index'] if not args.dry_run else start_index}"
        )

        if remaining > 0 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
