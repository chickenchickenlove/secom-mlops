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


def build_label_events(
        labels: list[tuple[int, str]],
        start_index: int,
        batch_size: int,
) -> list[dict[str, Any]]:
    events = []

    for item_index in range(batch_size):
        global_sample_index = start_index + item_index
        source_row_index = global_sample_index % len(labels)
        sample_id = f"secom-{global_sample_index:07d}"
        actual_value, actual_label = labels[source_row_index]

        event = {
            "label_event_id": f"label:{sample_id}:r1",
            "sample_id": sample_id,
            "label_revision": 1,
            "actual_value": actual_value,
            "actual_label": actual_label,
        }

        events.append(event)

    return events


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

    for event in events:
        event["measured_at"] = time.time()
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

    print(
        "label_events_built "
        f"count={len(events)} "
        f"pass={pass_count} "
        f"fail={fail_count} "
        f"first_sample_id={events[0]['sample_id']} "
        f"last_sample_id={events[-1]['sample_id']}"
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 1:
        raise ValueError("max-samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    if args.sleep_seconds < 0:
        raise ValueError("sleep-seconds must be >= 0")


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

        events = build_label_events(
            labels=labels,
            start_index=start_index,
            batch_size=batch_size,
        )

        summarize(events)

        if args.dry_run:
            for event in events[:5]:
                preview_event = {**event, "measured_at": time.time()}
                print(json.dumps(preview_event, indent=2, allow_nan=False))
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
