import json
from typing import Any

from confluent_kafka import Producer

from secom_mlops_common.config.kafka import (
    resolve_kafka_bootstrap_servers,
    resolve_prediction_event_client_id,
    resolve_prediction_event_flush_timeout_seconds,
    resolve_prediction_events_topic,
)


class PredictionEventProducer:
    def __init__(self) -> None:
        bootstrap_servers = resolve_kafka_bootstrap_servers()
        self._topic = resolve_prediction_events_topic()
        self._flush_timeout_seconds = resolve_prediction_event_flush_timeout_seconds()
        self._producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "client.id": resolve_prediction_event_client_id(),
            "broker.address.family": "v4",
        })

    def publish_many(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return

        delivery_errors: list[str] = []

        def callback(error, message) -> None:
            if error is not None:
                delivery_errors.append(
                    f"topic={message.topic()} partition={message.partition()} "
                    f"offset={message.offset()} error={error}"
                )

        for event in events:
            value = json.dumps(event, separators=(",", ":"), allow_nan=False).encode("utf-8")
            key = str(event["sample_id"]).encode("utf-8")

            while True:
                try:
                    self._producer.produce(
                        topic=self._topic,
                        key=key,
                        value=value,
                        on_delivery=callback,
                    )
                    break
                except BufferError:
                    self._producer.poll(1.0)

            self._producer.poll(0)

        remaining = self._producer.flush(self._flush_timeout_seconds)
        if remaining > 0:
            raise RuntimeError(f"prediction_event_flush_timeout remaining_messages={remaining}")
        if delivery_errors:
            raise RuntimeError("prediction_event_delivery_failed " + "; ".join(delivery_errors))

    def close(self) -> None:
        self._producer.flush(5.0)
