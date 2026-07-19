import json
import logging
from typing import Any

from confluent_kafka import Producer

from secom_mlops_common.config.kafka import (
    resolve_kafka_bootstrap_servers,
    resolve_prediction_event_client_id,
    resolve_prediction_event_flush_timeout_seconds,
    resolve_prediction_events_topic,
)

logger = logging.getLogger(__name__)

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

        def callback(error, message) -> None:
            # This should be lighten.
            # Consider warning logging level.
            if error is not None:
                logger.warning(
                    "prediction_event_delivery_failed "
                    "topic=%s partition=%s offset=%s error=%s",
                    message.topic(),
                    message.partition(),
                    message.offset(),
                    error
                )

        for event in events:
            value = json.dumps(event, separators=(",", ":"), allow_nan=False).encode("utf-8")
            key = str(event["sample_id"]).encode("utf-8")

            try:
                # Message will be stored in producer batch.
                # This does not means that messages are sent to broker right now.
                self._producer.produce(topic=self._topic, key=key, value=value, on_delivery=callback,)
            except BufferError:
                logger.warning(
                    "prediction_event_local_queue_full "
                    "sample_id=%s prediction_id=%s",
                    event.get("sample_id"),
                    event.get("prediction_id"),
                )
                self._producer.poll(0)
                continue

            self._producer.poll(0)


    def close(self) -> None:
        remaining = self._producer.flush(self._flush_timeout_seconds)
        if remaining > 0:
            logger.warning(
                "prediction_event_flush_timeout remaining_messages=%d",
                remaining,
            )
