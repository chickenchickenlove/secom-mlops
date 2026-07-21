import unittest
from unittest.mock import patch

from secom_mlops.monitor.prediction_events import PredictionEventProducer


class PredictionEventProducerConfigTest(unittest.TestCase):
    def test_uses_explicit_kafka_config(self) -> None:
        with patch(
            "secom_mlops.monitor.prediction_events.Producer"
        ) as producer_type:
            producer = PredictionEventProducer(
                bootstrap_servers="kafka:29092",
                topic="prediction-events",
                client_id="serving-api",
                flush_timeout_seconds=3.5,
            )

        producer_type.assert_called_once_with({
            "bootstrap.servers": "kafka:29092",
            "client.id": "serving-api",
            "broker.address.family": "v4",
        })

        producer_type.return_value.flush.return_value = 0
        producer.close()
        producer_type.return_value.flush.assert_called_once_with(3.5)


if __name__ == "__main__":
    unittest.main()
