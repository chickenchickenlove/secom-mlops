import unittest

from prometheus_client import CollectorRegistry

from secom_mlops.serving.api.metrics import PredictionMetrics


class PredictionMetricsTest(unittest.TestCase):
    def test_records_prediction_count_by_destination(self) -> None:
        registry = CollectorRegistry()
        metrics = PredictionMetrics(registry)

        metrics.record_dispatch("release", 3)
        metrics.record_dispatch("shadow", 2)

        self.assertEqual(
            3,
            registry.get_sample_value(
                "secom_serving_prediction_dispatch_total",
                {"destination": "release"},
            ),
        )
        self.assertEqual(
            2,
            registry.get_sample_value(
                "secom_serving_prediction_dispatch_total",
                {"destination": "shadow"},
            ),
        )


if __name__ == "__main__":
    unittest.main()
