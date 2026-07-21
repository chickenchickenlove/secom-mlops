import unittest

from prometheus_client import CollectorRegistry

from secom_mlops.serving.api.metrics import PredictionMetrics


class PredictionMetricsTest(unittest.TestCase):
    def test_initializes_every_prediction_destination(self) -> None:
        registry = CollectorRegistry()
        PredictionMetrics(registry)

        for destination in ("release", "canary", "shadow"):
            self.assertEqual(
                0,
                registry.get_sample_value(
                    "secom_serving_prediction_dispatch_total",
                    {"destination": destination},
                ),
            )

    def test_records_prediction_count_by_destination(self) -> None:
        registry = CollectorRegistry()
        metrics = PredictionMetrics(registry)

        metrics.record_dispatch("release", 3)
        metrics.record_dispatch("canary", 1)
        metrics.record_dispatch("shadow", 2)

        self.assertEqual(
            3,
            registry.get_sample_value(
                "secom_serving_prediction_dispatch_total",
                {"destination": "release"},
            ),
        )
        self.assertEqual(
            1,
            registry.get_sample_value(
                "secom_serving_prediction_dispatch_total",
                {"destination": "canary"},
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
