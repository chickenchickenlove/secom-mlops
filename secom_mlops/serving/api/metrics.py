from typing import Literal

from prometheus_client import CollectorRegistry, Counter, REGISTRY


PredictionDestination = Literal["release", "shadow"]
PREDICTION_DESTINATIONS: tuple[PredictionDestination, ...] = (
    "release",
    "shadow",
)


class PredictionMetrics:
    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self._dispatches = Counter(
            "secom_serving_prediction_dispatch_total",
            "Predictions dispatched by the Serving API.",
            ("destination",),
            registry=registry,
        )

        for destination in PREDICTION_DESTINATIONS:
            self._dispatches.labels(destination=destination)

    def record_dispatch(
        self,
        destination: PredictionDestination,
        prediction_count: int,
    ) -> None:
        self._dispatches.labels(destination=destination).inc(prediction_count)


prediction_metrics = PredictionMetrics()
