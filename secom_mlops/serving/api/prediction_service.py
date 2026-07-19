import logging

from typing import Any

from secom_mlops.serving.api.batch import PredictionBatcher
from secom_mlops.serving.api.model import (
    PredictionEventContext,
)

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, primary_batcher: PredictionBatcher) -> None:
        self._primary_batcher = primary_batcher
        self._shadow_batcher = None

    def start(self) -> None:
        self._primary_batcher.start()

    async def close(self) -> None:
        await self._primary_batcher.close()

    async def predict(self, features: list[float | None], *, event_context: PredictionEventContext) -> dict[str, Any]:
        return await self._primary_batcher.invoke(
            features,
            event_context=event_context,
        )

    async def predict_debug_many(
        self,
        inputs: list[list[float | None]],
    ) -> list[dict[str, Any]]:
        return await self._primary_batcher.invoke_many(inputs)
