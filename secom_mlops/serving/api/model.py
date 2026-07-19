import asyncio

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionEventContext:
    prediction_id: str
    request_id: str
    sample_id: str
    serving_snapshot_id: str
    snapshot_version: int
    feature_hash: str
    missing_count: int


@dataclass
class PendingInvocation:
    features: list[float | None]
    future: asyncio.Future[dict[str, Any]]
    event_context: PredictionEventContext | None
    started_at: float
