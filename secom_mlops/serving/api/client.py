import logging
import httpx

from typing import Any

from secom_mlops.serving.api.errors import (
    ModelGatewayError,
)


logger = logging.getLogger(__name__)


class ModelGatewayClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(
                timeout_seconds,
                connect=min(timeout_seconds, 2.0),
            ),
        )
        # TODO: Use argument.
        self.path = "/invocations"

    async def close(self) -> None:
        await self._client.aclose()

    async def invoke_batch(self, inputs: list[list[float | None]]) -> list[dict[str, Any]]:
        try:
            response = await self._client.post(
                self.path,
                json={"inputs": inputs}
            )
        except httpx.RequestError as error:
            raise ModelGatewayError("model gateway unavailable") from error

        if response.status_code >= 400:
            raise ModelGatewayError(
                f"model gateway failed: status={response.status_code} body={response.text[:1000]}"
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise ModelGatewayError("model gateway returned invalid JSON") from error

        if not isinstance(payload, dict):
            raise ModelGatewayError("invalid model gateway response")

        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(inputs):
            raise ModelGatewayError("invalid model gateway response")
        if any(not isinstance(prediction, dict) for prediction in predictions):
            raise ModelGatewayError("model gateway prediction must be an object")

        return predictions