import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from secom_mlops_common.config.serving import resolve_serving_api_url
from secom_mlops_common.metrics.stats import indexed_percentile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = PROJECT_ROOT / "runtime" / "online_workload_next_predict_state.json"


def load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "next_predict_index": 0,
        }

    state = json.loads(path.read_text(encoding="utf-8"))
    return {
        "next_predict_index": int(state.get("next_predict_index", 0)),
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**state, "updated_at": time.time()}

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def sample_id_for(index: int) -> str:
    return f"secom-{index:07d}"


async def predict_one(
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        index: int,
) -> dict[str, Any]:
    sample_id = sample_id_for(index)
    started_at = time.perf_counter()

    async with semaphore:
        try:
            response = await client.post(
                "/predict-by-id",
                json={"sample_id": sample_id},
            )
            payload = response.json()
        except httpx.RequestError as error:
            return {
                "index": index,
                "sample_id": sample_id,
                "status_code": None,
                "latency_ms": (time.perf_counter() - started_at) * 1000,
                "payload": {"error": str(error)},
            }
        except ValueError:
            return {
                "index": index,
                "sample_id": sample_id,
                "status_code": response.status_code,
                "latency_ms": (time.perf_counter() - started_at) * 1000,
                "payload": {"error": response.text[:1000]},
            }

        return {
            "index": index,
            "sample_id": sample_id,
            "status_code": response.status_code,
            "latency_ms": (time.perf_counter() - started_at) * 1000,
            "payload": payload,
        }


def classify_result(result: dict[str, Any]) -> str:
    status_code = result["status_code"]
    payload = result["payload"]

    if status_code == 200:
        return "predicted"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        if isinstance(payload, dict) and payload.get("retryable") is True:
            return "partial"
        return "not_ready_final"
    if status_code == 502:
        return "model_gateway_error"
    if status_code == 503:
        return "feature_store_unavailable"
    if status_code is None:
        return "request_error"

    return "failed"


async def predict_batch(args: argparse.Namespace, start_index: int, batch_size: int) -> list[dict[str, Any]]:
    timeout = httpx.Timeout(args.timeout_seconds, connect=min(args.timeout_seconds, 2.0))
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
            base_url=args.base_url.rstrip("/"),
            timeout=timeout,
    ) as client:
        return await asyncio.gather(*[
            predict_one(client, semaphore, index)
            for index in range(start_index, start_index + batch_size)
        ])


def leading_predicted_count(results: list[dict[str, Any]]) -> int:
    count = 0

    for result in sorted(results, key=lambda item: item["index"]):
        if classify_result(result) != "predicted":
            break
        count += 1

    return count


def print_summary(results: list[dict[str, Any]], print_failures: bool) -> None:
    categories = Counter(classify_result(result) for result in results)
    latencies = sorted(result["latency_ms"] for result in results)

    print("prediction_batch_complete")
    print(f"total={len(results)}")

    for name in sorted(categories):
        print(f"{name}={categories[name]}")

    if latencies:
        print(f"latency_ms_p50={indexed_percentile(latencies, 0.50):.2f}")
        print(f"latency_ms_p95={indexed_percentile(latencies, 0.95):.2f}")
        print(f"latency_ms_max={max(latencies):.2f}")

    if print_failures:
        for result in results:
            category = classify_result(result)
            if category == "predicted":
                continue

            print(
                "non_success "
                f"sample_id={result['sample_id']} "
                f"category={category} "
                f"status_code={result['status_code']} "
                f"payload={result['payload']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--base-url", default=resolve_serving_api_url())
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--print-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_samples < 1:
        raise ValueError("max-samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    if args.sleep_seconds < 0:
        raise ValueError("sleep-seconds must be >= 0")

    state_path = Path(args.state_path)
    state = load_state(state_path)

    remaining = args.max_samples
    total_requested = 0

    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        start_index = state["next_predict_index"]

        results = asyncio.run(predict_batch(args, start_index, batch_size))
        print_summary(results, args.print_failures)

        advance_count = leading_predicted_count(results)
        state["next_predict_index"] = start_index + advance_count
        save_state(state_path, state)

        total_requested += batch_size
        remaining -= batch_size

        print(
            "prediction_cursor_updated "
            f"requested_start={start_index} "
            f"requested_batch_size={batch_size} "
            f"leading_predicted={advance_count} "
            f"total_requested={total_requested} "
            f"next_predict_index={state['next_predict_index']}"
        )

        if advance_count < batch_size:
            print("prediction_request_stopped reason=non_success")
            break

        if remaining > 0 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
