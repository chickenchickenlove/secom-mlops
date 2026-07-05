import argparse
import asyncio
import time
from collections import Counter
from typing import Any

import httpx

from secom_mlops_common.config.serving import resolve_serving_api_url
from secom_mlops_common.metrics.stats import indexed_percentile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=resolve_serving_api_url())
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--print-failures", action="store_true")
    return parser.parse_args()


def sample_id_for(index: int) -> str:
    return f"secom-{index:07d}"


async def predict_one(
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        sample_id: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    async with semaphore:
        try:
            response = await client.post(
                "/predict-by-id",
                json={"sample_id": sample_id},
            )
            latency_ms = (time.perf_counter() - started_at) * 1000

            payload = response.json()
            return {
                "sample_id": sample_id,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "payload": payload,
            }
        except httpx.RequestError as error:
            latency_ms = (time.perf_counter() - started_at) * 1000
            return {
                "sample_id": sample_id,
                "status_code": None,
                "latency_ms": latency_ms,
                "payload": {"error": str(error)},
            }
        except ValueError:
            latency_ms = (time.perf_counter() - started_at) * 1000
            return {
                "sample_id": sample_id,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "payload": {"error": response.text[:1000]},
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

    if status_code == 422:
        return "bad_request"

    if status_code == 502:
        return "model_gateway_error"

    if status_code == 503:
        return "feature_store_unavailable"

    if status_code is None:
        return "request_error"

    return "failed"


async def run_smoke(args: argparse.Namespace) -> list[dict[str, Any]]:
    timeout = httpx.Timeout(args.timeout_seconds, connect=min(args.timeout_seconds, 2.0))
    semaphore = asyncio.Semaphore(args.concurrency)

    sample_ids = [
        sample_id_for(index)
        for index in range(args.start_index, args.start_index + args.count)
    ]

    async with httpx.AsyncClient(
            base_url=args.base_url.rstrip("/"),
            timeout=timeout,
    ) as client:
        results = []

        for offset in range(0, len(sample_ids), args.concurrency):
            chunk = sample_ids[offset:offset + args.concurrency]
            chunk_results = await asyncio.gather(*[
                predict_one(client, semaphore, sample_id)
                for sample_id in chunk
            ])
            results.extend(chunk_results)

            if args.sleep_seconds > 0:
                await asyncio.sleep(args.sleep_seconds)

        return results


def print_summary(results: list[dict[str, Any]], print_failures: bool) -> None:
    categories = Counter(classify_result(result) for result in results)
    status_codes = Counter(str(result["status_code"]) for result in results)
    latencies = sorted(result["latency_ms"] for result in results)

    print("predict_by_id_smoke_complete")
    print(f"total={len(results)}")

    for name in sorted(categories):
        print(f"{name}={categories[name]}")

    print("status_codes=" + ",".join(
        f"{status}:{count}"
        for status, count in sorted(status_codes.items())
    ))

    if latencies:
        p50 = indexed_percentile(latencies, 0.50)
        p95 = indexed_percentile(latencies, 0.95)
        max_latency = max(latencies)
        print(f"latency_ms_p50={p50:.2f}")
        print(f"latency_ms_p95={p95:.2f}")
        print(f"latency_ms_max={max_latency:.2f}")

    predicted = [
        result
        for result in results
        if classify_result(result) == "predicted"
    ]

    if predicted:
        fail_count = sum(
            1
            for result in predicted
            if result["payload"].get("prediction") == 1
        )
        pass_count = len(predicted) - fail_count
        print(f"predicted_pass={pass_count}")
        print(f"predicted_fail={fail_count}")

        model_versions = Counter(
            result["payload"].get("model_version")
            for result in predicted
        )
        print("model_versions=" + ",".join(
            f"{version}:{count}"
            for version, count in sorted(model_versions.items())
        ))

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


def main() -> None:
    args = parse_args()
    results = asyncio.run(run_smoke(args))
    print_summary(results, args.print_failures)


if __name__ == "__main__":
    main()
