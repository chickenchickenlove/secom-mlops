import argparse
import time
from uuid import uuid4

from secom_mlops.feature_store.offline_snapshot_reader import (
    load_offline_feature_snapshots,
    offline_snapshots_to_dataframe,
)
from secom_mlops.monitor.offline_prediction_logs import OfflinePredictionLogStore
from secom_mlops.models.predict_with_mlflow_model import bootstrap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cutoff-time", type=float, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-partial-patches", action="store_true")
    parser.add_argument("--write-offline-prediction-logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    snapshots = load_offline_feature_snapshots(
        build_cutoff_time=args.build_cutoff_time,
        only_complete=not args.include_partial_patches,
    )

    if args.limit is not None:
        snapshots = snapshots[:args.limit]

    if not snapshots:
        print("no_offline_snapshots_found")
        return

    features = offline_snapshots_to_dataframe(snapshots)
    model = bootstrap_model()

    started_at = time.perf_counter()
    predictions = model.predict(features.copy())
    predicted_at = time.time()
    latency_ms = (time.perf_counter() - started_at) * 1000

    log_rows = []

    print(
        "offline_prediction_complete "
        f"rows={len(snapshots)} "
        f"build_cutoff_time={args.build_cutoff_time} "
        f"model_run_id={model.run_id}"
    )

    for snapshot, prediction in zip(snapshots, predictions):
        print(
            "offline_prediction "
            f"sample_id={snapshot.sample_id} "
            f"offline_snapshot_id={snapshot.offline_snapshot_id} "
            f"fail_probability={prediction['fail_probability']:.6f} "
            f"prediction={prediction['prediction']} "
            f"label={prediction['label']} "
            f"threshold={prediction['threshold']} "
            f"missing_count={snapshot.missing_count}"
        )

        if args.write_offline_prediction_logs:
            log_rows.append({
                "offline_prediction_id": str(uuid4()),
                "offline_snapshot_id": snapshot.offline_snapshot_id,
                "sample_id": snapshot.sample_id,
                "build_cutoff_time": snapshot.build_cutoff_time,
                "model_run_id": model.run_id,
                "predicted_at": predicted_at,
                "fail_probability": prediction["fail_probability"],
                "predicted_value": prediction["prediction"],
                "predicted_label": prediction["label"],
                "threshold": prediction["threshold"],
                "missing_count": snapshot.missing_count,
                "latency_ms": latency_ms,
            })

    if args.write_offline_prediction_logs:
        result = OfflinePredictionLogStore().save_many(log_rows)
        print(
            "offline_prediction_logs_saved "
            f"attempted={result.attempted} "
            f"saved={result.saved}"
        )


if __name__ == "__main__":
    main()
