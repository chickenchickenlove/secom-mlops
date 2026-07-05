import argparse

from secom_mlops.feature_store.offline_snapshot import OfflineFeatureSnapshotStore
from secom_mlops.feature_store.reconstruction import reconstruct_feature_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cutoff-time", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = reconstruct_feature_rows(
        build_cutoff_time=args.build_cutoff_time,
    )

    print(
        "offline_reconstruction_complete "
        f"rows={len(rows)} "
        f"build_cutoff_time={args.build_cutoff_time}"
    )

    for row in rows[:args.limit]:
        print(
            "offline_snapshot_row "
            f"sample_id={row.sample_id} "
            f"feature_count={row.observed_feature_count} "
            f"missing_count={row.missing_count} "
            f"is_complete={row.patch_complete} "
            f"has_no_missing_values={row.has_no_missing_values} "
            f"source_event_count={row.source_event_count} "
            f"max_event_time={row.max_event_time}"
        )

    if args.dry_run:
        print("offline_snapshot_write_skipped dry_run=true")
        return

    result = OfflineFeatureSnapshotStore().save_many(rows)

    print(
        "offline_snapshot_write_complete "
        f"attempted={result.attempted} "
        f"saved={result.saved}"
    )


if __name__ == "__main__":
    main()
