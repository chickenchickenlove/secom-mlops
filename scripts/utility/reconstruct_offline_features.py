import argparse
import json

from secom_mlops.feature_store.reconstruction import reconstruct_feature_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cutoff-time", type=float, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--print-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = reconstruct_feature_rows(
        build_cutoff_time=args.build_cutoff_time,
    )

    print(
        "reconstruction_complete "
        f"rows={len(rows)} "
        f"build_cutoff_time={args.build_cutoff_time}"
    )

    for row in rows[:args.limit]:
        print(
            "feature_row "
            f"sample_id={row.sample_id} "
            f"observed_feature_count={row.observed_feature_count} "
            f"missing_count={row.missing_count} "
            f"patch_complete={row.patch_complete} "
            f"has_no_missing_values={row.has_no_missing_values} "
            f"source_event_count={row.source_event_count} "
            f"max_event_time={row.max_event_time}"
        )

        if args.print_features:
            print(json.dumps(row.features_json, sort_keys=True))


if __name__ == "__main__":
    main()