import argparse
import time

from psycopg.rows import dict_row

from secom_mlops.monitor.db import connect

SELECT_REQUEST_SQL = """
SELECT
  request_id,
  model_name,
  source_version,
  previous_version,
  target_alias,
  approval_status,
  rollout_status,
  deployed_at,
  rolled_back_at,
  notes
FROM model_deployment_requests
WHERE request_id = %s;
"""

MARK_ROLLED_BACK_SQL = """
UPDATE model_deployment_requests
SET
  rollout_status = 'rolled_back',
  rolled_back_at = %s,
  updated_at = %s
WHERE request_id = %s
RETURNING
  request_id,
  model_name,
  source_version,
  previous_version,
  target_alias,
  approval_status,
  rollout_status,
  deployed_at,
  rolled_back_at;
"""

MARK_ROLLED_BACK_WITH_NOTES_SQL = """
UPDATE model_deployment_requests
SET
  rollout_status = 'rolled_back',
  rolled_back_at = %s,
  updated_at = %s,
  notes = CASE
    WHEN notes IS NULL OR notes = '' THEN %s
    ELSE notes || E'\\n' || %s
  END
WHERE request_id = %s
RETURNING
  request_id,
  model_name,
  source_version,
  previous_version,
  target_alias,
  approval_status,
  rollout_status,
  deployed_at,
  rolled_back_at;
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--expected-source-version", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(SELECT_REQUEST_SQL, [args.request_id])
            request = cursor.fetchone()

            if request is None:
                raise RuntimeError(f"deployment request not found: request_id={args.request_id}")

            if request["rollout_status"] not in {
                "promoted",
                "canary_reloading",
                "canary_ready",
                "release_reloading",
                "deployed",
                "failed",
            }:
                raise RuntimeError(
                    "deployment request is not rollback-eligible: "
                    f"request_id={args.request_id} "
                    f"rollout_status={request['rollout_status']}"
                )

            if (
                    args.expected_source_version is not None
                    and str(request["source_version"]) != str(args.expected_source_version)
            ):
                raise RuntimeError(
                    "deployment request source version mismatch: "
                    f"request_id={args.request_id} "
                    f"expected={args.expected_source_version} "
                    f"actual={request['source_version']}"
                )

            if args.dry_run:
                print(
                    "model_deployment_rollback_mark_dry_run "
                    f"request_id={request['request_id']} "
                    f"model_name={request['model_name']} "
                    f"source_version={request['source_version']} "
                    f"rollback_version={request['previous_version']} "
                    f"target_alias={request['target_alias']} "
                    f"approval_status={request['approval_status']} "
                    f"rollout_status={request['rollout_status']}"
                )
                return

            now = time.time()

            if args.notes is None:
                cursor.execute(MARK_ROLLED_BACK_SQL, [now, now, args.request_id])
            else:
                cursor.execute(
                    MARK_ROLLED_BACK_WITH_NOTES_SQL,
                    [now, now, args.notes, args.notes, args.request_id],
                )

            updated = cursor.fetchone()

    print(
        "model_deployment_marked_rolled_back "
        f"request_id={updated['request_id']} "
        f"model_name={updated['model_name']} "
        f"source_version={updated['source_version']} "
        f"rollback_version={updated['previous_version']} "
        f"target_alias={updated['target_alias']} "
        f"approval_status={updated['approval_status']} "
        f"rollout_status={updated['rollout_status']} "
        f"rolled_back_at={updated['rolled_back_at']}"
    )


if __name__ == "__main__":
    main()
