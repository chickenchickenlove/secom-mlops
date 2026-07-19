import json
import unittest
from unittest.mock import patch

from secom_mlops.feature_store.online_snapshot_reader import (
    InvalidOnlineFeatureSnapshot,
    OnlineFeatureSnapshotStore,
    _parse_snapshot,
)

FEATURE_HASH = "sha256:v1:" + "0" * 64


class OnlineSnapshotReaderTest(unittest.TestCase):
    def test_builds_client_from_explicit_host_config(self) -> None:
        with patch(
            "secom_mlops.feature_store.online_snapshot_reader.valkey.Valkey"
        ) as valkey_type:
            store = OnlineFeatureSnapshotStore(
                valkey_url=None,
                valkey_host="valkey",
                valkey_port=6380,
                valkey_database=2,
                timeout_seconds=3.5,
                key_prefix="snapshots",
            )

        valkey_type.assert_called_once_with(
            host="valkey",
            port=6380,
            db=2,
            decode_responses=True,
            socket_timeout=3.5,
            socket_connect_timeout=3.5,
        )

        store.close()
        valkey_type.return_value.close.assert_called_once_with()

    def test_builds_client_from_explicit_url_config(self) -> None:
        with patch(
            "secom_mlops.feature_store.online_snapshot_reader.valkey.Valkey"
        ) as valkey_type:
            store = OnlineFeatureSnapshotStore(
                valkey_url="valkey://valkey:6379/3",
                valkey_host="ignored",
                valkey_port=1,
                valkey_database=0,
                timeout_seconds=1.5,
                key_prefix="snapshots",
            )

        valkey_type.from_url.assert_called_once_with(
            "valkey://valkey:6379/3",
            decode_responses=True,
            socket_timeout=1.5,
            socket_connect_timeout=1.5,
        )

        store.close()
        valkey_type.from_url.return_value.close.assert_called_once_with()

    def test_parses_matching_positive_snapshot_version(self) -> None:
        snapshot = _parse_snapshot(
            json.dumps(_valid_payload()),
            expected_sample_id="secom-0000001",
        )

        self.assertEqual("state:secom-0000001:1000:3", snapshot.serving_snapshot_id)
        self.assertEqual(3, snapshot.snapshot_version)
        self.assertEqual(FEATURE_HASH, snapshot.feature_hash)
        self.assertEqual("secom-0000001", snapshot.sample_id)
        self.assertEqual(590, len(snapshot.values))

    def test_rejects_missing_version_fields(self) -> None:
        for field in ("source_event_count", "snapshot_version"):
            with self.subTest(field=field):
                payload = _valid_payload()
                del payload[field]

                with self.assertRaisesRegex(
                    InvalidOnlineFeatureSnapshot,
                    rf"{field} must be an integer",
                ):
                    _parse_snapshot(
                        json.dumps(payload),
                        expected_sample_id="secom-0000001",
                    )

    def test_rejects_non_integer_version_fields(self) -> None:
        for field in ("source_event_count", "snapshot_version"):
            for value in (True, 1.0, "1", None):
                with self.subTest(field=field, value=value):
                    payload = _valid_payload()
                    payload[field] = value

                    with self.assertRaisesRegex(
                        InvalidOnlineFeatureSnapshot,
                        rf"{field} must be an integer",
                    ):
                        _parse_snapshot(
                            json.dumps(payload),
                            expected_sample_id="secom-0000001",
                        )

    def test_rejects_non_positive_version_fields(self) -> None:
        for field in ("source_event_count", "snapshot_version"):
            for value in (0, -1):
                with self.subTest(field=field, value=value):
                    payload = _valid_payload()
                    payload[field] = value

                    with self.assertRaisesRegex(
                        InvalidOnlineFeatureSnapshot,
                        rf"{field} must be >= 1",
                    ):
                        _parse_snapshot(
                            json.dumps(payload),
                            expected_sample_id="secom-0000001",
                        )

    def test_rejects_snapshot_version_mismatch(self) -> None:
        payload = _valid_payload()
        payload["snapshot_version"] = 4

        with self.assertRaisesRegex(
            InvalidOnlineFeatureSnapshot,
            "snapshot_version must match source_event_count",
        ):
            _parse_snapshot(
                json.dumps(payload),
                expected_sample_id="secom-0000001",
            )

    def test_rejects_missing_or_invalid_feature_hash(self) -> None:
        missing_hash = _valid_payload()
        del missing_hash["feature_hash"]

        with self.assertRaisesRegex(
            InvalidOnlineFeatureSnapshot,
            "feature_hash must be a non-empty string",
        ):
            _parse_snapshot(
                json.dumps(missing_hash),
                expected_sample_id="secom-0000001",
            )

        invalid_hash = _valid_payload()
        invalid_hash["feature_hash"] = "sha256:v1:not-a-hash"

        with self.assertRaisesRegex(
            InvalidOnlineFeatureSnapshot,
            "invalid feature_hash",
        ):
            _parse_snapshot(
                json.dumps(invalid_hash),
                expected_sample_id="secom-0000001",
            )


def _valid_payload() -> dict[str, object]:
    return {
        "serving_snapshot_id": "state:secom-0000001:1000:3",
        "source_event_count": 3,
        "snapshot_version": 3,
        "feature_hash": FEATURE_HASH,
        "sample_id": "secom-0000001",
        "snapshot_time": 1.0,
        "snapshot_status": "complete",
        "feature_count": 590,
        "missing_count": 0,
        "is_complete": True,
        "features": {
            f"f{index:03d}": float(index)
            for index in range(590)
        },
    }


if __name__ == "__main__":
    unittest.main()
