import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from secom_mlops.datasets.dataset_artifacts import write_artifacts
from secom_mlops.datasets.training_dataset import (
    DATASET_SCHEMA_VERSION,
    DATASET_TYPE,
    SELECTOR_VERSION,
    DatasetBuildConfig,
    DatasetMember,
    build_dataset_frame,
    build_dataset_identity,
    build_manifest,
    evaluate_readiness,
)
from secom_mlops.datasets.training_dataset_loader import (
    load_training_dataset,
    verify_downloaded_artifacts,
)
from secom_mlops_common.schemas.secom import FEATURE_KEYS


class TrainingDatasetLoaderTest(unittest.TestCase):

    def test_rejects_a_non_ready_catalog_row_before_download(self) -> None:
        catalog = {
            "dataset_id": "training_not_ready",
            "dataset_type": DATASET_TYPE,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "selector_version": SELECTOR_VERSION,
            "status": "BUILDING",
        }

        with patch(
            "secom_mlops.datasets.training_dataset_loader.get_dataset_build",
            return_value=catalog,
        ):
            with self.assertRaisesRegex(RuntimeError, "catalog.status"):
                load_training_dataset(
                    "training_not_ready",
                    tracking_uri="http://mlflow:5100",
                )

    def test_verifies_catalog_manifest_and_parquet_together(self) -> None:
        config = DatasetBuildConfig(
            cohort_start_time=0.0,
            cutoff_time=10.0,
            label_maturity_seconds=0.0,
            min_labeled_samples=1,
            min_label_coverage=1.0,
            min_fail_samples=1,
            min_pass_samples=1,
        )
        members = [self._member(0, 1), self._member(1, -1)]
        identity = build_dataset_identity(config, members)
        frame = build_dataset_frame(identity.dataset_id, members)
        stats = evaluate_readiness(config, members).stats
        manifest = build_manifest(config, identity, stats)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            artifact_sha256 = write_artifacts(artifact_dir, frame, manifest, stats)
            loaded = verify_downloaded_artifacts(
                artifact_dir,
                self._catalog(
                    config=config,
                    identity=identity,
                    stats=stats,
                    artifact_sha256=artifact_sha256,
                ),
            )

        self.assertEqual(identity.dataset_id, loaded.dataset_id)
        self.assertEqual(2, len(loaded.frame))

    def test_rejects_catalog_artifact_hash_mismatch(self) -> None:
        config = DatasetBuildConfig(
            cohort_start_time=0.0,
            cutoff_time=10.0,
            label_maturity_seconds=0.0,
            min_labeled_samples=1,
            min_label_coverage=1.0,
            min_fail_samples=1,
            min_pass_samples=1,
        )
        members = [self._member(0, 1), self._member(1, -1)]
        identity = build_dataset_identity(config, members)
        frame = build_dataset_frame(identity.dataset_id, members)
        stats = evaluate_readiness(config, members).stats
        manifest = build_manifest(config, identity, stats)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            write_artifacts(artifact_dir, frame, manifest, stats)
            catalog = self._catalog(
                config=config,
                identity=identity,
                stats=stats,
                artifact_sha256="sha256:v1:not-the-artifact",
            )
            with self.assertRaisesRegex(RuntimeError, "catalog.artifact_sha256"):
                verify_downloaded_artifacts(artifact_dir, catalog)

    @staticmethod
    def _member(index: int, actual_value: int) -> DatasetMember:
        actual_label = "fail" if actual_value == 1 else "pass"
        return DatasetMember(
            sample_id=f"sample-{index}",
            serving_snapshot_id=f"snapshot-{index}",
            snapshot_version=1,
            feature_hash="sha256:v1:" + str(index) * 64,
            snapshot_time=float(index),
            snapshot_available_at=float(index),
            window_start=float(index),
            window_end=float(index + 1),
            feature_count=len(FEATURE_KEYS),
            serving_missing_count=0,
            simulation_run_id=None,
            drift_segment=None,
            label_event_id=f"label-{index}",
            label_revision=1,
            label_measured_at=float(index + 1),
            label_available_at=float(index + 2),
            actual_value=actual_value,
            actual_label=actual_label,
            features_json={key: float(index) for key in FEATURE_KEYS},
        )

    @staticmethod
    def _catalog(*, config, identity, stats, artifact_sha256):
        return {
            "dataset_id": identity.dataset_id,
            "manifest_hash": identity.manifest_hash,
            "artifact_sha256": artifact_sha256,
            "mlflow_run_id": "dataset-run-1",
            "artifact_uri": "mlflow-artifacts:/data/dataset.parquet",
            "cohort_start_time": config.cohort_start_time,
            "cutoff_time": config.cutoff_time,
            "label_maturity_seconds": config.label_maturity_seconds,
            **{
                name: stats[name]
                for name in (
                    "eligible_sample_count",
                    "labeled_sample_count",
                    "unlabeled_sample_count",
                    "label_coverage",
                    "fail_count",
                    "pass_count",
                )
            },
        }


if __name__ == "__main__":
    unittest.main()
