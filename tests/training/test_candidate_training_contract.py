import unittest

import numpy as np
import pandas as pd

from secom_mlops.training.random_forest_training import split_indices
from secom_mlops.training.training_data_preparation import (
    DEVELOPMENT_SAMPLE_SELECTION,
    MAX_DEVELOPMENT_SAMPLES,
    VALIDATION_SIZE,
    _selection_hash,
    prepare_training_dataset,
    select_latest_labeled_training_rows,
    validate_training_data,
)
from secom_mlops.datasets.training_dataset_loader import LoadedTrainingDataset
from secom_mlops_common.schemas.secom import MODEL_COLUMNS


class CandidateTrainingContractTest(unittest.TestCase):

    def test_development_cohort_is_capped_at_one_thousand(self) -> None:
        self.assertEqual(1000, MAX_DEVELOPMENT_SAMPLES)
        self.assertEqual(
            "latest_labeled_snapshot_available_at",
            DEVELOPMENT_SAMPLE_SELECTION,
        )

    def test_selects_latest_one_thousand_after_filtering_labeled_rows(self) -> None:
        frame = pd.DataFrame({
            "sample_id": [f"sample-{index:04d}" for index in range(1002)],
            "serving_snapshot_id": [f"snapshot-{index:04d}" for index in range(1002)],
            "snapshot_version": [1] * 1002,
            "snapshot_available_at": [float(index) for index in range(1002)],
            "label_event_id": [f"label-{index:04d}" for index in range(1001)] + [None],
        })

        selected = select_latest_labeled_training_rows(
            frame,
            dataset_id="training_test",
        )

        self.assertEqual(MAX_DEVELOPMENT_SAMPLES, len(selected))
        self.assertNotIn("sample-1001", set(selected["sample_id"]))
        self.assertNotIn("sample-0000", set(selected["sample_id"]))
        self.assertEqual("sample-0001", selected.iloc[0]["sample_id"])
        self.assertEqual("sample-1000", selected.iloc[-1]["sample_id"])
        self.assertRegex(
            _selection_hash("training_test", selected),
            r"^sha256:v1:[0-9a-f]{64}$",
        )

    def test_training_selection_fails_below_one_thousand_labeled_rows(self) -> None:
        frame = pd.DataFrame({
            "sample_id": [f"sample-{index:04d}" for index in range(1000)],
            "serving_snapshot_id": [f"snapshot-{index:04d}" for index in range(1000)],
            "snapshot_version": [1] * 1000,
            "snapshot_available_at": [float(index) for index in range(1000)],
            "label_event_id": [f"label-{index:04d}" for index in range(999)] + [None],
        })

        with self.assertRaisesRegex(ValueError, "required=1000 actual=999"):
            select_latest_labeled_training_rows(
                frame,
                dataset_id="training_test",
            )

    def test_training_rejects_low_point_in_time_label_coverage(self) -> None:
        metadata = {
            "eligible_cohort_count": 1000,
            "labeled_cohort_count": 900,
            "label_coverage": 0.9,
            "sample_count": 900,
            "fail_count": 60,
            "pass_count": 840,
        }

        with self.assertRaisesRegex(
                ValueError,
                "point-in-time label coverage below training minimum",
        ):
            validate_training_data(
                metadata=metadata,
                min_samples=500,
                min_label_coverage=0.95,
                min_fail_samples=20,
                min_pass_samples=20,
            )

    def test_training_accepts_coverage_at_minimum(self) -> None:
        metadata = {
            "eligible_cohort_count": 1000,
            "labeled_cohort_count": 950,
            "label_coverage": 0.95,
            "sample_count": 950,
            "fail_count": 60,
            "pass_count": 890,
        }

        validate_training_data(
            metadata=metadata,
            min_samples=500,
            min_label_coverage=0.95,
            min_fail_samples=20,
            min_pass_samples=20,
        )

    def test_training_rejects_insufficient_samples_for_each_gate(self) -> None:
        valid = {
            "eligible_cohort_count": 1000,
            "labeled_cohort_count": 1000,
            "label_coverage": 1.0,
            "sample_count": 1000,
            "fail_count": 100,
            "pass_count": 900,
        }
        cases = (
            ({**valid, "sample_count": 999}, "not enough labeled"),
            ({**valid, "fail_count": 19}, "not enough fail"),
            ({**valid, "pass_count": 19}, "not enough pass"),
        )

        for metadata, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_training_data(
                        metadata=metadata,
                        min_samples=1000,
                        min_label_coverage=0.95,
                        min_fail_samples=20,
                        min_pass_samples=20,
                    )

    def test_prepares_model_inputs_and_dataset_lineage(self) -> None:
        row_count = MAX_DEVELOPMENT_SAMPLES
        data = {
            "sample_id": [f"sample-{index:04d}" for index in range(row_count)],
            "serving_snapshot_id": [
                f"snapshot-{index:04d}" for index in range(row_count)
            ],
            "snapshot_version": [1] * row_count,
            "snapshot_time": np.arange(row_count, dtype="float64"),
            "snapshot_available_at": np.arange(row_count, dtype="float64"),
            "window_start": np.arange(row_count, dtype="float64") - 1.0,
            "window_end": np.arange(row_count, dtype="float64"),
            "serving_missing_count": [0] * row_count,
            "label_event_id": [f"label-{index:04d}" for index in range(row_count)],
            "label_revision": [1] * row_count,
            "label_measured_at": np.arange(row_count, dtype="float64") + 1.0,
            "label_available_at": np.arange(row_count, dtype="float64") + 2.0,
            "actual_value": [1 if index % 2 == 0 else -1 for index in range(row_count)],
        }
        data.update({
            column: np.full(row_count, float(index), dtype="float64")
            for index, column in enumerate(MODEL_COLUMNS)
        })
        loaded = LoadedTrainingDataset(
            dataset_id="training_test",
            manifest_hash="sha256:v1:manifest",
            artifact_sha256="sha256:v1:artifact",
            mlflow_run_id="dataset-run",
            artifact_uri="mlflow-artifacts:/data/dataset.parquet",
            frame=pd.DataFrame(data),
            manifest={
                "identity": {
                    "cohort_start_time": 0.0,
                    "label_maturity_seconds": 60.0,
                    "simulation_run_id": "simulation-1",
                    "drift_segment": "stable",
                },
                "build_context": {
                    "cohort_end_time": 1000.0,
                    "cutoff_time": 1060.0,
                },
            },
        )

        prepared = prepare_training_dataset(loaded)

        self.assertEqual(list(MODEL_COLUMNS), list(prepared.features.columns))
        self.assertTrue((prepared.features.dtypes == "float64").all())
        self.assertEqual("int64", str(prepared.targets.dtype))
        self.assertEqual(row_count, len(prepared.sample_ids))
        self.assertEqual(
            prepared.sample_ids,
            prepared.selected_rows["sample_id"].astype(str).tolist(),
        )
        self.assertEqual("training_test", prepared.metadata["dataset_id"])
        self.assertEqual(
            "sha256:v1:manifest",
            prepared.metadata["dataset_manifest_hash"],
        )
        self.assertEqual(500, prepared.metadata["fail_count"])
        self.assertEqual(500, prepared.metadata["pass_count"])
        self.assertEqual(1.0, prepared.metadata["label_coverage"])
        self.assertRegex(
            prepared.metadata["dataset_selection_hash"],
            r"^sha256:v1:[0-9a-f]{64}$",
        )

    def test_split_is_disjoint_stratified_eighty_twenty(self) -> None:
        labels = pd.Series(
            [1] * 76 + [-1] * (MAX_DEVELOPMENT_SAMPLES - 76),
            dtype="int64",
        )

        train_indices, validation_indices = split_indices(
            labels,
            validation_size=VALIDATION_SIZE,
            random_state=42,
        )

        self.assertEqual(800, len(train_indices))
        self.assertEqual(200, len(validation_indices))
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(
            set(range(MAX_DEVELOPMENT_SAMPLES)),
            set(train_indices) | set(validation_indices),
        )

        train_labels = labels.iloc[np.asarray(train_indices)]
        validation_labels = labels.iloc[np.asarray(validation_indices)]
        self.assertEqual({-1, 1}, set(train_labels))
        self.assertEqual({-1, 1}, set(validation_labels))
        self.assertAlmostEqual(
            float((labels == 1).mean()),
            float((train_labels == 1).mean()),
            places=2,
        )
        self.assertAlmostEqual(
            float((labels == 1).mean()),
            float((validation_labels == 1).mean()),
            places=2,
        )

if __name__ == "__main__":
    unittest.main()
