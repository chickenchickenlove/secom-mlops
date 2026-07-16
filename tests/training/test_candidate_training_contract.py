import unittest

import numpy as np
import pandas as pd

from scripts.training.train_candidate_from_offline_point_in_time_features import (
    DEVELOPMENT_SAMPLE_SELECTION,
    MAX_DEVELOPMENT_SAMPLES,
    VALIDATION_SIZE,
    split_indices,
    validate_training_data,
)


class CandidateTrainingContractTest(unittest.TestCase):

    def test_development_cohort_is_capped_at_one_thousand(self) -> None:
        self.assertEqual(1000, MAX_DEVELOPMENT_SAMPLES)
        self.assertEqual("latest_eligible_available_at", DEVELOPMENT_SAMPLE_SELECTION)

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
