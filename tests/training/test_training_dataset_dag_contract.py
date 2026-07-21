import unittest
from pathlib import Path


class TrainingDatasetDagContractTest(unittest.TestCase):

    def test_candidate_training_requires_only_dataset_identity_for_data_selection(self) -> None:
        source = Path(
            "airflow/dags/train_candidate_from_offline_point_in_time_features.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"dataset_id": Param', source)
        self.assertIn('--dataset-id "{{ params.dataset_id }}"', source)
        self.assertNotIn('"cohort_start_time": Param', source)
        self.assertNotIn('"cutoff_time": Param', source)
        self.assertNotIn('"label_maturity_seconds": Param', source)


if __name__ == "__main__":
    unittest.main()
