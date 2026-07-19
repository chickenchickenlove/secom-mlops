import unittest
from pathlib import Path

from secom_mlops.datasets.serving_gate_dataset import DECISION_SELECTION


class CandidateServingGateContractTest(unittest.TestCase):

    def test_gate_uses_persisted_dataset_instead_of_querying_predictions(self) -> None:
        source = Path(
            "scripts/monitoring/compare_candidate_with_champion_serving.py"
        ).read_text(encoding="utf-8")

        self.assertIn("load_serving_gate_dataset", source)
        self.assertIn("log_evaluation_run", source)
        self.assertIn("set_candidate_evaluation_pointer", source)
        self.assertIn("evaluation_run_id=", source)
        self.assertNotIn("FROM prediction_logs", source)
        self.assertNotIn("max_decisions", source)

    def test_gate_selector_is_first_release_decision_per_sample_snapshot(self) -> None:
        self.assertEqual(
            "first_release_decision_per_sample_snapshot",
            DECISION_SELECTION,
        )

    def test_gate_dag_materializes_before_evaluation_and_passes_only_dataset_id(self) -> None:
        source = Path(
            "airflow/dags/evaluate_candidate_serving_snapshot_gate.py"
        ).read_text(encoding="utf-8")

        self.assertIn('task_id="materialize_serving_gate_dataset"', source)
        self.assertIn('task_id="evaluate_candidate_against_champion"', source)
        self.assertIn(
            "materialize_serving_gate_dataset >> evaluate_candidate_against_champion",
            source,
        )
        self.assertIn("ti.xcom_pull(task_ids='materialize_serving_gate_dataset')", source)
        self.assertIn("--dataset-id", source)
        self.assertNotIn("--max-decisions", source)
        self.assertIn('type=["null", "string"]', source)
        self.assertIn('"min_decisions": Param(1000', source)
        self.assertIn('"min_labeled_decisions": Param(1000', source)

    def test_deployment_request_dag_requires_an_evaluation_run_id(self) -> None:
        source = Path(
            "airflow/dags/record_serving_candidate_deployment_request.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"evaluation_run_id": Param(', source)
        self.assertIn("--evaluation-run-id", source)
        self.assertIn("evaluation_run_id_required", source)

if __name__ == "__main__":
    unittest.main()
