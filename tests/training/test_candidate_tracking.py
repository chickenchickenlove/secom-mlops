import unittest

from secom_mlops.training.candidate_tracking import (
    CandidateSplitStats,
    CandidateTrackingContext,
    build_candidate_tracking_record,
)


class CandidateTrackingTest(unittest.TestCase):

    def setUp(self) -> None:
        self.context = CandidateTrackingContext(
            tracking_uri="http://mlflow:5100",
            model_name="secom-fail-detector",
            model_alias="candidate",
            model_role="candidate",
            candidate_group="group-1",
            training_job_id="job-1",
            random_state=42,
            min_label_coverage=0.95,
            validation_size=0.2,
            gate_source="serving_feature_snapshots",
        )
        self.metadata = {
            "train_source": "versioned_training_dataset",
            "training_spine": "serving_feature_snapshots",
            "training_decision_time": "snapshot_available_at",
            "snapshot_selection": "first_complete",
            "label_selection": "available_at_lte_cutoff_then_max_revision",
            "dataset_id": "training_test",
            "dataset_manifest_hash": "sha256:v1:manifest",
            "dataset_artifact_sha256": "sha256:v1:artifact",
            "dataset_mlflow_run_id": "dataset-run",
            "dataset_artifact_uri": "mlflow-artifacts:/data/dataset.parquet",
            "dataset_selection_hash": "sha256:v1:selection",
            "cohort_start_time": 0.0,
            "cohort_end_time": 100.0,
            "cutoff_time": 120.0,
            "label_maturity_seconds": 20.0,
            "simulation_run_id": "simulation-1",
            "drift_segment": "stable",
            "development_sample_limit": 1000,
            "development_sample_selection": "latest_labeled_snapshot_available_at",
            "eligible_cohort_count": 1100,
            "unlabeled_cohort_count": 50,
            "label_coverage": 1050 / 1100,
            "sample_count": 1000,
            "fail_count": 80,
            "pass_count": 920,
            "first_sample_id": "sample-1",
            "last_sample_id": "sample-1000",
            "decision_time_min": 1.0,
            "decision_time_max": 1000.0,
        }
        self.best_row = {
            "n_estimators": 300,
            "min_samples_leaf": 3,
            "threshold": 0.2,
            "accuracy": 0.9,
            "balanced_accuracy": 0.8,
            "precision_1": 0.7,
            "recall_1": 0.6,
            "f1_1": 0.65,
            "pr_auc": 0.72,
            "tn": 170,
            "fp": 10,
            "fn": 8,
            "tp": 12,
        }
        self.split_stats = CandidateSplitStats(
            training_sample_count=800,
            validation_sample_count=200,
            training_fail_count=64,
            validation_fail_count=16,
        )

    def test_builds_consistent_dataset_lineage_across_tracking_surfaces(self) -> None:
        tracking = self._build()
        version_tags = tracking.model_version_tags("training-run")

        for field in (
                "training_dataset_id",
                "training_dataset_manifest_hash",
                "training_dataset_artifact_sha256",
                "training_dataset_mlflow_run_id",
                "training_dataset_selection_hash",
        ):
            self.assertEqual(str(tracking.params[field]), tracking.run_tags[field])
            self.assertEqual(tracking.params[field], version_tags[field])

        self.assertEqual("training-run", version_tags["source_run_id"])
        self.assertEqual(
            self.metadata["dataset_selection_hash"],
            tracking.dataset_input_tags["selection_hash"],
        )

    def test_preserves_legacy_and_explicit_validation_metrics(self) -> None:
        tracking = self._build()

        self.assertEqual({
            "accuracy",
            "balanced_accuracy",
            "precision_1",
            "recall_1",
            "f1_1",
            "pr_auc",
            "tn",
            "fp",
            "fn",
            "tp",
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_precision_1",
            "validation_recall_1",
            "validation_f1_1",
            "validation_pr_auc",
            "label_coverage",
            "training_sample_count",
            "validation_sample_count",
            "final_fit_sample_count",
            "training_fail_count",
            "validation_fail_count",
            "final_fit_fail_count",
        }, set(tracking.metrics))
        self.assertEqual(0.65, tracking.metrics["f1_1"])
        self.assertEqual(0.65, tracking.metrics["validation_f1_1"])
        self.assertEqual(800.0, tracking.metrics["training_sample_count"])
        self.assertEqual(80.0, tracking.metrics["final_fit_fail_count"])

    def test_reuses_tracking_values_in_summary_and_messages(self) -> None:
        tracking = self._build()

        self.assertEqual(self.metadata, tracking.summary["metadata"])
        dry_run = tracking.dry_run_message()
        self.assertIn("dataset_id=training_test", dry_run)
        self.assertNotIn("cohort_start_time=", dry_run)
        self.assertNotIn("final_fit_sample_count=", dry_run)
        registered = tracking.registered_message(
            model_version="7",
            run_id="training-run",
        )
        self.assertIn("version=7", registered)
        self.assertIn("run_id=training-run", registered)
        self.assertIn("dataset_selection_hash=sha256:v1:selection", registered)

    def test_preserves_run_and_model_version_tag_contracts(self) -> None:
        tracking = self._build()

        self.assertEqual({
            "project",
            "stage",
            "purpose",
            "role",
            "candidate_group",
            "training_job_id",
            "train_source",
            "training_spine",
            "training_decision_time",
            "snapshot_selection",
            "label_selection",
            "gate_source",
            "training_dataset_id",
            "training_dataset_manifest_hash",
            "training_dataset_artifact_sha256",
            "training_dataset_mlflow_run_id",
            "training_dataset_selection_hash",
            "cohort_start_time",
            "cohort_end_time",
            "cutoff_time",
            "label_maturity_seconds",
            "development_sample_selection",
            "min_label_coverage",
            "offline_metric_scope",
            "final_evaluation_source",
        }, set(tracking.run_tags))
        self.assertEqual({
            "registered_model_alias",
            "source_run_id",
            "role",
            "candidate_group",
            "training_job_id",
            "train_source",
            "training_spine",
            "training_decision_time",
            "snapshot_selection",
            "label_selection",
            "gate_source",
            "training_dataset_id",
            "training_dataset_manifest_hash",
            "training_dataset_artifact_sha256",
            "training_dataset_mlflow_run_id",
            "training_dataset_selection_hash",
            "development_sample_limit",
            "development_sample_selection",
            "min_label_coverage",
            "validation_size",
            "final_fit_scope",
            "final_evaluation_source",
            "cohort_start_time",
            "cohort_end_time",
            "cutoff_time",
            "label_maturity_seconds",
        }, set(tracking.model_version_tags("training-run")))

    def _build(self):
        return build_candidate_tracking_record(
            context=self.context,
            metadata=self.metadata,
            best_row=self.best_row,
            split_stats=self.split_stats,
        )


if __name__ == "__main__":
    unittest.main()
