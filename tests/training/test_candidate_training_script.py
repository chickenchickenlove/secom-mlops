import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scripts.training import train_candidate_from_offline_point_in_time_features as script
from secom_mlops.training.candidate_registration import RegisteredCandidate
from secom_mlops.training.random_forest_training import RandomForestTrainingResult
from secom_mlops.training.training_data_preparation import PreparedTrainingData


class CandidateTrainingScriptTest(unittest.TestCase):

    def test_dry_run_executes_training_pipeline_without_registration(self) -> None:
        args = self._args(dry_run=True)
        loaded_dataset = object()
        training_data = self._training_data()
        training_result = self._training_result()
        tracking = MagicMock()
        tracking.dry_run_message.return_value = "candidate dry run"

        output = io.StringIO()
        with (
            patch.object(script, "resolve_tracking_uri", return_value="http://mlflow:5100"),
            patch.object(script, "load_training_dataset", return_value=loaded_dataset) as load,
            patch.object(script, "prepare_training_dataset", return_value=training_data) as prepare,
            patch.object(script, "validate_training_data") as validate,
            patch.object(script, "train_random_forest", return_value=training_result) as train,
            patch.object(
                script,
                "build_candidate_tracking_record",
                return_value=tracking,
            ) as build_tracking,
            patch.object(script, "register_candidate") as register,
            contextlib.redirect_stdout(output),
        ):
            script.train_and_register(args)

        load.assert_called_once_with(
            "training_test",
            tracking_uri="http://mlflow:5100",
        )
        prepare.assert_called_once_with(loaded_dataset)
        validate.assert_called_once_with(
            metadata=training_data.metadata,
            min_samples=script.MAX_DEVELOPMENT_SAMPLES,
            min_label_coverage=0.95,
            min_fail_samples=20,
            min_pass_samples=20,
        )
        training_config = train.call_args.args[2]
        self.assertEqual([100, 300], training_config.n_estimators)
        self.assertEqual([1, 3], training_config.min_samples_leaf)
        self.assertEqual([0.2, 0.5], training_config.thresholds)
        self.assertEqual(script.VALIDATION_SIZE, training_config.validation_size)
        self.assertEqual(42, training_config.random_state)
        build_tracking.assert_called_once()
        register.assert_not_called()
        self.assertEqual("candidate dry run\n", output.getvalue())

    def test_non_dry_run_registers_and_reports_candidate(self) -> None:
        training_data = self._training_data()
        training_result = self._training_result()
        tracking = MagicMock()
        tracking.registered_message.return_value = "candidate registered"

        output = io.StringIO()
        with (
            patch.object(script, "resolve_tracking_uri", return_value="http://mlflow:5100"),
            patch.object(script, "load_training_dataset", return_value=object()),
            patch.object(script, "prepare_training_dataset", return_value=training_data),
            patch.object(script, "validate_training_data"),
            patch.object(script, "train_random_forest", return_value=training_result),
            patch.object(
                script,
                "build_candidate_tracking_record",
                return_value=tracking,
            ),
            patch.object(
                script,
                "register_candidate",
                return_value=RegisteredCandidate(
                    run_id="training-run",
                    model_version="7",
                ),
            ) as register,
            contextlib.redirect_stdout(output),
        ):
            script.train_and_register(self._args(dry_run=False))

        registration_config = register.call_args.kwargs["config"]
        self.assertEqual("http://mlflow:5100", registration_config.tracking_uri)
        self.assertEqual("secom-fail-detection", registration_config.experiment_name)
        self.assertEqual("secom-fail-detector", registration_config.model_name)
        self.assertEqual("candidate", registration_config.model_alias)
        tracking.registered_message.assert_called_once_with(
            model_version="7",
            run_id="training-run",
        )
        self.assertEqual("candidate registered\n", output.getvalue())

    def test_candidate_role_requires_candidate_identity(self) -> None:
        args = self._args(candidate_group=None, training_job_id=None)

        with self.assertRaisesRegex(
            ValueError,
            "candidate training requires .*candidate-group.*training-job-id",
        ):
            script.validate_args(args)

    @staticmethod
    def _args(**overrides) -> argparse.Namespace:
        values = {
            "tracking_uri": None,
            "model_name": "secom-fail-detector",
            "model_alias": "candidate",
            "model_role": "candidate",
            "candidate_group": "group-1",
            "training_job_id": "job-1",
            "dataset_id": "training_test",
            "min_label_coverage": 0.95,
            "min_fail_samples": 20,
            "min_pass_samples": 20,
            "random_state": 42,
            "n_estimators": "100,300",
            "min_samples_leaf": "1,3",
            "thresholds": "0.2,0.5",
            "dry_run": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def _training_data() -> PreparedTrainingData:
        return PreparedTrainingData(
            features=pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}),
            targets=pd.Series([-1, 1, -1, 1], dtype="int64"),
            sample_ids=["sample-0", "sample-1", "sample-2", "sample-3"],
            metadata={"dataset_id": "training_test"},
            selected_rows=pd.DataFrame({"sample_id": ["sample-0", "sample-1"]}),
        )

    @staticmethod
    def _training_result() -> RandomForestTrainingResult:
        return RandomForestTrainingResult(
            model=SimpleNamespace(),
            best_row={
                "n_estimators": 100,
                "min_samples_leaf": 1,
                "threshold": 0.5,
            },
            search_results=pd.DataFrame(),
            report={},
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            train_indices=np.array([0, 1]),
            validation_indices=np.array([2, 3]),
        )


if __name__ == "__main__":
    unittest.main()
