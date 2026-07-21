import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from secom_mlops.training.candidate_registration import (
    CandidateRegistrationConfig,
    _json_safe,
    _log_and_register_model,
    _log_artifacts,
    _log_tracking_record,
    _log_training_dataset,
    register_candidate,
)
from secom_mlops.training.random_forest_training import RandomForestTrainingResult
from secom_mlops.training.training_data_preparation import PreparedTrainingData


class CandidateRegistrationTest(unittest.TestCase):

    def test_logs_tracking_record_to_active_run(self) -> None:
        tracking = SimpleNamespace(
            run_tags={"stage": "candidate"},
            params={"n_estimators": 300},
            metrics={"validation_f1_1": 0.8},
        )

        with (
            patch("secom_mlops.training.candidate_registration.mlflow.set_tags") as set_tags,
            patch("secom_mlops.training.candidate_registration.mlflow.log_params") as log_params,
            patch("secom_mlops.training.candidate_registration.mlflow.log_metrics") as log_metrics,
        ):
            _log_tracking_record(tracking)

        set_tags.assert_called_once_with(tracking.run_tags)
        log_params.assert_called_once_with(tracking.params)
        log_metrics.assert_called_once_with(tracking.metrics)

    def test_logs_selected_dataset_with_digest_and_lineage(self) -> None:
        training_data = self._training_data()
        tracking = SimpleNamespace(dataset_input_tags={"selection": "latest"})
        tracked_dataset = object()

        with (
            patch(
                "secom_mlops.training.candidate_registration.mlflow.data.from_pandas",
                return_value=tracked_dataset,
            ) as from_pandas,
            patch(
                "secom_mlops.training.candidate_registration.mlflow.log_input"
            ) as log_input,
        ):
            _log_training_dataset(training_data, tracking)

        from_pandas.assert_called_once_with(
            training_data.selected_rows,
            name="training_test",
            targets="actual_value",
            digest="a" * 36,
        )
        log_input.assert_called_once_with(
            tracked_dataset,
            context="training",
            tags=tracking.dataset_input_tags,
        )

    def test_logs_reports_results_and_sample_ids_as_artifacts(self) -> None:
        logged: dict[str, str] = {}

        def capture_artifact(path: str, artifact_path: str) -> None:
            source = Path(path)
            logged[f"{artifact_path}/{source.name}"] = source.read_text(
                encoding="utf-8"
            )

        with patch(
            "secom_mlops.training.candidate_registration.mlflow.log_artifact",
            side_effect=capture_artifact,
        ) as log_artifact:
            _log_artifacts(
                training_data=self._training_data(),
                training_result=self._training_result(),
                tracking=SimpleNamespace(summary={"score": np.float64(0.8)}),
            )

        self.assertEqual(7, log_artifact.call_count)
        self.assertEqual(
            {
                "results/offline_feature_training_threshold_results.csv",
                "reports/validation_classification_report.json",
                "reports/validation_confusion_matrix.json",
                "reports/offline_feature_training_summary.json",
                "data/development_sample_ids.txt",
                "data/train_sample_ids.txt",
                "data/validation_sample_ids.txt",
            },
            set(logged),
        )
        self.assertEqual(
            "sample-0\nsample-1\nsample-2\nsample-3",
            logged["data/development_sample_ids.txt"],
        )
        self.assertEqual(
            "sample-0\nsample-2",
            logged["data/train_sample_ids.txt"],
        )
        self.assertEqual(
            "sample-1\nsample-3",
            logged["data/validation_sample_ids.txt"],
        )
        self.assertEqual(
            {"labels": [-1, 1], "matrix": [[1, 0], [1, 2]]},
            json.loads(logged["reports/validation_confusion_matrix.json"]),
        )
        self.assertEqual(
            {"score": 0.8},
            json.loads(logged["reports/offline_feature_training_summary.json"]),
        )

    def test_logs_model_then_registers_model_version_with_tags(self) -> None:
        training_data = self._training_data()
        training_result = self._training_result()
        tracking = MagicMock()
        tracking.model_version_tags.return_value = {
            "source_run_id": "training-run",
            "candidate_group": "group-1",
            "optional": None,
        }
        pyfunc_model = MagicMock()
        output_example = pd.DataFrame({"prediction": [-1, 1, -1, 1]})
        pyfunc_model.predict.return_value = output_example
        signature = object()

        with (
            patch(
                "secom_mlops.training.candidate_registration.SECOMFailDetectionPyfunc",
                return_value=pyfunc_model,
            ) as pyfunc_class,
            patch(
                "secom_mlops.training.candidate_registration.infer_signature",
                return_value=signature,
            ) as infer_signature,
            patch(
                "secom_mlops.training.candidate_registration.mlflow.pyfunc.log_model",
                return_value=SimpleNamespace(model_uri="runs:/training-run/model"),
            ) as log_model,
            patch(
                "secom_mlops.training.candidate_registration.mlflow.register_model",
                return_value=SimpleNamespace(version=7),
            ) as register_model,
        ):
            version = _log_and_register_model(
                run_id="training-run",
                training_data=training_data,
                training_result=training_result,
                tracking=tracking,
                config=self._config(),
            )

        self.assertEqual("7", version)
        pyfunc_class.assert_called_once_with(
            model=training_result.model,
            threshold=0.5,
            model_name="secom-fail-detector",
            model_run_id="training-run",
            positive_class=1,
        )
        pd.testing.assert_frame_equal(
            infer_signature.call_args.args[0],
            training_data.features.head(5),
        )
        pd.testing.assert_frame_equal(
            infer_signature.call_args.args[1],
            output_example,
        )
        self.assertEqual("model", log_model.call_args.kwargs["name"])
        self.assertIs(pyfunc_model, log_model.call_args.kwargs["python_model"])
        self.assertIs(signature, log_model.call_args.kwargs["signature"])
        register_model.assert_called_once_with(
            model_uri="runs:/training-run/model",
            name="secom-fail-detector",
            await_registration_for=300,
            tags={
                "source_run_id": "training-run",
                "candidate_group": "group-1",
            },
        )

    def test_register_candidate_owns_run_and_candidate_alias_wiring(self) -> None:
        active_run = SimpleNamespace(info=SimpleNamespace(run_id="training-run"))
        run_context = MagicMock()
        run_context.__enter__.return_value = active_run
        client = MagicMock()
        tracking = SimpleNamespace(
            context=SimpleNamespace(training_job_id="job-1"),
        )

        with (
            patch("secom_mlops.training.candidate_registration.mlflow.set_tracking_uri") as set_uri,
            patch("secom_mlops.training.candidate_registration.mlflow.set_experiment") as set_experiment,
            patch(
                "secom_mlops.training.candidate_registration.mlflow.start_run",
                return_value=run_context,
            ) as start_run,
            patch("secom_mlops.training.candidate_registration._log_tracking_record"),
            patch("secom_mlops.training.candidate_registration._log_training_dataset"),
            patch("secom_mlops.training.candidate_registration._log_artifacts"),
            patch(
                "secom_mlops.training.candidate_registration._log_and_register_model",
                return_value="7",
            ),
            patch(
                "secom_mlops.training.candidate_registration.MlflowClient",
                return_value=client,
            ),
        ):
            registered = register_candidate(
                training_data=self._training_data(),
                training_result=self._training_result(),
                tracking=tracking,
                config=self._config(),
            )

        set_uri.assert_called_once_with("http://mlflow:5100")
        set_experiment.assert_called_once_with("secom-fail-detection")
        start_run.assert_called_once_with(
            run_name="training_dataset_candidate_training_test_job-1"
        )
        client.set_registered_model_alias.assert_called_once_with(
            name="secom-fail-detector",
            alias="candidate",
            version="7",
        )
        self.assertEqual("training-run", registered.run_id)
        self.assertEqual("7", registered.model_version)

    def test_json_safe_converts_nested_numpy_values(self) -> None:
        self.assertEqual(
            {
                "integer": 1,
                "floating": 0.5,
                "boolean": True,
                "array": [1, 2],
                "tuple": [3],
            },
            _json_safe({
                "integer": np.int64(1),
                "floating": np.float64(0.5),
                "boolean": np.bool_(True),
                "array": np.array([1, 2]),
                "tuple": (np.int64(3),),
            }),
        )

    @staticmethod
    def _training_data() -> PreparedTrainingData:
        return PreparedTrainingData(
            features=pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]}),
            targets=pd.Series([-1, 1, -1, 1], dtype="int64"),
            sample_ids=["sample-0", "sample-1", "sample-2", "sample-3"],
            metadata={
                "dataset_id": "training_test",
                "dataset_selection_hash": "sha256:v1:" + "a" * 64,
            },
            selected_rows=pd.DataFrame({
                "sample_id": ["sample-0", "sample-1", "sample-2", "sample-3"],
                "actual_value": [-1, 1, -1, 1],
            }),
        )

    @staticmethod
    def _training_result() -> RandomForestTrainingResult:
        return RandomForestTrainingResult(
            model=object(),
            best_row={
                "n_estimators": 5,
                "min_samples_leaf": 1,
                "threshold": 0.5,
            },
            search_results=pd.DataFrame({"threshold": [0.5], "f1_1": [0.8]}),
            report={"1": {"f1-score": np.float64(0.8)}},
            confusion_matrix=np.array([[1, 0], [1, 2]]),
            train_indices=np.array([0, 2]),
            validation_indices=np.array([1, 3]),
        )

    @staticmethod
    def _config() -> CandidateRegistrationConfig:
        return CandidateRegistrationConfig(
            tracking_uri="http://mlflow:5100",
            experiment_name="secom-fail-detection",
            model_name="secom-fail-detector",
            model_alias="candidate",
        )


if __name__ == "__main__":
    unittest.main()
