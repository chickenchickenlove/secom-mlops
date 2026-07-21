import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from secom_mlops.training.random_forest_training import (
    RandomForestTrainingConfig,
    evaluate_predictions,
    select_model,
    train_random_forest,
)


class RandomForestTrainingTest(unittest.TestCase):

    def test_rejects_different_feature_and_target_lengths(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "features and targets must have the same number of rows",
        ):
            train_random_forest(
                features=pd.DataFrame({"feature": [0.0, 1.0]}),
                targets=pd.Series([1], dtype="int64"),
                config=self._config(),
            )

    def test_searches_every_hyperparameter_and_threshold_combination(self) -> None:
        x_train = pd.DataFrame({
            "feature": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0] * 4,
        })
        y_train = pd.Series([-1, -1, -1, 1, 1, 1] * 4, dtype="int64")
        x_validation = pd.DataFrame({
            "feature": [-2.5, -1.5, 1.5, 2.5] * 2,
        })
        y_validation = pd.Series([-1, -1, 1, 1] * 2, dtype="int64")

        results, best_row, report, matrix = select_model(
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            n_estimators_values=[5, 10],
            min_samples_leaf_values=[1, 2],
            threshold_values=[0.3, 0.5],
            random_state=42,
        )

        self.assertEqual(8, len(results))
        self.assertEqual({5, 10}, set(results["n_estimators"]))
        self.assertEqual({1, 2}, set(results["min_samples_leaf"]))
        self.assertEqual({0.3, 0.5}, set(results["threshold"]))
        self.assertIn(int(best_row["n_estimators"]), {5, 10})
        self.assertIn(int(best_row["min_samples_leaf"]), {1, 2})
        self.assertIn(float(best_row["threshold"]), {0.3, 0.5})
        self.assertIn("1", report)
        self.assertEqual((2, 2), matrix.shape)

    def test_refits_selected_configuration_on_complete_development_data(self) -> None:
        features = pd.DataFrame({"feature": range(10)}, dtype="float64")
        targets = pd.Series([-1] * 5 + [1] * 5, dtype="int64")
        train_indices = np.array([0, 1, 2, 5, 6, 7, 8, 9])
        validation_indices = np.array([3, 4])
        search_results = pd.DataFrame({"f1_1": [0.8]})
        best_row = {
            "n_estimators": 300,
            "min_samples_leaf": 3,
            "threshold": 0.2,
        }
        report = {"1": {"f1-score": 0.8}}
        matrix = np.array([[1, 0], [0, 1]])
        final_model = MagicMock()

        with (
            patch(
                "secom_mlops.training.random_forest_training.split_indices",
                return_value=(train_indices, validation_indices),
            ),
            patch(
                "secom_mlops.training.random_forest_training.select_model",
                return_value=(search_results, best_row, report, matrix),
            ) as select,
            patch(
                "secom_mlops.training.random_forest_training.build_model",
                return_value=final_model,
            ) as build,
        ):
            result = train_random_forest(
                features=features,
                targets=targets,
                config=self._config(),
            )

        pd.testing.assert_frame_equal(select.call_args.kwargs["x_train"], features.iloc[train_indices])
        pd.testing.assert_series_equal(select.call_args.kwargs["y_train"], targets.iloc[train_indices])
        pd.testing.assert_frame_equal(
            select.call_args.kwargs["x_validation"],
            features.iloc[validation_indices],
        )
        pd.testing.assert_series_equal(
            select.call_args.kwargs["y_validation"],
            targets.iloc[validation_indices],
        )
        build.assert_called_once_with(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
        )
        final_model.fit.assert_called_once()
        pd.testing.assert_frame_equal(final_model.fit.call_args.args[0], features)
        pd.testing.assert_series_equal(final_model.fit.call_args.args[1], targets)
        self.assertIs(final_model, result.model)
        self.assertIs(best_row, result.best_row)

    def test_evaluates_predictions_with_selected_fail_threshold(self) -> None:
        metrics, report, matrix = evaluate_predictions(
            y_true=pd.Series([-1, -1, 1, 1], dtype="int64"),
            fail_probability=np.array([0.1, 0.8, 0.9, 0.2]),
            threshold=0.5,
        )

        np.testing.assert_array_equal(matrix, np.array([[1, 1], [1, 1]]))
        for name in (
            "accuracy",
            "balanced_accuracy",
            "precision_1",
            "recall_1",
            "f1_1",
        ):
            self.assertEqual(0.5, metrics[name])
        self.assertEqual(
            {"tn": 1.0, "fp": 1.0, "fn": 1.0, "tp": 1.0},
            {name: metrics[name] for name in ("tn", "fp", "fn", "tp")},
        )
        self.assertEqual(0.5, report["1"]["f1-score"])

    @staticmethod
    def _config() -> RandomForestTrainingConfig:
        return RandomForestTrainingConfig(
            n_estimators=[100, 300],
            min_samples_leaf=[1, 3],
            thresholds=[0.2, 0.5],
            validation_size=0.2,
            random_state=42,
        )


if __name__ == "__main__":
    unittest.main()
