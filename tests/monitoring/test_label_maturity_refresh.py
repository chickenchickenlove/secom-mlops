import argparse
import unittest
from unittest.mock import MagicMock, call, patch

from scripts.monitoring.refresh_label_maturity_metrics import (
    MATERIALIZED_VIEW,
    REFRESH_SQL,
    SUMMARY_SQL,
    positive_int,
    refresh_once,
)


class LabelMaturityRefreshTest(unittest.TestCase):

    def test_refreshes_fixed_materialized_view_and_returns_summary(self) -> None:
        cursor = MagicMock()
        cursor.fetchone.return_value = (44, 123.5)

        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        connection_context = MagicMock()
        connection_context.__enter__.return_value = connection

        with patch(
                "scripts.monitoring.refresh_label_maturity_metrics.connect",
                return_value=connection_context,
        ):
            metric_rows, computed_at = refresh_once()

        self.assertEqual("label_maturity_cohort_age_metrics", MATERIALIZED_VIEW)
        self.assertEqual((44, 123.5), (metric_rows, computed_at))
        self.assertEqual(
            [
                call(REFRESH_SQL),
                call(SUMMARY_SQL),
            ],
            cursor.execute.call_args_list,
        )

    def test_positive_interval_rejects_zero(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")


if __name__ == "__main__":
    unittest.main()
