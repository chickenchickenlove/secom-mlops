import unittest
from unittest.mock import patch

from scripts.monitoring.compare_candidate_with_champion_serving import (
    DECISION_SELECTION,
    DEFAULT_MAX_DECISIONS,
    parse_args,
    validate_args,
)


class CandidateServingGateContractTest(unittest.TestCase):

    def parse(self, *extra_args: str):
        argv = [
            "compare_candidate_with_champion_serving.py",
            "--cohort-start-time",
            "1",
            "--cutoff-time",
            "10",
            "--label-maturity-seconds",
            "0",
            *extra_args,
        ]
        with patch("sys.argv", argv):
            return parse_args()

    def test_gate_defaults_to_latest_one_thousand_decisions(self) -> None:
        args = self.parse()

        self.assertEqual(1000, DEFAULT_MAX_DECISIONS)
        self.assertEqual(1000, args.max_decisions)
        self.assertEqual("latest_champion_decisions", DECISION_SELECTION)

    def test_legacy_limit_alias_sets_max_decisions(self) -> None:
        args = self.parse("--limit", "700")

        self.assertEqual(700, args.max_decisions)

    def test_min_decisions_cannot_exceed_max_decisions(self) -> None:
        args = self.parse(
            "--max-decisions",
            "400",
            "--min-decisions",
            "500",
        )

        with self.assertRaisesRegex(
                ValueError,
                "min_decisions must be <= max_decisions",
        ):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
