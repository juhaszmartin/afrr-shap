import unittest
from datetime import UTC, datetime, timedelta

import polars as pl

from scripts.preprocess_all import select_feature_columns


class FeatureFilterTests(unittest.TestCase):
    def test_drops_missing_and_constant_features_per_split(self):
        split = datetime(2025, 1, 1, tzinfo=UTC)
        timestamps = [
            split - timedelta(days=2),
            split - timedelta(days=1),
            split,
            split + timedelta(days=1),
        ]
        frame = pl.DataFrame(
            {
                "timestamp": timestamps,
                "kept": [1.0, 2.0, 3.0, 4.0],
                "mostly_missing": [1.0, 2.0, None, None],
                "constant": [7.0, 7.0, 7.0, 7.0],
            }
        )

        kept = select_feature_columns(
            frame, ["kept", "mostly_missing", "constant"], split
        )

        self.assertEqual(kept, ["kept"])

    def test_drops_features_missing_in_test_but_ignores_test_values(self):
        dev = datetime(2025, 1, 1, tzinfo=UTC)
        test = datetime(2026, 1, 1, tzinfo=UTC)
        timestamps = [
            dev - timedelta(days=2),
            dev - timedelta(days=1),
            dev,
            dev + timedelta(days=1),
            test,
            test + timedelta(days=1),
        ]
        frame = pl.DataFrame(
            {
                "timestamp": timestamps,
                "kept": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                # constant in test only: test values are not inspected, so it stays
                "constant_in_test": [1.0, 2.0, 3.0, 4.0, 9.0, 9.0],
                "stops_before_test": [1.0, 2.0, 3.0, 4.0, None, None],
            }
        )

        kept = select_feature_columns(
            frame,
            ["kept", "constant_in_test", "stops_before_test"],
            dev,
            eval_end_ts=test,
        )

        self.assertEqual(kept, ["kept", "constant_in_test"])


if __name__ == "__main__":
    unittest.main()
