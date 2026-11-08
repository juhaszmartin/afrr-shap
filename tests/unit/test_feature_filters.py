import unittest
from datetime import datetime, timedelta, timezone

import polars as pl

from scripts.preprocess_all import select_feature_columns


class FeatureFilterTests(unittest.TestCase):
    def test_drops_missing_and_constant_features_per_split(self):
        split = datetime(2025, 1, 1, tzinfo=timezone.utc)
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


if __name__ == "__main__":
    unittest.main()
