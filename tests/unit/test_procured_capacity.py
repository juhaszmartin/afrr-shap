import unittest
from datetime import UTC, datetime

import polars as pl

from scripts.preprocess_all import procured_capacity


def _make_col(*parts):
    """Create a column name as a stringified tuple, matching ENTSO-E format.
    All columns must be 3-element tuples for the procured_capacity function."""
    return str(tuple(parts))


def _make_ts():
    """Timestamp column as 3-element tuple."""
    return str(("timestamp", "", ""))


class ProcuredCapacityTests(unittest.TestCase):
    def test_aggregates_multiple_tenders_per_direction(self):
        """Volume-weighted price across tenders for each direction."""
        df = pl.DataFrame(
            {
                _make_ts(): [
                    datetime(2025, 1, 1, 0, tzinfo=UTC),
                    datetime(2025, 1, 1, 0, 15, tzinfo=UTC),
                ],
                _make_col("Up", "Tender1", "Price"): [50.0, 60.0],
                _make_col("Up", "Tender1", "Volume"): [100.0, 200.0],
                _make_col("Up", "Tender2", "Price"): [55.0, 65.0],
                _make_col("Up", "Tender2", "Volume"): [50.0, 100.0],
                _make_col("Down", "Tender1", "Price"): [40.0, 45.0],
                _make_col("Down", "Tender1", "Volume"): [80.0, 160.0],
            }
        )

        result = procured_capacity(df, "procured_balancing_capacity").sort("timestamp")

        # Directions are sorted alphabetically: Down, Up
        self.assertEqual(
            result.columns,
            [
                "timestamp",
                "procured_balancing_capacity - Down - Volume",
                "procured_balancing_capacity - Down - Price",
                "procured_balancing_capacity - Up - Volume",
                "procured_balancing_capacity - Up - Price",
            ],
        )
        up_vol = result["procured_balancing_capacity - Up - Volume"].to_list()
        up_price = result["procured_balancing_capacity - Up - Price"].to_list()
        down_vol = result["procured_balancing_capacity - Down - Volume"].to_list()
        down_price = result["procured_balancing_capacity - Down - Price"].to_list()

        self.assertEqual(up_vol, [150.0, 300.0])
        self.assertAlmostEqual(up_price[0], 51.666666666666664)
        self.assertAlmostEqual(up_price[1], 61.666666666666664)
        self.assertEqual(down_vol, [80.0, 160.0])
        self.assertEqual(down_price, [40.0, 45.0])

    def test_handles_missing_volumes(self):
        """Only reported tenders (non-null volume) contribute to the weighted price."""
        df = pl.DataFrame(
            {
                _make_ts(): [
                    datetime(2025, 1, 1, 0, tzinfo=UTC),
                ],
                _make_col("Up", "Tender1", "Price"): [50.0],
                _make_col("Up", "Tender1", "Volume"): [100.0],
                _make_col("Up", "Tender2", "Price"): [55.0],
                _make_col("Up", "Tender2", "Volume"): [None],  # Not reported
            }
        )

        result = procured_capacity(df, "procured_balancing_capacity")

        up_vol = result["procured_balancing_capacity - Up - Volume"].to_list()
        up_price = result["procured_balancing_capacity - Up - Price"].to_list()

        self.assertEqual(up_vol, [100.0])
        self.assertEqual(up_price, [50.0])

    def test_zero_volume_returns_null_price(self):
        """When total volume is zero, price should be null."""
        df = pl.DataFrame(
            {
                _make_ts(): [
                    datetime(2025, 1, 1, 0, tzinfo=UTC),
                ],
                _make_col("Up", "Tender1", "Price"): [50.0],
                _make_col("Up", "Tender1", "Volume"): [0.0],
                _make_col("Up", "Tender2", "Price"): [55.0],
                _make_col("Up", "Tender2", "Volume"): [0.0],
            }
        )

        result = procured_capacity(df, "procured_balancing_capacity")

        up_vol = result["procured_balancing_capacity - Up - Volume"].to_list()
        up_price = result["procured_balancing_capacity - Up - Price"].to_list()

        self.assertEqual(up_vol, [0.0])
        self.assertIsNone(up_price[0])

    def test_preserves_timestamp_timezone(self):
        """Timestamps should remain in UTC."""
        df = pl.DataFrame(
            {
                _make_ts(): [datetime(2025, 1, 1, 0, tzinfo=UTC)],
                _make_col("Up", "Tender1", "Price"): [50.0],
                _make_col("Up", "Tender1", "Volume"): [100.0],
            }
        )

        result = procured_capacity(df, "procured_balancing_capacity")

        ts = result["timestamp"][0]
        # zoneinfo.ZoneInfo('UTC') and datetime.UTC are equivalent
        self.assertEqual(str(ts.tzinfo), "UTC")


if __name__ == "__main__":
    unittest.main()
