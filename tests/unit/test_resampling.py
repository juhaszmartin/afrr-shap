import unittest
from datetime import datetime, timezone

import polars as pl

from scripts.preprocess_all import delay_hourly_actuals, hourly_energy, timelag_expressions


class ResamplingTests(unittest.TestCase):
    def test_hourly_energy_sums_quarter_hour_mwh(self):
        frame = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1, 0, minute, tzinfo=timezone.utc)
                    for minute in (0, 15, 30, 45)
                ],
                "volume": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = hourly_energy(frame, ["volume"])

        self.assertEqual(result["volume"].to_list(), [10.0])

    def test_lag_expressions_name_and_shift_values(self):
        frame = pl.DataFrame({"value": [10, 20, 30]})

        result = frame.select(
            [pl.col("value")] + timelag_expressions([1, -1], ["value"])
        )

        self.assertEqual(result.columns, ["value", "value_t+15min", "value_t-15min"])
        self.assertEqual(result["value_t+15min"].to_list(), [20, 30, None])
        self.assertEqual(result["value_t-15min"].to_list(), [None, 10, 20])

    def test_hourly_actuals_are_available_from_last_quarter(self):
        timestamps = [
            datetime(2025, 1, 1, 0, minute, tzinfo=timezone.utc)
            for minute in (0, 15, 30, 45)
        ] + [
            datetime(2025, 1, 1, 1, minute, tzinfo=timezone.utc)
            for minute in (0, 15, 30, 45)
        ]
        frame = pl.DataFrame(
            {"timestamp": timestamps, "actual": [10.0, None, None, None, 20.0, None, None, None]}
        )

        result = delay_hourly_actuals(frame, ["actual"])

        self.assertEqual(
            result["actual"].to_list(),
            [None, None, None, 10.0, 10.0, 10.0, 10.0, 20.0],
        )


if __name__ == "__main__":
    unittest.main()
