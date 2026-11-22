import unittest
from datetime import UTC, datetime

import polars as pl

from scripts.preprocess_all import add_time_and_holiday_features, known_in_advance


class TimeFeaturesTests(unittest.TestCase):
    def test_adds_cyclic_time_features_15min(self):
        """15-minute grid gets all time features including quarter-hour-of-hour."""
        # Use a timezone where UTC = local (e.g., UTC itself or a country without DST in Jan)
        # But the function always converts to LOCAL_TZ[country_code], so for DE it's CET (UTC+1 in winter)
        # 2025-01-01 00:00 UTC = 2025-01-01 01:00 CET -> qh_of_day = 4
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2025, 1, 1, 0, 15, tzinfo=UTC),
                    datetime(2025, 1, 1, 0, 30, tzinfo=UTC),
                    datetime(2025, 1, 1, 0, 45, tzinfo=UTC),
                ]
            }
        )

        result = add_time_and_holiday_features(df, "DE", is_hourly=False)

        # Check all 11 time features exist
        expected = [
            "qh_of_day_sin",
            "qh_of_day_cos",
            "qh_of_hour_sin",
            "qh_of_hour_cos",
            "dayofweek_sin",
            "dayofweek_cos",
            "is_weekend",
            "day_of_year_sin",
            "day_of_year_cos",
            "is_holiday",
        ]
        for col in expected:
            self.assertIn(col, result.columns)

        # In CET (UTC+1), 00:00 UTC = 01:00 local -> qh_of_day = 4
        qh_sin = result["qh_of_day_sin"].to_list()
        qh_cos = result["qh_of_day_cos"].to_list()
        # sin(2*pi*4/96) = sin(pi/12) ≈ 0.2588
        # cos(2*pi*4/96) = cos(pi/12) ≈ 0.9659
        self.assertAlmostEqual(qh_sin[0], 0.258819, places=4)
        self.assertAlmostEqual(qh_cos[0], 0.965925, places=4)

    def test_adds_cyclic_time_features_hourly(self):
        """Hourly grid drops quarter-hour-of-hour features."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2025, 1, 1, 1, 0, tzinfo=UTC),
                ]
            }
        )

        result = add_time_and_holiday_features(df, "DE", is_hourly=True)

        self.assertNotIn("qh_of_hour_sin", result.columns)
        self.assertNotIn("qh_of_hour_cos", result.columns)
        self.assertIn("qh_of_day_sin", result.columns)

    def test_weekend_flag(self):
        """is_weekend is 1 for Sat/Sun, 0 otherwise."""
        # 2025-01-01 is Wednesday
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1, 0, 0, tzinfo=UTC),  # Wed
                    datetime(2025, 1, 4, 0, 0, tzinfo=UTC),  # Sat
                    datetime(2025, 1, 5, 0, 0, tzinfo=UTC),  # Sun
                ]
            }
        )

        result = add_time_and_holiday_features(df, "DE", is_hourly=True)

        self.assertEqual(result["is_weekend"].to_list(), [0, 1, 1])

    def test_holiday_flag(self):
        """is_holiday is 1 on German holidays."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1, 0, 0, tzinfo=UTC),  # New Year
                    datetime(2025, 1, 2, 0, 0, tzinfo=UTC),  # Regular day
                ]
            }
        )

        result = add_time_and_holiday_features(df, "DE", is_hourly=True)

        self.assertEqual(result["is_holiday"].to_list(), [1, 0])

    def test_local_timezone_conversion(self):
        """Features are computed in local time, not UTC."""
        # 2025-01-01 00:00 UTC = 2025-01-01 01:00 CET (Europe/Berlin)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
                ]
            }
        )

        result = add_time_and_holiday_features(df, "DE", is_hourly=False)

        # In CET, this is 01:00 -> qh_of_day = 4 (1 * 4 + 0)
        qh_sin = result["qh_of_day_sin"][0]
        qh_cos = result["qh_of_day_cos"][0]
        # sin(2*pi*4/96) = sin(pi/12) ≈ 0.2588
        # cos(2*pi*4/96) = cos(pi/12) ≈ 0.9659
        self.assertAlmostEqual(qh_sin, 0.258819, places=4)
        self.assertAlmostEqual(qh_cos, 0.965925, places=4)


class KnownInAdvanceTests(unittest.TestCase):
    def test_identifies_forecast_columns(self):
        """Columns with 'forecast' in name are known in advance."""
        self.assertTrue(
            known_in_advance("generation_forecast - Wind Onshore - Actual Aggregated")
        )
        self.assertTrue(known_in_advance("load_forecast - value"))
        self.assertTrue(known_in_advance("wind_and_solar_forecast - Solar - value"))

    def test_identifies_scheduled_columns(self):
        """Columns with 'scheduled' in name are known in advance."""
        self.assertTrue(known_in_advance("generation_forecast - Scheduled Aggregated"))
        self.assertTrue(known_in_advance("net_position - Scheduled"))

    def test_identifies_day_ahead_prices(self):
        """Day-ahead prices are known in advance."""
        self.assertTrue(known_in_advance("prices - value"))
        self.assertTrue(known_in_advance("prices - Long"))
        self.assertTrue(known_in_advance("net_position - value"))

    def test_rejects_actual_columns(self):
        """Actual generation/load/balancing are not known in advance."""
        self.assertFalse(
            known_in_advance("generation - Wind Onshore - Actual Aggregated")
        )
        self.assertFalse(known_in_advance("load - value"))
        self.assertFalse(
            known_in_advance("activated_balancing_energy_prices - aFRR_Up")
        )
        self.assertFalse(known_in_advance("imbalance_volumes - value"))

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        self.assertTrue(known_in_advance("GENERATION_FORECAST - Wind"))
        self.assertTrue(known_in_advance("Load_Forecast"))
        self.assertTrue(known_in_advance("SCHEDULED"))


if __name__ == "__main__":
    unittest.main()
