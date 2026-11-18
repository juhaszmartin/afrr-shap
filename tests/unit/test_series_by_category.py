import unittest
from datetime import UTC, datetime

from scripts.get_entsoe_info import format_timestamp, series_by_category


class SeriesByCategoryTests(unittest.TestCase):
    def test_groups_by_base_name_ignoring_lags(self):
        """Lagged versions of a series are grouped under the base name.

        The function skips lagged columns (they have LAG_SUFFIX) and only processes
        the base column (without lag suffix). So we need to include base columns.
        """
        columns = [
            "timestamp",
            "imbalance_prices - Long",  # base version (t+0)
            "imbalance_prices - Long_t+15min",
            "imbalance_prices - Long_t+30min",
            "generation - Wind Onshore - Actual Aggregated",  # base version
            "generation - Wind Onshore - Actual Aggregated_t-15min",
            "load_forecast - value",  # base version
            "load_forecast - value_t+15min",
        ]

        result = series_by_category(columns)

        self.assertIn("target history", result)
        self.assertIn("imbalance_prices - Long", result["target history"])

        self.assertIn("generation", result)
        self.assertIn("Wind Onshore - Actual Aggregated", result["generation"])

        self.assertIn("load_forecast", result)
        self.assertIn("value", result["load_forecast"])

    def test_calendar_sin_cos_paired(self):
        """Sin/cos pairs of calendar features are grouped as one series."""
        columns = [
            "timestamp",
            "qh_of_day_sin",
            "qh_of_day_cos",
            "dayofweek_sin",
            "dayofweek_cos",
            "is_weekend",
        ]

        result = series_by_category(columns)

        self.assertIn("calendar", result)
        self.assertIn("qh_of_day", result["calendar"])
        self.assertIn("dayofweek", result["calendar"])
        # is_weekend is not a sin/cos pair
        self.assertIn("is_weekend", result["calendar"])

    def test_weather_grouped_separately(self):
        """Weather columns go into 'weather' category."""
        # Weather keywords: temperature, ssrd, wind_speed
        # Need base versions (without lag suffix)
        columns = [
            "timestamp",
            "temperature_mean_celsius",
            "temperature_std_celsius",
            "ssrd_mean_j_m2",
            "10m_wind_speed_mean",
            "100m_wind_speed_std",
        ]

        result = series_by_category(columns)

        self.assertIn("weather", result)
        self.assertEqual(len(result["weather"]), 5)

    def test_excludes_timestamp(self):
        """Timestamp column is not included in any category."""
        columns = ["timestamp", "imbalance_prices - Long"]

        result = series_by_category(columns)

        for cat_series in result.values():
            self.assertNotIn("timestamp", cat_series)

    def test_unknown_category_fallback(self):
        """Columns not matching known patterns get their prefix as category."""
        columns = [
            "timestamp",
            "custom_category - some_value",  # base version without lag suffix
        ]

        result = series_by_category(columns)

        self.assertIn("custom_category", result)
        self.assertIn("some_value", result["custom_category"])


class FormatTimestampTests(unittest.TestCase):
    def test_formats_utc_datetime(self):
        """UTC datetimes are formatted without timezone."""
        ts = datetime(2025, 1, 1, 12, 30, tzinfo=UTC)
        result = format_timestamp(ts)
        self.assertEqual(result, "2025-01-01 12:30")

    def test_formats_naive_datetime(self):
        """Naive datetimes are formatted as-is."""
        ts = datetime(2025, 1, 1, 12, 30)
        result = format_timestamp(ts)
        self.assertEqual(result, "2025-01-01 12:30")

    def test_returns_dash_for_none(self):
        """None returns a dash."""
        result = format_timestamp(None)
        self.assertEqual(result, "-")


if __name__ == "__main__":
    unittest.main()
