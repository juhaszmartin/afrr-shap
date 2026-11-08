import unittest
from datetime import date, datetime, timezone

import polars as pl

from scripts import preprocess_all


class CurrencyConversionTests(unittest.TestCase):
    def setUp(self):
        preprocess_all._fx_rates = pl.DataFrame(
            {
                "date": [date(2025, 1, 3)],
                "currency": ["HUF"],
                "per_eur": [400.0],
            }
        )

    def tearDown(self):
        preprocess_all._fx_rates = None

    def test_converts_local_currency_using_last_published_rate(self):
        frame = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 3, 12, tzinfo=timezone.utc),
                    datetime(2025, 1, 4, 12, tzinfo=timezone.utc),
                ],
                "Long": [800.0, 400.0],
            }
        )

        result = preprocess_all.convert_to_eur(frame, "imbalance_prices", "HU")

        self.assertEqual(result["Long"].to_list(), [2.0, 1.0])

    def test_raises_when_no_rate_exists_for_non_euro_value(self):
        preprocess_all._fx_rates = pl.DataFrame(
            {"date": [date(2025, 1, 1)], "currency": ["PLN"], "per_eur": [4.0]}
        )
        frame = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 3, tzinfo=timezone.utc)],
                "Long": [800.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "without an HUF rate"):
            preprocess_all.convert_to_eur(frame, "imbalance_prices", "HU")


if __name__ == "__main__":
    unittest.main()
