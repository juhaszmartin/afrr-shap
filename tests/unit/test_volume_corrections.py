import unittest
from datetime import UTC, datetime

import polars as pl

from scripts.preprocess_all import (
    BALTIC_HOURLY_UNTIL,
    fix_rs_volume,
    fix_volume_sign,
    spread_repeated_quarters,
)


def frame(timestamps, values, tz="UTC"):
    return pl.DataFrame(
        {"timestamp": timestamps, "value": values},
        schema={"timestamp": pl.Datetime("us", "UTC"), "value": pl.Float64},
    ).with_columns(pl.col("timestamp").dt.convert_time_zone(tz))


class VolumeSignTests(unittest.TestCase):
    def test_at_flips_only_old_convention_months(self):
        ts = [
            datetime(2020, 4, 30, 21, 45, tzinfo=UTC),  # 30 Apr 23:45 Vienna: flipped
            datetime(2020, 4, 30, 22, 0, tzinfo=UTC),  # 1 May 00:00 Vienna: already surplus-positive
            datetime(2024, 7, 15, tzinfo=UTC),  # July 2024: already surplus-positive
            datetime(2024, 11, 30, 22, 45, tzinfo=UTC),  # 30 Nov 23:45 Vienna: flipped
            datetime(2024, 11, 30, 23, 0, tzinfo=UTC),  # 1 Dec 00:00 Vienna: new convention
        ]
        out = fix_volume_sign(frame(ts, [1.0] * 5, "Europe/Vienna"), "AT")
        self.assertEqual(out["value"].to_list(), [-1.0, 1.0, 1.0, -1.0, 1.0])

    def test_hu_flips_everything_and_others_are_untouched(self):
        ts = [datetime(2019, 6, 1, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC)]
        self.assertEqual(fix_volume_sign(frame(ts, [2.0, -3.0]), "HU")["value"].to_list(), [-2.0, 3.0])
        self.assertEqual(fix_volume_sign(frame(ts, [2.0, -3.0]), "DE")["value"].to_list(), [2.0, -3.0])

    def test_ee_is_not_flipped(self):
        # EE volumes come from the Baltic dashboard, whose sign is surplus-positive throughout
        ts = [datetime(2025, 6, 1, tzinfo=UTC)]
        self.assertEqual(fix_volume_sign(frame(ts, [1.0], "Europe/Tallinn"), "EE")["value"].to_list(), [1.0])


class BalticTests(unittest.TestCase):
    def test_repeated_quarters_are_spread_until_the_switch(self):
        until = BALTIC_HOURLY_UNTIL["EE"]  # 2025-03-31 21:00 UTC, local midnight 1 April
        ts = [
            datetime(2024, 12, 31, 23, 0, tzinfo=UTC),  # hourly row: unchanged
            datetime(2025, 3, 31, 20, 0, tzinfo=UTC),  # hourly value repeated per quarter: / 4
            datetime(2025, 3, 31, 20, 15, tzinfo=UTC),
            datetime(2025, 3, 31, 20, 30, tzinfo=UTC),
            datetime(2025, 3, 31, 20, 45, tzinfo=UTC),
            datetime(2025, 3, 31, 21, 0, tzinfo=UTC),  # real quarter-hours from here: unchanged
            datetime(2025, 3, 31, 21, 15, tzinfo=UTC),
        ]
        out = spread_repeated_quarters(frame(ts, [40.0, 44.0, 44.0, 44.0, 44.0, 10.0, 12.0]), until)
        self.assertEqual(out["value"].to_list(), [40.0, 11.0, 11.0, 11.0, 11.0, 10.0, 12.0])


class RsVolumeTests(unittest.TestCase):
    def test_hourly_era_is_unchanged(self):
        out = fix_rs_volume(frame([datetime(2024, 6, 1, 12, tzinfo=UTC)], [20.0]))
        self.assertEqual(out["timestamp"][0], datetime(2024, 6, 1, 12, tzinfo=UTC))
        self.assertEqual(out["value"][0], 20.0)

    def test_2025_shift_follows_belgrade_offset_and_mw_become_mwh(self):
        ts = [datetime(2025, 1, 15, 10, tzinfo=UTC), datetime(2025, 7, 15, 10, tzinfo=UTC)]
        out = fix_rs_volume(frame(ts, [40.0, 40.0]))
        self.assertEqual(
            out["timestamp"].to_list(),
            [datetime(2025, 1, 15, 11, tzinfo=UTC), datetime(2025, 7, 15, 12, tzinfo=UTC)],
        )
        self.assertEqual(out["value"].to_list(), [10.0, 10.0])

    def test_2026_shift_is_one_hour_all_year(self):
        out = fix_rs_volume(frame([datetime(2026, 7, 15, 10, tzinfo=UTC)], [8.0]))
        self.assertEqual(out["timestamp"][0], datetime(2026, 7, 15, 11, tzinfo=UTC))
        self.assertEqual(out["value"][0], 2.0)


if __name__ == "__main__":
    unittest.main()
