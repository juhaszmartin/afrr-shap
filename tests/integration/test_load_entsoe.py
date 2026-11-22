"""Integration tests for load_entsoe_data using programmatically generated fixtures.

These tests create temporary parquet files, run the full loader, and verify
the output. This tests the complete I/O + transformation pipeline.
"""

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from scripts.preprocess_all import load_entsoe_data


@pytest.fixture(scope="module")
def temp_entsoe_dir():
    """Create a temporary ENTSOE_DIR with test fixture files."""
    base = Path(tempfile.mkdtemp(prefix="entsoe_test_"))
    try:
        # Create test fixture structure
        yield _create_fixtures(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _create_fixtures(base: Path) -> Path:
    """Create test parquet files mimicking ENTSO-E structure."""
    # ----- imbalance_prices -----
    _write_parquet(
        base / "imbalance_prices" / "country=DE" / "year=2024" / "data.parquet",
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 30, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 45, tzinfo=UTC),
            ],
            "Long": [50.0, 51.0, 52.0, 53.0],
            "Short": [40.0, 41.0, 42.0, 43.0],
        },
    )
    _write_parquet(
        base / "imbalance_prices" / "country=DE" / "year=2024" / "fill.parquet",
        {
            "timestamp": [
                datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
            ],
            "Long": [54.0],
            "Short": [44.0],
        },
    )

    # ----- imbalance_volumes -----
    _write_parquet(
        base / "imbalance_volumes" / "country=DE" / "year=2024" / "data.parquet",
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ],
            "value": [100.0, 200.0],
        },
    )

    # ----- generation (with tuple columns) -----
    _write_parquet(
        base / "generation" / "country=DE" / "year=2024" / "data.parquet",
        {
            "('timestamp', '')": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ],
            "('Wind Onshore', 'Actual Aggregated')": [1000.0, 1100.0],
            "('Solar', 'Actual Aggregated')": [500.0, 550.0],
        },
    )

    # ----- generation with year transition (PL style: Biomass + Biomass - Actual Aggregated) -----
    _write_parquet(
        base / "generation" / "country=PL" / "year=2024" / "data.parquet",
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            ],
            "Biomass": [100.0],
        },
    )
    _write_parquet(
        base / "generation" / "country=PL" / "year=2025" / "data.parquet",
        {
            "timestamp": [
                datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
            ],
            "Biomass - Actual Aggregated": [200.0],
        },
    )

    # ----- activated_balancing_energy_prices (long format) -----
    _write_parquet(
        base
        / "activated_balancing_energy_prices"
        / "country=DE"
        / "year=2024"
        / "data.parquet",
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ],
            "Direction": ["Up", "Down", "Up", "Down"],
            "ReserveType": ["aFRR", "aFRR", "aFRR", "aFRR"],
            "Price": [50.0, 40.0, 55.0, 45.0],
        },
    )

    # ----- load_forecast -----
    _write_parquet(
        base / "load_forecast" / "country=DE" / "year=2024" / "data.parquet",
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            ],
            "value": [50000.0, 51000.0],
        },
    )

    return base


def _write_parquet(path: Path, data: dict):
    """Write a DataFrame to parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(data)
    df.write_parquet(path)


class TestLoadEntsoeData:
    """Integration tests for load_entsoe_data."""

    def test_loads_multiple_categories(self, temp_entsoe_dir, monkeypatch):
        """Loads and joins multiple categories correctly."""
        # Patch ENTSOE_DIR to point to our test fixtures
        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("DE", is_hourly=False)

        # Should have data from all categories
        assert df is not None
        assert df.height > 0
        assert "timestamp" in df.columns

        # Check expected columns from each category
        expected_prefixes = [
            "imbalance_price_",  # targets are renamed by TARGET_RENAME
            "imbalance_volume",
            "generation - ",
            "activated_balancing_energy_prices - ",
            "load_forecast - ",
        ]
        for prefix in expected_prefixes:
            assert any(c.startswith(prefix) for c in df.columns), (
                f"Missing {prefix} columns"
            )

    def test_fill_priority_over_data(self, temp_entsoe_dir, monkeypatch):
        """fill.parquet rows take priority over data.parquet for same timestamp."""
        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("DE", is_hourly=False)

        # fill.parquet has timestamp 2024-01-01 01:00 with Long=54.0
        # data.parquet has timestamps 00:00-00:45
        # Both should appear (different timestamps)
        long_col = "imbalance_price_long"
        assert long_col in df.columns

        # Check fill row exists
        fill_row = df.filter(
            pl.col("timestamp") == datetime(2024, 1, 1, 1, 0, tzinfo=UTC)
        )
        assert fill_row.height == 1
        assert fill_row[long_col][0] == 54.0

    def test_generation_tuple_columns_normalized(self, temp_entsoe_dir, monkeypatch):
        """Tuple-string columns are parsed and renamed."""
        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("DE", is_hourly=False)

        assert "generation - Wind Onshore - Actual Aggregated" in df.columns
        assert "generation - Solar - Actual Aggregated" in df.columns

    def test_balancing_prices_pivoted(self, temp_entsoe_dir, monkeypatch):
        """Long-format balancing prices are pivoted to wide."""
        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("DE", is_hourly=False)

        assert "activated_balancing_energy_prices - aFRR_Up" in df.columns
        assert "activated_balancing_energy_prices - aFRR_Down" in df.columns

        # Check values
        up_col = "activated_balancing_energy_prices - aFRR_Up"
        down_col = "activated_balancing_energy_prices - aFRR_Down"
        row = df.filter(pl.col("timestamp") == datetime(2024, 1, 1, 0, 0, tzinfo=UTC))
        assert row[up_col][0] == 50.0
        assert row[down_col][0] == 40.0

    def test_coalesces_duplicate_columns_across_years(
        self, temp_entsoe_dir, monkeypatch
    ):
        """Columns split across years (Biomass + Biomass - Actual Aggregated) are merged."""
        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("PL", is_hourly=False)

        # Should have merged column
        assert "generation - Biomass - Actual Aggregated" in df.columns
        # Original "Biomass" should be dropped
        assert "generation - Biomass" not in df.columns

        # Check both years' data present
        assert df.height == 2
        vals = df["generation - Biomass - Actual Aggregated"].to_list()
        assert 100.0 in vals  # 2024
        assert 200.0 in vals  # 2025

    def test_deduplicates_timestamps_within_category(
        self, temp_entsoe_dir, monkeypatch
    ):
        """Duplicate timestamps within a category are deduplicated (keep first)."""
        # Add duplicate timestamp in imbalance_prices
        dup_path = (
            temp_entsoe_dir
            / "imbalance_prices"
            / "country=DE"
            / "year=2024"
            / "data.parquet"
        )
        df_dup = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),  # duplicate
                    datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
                ],
                "Long": [50.0, 999.0, 52.0],  # second should be dropped
                "Short": [40.0, 999.0, 42.0],
            }
        )
        df_dup.write_parquet(dup_path)

        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("DE", is_hourly=False)

        row = df.filter(pl.col("timestamp") == datetime(2024, 1, 1, 0, 0, tzinfo=UTC))
        assert row["imbalance_price_long"][0] == 50.0  # first kept
        assert row["imbalance_price_short"][0] == 40.0

    def test_cutoff_year_filters_old_files(self, temp_entsoe_dir, monkeypatch):
        """Files before cutoff year are skipped for 15-min countries."""
        # ES cutoff is 2022, so 2021 file should be skipped
        _write_parquet(
            temp_entsoe_dir
            / "imbalance_prices"
            / "country=ES"
            / "year=2021"
            / "data.parquet",
            {
                "timestamp": [datetime(2021, 1, 1, 0, 0, tzinfo=UTC)],
                "Long": [100.0],
                "Short": [90.0],
            },
        )
        _write_parquet(
            temp_entsoe_dir
            / "imbalance_prices"
            / "country=ES"
            / "year=2023"
            / "data.parquet",
            {
                "timestamp": [datetime(2023, 1, 1, 0, 0, tzinfo=UTC)],
                "Long": [200.0],
                "Short": [190.0],
            },
        )

        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("ES", is_hourly=False)

        # Only 2023 data should be loaded
        assert df.height == 1
        assert df["imbalance_price_long"][0] == 200.0

    def test_no_cutoff_for_hourly_countries(self, temp_entsoe_dir, monkeypatch):
        """Hourly countries load all years regardless of cutoff."""
        # FR is hourly, no cutoff
        _write_parquet(
            temp_entsoe_dir
            / "imbalance_prices"
            / "country=FR"
            / "year=2019"
            / "data.parquet",
            {
                "timestamp": [datetime(2019, 1, 1, 0, 0, tzinfo=UTC)],
                "Long": [100.0],
                "Short": [90.0],
            },
        )

        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("FR", is_hourly=True)

        assert df.height == 1
        assert df["imbalance_price_long"][0] == 100.0

    def test_returns_none_for_missing_country(self, temp_entsoe_dir, monkeypatch):
        """Returns None if no data found for country."""
        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("XX", is_hourly=False)  # non-existent
        assert df is None

    def test_timestamp_is_utc(self, temp_entsoe_dir, monkeypatch):
        """Output timestamps are in UTC."""
        monkeypatch.setattr("scripts.preprocess_all.ENTSOE_DIR", temp_entsoe_dir)

        df = load_entsoe_data("DE", is_hourly=False)

        assert df["timestamp"].dtype == pl.Datetime(time_zone="UTC")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
