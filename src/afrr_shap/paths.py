"""Canonical paths for source data and generated data products."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

ENTSOE_DIR = RAW_DIR / "entsoe"
WEATHER_DIR = RAW_DIR / "weather"
ERA5_RAW_DIR = INTERIM_DIR / "era5_raw"
