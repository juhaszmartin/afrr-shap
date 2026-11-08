"""Loading and accessors for versioned pipeline configuration."""

import os
from datetime import UTC, datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"


def load_config(path=None):
    """Load a pipeline config and its country metadata."""
    config_path = Path(path or os.environ.get("AFRR_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open() as f:
        config = yaml.safe_load(f)

    countries_path = config_path.parent / config["countries_file"]
    with countries_path.open() as f:
        config["countries"] = yaml.safe_load(f)
    return config


def test_start(config):
    return datetime.fromisoformat(config["data"]["test_start"]).astimezone(UTC)


def shap_seeds(config):
    seeds = config["shap"]["seeds"]
    return range(seeds["start"], seeds["stop"])


DEFAULT_CONFIG = load_config()
TEST_START = test_start(DEFAULT_CONFIG)
TIME_FEATURES = [
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
SHAP_SEEDS = shap_seeds(DEFAULT_CONFIG)
