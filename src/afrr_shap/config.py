"""Shared pipeline configuration constants."""

from datetime import datetime, timezone

TEST_START = datetime(2025, 1, 1, tzinfo=timezone.utc)

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

SHAP_SEEDS = range(5)
