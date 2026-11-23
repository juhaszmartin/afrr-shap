"""Loading the processed data of a country and keeping trained models tied to the exact feature layout
they were trained on.

Every country has one model directory, models/{CC}/, which is emptied before the country is trained, so
it never mixes files from different runs. manifest.json records the feature and target names in model
order; anything that loads a model checks the processed data against it and refuses to run on a mismatch.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl
import polars.selectors as cs

from afrr_shap.config import DEV_START, TEST_START
from afrr_shap.paths import MODELS_DIR, PROCESSED_DIR

SPLITS = ("train", "dev", "test")


@dataclass
class CountryData:
    country: str
    timestamps: np.ndarray  # numpy datetime64[us], UTC
    X: np.ndarray  # (samples, features), float64
    y: np.ndarray  # (samples, targets), float64
    feature_names: list
    target_names: list
    masks: dict  # split -> boolean mask

    def split(self, name):
        m = self.masks[name]
        return self.X[m], self.y[m]


def load_country(country_code):
    X = pl.read_parquet(PROCESSED_DIR / f"X_{country_code}.parquet")
    y = pl.read_parquet(PROCESSED_DIR / f"y_{country_code}.parquet")
    ts = X["timestamp"]
    masks = {
        "train": (ts < DEV_START).to_numpy(),
        "dev": ((ts >= DEV_START) & (ts < TEST_START)).to_numpy(),
        "test": (ts >= TEST_START).to_numpy(),
    }
    Xn = X.select(cs.numeric())
    yn = y.select(cs.numeric())
    return CountryData(
        country=country_code,
        timestamps=ts.dt.replace_time_zone(None).to_numpy(),
        X=Xn.to_numpy().astype(np.float64),
        y=yn.to_numpy().astype(np.float64),
        feature_names=Xn.columns,
        target_names=yn.columns,
        masks=masks,
    )


def model_dir(country_code):
    return MODELS_DIR / country_code


def reset_model_dir(country_code):
    """Removes everything from a previous run of this country, so no stale model can be picked up."""
    d = model_dir(country_code)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def layout_hash(names):
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def write_manifest(country_code, data, extra):
    manifest = {
        "country": country_code,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dev_start": DEV_START.isoformat(),
        "test_start": TEST_START.isoformat(),
        "rows": {s: int(m.sum()) for s, m in data.masks.items()},
        "feature_layout": layout_hash(data.feature_names),
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        **extra,
    }
    (model_dir(country_code) / "manifest.json").write_text(
        json.dumps(manifest, indent=1)
    )
    return manifest


def load_manifest(country_code, data=None):
    """Reads the manifest; with data, also checks that the processed data has the trained layout."""
    path = model_dir(country_code) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: train {country_code} first")
    manifest = json.loads(path.read_text())
    if data is not None:
        if data.feature_names != manifest["feature_names"]:
            raise ValueError(
                f"{country_code}: data/processed/X_{country_code}.parquet does not have the feature layout the "
                f"models were trained on ({layout_hash(data.feature_names)} vs {manifest['feature_layout']}). "
                "Retrain after preprocessing."
            )
        if data.target_names != manifest["target_names"]:
            raise ValueError(f"{country_code}: targets differ from the trained models")
    return manifest
