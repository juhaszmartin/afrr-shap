"""Fills holes in data/raw/entsoe/ from the ENTSO-E API and spot-checks existing data against it.

    python scripts/fill_entsoe_gaps.py fill     # writes year=YYYY/fill.parquet next to data.parquet
    python scripts/fill_entsoe_gaps.py verify   # re-downloads random days and compares with data.parquet
    python scripts/fill_entsoe_gaps.py fill HU,DE prices,import   # optional country / category subsets

Original data.parquet files are never modified. Rows are only filled for timestamps that are
absent from data.parquet, so existing values always win. Needs ENTSOE_API_KEY in .env.
"""
import ast
import glob
import json
import os
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"], retry_count=5, retry_delay=10)

sys.path.insert(0, str(ROOT / "scripts"))
from preprocess_all import CUTOFF_YEARS, HOURLY_COUNTRIES  # noqa: E402
from afrr_shap.paths import ENTSOE_DIR

START, END = pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
GAP = pd.Timedelta(hours=2)          # holes shorter than this are resolution/DST artefacts
MERGE = pd.Timedelta(days=3)         # neighbouring holes are fetched in one request
LOG = ENTSOE_DIR / "fill_log.json"

# Germany is split into control areas for balancing data and uses the DE_LU bidding zone for markets
DE_AREAS = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"]
AREA = {"DE": {"prices": "DE_LU", "import": "DE_LU", "net_position": "DE_LU"}}


def _de_volume(s, e):
    """German imbalance volume = sum over the four control areas."""
    parts = []
    for a in DE_AREAS:
        try:
            parts.append(client.query_imbalance_volumes(a, start=s, end=e).squeeze().rename(a))
        except NoMatchingDataError:
            pass
    if not parts:
        raise NoMatchingDataError
    return pd.concat(parts, axis=1).sum(axis=1, min_count=1)


def _de_balancing(s, e):
    """German aFRR/mFRR prices are cleared Germany-wide; take the mean over the areas that publish."""
    parts = []
    for a in DE_AREAS:
        try:
            parts.append(client.query_activated_balancing_energy_prices(a, start=s, end=e))
        except NoMatchingDataError:
            pass
    if not parts:
        raise NoMatchingDataError
    df = pd.concat(parts).reset_index(names="timestamp")
    return df.groupby(["timestamp", "ReserveType", "Direction"], as_index=False)["Price"].mean().set_index("timestamp")


QUERIES = {
    "imbalance_prices": lambda cc, s, e: client.query_imbalance_prices(cc, start=s, end=e),
    "imbalance_volumes": lambda cc, s, e: _de_volume(s, e) if cc == "DE" else client.query_imbalance_volumes(cc, start=s, end=e),
    "activated_balancing_energy_prices": lambda cc, s, e: _de_balancing(s, e) if cc == "DE"
        else client.query_activated_balancing_energy_prices(cc, start=s, end=e),
    "prices": lambda cc, s, e: client.query_day_ahead_prices(cc, start=s, end=e),
    "load": lambda cc, s, e: client.query_load(cc, start=s, end=e),
    "load_forecast": lambda cc, s, e: client.query_load_forecast(cc, start=s, end=e),
    "generation": lambda cc, s, e: client.query_generation(cc, start=s, end=e),
    "generation_forecast": lambda cc, s, e: client.query_generation_forecast(cc, start=s, end=e),
    "wind_and_solar_forecast": lambda cc, s, e: client.query_wind_and_solar_forecast(cc, start=s, end=e),
    "intraday_wind_and_solar_forecast": lambda cc, s, e: client.query_intraday_wind_and_solar_forecast(cc, start=s, end=e),
    "import": lambda cc, s, e: client.query_import(cc, start=s, end=e),
    "net_position": lambda cc, s, e: client.query_net_position(cc, start=s, end=e, dayahead=True),
}
LONG_FORMAT = {"activated_balancing_energy_prices"}  # several rows per timestamp


def norm(name):
    """Comparable column key across entsoe-py versions and header layouts."""
    if isinstance(name, str) and name.startswith("("):
        name = ast.literal_eval(name)
    if isinstance(name, tuple):
        name = " - ".join(str(x) for x in name if x)
    return name.replace("pondage", "poundage").replace(" - Actual Aggregated", "").strip()


def existing_files(cat, cc, name="data"):
    return sorted(glob.glob(str(ENTSOE_DIR / cat / f"country={cc}" / "year=*" / f"{name}.parquet")))


def read_existing(files):
    """Existing rows as pandas with a UTC 'timestamp' column (original column names kept)."""
    out = []
    for f in files:
        df = pl.read_parquet(f).to_pandas()
        ts = [c for c in df.columns if "timestamp" in c][0]
        df = df.rename(columns={ts: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame({"timestamp": pd.to_datetime([], utc=True)})


def to_frame(res, cat, existing_cols):
    """API result -> DataFrame with UTC 'timestamp' and column names matching the existing files."""
    if isinstance(res, pd.Series):
        data = [c for c in existing_cols if c != "timestamp"]
        name = data[0] if len(data) == 1 else ("Scheduled Aggregated" if cat == "generation_forecast" and "Scheduled Aggregated" in data else "value")
        res = res.rename(name).to_frame()
    res = res.copy()
    res.columns = [str(c) if isinstance(c, tuple) else c for c in res.columns]
    res.index = pd.to_datetime(res.index, utc=True)
    res = res.reset_index(names="timestamp") if "timestamp" not in res.columns else res.reset_index(drop=True)
    if cat == "generation_forecast":
        res = res.rename(columns={"Actual Aggregated": "Scheduled Aggregated", "Actual Consumption": "Scheduled Consumption"})
    if cat == "imbalance_volumes":
        res = res.rename(columns={c: "value" for c in res.columns if c != "timestamp"})
    by_norm = {norm(c): c for c in existing_cols if c != "timestamp"}
    rename = {}
    for c in res.columns:
        if c == "timestamp":
            continue
        rename[c] = by_norm.get(norm(c), str(c) if isinstance(c, tuple) else c)
    return res.rename(columns=rename)


def find_holes(ts, start):
    """Missing intervals (start, end) in a sorted array of UTC timestamps, merged and split by year."""
    ts = pd.DatetimeIndex(sorted(set(ts)))
    edges = [start] + list(ts) + [END]
    holes = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a > GAP:
            lo = a if a == start and (len(ts) == 0 or a < ts[0]) else a + pd.Timedelta(minutes=15)
            holes.append([lo, b])
    merged = []
    for h in holes:
        if merged and h[0] - merged[-1][1] <= MERGE:
            merged[-1][1] = h[1]
        else:
            merged.append(h)
    out = []
    for lo, hi in merged:
        while lo < hi:
            cut = min(hi, pd.Timestamp(f"{lo.year + 1}-01-01", tz="UTC"))
            out.append((lo, cut))
            lo = cut
    return out


def fill_series(cc, cat):
    start = pd.Timestamp(f"{CUTOFF_YEARS.get(cc, 2019) if cc not in HOURLY_COUNTRIES else 2019}-01-01", tz="UTC")
    files = existing_files(cat, cc)
    have = read_existing(files)
    have = have[have["timestamp"] >= start]
    filled = read_existing(existing_files(cat, cc, "fill"))  # rows added by an earlier run
    known = set(have["timestamp"]) | set(filled["timestamp"])
    holes = find_holes(list(known), start)
    rep = {"holes": len(holes), "hours_missing": round(sum((b - a).total_seconds() for a, b in holes) / 3600), "filled_rows": 0, "unavailable": []}
    new_by_year = {}
    for lo, hi in holes:
        try:
            res = QUERIES[cat](AREA.get(cc, {}).get(cat, cc), lo, hi)
        except NoMatchingDataError:
            rep["unavailable"].append([str(lo), str(hi)])
            continue
        except Exception as ex:  # noqa: BLE001 - keep going, but record it
            rep["unavailable"].append([str(lo), str(hi), f"{type(ex).__name__}: {str(ex)[:120]}"])
            continue
        df = to_frame(res, cat, list(have.columns))
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi) & ~df["timestamp"].isin(known)]
        if df.empty:
            rep["unavailable"].append([str(lo), str(hi), "no new rows"])
            continue
        for yr, part in df.groupby(df["timestamp"].dt.year):
            new_by_year.setdefault(yr, []).append(part)
    for yr, parts in new_by_year.items():
        df = pd.concat([filled[filled["timestamp"].dt.year == yr]] + parts, ignore_index=True)
        key = ["timestamp", "ReserveType", "Direction"] if cat in LONG_FORMAT else ["timestamp"]
        df = df.drop_duplicates(subset=key).sort_values("timestamp")
        out = ENTSOE_DIR / cat / f"country={cc}" / f"year={yr}"
        out.mkdir(parents=True, exist_ok=True)
        pl.from_pandas(df).write_parquet(out / "fill.parquet")
        rep["filled_rows"] += len(df)
    return rep


def refresh_year(cc, cat, year):
    """Replaces one stored year with the current API version, for series the TSO revised after download.
    The originals are kept as data.parquet.bak / fill.parquet.bak, which the loaders do not read."""
    out = ENTSOE_DIR / cat / f"country={cc}" / f"year={year}"
    lo, hi = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    have = read_existing(existing_files(cat, cc))
    df = to_frame(QUERIES[cat](AREA.get(cc, {}).get(cat, cc), lo, hi), cat, list(have.columns))
    df = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)].sort_values("timestamp")
    for name in ("data", "fill"):
        f = out / f"{name}.parquet"
        if f.exists() and not (out / f"{name}.parquet.bak").exists():
            f.rename(out / f"{name}.parquet.bak")
        f.unlink(missing_ok=True)
    pl.from_pandas(df).write_parquet(out / "data.parquet")
    return {"rows": len(df)}


def verify_series(cc, cat, n_days=3, seed=0):
    files = existing_files(cat, cc)
    have = read_existing(files)
    if have.empty:
        return None
    rng = random.Random(f"{seed}-{cc}-{cat}")
    days = sorted(set(have["timestamp"].dt.floor("D")))
    rep = {"days": [], "cells": 0, "equal": 0, "examples": []}
    for d in rng.sample(days, min(n_days, len(days))):
        try:
            res = QUERIES[cat](AREA.get(cc, {}).get(cat, cc), d, d + pd.Timedelta(days=1))
        except Exception as ex:  # noqa: BLE001
            rep["days"].append([str(d.date()), f"{type(ex).__name__}"])
            continue
        new = to_frame(res, cat, list(have.columns))
        old = have[(have["timestamp"] >= d) & (have["timestamp"] < d + pd.Timedelta(days=1))]
        key = ["timestamp", "ReserveType", "Direction"] if cat in LONG_FORMAT else ["timestamp"]
        m = old.merge(new, on=key, suffixes=("_old", "_new"))
        cols = [c[:-4] for c in m.columns if c.endswith("_old") and c[:-4] + "_new" in m.columns]
        if cat in LONG_FORMAT:
            cols = ["Price"]
        n_eq = n = 0
        for c in cols:
            a, b = pd.to_numeric(m[f"{c}_old"], errors="coerce"), pd.to_numeric(m[f"{c}_new"], errors="coerce")
            both = a.notna() & b.notna()
            eq = (np.abs(a - b) <= 1e-3 + 1e-6 * np.abs(a)) | (a.isna() & b.isna())
            n += int((both | a.isna() | b.isna()).sum())
            n_eq += int(eq.sum())
            bad = m.loc[~eq & both, ["timestamp", f"{c}_old", f"{c}_new"]].head(2)
            for r in bad.itertuples(index=False):
                rep["examples"].append([str(r[0]), c, float(r[1]), float(r[2])])
        rep["days"].append([str(d.date()), len(m), len(old), len(new)])
        rep["cells"] += n
        rep["equal"] += n_eq
    return rep


def run(job, countries=None, cats=None, workers=8):
    countries = countries or sorted(
        Path(d).name.split("=")[1]
        for d in glob.glob(str(ENTSOE_DIR / "imbalance_prices" / "country=*"))
    )
    tasks = [(cc, cat) for cc in countries for cat in (cats or QUERIES)]
    results = {}

    def one(t):
        t0 = time.time()
        try:
            r = (fill_series if job == "fill" else verify_series)(*t)
        except Exception as ex:  # noqa: BLE001 - one broken series must not stop the run
            r = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        print(f"{t[0]} {t[1]}: {json.dumps(r)[:300]} ({time.time() - t0:.0f}s)", flush=True)
        return t, r

    with ThreadPoolExecutor(workers) as ex:
        for (cc, cat), r in ex.map(one, tasks):
            results.setdefault(cc, {})[cat] = r
    out = LOG if job == "fill" else LOG.with_name("verify_log.json")
    if out.exists():  # partial runs update the log instead of replacing it
        old = json.loads(out.read_text())
        for cc, r in results.items():
            old.setdefault(cc, {}).update(r)
        results = old
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"written {out}")


if __name__ == "__main__":
    args = sys.argv[1:] + [None] * 3
    if args[0] == "refresh":  # python scripts/fill_entsoe_gaps.py refresh BA load 2025
        print(refresh_year(args[1], args[2], int(args[3])))
        sys.exit()
    run(args[0] or "fill", args[1] and args[1].split(","), args[2] and args[2].split(","))
