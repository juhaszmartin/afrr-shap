"""Maintains data/raw/entsoe/ files using the ENTSO-E API.

    python scripts/fill_entsoe_gaps.py fill
    python scripts/fill_entsoe_gaps.py update
    python scripts/fill_entsoe_gaps.py verify
    python scripts/fill_entsoe_gaps.py refresh --country BA --category load --year 2025
    python scripts/fill_entsoe_gaps.py download --countries CZ,SI,EE   # a new country: every category, 2019 to DOWNLOAD_END

Historical gap fills go into fill.parquet. The update command adds recent API data directly
to data.parquet according to the configured lookback policy. Needs ENTSOE_API_KEY in .env.
"""

import argparse
import ast
import functools
import glob
import itertools
import json
import os
import random
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from preprocess_all import CUTOFF_YEARS, HOURLY_COUNTRIES

from afrr_shap.env import load_env
from afrr_shap.paths import ENTSOE_DIR

warnings.filterwarnings("ignore")
load_env()


@functools.cache
def client():
    """ENTSO-E client, built on first use; needs ENTSOE_API_KEY in .env."""
    return EntsoePandasClient(
        api_key=os.environ["ENTSOE_API_KEY"], retry_count=5, retry_delay=10
    )


START, END = pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
GAP = pd.Timedelta(hours=2)  # holes shorter than this are resolution/DST artefacts
MERGE = pd.Timedelta(days=3)  # neighbouring holes are fetched in one request
LOG = ENTSOE_DIR / "fill_log.json"
UPDATE_LOG = ENTSOE_DIR / "update_log.json"
UPDATE_LOOKBACK_DAYS = 0
UPDATE_DELAY_DAYS = 1  # only update through a UTC midnight at least one day old
UPDATE_MAX_CATCHUP_YEARS = 1
UPDATE_LOOKBACK_POLICY = "override"  # "override" or "keep"

# Germany is split into control areas for balancing data and uses the DE_LU bidding zone for markets
DE_AREAS = ["DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET"]
AREA = {"DE": {"prices": "DE_LU", "import": "DE_LU", "net_position": "DE_LU"}}


def _de_volume(s, e):
    """German imbalance volume = sum over the four control areas."""
    parts = []
    for a in DE_AREAS:
        try:
            parts.append(
                client().query_imbalance_volumes(a, start=s, end=e).squeeze().rename(a)
            )
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
            parts.append(
                client().query_activated_balancing_energy_prices(a, start=s, end=e)
            )
        except NoMatchingDataError:
            pass
    if not parts:
        raise NoMatchingDataError
    df = pd.concat(parts).reset_index(names="timestamp")
    return (
        df.groupby(["timestamp", "ReserveType", "Direction"], as_index=False)["Price"]
        .mean()
        .set_index("timestamp")
    )


QUERIES = {
    "imbalance_prices": lambda cc, s, e: client().query_imbalance_prices(
        cc, start=s, end=e
    ),
    "imbalance_volumes": lambda cc, s, e: (
        _de_volume(s, e)
        if cc == "DE"
        else client().query_imbalance_volumes(cc, start=s, end=e)
    ),
    "activated_balancing_energy_prices": lambda cc, s, e: (
        _de_balancing(s, e)
        if cc == "DE"
        else client().query_activated_balancing_energy_prices(cc, start=s, end=e)
    ),
    "prices": lambda cc, s, e: client().query_day_ahead_prices(cc, start=s, end=e),
    "load": lambda cc, s, e: client().query_load(cc, start=s, end=e),
    "load_forecast": lambda cc, s, e: client().query_load_forecast(cc, start=s, end=e),
    "generation": lambda cc, s, e: client().query_generation(cc, start=s, end=e),
    "generation_forecast": lambda cc, s, e: client().query_generation_forecast(
        cc, start=s, end=e
    ),
    "wind_and_solar_forecast": lambda cc, s, e: client().query_wind_and_solar_forecast(
        cc, start=s, end=e
    ),
    "intraday_wind_and_solar_forecast": lambda cc, s, e: (
        client().query_intraday_wind_and_solar_forecast(cc, start=s, end=e)
    ),
    "import": lambda cc, s, e: client().query_import(cc, start=s, end=e),
    "net_position": lambda cc, s, e: client().query_net_position(
        cc, start=s, end=e, dayahead=True
    ),
}
LONG_FORMAT = {"activated_balancing_energy_prices"}  # several rows per timestamp


def norm(name):
    """Comparable column key across entsoe-py versions and header layouts."""
    if isinstance(name, str) and name.startswith("("):
        name = ast.literal_eval(name)
    if isinstance(name, tuple):
        name = " - ".join(str(x) for x in name if x)
    return (
        name.replace("pondage", "poundage").replace(" - Actual Aggregated", "").strip()
    )


def existing_files(cat, cc, name="data"):
    return sorted(
        glob.glob(
            str(ENTSOE_DIR / cat / f"country={cc}" / "year=*" / f"{name}.parquet")
        )
    )


def read_series_files(cat, cc, name):
    """Read one overlay's files for a country/category."""
    return read_existing(existing_files(cat, cc, name))


def read_existing(files):
    """Existing rows as pandas with a UTC 'timestamp' column (original column names kept)."""
    out = []
    for f in files:
        df = pl.read_parquet(f).to_pandas()
        ts = next(c for c in df.columns if "timestamp" in c)
        df = df.rename(columns={ts: "timestamp"})
        df = coalesce_duplicate_columns(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        out.append(df)
    return (
        pd.concat(out, ignore_index=True)
        if out
        else pd.DataFrame({"timestamp": pd.to_datetime([], utc=True)})
    )


def coalesce_duplicate_columns(df):
    """Combine duplicate source headers, keeping the first non-null value per row."""
    if not df.columns.duplicated().any():
        return df
    result = pd.DataFrame(index=df.index)
    for name in pd.unique(df.columns):
        columns = df.loc[:, df.columns == name]
        result[name] = columns.bfill(axis=1).iloc[:, 0]
    return result


def to_frame(res, cat, existing_cols):
    """API result -> DataFrame with UTC 'timestamp' and column names matching the existing files."""
    if isinstance(res, pd.Series):
        data = [c for c in existing_cols if c != "timestamp"]
        name = (
            data[0]
            if len(data) == 1
            else (
                "Scheduled Aggregated"
                if cat == "generation_forecast" and "Scheduled Aggregated" in data
                else "value"
            )
        )
        res = res.rename(name).to_frame()
    res = res.copy()
    res.columns = [str(c) if isinstance(c, tuple) else c for c in res.columns]
    res.index = pd.to_datetime(res.index, utc=True)
    res = (
        res.reset_index(names="timestamp")
        if "timestamp" not in res.columns
        else res.reset_index(drop=True)
    )
    if cat == "generation_forecast":
        res = res.rename(
            columns={
                "Actual Aggregated": "Scheduled Aggregated",
                "Actual Consumption": "Scheduled Consumption",
            }
        )
    if cat == "imbalance_volumes":
        res = res.rename(columns={c: "value" for c in res.columns if c != "timestamp"})
    by_norm = {norm(c): c for c in existing_cols if c != "timestamp"}
    rename = {}
    for c in res.columns:
        if c == "timestamp":
            continue
        rename[c] = by_norm.get(norm(c), str(c) if isinstance(c, tuple) else c)
    return coalesce_duplicate_columns(res.rename(columns=rename))


def series_key(cat):
    """Columns that identify one stored observation for a category."""
    return (
        ["timestamp", "ReserveType", "Direction"]
        if cat in LONG_FORMAT
        else ["timestamp"]
    )


def query_series(cc, cat, start, end, existing_cols):
    """Query one country/category interval and normalize it to stored columns."""
    res = QUERIES[cat](AREA.get(cc, {}).get(cat, cc), start, end)
    return to_frame(res, cat, existing_cols)


def find_holes(ts, start):
    """Missing intervals (start, end) in a sorted array of UTC timestamps, merged and split by year."""
    ts = pd.DatetimeIndex(sorted(set(ts)))
    edges = [start] + list(ts) + [END]
    holes = []
    for a, b in itertools.pairwise(edges):
        if b - a > GAP:
            lo = (
                a
                if a == start and (len(ts) == 0 or a < ts[0])
                else a + pd.Timedelta(minutes=15)
            )
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
    start = pd.Timestamp(
        f"{CUTOFF_YEARS.get(cc, 2019) if cc not in HOURLY_COUNTRIES else 2019}-01-01",
        tz="UTC",
    )
    files = existing_files(cat, cc)
    have = read_existing(files)
    have = have[have["timestamp"] >= start]
    filled = read_series_files(cat, cc, "fill")  # rows added by an earlier run
    known = set(have["timestamp"]) | set(filled["timestamp"])
    holes = find_holes(list(known), start)
    rep = {
        "holes": len(holes),
        "hours_missing": round(sum((b - a).total_seconds() for a, b in holes) / 3600),
        "filled_rows": 0,
        "unavailable": [],
    }
    new_by_year = {}
    for lo, hi in holes:
        try:
            res = query_series(cc, cat, lo, hi, list(have.columns))
        except NoMatchingDataError:
            rep["unavailable"].append([str(lo), str(hi)])
            continue
        except Exception as ex:  # noqa: BLE001 - keep going, but record it
            rep["unavailable"].append(
                [str(lo), str(hi), f"{type(ex).__name__}: {str(ex)[:120]}"]
            )
            continue
        df = res[
            (res["timestamp"] >= lo)
            & (res["timestamp"] < hi)
            & ~res["timestamp"].isin(known)
        ]
        if df.empty:
            rep["unavailable"].append([str(lo), str(hi), "no new rows"])
            continue
        for yr, part in df.groupby(df["timestamp"].dt.year):
            new_by_year.setdefault(yr, []).append(part)
    for yr, parts in new_by_year.items():
        df = pd.concat(
            [filled[filled["timestamp"].dt.year == yr]] + parts, ignore_index=True
        )
        df = df.drop_duplicates(subset=series_key(cat)).sort_values("timestamp")
        out = ENTSOE_DIR / cat / f"country={cc}" / f"year={yr}"
        out.mkdir(parents=True, exist_ok=True)
        pl.from_pandas(df).write_parquet(out / "fill.parquet")
        rep["filled_rows"] += len(df)
    return rep


def update_end(now=None):
    """Return the exclusive UTC cutoff for stable, published data."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    return now.floor("D") - pd.Timedelta(days=UPDATE_DELAY_DAYS)


def update_interval(cc):
    """Return the expected settlement interval for a country."""
    return pd.Timedelta(hours=1) if cc in HOURLY_COUNTRIES else pd.Timedelta(minutes=15)


def _missing(value):
    return bool(pd.isna(value))


def _key(row, columns):
    return tuple(row[column] for column in columns)


def _merge_update_rows(data, stored, new, key, policy):
    """Merge API rows into data, using stored rows for lookback conflicts."""
    result = data.copy()
    all_columns = list(dict.fromkeys([*result.columns, *stored.columns, *new.columns]))
    result = result.reindex(columns=all_columns)
    stored = stored.reindex(columns=all_columns)
    new = new.reindex(columns=all_columns)
    data_indices = {_key(row, key): index for index, row in result.iterrows()}
    stored_rows = {
        _key(row, key): row
        for _, row in stored.drop_duplicates(subset=key, keep="first").iterrows()
    }
    added = overridden = kept = 0
    rows_to_append = []

    for _, incoming in new.iterrows():
        observation_key = _key(incoming, key)
        old = stored_rows.get(observation_key)
        if old is not None and policy == "keep":
            kept += 1
            continue

        if old is None:
            merged = incoming.copy()
            added += 1
        else:
            merged = old.copy()
            for column in all_columns:
                if column not in key and not _missing(incoming[column]):
                    merged[column] = incoming[column]
            overridden += 1

        if observation_key in data_indices:
            index = data_indices[observation_key]
            for column in all_columns:
                result.at[index, column] = merged[column]
        else:
            rows_to_append.append(merged)

    if rows_to_append:
        result = pd.concat([result, pd.DataFrame(rows_to_append)], ignore_index=True)
    return result, added, overridden, kept


def _write_data_partition(df, cat, cc, year):
    """Atomically replace one data partition and preserve the pre-update file."""
    out = ENTSOE_DIR / cat / f"country={cc}" / f"year={year}"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "data.parquet"
    backup = out / "data.parquet.pre_update.bak"
    temporary = out / "data.parquet.update.tmp"
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    pl.from_pandas(df.sort_values("timestamp")).write_parquet(temporary)
    temporary.replace(target)


def update_series(cc, cat):
    """Refresh a recent rolling window directly in data.parquet."""
    if UPDATE_LOOKBACK_POLICY not in {"override", "keep"}:
        raise ValueError("UPDATE_LOOKBACK_POLICY must be either 'override' or 'keep'")
    data = read_series_files(cat, cc, "data")
    filled = read_series_files(cat, cc, "fill")
    existing = pd.concat(
        [df for df in (data, filled) if not df.empty], ignore_index=True
    )
    if existing.empty:
        return {"status": "no_existing_data"}

    existing_cols = list(data.columns)
    if len(existing_cols) == 1 and not filled.empty:
        existing_cols = list(filled.columns)

    latest = existing["timestamp"].max()
    end = update_end()
    interval = update_interval(cc)
    next_expected = latest + interval
    if end <= next_expected:
        return {
            "status": "already_current",
            "latest": str(latest),
            "end": str(end),
            "rows_added": 0,
            "rows_replaced": 0,
        }
    oldest_allowed = end - pd.DateOffset(years=UPDATE_MAX_CATCHUP_YEARS)
    if latest < oldest_allowed:
        lag_days = (end - latest).total_seconds() / (24 * 60 * 60)
        return {
            "status": "skipped_too_far_behind",
            "latest": str(latest),
            "end": str(end),
            "lag_days": round(lag_days, 2),
            "max_catchup_years": UPDATE_MAX_CATCHUP_YEARS,
        }

    start = latest - pd.Timedelta(days=UPDATE_LOOKBACK_DAYS)
    key = series_key(cat)
    try:
        new = query_series(cc, cat, start, end, existing_cols)
    except NoMatchingDataError:
        return {
            "status": "unavailable",
            "start": str(start),
            "end": str(end),
            "rows_added": 0,
            "rows_replaced": 0,
        }
    except Exception as ex:  # noqa: BLE001 - report this series and continue others
        return {
            "status": "error",
            "start": str(start),
            "end": str(end),
            "error": f"{type(ex).__name__}: {str(ex)[:200]}",
        }

    new = new[(new["timestamp"] >= start) & (new["timestamp"] < end)]
    new = coalesce_duplicate_columns(new)
    new = new.drop_duplicates(subset=key).sort_values("timestamp")
    if new.empty:
        return {
            "status": "empty_response",
            "start": str(start),
            "end": str(end),
            "rows_added": 0,
            "rows_replaced": 0,
        }

    merged, added, overridden, kept = _merge_update_rows(
        data, existing, new, key, UPDATE_LOOKBACK_POLICY
    )
    if added or overridden:
        affected_years = set(new["timestamp"].dt.year)
        for year in affected_years:
            _write_data_partition(
                merged[merged["timestamp"].dt.year == year], cat, cc, year
            )

    return {
        "status": "updated",
        "latest_before": str(latest),
        "start": str(start),
        "end": str(end),
        "rows_downloaded": len(new),
        "policy": UPDATE_LOOKBACK_POLICY,
        "rows_added": added,
        "rows_overridden": overridden,
        "rows_kept": kept,
    }


DOWNLOAD_END = pd.Timestamp(
    "2026-11-10", tz="UTC"
)  # in line with the end of the stored 2026 data


def download_series(cc, cat):
    """Downloads a series of a country that is not stored yet, one data.parquet per calendar year (UTC),
    2019 up to DOWNLOAD_END. Years that already have a data.parquet are left alone."""
    rep = {"years": [], "rows": 0, "unavailable": []}
    for year in range(2019, DOWNLOAD_END.year + 1):
        out = ENTSOE_DIR / cat / f"country={cc}" / f"year={year}"
        if (out / "data.parquet").exists():
            continue
        lo = pd.Timestamp(f"{year}-01-01", tz="UTC")
        hi = min(pd.Timestamp(f"{year + 1}-01-01", tz="UTC"), DOWNLOAD_END)
        try:
            df = query_series(cc, cat, lo, hi, ["timestamp"])
        except NoMatchingDataError:
            rep["unavailable"].append(year)
            continue
        except Exception as ex:  # noqa: BLE001 - keep going, but record it
            rep["unavailable"].append([year, f"{type(ex).__name__}: {str(ex)[:120]}"])
            continue
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)]
        df = df.drop_duplicates(subset=series_key(cat)).sort_values("timestamp")
        if df.empty:
            rep["unavailable"].append(year)
            continue
        out.mkdir(parents=True, exist_ok=True)
        pl.from_pandas(df).write_parquet(out / "data.parquet")
        rep["years"].append(year)
        rep["rows"] += len(df)
    return rep


def refresh_year(cc, cat, year):
    """Replaces one stored year with the current API version, for series the TSO revised after download.
    The originals are kept as data.parquet.bak / fill.parquet.bak, which the loaders do not read."""
    out = ENTSOE_DIR / cat / f"country={cc}" / f"year={year}"
    lo, hi = (
        pd.Timestamp(f"{year}-01-01", tz="UTC"),
        pd.Timestamp(f"{year + 1}-01-01", tz="UTC"),
    )
    have = read_existing(existing_files(cat, cc))
    df = query_series(cc, cat, lo, hi, list(have.columns))
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
            res = QUERIES[cat](
                AREA.get(cc, {}).get(cat, cc), d, d + pd.Timedelta(days=1)
            )
        except Exception as ex:  # noqa: BLE001
            rep["days"].append([str(d.date()), f"{type(ex).__name__}"])
            continue
        new = to_frame(res, cat, list(have.columns))
        old = have[
            (have["timestamp"] >= d) & (have["timestamp"] < d + pd.Timedelta(days=1))
        ]
        key = series_key(cat)
        m = old.merge(new, on=key, suffixes=("_old", "_new"))
        cols = [
            c[:-4]
            for c in m.columns
            if c.endswith("_old") and c[:-4] + "_new" in m.columns
        ]
        if cat in LONG_FORMAT:
            cols = ["Price"]
        n_eq = n = 0
        for c in cols:
            a, b = (
                pd.to_numeric(m[f"{c}_old"], errors="coerce"),
                pd.to_numeric(m[f"{c}_new"], errors="coerce"),
            )
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


def available_countries():
    """Return countries represented in the raw ENTSO-E directory."""
    return sorted(
        Path(d).name.split("=")[1]
        for d in glob.glob(str(ENTSOE_DIR / "imbalance_prices" / "country=*"))
    )


def run_series_job(job, countries=None, cats=None, workers=8, log_path=LOG, **kwargs):
    """Run a country/category job in parallel and update its log incrementally."""
    countries = countries or available_countries()
    cats = cats or list(QUERIES)
    tasks = [(cc, cat) for cc in countries for cat in cats]
    results = {}

    def one(t):
        t0 = time.time()
        try:
            r = job(*t, **kwargs)
        except Exception as ex:  # noqa: BLE001 - one broken series must not stop the run
            r = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
        print(
            f"{t[0]} {t[1]}: {json.dumps(r)[:300]} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        return t, r

    with ThreadPoolExecutor(workers) as ex:
        for (cc, cat), r in ex.map(one, tasks):
            results.setdefault(cc, {})[cat] = r
    out = log_path
    if out.exists():  # partial runs update the log instead of replacing it
        old = json.loads(out.read_text())
        for cc, r in results.items():
            old.setdefault(cc, {}).update(r)
        results = old
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"written {out}")


def csv_values(value):
    """argparse converter for comma-separated country/category lists."""
    return [item.strip() for item in value.split(",") if item.strip()]


def positive_int(value):
    """argparse converter for positive integer options."""
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def add_series_options(parser):
    """Add options shared by series-wide commands."""
    parser.add_argument(
        "--countries",
        type=csv_values,
        help="comma-separated country codes (default: all available countries)",
    )
    parser.add_argument(
        "--categories",
        type=csv_values,
        help="comma-separated ENTSO-E categories (default: all categories)",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=8,
        help="number of parallel country/category jobs (default: 8)",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fill = commands.add_parser("fill", help="download missing historical intervals")
    add_series_options(fill)

    update = commands.add_parser(
        "update", help="refresh the recent rolling window from the API"
    )
    add_series_options(update)

    verify = commands.add_parser(
        "verify", help="compare random stored days with the API"
    )
    add_series_options(verify)
    verify.add_argument(
        "--days",
        type=positive_int,
        default=3,
        help="random days to check per series (default: 3)",
    )

    download = commands.add_parser(
        "download",
        help="download every category of a new country, 2019 to DOWNLOAD_END",
    )
    add_series_options(download)

    refresh = commands.add_parser("refresh", help="replace one complete stored year")
    refresh.add_argument("--country", required=True, help="country code")
    refresh.add_argument("--category", required=True, choices=list(QUERIES))
    refresh.add_argument("--year", required=True, type=int)

    args = parser.parse_args(argv)
    if args.command == "download" and not args.countries:
        parser.error("download needs --countries (new countries are not on disk yet)")
    if args.command in {"fill", "update", "verify", "download"} and args.categories:
        invalid = sorted(set(args.categories) - set(QUERIES))
        if invalid:
            parser.error(f"unknown category/categories: {', '.join(invalid)}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not os.environ.get("ENTSOE_API_KEY"):
        sys.exit("missing ENTSOE_API_KEY: add it to .env (see README)")

    if args.command == "fill":
        run_series_job(
            fill_series,
            countries=args.countries,
            cats=args.categories,
            workers=args.workers,
        )
    elif args.command == "update":
        run_series_job(
            update_series,
            countries=args.countries,
            cats=args.categories,
            workers=args.workers,
            log_path=UPDATE_LOG,
        )
    elif args.command == "verify":
        run_series_job(
            verify_series,
            countries=args.countries,
            cats=args.categories,
            workers=args.workers,
            log_path=LOG.with_name("verify_log.json"),
            n_days=args.days,
        )
    elif args.command == "download":
        run_series_job(
            download_series,
            countries=args.countries,
            cats=args.categories,
            workers=args.workers,
            log_path=LOG.with_name("download_log.json"),
        )
    else:
        print(refresh_year(args.country, args.category, args.year))


if __name__ == "__main__":
    main()
