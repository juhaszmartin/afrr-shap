import sys
import os
import glob
import math
import polars as pl
import holidays
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

# Mapping from Country Code to ERA5 Country Name
CC_TO_NAME = {
    "AT": "austria", "BE": "belgium", "BA": "bosnia_and_herz", "HR": "croatia",
    "FI": "finland", "GR": "greece", "HU": "hungary", "LV": "latvia",
    "LT": "lithuania", "PL": "poland", "PT": "portugal", "RO": "romania",
    "RS": "serbia", "SK": "slovakia", "ES": "spain", "CH": "switzerland",
    "UA": "ukraine", "DE": "germany", "FR": "france"
}

# Countries where imbalance settlement / aFRR is natively hourly (or 30-min in FR) for most of the data
HOURLY_COUNTRIES = {"BA", "LT", "LV", "PT", "RS", "FR"}

# Cutoff years for countries that transitioned to 15-min settlement under EU EBGL
CUTOFF_YEARS = {"ES": 2022, "RO": 2021, "FI": 2023, "HR": 2024, "PL": 2024}

# 2025 is the test year for every country; everything before it is training data
TEST_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
# Countries whose market changed so much that older data describes a different system
TRAIN_START = {
    # PSE balancing market reform: 15-min settlement and new imbalance pricing from 14 June 2024 (local midnight)
    "PL": datetime(2024, 6, 13, 22, tzinfo=timezone.utc),
    # Before March 2021 the published RS volume is almost always one-signed and ~10x larger
    "RS": datetime(2021, 3, 1, tzinfo=timezone.utc),
}
# Local clock per country for calendar features
LOCAL_TZ = {
    "AT": "Europe/Vienna", "BA": "Europe/Sarajevo", "BE": "Europe/Brussels", "CH": "Europe/Zurich",
    "DE": "Europe/Berlin", "ES": "Europe/Madrid", "FI": "Europe/Helsinki", "FR": "Europe/Paris",
    "GR": "Europe/Athens", "HR": "Europe/Zagreb", "HU": "Europe/Budapest", "LT": "Europe/Vilnius",
    "LV": "Europe/Riga", "PL": "Europe/Warsaw", "PT": "Europe/Lisbon", "RO": "Europe/Bucharest",
    "RS": "Europe/Belgrade", "SK": "Europe/Bratislava",
}
# generation_import repeats the generation and import categories column for column
SKIP_CATEGORIES = {"generation_import"}
# Harmonised technical price limit of the European balancing platforms (PICASSO/MARI);
# balancing-energy prices beyond it are publication errors (e.g. SK aFRR Up -529,189 EUR/MWh on 2025-09-21)
BALANCING_PRICE_LIMIT = 15_000
# Periods judged to be publication errors; every value of the category is blanked, so these rows drop out.
# ES volumes on the local days 9 Feb and 18-24 Feb 2025 swing to +-1,800 MWh per quarter-hour (2022-24 max: 715),
# flat for hours and starting/ending on local midnight, while the imbalance price stays normal. The API serves the same values.
BAD_DATA = {
    ("ES", "imbalance_volumes"): [
        (datetime(2025, 2, 8, 23, tzinfo=timezone.utc), datetime(2025, 2, 9, 23, tzinfo=timezone.utc)),
        (datetime(2025, 2, 17, 23, tzinfo=timezone.utc), datetime(2025, 2, 24, 23, tzinfo=timezone.utc)),
    ],
}

# ENTSO-E passes prices through in the TSO's reporting currency. These series are converted
# to EUR with the ECB daily reference rate (fx_data/ecb_eur_rates.csv, see get_fx_rates.py).
# category -> (currency, UTC instant from which the series is already published in EUR, or None)
LOCAL_CURRENCY = {
    "HU": {"imbalance_prices": ("HUF", None), "activated_balancing_energy_prices": ("HUF", None),
           "procured_balancing_capacity": ("HUF", None)},
    "PL": {"imbalance_prices": ("PLN", None), "activated_balancing_energy_prices": ("PLN", None),
           "prices": ("PLN", datetime(2019, 11, 19, 23))},  # day-ahead moved to EUR with SDAC coupling
    "RO": {"imbalance_prices": ("RON", None), "activated_balancing_energy_prices": ("RON", None),
           "prices": ("RON", datetime(2021, 6, 17, 22))},  # day-ahead moved to EUR with SDAC coupling
    "HR": {"imbalance_prices": ("HRK", datetime(2022, 12, 31, 23))},  # euro adoption
}
PRICE_COLUMNS = {
    "imbalance_prices": ["Long", "Short"],
    "activated_balancing_energy_prices": ["Price"],
    "prices": ["value"],
    "procured_balancing_capacity": ["procured_balancing_capacity - Up - Price", "procured_balancing_capacity - Down - Price"],
}

_fx_rates = None

def convert_to_eur(df_cat, cat, country_code):
    """Converts local-currency price columns of a raw ENTSO-E frame to EUR."""
    rule = LOCAL_CURRENCY.get(country_code, {}).get(cat)
    if rule is None:
        return df_cat
    currency, eur_from = rule

    global _fx_rates
    if _fx_rates is None:
        _fx_rates = pl.read_csv("fx_data/ecb_eur_rates.csv", try_parse_dates=True)

    df_cat = df_cat.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC")).sort("timestamp")
    rates = (
        _fx_rates.filter(pl.col("currency") == currency)
        .select(pl.col("date").cast(pl.Datetime).dt.replace_time_zone("UTC").cast(df_cat.schema["timestamp"]).alias("fx_ts"), "per_eur")
        .sort("fx_ts")
    )
    # Weekends/holidays carry the last published ECB rate
    df_cat = df_cat.join_asof(rates, left_on="timestamp", right_on="fx_ts", strategy="backward")

    is_local = pl.lit(True) if eur_from is None else pl.col("timestamp") < pl.lit(eur_from).dt.replace_time_zone("UTC")
    cols = [c for c in PRICE_COLUMNS[cat] if c in df_cat.columns]
    missing = df_cat.filter(is_local & pl.col("per_eur").is_null() & pl.any_horizontal([pl.col(c).is_not_null() for c in cols])).height
    if missing:
        raise ValueError(f"{country_code}/{cat}: {missing} rows without an {currency} rate")
    df_cat = df_cat.with_columns([
        pl.when(is_local).then(pl.col(c) / pl.col("per_eur")).otherwise(pl.col(c)).alias(c) for c in cols
    ])
    return df_cat.drop("fx_ts", "per_eur")

def procured_capacity(df, cat):
    """Procured balancing capacity (only HU has it) comes with one (direction, tender id, Price/Volume) column
    triple per tender, and the tender ids differ every year. It is aggregated per direction into the total
    procured volume and its volume-weighted price."""
    import ast
    cols = {c: ast.literal_eval(c) for c in df.columns}
    ts = next(c for c, t in cols.items() if t[0] == "timestamp")
    exprs = [pl.col(ts).dt.convert_time_zone("UTC").alias("timestamp")]
    for direction in sorted({t[0] for t in cols.values() if t[0] != "timestamp"}):
        pairs = [(next(p for p, q in cols.items() if q == (d, i, "Price")), c)
                 for c, (d, i, kind) in cols.items() if d == direction and kind == "Volume"]
        vol = pl.sum_horizontal([pl.col(v) for _, v in pairs])
        reported = pl.any_horizontal([pl.col(v).is_not_null() for _, v in pairs])
        weighted = pl.sum_horizontal([pl.col(p) * pl.col(v) for p, v in pairs])
        exprs += [pl.when(reported).then(vol).alias(f"{cat} - {direction} - Volume"),
                  pl.when(reported & (vol > 0)).then(weighted / vol).alias(f"{cat} - {direction} - Price")]
    return df.select(exprs).unique(subset=["timestamp"], keep="first")


def load_entsoe_data(country_code, is_hourly=False):
    """Loads and joins all available ENTSO-E parquet data for a given country."""
    base_dir = Path("entsoe_data")
    if not base_dir.exists():
        raise FileNotFoundError("entsoe_data directory not found.")
        
    categories = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name not in SKIP_CATEGORIES]
    start_year = CUTOFF_YEARS.get(country_code, 2019)
    
    joined_df = None
    for cat in categories:
        # data.parquet is the original download, fill.parquet holds rows added by fill_entsoe_gaps.py
        pattern = f"entsoe_data/{cat}/country={country_code}/year=*/*.parquet"
        files = glob.glob(pattern)
        if not files:
            continue
            
        dfs = []
        for f in files:
            # Check cutoff year for 15-min transitioned countries
            yr = int(f.split("/")[-2].split("=")[1])
            if not is_hourly and yr < start_year:
                continue

            # A file that cannot be read stops the run instead of silently dropping a category
            try:
                df_cat = pl.read_parquet(f)
                if cat == "procured_balancing_capacity":
                    dfs.append(convert_to_eur(procured_capacity(df_cat, cat), cat, country_code))
                    continue
                df_cat = convert_to_eur(df_cat, cat, country_code)
                if cat == "imbalance_volumes":
                    # BA 2019-2021 files call the column "Imbalance Volume", later ones "value"
                    df_cat = df_cat.rename({c: "value" for c in df_cat.columns if c == "Imbalance Volume"})
                for lo, hi in BAD_DATA.get((country_code, cat), []):
                    ts = pl.col("timestamp").dt.convert_time_zone("UTC")
                    bad = (ts >= lo) & (ts < hi)
                    df_cat = df_cat.with_columns([pl.when(bad).then(None).otherwise(pl.col(c)).alias(c)
                                                  for c in df_cat.columns if c != "timestamp"])

                # Fix pandas multi-index tuple columns saved as strings
                if cat == "activated_balancing_energy_prices":
                    if "Direction" in df_cat.columns and "ReserveType" in df_cat.columns and "Price" in df_cat.columns:
                        df_cat = df_cat.with_columns(
                            pl.when(pl.col("Price").abs() <= BALANCING_PRICE_LIMIT).then(pl.col("Price")).alias("Price")
                        ).with_columns(
                            (pl.col("ReserveType") + "_" + pl.col("Direction")).alias("res_dir")
                        )
                        df_cat = df_cat.pivot(
                            values="Price",
                            index="timestamp",
                            on="res_dir",
                            aggregate_function="mean"
                        )
                        new_cols = ["timestamp"] + [f"{cat} - {c}" for c in df_cat.columns if c != "timestamp"]
                        df_cat.columns = new_cols
                        df_cat = df_cat.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
                        dfs.append(df_cat)
                        continue

                new_cols = []
                for c in df_cat.columns:
                    if c in ["('timestamp', '')", "('timestamp', ' ')", "('timestamp', 'UTC')"]:
                        new_cols.append("timestamp")
                    elif c.startswith("('") and c.endswith("')"):
                        import ast
                        try:
                            c_tuple = ast.literal_eval(c)
                            new_cols.append(f"{cat} - " + " - ".join([str(x) for x in c_tuple if x]).strip())
                        except:
                            new_cols.append(f"{cat} - {c}")
                    elif c != "timestamp":
                        new_cols.append(f"{cat} - {c}")
                    else:
                        new_cols.append(c)
                df_cat.columns = new_cols
                
                if "timestamp" not in df_cat.columns:
                    raise ValueError("no timestamp column found")

                df_cat = df_cat.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
                df_cat = df_cat.unique(subset=["timestamp"], keep="first")
                dfs.append(df_cat)
            except Exception as e:
                raise RuntimeError(f"Cannot read {f}: {e}") from e
                
        if dfs:
            combined_cat_df = pl.concat(dfs, how="diagonal").sort("timestamp")
            # Some TSOs switch header layout between years (PL generation: "Biomass" until 2024,
            # ("Biomass", "Actual Aggregated") from 2025); merge both into one column
            for c in [c for c in combined_cat_df.columns if f"{c} - Actual Aggregated" in combined_cat_df.columns]:
                combined_cat_df = combined_cat_df.with_columns(
                    pl.coalesce(f"{c} - Actual Aggregated", c).alias(f"{c} - Actual Aggregated")).drop(c)
            combined_cat_df = combined_cat_df.unique(subset=["timestamp"], keep="last")
            
            if joined_df is None:
                joined_df = combined_cat_df
            else:
                overlap = set(joined_df.columns).intersection(set(combined_cat_df.columns)) - {"timestamp"}
                combined_cat_df = combined_cat_df.drop(list(overlap))
                joined_df = joined_df.join(combined_cat_df, on="timestamp", how="full", coalesce=True)
                
    if joined_df is not None:
        joined_df = joined_df.sort("timestamp")
    return joined_df

def load_era5_data(country_code):
    """Loads and joins ERA5 weather data for a country."""
    if country_code not in CC_TO_NAME:
        return None
    name = CC_TO_NAME[country_code]
    
    files = {
        "temp_mean": f"weather_data/{name}_hourly_temperature_2019_2025.csv",
        "temp_std": f"weather_data/{name}_hourly_temperature_volatility_2019_2025.csv",
        "ssrd_mean": f"weather_data/{name}_hourly_ssrd_2019_2025.csv",
        "ssrd_std": f"weather_data/{name}_hourly_ssrd_volatility_2019_2025.csv",
        "10m_wind_speed_mean": f"weather_data/{name}_hourly_10m_wind_mean_2019_2025.csv",
        "10m_wind_speed_std": f"weather_data/{name}_hourly_10m_wind_std_2019_2025.csv",
        "100m_wind_speed_mean": f"weather_data/{name}_hourly_100m_wind_mean_2019_2025.csv",
        "100m_wind_speed_std": f"weather_data/{name}_hourly_100m_wind_std_2019_2025.csv",
    }
    
    joined_weather = None
    for col_name, filename in files.items():
        if os.path.exists(filename):
            df = pl.read_csv(filename)
            if "datetime" in df.columns:
                df = df.rename({"datetime": "timestamp"})
                df = df.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                )
                df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
                
                if joined_weather is None:
                    joined_weather = df
                else:
                    overlap = set(joined_weather.columns).intersection(set(df.columns)) - {"timestamp"}
                    df = df.drop(list(overlap))
                    joined_weather = joined_weather.join(df, on="timestamp", how="full", coalesce=True)
    
    if joined_weather is not None:
        joined_weather = joined_weather.sort("timestamp")
    return joined_weather

TIME_FEATURES = ["qh_of_day_sin", "qh_of_day_cos", "qh_of_hour_sin", "qh_of_hour_cos",
                 "dayofweek_sin", "dayofweek_cos", "is_weekend", "day_of_year_sin", "day_of_year_cos", "is_holiday"]

def add_time_and_holiday_features(df, country_code, is_hourly):
    """Calendar features on the country's local clock, encoded cyclically so 23:45 sits next to 00:00."""
    local = pl.col("timestamp").dt.convert_time_zone(LOCAL_TZ[country_code])
    qh_of_day = local.dt.hour() * 4 + local.dt.minute() // 15
    df = df.with_columns(
        (2 * np.pi * qh_of_day / 96).sin().alias("qh_of_day_sin"),
        (2 * np.pi * qh_of_day / 96).cos().alias("qh_of_day_cos"),
        # Position within the hour; schedules change on the full hour, so imbalances follow an intra-hour pattern
        (2 * np.pi * (local.dt.minute() // 15) / 4).sin().alias("qh_of_hour_sin"),
        (2 * np.pi * (local.dt.minute() // 15) / 4).cos().alias("qh_of_hour_cos"),
        (2 * np.pi * (local.dt.weekday() - 1) / 7).sin().alias("dayofweek_sin"),
        (2 * np.pi * (local.dt.weekday() - 1) / 7).cos().alias("dayofweek_cos"),
        (local.dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        (2 * np.pi * (local.dt.ordinal_day() - 1) / 365.25).sin().alias("day_of_year_sin"),
        (2 * np.pi * (local.dt.ordinal_day() - 1) / 365.25).cos().alias("day_of_year_cos"),
        local.dt.date().alias("local_date"),
    )
    if is_hourly:
        df = df.drop("qh_of_hour_sin", "qh_of_hour_cos")

    dates = df.select("local_date").unique()
    country_holidays = holidays.country_holidays(country_code, years=range(2018, 2027))
    dates = dates.with_columns(
        pl.Series("is_holiday", [int(d in country_holidays) for d in dates["local_date"]], dtype=pl.Int8))
    return df.join(dates, on="local_date", how="left").drop("local_date").sort("timestamp")

def timelag_expressions(lags, cols, is_hourly=False):
    if isinstance(lags, int):
        if lags > 0:
            lags = range(1, lags + 1)
        else:
            lags = range(lags, 0)
            
    exprs = []
    unit_str = "h" if is_hourly else "min"
    step_mult = 1 if is_hourly else 15
    for i in lags:
        for colname in cols:
            exprs.append(pl.col(colname).shift(-i).alias(f"{colname}_t{i*step_mult:+} {unit_str}".replace(" ", "")))
    return exprs

# Forward fill bridges at most one hour, e.g. hourly series on the 15-min grid; longer holes stay null
FFILL_LIMIT = {False: 3, True: 1}
MAX_NULL_RATIO = 0.10      # per column, checked separately on the train and test portions
MAX_DOMINANT_SHARE = 0.99  # quasi-constant: one value covers more than 99% of a portion
GAP_FLAG = timedelta(days=1)

def report_gaps(df, target_cols, step, country_code, split_ts):
    """Flags holes longer than GAP_FLAG in the target series and writes them to processed_data."""
    ts = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in target_cols])).select("timestamp")
    gaps = (
        ts.with_columns(pl.col("timestamp").shift(1).alias("gap_start"))
        .with_columns((pl.col("timestamp") - pl.col("gap_start")).alias("gap"))
        .filter(pl.col("gap") > step)
        .select("gap_start", pl.col("timestamp").alias("gap_end"), "gap")
    )
    big = gaps.filter(pl.col("gap") > GAP_FLAG)
    for r in big.iter_rows(named=True):
        where = "TEST" if r["gap_end"] >= split_ts else "train"
        print(f"  WARNING {country_code}: {r['gap']} target gap {r['gap_start']} -> {r['gap_end']} ({where} period)")
    Path("processed_data").mkdir(exist_ok=True)
    gaps.with_columns((pl.col("gap").dt.total_minutes() / 60).alias("gap_hours")).drop("gap").write_csv(
        f"processed_data/gaps_{country_code}.csv")
    print(f"  {gaps.height} target gaps in total, {big.height} longer than {GAP_FLAG}")

def select_feature_columns(df, feature_cols, split_ts):
    """Drops columns that are mostly missing or (quasi-)constant in the train or the test portion."""
    portions = {
        "train": df.filter(pl.col("timestamp") < split_ts),
        "test": df.filter(pl.col("timestamp") >= split_ts),
    }
    dropped = {"missing": [], "constant": []}
    keep = []
    for c in feature_cols:
        reason = None
        for part in portions.values():
            col = part[c]
            if col.null_count() / part.height > MAX_NULL_RATIO:
                reason = "missing"
                break
            # Rounding catches float noise around a constant (e.g. 1e-19 from averaging zeros)
            values = col.drop_nulls().round(6)
            if values.len() == 0 or values.value_counts(sort=True)["count"][0] / values.len() > MAX_DOMINANT_SHARE:
                reason = "constant"
                break
        if reason:
            dropped[reason].append(c)
        else:
            keep.append(c)
    for reason, cols in dropped.items():
        print(f"  dropped {len(cols)} {reason} columns" + (f": {sorted(cols)}" if cols else ""))
    return keep

def hourly_energy(df, vol_cols):
    """Imbalance volumes are MWh per settlement period, so an hour is the sum of its periods, not
    their mean. The hourly value is mean x (60 / period length), which equals the sum for complete
    hours and scales up hours with a missing quarter. The period length is the smallest spacing
    within the hour, or within the UTC day when an hour has a single value."""
    out = None
    for c in vol_cols:
        v = (
            df.select("timestamp", c).drop_nulls().sort("timestamp")
            .with_columns(pl.col("timestamp").dt.truncate("1h").alias("hour"),
                          (pl.col("timestamp").diff().dt.total_minutes()).alias("step"))
        )
        h = v.group_by("hour").agg(pl.col(c).mean(), pl.col("step").filter(pl.col("timestamp").dt.minute() > 0).min())
        h = h.with_columns(pl.col("step").min().over(pl.col("hour").dt.date()).alias("day_step"))
        step = pl.coalesce("step", "day_step", pl.lit(60)).clip(15, 60)
        h = h.select(pl.col("hour").alias("timestamp"), (pl.col(c) * 60 / step).alias(c))
        out = h if out is None else out.join(h, on="timestamp", how="full", coalesce=True)
    return out

def known_in_advance(col):
    """Forecasts and day-ahead results are known before t, so they also get future values and are not delayed."""
    return "forecast" in col.lower() or "scheduled" in col.lower() or col.startswith(("prices - ", "net_position - "))

def delay_hourly_actuals(df, cols):
    """A value labelled t covers the period [t, t + step) and is treated as known at t. An hourly actual labelled
    hh:00 covers hh:00-hh+1:00, so on the 15-min grid it may only be used from the row of the hour's last quarter
    (hh:45) on; at hh:00-hh:30 it would hold up to 45 minutes of the future. Days on which a series reports only
    on the full hour are hourly for that series: their values are moved to hh:45 and carried over the next three
    quarters (hh+1:00-hh+1:30). 15-min days are left as they are. Needs the regular 15-min grid."""
    day = pl.col("timestamp").dt.date()
    off_hour = pl.col("timestamp").dt.minute() != 0
    exprs, delayed = [], []
    for c in cols:
        hourly = pl.col(c).is_not_null() & ((pl.col(c).is_not_null() & off_hour).sum().over(day) == 0)
        if not df.select(hourly.any()).item():
            continue
        late = pl.when(hourly).then(pl.col(c)).shift(3).fill_null(strategy="forward", limit=3)
        exprs.append(pl.coalesce(pl.when(~hourly).then(pl.col(c)), late).alias(c))
        delayed.append(c)
    print(f"  {len(delayed)} actual columns have hourly values, delayed to the end of their hour")
    return df.with_columns(exprs)

def process_country(country_code):
    is_hourly = country_code in HOURLY_COUNTRIES
    mode_str = "1-HOUR" if is_hourly else "15-MINUTE"
    step = timedelta(hours=1) if is_hourly else timedelta(minutes=15)
    print(f"\nProcessing {country_code} in {mode_str} mode...")
    
    df_entsoe = load_entsoe_data(country_code, is_hourly=is_hourly)
    if df_entsoe is None:
        print(f"No ENTSO-E data found for {country_code}.")
        return
        
    df_era5 = load_era5_data(country_code)
    weather_cols = set(df_era5.columns) - {"timestamp"} if df_era5 is not None else set()
    if df_era5 is not None:
        df = df_entsoe.join(df_era5, on="timestamp", how="full", coalesce=True)
    else:
        df = df_entsoe

    if country_code in TRAIN_START:
        df = df.filter(pl.col("timestamp") >= TRAIN_START[country_code])

    # Resample everything onto the target resolution (hourly countries average their 15-min data)
    vol_cols = [c for c in df.columns if c.startswith("imbalance_volumes")]
    hourly_volumes = hourly_energy(df, vol_cols) if is_hourly else None
    df = (
        df.with_columns(pl.col("timestamp").dt.truncate("1h" if is_hourly else "15m"))
        .group_by("timestamp")
        .agg(pl.all().mean())
        .sort("timestamp")
    )
    if hourly_volumes is not None:
        df = df.drop(vol_cols).join(hourly_volumes, on="timestamp", how="left")
        
    # Target columns in ENTSO-E
    pos_price_col = [c for c in df.columns if "imbalance_prices" in c and c.endswith("Long")]
    neg_price_col = [c for c in df.columns if "imbalance_prices" in c and c.endswith("Short")]
    imb_vol_col = [c for c in df.columns if "imbalance_volumes" in c and c.endswith("value")]
    
    target_cols = pos_price_col + neg_price_col + imb_vol_col
    if not target_cols:
        print(f"No target columns found for {country_code}. Skipping.")
        return

    # Fix Slovakia single pricing: if Short is 0 and Long != 0, populate Short = Long
    if country_code == "SK" and pos_price_col and neg_price_col:
        df = df.with_columns(
            pl.when((pl.col(neg_price_col[0]) == 0) & (pl.col(pos_price_col[0]) != 0))
            .then(pl.col(pos_price_col[0]))
            .otherwise(pl.col(neg_price_col[0]))
            .alias(neg_price_col[0])
        )

    # Regular time grid over the span with targets, so that a row shift is exactly one time step
    # and lags across missing periods become null instead of silently reaching back days
    has_target = pl.all_horizontal([pl.col(c).is_not_null() for c in target_cols])
    t_min, t_max = df.filter(has_target).select(pl.col("timestamp").min().alias("lo"), pl.col("timestamp").max().alias("hi")).row(0)
    grid = pl.DataFrame({"timestamp": pl.datetime_range(t_min, t_max, step, eager=True, time_zone="UTC")})
    grid = grid.with_columns(pl.col("timestamp").cast(df.schema["timestamp"]))
    df = grid.join(df, on="timestamp", how="left")

    split_ts = TEST_START
    if t_max < split_ts or t_min >= split_ts:
        print(f"Skipping {country_code}: targets run {t_min} → {t_max}, needs data both before and in {split_ts.year}.")
        return
    report_gaps(df, target_cols, step, country_code, split_ts)

    df = add_time_and_holiday_features(df, country_code, is_hourly)
    time_features = [c for c in TIME_FEATURES if c in df.columns]
    # The import sum in the raw files only adds the borders reporting at that timestamp; it is rebuilt below
    exclude_cols = {"timestamp", "Situation", "Óraátállítás", "import - sum"}

    feature_cols = [c for c in df.columns if c not in target_cols and c not in time_features and c not in exclude_cols]
    if not is_hourly:
        # Weather is instantaneous (SSRD accumulates over the preceding hour), so only ENTSO-E actuals are delayed
        df = delay_hourly_actuals(df, [c for c in feature_cols if c not in weather_cols and not known_in_advance(c)])
    # ENTSO-E leaves a series blank when nothing happened: a production type that produced nothing
    # (e.g. BE offshore wind in calm hours) or a balancing product that was not activated (e.g. CH RR).
    # A gap while the rest of the category reports is therefore zero; rows where the whole category
    # is missing stay null.
    for prefix in ["generation - ", "activated_balancing_energy_prices - "]:
        cat_cols = [c for c in feature_cols if c.startswith(prefix)]
        if cat_cols:
            reported = pl.any_horizontal([pl.col(c).is_not_null() for c in cat_cols])
            df = df.with_columns([pl.when(reported).then(pl.col(c).fill_null(0)).otherwise(pl.col(c)).alias(c) for c in cat_cols])
    df = df.with_columns([pl.col(c).fill_null(strategy="forward", limit=FFILL_LIMIT[is_hourly]) for c in feature_cols])

    # Column checks only look at rows that can become samples
    non_time_cols = select_feature_columns(df.filter(has_target), feature_cols, split_ts)
    border_cols = [c for c in non_time_cols if c.startswith("import - ")]
    if border_cols:
        # Total import over the kept borders, only where every border is known (after the 1-hour fill)
        df = df.with_columns(
            pl.when(pl.all_horizontal([pl.col(c).is_not_null() for c in border_cols]))
            .then(pl.sum_horizontal(border_cols)).alias("import - sum"))
        non_time_cols.append("import - sum")
    df = df.select(["timestamp"] + [c for c in time_features if c in df.columns] + non_time_cols + target_cols)
    target_cols_exist = target_cols
    
    weather_cols_exist = [c for c in non_time_cols if c in weather_cols]
    other_cols = list(set(non_time_cols) - set(weather_cols_exist))
    
    # Forecasts and day-ahead results are known in advance, so they also get future values
    forecast_cols = [c for c in other_cols if known_in_advance(c)]
    actual_cols = list(set(other_cols) - set(forecast_cols))
    
    X_exprs = [pl.col("timestamp")]  # Keep timestamp for precise inference alignment
    X_exprs += [pl.col(c) for c in time_features if c in df.columns]
    X_exprs += [pl.col(c) for c in non_time_cols]  # current values (t=0)
    
    # Lags: 15-min vs Hourly (1/4th of lags rounded up)
    if is_hourly:
        # Backward lags: 5 for targets, 3 for actuals/forecasts/weather
        if weather_cols_exist:
            X_exprs += timelag_expressions(range(-3, 0), weather_cols_exist, is_hourly=True)
        if forecast_cols:
            X_exprs += timelag_expressions(range(-3, 0), forecast_cols, is_hourly=True)
        if actual_cols:
            X_exprs += timelag_expressions(range(-3, 0), actual_cols, is_hourly=True)
        if target_cols_exist:
            X_exprs += timelag_expressions(range(-4, 1), target_cols_exist, is_hourly=True)
            
        # Future forecast lags (1/4th rounded up: 3 forward lags for weather and forecasts)
        if weather_cols_exist:
            X_exprs += timelag_expressions(range(1, 4), weather_cols_exist, is_hourly=True)
        if forecast_cols:
            X_exprs += timelag_expressions(range(1, 4), forecast_cols, is_hourly=True)
            
        # Target predictions: 2 forward lags (ceil(5/4) = 2 steps = t+1h, t+2h)
        y_exprs = timelag_expressions(range(1, 3), target_cols_exist, is_hourly=True)
    else:
        # Standard 15-minute lags
        if weather_cols_exist:
            X_exprs += timelag_expressions(range(-10, 0), weather_cols_exist, is_hourly=False)
        if forecast_cols:
            X_exprs += timelag_expressions(range(-10, 0), forecast_cols, is_hourly=False)
        if actual_cols:
            X_exprs += timelag_expressions(range(-10, 0), actual_cols, is_hourly=False)
        if target_cols_exist:
            X_exprs += timelag_expressions(range(-19, 1), target_cols_exist, is_hourly=False)
            
        if weather_cols_exist:
            X_exprs += timelag_expressions(range(1, 11), weather_cols_exist, is_hourly=False)
        if forecast_cols:
            X_exprs += timelag_expressions(range(1, 11), forecast_cols, is_hourly=False)
            
        y_exprs = timelag_expressions(range(1, 6), target_cols_exist, is_hourly=False)
        
    X = df.select(X_exprs)
    y = df.select(y_exprs)
    
    combined = pl.concat([X, y], how="horizontal").drop_nulls()
    X = combined.select(X.columns)
    y = combined.select(y.columns)
    
    out_dir = Path("processed_data")
    out_dir.mkdir(exist_ok=True)
    X.write_parquet(out_dir / f"X_{country_code}.parquet")
    y.write_parquet(out_dir / f"y_{country_code}.parquet")
    print(f"Saved {country_code}: X shape {X.shape}, y shape {y.shape}")

if __name__ == "__main__":
    imbalance_dirs = sorted(glob.glob("entsoe_data/imbalance_prices/country=*"))
    countries = [Path(d).name.split("=")[1] for d in imbalance_dirs]
    if len(sys.argv) > 1:  # optional subset: python scripts/preprocess_all.py DE,FR
        countries = sys.argv[1].split(",")
    for cc in countries:
        process_country(cc)
