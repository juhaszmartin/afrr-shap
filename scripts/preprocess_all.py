import glob
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import holidays
import numpy as np
import polars as pl

from afrr_shap.config import DEFAULT_CONFIG, DEV_START, TEST_START, TIME_FEATURES
from afrr_shap.paths import (
    BALTIC_DIR,
    ENTSOE_DIR,
    FX_RATES_PATH,
    PROCESSED_DIR,
    WEATHER_DIR,
)

# Mapping from Country Code to ERA5 Country Name
CC_TO_NAME = {
    "AT": "austria",
    "BE": "belgium",
    "BA": "bosnia_and_herz",
    "HR": "croatia",
    "FI": "finland",
    "GR": "greece",
    "HU": "hungary",
    "LV": "latvia",
    "LT": "lithuania",
    "PL": "poland",
    "PT": "portugal",
    "RO": "romania",
    "RS": "serbia",
    "SK": "slovakia",
    "ES": "spain",
    "CH": "switzerland",
    "UA": "ukraine",
    "DE": "germany",
    "FR": "france",
    "CZ": "czechia",
    "SI": "slovenia",
    "EE": "estonia",
}

# Countries where imbalance settlement / aFRR is natively hourly (or 30-min in FR) for most of the data
# EE: hourly until 2024; its 15-min volumes only start on 2025-05-27
HOURLY_COUNTRIES = {"BA", "LT", "LV", "PT", "RS", "FR", "EE"}

# Countries left out of the study, with the reason (checked against the TSO's and regulator's own data)
EXCLUDED_COUNTRIES = {
    "BA": "the ENTSO-E imbalance volume is not the system imbalance (2019-21 unsigned and mostly zero, "
    "2022-25 18-63x the regulator's annual BRP imbalance totals, uncorrelated with NOSBiH's published "
    "deviation); prices are administered and in BAM; no ENTSO-E data after 8 Dec 2025, so no test year",
}

# Cutoff years for countries that transitioned to 15-min settlement under EU EBGL
CUTOFF_YEARS = {
    "ES": 2022,
    "RO": 2021,
    "FI": 2023,
    "HR": 2024,
    "PL": 2024,
    "SI": 2021,
    "CZ": 2024,
}

# 2025 is the test year for every country; everything before it is training data
# Countries whose market changed so much that older data describes a different system
TRAIN_START = {
    # PSE balancing market reform: 15-min settlement and new imbalance pricing from 14 June 2024 (local midnight)
    "PL": datetime(2024, 6, 13, 22, tzinfo=UTC),
    # Before March 2021 the published RS volume is almost always one-signed and ~10x larger
    "RS": datetime(2021, 3, 1, tzinfo=UTC),
    # CZ (OTE) settles every 15 min with a single imbalance price from 1 July 2024 (local midnight). Before, ENTSO-E
    # "Long" held the imbalance price in the system's direction and "Short" the counter-imbalance price, hourly
    "CZ": datetime(2024, 6, 30, 22, tzinfo=UTC),
}
# Local clock per country for calendar features
LOCAL_TZ = {
    "AT": "Europe/Vienna",
    "BA": "Europe/Sarajevo",
    "BE": "Europe/Brussels",
    "CH": "Europe/Zurich",
    "DE": "Europe/Berlin",
    "ES": "Europe/Madrid",
    "FI": "Europe/Helsinki",
    "FR": "Europe/Paris",
    "GR": "Europe/Athens",
    "HR": "Europe/Zagreb",
    "HU": "Europe/Budapest",
    "LT": "Europe/Vilnius",
    "LV": "Europe/Riga",
    "PL": "Europe/Warsaw",
    "PT": "Europe/Lisbon",
    "RO": "Europe/Bucharest",
    "RS": "Europe/Belgrade",
    "SK": "Europe/Bratislava",
    "CZ": "Europe/Prague",
    "SI": "Europe/Ljubljana",
    "EE": "Europe/Tallinn",
}
# generation_import repeats the generation and import categories column for column
SKIP_CATEGORIES = {"generation_import"}
# Harmonised technical price limit of the European balancing platforms (PICASSO/MARI);
# balancing-energy prices beyond it are publication errors (e.g. SK aFRR Up -529,189 EUR/MWh on 2025-09-21)
BALANCING_PRICE_LIMIT = 15_000
# Target series: raw ENTSO-E column -> name used everywhere downstream (features, targets, results, figures).
# ENTSO-E "Long" is the price for a balance responsible party that is long (surplus), "Short" the price for
# one that is short (deficit). Under dual pricing short >= long; under single pricing they are equal.
TARGET_RENAME = {
    "imbalance_prices - Long": "imbalance_price_long",
    "imbalance_prices - Short": "imbalance_price_short",
    "imbalance_volumes - value": "imbalance_volume",
}
LONG_PRICE, SHORT_PRICE, VOLUME = TARGET_RENAME.values()
# Periods in which the TSO published Long and Short the wrong way round. HU until 1 May 2021 (local midnight):
# 'Long' is above 'Short' in every quarter-hour of 2019-2020, which dual pricing rules out; from that day HU is
# single-priced and the two are equal. The two columns are swapped back before this instant (UTC).
SWAPPED_LONG_SHORT = {"HU": datetime(2021, 4, 30, 22, tzinfo=UTC)}
# Periods judged to be publication errors; every value of the category is blanked, so these rows drop out.
# ES volumes on the local days 9 Feb and 18-24 Feb 2025 swing to +-1,800 MWh per quarter-hour (2022-24 max: 715),
# flat for hours and starting/ending on local midnight, while the imbalance price stays normal. The API serves the same values.
BAD_DATA = {
    ("ES", "imbalance_volumes"): [
        (
            datetime(2025, 2, 8, 23, tzinfo=UTC),
            datetime(2025, 2, 9, 23, tzinfo=UTC),
        ),
        (
            datetime(2025, 2, 17, 23, tzinfo=UTC),
            datetime(2025, 2, 24, 23, tzinfo=UTC),
        ),
    ],
    # HU volumes in Jan-Mar 2019 are all positive (absolute values without a sign); signed from 1 April 2019
    ("HU", "imbalance_volumes"): [
        (
            datetime(2018, 12, 31, 23, tzinfo=UTC),
            datetime(2019, 3, 31, 22, tzinfo=UTC),
        ),
    ],
    # RS volumes for 30-31 Dec 2024 are published quarter-hourly while EMS still settled hourly, at the
    # inflated 15-min magnitude and not alignable with the EMS balancing data at any time shift
    ("RS", "imbalance_volumes"): [
        (
            datetime(2024, 12, 30, 0, tzinfo=UTC),
            datetime(2025, 1, 1, 0, tzinfo=UTC),
        ),
    ],
}

# Imbalance volume sign convention, for every country: positive = system long (surplus), negative = short
# (deficit), as the EB GL 'imbalance' (Art. 2(8)) and as published by most TSOs. Periods in which the
# published volume has the opposite sign are multiplied by -1. Bounds are local dates [start, end); None = open.
VOLUME_SIGN_FLIP = {
    # Checked against APG's own control-area imbalance (DRZ, positive = surplus; transparency.apg.at API
    # from 2023, Deltaregelzone archive before): ENTSO-E AT = -APG/4 in every month except May 2020,
    # Nov 2022, Jan 2023, Jul 2024 and from Dec 2024, where it is +APG/4.
    "AT": [
        (None, (2020, 5, 1)),
        ((2020, 6, 1), (2022, 11, 1)),
        ((2022, 12, 1), (2023, 1, 1)),
        ((2023, 2, 1), (2024, 7, 1)),
        ((2024, 8, 1), (2024, 12, 1)),
    ],
    # MAVIR publishes 'KE' (imbalance energy), whose sign is opposite to the EB GL imbalance (MAVIR BRP working
    # group, 14 Aug 2025); the HU volume rises with the imbalance price in every month from April 2019.
    "HU": [(None, None)],
}

# Categories read from the Baltic TSOs' own Transparency Dashboard (scripts/baltic_download.py) instead of
# ENTSO-E, per country: (dashboard area, {ENTSO-E category: dashboard report}). Checked for EE, 2019-2026:
# - imbalance volumes: identical to ENTSO-E until Oct 2024; ENTSO-E then has none until 2025-05-27, is
#   patchy until Dec 2025 and holds -1 x the dashboard value in 2025 (inverted sign); in 2026 ENTSO-E differs
#   from the dashboard, and the dashboard volume tracks the imbalance price more closely in every month;
# - imbalance prices: ENTSO-E holds Elering's preliminary price from Aug 2025 (Jul 2025 matches neither);
#   the dashboard's final price is used, its preliminary one only where no final exists yet (the last weeks);
# - balancing prices: ENTSO-E EE mFRR prices end in Oct 2024 and are blank in hours without activation;
#   the dashboard's balancing energy reference price equals them where they exist and is complete 2019-2026.
BALTIC_SOURCES = {
    "EE": (
        "Estonia",
        {
            "imbalance_prices": "imbalance_prices",
            "imbalance_volumes": "imbalance_volumes_v2",
            "activated_balancing_energy_prices": "balancing_energy_prices",
        },
    ),
}
# Until this instant (EE: local midnight 1 April 2025) the dashboard repeats the hourly value in all four
# quarter-hours of the hour (Jan-Mar 2025; hourly rows before). Volumes are MWh per period, so these quarter-hour
# repeats are divided by 4; summed per hour they give the hourly value instead of 4x it. Hourly rows are untouched.
BALTIC_HOURLY_UNTIL = {"EE": datetime(2025, 3, 31, 21, tzinfo=UTC)}

# RS from 2025 (quarter-hourly publication), checked against EMS's own balancing data (aFRR, mFRR, IGCC):
# - timestamps are early by the Belgrade UTC offset in 2025 (1 h in CET, 2 h in CEST) and by 1 h in 2026;
#   the value labelled t belongs to t + shift. Unshifted, the volume does not track the EMS IGCC exchange
#   (corr ~ -0.07); shifted it does (-0.4 to -0.55, every month), as the hourly data does unshifted.
# - values are MW (average power) rather than MWh per quarter-hour: after alignment the volume is ~5x as
#   large relative to EMS's IGCC energy as before 2025, while EMS's own energies scale correctly; /4 makes
#   the hourly-equivalent volume continuous with 2023-24 instead of jumping 6.5x on 1 January 2025.
RS_15MIN_FROM = datetime(2024, 12, 31, 23, tzinfo=UTC)
RS_2026_FROM = datetime(2025, 12, 31, 23, tzinfo=UTC)


def local_instant(country_code, ymd):
    """Local midnight of a (year, month, day) date as a UTC datetime."""
    from zoneinfo import ZoneInfo

    return datetime(*ymd, tzinfo=ZoneInfo(LOCAL_TZ[country_code])).astimezone(UTC)


def fix_volume_sign(df_cat, country_code):
    """Multiplies the volume by -1 in the periods of VOLUME_SIGN_FLIP (see there)."""
    periods = VOLUME_SIGN_FLIP.get(country_code)
    if not periods:
        return df_cat
    ts = pl.col("timestamp").dt.convert_time_zone("UTC")
    flip = pl.lit(False)
    for start, end in periods:
        cond = pl.lit(True)
        if start is not None:
            cond = cond & (ts >= local_instant(country_code, start))
        if end is not None:
            cond = cond & (ts < local_instant(country_code, end))
        flip = flip | cond
    return df_cat.with_columns(
        pl.when(flip).then(-pl.col("value")).otherwise(pl.col("value")).alias("value")
    )


def fix_rs_volume(df_cat):
    """Moves the 2025+ RS volumes to their true quarter-hour and converts MW to MWh (see RS_15MIN_FROM)."""
    ts = pl.col("timestamp").dt.convert_time_zone("UTC")
    shift = (
        pl.when(ts < RS_2026_FROM)
        .then(
            pl.duration(hours=1)
            + ts.dt.convert_time_zone("Europe/Belgrade").dt.dst_offset()
        )
        .otherwise(pl.duration(hours=1))
    )
    is_15min = ts >= RS_15MIN_FROM
    return (
        df_cat.with_columns(
            pl.when(is_15min)
            .then(ts + shift)
            .otherwise(ts)
            .cast(pl.Datetime("us", "UTC"))
            .alias("timestamp"),
            pl.when(is_15min)
            .then(pl.col("value") / 4)
            .otherwise(pl.col("value"))
            .alias("value"),
        )
        # a DST change in the shift maps two labels onto one quarter-hour; keep the first
        .unique(subset=["timestamp"], keep="first", maintain_order=True)
    )


# ENTSO-E passes prices through in the TSO's reporting currency. These series are converted
# to EUR with the ECB daily reference rate (data/reference/ecb_eur_rates.csv, see get_fx_rates.py).
# category -> (currency, UTC instant from which the series is already published in EUR, or None)
LOCAL_CURRENCY = {
    "HU": {
        "imbalance_prices": ("HUF", None),
        "activated_balancing_energy_prices": ("HUF", None),
        "procured_balancing_capacity": ("HUF", None),
    },
    "PL": {
        "imbalance_prices": ("PLN", None),
        "activated_balancing_energy_prices": ("PLN", None),
        "prices": ("PLN", datetime(2019, 11, 19, 23)),
    },  # day-ahead moved to EUR with SDAC coupling
    "RO": {
        "imbalance_prices": ("RON", None),
        "activated_balancing_energy_prices": ("RON", None),
        "prices": ("RON", datetime(2021, 6, 17, 22)),
    },  # day-ahead moved to EUR with SDAC coupling
    "HR": {"imbalance_prices": ("HRK", datetime(2022, 12, 31, 23))},  # euro adoption
    # CZ imbalance prices are CZK throughout (XML checked 2019-2026). aFRR/mFRR prices were CZK until March 2022 and
    # EUR from April 2022 (RR always EUR); CZ starts in 2024 (TRAIN_START), so only the imbalance prices need converting
    "CZ": {"imbalance_prices": ("CZK", None)},
}
PRICE_COLUMNS = {
    "imbalance_prices": ["Long", "Short"],
    "activated_balancing_energy_prices": ["Price"],
    "prices": ["value"],
    "procured_balancing_capacity": [
        "procured_balancing_capacity - Up - Price",
        "procured_balancing_capacity - Down - Price",
    ],
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
        _fx_rates = pl.read_csv(FX_RATES_PATH, try_parse_dates=True)

    df_cat = df_cat.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC")).sort(
        "timestamp"
    )
    rates = (
        _fx_rates.filter(pl.col("currency") == currency)
        .select(
            pl.col("date")
            .cast(pl.Datetime)
            .dt.replace_time_zone("UTC")
            .cast(df_cat.schema["timestamp"])
            .alias("fx_ts"),
            "per_eur",
        )
        .sort("fx_ts")
    )
    # Weekends/holidays carry the last published ECB rate
    df_cat = df_cat.join_asof(
        rates, left_on="timestamp", right_on="fx_ts", strategy="backward"
    )

    is_local = (
        pl.lit(True)
        if eur_from is None
        else pl.col("timestamp") < pl.lit(eur_from).dt.replace_time_zone("UTC")
    )
    cols = [c for c in PRICE_COLUMNS[cat] if c in df_cat.columns]
    missing = df_cat.filter(
        is_local
        & pl.col("per_eur").is_null()
        & pl.any_horizontal([pl.col(c).is_not_null() for c in cols])
    ).height
    if missing:
        raise ValueError(
            f"{country_code}/{cat}: {missing} rows without an {currency} rate"
        )
    df_cat = df_cat.with_columns(
        [
            pl.when(is_local)
            .then(pl.col(c) / pl.col("per_eur"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in cols
        ]
    )
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
        pairs = [
            (next(p for p, q in cols.items() if q == (d, i, "Price")), c)
            for c, (d, i, kind) in cols.items()
            if d == direction and kind == "Volume"
        ]
        vol = pl.sum_horizontal([pl.col(v) for _, v in pairs])
        reported = pl.any_horizontal([pl.col(v).is_not_null() for _, v in pairs])
        weighted = pl.sum_horizontal([pl.col(p) * pl.col(v) for p, v in pairs])
        exprs += [
            pl.when(reported).then(vol).alias(f"{cat} - {direction} - Volume"),
            pl.when(reported & (vol > 0))
            .then(weighted / vol)
            .alias(f"{cat} - {direction} - Price"),
        ]
    return df.select(exprs).unique(subset=["timestamp"], keep="first")


def spread_repeated_quarters(df, until):
    """Before `until` the source repeats the hourly volume (MWh per hour) in each quarter-hour: divide by 4, so that
    each quarter holds its share and the hour sums to the hourly value (whatever period length is inferred).
    Hours with a single (hourly) row are left as they are."""
    ts = pl.col("timestamp")
    early = (ts < until) & (pl.len().over(ts.dt.truncate("1h")) > 1)
    return df.with_columns(
        [
            pl.when(early).then(pl.col(c) / 4).otherwise(pl.col(c)).alias(c)
            for c in df.columns
            if c != "timestamp"
        ]
    )


def load_baltic(country_code, cat):
    """One category from the Baltic Transparency Dashboard, in the column layout load_entsoe_data builds."""
    area, reports = BALTIC_SOURCES[country_code]
    files = sorted(
        glob.glob(str(BALTIC_DIR / reports[cat] / "year=*" / "data.parquet"))
    )
    if not files:
        raise FileNotFoundError(
            f"no {reports[cat]} in {BALTIC_DIR}; run scripts/baltic_download.py"
        )
    df = pl.concat([pl.read_parquet(f) for f in files]).filter(pl.col("area") == area)
    wide = df.pivot("label", index="timestamp", values="value").sort("timestamp")
    if cat == "imbalance_prices":  # single price: the same value for long and short
        price = pl.coalesce("Final", "Preliminary")
        return wide.select(
            "timestamp", price.alias(f"{cat} - Long"), price.alias(f"{cat} - Short")
        )
    if cat == "imbalance_volumes":
        wide = wide.select("timestamp", pl.col("value").alias(f"{cat} - value"))
        if country_code in BALTIC_HOURLY_UNTIL:
            wide = spread_repeated_quarters(wide, BALTIC_HOURLY_UNTIL[country_code])
        return wide
    if cat == "activated_balancing_energy_prices":
        limit = lambda c: pl.when(pl.col(c).abs() <= BALANCING_PRICE_LIMIT).then(
            pl.col(c)
        )
        return wide.select(
            "timestamp",
            limit("Upward").alias(f"{cat} - reference_Up"),
            limit("Downward").alias(f"{cat} - reference_Down"),
        )
    raise ValueError(f"no Baltic source for {cat}")


def load_entsoe_data(country_code, is_hourly=False):
    """Loads and joins all available ENTSO-E parquet data for a given country."""
    base_dir = ENTSOE_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} directory not found.")

    # Sorted, so that the column order of the output never depends on the file system
    categories = sorted(
        d.name
        for d in base_dir.iterdir()
        if d.is_dir() and d.name not in SKIP_CATEGORIES
    )
    start_year = CUTOFF_YEARS.get(country_code, 2019)

    joined_df = None
    for cat in categories:
        # data.parquet is the original/updateable download; fill.parquet holds historical
        # rows added by fill_entsoe_gaps.py.
        pattern = str(
            ENTSOE_DIR / cat / f"country={country_code}" / "year=*" / "*.parquet"
        )
        files = sorted(
            (
                f
                for f in glob.glob(pattern)
                if Path(f).name in {"data.parquet", "fill.parquet"}
            ),
            key=lambda f: {"fill.parquet": 0, "data.parquet": 1}[Path(f).name],
        )
        if not files:
            continue

        dfs = []
        for f in files:
            # Check cutoff year for 15-min transitioned countries
            yr = int(f.split("/")[-2].split("=")[1])
            if not is_hourly and yr < start_year:
                continue
            priority = {
                "fill.parquet": 0,
                "data.parquet": 1,
            }.get(Path(f).name, -1)

            # A file that cannot be read stops the run instead of silently dropping a category
            try:
                df_cat = pl.read_parquet(f)
                if cat == "procured_balancing_capacity":
                    processed = convert_to_eur(
                        procured_capacity(df_cat, cat), cat, country_code
                    )
                    dfs.append(
                        processed.with_columns(
                            pl.lit(priority).alias("_source_priority")
                        )
                    )
                    continue
                df_cat = convert_to_eur(df_cat, cat, country_code)
                if cat == "imbalance_prices" and country_code in SWAPPED_LONG_SHORT:
                    # with_columns evaluates both expressions on the input frame, so this is a swap
                    swapped = pl.col("timestamp").dt.convert_time_zone("UTC") < pl.lit(
                        SWAPPED_LONG_SHORT[country_code]
                    )
                    df_cat = df_cat.with_columns(
                        pl.when(swapped)
                        .then(pl.col("Short"))
                        .otherwise(pl.col("Long"))
                        .alias("Long"),
                        pl.when(swapped)
                        .then(pl.col("Long"))
                        .otherwise(pl.col("Short"))
                        .alias("Short"),
                    )
                if cat == "imbalance_volumes":
                    # BA 2019-2021 files call the column "Imbalance Volume", later ones "value"
                    df_cat = df_cat.rename(
                        {c: "value" for c in df_cat.columns if c == "Imbalance Volume"}
                    )
                    df_cat = fix_volume_sign(df_cat, country_code)
                    if country_code == "RS":
                        df_cat = fix_rs_volume(df_cat)
                for lo, hi in BAD_DATA.get((country_code, cat), []):
                    ts = pl.col("timestamp").dt.convert_time_zone("UTC")
                    bad = (ts >= lo) & (ts < hi)
                    df_cat = df_cat.with_columns(
                        [
                            pl.when(bad).then(None).otherwise(pl.col(c)).alias(c)
                            for c in df_cat.columns
                            if c != "timestamp"
                        ]
                    )

                # Fix pandas multi-index tuple columns saved as strings
                if cat == "activated_balancing_energy_prices":  # noqa: SIM102
                    if (
                        "Direction" in df_cat.columns
                        and "ReserveType" in df_cat.columns
                        and "Price" in df_cat.columns
                    ):
                        df_cat = df_cat.with_columns(
                            pl.when(pl.col("Price").abs() <= BALANCING_PRICE_LIMIT)
                            .then(pl.col("Price"))
                            .alias("Price")
                        ).with_columns(
                            (pl.col("ReserveType") + "_" + pl.col("Direction")).alias(
                                "res_dir"
                            )
                        )
                        df_cat = df_cat.pivot(
                            values="Price",
                            index="timestamp",
                            on="res_dir",
                            aggregate_function="mean",
                        )
                        new_cols = ["timestamp"] + [
                            f"{cat} - {c}" for c in df_cat.columns if c != "timestamp"
                        ]
                        df_cat.columns = new_cols
                        df_cat = df_cat.with_columns(
                            pl.col("timestamp").dt.convert_time_zone("UTC")
                        )
                        dfs.append(
                            df_cat.with_columns(
                                pl.lit(priority).alias("_source_priority")
                            )
                        )
                        continue

                new_cols = []
                for c in df_cat.columns:
                    if c in [
                        "('timestamp', '')",
                        "('timestamp', ' ')",
                        "('timestamp', 'UTC')",
                    ]:
                        new_cols.append("timestamp")
                    elif c.startswith("('") and c.endswith("')"):
                        import ast

                        try:
                            c_tuple = ast.literal_eval(c)
                            new_cols.append(
                                f"{cat} - "
                                + " - ".join([str(x) for x in c_tuple if x]).strip()
                            )
                        except ValueError, SyntaxError:
                            new_cols.append(f"{cat} - {c}")
                    elif c != "timestamp":
                        new_cols.append(f"{cat} - {c}")
                    else:
                        new_cols.append(c)
                df_cat.columns = new_cols

                if "timestamp" not in df_cat.columns:
                    raise ValueError("no timestamp column found")

                df_cat = df_cat.with_columns(
                    pl.col("timestamp").dt.convert_time_zone("UTC")
                )
                df_cat = df_cat.unique(subset=["timestamp"], keep="first")
                dfs.append(
                    df_cat.with_columns(pl.lit(priority).alias("_source_priority"))
                )
            except Exception as e:
                raise RuntimeError(f"Cannot read {f}: {e}") from e

        if cat in BALTIC_SOURCES.get(country_code, (None, {}))[1]:
            dfs = [load_baltic(country_code, cat)]
        if dfs:
            combined_cat_df = pl.concat(dfs, how="diagonal").sort(
                ["timestamp", "_source_priority"]
            )
            # Some TSOs switch header layout between years (PL generation: "Biomass" until 2024,
            # ("Biomass", "Actual Aggregated") from 2025); merge both into one column
            for c in [
                c
                for c in combined_cat_df.columns
                if f"{c} - Actual Aggregated" in combined_cat_df.columns
            ]:
                combined_cat_df = combined_cat_df.with_columns(
                    pl.coalesce(f"{c} - Actual Aggregated", c).alias(
                        f"{c} - Actual Aggregated"
                    )
                ).drop(c)
            combined_cat_df = combined_cat_df.unique(
                subset=["timestamp"], keep="last"
            ).drop("_source_priority")

            if joined_df is None:
                joined_df = combined_cat_df
            else:
                overlap = set(joined_df.columns).intersection(
                    set(combined_cat_df.columns)
                ) - {"timestamp"}
                combined_cat_df = combined_cat_df.drop(list(overlap))
                joined_df = joined_df.join(
                    combined_cat_df, on="timestamp", how="full", coalesce=True
                )

    if joined_df is not None:
        joined_df = joined_df.sort("timestamp").rename(
            {k: v for k, v in TARGET_RENAME.items() if k in joined_df.columns}
        )
    return joined_df


def load_era5_data(country_code):
    """Loads and joins ERA5 weather data for a country."""
    if country_code not in CC_TO_NAME:
        return None
    name = CC_TO_NAME[country_code]

    files = {
        "temp_mean": WEATHER_DIR / f"{name}_hourly_temperature_2019_2025.csv",
        "temp_std": WEATHER_DIR / f"{name}_hourly_temperature_volatility_2019_2025.csv",
        "ssrd_mean": WEATHER_DIR / f"{name}_hourly_ssrd_2019_2025.csv",
        "ssrd_std": WEATHER_DIR / f"{name}_hourly_ssrd_volatility_2019_2025.csv",
        "10m_wind_speed_mean": WEATHER_DIR
        / f"{name}_hourly_10m_wind_mean_2019_2025.csv",
        "10m_wind_speed_std": WEATHER_DIR / f"{name}_hourly_10m_wind_std_2019_2025.csv",
        "100m_wind_speed_mean": WEATHER_DIR
        / f"{name}_hourly_100m_wind_mean_2019_2025.csv",
        "100m_wind_speed_std": WEATHER_DIR
        / f"{name}_hourly_100m_wind_std_2019_2025.csv",
    }

    joined_weather = None
    for filename in files.values():
        if filename.exists():
            df = pl.read_csv(filename)
            if "datetime" in df.columns:
                df = df.rename({"datetime": "timestamp"})
                df = df.with_columns(
                    pl.col("timestamp").str.strptime(
                        pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False
                    )
                )
                df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

                if joined_weather is None:
                    joined_weather = df
                else:
                    overlap = set(joined_weather.columns).intersection(
                        set(df.columns)
                    ) - {"timestamp"}
                    df = df.drop(list(overlap))
                    joined_weather = joined_weather.join(
                        df, on="timestamp", how="full", coalesce=True
                    )

    if joined_weather is not None:
        joined_weather = joined_weather.sort("timestamp")
    return joined_weather


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
        (2 * np.pi * (local.dt.ordinal_day() - 1) / 365.25)
        .sin()
        .alias("day_of_year_sin"),
        (2 * np.pi * (local.dt.ordinal_day() - 1) / 365.25)
        .cos()
        .alias("day_of_year_cos"),
        local.dt.date().alias("local_date"),
    )
    if is_hourly:
        df = df.drop("qh_of_hour_sin", "qh_of_hour_cos")

    dates = df.select("local_date").unique()
    country_holidays = holidays.country_holidays(country_code, years=range(2018, 2027))
    dates = dates.with_columns(
        pl.Series(
            "is_holiday",
            [int(d in country_holidays) for d in dates["local_date"]],
            dtype=pl.Int8,
        )
    )
    return (
        df.join(dates, on="local_date", how="left").drop("local_date").sort("timestamp")
    )


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
            exprs.append(
                pl.col(colname)
                .shift(-i)
                .alias(f"{colname}_t{i * step_mult:+} {unit_str}".replace(" ", ""))
            )
    return exprs


# Forward fill bridges at most one hour, e.g. hourly series on the 15-min grid; longer holes stay null
PREPROCESSING_CONFIG = DEFAULT_CONFIG["preprocessing"]
FFILL_LIMIT = {
    False: PREPROCESSING_CONFIG["forward_fill_limit"]["fifteen_minute"],
    True: PREPROCESSING_CONFIG["forward_fill_limit"]["hourly"],
}
MAX_NULL_RATIO = PREPROCESSING_CONFIG["max_null_ratio"]
MAX_DOMINANT_SHARE = PREPROCESSING_CONFIG["max_dominant_share"]
GAP_FLAG = timedelta(days=PREPROCESSING_CONFIG["gap_warning_days"])


def report_gaps(df, target_cols, step, country_code, split_ts):
    """Flags holes longer than GAP_FLAG in the target series and writes them to data/processed/.
    split_ts is the start of the dev period."""
    ts = df.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in target_cols])
    ).select("timestamp")
    gaps = (
        ts.with_columns(pl.col("timestamp").shift(1).alias("gap_start"))
        .with_columns((pl.col("timestamp") - pl.col("gap_start")).alias("gap"))
        .filter(pl.col("gap") > step)
        .select("gap_start", pl.col("timestamp").alias("gap_end"), "gap")
    )
    big = gaps.filter(pl.col("gap") > GAP_FLAG)
    for r in big.iter_rows(named=True):
        where = (
            "train"
            if r["gap_end"] < split_ts
            else "dev"
            if r["gap_end"] < TEST_START
            else "test"
        )
        print(
            f"  WARNING {country_code}: {r['gap']} target gap {r['gap_start']} -> {r['gap_end']} ({where} period)"
        )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gaps.with_columns((pl.col("gap").dt.total_minutes() / 60).alias("gap_hours")).drop(
        "gap"
    ).write_csv(PROCESSED_DIR / f"gaps_{country_code}.csv")
    print(f"  {gaps.height} target gaps in total, {big.height} longer than {GAP_FLAG}")


def select_feature_columns(df, feature_cols, split_ts, eval_end_ts=None):
    """Drops columns that are mostly missing or (quasi-)constant in the train or the dev portion.

    With eval_end_ts (the start of the test period), columns that are mostly missing in the test portion
    are dropped too. Only availability is checked there, never the values: an input the TSO stopped
    publishing cannot be used by a forecaster in that period, and keeping it would silently remove every
    test row in the final drop of incomplete samples (ES balancing prices and GR generation in 2026)."""
    dev_filter = (
        (pl.col("timestamp") >= split_ts) & (pl.col("timestamp") < eval_end_ts)
        if eval_end_ts
        else (pl.col("timestamp") >= split_ts)
    )
    portions = {
        "train": df.filter(pl.col("timestamp") < split_ts),
        "dev": df.filter(dev_filter),
    }
    test = df.filter(pl.col("timestamp") >= eval_end_ts) if eval_end_ts else df.clear()
    dropped = {"missing": [], "constant": [], "missing in test": []}
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
            if (
                values.len() == 0
                or values.value_counts(sort=True)["count"][0] / values.len()
                > MAX_DOMINANT_SHARE
            ):
                reason = "constant"
                break
        if (
            reason is None
            and test.height
            and test[c].null_count() / test.height > MAX_NULL_RATIO
        ):
            reason = "missing in test"
        if reason:
            dropped[reason].append(c)
        else:
            keep.append(c)
    for reason, cols in dropped.items():
        print(
            f"  dropped {len(cols)} {reason} columns"
            + (f": {sorted(cols)}" if cols else "")
        )
    return keep


def hourly_energy(df, vol_cols):
    """Imbalance volumes are MWh per settlement period, so an hour is the sum of its periods, not
    their mean. The hourly value is mean x (60 / period length), which equals the sum for complete
    hours and scales up hours with a missing quarter. The period length is the smallest spacing
    within the hour, or within the UTC day when an hour has a single value."""
    out = None
    for c in vol_cols:
        v = (
            df.select("timestamp", c)
            .drop_nulls()
            .sort("timestamp")
            .with_columns(
                pl.col("timestamp").dt.truncate("1h").alias("hour"),
                (pl.col("timestamp").diff().dt.total_minutes()).alias("step"),
            )
        )
        h = v.group_by("hour").agg(
            pl.col(c).mean(),
            pl.col("step").filter(pl.col("timestamp").dt.minute() > 0).min(),
        )
        h = h.with_columns(
            pl.col("step").min().over(pl.col("hour").dt.date()).alias("day_step")
        )
        step = pl.coalesce("step", "day_step", pl.lit(60)).clip(15, 60)
        h = h.select(
            pl.col("hour").alias("timestamp"), (pl.col(c) * 60 / step).alias(c)
        )
        out = (
            h if out is None else out.join(h, on="timestamp", how="full", coalesce=True)
        )
    return out


def known_in_advance(col):
    """Forecasts and day-ahead results are known before t, so they also get future values and are not delayed."""
    return (
        "forecast" in col.lower()
        or "scheduled" in col.lower()
        or col.startswith(("prices - ", "net_position - "))
    )


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
        hourly = pl.col(c).is_not_null() & (
            (pl.col(c).is_not_null() & off_hour).sum().over(day) == 0
        )
        if not df.select(hourly.any()).item():
            continue
        late = (
            pl.when(hourly)
            .then(pl.col(c))
            .shift(3)
            .fill_null(strategy="forward", limit=3)
        )
        exprs.append(pl.coalesce(pl.when(~hourly).then(pl.col(c)), late).alias(c))
        delayed.append(c)
    print(
        f"  {len(delayed)} actual columns have hourly values, delayed to the end of their hour"
    )
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
    weather_cols = (
        set(df_era5.columns) - {"timestamp"} if df_era5 is not None else set()
    )
    if df_era5 is not None:
        df = df_entsoe.join(df_era5, on="timestamp", how="full", coalesce=True)
    else:
        df = df_entsoe

    if country_code in TRAIN_START:
        df = df.filter(pl.col("timestamp") >= TRAIN_START[country_code])

    # Resample everything onto the target resolution (hourly countries average their 15-min data)
    vol_cols = [c for c in df.columns if c == VOLUME]
    hourly_volumes = hourly_energy(df, vol_cols) if is_hourly else None
    df = (
        df.with_columns(pl.col("timestamp").dt.truncate("1h" if is_hourly else "15m"))
        .group_by("timestamp")
        .agg(pl.all().mean())
        .sort("timestamp")
    )
    if hourly_volumes is not None:
        df = df.drop(vol_cols).join(hourly_volumes, on="timestamp", how="left")

    # Targets, in a fixed order: long price, short price, volume
    target_cols = [c for c in TARGET_RENAME.values() if c in df.columns]
    if not target_cols:
        print(f"No target columns found for {country_code}. Skipping.")
        return

    # SK publishes a short price of 0 in single-priced periods; there the short price equals the long price
    if country_code == "SK" and {LONG_PRICE, SHORT_PRICE} <= set(target_cols):
        df = df.with_columns(
            pl.when((pl.col(SHORT_PRICE) == 0) & (pl.col(LONG_PRICE) != 0))
            .then(pl.col(LONG_PRICE))
            .otherwise(pl.col(SHORT_PRICE))
            .alias(SHORT_PRICE)
        )

    # Regular time grid over the span with targets, so that a row shift is exactly one time step
    # and lags across missing periods become null instead of silently reaching back days
    has_target = pl.all_horizontal([pl.col(c).is_not_null() for c in target_cols])
    t_min, t_max = (
        df.filter(has_target)
        .select(
            pl.col("timestamp").min().alias("lo"), pl.col("timestamp").max().alias("hi")
        )
        .row(0)
    )
    grid = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                t_min, t_max, step, eager=True, time_zone="UTC"
            )
        }
    )
    grid = grid.with_columns(pl.col("timestamp").cast(df.schema["timestamp"]))
    df = grid.join(df, on="timestamp", how="left")

    split_ts = DEV_START
    if t_max < split_ts or t_min >= split_ts:
        print(
            f"Skipping {country_code}: targets run {t_min} → {t_max}, needs data both before and in {split_ts.year}."
        )
        return
    report_gaps(df, target_cols, step, country_code, split_ts)

    df = add_time_and_holiday_features(df, country_code, is_hourly)
    time_features = [c for c in TIME_FEATURES if c in df.columns]
    # The import sum in the raw files only adds the borders reporting at that timestamp; it is rebuilt below
    exclude_cols = {"timestamp", "Situation", "Óraátállítás", "import - sum"}

    feature_cols = [
        c
        for c in df.columns
        if c not in target_cols and c not in time_features and c not in exclude_cols
    ]
    if not is_hourly:
        # Weather is instantaneous (SSRD accumulates over the preceding hour), so only ENTSO-E actuals are delayed
        df = delay_hourly_actuals(
            df,
            [
                c
                for c in feature_cols
                if c not in weather_cols and not known_in_advance(c)
            ],
        )
    # ENTSO-E leaves a series blank when nothing happened: a production type that produced nothing
    # (e.g. BE offshore wind in calm hours) or a balancing product that was not activated (e.g. CH RR).
    # A gap while the rest of the category reports is therefore zero; rows where the whole category
    # is missing stay null.
    for prefix in ["generation - ", "activated_balancing_energy_prices - "]:
        cat_cols = [c for c in feature_cols if c.startswith(prefix)]
        if cat_cols:
            reported = pl.any_horizontal([pl.col(c).is_not_null() for c in cat_cols])
            df = df.with_columns(
                [
                    pl.when(reported)
                    .then(pl.col(c).fill_null(0))
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in cat_cols
                ]
            )
    df = df.with_columns(
        [
            pl.col(c).fill_null(strategy="forward", limit=FFILL_LIMIT[is_hourly])
            for c in feature_cols
        ]
    )

    # Column checks only look at rows that can become samples
    non_time_cols = select_feature_columns(
        df.filter(has_target), feature_cols, split_ts, eval_end_ts=TEST_START
    )
    border_cols = [c for c in non_time_cols if c.startswith("import - ")]
    if border_cols:
        # Total import over the kept borders, only where every border is known (after the 1-hour fill)
        df = df.with_columns(
            pl.when(pl.all_horizontal([pl.col(c).is_not_null() for c in border_cols]))
            .then(pl.sum_horizontal(border_cols))
            .alias("import - sum")
        )
        non_time_cols.append("import - sum")
    df = df.select(
        ["timestamp"]
        + [c for c in time_features if c in df.columns]
        + non_time_cols
        + target_cols
    )
    target_cols_exist = target_cols

    # Sorted lists, never sets: set order changes between Python processes (string hashing is randomised),
    # which used to give every preprocessing run a different column order
    weather_cols_exist = sorted(c for c in non_time_cols if c in weather_cols)
    other_cols = sorted(c for c in non_time_cols if c not in weather_cols)

    # Forecasts and day-ahead results are known in advance, so they also get future values
    forecast_cols = [c for c in other_cols if known_in_advance(c)]
    actual_cols = [c for c in other_cols if not known_in_advance(c)]

    X_exprs = [pl.col("timestamp")]  # Keep timestamp for precise inference alignment
    X_exprs += [pl.col(c) for c in time_features if c in df.columns]
    X_exprs += [pl.col(c) for c in non_time_cols]  # current values (t=0)

    # Lags: 15-min vs Hourly (1/4th of lags rounded up)
    if is_hourly:
        # Backward lags: 5 for targets, 3 for actuals/forecasts/weather
        if weather_cols_exist:
            X_exprs += timelag_expressions(
                range(-3, 0), weather_cols_exist, is_hourly=True
            )
        if forecast_cols:
            X_exprs += timelag_expressions(range(-3, 0), forecast_cols, is_hourly=True)
        if actual_cols:
            X_exprs += timelag_expressions(range(-3, 0), actual_cols, is_hourly=True)
        if target_cols_exist:
            X_exprs += timelag_expressions(
                range(-4, 1), target_cols_exist, is_hourly=True
            )

        # Future forecast lags (1/4th rounded up: 3 forward lags for weather and forecasts)
        if weather_cols_exist:
            X_exprs += timelag_expressions(
                range(1, 4), weather_cols_exist, is_hourly=True
            )
        if forecast_cols:
            X_exprs += timelag_expressions(range(1, 4), forecast_cols, is_hourly=True)

        # Target predictions: 2 forward lags (ceil(5/4) = 2 steps = t+1h, t+2h)
        y_exprs = timelag_expressions(range(1, 3), target_cols_exist, is_hourly=True)
    else:
        # Standard 15-minute lags
        if weather_cols_exist:
            X_exprs += timelag_expressions(
                range(-10, 0), weather_cols_exist, is_hourly=False
            )
        if forecast_cols:
            X_exprs += timelag_expressions(
                range(-10, 0), forecast_cols, is_hourly=False
            )
        if actual_cols:
            X_exprs += timelag_expressions(range(-10, 0), actual_cols, is_hourly=False)
        if target_cols_exist:
            X_exprs += timelag_expressions(
                range(-19, 1), target_cols_exist, is_hourly=False
            )

        if weather_cols_exist:
            X_exprs += timelag_expressions(
                range(1, 11), weather_cols_exist, is_hourly=False
            )
        if forecast_cols:
            X_exprs += timelag_expressions(range(1, 11), forecast_cols, is_hourly=False)

        y_exprs = timelag_expressions(range(1, 6), target_cols_exist, is_hourly=False)

    X = df.select(X_exprs)
    y = df.select(y_exprs)

    # Rows that have every target, before incomplete feature rows are dropped (for the report below)
    split_of = (
        pl.when(pl.col("timestamp") < DEV_START)
        .then(pl.lit("train"))
        .when(pl.col("timestamp") < TEST_START)
        .then(pl.lit("dev"))
        .otherwise(pl.lit("test"))
    )
    with_targets = (
        X.select("timestamp")
        .hstack(y)
        .drop_nulls()
        .group_by(split_of.alias("split"))
        .len()
    )
    combined = X.hstack(y).drop_nulls()
    # Features in sorted order, so the model input layout is the same in every run
    X = combined.select(
        ["timestamp", *sorted(c for c in X.columns if c != "timestamp")]
    )
    y = combined.select(y.columns)

    kept = combined.group_by(split_of.alias("split")).len()
    report = with_targets.join(kept, on="split", how="left", suffix="_kept").fill_null(
        0
    )
    for split in ("train", "dev", "test"):
        row = report.filter(pl.col("split") == split)
        n_targets, n_kept = (row.row(0)[1], row.row(0)[2]) if row.height else (0, 0)
        print(f"  {split}: {n_kept} samples kept of {n_targets} with targets")
        if n_targets and n_kept < 0.8 * n_targets:
            print(
                f"  WARNING {country_code}: incomplete features remove "
                f"{1 - n_kept / n_targets:.0%} of the {split} rows"
            )

    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    X.write_parquet(out_dir / f"X_{country_code}.parquet")
    y.write_parquet(out_dir / f"y_{country_code}.parquet")
    print(f"Saved {country_code}: X shape {X.shape}, y shape {y.shape}")


if __name__ == "__main__":
    imbalance_dirs = sorted(
        glob.glob(str(ENTSOE_DIR / "imbalance_prices" / "country=*"))
    )
    countries = [Path(d).name.split("=")[1] for d in imbalance_dirs]
    if len(sys.argv) > 1:  # optional subset: python scripts/preprocess_all.py DE,FR
        countries = sys.argv[1].split(",")
    for cc in countries:
        if cc in EXCLUDED_COUNTRIES:
            print(f"\nSkipping {cc}: {EXCLUDED_COUNTRIES[cc]}")
            for stale in (
                PROCESSED_DIR / f"X_{cc}.parquet",
                PROCESSED_DIR / f"y_{cc}.parquet",
            ):
                stale.unlink(missing_ok=True)
            continue
        process_country(cc)
