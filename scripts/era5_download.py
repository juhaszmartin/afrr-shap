"""Downloads ERA5 weather and writes weather_data/{country}_hourly_*_2019_2025.csv: the hourly box mean and std
of 2m temperature, SSRD and 10m/100m wind speed over a lat/lon box per country.

The CDS queue, not the data volume, is the bottleneck. Neighbouring countries therefore share a region: one request
per region and 4 months (the CDS limit is 121k fields = variables x days x hours) over a box covering the whole
region, from which each country's box is cropped locally. The crop reproduces the per-country downloads of the
original notebook (legacy/era5_eu.ipynb) to float32 precision (checked for BE and CH).

Needs a CDS key in ~/.cdsapirc.
    python scripts/era5_download.py                    # every country whose CSVs are missing from weather_data/
    python scripts/era5_download.py germany,france     # these countries (re-download only what is not in era5_raw/)
"""
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

# Bounding boxes [North, West, South, East], mainland only, grouped into download regions
REGIONS = {
    "de_fr": {
        "germany": [55.1, 5.8, 47.2, 15.1],
        "france": [51.1, -4.8, 42.3, 8.2],
        "belgium": [51.5, 2.5, 49.4, 6.5],
        "switzerland": [47.9, 5.9, 45.8, 10.5],
    },
    "central": {
        "austria": [49.0, 9.5, 46.3, 17.2],
        "slovakia": [49.6, 16.8, 47.7, 22.6],
        "hungary": [48.61, 16.0765, 45.7, 22.9],
        "croatia": [46.6, 13.4, 42.3, 19.5],
        "bosnia_and_herz": [45.3, 15.7, 42.5, 19.7],
        "serbia": [46.2, 18.8, 42.2, 23.1],
    },
    "southeast": {
        "romania": [48.3, 20.2, 43.6, 29.8],
        "greece": [41.8, 19.3, 34.8, 28.3],
    },
    "iberia": {
        "spain": [43.8, -9.4, 35.9, 4.4],
        "portugal": [42.2, -9.6, 36.9, -6.1],
    },
    "north": {
        "finland": [70.1, 20.5, 59.8, 31.6],
        "latvia": [58.1, 20.9, 55.6, 28.3],
        "lithuania": [56.5, 20.9, 53.8, 26.9],
        "poland": [54.9, 14.1, 49.0, 24.2],
    },
    "ukraine": {
        "ukraine": [52.4, 22.1, 44.3, 40.3],
    },
}
COUNTRIES = {c: box for boxes in REGIONS.values() for c, box in boxes.items()}
REGION_OF = {c: r for r, boxes in REGIONS.items() for c in boxes}
YEARS = range(2019, 2026)
VARIABLES = ["2m_temperature", "surface_solar_radiation_downwards",
             "10m_u_component_of_wind", "10m_v_component_of_wind",
             "100m_u_component_of_wind", "100m_v_component_of_wind"]
PARTS = {"p1": range(1, 5), "p2": range(5, 9), "p3": range(9, 13)}  # 6 variables x 4 months is just under the CDS limit
RAW = Path("era5_raw")
OUT = Path("weather_data")


def region_area(region):
    boxes = REGIONS[region].values()
    return [max(b[0] for b in boxes), min(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes)]


def download(job):
    region, year, part = job
    target = RAW / region / f"{year}_{part}"
    if target.is_dir() and any(target.glob("*.nc")):
        return
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / "download.part"
    for _ in range(30):
        try:
            cdsapi.Client(quiet=True).retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": VARIABLES,
                    "year": [str(year)],
                    "month": [f"{m:02d}" for m in PARTS[part]],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": region_area(region),
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                str(tmp),
            )
            break
        except Exception as e:  # CDS rejects jobs when too many are queued; wait and retry
            if "queued requests" not in str(e):
                raise
            time.sleep(120)
    # Variables with different step types (instantaneous vs accumulated) can arrive as a zip of .nc files
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            z.extractall(target)
        tmp.unlink()
    else:
        tmp.rename(target / f"era5_{year}_{part}.nc")
    print(f"downloaded {region} {year} {part}", flush=True)


def country_stats(ds, box):
    north, west, south, east = box
    sub = ds.sel(latitude=slice(north, south), longitude=slice(west, east))
    dims = ["latitude", "longitude"]
    fields = {
        "temperature": sub["t2m"] - 273.15,
        "ssrd": sub["ssrd"],
        "10m_wind": np.sqrt(sub["u10"] ** 2 + sub["v10"] ** 2),
        "100m_wind": np.sqrt(sub["u100"] ** 2 + sub["v100"] ** 2),
    }
    return {k: (v.mean(dims).to_series(), v.std(dims).to_series()) for k, v in fields.items()}


# File name and column of the mean and std CSVs
FILES = {
    "temperature": [("hourly_temperature", "temperature_mean_celsius"), ("hourly_temperature_volatility", "temperature_std_celsius")],
    "ssrd": [("hourly_ssrd", "ssrd_mean_j_m2"), ("hourly_ssrd_volatility", "ssrd_std_j_m2")],
    "10m_wind": [("hourly_10m_wind_mean", "10m_wind_speed_mean"), ("hourly_10m_wind_std", "10m_wind_speed_std")],
    "100m_wind": [("hourly_100m_wind_mean", "100m_wind_speed_mean"), ("hourly_100m_wind_std", "100m_wind_speed_std")],
}


def csv_paths(country):
    return [OUT / f"{country}_{name}_2019_2025.csv" for files in FILES.values() for name, _ in files]


def process(countries):
    parts = {c: {k: ([], []) for k in FILES} for c in countries}
    for region in sorted({REGION_OF[c] for c in countries}):
        for year, part in [(y, p) for y in YEARS for p in PARTS]:
            ds = xr.merge([xr.open_dataset(f) for f in sorted((RAW / region / f"{year}_{part}").glob("*.nc"))], compat="override")
            ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.variables])
            if "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            for country in [c for c in countries if REGION_OF[c] == region]:
                for k, (mean, std) in country_stats(ds, COUNTRIES[country]).items():
                    parts[country][k][0].append(mean)
                    parts[country][k][1].append(std)
            print(f"processed {region} {year} {part}", flush=True)
    OUT.mkdir(exist_ok=True)
    for country in countries:
        for k, series in parts[country].items():
            for (name, column), s in zip(FILES[k], series):
                s = pd.concat(s).sort_index()
                s = s[~s.index.duplicated()]
                df = pd.DataFrame({"datetime": s.index.strftime("%Y-%m-%d %H:%M:%S"), column: s.values})
                df.to_csv(OUT / f"{country}_{name}_2019_2025.csv", index=False)
        print(f"wrote {country}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        countries = sys.argv[1].split(",")
        unknown = set(countries) - set(COUNTRIES)
        if unknown:
            sys.exit(f"unknown countries {sorted(unknown)}; known: {sorted(COUNTRIES)}")
    else:
        countries = [c for c in COUNTRIES if not all(p.exists() for p in csv_paths(c))]
    if not countries:
        print("weather_data/ is complete; pass country names to rebuild some")
        sys.exit()
    regions = sorted({REGION_OF[c] for c in countries})
    print(f"countries {countries} -> regions {regions}")
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(download, [(r, y, p) for r in regions for y in YEARS for p in PARTS]))
    process(countries)
