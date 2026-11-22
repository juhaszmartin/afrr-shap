"""Downloads ERA5 weather and writes data/raw/weather/{country}_hourly_*_2019_2025.csv: the hourly box mean and std
of 2m temperature, SSRD and 10m/100m wind speed over a lat/lon box per country.

By default every 0.25 degree grid point inside the box is fetched on its own from the ERA5 point time-series dataset
(~35 s per point for 2019-2026), one grid point at a time per worker; the values equal the gridded data.

With --grid, the gridded dataset is used instead. Its CDS queue, not the data volume, is the bottleneck (waits of up to
~10 h per request were seen in Nov 2026). Neighbouring countries therefore share a region: one request
per region and 4 months (the CDS limit is 121k fields = variables x days x hours) over a box covering the whole
region, from which each country's box is cropped locally. The crop reproduces the per-country downloads of the
original notebook (legacy/notebooks/era5_eu.ipynb) to float32 precision (checked for BE and CH).

Needs CDSAPI_URL and CDSAPI_KEY in .env.
    python scripts/era5_download.py                    # every country whose CSVs are missing from data/raw/weather/
    python scripts/era5_download.py germany,france     # these countries (re-download only what is not in data/interim/era5_raw/)
    python scripts/era5_download.py --grid germany     # gridded requests per region instead of grid points
"""

import functools
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cdsapi
import numpy as np
import pandas as pd
import requests
import xarray as xr

from afrr_shap.env import load_env
from afrr_shap.paths import ERA5_RAW_DIR, WEATHER_DIR

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
    # Added 2026-11-18; one region, since the CDS queue (not the area) limits the download
    "cz_si_ee": {
        "czechia": [51.1, 12.1, 48.5, 18.9],
        "slovenia": [46.9, 13.4, 45.4, 16.6],
        "estonia": [59.7, 23.4, 57.5, 28.2],
    },
}
COUNTRIES = {c: box for boxes in REGIONS.values() for c, box in boxes.items()}
REGION_OF = {c: r for r, boxes in REGIONS.items() for c in boxes}
YEARS = range(2019, 2027)  # cached region files are reused, so only missing years are downloaded
VARIABLES = [
    "2m_temperature",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
]
PARTS = {
    "p1": range(1, 5),
    "p2": range(5, 9),
    "p3": range(9, 13),
}  # 6 variables x 4 months is just under the CDS limit
RAW = ERA5_RAW_DIR
OUT = WEATHER_DIR


def region_area(region):
    boxes = REGIONS[region].values()
    return [
        max(b[0] for b in boxes),
        min(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _cds():
    return os.environ["CDSAPI_URL"].rstrip("/") + "/retrieve/v1", {"PRIVATE-TOKEN": os.environ["CDSAPI_KEY"]}


def _key(req):
    return tuple(tuple(req[k]) for k in ("year", "month", "area", "variable"))


@functools.cache
def earlier_jobs():
    """Requests already at CDS (queued, running or finished), keyed by year, months, area and variables, so that a
    restarted run collects them instead of queueing them again (CDS allows only a few queued requests per dataset)."""
    base, h = _cds()
    jobs = {}
    for status in ("successful", "running", "accepted"):
        listed = requests.get(f"{base}/jobs", params={"limit": 100, "status": status}, headers=h, timeout=60)
        for j in listed.json().get("jobs", []):
            meta = requests.get(f"{base}/jobs/{j['jobID']}", params={"request": "true"}, headers=h, timeout=60)
            ids = meta.json().get("metadata", {}).get("request", {}).get("ids", {})
            if all(k in ids for k in ("year", "month", "area", "variable")):
                jobs.setdefault(_key(ids), j["jobID"])
    return jobs


def fetch_job(job_id, path):
    """Waits for an earlier CDS job and downloads its result to path; False if it failed or its result expired."""
    base, h = _cds()
    while True:
        status = requests.get(f"{base}/jobs/{job_id}", headers=h, timeout=60).json().get("status")
        if status == "successful":
            break
        if status not in ("accepted", "running"):
            return False
        time.sleep(60)
    res = requests.get(f"{base}/jobs/{job_id}/results", headers=h, timeout=60)
    if res.status_code != 200:
        return False
    with requests.get(res.json()["asset"]["value"]["href"], stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            f.writelines(r.iter_content(1 << 20))
    return True


def download(job):
    region, year, part = job
    target = RAW / region / f"{year}_{part}"
    if target.is_dir() and any(target.glob("*.nc")):
        return
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / "download.part"
    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in PARTS[part]],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": region_area(region),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    earlier = earlier_jobs().get(_key(request))
    if not (earlier and fetch_job(earlier, tmp)):
        for _ in range(200):
            try:
                cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels", request, str(tmp))
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
    return {
        k: (v.mean(dims).to_series(), v.std(dims).to_series())
        for k, v in fields.items()
    }


# File name and column of the mean and std CSVs
FILES = {
    "temperature": [
        ("hourly_temperature", "temperature_mean_celsius"),
        ("hourly_temperature_volatility", "temperature_std_celsius"),
    ],
    "ssrd": [
        ("hourly_ssrd", "ssrd_mean_j_m2"),
        ("hourly_ssrd_volatility", "ssrd_std_j_m2"),
    ],
    "10m_wind": [
        ("hourly_10m_wind_mean", "10m_wind_speed_mean"),
        ("hourly_10m_wind_std", "10m_wind_speed_std"),
    ],
    "100m_wind": [
        ("hourly_100m_wind_mean", "100m_wind_speed_mean"),
        ("hourly_100m_wind_std", "100m_wind_speed_std"),
    ],
}


def csv_paths(country):
    return [
        OUT / f"{country}_{name}_2019_2025.csv"
        for files in FILES.values()
        for name, _ in files
    ]


def process(countries):
    parts = {c: {k: ([], []) for k in FILES} for c in countries}
    for region in sorted({REGION_OF[c] for c in countries}):
        for year, part in [(y, p) for y in YEARS for p in PARTS]:
            ds = xr.merge(
                [
                    xr.open_dataset(f)
                    for f in sorted((RAW / region / f"{year}_{part}").glob("*.nc"))
                ],
                compat="override",
            )
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
                df = pd.DataFrame(
                    {
                        "datetime": s.index.strftime("%Y-%m-%d %H:%M:%S"),
                        column: s.values,
                    }
                )
                df.to_csv(OUT / f"{country}_{name}_2019_2025.csv", index=False)
        print(f"wrote {country}", flush=True)


# Grid-point mode (--points): the CDS ERA5 time-series dataset serves the whole 2019-2026 series of one grid point
# in about 30 s, without the multi-hour queue of the gridded requests (about 10 h per request in Nov 2026). The box
# statistics are then computed over the 0.25 degree grid points inside the country box, the same points the gridded
# crop uses; the point values equal the gridded ones (checked at 46.0N 14.5E, Jan-Apr 2026: max difference 0.0003 K
# for t2m, 0 for ssrd, 2e-5 m/s for wind).
POINT_DIR = RAW / "points"
POINT_DATES = "2019-01-01/2026-11-10"


def grid_points(box):
    """ERA5 0.25 degree grid points inside a [North, West, South, East] box, edges included."""
    north, west, south, east = box
    lats = np.arange(np.ceil(south / 0.25) * 0.25, north + 1e-9, 0.25)
    lons = np.arange(np.ceil(west / 0.25) * 0.25, east + 1e-9, 0.25)
    return [(round(float(la), 2), round(float(lo), 2)) for la in lats for lo in lons]


def point_path(point):
    return POINT_DIR / f"{point[0]:.2f}_{point[1]:.2f}.csv"


def download_point(point):
    out = point_path(point)
    if out.exists():
        return
    POINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    request = {
        "variable": VARIABLES,
        "location": {"latitude": point[0], "longitude": point[1]},
        "date": [POINT_DATES],
        "data_format": "csv",
    }
    for _ in range(500):
        try:
            cdsapi.Client(quiet=True).retrieve("reanalysis-era5-single-levels-timeseries", request, str(tmp))
            break
        except Exception as e:  # CDS rejects jobs when too many are queued; wait and retry
            if "queued requests" not in str(e):
                raise
            time.sleep(20)
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            csv = next(n for n in z.namelist() if n.endswith(".csv"))
            out.with_suffix(".csv.part").write_bytes(z.read(csv))
        tmp.unlink()
        out.with_suffix(".csv.part").rename(out)
    else:
        tmp.rename(out)


def process_points(countries):
    """Box mean and std per hour from the grid-point series, written like process()."""
    OUT.mkdir(exist_ok=True)
    for country in countries:
        frames = [
            pd.read_csv(point_path(pt), parse_dates=["valid_time"], index_col="valid_time")
            for pt in grid_points(COUNTRIES[country])
        ]
        fields = {
            "temperature": pd.concat([f["t2m"] - 273.15 for f in frames], axis=1),
            "ssrd": pd.concat([f["ssrd"] for f in frames], axis=1),
            "10m_wind": pd.concat([np.sqrt(f["u10"] ** 2 + f["v10"] ** 2) for f in frames], axis=1),
            "100m_wind": pd.concat([np.sqrt(f["u100"] ** 2 + f["v100"] ** 2) for f in frames], axis=1),
        }
        for k, values in fields.items():
            values = values.sort_index()
            for (name, column), s in zip(FILES[k], (values.mean(axis=1), values.std(axis=1, ddof=0))):
                pd.DataFrame({"datetime": s.index.strftime("%Y-%m-%d %H:%M:%S"), column: s.values}).to_csv(
                    OUT / f"{country}_{name}_2019_2025.csv", index=False
                )
        print(f"wrote {country} from {len(frames)} grid points", flush=True)


if __name__ == "__main__":
    load_env()
    if not os.environ.get("CDSAPI_KEY"):
        sys.exit(
            "missing CDSAPI_KEY: add CDSAPI_URL and CDSAPI_KEY to .env (see README)"
        )
    args = sys.argv[1:]
    # grid points are the default: ~35 s per point, while gridded requests waited up to ~10 h in the CDS queue
    points_mode = "--grid" not in args
    args = [a for a in args if a not in ("--points", "--grid")]
    if args:
        countries = args[0].split(",")
        unknown = set(countries) - set(COUNTRIES)
        if unknown:
            sys.exit(f"unknown countries {sorted(unknown)}; known: {sorted(COUNTRIES)}")
    else:
        countries = [c for c in COUNTRIES if not all(p.exists() for p in csv_paths(c))]
    if not countries:
        print(f"{WEATHER_DIR} is complete; pass country names to rebuild some")
        sys.exit()
    # CDS queues at most 3 requests per user (across datasets); the download functions retry while it is full
    if points_mode:
        points = sorted({pt for c in countries for pt in grid_points(COUNTRIES[c])})
        print(f"countries {countries} -> {len(points)} grid points", flush=True)
        with ThreadPoolExecutor(3) as pool:
            for i, _ in enumerate(pool.map(download_point, points), 1):
                if i % 25 == 0 or i == len(points):
                    print(f"downloaded {i}/{len(points)} points", flush=True)
        process_points(countries)
        sys.exit()
    regions = sorted({REGION_OF[c] for c in countries})
    print(f"countries {countries} -> regions {regions}")
    with ThreadPoolExecutor(3) as pool:
        list(
            pool.map(
                download, [(r, y, p) for r in regions for y in YEARS for p in PARTS]
            )
        )
    process(countries)
