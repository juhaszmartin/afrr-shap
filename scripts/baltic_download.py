"""Downloads balancing data from the Baltic Transparency Dashboard (Elering, AST, Litgrid).

    python scripts/baltic_download.py            # every report in REPORTS, 2019 to END
    python scripts/baltic_download.py imbalance_prices

ENTSO-E has no EE imbalance volumes from 2024-10-10 to 2025-05-27 and no EE balancing prices after
2024; the dashboard has both. Public API, no key (CC BY 4.0), spec at
https://api-baltic.transparency-dashboard.eu/api/v1/openapi/btd_api. One request per report and month;
months already on disk are skipped, except the last two, which may still be revised.

Writes data/raw/baltic/{report}/year=YYYY/data.parquet in long format:
timestamp (UTC, period start), area, label (e.g. "Final", "Upward"), value.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime

import pandas as pd
import polars as pl

from afrr_shap.paths import BALTIC_DIR

URL = "https://api-baltic.transparency-dashboard.eu/api/v1/export"
REPORTS = [
    "imbalance_volumes",  # legacy: until the 2025 balancing reform
    "imbalance_volumes_v2",
    "imbalance_prices",
    "local_marginal_price_afrr",
    "local_marginal_price_mfrr",
    "balancing_energy_prices",
]
START, END = datetime(2019, 1, 1, tzinfo=UTC), datetime(2026, 11, 10, tzinfo=UTC)


def fetch(report, start, end):
    query = urllib.parse.urlencode({
        "id": report,
        "start_date": start.strftime("%Y-%m-%dT%H:%M"),
        "end_date": end.strftime("%Y-%m-%dT%H:%M"),
        "output_time_zone": "UTC",
        "output_format": "json",
        "json_header_groups": 1,
    })
    request = urllib.request.Request(f"{URL}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as r:
                body = json.load(r)
            break
        except OSError as e:
            if attempt == 4:
                raise
            print(f"  {report} {start:%Y-%m}: {e}, retrying")
            time.sleep(10 * (attempt + 1))
    if body.get("error"):
        raise RuntimeError(f"{report} {start:%Y-%m}: {body.get('message')}")
    data = body["data"]
    columns = [(c["group_level_0"], c.get("label") or "value") for c in data["columns"]]
    rows = [
        (pd.Timestamp(t["from"]), area, label, v)
        for t in data["timeseries"]
        for (area, label), v in zip(columns, t["values"])
        if v is not None
    ]
    return pl.DataFrame(
        rows, schema={"timestamp": pl.Datetime("us", "UTC"), "area": pl.Utf8, "label": pl.Utf8, "value": pl.Float64},
        orient="row",
    )


def download(report):
    months = pd.date_range(START, END, freq="MS", tz="UTC")
    recent = set(months[-2:])
    for year in sorted({m.year for m in months}):
        path = BALTIC_DIR / report / f"year={year}" / "data.parquet"
        year_months = [m for m in months if m.year == year]
        if path.exists() and not recent.intersection(year_months):
            continue
        frames = []
        for m in year_months:
            frames.append(fetch(report, m, min(m + pd.offsets.MonthBegin(1), pd.Timestamp(END))))
        df = pl.concat(frames).unique(["timestamp", "area", "label"], keep="last").sort("timestamp", "area", "label")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        print(f"{report} {year}: {df.height} values, areas {sorted(df['area'].unique())}")


if __name__ == "__main__":
    for report in sys.argv[1].split(",") if len(sys.argv) > 1 else REPORTS:
        download(report)
