"""Downloads daily ECB euro reference rates (units of currency per 1 EUR)
for the currencies some TSOs report ENTSO-E prices in."""
import io
import urllib.request
from pathlib import Path

import polars as pl

CURRENCIES = ["HUF", "PLN", "RON", "HRK"]
START, END = "2018-12-01", "2025-12-31"
URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.{cur}.EUR.SP00.A?format=csvdata&startPeriod={start}&endPeriod={end}"

if __name__ == "__main__":
    frames = []
    for cur in CURRENCIES:
        with urllib.request.urlopen(URL.format(cur=cur, start=START, end=END), timeout=60) as r:
            df = pl.read_csv(io.BytesIO(r.read()), columns=["TIME_PERIOD", "OBS_VALUE"])
        frames.append(df.select(
            pl.col("TIME_PERIOD").str.to_date().alias("date"),
            pl.lit(cur).alias("currency"),
            pl.col("OBS_VALUE").cast(pl.Float64).alias("per_eur"),
        ))
        print(f"{cur}: {df.height} daily rates")

    out = Path("fx_data")
    out.mkdir(exist_ok=True)
    pl.concat(frames).sort("currency", "date").write_csv(out / "ecb_eur_rates.csv")
