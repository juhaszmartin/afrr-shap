"""Downloads daily ECB euro reference rates (units of currency per 1 EUR)
for the currencies some TSOs report ENTSO-E prices in."""
import io
import urllib.request

import polars as pl

from afrr_shap.paths import FX_RATES_PATH

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

    FX_RATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).sort("currency", "date").write_csv(FX_RATES_PATH)
