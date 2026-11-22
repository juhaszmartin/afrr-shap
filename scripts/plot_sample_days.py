"""3-day plots of actuals against every model's forecast, from results/predictions/{CC}.parquet
(written by train_models.py), for the last three days of the dev and the test split.

Usage: python scripts/plot_sample_days.py [--split dev|test|all] [CC,CC,...]
"""

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

from afrr_shap.modeling.metrics import base_series, horizon
from afrr_shap.paths import RESULTS_DIR, SAMPLES_DIR

plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "legend.fontsize": 10})

MODEL_STYLE = {
    "naive": {"color": "#9e9e9e", "linestyle": ":", "linewidth": 1.4},
    "ridge": {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.6},
    "lightgbm": {"color": "#2ca02c", "linestyle": "--", "linewidth": 1.6},
    "nn_ensemble": {"color": "#d62728", "linestyle": "-", "linewidth": 1.8},
}
SERIES_LABEL = {
    "imbalance_price_long": "long price",
    "imbalance_price_short": "short price",
    "imbalance_volume": "volume",
}


def plot_country(country_code, split):
    path = RESULTS_DIR / "predictions" / f"{country_code}.parquet"
    if not path.exists():
        print(f"Skipping {country_code}: {path} not found.")
        return
    df = pl.read_parquet(path).filter(pl.col("split") == split)
    if df.height == 0:
        print(f"Skipping {country_code}: no {split} rows.")
        return
    last = df["timestamp"].max()
    df = df.filter(pl.col("timestamp") > last - pl.duration(days=3)).sort("timestamp")
    ts = df["timestamp"].dt.replace_time_zone(None).to_numpy()

    targets = [c for c in df.columns if c not in ("timestamp", "split") and "|" not in c]
    models = [m for m in MODEL_STYLE if f"{targets[0]}|{m}" in df.columns]
    out_dir = SAMPLES_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{country_code}_*.png"):
        old.unlink()

    for target in targets:
        series = base_series(target)
        # Single-priced short prices repeat the long price; skip them
        if series == "imbalance_price_short" and (
            df[target] - df[target.replace("_short", "_long")]
        ).abs().max() < 1e-6:
            continue
        fig, ax = plt.subplots(figsize=(14, 5.5))
        ax.plot(ts, df[target].to_numpy(), color="black", linewidth=2.0, label="actual")
        for m in models:
            pred = df[f"{target}|{m}"].to_numpy()
            mae = abs(pred - df[target].to_numpy()).mean()
            ax.plot(ts, pred, label=f"{m} (MAE {mae:.1f})", **MODEL_STYLE[m])
        if series == "imbalance_volume":
            ax.axhline(0, color="gray", linewidth=0.8)
        unit = "MWh" if series == "imbalance_volume" else "EUR/MWh"
        ax.set_title(
            f"{country_code} {SERIES_LABEL[series]}, {horizon(target)} ahead: last 3 days of the {split} split"
        )
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(f"{SERIES_LABEL[series]} [{unit}]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{country_code}_{series}_{horizon(target)}.png", dpi=150)
        plt.close(fig)
    print(f"Plotted {country_code} ({split}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("countries", nargs="?", default=None)
    args = parser.parse_args()
    files = sorted((RESULTS_DIR / "predictions").glob("*.parquet"))
    countries = args.countries.split(",") if args.countries else [f.stem for f in files]
    for split in ["dev", "test"] if args.split == "all" else [args.split]:
        for cc in countries:
            plot_country(cc, split)


if __name__ == "__main__":
    main()
