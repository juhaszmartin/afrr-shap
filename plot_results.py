"""Charts comparing the models per country and horizon, from results/{dev,test}/metrics_*.csv.

One chart per split, target group and metric in figures/performance/{split}/: countries on the x axis,
one bar per model, one panel per horizon. A second set shows the MAE skill against the naive forecast
(1 - MAE / MAE naive; above 0 = better than naive).

Usage: python plot_results.py [--split dev|test|all]
"""

import argparse

import altair as alt
import pandas as pd

from afrr_shap.modeling.metrics import GROUP_SINGLE
from afrr_shap.paths import FIGURES_DIR, RESULTS_DIR

MODEL_ORDER = ["naive", "ridge", "lightgbm", "nn", "nn_ensemble"]
MODEL_COLORS = ["#9e9e9e", "#1f77b4", "#2ca02c", "#ff9896", "#d62728"]


def horizon_key(h):
    """15-min-country horizons first, then hourly ones, each by minutes ahead."""
    return (h.endswith("h"), int(h[2:-3]) if h.endswith("min") else 60 * int(h[2:-1]))


def chart(df, value, title):
    """One row of panels per horizon (stacked charts rather than a row facet, whose custom sort
    misplaced panels), one panel per country, one bar per model."""
    models = [m for m in MODEL_ORDER if m in set(df["Model"])]
    colors = [MODEL_COLORS[MODEL_ORDER.index(m)] for m in models]
    y_scale = alt.Scale(domain=[float(df[value].min()), float(df[value].max())], nice=True)
    rows = []
    for h in sorted(df["Horizon"].unique(), key=horizon_key):
        rows.append(
            alt.Chart(df[df["Horizon"] == h], title=f"{h} ahead")
            .mark_bar()
            .encode(
                x=alt.X("Model:N", sort=models, axis=None),
                y=alt.Y(f"{value}:Q", title=value, scale=y_scale),
                color=alt.Color(
                    "Model:N", sort=models, scale=alt.Scale(domain=models, range=colors)
                ).legend(orient="bottom", direction="horizontal"),
                column=alt.Column("Country:N", title=None, header=alt.Header(labelFontSize=12)),
            )
            .properties(width=55, height=110)
        )
    return alt.vconcat(*rows, title=title)


def process_split(split):
    files = sorted((RESULTS_DIR / split).glob("metrics_*.csv"))
    if not files:
        print(f"No metrics in {RESULTS_DIR / split}, skipping {split}.")
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[df["Group"] != GROUP_SINGLE]
    naive = df[df["Model"] == "naive"][["Country", "Target", "MAE"]]
    df = df.merge(naive, on=["Country", "Target"], suffixes=("", "_naive"))
    df["MAE skill vs naive"] = 1 - df["MAE"] / df["MAE_naive"]
    out_dir = FIGURES_DIR / "performance" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    for group, g in df.groupby("Group"):
        name = group.split(" [")[0].replace(" ", "_")
        for value in ["MAE", "RMSE", "R2", "MAE skill vs naive"]:
            data = g if value != "MAE skill vs naive" else g[g["Model"] != "naive"]
            chart(data, value, f"{group}, {split}").save(
                out_dir / f"{value.replace(' ', '_')}_{name}.png", ppi=150
            )
    print(f"Charts for {split} in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    args = parser.parse_args()
    for split in ["dev", "test"] if args.split == "all" else [args.split]:
        process_split(split)


if __name__ == "__main__":
    main()
