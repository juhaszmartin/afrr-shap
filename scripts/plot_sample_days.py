import os
import glob
import re
import pickle
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import torch

from afrr_shap.config import TEST_START
from afrr_shap.modeling.network import MultiOutputNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# High-quality visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True
})

def plot_sample_days_for_country(country_code, samples_dir):
    x_path = Path(f"processed_data/X_{country_code}.parquet")
    y_path = Path(f"processed_data/y_{country_code}.parquet")
    model_path = Path(f"models/nn_100seeds_best_{country_code}.pt")
    scaler_x_path = Path(f"models/scaler_X_100seeds_best_{country_code}.pkl")
    scaler_y_path = Path(f"models/scaler_y_100seeds_best_{country_code}.pkl")

    if not all(p.exists() for p in [x_path, y_path, model_path, scaler_x_path, scaler_y_path]):
        print(f"Skipping {country_code}: missing data or model files.")
        return

    print(f"\nGenerating sample inference plots for {country_code}...")

    # Load data
    X_full = pl.read_parquet(x_path)
    y_full = pl.read_parquet(y_path)

    timestamps = X_full["timestamp"].to_list()
    X_num = X_full.select(cs.numeric()).to_pandas()
    y_num = y_full.select(cs.numeric()).to_pandas()

    target_names = list(y_num.columns)
    is_test = (X_full["timestamp"] >= TEST_START).to_numpy()
    if not is_test.any():
        print(f"Skipping {country_code}: no test-year data.")
        return

    X_test = X_num[is_test]
    y_test = y_num[is_test]
    ts_test = [t for t, m in zip(timestamps, is_test) if m]

    # Load scalers and model
    with open(scaler_x_path, "rb") as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)

    input_dim = X_test.shape[1]
    output_dim = y_test.shape[1]

    model = MultiOutputNN(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Calculate inference on test split
    X_test_scaled = scaler_X.transform(X_test.to_numpy())
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    pred_df = pd.DataFrame(y_pred, columns=target_names, index=y_test.index)

    # Determine last 3 days of the test split
    ts_test_series = pd.Series(pd.to_datetime(ts_test))
    max_time = ts_test_series.max()
    min_3days = max_time - pd.Timedelta(days=3)

    mask_3days = (ts_test_series >= min_3days).to_numpy()
    if mask_3days.sum() < 10:  # Fallback if timestamps are sparse
        mask_3days = np.zeros(len(ts_test), dtype=bool)
        mask_3days[-72:] = True

    plot_ts = ts_test_series[mask_3days].dt.tz_localize(None)
    plot_y_true = y_test.iloc[mask_3days]
    plot_y_pred = pred_df.iloc[mask_3days]

    # Extract distinct timelags from target names
    # e.g. '_t+15min', '_t+30min', '_t+1h', '_t+2h'
    timelags = []
    for col in target_names:
        m = re.search(r"(_t\+[0-9]+(?:min|h))", col)
        if m and m.group(1) not in timelags:
            timelags.append(m.group(1))

    # For each forecasted timelag, plot Price and Sysdir (Volume)
    for lag in timelags:
        lag_clean = lag.lstrip("_")

        # 1. PRICE PLOT (Long and Short)
        price_cols = [c for c in target_names if "imbalance_prices" in c and lag in c]
        if price_cols:
            fig, ax = plt.subplots(figsize=(14, 6))

            for col in price_cols:
                is_short = "Short" in col
                color_fact = "#1f77b4" if not is_short else "#ff7f0e"
                color_pred = "#2ca02c" if not is_short else "#d62728"
                label_suffix = "Short" if is_short else "Long"

                ax.plot(
                    plot_ts,
                    plot_y_true[col],
                    label=f"Fact ({label_suffix})",
                    color=color_fact,
                    linewidth=2.0,
                    alpha=0.85
                )
                ax.plot(
                    plot_ts,
                    plot_y_pred[col],
                    label=f"Model Est ({label_suffix})",
                    color=color_pred,
                    linewidth=2.0,
                    linestyle="--",
                    alpha=0.9
                )

            # Metric on this 3-day sample
            col0 = price_cols[0]
            mae_sample = np.mean(np.abs(plot_y_true[col0] - plot_y_pred[col0]))

            ax.set_title(
                f"[{country_code}] Imbalance Price Forecast ({lag_clean}) — Last 3 Days of Test Split\n"
                f"Sample MAE ({col0}): {mae_sample:.2f}",
                fontweight="bold"
            )
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Price")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
            ax.legend(loc="upper right", frameon=True)
            plt.tight_layout()

            save_file = samples_dir / f"{country_code}_sample_3days_price_{lag_clean}.png"
            plt.savefig(save_file, dpi=200)
            plt.close(fig)

        # 2. SYSDIR / VOLUME PLOT
        vol_cols = [c for c in target_names if "imbalance_volumes" in c and lag in c]
        if vol_cols:
            vol_col = vol_cols[0]
            fig, ax = plt.subplots(figsize=(14, 6))

            ax.plot(
                plot_ts,
                plot_y_true[vol_col],
                label="Fact Sysdir / Volume",
                color="#004c6d",
                linewidth=2.2,
                alpha=0.9
            )
            ax.plot(
                plot_ts,
                plot_y_pred[vol_col],
                label="Model Est Sysdir / Volume",
                color="#e34a33",
                linewidth=2.0,
                linestyle="--",
                alpha=0.9
            )

            # Draw zero reference line for system direction balance
            ax.axhline(0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, label="System Balanced (0 MW)")

            mae_vol = np.mean(np.abs(plot_y_true[vol_col] - plot_y_pred[vol_col]))
            ax.set_title(
                f"[{country_code}] System Direction / Imbalance Volume Forecast ({lag_clean}) — Last 3 Days\n"
                f"Sample MAE: {mae_vol:.2f} MW",
                fontweight="bold"
            )
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Volume (MW)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
            ax.legend(loc="upper right", frameon=True)
            plt.tight_layout()

            save_file = samples_dir / f"{country_code}_sample_3days_sysdir_{lag_clean}.png"
            plt.savefig(save_file, dpi=200)
            plt.close(fig)

    print(f"Finished sample plots for {country_code}.")

def main():
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)

    x_files = sorted(glob.glob("processed_data/X_*.parquet"))
    countries = [Path(f).stem.split("_")[1] for f in x_files]

    print(f"Generating inference sample plots for {len(countries)} countries in 'samples/'...")
    for cc in countries:
        try:
            plot_sample_days_for_country(cc, samples_dir)
        except Exception as e:
            print(f"Error plotting {cc}: {e}")

    print("\nAll sample plots generated successfully in 'samples/'!")

if __name__ == "__main__":
    main()
