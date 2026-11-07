#!/bin/bash
# Full pipeline from raw data to plots, run from the repository root. Data downloads
# (get_fx_rates.py, fill_entsoe_gaps.py, era5_download.py) are separate one-off steps, see PIPELINE.md.
set -e
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"

echo "Preprocessing..."
venv/bin/python scripts/preprocess_all.py
venv/bin/python scripts/get_entsoe_info.py

echo "Training 100 seeds per country..."
venv/bin/python scripts/train_all_100_seeds.py

echo "SHAP (5 seeds, 1000 test samples)..."
venv/bin/python scripts/run_shap_all.py

echo "Plots..."
venv/bin/python scripts/plot_shap_results.py
venv/bin/python scripts/plot_sample_days.py
venv/bin/python plot_results.py
echo "PIPELINE COMPLETE."
