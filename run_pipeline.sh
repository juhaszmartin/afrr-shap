#!/bin/bash
# Full pipeline from raw data to plots, run from the repository root. Data downloads
# (get_fx_rates.py, fill_entsoe_gaps.py, era5_download.py) are separate one-off steps, see PIPELINE.md.
set -e
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export AFRR_CONFIG="${AFRR_CONFIG:-configs/default.yaml}"

echo "Running pipeline..."
.venv/bin/python -m afrr_shap.cli pipeline
echo "PIPELINE COMPLETE."
