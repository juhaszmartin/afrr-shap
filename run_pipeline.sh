#!/bin/bash
set -e

echo "Starting 100-seed Neural Network training for all countries..."
venv/bin/python scripts/train_all_100_seeds.py
echo "Training finished successfully!"

echo "Starting SHAP analysis on the best models..."
venv/bin/python scripts/run_shap_all.py
echo "SHAP analysis finished successfully!"
echo "PIPELINE COMPLETE."
