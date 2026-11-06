# Legacy code

Kept for reference only. None of these files is part of the pipeline, and they may not run against the current data layout.

| File | What it was | Replaced by |
|---|---|---|
| `train_nn.py` | single-seed training, 80/20 row split | `scripts/train_all_100_seeds.py` |
| `train_hu_100_seeds.py` | 100-seed training for HU only, 80/20 row split | `scripts/train_all_100_seeds.py` |
| `run_shap.py` | SHAP on one model, 80/20 row split | `scripts/run_shap_all.py` |
| `entsoe_data_getter.py` | interactive explorer of imbalance data per country (reads a `country_data.csv` that is not in the repo) | — |
| `entsoe.ipynb` | exploration of ENTSO-E endpoints and availability | — |
| `era5_eu.ipynb` | first ERA5 download for 17 countries, one request per country and month | `scripts/era5_download.py` (same boxes; the notebook's column names differ from the stored CSVs) |
