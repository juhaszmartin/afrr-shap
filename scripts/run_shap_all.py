import sys
import glob
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import shap
import torch
import torch.nn as nn

from preprocess_all import TEST_START, TIME_FEATURES
from train_all_100_seeds import SHAP_SEEDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_BACKGROUND = 500  # training rows used as the SHAP reference distribution
N_EXPLAIN = 1000    # 2025 rows explained, the same rows for every seed
LAG_SUFFIX = re.compile(r"_t([+-]\d+)(min|h)$")
CYCLIC = re.compile(r"_(sin|cos)$")
CALENDAR_GROUPS = {CYCLIC.sub("", c) for c in TIME_FEATURES}


class MultiOutputNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def feature_groups(feature_names):
    """Maps every column to its base feature, so that the value at t and all lags/leads of a series
    form one group, and the sin and cos of a calendar feature form one group. Lagged names have their spaces removed, so bases are compared without spaces.
    Returns {group name: (column indices, index of the column closest to t)}."""
    groups = {}
    for i, name in enumerate(feature_names):
        if name in TIME_FEATURES:  # a cyclic calendar feature is its sin/cos pair
            groups.setdefault(CYCLIC.sub("", name), []).append((i, 0, CYCLIC.sub("", name)))
            continue
        m = LAG_SUFFIX.search(name)
        key = (LAG_SUFFIX.sub("", name) if m else name).replace(" ", "")
        offset = abs(int(m.group(1))) if m else 0
        groups.setdefault(key, []).append((i, offset, name))
    out = {}
    for key, cols in groups.items():
        # Display name: the original (spaced) name of the value at t when there is one
        at_t = [n for _, o, n in cols if o == 0 and not LAG_SUFFIX.search(n)]
        display = at_t[0] if at_t else LAG_SUFFIX.sub("", cols[0][2])
        nearest = min(cols, key=lambda c: c[1])[0]
        out[display] = ([i for i, _, _ in cols], nearest)
    return out


def feature_category(group_name, weather_names):
    if group_name in CALENDAR_GROUPS:
        return "calendar"
    if group_name in weather_names:
        return "weather"
    if group_name.startswith("imbalance_"):
        return "target history"
    return re.split(r" - |-", group_name)[0]


def run_shap_for_country(country_code):
    print(f"--- Running SHAP ({len(SHAP_SEEDS)} seeds, {N_EXPLAIN} samples) for {country_code} ---")
    x_path = Path("processed_data") / f"X_{country_code}.parquet"
    y_path = Path("processed_data") / f"y_{country_code}.parquet"
    model_paths = [Path("models") / f"nn_seed{s}_{country_code}.pt" for s in SHAP_SEEDS]
    scaler_X_path = Path("models") / f"scaler_X_100seeds_best_{country_code}.pkl"

    if not all(p.exists() for p in [x_path, y_path, scaler_X_path, *model_paths]):
        print(f"Missing artifacts for {country_code}. Skipping.")
        return

    X_pl = pl.read_parquet(x_path)
    is_test = (X_pl["timestamp"] >= TEST_START).to_numpy()
    X_df = X_pl.select(cs.numeric()).to_pandas()
    y_df = pl.read_parquet(y_path).select(cs.numeric()).to_pandas()

    with open(scaler_X_path, "rb") as f:
        scaler_X = pickle.load(f)

    X_train_scaled = scaler_X.transform(X_df[~is_test])
    X_test_scaled = scaler_X.transform(X_df[is_test])

    rng = np.random.default_rng(42)
    bg_indices = rng.choice(len(X_train_scaled), min(N_BACKGROUND, len(X_train_scaled)), replace=False)
    test_indices = np.sort(rng.choice(len(X_test_scaled), min(N_EXPLAIN, len(X_test_scaled)), replace=False))
    background = torch.tensor(X_train_scaled[bg_indices], dtype=torch.float32, device=device)
    test_samples = torch.tensor(X_test_scaled[test_indices], dtype=torch.float32, device=device)

    target_names = list(y_df.columns)
    feature_names = list(X_df.columns)
    n_targets = len(target_names)

    # SHAP values are linear in the model, so the mean over seeds explains the seed ensemble
    per_seed = []
    for path in model_paths:
        model = MultiOutputNN(len(feature_names), n_targets)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device).eval()
        sv = shap.DeepExplainer(model, background).shap_values(test_samples, check_additivity=False)
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv  # (samples, features, targets)
        per_seed.append(sv.astype(np.float32))
        print(f"  {path.name} done")
    per_seed = np.stack(per_seed)  # (seeds, samples, features, targets)

    # Sum each feature's columns (value at t + lags + leads) into one group
    groups = feature_groups(feature_names)
    group_names = list(groups)
    grouped = np.stack([per_seed[:, :, idx, :].sum(axis=2) for idx, _ in groups.values()], axis=2)
    group_values = np.stack([X_test_scaled[test_indices][:, nearest] for _, nearest in groups.values()], axis=1)
    # A sin/cos pair is coloured by its position in the cycle (0 = start of day/week/year, 1 = end)
    X_test_raw = X_df[is_test].to_numpy()[test_indices]
    for g, name in enumerate(group_names):
        if f"{name}_sin" in feature_names:
            sin, cos = (X_test_raw[:, feature_names.index(f"{name}_{f}")] for f in ("sin", "cos"))
            group_values[:, g] = np.mod(np.arctan2(sin, cos), 2 * np.pi) / (2 * np.pi)
    mean_grouped = grouped.mean(axis=0)  # (samples, groups, targets)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    for old in glob.glob(str(out_dir / f"shap_raw_{country_code}_*.npy")):
        Path(old).unlink()

    weather = {c for c in group_names if any(k in c for k in ("temperature", "ssrd", "wind_speed"))}
    mean_abs = pd.DataFrame({"Feature": group_names, "Category": [feature_category(g, weather) for g in group_names]})
    stability = []
    for t, target_name in enumerate(target_names):
        mean_abs[target_name] = np.abs(mean_grouped[:, :, t]).mean(axis=0)
        # Agreement between seeds: Spearman correlation of the per-seed feature rankings
        imp = np.abs(grouped[:, :, :, t]).mean(axis=1)  # (seeds, groups)
        ranks = imp.argsort(axis=1).argsort(axis=1)
        corr = np.corrcoef(ranks)
        top10 = [set(np.argsort(-imp[s])[:10]) for s in range(len(imp))]
        stability.append({
            "Target": target_name,
            "rank_corr_mean": corr[np.triu_indices(len(imp), 1)].mean(),
            "top10_overlap_mean": np.mean([len(a & b) / 10 for i, a in enumerate(top10) for b in top10[i + 1:]]),
        })
        safe_target_name = "".join(c for c in target_name if c.isalnum() or c in (' ', '_')).rstrip().replace(" ", "_")
        np.save(out_dir / f"shap_raw_{country_code}_{safe_target_name}.npy", mean_grouped[:, :, t])

    np.save(out_dir / f"shap_test_features_{country_code}.npy", group_values)
    with open(out_dir / f"shap_feature_names_{country_code}.txt", "w") as f:
        f.write("\n".join(group_names))
    mean_abs.to_csv(out_dir / f"shap_mean_abs_{country_code}.csv", index=False)
    mean_abs.drop(columns="Feature").groupby("Category").sum().to_csv(out_dir / f"shap_category_{country_code}.csv")
    pd.DataFrame(stability).to_csv(out_dir / f"shap_stability_{country_code}.csv", index=False)
    print(f"Saved SHAP for {country_code}: {len(feature_names)} columns in {len(group_names)} groups, "
          f"seed rank agreement {np.mean([s['rank_corr_mean'] for s in stability]):.2f}")


if __name__ == "__main__":
    x_files = sorted(glob.glob("processed_data/X_*.parquet"))
    countries = [Path(f).stem.split("_")[1] for f in x_files]
    if len(sys.argv) > 1:  # optional subset, e.g. DE,FR
        countries = sys.argv[1].split(",")

    print(f"Running SHAP for {len(countries)} countries: {countries}")

    for cc in countries:
        run_shap_for_country(cc)
