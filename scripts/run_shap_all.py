"""SHAP for the network of every country (DeepExplainer on the saved seeds, averaged over seeds).

The models and scalers come from models/{CC}/ of the latest training run; the processed data must have
the feature layout recorded in models/{CC}/manifest.json, otherwise the country is refused.
Usage: python scripts/run_shap_all.py [CC,CC,...]
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

from afrr_shap.config import DEFAULT_CONFIG, TIME_FEATURES
from afrr_shap.modeling.artifacts import load_country, load_manifest, model_dir
from afrr_shap.modeling.network import MultiOutputNN
from afrr_shap.paths import FIGURES_DIR, PROCESSED_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHAP_CONFIG = DEFAULT_CONFIG["shap"]
N_BACKGROUND = SHAP_CONFIG["background_samples"]
N_EXPLAIN = SHAP_CONFIG["explain_samples"]
SHAP_SPLIT = SHAP_CONFIG.get("split", "test")
LAG_SUFFIX = re.compile(r"_t([+-]\d+)(min|h)$")
CYCLIC = re.compile(r"_(sin|cos)$")
CALENDAR_GROUPS = {CYCLIC.sub("", c) for c in TIME_FEATURES}


def feature_groups(feature_names):
    """Maps every column to its base feature, so that the value at t and all lags/leads of a series
    form one group, and the sin and cos of a calendar feature form one group. Lagged names have their
    spaces removed, so bases are compared without spaces.
    Returns {group name: (column indices, index of the column closest to t)}."""
    groups = {}
    for i, name in enumerate(feature_names):
        if name in TIME_FEATURES:  # a cyclic calendar feature is its sin/cos pair
            groups.setdefault(CYCLIC.sub("", name), []).append(
                (i, 0, CYCLIC.sub("", name))
            )
            continue
        m = LAG_SUFFIX.search(name)
        key = (LAG_SUFFIX.sub("", name) if m else name).replace(" ", "")
        offset = abs(int(m.group(1))) if m else 0
        groups.setdefault(key, []).append((i, offset, name))
    out = {}
    for key, cols in sorted(groups.items()):
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


def safe_name(target_name):
    return "".join(c for c in target_name if c.isalnum() or c == "_")


def clear_country_outputs(country_code):
    for pattern in (
        f"shap_raw_{country_code}_*.npy",
        f"shap_summary_{country_code}_*.png",
        f"shap_*_{country_code}.csv",
        f"shap_*_{country_code}.npy",
        f"shap_*_{country_code}.txt",
        f"shap_*_{country_code}.json",
    ):
        for old in FIGURES_DIR.glob(pattern):
            old.unlink()


def run_shap_for_country(country_code):
    if not (PROCESSED_DIR / f"X_{country_code}.parquet").exists():
        print(f"Skipping {country_code}: no processed data.")
        return
    data = load_country(country_code)
    manifest = load_manifest(
        country_code, data
    )  # refuses a feature layout that differs from training
    seeds = manifest.get("nn", {}).get("saved_seeds", [])
    model_paths = [model_dir(country_code) / f"nn_seed{s}.pt" for s in seeds]
    if not model_paths or not all(p.exists() for p in model_paths):
        print(f"Skipping {country_code}: no saved network seeds.")
        return

    split = SHAP_SPLIT if data.masks[SHAP_SPLIT].any() else "dev"
    print(
        f"--- SHAP for {country_code}: {len(seeds)} seeds, {N_EXPLAIN} {split} rows"
        + (f" (no {SHAP_SPLIT} rows)" if split != SHAP_SPLIT else "")
        + " ---"
    )
    with open(model_dir(country_code) / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    X_train = scaler_X.transform(data.X[data.masks["train"]])
    X_eval_raw = data.X[data.masks[split]]
    X_eval = scaler_X.transform(X_eval_raw)

    rng = np.random.default_rng(SHAP_CONFIG["sample_seed"])
    bg_idx = rng.choice(len(X_train), min(N_BACKGROUND, len(X_train)), replace=False)
    ex_idx = np.sort(
        rng.choice(len(X_eval), min(N_EXPLAIN, len(X_eval)), replace=False)
    )
    background = torch.tensor(X_train[bg_idx], dtype=torch.float32, device=device)
    samples = torch.tensor(X_eval[ex_idx], dtype=torch.float32, device=device)

    feature_names, target_names = data.feature_names, data.target_names
    per_seed, additivity = [], []
    for path in model_paths:
        model = MultiOutputNN(len(feature_names), len(target_names))
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device).eval()
        explainer = shap.DeepExplainer(model, background)
        sv = explainer.shap_values(samples, check_additivity=False)
        sv = (
            np.stack(sv, axis=-1) if isinstance(sv, list) else sv
        )  # (samples, features, targets)
        # Additivity: SHAP values plus the expected value must give the model output
        with torch.no_grad():
            out = model(samples).cpu().numpy()
        recon = sv.sum(axis=1) + np.asarray(explainer.expected_value).reshape(1, -1)
        additivity.append(float(np.abs(recon - out).max() / (np.abs(out).max() + 1e-9)))
        per_seed.append(sv.astype(np.float32))
        print(f"  {path.name} done, max relative additivity error {additivity[-1]:.1e}")
    per_seed = np.stack(per_seed)  # (seeds, samples, features, targets)

    # Sum each feature's columns (value at t + lags + leads) into one group
    groups = feature_groups(feature_names)
    group_names = list(groups)
    grouped = np.stack(
        [per_seed[:, :, idx, :].sum(axis=2) for idx, _ in groups.values()], axis=2
    )
    group_values = np.stack(
        [X_eval[ex_idx][:, nearest] for _, nearest in groups.values()], axis=1
    )
    # A sin/cos pair is coloured by its position in the cycle (0 = start of day/week/year, 1 = end)
    raw = X_eval_raw[ex_idx]
    for g, name in enumerate(group_names):
        if f"{name}_sin" in feature_names:
            sin, cos = (
                raw[:, feature_names.index(f"{name}_{f}")] for f in ("sin", "cos")
            )
            group_values[:, g] = np.mod(np.arctan2(sin, cos), 2 * np.pi) / (2 * np.pi)
    mean_grouped = grouped.mean(axis=0)  # (samples, groups, targets)

    FIGURES_DIR.mkdir(exist_ok=True)
    clear_country_outputs(country_code)
    weather = {
        c
        for c in group_names
        if any(k in c for k in ("temperature", "ssrd", "wind_speed"))
    }
    mean_abs = pd.DataFrame(
        {
            "Feature": group_names,
            "Category": [feature_category(g, weather) for g in group_names],
        }
    )
    stability = []
    for t, target_name in enumerate(target_names):
        mean_abs[target_name] = np.abs(mean_grouped[:, :, t]).mean(axis=0)
        # Agreement between seeds: Spearman correlation of the per-seed feature rankings
        imp = np.abs(grouped[:, :, :, t]).mean(axis=1)  # (seeds, groups)
        ranks = imp.argsort(axis=1).argsort(axis=1)
        corr = np.corrcoef(ranks)
        top10 = [set(np.argsort(-imp[s])[:10]) for s in range(len(imp))]
        stability.append(
            {
                "Target": target_name,
                "rank_corr_mean": corr[np.triu_indices(len(imp), 1)].mean(),
                "top10_overlap_mean": np.mean(
                    [
                        len(a & b) / 10
                        for i, a in enumerate(top10)
                        for b in top10[i + 1 :]
                    ]
                ),
            }
        )
        np.save(
            FIGURES_DIR / f"shap_raw_{country_code}_{safe_name(target_name)}.npy",
            mean_grouped[:, :, t],
        )

    np.save(FIGURES_DIR / f"shap_test_features_{country_code}.npy", group_values)
    (FIGURES_DIR / f"shap_feature_names_{country_code}.txt").write_text(
        "\n".join(group_names)
    )
    mean_abs.to_csv(FIGURES_DIR / f"shap_mean_abs_{country_code}.csv", index=False)
    mean_abs.drop(columns="Feature").groupby("Category").sum().to_csv(
        FIGURES_DIR / f"shap_category_{country_code}.csv"
    )
    pd.DataFrame(stability).to_csv(
        FIGURES_DIR / f"shap_stability_{country_code}.csv", index=False
    )
    (FIGURES_DIR / f"shap_info_{country_code}.json").write_text(
        json.dumps(
            {
                "split": split,
                "explained_rows": len(ex_idx),
                "background_rows": len(bg_idx),
                "seeds": seeds,
                "feature_layout": manifest["feature_layout"],
                "trained_utc": manifest["created_utc"],
                "max_relative_additivity_error": max(additivity),
            },
            indent=1,
        )
    )
    print(
        f"Saved SHAP for {country_code}: {len(feature_names)} columns in {len(group_names)} groups, "
        f"seed rank agreement {np.mean([s['rank_corr_mean'] for s in stability]):.2f}"
    )


if __name__ == "__main__":
    countries = sorted(
        Path(f).stem.split("_")[1] for f in PROCESSED_DIR.glob("X_*.parquet")
    )
    if len(sys.argv) > 1:  # optional subset, e.g. DE,FR
        countries = sys.argv[1].split(",")
    print(f"Running SHAP for {len(countries)} countries: {countries}")
    for cc in countries:
        run_shap_for_country(cc)
