"""Trains and evaluates every model of every country.

Models (configs/default.yaml `models`):
- naive: the value at t, carried forward to every horizon;
- ridge: linear regression on the standardised inputs, one L2 penalty per target picked on dev;
- lightgbm: one gradient-boosted tree model per target, early stopping on dev;
- nn: the multi-output network, trained with `training.seeds` seeds, early stopping on dev. It is reported
  per seed (mean +- std over seeds, "nn") and as the average prediction of all seeds ("nn_ensemble").

Every model is fitted on train (before dev_start); dev (dev_start to test_start) is used for early
stopping and for picking hyperparameters; test (from test_start) is touched only for the final metrics.

Writes:
- results/{dev,test}/metrics_{CC}.csv: one row per model and target;
- results/{dev,test}/summary.txt: per-group averages over horizons for all countries;
- results/predictions/{CC}.parquet: actuals and every model's forecast on dev and test;
- models/{CC}/: network seeds kept for SHAP, scalers, ridge weights, LightGBM boosters and manifest.json.

Usage: python scripts/train_models.py [CC,CC,...]
"""

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # before CUDA starts, for deterministic cuBLAS

import copy
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from afrr_shap.config import DEFAULT_CONFIG, SHAP_SEEDS
from afrr_shap.modeling.artifacts import load_country, reset_model_dir, write_manifest
from afrr_shap.modeling.metrics import (
    GROUP_SINGLE,
    base_series,
    error_metrics,
    horizon,
    naive_columns,
    target_groups,
)
from afrr_shap.modeling.network import MultiOutputNN
from afrr_shap.paths import PROCESSED_DIR, RESULTS_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = list(dict.fromkeys(["naive", *DEFAULT_CONFIG.get("models", ["nn"])]))
NN_CONFIG = DEFAULT_CONFIG["training"]
RIDGE_CONFIG = DEFAULT_CONFIG.get("ridge", {"alphas": [1, 10, 100, 1000, 10000]})
LGB_CONFIG = DEFAULT_CONFIG.get("lightgbm", {})
EVAL_SPLITS = ("dev", "test")
METRICS = ("MAE", "RMSE", "R2", "SMAPE")

if NN_CONFIG.get("deterministic", False):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def to_original(scaler_y, pred):
    """Back to original units; empty arrays (a split without rows, e.g. BA test) pass through."""
    return scaler_y.inverse_transform(pred) if len(pred) else pred


def fit_ridge(Xs, ys, scaler_y, data):
    """Closed-form ridge on standardised data (no intercept needed: both are centred on train).
    One alpha per target, the one with the lowest dev MAE in original units."""
    tr, dv = data.masks["train"], data.masks["dev"]
    XtX = Xs[tr].T @ Xs[tr]
    Xty = Xs[tr].T @ ys[tr]
    eye = np.eye(XtX.shape[0])
    alphas = RIDGE_CONFIG["alphas"]
    weights, dev_mae = [], []
    for a in alphas:
        W = np.linalg.solve(XtX + a * eye, Xty)
        weights.append(W)
        pred = scaler_y.inverse_transform(Xs[dv] @ W)
        dev_mae.append(np.abs(pred - data.y[dv]).mean(axis=0))
    best = np.argmin(np.array(dev_mae), axis=0)  # per target
    W = np.stack([weights[b][:, i] for i, b in enumerate(best)], axis=1)
    chosen = [alphas[b] for b in best]
    edge = sorted({a for a in chosen if a in (alphas[0], alphas[-1])})
    print(
        f"  ridge: alpha per target {chosen}"
        + (f" (WARNING: at the edge of the grid: {edge})" if edge else "")
    )
    return W, chosen


def fit_lightgbm(data, ys, out_dir):
    """One booster per target on the standardised target (so the Huber delta of 1 matches the network)."""
    tr, dv = data.masks["train"], data.masks["dev"]
    params = dict(LGB_CONFIG.get("params", {}))
    rounds = LGB_CONFIG.get("num_boost_round", 2000)
    stop = LGB_CONFIG.get("early_stopping_rounds", 100)
    # The binned training set is built once and reused for every target by swapping the label
    train_set = lgb.Dataset(
        data.X[tr].astype(np.float32),
        label=ys[tr, 0],
        feature_name=[f"f{i}" for i in range(data.X.shape[1])],
        free_raw_data=False,
        params={"max_bin": params.get("max_bin", 255), "verbose": -1},
    ).construct()
    dev_set = lgb.Dataset(
        data.X[dv].astype(np.float32), label=ys[dv, 0], reference=train_set
    ).construct()
    (out_dir / "lightgbm").mkdir()
    boosters, iterations = [], []
    for i, target in enumerate(data.target_names):
        train_set.set_label(ys[tr, i])
        dev_set.set_label(ys[dv, i])
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=rounds,
            valid_sets=[dev_set],
            callbacks=[lgb.early_stopping(stop, verbose=False)],
        )
        booster.save_model(
            str(out_dir / "lightgbm" / f"{target}.txt"),
            num_iteration=booster.best_iteration,
        )
        boosters.append(booster)
        iterations.append(booster.best_iteration)
    print(f"  lightgbm: best iterations {iterations}")
    return boosters, iterations


def train_nn_seeds(data, Xs, ys, scaler_y, out_dir):
    """Trains the network once per seed; returns per-seed metrics and the seed-average predictions."""
    tr, dv = data.masks["train"], data.masks["dev"]
    as_tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    X_train, y_train = as_tensor(Xs[tr]), as_tensor(ys[tr])
    X_dev, y_dev = as_tensor(Xs[dv]), as_tensor(ys[dv])
    X_eval = {s: as_tensor(Xs[data.masks[s]]) for s in EVAL_SPLITS}

    n_seeds = NN_CONFIG["seeds"]
    batch_size = NN_CONFIG["batch_size"]
    n_samples = len(X_train)
    per_seed = {s: [] for s in EVAL_SPLITS}  # list of metric dicts
    pred_sum = {
        s: np.zeros((int(data.masks[s].sum()), ys.shape[1])) for s in EVAL_SPLITS
    }
    dev_losses, epochs_run = [], []
    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = MultiOutputNN(X_train.shape[1], y_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=NN_CONFIG["learning_rate"])
        criterion = nn.HuberLoss(delta=NN_CONFIG["huber_delta"])
        best_loss, best_state, bad_epochs, epoch = float("inf"), None, 0, 0
        for epoch in range(1, NN_CONFIG["epochs"] + 1):
            model.train()
            perm = torch.randperm(n_samples, device=device)
            for b in range(0, n_samples, batch_size):
                idx = perm[b : b + batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_train[idx]), y_train[idx])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                dev_loss = criterion(model(X_dev), y_dev).item()
            if dev_loss < best_loss:
                best_loss, bad_epochs = dev_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= NN_CONFIG["early_stopping_patience"]:
                    break
        model.load_state_dict(best_state)
        model.eval()
        dev_losses.append(best_loss)
        epochs_run.append(epoch)
        with torch.no_grad():
            for s in EVAL_SPLITS:
                if len(X_eval[s]) == 0:
                    continue
                pred = to_original(scaler_y, model(X_eval[s]).cpu().numpy())
                pred_sum[s] += pred
                per_seed[s].append(error_metrics(data.y[data.masks[s]], pred))
        if seed in SHAP_SEEDS:
            torch.save(best_state, out_dir / f"nn_seed{seed}.pt")
    print(
        f"  nn: {n_seeds} seeds, epochs run {min(epochs_run)}-{max(epochs_run)} "
        f"(median {int(np.median(epochs_run))}), best dev loss seed {int(np.argmin(dev_losses))}"
    )
    ensemble = {s: pred_sum[s] / n_seeds for s in EVAL_SPLITS}
    return per_seed, ensemble, {"dev_loss": dev_losses, "epochs": epochs_run}


def metric_rows(country, split, data, groups, model, metrics, std=None):
    rows = []
    for i, t in enumerate(data.target_names):
        row = {
            "Country": country,
            "Split": split,
            "Model": model,
            "Target": t,
            "Series": base_series(t),
            "Horizon": horizon(t),
            "Group": groups[t],
        }
        for k in METRICS:
            row[k] = metrics[k][i]
            row[f"{k}_STD"] = std[k][i] if std is not None else 0.0
        rows.append(row)
    return rows


def train_country(country_code):
    print("\n=============================================")
    print(f"{country_code} on {device}")
    if not (PROCESSED_DIR / f"X_{country_code}.parquet").exists():
        print(f"Skipping {country_code}: no processed data.")
        return
    t0 = time.time()
    data = load_country(country_code)
    n = {s: int(m.sum()) for s, m in data.masks.items()}
    print(f"  {data.X.shape[1]} features, {len(data.target_names)} targets, rows {n}")
    if n["train"] == 0 or n["dev"] == 0:
        print(f"Skipping {country_code}: needs train and dev rows.")
        return
    if n["test"] == 0:
        print(f"  WARNING {country_code}: no test rows, only dev metrics are written")

    out_dir = reset_model_dir(country_code)
    tr = data.masks["train"]
    scaler_X = StandardScaler().fit(data.X[tr])
    scaler_y = StandardScaler().fit(data.y[tr])
    Xs = scaler_X.transform(data.X)
    ys = scaler_y.transform(data.y)
    for name, obj in (("scaler_X", scaler_X), ("scaler_y", scaler_y)):
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    preds = {s: {} for s in EVAL_SPLITS}  # split -> model -> (samples, targets)
    seed_metrics = None
    extra = {"models": MODELS}

    naive_idx = naive_columns(data.target_names, data.feature_names)
    for s in EVAL_SPLITS:
        preds[s]["naive"] = data.X[data.masks[s]][:, naive_idx]

    if "ridge" in MODELS:
        t = time.time()
        W, alphas = fit_ridge(Xs, ys, scaler_y, data)
        np.save(out_dir / "ridge_weights.npy", W)
        extra["ridge_alphas"] = alphas
        for s in EVAL_SPLITS:
            preds[s]["ridge"] = to_original(scaler_y, Xs[data.masks[s]] @ W)
        print(f"  ridge done in {time.time() - t:.0f}s")

    if "lightgbm" in MODELS:
        t = time.time()
        boosters, iterations = fit_lightgbm(data, ys, out_dir)
        extra["lightgbm_iterations"] = iterations
        for s in EVAL_SPLITS:
            Xe = data.X[data.masks[s]].astype(np.float32)
            preds[s]["lightgbm"] = to_original(
                scaler_y,
                np.stack(
                    [b.predict(Xe, num_iteration=b.best_iteration) for b in boosters],
                    axis=1,
                )
                if len(Xe)
                else np.zeros((0, len(boosters))),
            )
        print(f"  lightgbm done in {time.time() - t:.0f}s")

    if "nn" in MODELS:
        t = time.time()
        seed_metrics, ensemble, nn_info = train_nn_seeds(
            data, Xs, ys, scaler_y, out_dir
        )
        extra["nn"] = {
            "seeds": NN_CONFIG["seeds"],
            "saved_seeds": [s for s in SHAP_SEEDS if s < NN_CONFIG["seeds"]],
            **nn_info,
        }
        for s in EVAL_SPLITS:
            preds[s]["nn_ensemble"] = ensemble[s]
        print(f"  nn done in {time.time() - t:.0f}s")

    write_manifest(country_code, data, extra)

    # Metrics per split and model
    for s in EVAL_SPLITS:
        out = RESULTS_DIR / s / f"metrics_{country_code}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if n[s] == 0:
            out.unlink(missing_ok=True)
            continue
        y_true = data.y[data.masks[s]]
        groups = target_groups(data.target_names, y_true)
        rows = []
        if seed_metrics is not None and seed_metrics[s]:
            stacked = {k: np.stack([m[k] for m in seed_metrics[s]]) for k in METRICS}
            rows += metric_rows(
                country_code,
                s,
                data,
                groups,
                "nn",
                {k: v.mean(axis=0) for k, v in stacked.items()},
                {k: v.std(axis=0) for k, v in stacked.items()},
            )
        for model, pred in preds[s].items():
            rows += metric_rows(
                country_code, s, data, groups, model, error_metrics(y_true, pred)
            )
        pd.DataFrame(rows).to_csv(out, index=False)

    # Actuals and forecasts, for the sample plots and any later analysis
    frames = []
    for s in EVAL_SPLITS:
        m = data.masks[s]
        if not m.any():
            continue
        cols = {"timestamp": data.timestamps[m], "split": [s] * int(m.sum())}
        for i, t in enumerate(data.target_names):
            cols[t] = data.y[m][:, i]
            for model, pred in preds[s].items():
                cols[f"{t}|{model}"] = pred[:, i]
        frames.append(pl.DataFrame(cols))
    pred_dir = RESULTS_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC")
    ).write_parquet(pred_dir / f"{country_code}.parquet")
    print(f"{country_code} finished in {time.time() - t0:.0f}s")


def write_summary(split):
    """Averages over horizons per target group for every country, from the metrics files on disk."""
    files = sorted((RESULTS_DIR / split).glob("metrics_*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[df["Group"] != GROUP_SINGLE]
    mean = (
        df.groupby(["Country", "Group", "Model"], sort=False)[list(METRICS)]
        .mean()
        .reset_index()
    )
    naive = mean.loc[mean["Model"] == "naive", ["Country", "Group", "MAE"]]
    mean = mean.merge(naive, on=["Country", "Group"], suffixes=("", "_naive"))
    mean["MAE skill vs naive"] = 1 - mean["MAE"] / mean.pop("MAE_naive")
    lines = [
        f"Averages over horizons per target group, {split} split. "
        "MAE skill vs naive = 1 - MAE / MAE(naive). Single-priced short prices are left out.\n\n"
    ]
    order = ["naive", "ridge", "lightgbm", "nn", "nn_ensemble"]
    for (cc, group), g in mean.groupby(["Country", "Group"], sort=False):
        lines.append(f"{cc} | {group}\n")
        lines.append(
            g.drop(columns=["Country", "Group"])
            .set_index("Model")
            .reindex([m for m in order if m in set(g["Model"])])
            .to_string(float_format=lambda v: f"{v:.3f}")
            + "\n\n"
        )
    (RESULTS_DIR / split / "summary.txt").write_text("".join(lines))
    print(f"wrote {RESULTS_DIR / split / 'summary.txt'}")


if __name__ == "__main__":
    countries = sorted(
        Path(f).stem.split("_")[1] for f in PROCESSED_DIR.glob("X_*.parquet")
    )
    if len(sys.argv) > 1:  # optional subset, e.g. DE,FR
        countries = sys.argv[1].split(",")
    print(f"Training {MODELS} for {len(countries)} countries: {countries}")
    for cc in countries:
        train_country(cc)
    for split in EVAL_SPLITS:
        write_summary(split)
