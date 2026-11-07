import sys
import os
import glob
import copy
import re
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import polars.selectors as cs

from afrr_shap.config import SHAP_SEEDS, TEST_START
from afrr_shap.modeling.metrics import smape, target_groups
from afrr_shap.modeling.network import MultiOutputNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_country(country_code, output_txt_file):
    print(f"\n=============================================")
    print(f"Loading data for {country_code} on device: {device}...")
    x_path = Path(f"processed_data/X_{country_code}.parquet")
    y_path = Path(f"processed_data/y_{country_code}.parquet")
    
    if not x_path.exists() or not y_path.exists():
        print(f"Error: {country_code} parquet files not found.")
        return

    X_pl = pl.read_parquet(x_path)
    is_test = (X_pl["timestamp"] >= TEST_START).to_numpy()
    if is_test.sum() == 0 or is_test.all():
        print(f"Skipping {country_code}: needs data both before and in {TEST_START.year}.")
        return
    X_df = X_pl.select(cs.numeric()).to_pandas()
    y_df = pl.read_parquet(y_path).select(cs.numeric()).to_pandas()
    
    feature_names = list(X_df.columns)
    target_names = list(y_df.columns)

    X_np = X_df.to_numpy()
    y_np = y_df.to_numpy()

    print(f"Data shape: X={X_np.shape}, y={y_np.shape}")

    # Chronological split: everything before 2025 trains, 2025 tests
    X_train, X_test = X_np[~is_test], X_np[is_test]
    y_train, y_test = y_np[~is_test], y_np[is_test]
    print(f"Train {len(X_train)} rows, test (2025) {len(X_test)} rows")

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Place tensors directly onto GPU for fast vector execution
    X_train_gpu = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_gpu = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    X_test_gpu = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
    y_test_gpu = torch.tensor(y_test_scaled, dtype=torch.float32, device=device)

    del X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_np, y_np

    batch_size = 1024
    n_samples = len(X_train_gpu)
    n_batches = (n_samples + batch_size - 1) // batch_size

    input_dim = X_train_gpu.shape[1]
    output_dim = y_train_gpu.shape[1]
    num_seeds = 100
    num_targets = output_dim

    mae_all = np.zeros((num_seeds, num_targets))
    rmse_all = np.zeros((num_seeds, num_targets))
    r2_all = np.zeros((num_seeds, num_targets))
    smape_all = np.zeros((num_seeds, num_targets))

    best_overall_val_loss = float('inf')
    best_overall_state = None

    import time
    t0_country = time.time()

    for s in range(num_seeds):
        seed = s
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model_s = MultiOutputNN(input_dim, output_dim).to(device)
        optimizer_s = optim.Adam(model_s.parameters(), lr=1e-4)
        criterion_s = nn.HuberLoss(delta=1.0, reduction='mean')

        best_val_loss = float('inf')
        counter = 0
        patience = 5
        best_state = copy.deepcopy(model_s.state_dict())

        epochs = 100
        for epoch in range(1, epochs + 1):
            model_s.train()
            perm = torch.randperm(n_samples, device=device)
            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                optimizer_s.zero_grad()
                outputs = model_s(X_train_gpu[idx])
                loss = criterion_s(outputs, y_train_gpu[idx])
                loss.backward()
                optimizer_s.step()

            model_s.eval()
            with torch.no_grad():
                val_loss = criterion_s(model_s(X_test_gpu), y_test_gpu).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_state = copy.deepcopy(model_s.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    break

        model_s.load_state_dict(best_state)
        model_s.eval()
        with torch.no_grad():
            y_pred_scaled = model_s(X_test_gpu).cpu().numpy()
        y_true_scaled = y_test_gpu.cpu().numpy()

        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_true_scaled)

        for i in range(num_targets):
            mae_all[s, i] = mean_absolute_error(y_true[:, i], y_pred[:, i])
            rmse_all[s, i] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            r2_all[s, i] = r2_score(y_true[:, i], y_pred[:, i])
        smape_all[s, :] = smape(y_true, y_pred)
        
        if s in SHAP_SEEDS:
            out_dir = Path("models")
            out_dir.mkdir(exist_ok=True)
            torch.save(best_state, out_dir / f"nn_seed{s}_{country_code}.pt")

        if best_val_loss < best_overall_val_loss:
            best_overall_val_loss = best_val_loss
            best_overall_state = copy.deepcopy(best_state)

    print(f"{country_code} 100 seeds finished in {time.time() - t0_country:.2f}s! Best Val Loss: {best_overall_val_loss:.4f}")

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    torch.save(best_overall_state, out_dir / f"nn_100seeds_best_{country_code}.pt")
    with open(out_dir / f"scaler_X_100seeds_best_{country_code}.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(out_dir / f"scaler_y_100seeds_best_{country_code}.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
        
    mae_mean = mae_all.mean(axis=0)
    rmse_mean = rmse_all.mean(axis=0)
    r2_mean = r2_all.mean(axis=0)
    smape_mean = smape_all.mean(axis=0)

    mae_std = mae_all.std(axis=0)
    rmse_std = rmse_all.std(axis=0)
    r2_std = r2_all.std(axis=0)
    smape_std = smape_all.std(axis=0)

    metrics_df = pd.DataFrame({
        "Target": target_names,
        "MAE": mae_mean,
        "RMSE": rmse_mean,
        "R2": r2_mean,
        "SMAPE": smape_mean,
        "MAE_STD": mae_std,
        "RMSE_STD": rmse_std,
        "R2_STD": r2_std,
        "SMAPE_STD": smape_std
    })

    # Naive baseline: the value known at t, carried forward to every horizon
    naive = {"NAIVE_MAE": [], "NAIVE_RMSE": [], "NAIVE_R2": []}
    for i, t in enumerate(target_names):
        base = re.sub(r"_t\+\d+(min|h)$", "", t)
        now_col = next(c for c in feature_names if c in (f"{base}_t+0min", f"{base}_t+0h"))
        y_naive = X_test[:, feature_names.index(now_col)]
        naive["NAIVE_MAE"].append(mean_absolute_error(y_test[:, i], y_naive))
        naive["NAIVE_RMSE"].append(np.sqrt(mean_squared_error(y_test[:, i], y_naive)))
        naive["NAIVE_R2"].append(r2_score(y_test[:, i], y_naive))
    for k, v in naive.items():
        metrics_df[k] = v

    groups = target_groups(target_names, y_test)
    metrics_df.insert(1, "Group", [groups[t] for t in target_names])
    group_summary = (
        metrics_df[~metrics_df["Group"].str.startswith("price Short = Long")]
        .groupby("Group", sort=False)[["MAE", "RMSE", "R2", "SMAPE", "NAIVE_MAE", "NAIVE_RMSE", "NAIVE_R2"]].mean()
    )

    print(f"Results for {country_code}:")
    print(metrics_df)
    print(f"Averages over horizons, per target group:\n{group_summary}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    metrics_df.to_csv(results_dir / f"NN_100_seeds_Huber_{country_code}_data.csv", index=False)
    
    with open(output_txt_file, "a") as f:
        f.write(f"\n===============================\n")
        f.write(f"FINAL METRICS FOR {country_code} (100 Seeds)\n")
        f.write(f"===============================\n")
        f.write(metrics_df.to_string() + "\n")
        f.write(f"\nAverages over horizons, per target group:\n")
        f.write(group_summary.to_string() + "\n")
        f.write("\nConcise summary per target (Mean ± Std):\n")
        for i, t in enumerate(target_names):
            f.write(f"{t}: MAE {mae_mean[i]:.3f} ± {mae_std[i]:.3f}, "
                    f"RMSE {rmse_mean[i]:.3f} ± {rmse_std[i]:.3f}, "
                    f"R2 {r2_mean[i]:.3f} ± {r2_std[i]:.3f}, "
                    f"SMAPE {smape_mean[i]:.3f} ± {smape_std[i]:.3f}\n")

if __name__ == "__main__":
    output_txt = Path("results") / "all_countries_100_seeds_metrics.txt"
    x_files = sorted(glob.glob("processed_data/X_*.parquet"))
    countries = [Path(f).stem.split("_")[1] for f in x_files]
    if len(sys.argv) > 1:  # optional subset, e.g. DE,FR; appends to the combined metrics file
        countries = sys.argv[1].split(",")
    elif output_txt.exists():
        output_txt.unlink()
    
    print(f"Will run 100-seed training for {len(countries)} countries: {countries}")
    
    for cc in countries:
        train_country(cc, output_txt)
