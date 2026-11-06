import os
import copy
import re
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import polars.selectors as cs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smape(y_true, y_pred, eps=1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom, axis=0)

class MultiOutputNN(nn.Module):
    def __init__(self, input_dim, output_dim, lag_groups):
        super().__init__()
        self.lag_groups = list(lag_groups.values())

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_hu():
    print(f"Loading data for Hungary (HU) on device: {device}...")
    x_path = Path("processed_data/X_HU.parquet")
    y_path = Path("processed_data/y_HU.parquet")
    
    if not x_path.exists() or not y_path.exists():
        print("Error: HU parquet files not found in processed_data/")
        return

    # Select only numeric columns to avoid string conversion errors
    X_df = pl.read_parquet(x_path).select(cs.numeric()).to_pandas()
    y_df = pl.read_parquet(y_path).select(cs.numeric()).to_pandas()
    
    feature_names = list(X_df.columns)
    target_names = list(y_df.columns)

    X_np = X_df.to_numpy()
    y_np = y_df.to_numpy()

    print(f"Data shape: X={X_np.shape}, y={y_np.shape}")

    # Train-test split (80-20 chronologically as per previous split)
    split_idx = int(len(X_np) * 0.8)
    X_train, X_test = X_np[:split_idx], X_np[split_idx:]
    y_train, y_test = y_np[:split_idx], y_np[split_idx:]

    print("Scaling data...")
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    # Convert to PyTorch tensors on CPU
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    del X_train, X_test, y_train, y_test

    batch_size = 256  
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    lag_groups = {}
    lagged_bases = set()
    for fname in feature_names:
        m = re.match(r"(.+)_t.*", fname)
        if m:
            lagged_bases.add(m.group(1))

    for i, fname in enumerate(feature_names):
        m = re.match(r"(.+)_t.*", fname)
        if m:
            base = m.group(1)
        else:
            base = fname
        if base in lagged_bases:
            lag_groups.setdefault(base, []).append(i)

    input_dim = X_train_t.shape[1]
    output_dim = y_train_t.shape[1]
    num_seeds = 100
    num_targets = output_dim

    mae_all = np.zeros((num_seeds, num_targets))
    rmse_all = np.zeros((num_seeds, num_targets))
    r2_all = np.zeros((num_seeds, num_targets))
    smape_all = np.zeros((num_seeds, num_targets))

    print(f"Starting training over {num_seeds} seeds...")

    for s in range(num_seeds):
        seed = s
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model_s = MultiOutputNN(input_dim, output_dim, lag_groups).to(device)
        optimizer_s = optim.Adam(model_s.parameters(), lr=1e-4)
        criterion_s = nn.HuberLoss(delta=1.0, reduction='mean')

        gen = torch.Generator()
        gen.manual_seed(seed)
        train_loader_s = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen)
        test_loader_s = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        counter = 0
        patience = 5
        best_state = copy.deepcopy(model_s.state_dict())

        epochs = 100
        for epoch in range(1, epochs + 1):
            model_s.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader_s:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer_s.zero_grad()
                outputs = model_s(X_batch)
                loss = criterion_s(outputs, y_batch)
                loss.backward()
                optimizer_s.step()

                running_loss += loss.item() * X_batch.size(0)

            model_s.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for X_batch, y_batch in test_loader_s:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = model_s(X_batch)
                    val_loss_total += criterion_s(outputs, y_batch).item() * X_batch.size(0)
            val_loss = val_loss_total / len(test_loader_s.dataset)

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
        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader_s:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model_s(X_batch)
                y_pred_list.append(outputs.cpu())
                y_true_list.append(y_batch.cpu())

        y_pred_scaled = torch.cat(y_pred_list).numpy()
        y_true_scaled = torch.cat(y_true_list).numpy()

        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_true_scaled)

        for i in range(num_targets):
            mae_all[s, i] = mean_absolute_error(y_true[:, i], y_pred[:, i])
            rmse_all[s, i] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            r2_all[s, i] = r2_score(y_true[:, i], y_pred[:, i])
        smape_all[s, :] = smape(y_true, y_pred)
        
        if (s+1) % 10 == 0 or s == 0:
            print(f"Completed seed {s+1}/{num_seeds}")

    # compute mean across seeds
    mae_mean_per_target = mae_all.mean(axis=0)
    rmse_mean_per_target = rmse_all.mean(axis=0)
    r2_mean_per_target = r2_all.mean(axis=0)
    smape_mean_per_target = smape_all.mean(axis=0)

    # compute std across seeds
    mae_std_per_target = mae_all.std(axis=0)
    rmse_std_per_target = rmse_all.std(axis=0)
    r2_std_per_target = r2_all.std(axis=0)
    smape_std_per_target = smape_all.std(axis=0)

    metrics_df = pd.DataFrame({
        "Target": target_names,
        "MAE": mae_mean_per_target,
        "RMSE": rmse_mean_per_target,
        "R2": r2_mean_per_target,
        "SMAPE": smape_mean_per_target,
        "MAE_STD": mae_std_per_target,
        "RMSE_STD": rmse_std_per_target,
        "R2_STD": r2_std_per_target,
        "SMAPE_STD": smape_std_per_target
    })

    overall = {
        "MAE": mae_mean_per_target.mean(),
        "RMSE": rmse_mean_per_target.mean(),
        "R2": r2_mean_per_target.mean(),
        "SMAPE": smape_mean_per_target.mean()
    }
    
    print("\n===============================")
    print("FINAL METRICS (100 Seeds)")
    print("===============================")
    print(metrics_df.to_string())
    
    print("\nOverall averages across targets (mean over seeds then targets):")
    print(overall)
    
    print("\nConcise summary per target (Mean ± Std):")
    for i, t in enumerate(target_names):
        print(f"{t}: MAE {mae_mean_per_target[i]:.3f} ± {mae_std_per_target[i]:.3f}, "
              f"RMSE {rmse_mean_per_target[i]:.3f} ± {rmse_std_per_target[i]:.3f}, "
              f"R2 {r2_mean_per_target[i]:.3f} ± {r2_std_per_target[i]:.3f}, "
              f"SMAPE {smape_mean_per_target[i]:.3f} ± {smape_std_per_target[i]:.3f}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    metrics_df.to_csv(results_dir / "NN_100_seeds_Huber_HU_data.csv", index=False)
    print("Metrics saved to NN_100_seeds_Huber_HU_data.csv")

if __name__ == "__main__":
    train_hu()
