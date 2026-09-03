import os
import glob
import json
import pickle
import numpy as np
import polars as pl
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def smape(y_true, y_pred, eps=1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom, axis=0)

class MultiOutputNN(nn.Module):
    def __init__(self, input_dim, output_dim, lag_groups):
        super().__init__()
        # lag_groups could be used for specific architecture variations, but we'll stick to simple dense for now as in the original
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

def get_lag_groups(feature_names):
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
    return lag_groups

def train_country(country_code):
    print(f"--- Training model for {country_code} ---")
    x_path = Path("processed_data") / f"X_{country_code}.parquet"
    y_path = Path("processed_data") / f"y_{country_code}.parquet"
    
    if not x_path.exists() or not y_path.exists():
        print(f"Data not found for {country_code}.")
        return None
        
    import polars.selectors as cs
    X_df = pl.read_parquet(x_path).select(cs.numeric()).to_pandas()
    y_df = pl.read_parquet(y_path).select(cs.numeric()).to_pandas()
    
    # Train-test split (80-20 chronologically)
    split_idx = int(len(X_df) * 0.8)
    X_train, X_test = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
    y_train, y_test = y_df.iloc[:split_idx], y_df.iloc[split_idx:]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    batch_size = 256
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train_t.shape[1]
    output_dim = y_train_t.shape[1]
    lag_groups = get_lag_groups(X_df.columns)
    
    model = MultiOutputNN(input_dim, output_dim, lag_groups).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.HuberLoss(delta=1.0, reduction='mean')
    
    epochs = 30 # Reduced for standard loop; can be increased
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
        
        val_loss /= len(test_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    if best_state is not None:
        model.load_state_dict(best_state)
        
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device))
            y_pred_list.append(outputs.cpu())
            y_true_list.append(y_batch)
            
    y_pred_scaled = torch.cat(y_pred_list).numpy()
    y_true_scaled = torch.cat(y_true_list).numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled)
    
    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "SMAPE": float(np.mean(smape(y_true, y_pred)))
    }
    
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), out_dir / f"nn_{country_code}.pt")
    with open(out_dir / f"scaler_X_{country_code}.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(out_dir / f"scaler_y_{country_code}.pkl", "wb") as f:
        pickle.dump(scaler_y, f)
        
    print(f"Metrics for {country_code}: {metrics}")
    return metrics

if __name__ == "__main__":
    x_files = glob.glob("processed_data/X_*.parquet")
    countries = [Path(f).stem.split("_")[1] for f in x_files]
    
    all_metrics = {}
    for cc in countries:
        metrics = train_country(cc)
        if metrics:
            all_metrics[cc] = metrics
            
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
