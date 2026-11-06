import os
import glob
import pickle
import numpy as np
import polars as pl
from pathlib import Path
import re
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import shap

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

def analyze_country(country_code):
    print(f"--- Running SHAP for {country_code} ---")
    x_path = Path("processed_data") / f"X_{country_code}.parquet"
    y_path = Path("processed_data") / f"y_{country_code}.parquet"
    model_path = Path("models") / f"nn_{country_code}.pt"
    scaler_X_path = Path("models") / f"scaler_X_{country_code}.pkl"
    scaler_y_path = Path("models") / f"scaler_y_{country_code}.pkl"
    
    if not all(p.exists() for p in [x_path, y_path, model_path, scaler_X_path, scaler_y_path]):
        print(f"Missing artifacts for {country_code}. Skipping.")
        return
        
    import polars.selectors as cs
    X_df = pl.read_parquet(x_path).select(cs.numeric()).to_pandas()
    y_df = pl.read_parquet(y_path).select(cs.numeric()).to_pandas()
    
    with open(scaler_X_path, "rb") as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, "rb") as f:
        scaler_y = pickle.load(f)
        
    # Standard split
    split_idx = int(len(X_df) * 0.8)
    X_train, X_test = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
    y_test = y_df.iloc[split_idx:]
    
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = X_train_scaled.shape[1]
    output_dim = y_test.shape[1]
    lag_groups = get_lag_groups(X_df.columns)
    
    model = MultiOutputNN(input_dim, output_dim, lag_groups)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # SHAP requires a background dataset
    # 500 samples are usually sufficient for DeepExplainer
    np.random.seed(42)
    bg_indices = np.random.choice(X_train_scaled.shape[0], 500, replace=False)
    background = torch.tensor(X_train_scaled[bg_indices], dtype=torch.float32).to(device)
    
    # Test dataset to explain (e.g., 200 samples to keep computation time reasonable)
    test_indices = np.random.choice(X_test_scaled.shape[0], 200, replace=False)
    test_samples = torch.tensor(X_test_scaled[test_indices], dtype=torch.float32).to(device)
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)
    
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    
    target_names = y_df.columns
    
    # shap_values is a list of arrays (one per output) or a single array if output_dim is 1
    # For PyTorch, DeepExplainer often returns a list
    if not isinstance(shap_values, list):
        shap_values = [shap_values[:, :, i] for i in range(output_dim)]
        
    for i, target_name in enumerate(target_names):
        plt.figure(figsize=(10, 8))
        # Ensure we pass the feature names
        shap.summary_plot(shap_values[i], X_test_scaled[test_indices], feature_names=X_df.columns.tolist(), show=False)
        # Sanitize target name for filename
        safe_target_name = "".join(c for c in target_name if c.isalnum() or c in (' ', '_')).rstrip()
        safe_target_name = safe_target_name.replace(" ", "_")
        
        plt.title(f"SHAP Summary: {country_code} - {target_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_summary_{country_code}_{safe_target_name}.png")
        plt.close()

if __name__ == "__main__":
    x_files = glob.glob("processed_data/X_*.parquet")
    countries = [Path(f).stem.split("_")[1] for f in x_files]
    
    for cc in countries:
        analyze_country(cc)
