import os
import glob
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

def plot_all_shap_results():
    out_dir = Path("figures")
    if not out_dir.exists():
        print("Figures directory not found. Please run SHAP analysis first.")
        return
        
    # Find all feature files
    feature_files = glob.glob(str(out_dir / "shap_test_features_*.npy"))
    
    for feat_file in feature_files:
        country_code = Path(feat_file).stem.split("_")[-1]
        print(f"Plotting SHAP for {country_code}...")
        
        # Load features and names
        X_test = np.load(feat_file)
        names_file = out_dir / f"shap_feature_names_{country_code}.txt"
        
        if not names_file.exists():
            print(f"Missing feature names for {country_code}. Skipping.")
            continue
            
        with open(names_file, "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
            
        # Find all target SHAP arrays for this country
        shap_files = glob.glob(str(out_dir / f"shap_raw_{country_code}_*.npy"))
        
        for shap_file in shap_files:
            # Extract target name from filename: shap_raw_CC_Target_Name.npy
            filename = Path(shap_file).stem
            # filename format is: shap_raw_{country_code}_{target_name}
            target_name = filename.replace(f"shap_raw_{country_code}_", "").replace("_", " ")
            
            shap_values = np.load(shap_file)
            
            # Create a WIDER plot as requested
            plt.figure(figsize=(16, 10))
            
            shap.summary_plot(
                shap_values, 
                X_test, 
                feature_names=feature_names, 
                show=False,
                plot_size=(16, 10) # Enforce wider size in shap directly
            )
            
            plt.title(f"SHAP Summary: {country_code} - {target_name}", fontsize=14)
            plt.tight_layout()
            
            safe_target_name = filename.replace(f"shap_raw_{country_code}_", "")
            save_path = out_dir / f"shap_summary_{country_code}_{safe_target_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved plot: {save_path.name}")

if __name__ == "__main__":
    plot_all_shap_results()
