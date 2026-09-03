import glob
import polars as pl
from collections import defaultdict

x_files = sorted(glob.glob("processed_data/X_*.parquet"))
doc_lines = ["# ENTSO-E Data Availability by Country\n\n"]
doc_lines.append("This document summarizes the raw ENTSO-E datasets and specific variables that were successfully extracted and fed into the neural network for each country.\n\n")

from pathlib import Path

for f in x_files:
    cc = Path(f).stem.split("_")[1]
    df = pl.read_parquet(f).head(1)
    
    # We just want the base features, not the lags
    base_cols = set()
    for col in df.columns:
        if "_t-" not in col and "_t+" not in col:
            base_cols.add(col)
            
    # Remove time and weather base columns
    time_and_weather = {"hour", "dayofweek", "dayofweek_sin", "dayofweek_cos", "day_of_year", "is_holiday",
                        "temp_mean", "temp_std", "ssrd_mean", "ssrd_std", "10m_wind_speed_mean", "10m_wind_speed_std", 
                        "100m_wind_speed_mean", "100m_wind_speed_std"}
    
    entsoe_cols = base_cols - time_and_weather
    
    categories = defaultdict(list)
    for col in entsoe_cols:
        if " - " in col:
            cat = col.split(" - ")[0]
            subcol = " - ".join(col.split(" - ")[1:])
            categories[cat].append(subcol)
        else:
            categories["Other"].append(col)
            
    doc_lines.append(f"## {cc}\n")
    for cat, subcols in sorted(categories.items()):
        doc_lines.append(f"- **{cat}**:\n")
        for sc in sorted(subcols):
            doc_lines.append(f"  - `{sc}`\n")
    doc_lines.append("\n")

with open("entsoe_availability.md", "w") as out:
    out.writelines(doc_lines)
print("done")
