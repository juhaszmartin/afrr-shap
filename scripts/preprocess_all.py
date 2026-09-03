import os
import glob
import polars as pl
import holidays
from pathlib import Path
import numpy as np

# Mapping from Country Code to ERA5 Country Name
CC_TO_NAME = {
    "AT": "austria", "BE": "belgium", "BA": "bosnia_and_herz", "HR": "croatia",
    "FI": "finland", "GR": "greece", "HU": "hungary", "LV": "latvia",
    "LT": "lithuania", "PL": "poland", "PT": "portugal", "RO": "romania",
    "RS": "serbia", "SK": "slovakia", "ES": "spain", "CH": "switzerland",
    "UA": "ukraine"
}

def load_entsoe_data(country_code):
    """Loads and joins all available ENTSO-E parquet data for a given country."""
    base_dir = Path("entsoe_data")
    if not base_dir.exists():
        raise FileNotFoundError("entsoe_data directory not found.")
        
    categories = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    joined_df = None
    for cat in categories:
        pattern = f"entsoe_data/{cat}/country={country_code}/year=*/data.parquet"
        files = glob.glob(pattern)
        if not files:
            continue
            
        dfs = []
        for f in files:
            try:
                df_cat = pl.read_parquet(f)
                
                # Fix pandas multi-index tuple columns saved as strings
                new_cols = []
                for c in df_cat.columns:
                    if c == "('timestamp', '')" or c == "('timestamp', ' ')" or c == "('timestamp', 'UTC')":
                        new_cols.append("timestamp")
                    elif c.startswith("('") and c.endswith("')"):
                        import ast
                        try:
                            c_tuple = ast.literal_eval(c)
                            new_cols.append(f"{cat} - " + " - ".join([str(x) for x in c_tuple if x]).strip())
                        except:
                            new_cols.append(f"{cat} - {c}")
                    elif c != "timestamp":
                        new_cols.append(f"{cat} - {c}")
                    else:
                        new_cols.append(c)
                df_cat.columns = new_cols
                
                if "timestamp" not in df_cat.columns:
                    print(f"Skipping {f}: no timestamp column found")
                    continue
                    
                df_cat = df_cat.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
                df_cat = df_cat.unique(subset=["timestamp"], keep="first")
                dfs.append(df_cat)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        if dfs:
            combined_cat_df = pl.concat(dfs, how="diagonal").sort("timestamp")
            combined_cat_df = combined_cat_df.unique(subset=["timestamp"], keep="last")
            
            if joined_df is None:
                joined_df = combined_cat_df
            else:
                overlap = set(joined_df.columns).intersection(set(combined_cat_df.columns)) - {"timestamp"}
                combined_cat_df = combined_cat_df.drop(list(overlap))
                # Handle how="full" and coalesce=True for newer polars
                joined_df = joined_df.join(combined_cat_df, on="timestamp", how="full", coalesce=True)
                
    if joined_df is not None:
        joined_df = joined_df.sort("timestamp")
    return joined_df

def load_era5_data(country_code):
    """Loads and joins ERA5 weather data for a country."""
    if country_code not in CC_TO_NAME:
        return None
    name = CC_TO_NAME[country_code]
    
    files = {
        "temp_mean": f"weather_data/{name}_hourly_temperature_2019_2025.csv",
        "temp_std": f"weather_data/{name}_hourly_temperature_volatility_2019_2025.csv",
        "ssrd_mean": f"weather_data/{name}_hourly_ssrd_2019_2025.csv",
        "ssrd_std": f"weather_data/{name}_hourly_ssrd_volatility_2019_2025.csv",
        "10m_wind_speed_mean": f"weather_data/{name}_hourly_10m_wind_mean_2019_2025.csv",
        "10m_wind_speed_std": f"weather_data/{name}_hourly_10m_wind_std_2019_2025.csv",
        "100m_wind_speed_mean": f"weather_data/{name}_hourly_100m_wind_mean_2019_2025.csv",
        "100m_wind_speed_std": f"weather_data/{name}_hourly_100m_wind_std_2019_2025.csv",
    }
    
    joined_weather = None
    for col_name, filename in files.items():
        if os.path.exists(filename):
            df = pl.read_csv(filename)
            # rename datetime to timestamp and convert to UTC datetime
            if "datetime" in df.columns:
                df = df.rename({"datetime": "timestamp"})
                df = df.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                )
                df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
                
                # rename the data column if necessary
                # the era5 script saves the metric column with the same name as we want
                
                if joined_weather is None:
                    joined_weather = df
                else:
                    overlap = set(joined_weather.columns).intersection(set(df.columns)) - {"timestamp"}
                    df = df.drop(list(overlap))
                    joined_weather = joined_weather.join(df, on="timestamp", how="full", coalesce=True)
    
    if joined_weather is not None:
        joined_weather = joined_weather.sort("timestamp")
    return joined_weather

def add_time_and_holiday_features(df, country_code):
    # Base time features
    df = df.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("dayofweek"),
        (pl.col("timestamp").dt.weekday() * 2 * np.pi / 7).sin().alias("dayofweek_sin"),
        (pl.col("timestamp").dt.weekday() * 2 * np.pi / 7).cos().alias("dayofweek_cos"),
        (pl.col("timestamp").dt.ordinal_day() / 366).alias("day_of_year"),
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
        pl.col("timestamp").dt.day().alias("day")
    ])
    
    # Holiday feature
    # Using holidays library
    try:
        # Get years in dataset
        years = df.select("year").unique().to_series().to_list()
        # Some countries might not be supported by holidays library directly under their 2-letter code
        # like BA, RS etc. Let's try to get them, if it fails, fallback to 0
        country_holidays = holidays.country_holidays(country_code, years=years)
        
        # Create a boolean list for each date
        def is_holiday(year, month, day):
            return 1 if (year, month, day) in country_holidays else 0
            
        # apply isn't great in polars but we can do a join
        # create a dataframe of dates and holiday flag
        date_df = df.select(["year", "month", "day"]).unique()
        holiday_flags = []
        for row in date_df.iter_rows(named=True):
            y, m, d = row["year"], row["month"], row["day"]
            try:
                dt = f"{y}-{m:02d}-{d:02d}"
                flag = 1 if dt in country_holidays else 0
            except:
                flag = 0
            holiday_flags.append(flag)
            
        date_df = date_df.with_columns(pl.Series("is_holiday", holiday_flags))
        df = df.join(date_df, on=["year", "month", "day"], how="left")
        
    except Exception as e:
        print(f"Warning: Could not load holidays for {country_code}. Setting to 0. {e}")
        df = df.with_columns(pl.lit(0).alias("is_holiday"))
        
    df = df.drop(["year", "month", "day"])
    return df

def timelag_expressions(lags, cols):
    if isinstance(lags, int):
        if lags > 0:
            lags = range(1, lags + 1)
        else:
            lags = range(lags, 0)
            
    exprs = []
    for i in lags:
        for colname in cols:
            exprs.append(pl.col(colname).shift(-i).alias(f"{colname}_t{i*15:+}min"))
    return exprs

def process_country(country_code):
    print(f"Processing {country_code}...")
    df_entsoe = load_entsoe_data(country_code)
    if df_entsoe is None:
        print(f"No ENTSO-E data found for {country_code}.")
        return
        
    df_era5 = load_era5_data(country_code)
    if df_era5 is not None:
        # Join Weather using a nearest or simply outer join, weather is hourly but entsoe might be 15min.
        # Let's upsample weather to 15min using forward fill.
        # First outer join, then sort, then forward fill the weather columns
        weather_cols = [c for c in df_era5.columns if c != "timestamp"]
        df = df_entsoe.join(df_era5, on="timestamp", how="full", coalesce=True)
        df = df.sort("timestamp")
        df = df.with_columns([pl.col(c).fill_null(strategy="forward") for c in weather_cols])
    else:
        df = df_entsoe
        
    # Drop rows where entsoe is heavily null (outer join might have added weather rows outside entsoe range)
    # We only care about valid ENTSO-E targets
    
    # Target columns in ENTSO-E
    # Depending on currency, it might be `+ Imbalance Price (EUR)` etc.
    # Let's dynamically find them
    pos_price_col = [c for c in df.columns if "imbalance_prices" in c and c.endswith("Long")]
    neg_price_col = [c for c in df.columns if "imbalance_prices" in c and c.endswith("Short")]
    imb_vol_col = [c for c in df.columns if "imbalance_volumes" in c and c.endswith("value")]
    
    target_cols = pos_price_col + neg_price_col + imb_vol_col
    if not target_cols:
        print(f"No target columns found for {country_code}. Skipping.")
        return
        
    df = df.drop_nulls(subset=target_cols)
    
    # Add time and holidays
    df = add_time_and_holiday_features(df, country_code)
    
    # We apply same logic as original preprocess_dataset_for_training
    # Remove columns with too many nulls
    null_ratios = df.select([pl.col(c).null_count() / df.height for c in df.columns]).to_dicts()[0]
    keep_cols = [c for c, ratio in null_ratios.items() if ratio <= 0.1]
    
    df = df.select(keep_cols).fill_null(strategy="forward")
    
    # Separate into X and y using time lags
    # Assuming targets exist
    target_cols_exist = [c for c in target_cols if c in df.columns]
    
    time_features = ["hour", "dayofweek", "dayofweek_sin", "dayofweek_cos", "day_of_year", "is_holiday"]
    exclude_cols = {"timestamp", "Situation", "Óraátállítás"}
    
    existing_cols = set(df.columns)
    
    # We don't know the exact ENTSO-E columns that match "cols_10_h" from the Hungarian dataset.
    # We will just lag ALL non-time, non-exclude columns by -40 (past 10 hours if 15min)
    non_time_cols = list(existing_cols - set(time_features) - exclude_cols - set(target_cols_exist))
    
    # Current state of the art: just use lags of all available predictive features
    # Let's lag non-time features by -40 steps (past 10 hours)
    # Wait, creating 40 lags of 20 columns is 800 columns. Might be slow but ok.
    # To mimic original behavior, we can do 5 lags of everything (past 1h15m) and maybe step lags.
    # For now, let's just do -40 for target cols (past prices and volumes) and -5 for other columns.
    
    # Original did: 
    # 40 lags for cols_10_h
    # 48 lags for cols_12_h
    # 5 lags for weather
    # 40 lags backwards (-40) for everything else
    
    weather_cols_exist = [c for c in existing_cols if "temp_" in c or "wind" in c or "ssrd" in c]
    other_cols = list(set(non_time_cols) - set(weather_cols_exist))
    
    forecast_cols = [c for c in other_cols if "forecast" in c.lower() or "scheduled" in c.lower()]
    actual_cols = list(set(other_cols) - set(forecast_cols))
    
    X_exprs = [pl.col(c) for c in time_features if c in df.columns]
    X_exprs += [pl.col(c) for c in non_time_cols] # current values (t=0)
    
    # Past values (Backward lags)
    if weather_cols_exist:
        X_exprs += timelag_expressions(-10, weather_cols_exist)
    if forecast_cols:
        X_exprs += timelag_expressions(-10, forecast_cols)
    if actual_cols:
        X_exprs += timelag_expressions(-10, actual_cols) 
    if target_cols_exist:
        X_exprs += timelag_expressions(-20, target_cols_exist)
        
    # Future values (Forward lags)
    if weather_cols_exist:
        X_exprs += timelag_expressions(range(1, 11), weather_cols_exist)
    if forecast_cols:
        X_exprs += timelag_expressions(range(1, 11), forecast_cols)
        
    y_exprs = timelag_expressions(range(1, 6), target_cols_exist)
    
    X = df.select(X_exprs)
    y = df.select(y_exprs)
    
    # Drop nulls caused by shifting
    combined = pl.concat([X, y], how="horizontal").drop_nulls()
    X = combined.select(X.columns)
    y = combined.select(y.columns)
    
    # Save
    out_dir = Path("processed_data")
    out_dir.mkdir(exist_ok=True)
    X.write_parquet(out_dir / f"X_{country_code}.parquet")
    y.write_parquet(out_dir / f"y_{country_code}.parquet")
    print(f"Saved {country_code}: X shape {X.shape}, y shape {y.shape}")

if __name__ == "__main__":
    imbalance_dirs = glob.glob("entsoe_data/imbalance_prices/country=*")
    countries = [Path(d).name.split("=")[1] for d in imbalance_dirs]
    for cc in countries:
        process_country(cc)
