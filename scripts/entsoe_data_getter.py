import requests
import polars as pl
import xml.etree.ElementTree as ET
import os
import zipfile
import io
import re
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# --- XML PARSER ---
def parse_entsoe_xml(content, data_type):
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            content = z.read(z.namelist()[0])
    elif b"No matching data found for Data item" in content:
        raise FileNotFoundError
    root = ET.fromstring(content)
    m = re.match(r"\{(.*)\}", root.tag)
    ns = {"ns": m.group(1)} if m else {}

    records = []
    for time_series in root.findall(".//ns:TimeSeries", ns):
        direction = None
        category = None

        # Extract Metadata
        if data_type == "volume":
            node = time_series.find(".//ns:flowDirection.direction", ns)
            if node is not None:
                direction = "Up" if node.text == "A01" else "Down"

        elif data_type == "price":
            node = time_series.find(".//ns:imbalance_Price.category", ns)
            if node is not None:
                category = node.text

        for period in time_series.findall(".//ns:Period", ns):
            start_str = period.find(".//ns:timeInterval/ns:start", ns).text
            start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            res = period.find(".//ns:resolution", ns).text
            step = timedelta(minutes=15) if res == "PT15M" else timedelta(hours=1)

            for point in period.findall(".//ns:Point", ns):
                pos = int(point.find(".//ns:position", ns).text)
                ts = start + (step * (pos - 1))

                if data_type == "volume":
                    val = float(point.find(".//ns:quantity", ns).text)
                    records.append({"timestamp": ts, "direction": direction, "val": val})
                elif data_type == "price":
                    p_node = point.find(".//ns:imbalance_Price.amount", ns)
                    if p_node is None:
                        p_node = point.find(".//ns:price.amount", ns)
                    if p_node is not None:
                        records.append({"timestamp": ts, "category": category, "val": float(p_node.text)})

    df = pl.DataFrame(records)
    if not df.is_empty():
        df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
    return df


def get_situation(net):
    if net > 0.001:
        return "A01"
    if net < -0.001:
        return "A02"
    return "Balanced"


# Timezone: Hungary (CET/CEST)


def get_imbalance_data(country_code, currency, start_date: datetime, end_date: datetime):
    start_str = start_date.strftime("%Y%m%d%H%M")
    end_str = end_date.strftime("%Y%m%d%H%M")

    # --- 1. FETCH & PROCESS VOLUME ---
    params_vol = {
        "securityToken": API_KEY,
        "documentType": "A86",
        "controlArea_Domain": country_code,
        "processType": "A51",
        "periodStart": start_str,
        "periodEnd": end_str,
    }
    r_vol = requests.get("https://web-api.tp.entsoe.eu/api", params=params_vol)
    df_vol = parse_entsoe_xml(r_vol.content, "volume")

    # Calculate Net Position (Up - Down)

    df_vol = df_vol.pivot(index="timestamp", on="direction", values="val").fill_null(0)
    if "Down" not in df_vol.columns:
        df_vol = df_vol.with_columns(pl.lit(0.0).alias("Down"))
    if "Up" not in df_vol.columns:
        df_vol = df_vol.with_columns(pl.lit(0.0).alias("Up"))

    df_vol = df_vol.with_columns((pl.col("Up") - pl.col("Down")).alias("Imbalance (MWh)")).with_columns(
        pl.when(pl.col("Imbalance (MWh)") > 0.001)
        .then(pl.lit("A01"))
        .otherwise(pl.when(pl.col("Imbalance (MWh)") < -0.001).then(pl.lit("A02")).otherwise(pl.lit("Balanced")))
        .alias("Situation")
    )
    # --- 2. FETCH & PROCESS PRICES ---
    params_price = {
        "securityToken": API_KEY,
        "documentType": "A85",
        "controlArea_Domain": country_code,
        "periodStart": start_str,
        "periodEnd": end_str,
    }
    r_price = requests.get("https://web-api.tp.entsoe.eu/api", params=params_price)
    df_price = parse_entsoe_xml(r_price.content, "price")
    df_price = df_price.pivot(index="timestamp", on="category", values="val")
    # --- 3. MERGE & FORMAT ---
    # Create strict 15-min index in UTC
    grid = pl.datetime_range(start_date, end_date - timedelta(minutes=15), interval="15m", time_zone="UTC", eager=True).alias("timestamp")
    df_final = pl.DataFrame({"timestamp": grid})

    # Join
    df_final = (
        df_final.join(df_price, on="timestamp", how="left")
        .join(df_vol.select("timestamp", "Imbalance (MWh)", "Situation"), on="timestamp", how="left")
        .fill_null(strategy="forward")
    )
    # Rename Columns to match Website Header
    df_final = df_final.rename({"A05": f"+ Imbalance Price ({currency})", "A04": f"- Imbalance Price ({currency})"})

    return df_final


if __name__ == "__main__":

    skipto = input("Wanna skip to somewhere? [0]: ")
    skipto = int(skipto) if skipto else 0

    plt.subplots(figsize=(16, 4))
    plt.xlabel("Time")
    plt.ylabel("Price (EUR/MWh)")
    plt.title("Split-Price Data Over Time")

    # --- SETUP ---
    load_dotenv()
    API_KEY = os.getenv("ENTSOE_API_KEY")

    data = pl.read_csv("country_data.csv")

    for i, (name, code, tz, currency) in enumerate(data.rows()):
        if i < skipto:
            continue
        print(f" {name} ".center(40, "="))
        for year in range(2025, 2014, -1):
            start_dt = datetime(year=year, month=1, day=1)
            end_dt = datetime(year=year, month=1, day=2)
            # print(f"Fetching {name} Data for Website Replica: {start_dt.year}...")
            try:
                df_final = get_imbalance_data(code, currency, start_dt, end_dt)
            except FileNotFoundError:
                print(f"{i:02} {name} {year} is not available on ENTSOE")
                continue

            # --- DISPLAY ---
            # Define Display Columns
            cols = [f"+ Imbalance Price ({currency})", f"- Imbalance Price ({currency})", "Imbalance (MWh)", "Situation"]
            plt.plot(df_final["timestamp"], df_final["Imbalance (MWh)"], label="Imbalance (MWh)")
            plt.title(f"Split-Price Data Over Time - {name} - {year}")
            plt.savefig("tmp.png")
            x = input(f"Do you wanna skip {name} ({i:02})? [y/N]: ")
            plt.cla()
            if x.lower() == "y":
                break
