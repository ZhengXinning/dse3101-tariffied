import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
geojson_path = BASE_DIR / "world_countries.json"
file_path = BASE_DIR.parent / "backend" / "temp_df" / "df_final.parquet"

df = pd.read_parquet(file_path)

with open(geojson_path, encoding="utf-8") as f:
    geojson = json.load(f)

geojson_isos = {f["properties"].get("iso_a3") for f in geojson["features"]}
dataset_isos = set(df["partnerISO"].unique())  # use original column name before rename

print("Dataset ISOs not in GeoJSON:", dataset_isos - geojson_isos)
print("GeoJSON ISOs not in dataset:", geojson_isos - dataset_isos)