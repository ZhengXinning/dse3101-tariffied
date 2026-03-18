import streamlit as st
import pandas as pd
from pathlib import Path

# load dummy dataset
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dummy_dataset.csv"

df = pd.read_csv(file_path, keep_default_na=False)

st.title("test filters")

col1, col2, col3 = st.columns(3) # arrangement of filters

# Country Searchbox
countries = sorted(df["country"].unique()) # list of countries

with col1:
    country = st.selectbox(" ", ["Search Country"] + countries)

# Filter dropdowns
industries = ["All"] + sorted(df["industry"].unique()) # list of industries
regions = ["All"] + sorted(df["region"].unique()) # list of regions

with col2:
    industry = st.selectbox("Industry", industries)

with col3:
    region = st.selectbox("Region", regions)

# Filtering results
filtered = df.copy()

if industry != "All": # filter by industry
    filtered = filtered[filtered["industry"] == industry]

if region != "All": # filter by region
    filtered = filtered[filtered["region"] == region]

if country != "Search Country": # filter by country
    filtered = filtered[filtered["country"] == country]
else: # if no country selected, show top 5 countries by trade value
    top5_countries = (
        filtered.groupby("country")["trade_value"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )
    filtered = filtered[filtered["country"].isin(top5_countries)]

# checking - TO REMOVE LATER
st.write(filtered)