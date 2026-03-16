import streamlit as st
import pandas as pd

# load dummy dataset
df = pd.read_csv("dummy_dataset.csv", keep_default_na = False)

st.title("test filters")

col1, col2, col3 = st.columns(3)

# Country Searchbox
countries = sorted(df["country"].unique()) # list of countries

top5 = (
    df.groupby("country")["trade_value"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)

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
filtered = df
if country == "Search Country":
    filtered = filtered[(filtered["country"].isin(top5))]
else: 
    filtered = filtered[(filtered["country"] == country)]
if industry != "All":
    filtered = filtered[filtered["industry"] == industry]
if region != "All":
    filtered = filtered[filtered["region"] == region]

# checking - TO REMOVE LATER
st.write(filtered)