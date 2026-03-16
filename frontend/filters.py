import streamlit as st
import pandas as pd

# load dummy dataset
df = pd.read_csv("dummy_dataset.csv", keep_default_na = False)

st.title("test filters")

col1, col2, col3 = st.columns(3)

# Country Searchbox
countries = sorted(df["country"].unique()) # list of countries

with col1:
    country = st.selectbox("Search Country", countries)

# Filter dropdowns
industries = sorted(df["industry"].unique()) # list of industries
regions = sorted(df["region"].unique()) # list of regions

with col2:
    industry = st.multiselect("Industry", industries)

with col3:
    region = st.multiselect("Region", regions)

# Setting default to select all
if not industry:
    industry = industries
if not region:
    region = regions

# Filtering results
filtered = df[
    (df["country"] == country) &
    (df["industry"].isin(industry)) &
    (df["region"].isin(region))]

# checking - TO REMOVE LATER
st.write(filtered)