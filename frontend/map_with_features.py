# -------------------------------
# Import libraries
# -------------------------------
import pandas as pd
import streamlit as st

from streamlit_folium import st_folium
import folium
from folium.plugins import AntPath
from folium import DivIcon, Tooltip
from folium.features import GeoJsonTooltip


import json
import pycountry
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dummy_dataset_global.csv"

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(file_path, keep_default_na=False)

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(layout="wide")

# -------------------------------
# Global styling (Dashboard style)
# -------------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h2, h3 {
    font-weight: 600;
}

.subtitle {
    color: #6B7280;
    font-size: 14px;
    margin-bottom: 20px;
}
</style>
            
<style>
/* Default (light mode) */
.legend-box {
    position: fixed;
    bottom: 10px;
    right: 10px;
    z-index: 1000;
    background-color: #F9FAFB;
    color: #111827;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    font-size: 13px;
    max-width: 260px;
}

/* Dark mode override */
.stApp[data-theme="dark"] .legend-box {
    background-color: #1F2937;
    color: #F9FAFB;
    border: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<h2>Singapore Trade Opportunity Dashboard</h2>
<div class="subtitle">
Identify high-potential trade partners based on risk and sector strength
</div>
""", unsafe_allow_html=True)

# Filters
col1, col2, col3, col4 = st.columns(4) 

# Add origin selector
with col1:
    origin = st.selectbox(
        "Origin Country",
        sorted(df["origin"].unique())
    )

# Region Searchbox
regions = ["All"] + sorted(df["region"].unique()) # list of regions

with col2:
    region = st.selectbox("Region", regions)

# Industry Searchbox
with col3:
    selected_industry = st.selectbox(
        "Industry",
        ["All"] + sorted(df['industry'].unique())
    )

# Country Searchbox
countries = sorted(df["country"].unique()) # list of countries

with col4:
    country = st.selectbox("Trading Partners", ["Search Country"] + countries)

# -------------------------------
# Filtering results
# -------------------------------
filtered = df[df["origin"] == origin].copy()

if selected_industry != "All": # filter by industry
    filtered = filtered[filtered["industry"] == selected_industry]

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


# -------------------------------
# Create base map
# -------------------------------
st.markdown("### Global Trade Network")


# -------------------------------
# Fixed-position legend/info over map
# -------------------------------
show_legend = st.checkbox("Show Legend / Info", value=True)

if show_legend:
    st.markdown(
    """
    <div class="legend-box">
        <b>Legend / Info</b><br>
        <hr style="margin:6px 0;">
        <div><b>Risk Index:</b> 0–100 (lower = better)</div>
        <div><b>Marker Color:</b> Green = low risk, Yellow = medium risk, Red = high risk</div>
        <div><b>Actual vs Expected Trade:</b> <100% = trade opportunities present, >100% = potentially overtrading</div>
        <div><b>Arrow Width:</b> Proportional to trade with Singapore (% of SG GDP)</div>
        <div>Click on markers for more trade information</div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Origin coordinates
# -------------------------------
ORIGIN_COORDS = {
    "Singapore": (1.3521, 103.8198),
    "United States of America": (37.09, -95.71),
    "China": (35.86, 104.19)
}

origin_coords = ORIGIN_COORDS[origin]

m = folium.Map(
    location=[20, 0],   # Africa-centered view
    zoom_start=2,
    tiles="CartoDB Voyager"
)

# -------------------------------
# Top 5 countries
# -------------------------------
df_filtered = filtered[filtered["country"] != origin]

top5 = (
    df_filtered.groupby("country")["risk_index"]
    .mean()
    .sort_values(ascending=True)   # lower = better
    .head(5)
    .index
    .tolist()
)

country_scores = (
    df_filtered.groupby("country")["risk_index"]
    .mean()
    .to_dict()
)

# imports/exports over gdp
country_totals = df_filtered.groupby('country').apply(
    lambda x: pd.Series({
        'imports_pct': (x['imports_vol'] * x['trade_pct_gdp']/ x['trade_value']).sum(),
        'exports_pct': (x['exports_vol'] * x['trade_pct_gdp']/ x['trade_value']).sum(),
        'arrow_width_factor': x['trade_pct_gdp'].sum()
    })
).to_dict('index')

# -------------------------------
# Arrow function
# -------------------------------
def Arrow(path, name, color, width):
    AntPath(
        path,
        delay=100,
        weight=width,
        color=color,
        pulse_color=color,
        dash_array=[30, 15],
        tooltip=name
    ).add_to(m)


def get_arrow_width(trade_factor):
    min_width = 2
    max_width = 200
    return min_width + (max_width - min_width) * trade_factor


# -------------------------------
# Helper: ISO2
# -------------------------------
def get_iso2(country_name):
    # because pycountry names it differently from Nat Geo
    special_cases = {
        "United States of America": "US",
        "United Arab Emirates": "AE"
    }
    
    if country_name in special_cases:
        return special_cases[country_name]
    
    try:
        return pycountry.countries.get(name=country_name).alpha_2
    except:
        try:
            return pycountry.countries.get(common_name=country_name).alpha_2
        except:
            return "UN"


# -------------------------------
# Colours
# -------------------------------
def get_color(score):
    if score <= 30:
        return "#2E8B57"   # green
    elif score <= 70:
        return "#F2C94C"   # yellow
    else:
        return "#E15759"   # red
    

# -------------------------------
# Markers + arrows
# -------------------------------
for rank, country in enumerate(top5, start=1):
    country_data = df_filtered[df_filtered["country"] == country]

    # skip if no data after filtering
    if country_data.empty:
        continue

    row = country_data.iloc[0]
    endLA = row["latitude"]
    endLO = row["longitude"]

    # Top sectors
    country_data = df_filtered[df_filtered["country"] == country]
    top_sectors = (
        country_data.sort_values("trade_value", ascending=False)
        .head(3)[["industry", "trade_value", "year"]]
        .values.tolist()
    )
    
    # Weighted Actual vs Expected ratio by industry
    total_weight = country_data["industry_weight"].sum()

    if total_weight > 0:
        weighted_ae = (
            (country_data["actual_vs_expected"] * country_data["industry_weight"]).sum()
            / total_weight
        )
    else:
        weighted_ae = 0

    
    # Colour based on compatibility score 
    color = get_color(country_scores[country])

    # Get weighted imports/exports and arrow width
    imports_vol = country_totals[country]['imports_pct']
    exports_vol = country_totals[country]['exports_pct']
    arrow_factor = country_totals[country]['arrow_width_factor']

    # Map trade_pct_gdp sum to arrow width
    width = get_arrow_width(arrow_factor)

    flag_url = f"https://flagcdn.com/w40/{get_iso2(country).lower()}.png"

    # Professional pop-up
    popup_html = f"""
    <div style="font-family: Arial; font-size: 12px; padding: 8px; min-width:200px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <img src="{flag_url}" style="width:24px;">
            <span style="font-size:14px; font-weight:600;">{country}</span>
        </div>

        <hr style="margin:6px 0;">

        <div>Rank: <b>#{rank}</b></div>
        <div>Risk Index: <b>{row['risk_index']:.2f}</b></div>
        <div>Actual vs Expected Trade: <b>{weighted_ae:.0f}%</b></div>
        
        <div style="margin-top:6px;">
        <div><b>Imports</b>: {imports_vol:.2f}% of {origin} GDP</div>
        <div><b>Exports</b>: {exports_vol:.2f}% of {origin} GDP</div>

        <div style="margin-top:6px;">
            <b>Top Sectors</b>
            <ul style="padding-left:16px; margin:4px 0;">
                {''.join([f"<li>{s}</li>" for s,_,_ in top_sectors])}
            </ul>
        </div>
    </div>
    """
    
    # Gradient marker
    tooltip = Tooltip(
        f"<b>{country}</b>",
        sticky=True,
        style="""
            font-size: 11px;
            border: none;
            border-radius: 6px;
            background-color: rgba(0,0,0,0.75);
            color: white;
            padding: 6px;
        """
    )
     
    

    folium.Marker(
        [endLA, endLO],
        icon=DivIcon(
            icon_size=(32,32),
            icon_anchor=(16,16),
            html=f"""
            <div style="
                font-size:12px;
                font-weight:bold;
                color:white;
                background:{color};
                border-radius:50%;
                width:30px;height:30px;
                text-align:center;
                line-height:30px;">
                {rank}
            </div>
            """
        ),
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=tooltip
    ).add_to(m)

    Arrow([origin_coords, (endLA, endLO)], country, color, width)
  

# -------------------------------
# GeoJSON
# -------------------------------
geojson_path = BASE_DIR / "world_countries.json"

with open(geojson_path, encoding="utf-8") as f:
    geojson = json.load(f)

def style_function(feature):
    if feature['properties']['name'] in top5:
        return {
            'fillColor': "#858AEE",
            'color': "#7A68C2",
            'weight': 1,
            'fillOpacity': 0.4
        }
    else:
        return {
            'fillColor': 'white',
            'color': 'gray',
            'weight': 0.5,
            'fillOpacity': 0.1
        }

highlight_function = lambda x: {
    "fillColor": "lightblue",
    "weight": 2,
    "fillOpacity": 0.5,
}

tooltip_geo = GeoJsonTooltip(
    fields=["name"],
    aliases=[""],
    localize=True,
    sticky=True,
    labels=True,
    style="""
        font-size: 11px;
        background-color: white;
        border: 1px solid black;
        border-radius: 3px;
        text-align:center;
    """,
    max_width=200
)

folium.GeoJson(
    geojson,
    style_function=style_function,
    highlight_function=highlight_function,
    tooltip=tooltip_geo
).add_to(m)

# -------------------------------
# Dynamic Origin Marker
# -------------------------------
origin_iso = get_iso2(origin).lower()
origin_flag_url = f"https://flagcdn.com/w40/{origin_iso}.png"

popup_html_origin = f"""
<div style="font-size:12px;">
    <div style="display:flex; align-items:center; gap:8px;">
        <img src="{origin_flag_url}" style="width:24px;">
        <span style="font-size:14px; font-weight:600;">{origin} (Origin)</span>
    </div>
</div>
"""

tooltip_origin = Tooltip(
    f"<b>{origin} (Origin)</b>",
    sticky=True,
    style="""
        font-size: 11px;
        border: none;
        border-radius: 6px;
        background-color: rgba(0,0,0,0.75);
        color: white;
        padding: 6px;
    """
)

folium.Marker(
    origin_coords,
    tooltip=tooltip_origin,
    icon=folium.Icon(color="red"),
    popup=folium.Popup(popup_html_origin, max_width=250)
).add_to(m)


# -------------------------------
# Display map
# -------------------------------
st_folium(m, use_container_width=True, height=550)


# -------------------------------
# Charts section
# -------------------------------
with st.expander("", expanded=True):        
    st.markdown("### Trade Insights")
    st.markdown(
        '<div class="subtitle">Comparison of partner strength and sector activity</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Chart 1
    with col1:
        chart_data = (
        filtered.groupby("country")
        .agg(
            risk_index=("risk_index", "mean"),
            trade_value=("trade_value", "sum"),
        )
        .reset_index()
    )
 
    # checking for filters
        if country != "Search Country":
            chart_countries = chart_data
            best_trade_country = country
        else:
            top5 = (
                chart_data.nlargest(5, "trade_value")["country"].tolist()
            )
            chart_countries = chart_data[chart_data["country"].isin(top5)]
            best_trade_country = chart_countries.sort_values("trade_value", ascending=False).iloc[0]["country"]
        
        chart_countries = chart_countries.sort_values("trade_value", ascending=False)
        chart_sorted = chart_countries.copy()
        chart_sorted["risk_index"] = (chart_countries["risk_index"]).round(0).astype(int)
        
  
        
        chart_sorted = chart_sorted.sort_values('risk_index')
        chart_sorted['colour'] = chart_sorted['risk_index'].apply(get_color)

        fig1 = px.bar(
            chart_sorted.sort_values('risk_index', ascending=False),
            x='risk_index',
            y='country',
            orientation='h',
            text='risk_index',
            labels={
                'risk_index': 'Risk Index',
                'country': 'Country'
            },
            color='colour',
            color_discrete_map='identity',
        )

        fig1.update_traces(textposition="outside", textfont_size=10)
        fig1.update_layout(
            template="plotly_white",
            font=dict(size=12),
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
            height = 300
        )

        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2
    with col2:
        fig2 = px.bar(
            chart_sorted.sort_values('trade_value', ascending=True),
            x='trade_value',
            y='country',
            orientation='h',
            text='trade_value',
            labels={
                'trade_value': f'{selected_industry} Trade Volume (kg/month)',
                'country': 'Country'
            }
        )

        fig2.update_traces(texttemplate='%{text:.0f}', textposition="outside", textfont_size=10)
        fig2.update_layout(
            template="plotly_white",
            font=dict(size=12),
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
            height = 300
        )

        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------
    # Interpretation
    # -------------------------------
    st.markdown("### Interpretation")
    most_compat_country = chart_sorted.iloc[0]["country"]

    st.info(
        f"{most_compat_country} emerges as the strongest partner amongst countries in {region} region based on risk index. "
        f"The selected industry ({selected_industry}) shows varying trade intensity across countries, "
        "highlighting potential specialization opportunities."
    )
