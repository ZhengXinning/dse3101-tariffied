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
file_path = BASE_DIR.parent / "backend" / "temp_df" / "df_final.parquet"

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_parquet(file_path)
#print(df.columns)
#print(df.head())

# -------------------------------
# Initialise session state
# -------------------------------
if "policies" not in st.session_state:
    st.session_state.policies = []

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
    left: 10px;
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
            

<style>
/* Reduce font size for the right panel */
.policy-panel {
    font-size: 12px;
}

/* Make sliders more compact */
div[data-testid="stSlider"] {
    font-size: 12px;
}

/* Reduce spacing */
div.block-container {
    padding-top: 1rem;
}
</style>           
""", unsafe_allow_html=True)


st.markdown("""
<div style="margin-top: 20px;">
    <h2>Singapore Trade Opportunity Dashboard</h2>
    <div class="subtitle">
        Identify high-potential trade partners based on risk and sector strength
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Tab Creation
# -------------------------------
tab1, tab2 = st.tabs(["Map & Charts", "Indicators"])

# -------------------------------
# Map & Charts Tab
# -------------------------------
with tab1:
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

    # Country Multibox
    countries = sorted(df["country"].unique()) # list of countries
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
            <div><b>Arrow Width:</b> Proportional to trade with Origin Country (% of OC GDP)</div>
            <div>Click on markers for more trade information</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    
    # -------------------------------
    # Policy simulation
    # -------------------------------
    col_map, col_panel = st.columns([3, 1])

    with col_panel:
        st.markdown('<div class="policy-panel">', unsafe_allow_html=True)
        st.markdown("### Add Trade Policy")

        policy_origin = st.selectbox("Origin", sorted(df["origin"].unique()), key="p1")
        policy_country = st.selectbox("Partner Country", sorted(df["country"].unique()), key="p2")
        policy_industry = st.selectbox("Industry", ["All"] + sorted(df["industry"].unique()), key="p3")

        col1, col2, col3 = st.columns(3)

        with col1:
            trade_mult = st.slider("Trade Multiplier", -5.0, 5.0, 0.0)

        with col2:
            risk_mult = st.slider("Risk Multiplier", -5.0, 5.0, 0.0)

        with col3:
            ae_adj = st.slider("AE Adjustment", -20, 20, 0)

        if st.button("Launch New Policy"):
            st.session_state.policies.append({
                "origin": policy_origin,
                "country": policy_country,
                "industry": policy_industry,
                "trade_multiplier": trade_mult,
                "risk_multiplier": risk_mult,
                "ae_adjustment": ae_adj
            })

        st.markdown("### Active Policies")

        for i, p in enumerate(st.session_state.policies):
            st.write(
                f"{i+1}. {p['origin']} → {p['country']} | {p['industry']} | "
                f"Trade x{p['trade_multiplier']}, Risk x{p['risk_multiplier']}"
            )

        if st.button("Clear All Policies"):
            st.session_state.policies = []

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # Apply policies
    # -------------------------------
    df_sim = df.copy()

    for policy in st.session_state.policies:

        condition = (
            (df_sim["origin"] == policy["origin"]) &
            (df_sim["country"] == policy["country"])
        )

        if policy["industry"] != "All":
            condition &= (df_sim["industry"] == policy["industry"])

        df_sim.loc[condition, "trade_value"] *= policy["trade_multiplier"]
        df_sim.loc[condition, "risk_index"] *= policy["risk_multiplier"]
        df_sim.loc[condition, "actual_vs_expected"] += policy["ae_adjustment"]


    #--------------------------------
    # Multiselect trading partners
    #--------------------------------


    #funtion to find list of countries in region
    def filter_region(x):
        condition=df['region']==x
        a=df[condition]
        return sorted(a['country'].unique())
    
    #Reducing the number of countries that can be selected by region
    Clist= countries
    Clist.remove(origin)
    if region != "All":
            Clist= filter_region(region)

    #function to find top5 countries for default
    def find_5(data):
        return (
            data.groupby("country")["risk_index"]
            .sum()
            .sort_values(ascending=True)
            .head(5)
            .index
        )
    
    #Getting list of top 5 countires for multiselect default
    default_list=countries
    df_default=df_sim[df_sim["origin"] == origin].copy()
    if region !="All":
        df_region=df_default[df_default["region"]==region].copy()
        top5_countries =  find_5(df_region)
        default_list=top5_countries
    else:
        top5_countries = find_5(df_default)
        default_list=top5_countries

    with col4:
        
        country= st.multiselect("Trading Partners",Clist,default=default_list)

    



    # -------------------------------
    # Filtering results
    # -------------------------------
    filtered = df_sim[df_sim["origin"] == origin].copy()

    if selected_industry != "All": # filter by industry
        filtered = filtered[filtered["industry"] == selected_industry]

    if region != "All": # filter by region
        filtered = filtered[filtered["region"] == region]


    if country != []:
        filtered=filtered[filtered["country"].isin(country)]

    else: # if no country selected, show top 5 countries by risk_index
        top5_countries = (
            filtered.groupby("country")["risk_index"]
            .sum()
            .sort_values(ascending=True)
            .head(5)
            .index
        )
        filtered = filtered[filtered["country"].isin(top5_countries)]

    

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
            'imports_pct': ((x['imports_vol'] * x['trade_pct_gdp'])/ x['trade_value']).sum(),
            'exports_pct': ((x['exports_vol'] * x['trade_pct_gdp'])/ x['trade_value']).sum(),
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
        max_width = 30
        return min_width + (max_width - min_width) * trade_factor


    # -------------------------------
    # Helper: ISO2
    # -------------------------------
    def get_iso2(country_name):
        # because pycountry names it differently from Nat Geo
        special_cases = {
            "United States of America": "US",
            "United Arab Emirates": "AE",
            "Russia": "RU",
            "Turkey": "TR"
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
            return "#269E54"   # green
        elif score <= 70:
            return "#E8BE3F"   # yellow
        else:
            return "#EE4A4D"   # red
        

    # -------------------------------
    # Map
    # -------------------------------
    with col_map:
        st.markdown("### Global Trade Network")
        
        # Origin coordinates
        ORIGIN_COORDS = {
            "Singapore": (1.3521, 103.8198),
            "United States of America": (37.09, -95.71),
            "China": (35.86, 104.19)
        }

        origin_coords = ORIGIN_COORDS[origin]

        m = folium.Map(
            location=[20, 0],
            zoom_start=2,
            tiles="CartoDB Voyager"
        )

        comparison_data = []

        # -------------------------------
        # Markers + arrows (TOP 5)
        # -------------------------------
        for rank, country in enumerate(top5, start=1):

            country_data = df_filtered[df_filtered["country"] == country]
            if country_data.empty:
                continue

            row = country_data.iloc[0]
            endLA = row["latitude"]
            endLO = row["longitude"]

            # Weighted AE
            total_weight = country_data["industry_weight"].sum()
            if total_weight > 0:
                weighted_ae = (
                    (country_data["actual_vs_expected"] * country_data["industry_weight"]).sum()
                    / total_weight
                )
            else:
                weighted_ae = 0

            color = get_color(country_scores[country])

            imports_vol = country_totals[country]['imports_pct']
            exports_vol = country_totals[country]['exports_pct']
            arrow_factor = country_totals[country]['arrow_width_factor']

            width = get_arrow_width(arrow_factor)

            flag_url = f"https://flagcdn.com/w40/{get_iso2(country).lower()}.png"

            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; padding: 8px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <img src="{flag_url}" style="width:24px;">
                    <span style="font-size:14px; font-weight:600;">{country}</span>
                </div>

                <hr style="margin:6px 0;">

                <div>Rank: <b>#{rank}</b></div>
                <div>Risk Index: <b>{row['risk_index']:.2f}</b></div>
                <div>Actual vs Expected: <b>{weighted_ae:.0f}%</b></div>

                <div><b>Imports</b>: {imports_vol:.2f}%</div>
                <div><b>Exports</b>: {exports_vol:.2f}%</div>
            </div>
            """

            tooltip = Tooltip(
                f"<b>{country}</b>",
                sticky=True,
                style="font-size:11px;background-color:rgba(0,0,0,0.75);color:white;padding:6px;"
            )

            folium.Marker(
                [endLA, endLO],
                icon=DivIcon(
                    html=f"""
                    <div style="background:{color};color:white;border-radius:50%;
                                width:30px;height:30px;text-align:center;line-height:30px;">
                        {rank}
                    </div>
                    """
                ),
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=tooltip
            ).add_to(m)

            AntPath(
                [origin_coords, (endLA, endLO)],
                color=color,
                weight=width,
                tooltip=country
            ).add_to(m)

            comparison_data.append({
                "Rank": rank,
                "Country": country,
                "Risk Index": row['risk_index'],      
                "Actual vs Expected": weighted_ae, 
                "Imports %": imports_vol,    
                "Exports %": exports_vol 
            })

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
        # Origin marker
        # -------------------------------
        origin_iso = get_iso2(origin).lower()
        origin_flag_url = f"https://flagcdn.com/w40/{origin_iso}.png"

        folium.Marker(
            origin_coords,
            icon=folium.Icon(color="red"),
            tooltip=Tooltip(f"<b>{origin} (Origin)</b>"),
            popup=folium.Popup(
                f"""
                <div>
                    <img src="{origin_flag_url}" style="width:24px;">
                    <b>{origin} (Origin)</b>
                </div>
                """,
                max_width=250
            )
        ).add_to(m)

        # -------------------------------
        # Render map
        # -------------------------------
        st_folium(m, use_container_width=True, height=550)

    df_compare = pd.DataFrame(comparison_data)

    st.subheader("Trading Partner Overview")
    cols = st.columns(5)

    for i, data in enumerate(comparison_data):
        with cols[i]:
            flag_url = f"https://flagcdn.com/w40/{get_iso2(data['Country']).lower()}.png"
            risk_score = data['Risk Index']
            text_color = get_color(risk_score)
            st.markdown(f"""
                <div style="
                        line-height: 1.2;
                        padding: 12px;
                        border: 1px solid #E5E7EB;
                        border-radius: 12px;
                        background-color: #FFFFFF;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                        ">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom: 5px;">
                        <img src="{flag_url}" style="width:24px; border: 1px solid #eee;">
                        <span style="font-size:20px; font-weight:700;">{data['Country']}</span>
                    </div>
                    <div style="color: gray; font-size:15px;margin-bottom: 5px;"> Rank: #{data['Rank']}</div>
                    <div style="margin-bottom: 3px;"> Risk Index: <span style="color:{text_color};"><b>{data['Risk Index']:.2f}</span></b></div>
                    <div style="margin-bottom: 3px;"> Actual vs Expected: <b>{data['Actual vs Expected']:.0f}%</b></div>
                    <div style="margin-bottom: 3px;"> Imports: {data['Imports %']:.2f}%</div>
                    <div>Exports: {data['Exports %']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

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
            if country != []:
                chart_countries = chart_data
                least_risk_country = country
            else:
                top5 = (
                    chart_data.nlargest(5, "risk_index")["country"].tolist()
                )
                chart_countries = chart_data[chart_data["country"].isin(top5)]
                least_risk_country = chart_countries.sort_values("risk_index", ascending=True).iloc[0]["country"]
            
            chart_countries = chart_countries.sort_values("risk_index", ascending=True)
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

# -------------------------------
# Indicators Tab
# -------------------------------
with tab2:
    st.markdown("### Customise Risk Index Indicators")
    st.write("Create your own risk index by selecting which indicators to include in the calculation. You may observe how the risk index and partner rankings change in the Map & Charts tab.")
