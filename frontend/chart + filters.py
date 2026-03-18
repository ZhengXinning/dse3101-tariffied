import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# load dummy dataset
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dummy_dataset.csv"

df = pd.read_csv(file_path, keep_default_na=False)

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

# charts section
with st.expander("", expanded=True):
    left_col, mid_col, right_col = st.columns([1, 2, 2])

    # leftmost text
    with left_col: 
        st.markdown(
            """
            **Scores** are compatibility ratings **(0–100)**:  
            higher scores mean lower geopolitical risk.
 
            Hover your cursor over a particular country or bar for more details.
            """
        )
 
    chart_data = (
        filtered.groupby("country")
        .agg(
            compatibility_score=("compatibility_score", "mean"),
            trade_value=("trade_value", "sum"),
        )
        .reset_index()
    )
 
    # checking for filters
    if country != "Search Country":
        chart_countries = chart_data
    else:
        top5 = (
            chart_data.nlargest(5, "trade_value")["country"].tolist()
        )
        chart_countries = chart_data[chart_data["country"].isin(top5)]
 
    chart_countries = chart_countries.sort_values("trade_value", ascending=False)
    scores_scaled = (chart_countries["compatibility_score"] * 100).round(0).astype(int)
 
    # colour coding for compatibility score 
    def score_colour(s):
        if s < 40:
            return "#e05252"   
        elif s <= 70:
            return "#f0c040"   
        else:
            return "#4caf7d"   
    
    bar_colours = scores_scaled.apply(score_colour).tolist()
        
    # compatibility score bar chart
    with mid_col:
        st.markdown("##### Compatibility Score of Singapore's trading relationship")
        fig_compat = go.Figure(
            go.Bar(  
                orientation = "h",
                y = chart_countries["country"],
                x = scores_scaled,
                marker=dict(color = bar_colours),
                hovertemplate = (
                    "<b>%{x}</b><br>"
                    "Compatibility Score: %{y:.1f}<extra></extra>"
                ),
                text = scores_scaled, 
                textposition = "outside",
                showlegend = False,
            )
        )
        fig_compat.update_layout(
            xaxis = dict(title="Compatibility Score", range=[0, 100], showgrid=True),
            yaxis = dict(title=""),
            margin = dict(t=20, b=40, l=0, r=0),
            height = 320,
            plot_bgcolor = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
            showlegend = False,
        )
        st.plotly_chart(fig_compat, width = "stretch")
    
    # trade value bar chart title 
    if industry != "All":
        trade_title = (
            f"Amount of {industry} traded between Singapore and selected trading partners"
        )
    else:
        trade_title = "Overall amount of trade between Singapore and selected trading partners"

    # trade value bar chart
    with right_col:
        st.markdown(f"##### {trade_title}")
        fig_trade = go.Figure(
            go.Bar(
                orientation = "h",
                y = chart_countries["country"],
                x = chart_countries["trade_value"],
                marker=dict(color="#4a90d9"),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Trade Value: %{y:,.0f}<extra></extra>"
                ),
                text = chart_countries["trade_value"].apply(lambda v: f"{round(v):,}"),
                textposition = "outside",
                showlegend = False,
            )
        )
        fig_trade.update_layout(
            xaxis=dict(
                title="Trade Value",
                showgrid=True,
                tickangle = 0,
                range=[0, chart_countries["trade_value"].max() * 1.25]
                if not chart_countries.empty else [0, 1],
            ),
            yaxis=dict(title=""),
            margin=dict(t=20, b=40, l=0, r=0),
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend = False,
        )
        st.plotly_chart(fig_trade, width = "stretch")