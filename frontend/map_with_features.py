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
import feedparser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dummy_dataset_global_indicators.csv"

# BASE_DIR = Path(__file__).resolve().parent
# file_path = BASE_DIR.parent / "backend" / "temp_df" / "df_final.parquet"

TRADE_KEYWORDS = [
    "trade", "tariff", "geopolit", "sanction", "export", "import",
    "wto", "supply chain", "bilateral", "fta", "free trade", "customs",
    "embargo", "diplomatic", "foreign policy", "alliance", "treaty", "agreement",
    "protectionism", "dumping", "quota", "trade war", "trade deal", "shortage"
]

POSITIVE_KEYWORDS = [
    "boost", "growth", "deal", "agreement", "cooperation", "rise", "gain",
    "opportunity", "strengthen", "recovery", "progress", "approval", "approve", 
    "partnership", "expand", "benefit", "positive", "optimism", "rebound", "record", "friendship"
]

NEGATIVE_KEYWORDS = [
    "war", "sanction", "ban", "decline", "loss", "risk", "tension", "conflict",
    "crisis", "recession", "drop", "fall", "restriction", "penalty", "threat",
    "collapse", "slowdown", "downturn", "tariff hike", "retaliation", "cut",
    "shutdown", "freeze", "dispute", "protest", "deficit", "inflation", "warns"
]

def format_date(published_str):
    if not published_str:
        return ""
    from datetime import datetime, timezone
    try:
        dt = datetime.strptime(published_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hours_ago = (now - dt).total_seconds() / 3600
        if hours_ago < 1:
            mins = int((now - dt).total_seconds() / 60)
            return f"{mins}m ago"
        elif hours_ago < 24:
            return f"{int(hours_ago)}h ago"
        else:
            return dt.strftime("%d %b %Y")
    except ValueError:
        return published_str

def get_sentiment(text):
    text = text.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


# Keyword → industry name (must match dataset values exactly)
INDUSTRY_KEYWORD_MAP = {
    "chip": "Electronics", "semiconductor": "Electronics", "electronic": "Electronics",
    "tech": "Electronics", "circuit": "Electronics",
    "oil": "Energy", "gas": "Energy", "fuel": "Energy", "opec": "Energy", "lng": "Energy",
    "energy": "Energy", "coal": "Energy", "nuclear": "Energy",
    "auto": "Automotive", "vehicle": "Automotive", "car": "Automotive", "ev ": "Automotive",
    "pharmaceutical": "Pharmaceuticals", "drug": "Pharmaceuticals",
    "vaccine": "Pharmaceuticals", "medicine": "Pharmaceuticals",
    "chemical": "Chemicals", "fertilizer": "Chemicals",
    "agriculture": "Agriculture", "food": "Agriculture", "grain": "Agriculture",
    "wheat": "Agriculture", "soy": "Agriculture", "rice": "Agriculture", "corn": "Agriculture",
    "machinery": "Machinery", "equipment": "Machinery", "industrial": "Machinery",
    "financial": "Financial Services", "banking": "Financial Services",
    "insurance": "Financial Services", "fintech": "Financial Services",
}


def extract_policy_from_article(title, all_origins, all_countries, all_industries):
    """Heuristically extract a trade policy from a news article title."""
    text = title.lower()

    # --- Country detection ---
    # Check for origin countries first (small list)
    found_origins = [c for c in all_origins if c.lower() in text]
    # Check for partner countries (also covers origins that are partners)
    found_countries = [c for c in all_countries if c.lower() in text]

    # Pick origin: prefer a found origin, else default to first origin
    origin_result = found_origins[0] if found_origins else all_origins[0]

    # Pick partner: first found country that differs from origin
    partner_candidates = [c for c in found_countries if c != origin_result]
    partner_result = partner_candidates[0] if partner_candidates else (
        [c for c in all_countries if c != origin_result][0]
    )

    # --- Industry detection ---
    found_industry = "All"
    for kw, industry in INDUSTRY_KEYWORD_MAP.items():
        if kw in text and industry in all_industries:
            found_industry = industry
            break

    # --- Impact estimation ---
    severe_negative = any(kw in text for kw in [
        "tariff", "sanction", "ban", "embargo", "restriction", "hike", "retaliation", "penalty", "damage",
    ])
    trade_deal = any(kw in text for kw in [
        "deal", "agreement", "fta", "free trade", "partnership", "cooperation", "treaty"
    ])
    sentiment = get_sentiment(title)

    if severe_negative:
        trade_mult = -1.5
        risk_mult = 1.5
        ae_adj = -10
    elif trade_deal:
        trade_mult = 1.5
        risk_mult = -0.5
        ae_adj = 10
    elif sentiment == "negative":
        trade_mult = -0.5
        risk_mult = 0.5
        ae_adj = -5
    elif sentiment == "positive":
        trade_mult = 0.5
        risk_mult = -0.3
        ae_adj = 5
    else:
        trade_mult = 0.0
        risk_mult = 0.0
        ae_adj = 0

    return {
        "origin": origin_result,
        "country": partner_result,
        "industry": found_industry,
        "trade_multiplier": round(trade_mult, 1),
        "risk_multiplier": round(risk_mult, 1),
        "ae_adjustment": ae_adj,
        "from_news": title,
    }

@st.cache_data(ttl=300)  # refresh every 5 minutes
def get_news():
    feeds = [
        ("Reuters Business", "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Financial Times", "https://www.ft.com/rss/home"),
        ("Reuters World", "https://feeds.reuters.com/reuters/worldNews"),
        ("Al Jazeera Economy", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("Guardian World", "https://www.theguardian.com/world/rss"), 
        ("Reuters Energy", "https://news.google.com/rss/search?q=site:reuters.com+(oil+OR+gas+OR+energy+OR+OPEC)+when:3d&hl=en-US&gl=US&ceid=US:en")
    ]

    articles = []

    for source, url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:20]:  # check more entries per source for keyword filtering
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            text = (title + " " + summary).lower()
            if any(kw in text for kw in TRADE_KEYWORDS):
                # Parse published datetime
                published = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    t = entry.published_parsed
                    published = f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d} {t.tm_hour:02d}:{t.tm_min:02d}"
                elif entry.get("published"):
                    published = entry.get("published", "")[:16]

                articles.append({
                    "title": title,
                    "link": entry.link,
                    "source": source,
                    "published": published,
                    "sentiment": get_sentiment(title + " " + summary)
                })

    return articles

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(file_path, keep_default_na=False)

#df = pd.read_parquet(file_path)
#df = df.rename(columns={
    # "refYear": "year",
    # "cmdCode": "industry_code",
    # "cmdDesc": "industry",
    # "reporterCode": "origin_code",
    # "reporterISO": "origin_iso",
    # "reporterDesc": "origin",
    # "reporterRegion": "origin_region",
    # "reporterGdp": "origin_gdp",
    # "reporterPopulation": "origin_population",
    # "reporter_gdp/capita": "origin_gdp_per_capita",
    # "reporterlat": "origin_latitude",
    # "reporterlong": "origin_longitude",
    # "partnerCode": "partner_code",
    # "partnerISO": "partner_iso",
    # "partnerDesc": "country",
    # "partnerRegion": "region",
    # "partnerGdp": "partner_gdp",
    # "partnerPopulation": "partner_population",
    # "partner_gdp/capita": "partner_gdp_per_capita",
    # "partnerlat": "latitude",
    # "partnerlong": "longitude",
    # "exportFlow": "exports_vol",
    # "importFlow": "imports_vol",
    # "totalFlow": "trade_value",
    # "predicted_exportFlow": "predicted_exports",
    # "tradeRatio": "actual_vs_expected",
    # "riskIndex": "risk_index",
    # "reporterTradePctGdp": "origin_trade_pct_gdp",
    # "partnerTradePctGdp": "trade_pct_gdp",
    # "Tariff": "tariff_rate"
#})

# -------------------------------
# Initialise session state
# -------------------------------
if "policies" not in st.session_state:
    st.session_state.policies = []

if "last_news_policy" not in st.session_state:
    st.session_state.last_news_policy = None

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

<style>
@keyframes flash-red {
    0%, 100% { background-color: #EF4444; opacity: 1; }
    50%       { background-color: #eb7a7a; opacity: 0.9; }
}
.alert-flash {
    animation: flash-red 1.2s ease-in-out infinite;
    color: #F9FAFB;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 700;
    white-space: nowrap;
    display: inline-block;
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

        if st.session_state.last_news_policy:
            news_title = st.session_state.last_news_policy
            short_title = news_title[:60] + "…" if len(news_title) > 60 else news_title
            st.success(f"📰 Last added from news: *{short_title}*")
            if st.button("Dismiss", key="dismiss_news_banner"):
                st.session_state.last_news_policy = None
                st.rerun()

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
            news_tag = " 📰" if p.get("from_news") else ""
            st.write(
                f"{i+1}. {p['origin']} → {p['country']} | {p['industry']} | "
                f"Trade x{p['trade_multiplier']}, Risk x{p['risk_multiplier']}{news_tag}"
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

    # Create comparison cards
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
                        border: 1px solid rgba(128,128,128,0.5);
                        border-radius: 12px;
                        background-color: var(--secondary-background-color);
                        color: var(--text-color);
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


# -------------------------------
# Live News Section (SIDEBAR)
# -------------------------------
with st.sidebar:
    st.markdown("# WORLD NEWS")

    with st.spinner("Loading news..."):
        news = get_news()

    if not news:
        st.info("No trade/geopolitics news found at the moment. Try again shortly.")
    else:
        all_origins = sorted(df["origin"].unique())
        all_countries = sorted(df["country"].unique())
        all_industries = sorted(df["industry"].unique())

        for i, article in enumerate(news):
            is_negative = article["sentiment"] == "negative"
            alert_badge = '<span class="alert-flash">ALERT</span>' if is_negative else ""
            date_str = format_date(article["published"])
            title_lower = article["title"].lower()
            detected_origins = [c for c in all_origins if c.lower() in title_lower]
            detected_partners = [c for c in all_countries if c.lower() in title_lower]
            # Need at least one origin AND one distinct partner to enable the button
            partner_only = [c for c in detected_partners if c not in detected_origins]
            has_countries = bool(detected_origins and partner_only)

            st.markdown(f"""
<div style="margin-bottom:6px;">
  <div style="font-size:11px; font-weight:600; opacity:0.5; color:var(--text-color); margin-bottom:2px;">
    {article['source']}
  </div>
  {alert_badge}
  <a href="{article['link']}" target="_blank"
     style="font-size:13px; font-weight:600; text-decoration:none; color:var(--text-color);">
    {article['title']}
  </a>
  <div style="font-size:11px; margin-top:3px; opacity:0.45; color:var(--text-color);">
    {date_str}
  </div>
</div>
""", unsafe_allow_html=True)

            if has_countries:
                if st.button("Add to Map", key=f"news_policy_{i}", use_container_width=True):
                    policy = extract_policy_from_article(
                        article["title"], all_origins, all_countries, all_industries
                    )
                    st.session_state.policies.append(policy)
                    st.session_state.last_news_policy = article["title"]
                    try:
                        st.toast(
                            f"Policy added: {policy['origin']} → {policy['country']} | {policy['industry']}",
                            icon="📰"
                        )
                    except Exception:
                        pass
                    st.rerun()

            st.markdown('<hr style="margin:6px 0; opacity:0.2;">', unsafe_allow_html=True)