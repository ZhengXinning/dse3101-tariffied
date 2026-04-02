# -------------------------------
# Import libraries
# -------------------------------
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
import anthropic
import os
from dotenv import load_dotenv

#BASE_DIR = Path(__file__).resolve().parent
#file_path = BASE_DIR / "dummy_dataset_global_indicators.csv"

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("DSE3101_KEY"))

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR.parent / "backend" / "temp_df" / "df_final.parquet"

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
        ("Reuters Energy", "https://news.google.com/rss/search?q=site:reuters.com+(oil+OR+gas+OR+energy+OR+OPEC)+when:3d&hl=en-US&gl=US&ceid=US:en"), 
        ("South China Morning Post", "https://www.scmp.com/rss/91/feed/"), 
        ("CNA", "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml")
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
#df = pd.read_csv(file_path, encoding='latin1', keep_default_na=False)

df = pd.read_parquet(file_path, engine = "fastparquet") # renaming columns
df = df.rename(columns={
     "refYear": "year",
     "cmdCode": "industry_code",
     "cmdDesc": "industry",
     "reporterCode": "origin_code",
     "reporterISO": "origin_iso",
     "reporterDesc": "origin",
     "reporterRegion": "origin_region",
     "reporterGdp": "origin_gdp",
     "reporterPopulation": "origin_population",
     "reporter_gdp/capita": "origin_gdp_per_capita",
     "reporterlat": "origin_latitude",
     "reporterlong": "origin_longitude",
     "partnerCode": "partner_code",
     "partnerISO": "partner_iso",
     "partnerDesc": "country",
     "partnerRegion": "region",
     "partnerGdp": "partner_gdp",
     "partnerPopulation": "partner_population",
     "partner_gdp/capita": "partner_gdp_per_capita",
     "partnerlat": "latitude",
     "partnerlong": "longitude",
     "exportFlow": "exports_vol",
     "importFlow": "imports_vol",
     "totalFlow": "trade_value",
     "predicted_exportFlow": "predicted_exports",
     "tradeRatio": "actual_vs_expected",
     "reporterTradePctGdp": "origin_trade_pct_gdp",
     "partnerTradePctGdp": "trade_pct_gdp",
     "Risk_Index_Raw": "risk_index_raw"
})


#setting year for the data
df=df[df["year"] ==2021] 
#Normalising the risk
risk_max= df["risk_index_raw"].max()
risk_min= df["risk_index_raw"].min()
df["risk_index"]=100* ((df["risk_index_raw"]- risk_min)/(risk_max-risk_min))

#Only countries with risk_index are shown
df=df[df["risk_index"]>=0]

#Shortens description of industry without examples
df["industry"]=df["industry"].str.split(';').str[0]

#Created industry weights column which measures export volume over total export
df["total_export"]=df.groupby(["origin","country"])["exports_vol"].transform("sum")
df["industry_weight"]=df["exports_vol"]/df["total_export"]

# -------------------------------
# Use ISO-3 to match with pycountry and geo json
# -------------------------------
df["iso3"] = df["partner_iso"]   # for partner countries
df["origin_iso3"] = df["origin_iso"]  # for origin

# -------------------------------
# Standard display names (UI only)
# -------------------------------
display_names = {
    "KOR": "South Korea",
    "USA": "United States",
    "RUS": "Russia",
    "LAO": "Laos",
    "BRN": "Brunei",
    "TUR": "Turkey",
    "CZE": "Czech Republic",
    "SVK": "Slovakia",
    "MKD": "North Macedonia",
    "DOM": "Dominican Republic",
}

df["country_display"] = df["partner_iso"].map(display_names).fillna(df["country"])

# Mapping: display → actual country name
display_to_country = (
    df.drop_duplicates("country_display")
      .set_index("country_display")["country"]
      .to_dict()
)

# Reverse mapping (for showing policies later)
country_to_display = (
    df.drop_duplicates("country")
      .set_index("country")["country_display"]
      .to_dict()
)

# -------------------------------
# Initialise session state
# -------------------------------
if "policies" not in st.session_state:
    st.session_state.policies = []

if "last_news_policy" not in st.session_state:
    st.session_state.last_news_policy = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

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

/* Chat bubble styles */
.chat-user {
    background-color: #1D4ED8;
    color: white;
    padding: 8px 12px;
    border-radius: 12px 12px 2px 12px;
    margin: 4px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 13px;
}
.chat-assistant {
    background-color: #F3F4F6;
    color: #111827;
    padding: 8px 12px;
    border-radius: 12px 12px 12px 2px;
    margin: 4px 0;
    max-width: 85%;
    font-size: 13px;
}
</style>

  
""", unsafe_allow_html=True)


_title_col, _fab_col = st.columns([5, 1])
with _title_col:
    st.markdown("""
<div style="margin-top: 20px;">
    <h2>Singapore Trade Opportunity Dashboard</h2>
    <div class="subtitle">
        Identify high-potential trade partners based on risk and sector strength
    </div>
</div>
""", unsafe_allow_html=True)
with _fab_col:
    st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
    _chat_lbl = "✕ Close" if st.session_state.show_chat else "Chat"
    if st.button(_chat_lbl, key="chat_fab", use_container_width=True):
        st.session_state.show_chat = not st.session_state.show_chat
        st.rerun()

# -------------------------------
# Chatbot helpers
# -------------------------------
def build_dashboard_context():
    try:
        ctx_origin = origin
        ctx_region = region
        ctx_industry = selected_industry
        ctx_comparison = comparison_data
        ctx_policies = st.session_state.policies
    except NameError:
        return "Dashboard data not yet loaded."

    lines = [
        f"Origin Country: {ctx_origin}",
        f"Selected Region: {ctx_region}",
        f"Selected Industry: {ctx_industry}",
        "",
        "Top Trading Partners currently displayed on the map:",
    ]
    for d in ctx_comparison:
        lines.append(
            f"  - {d['Country']} | Rank #{d['Rank']} | Risk Index: {d['Risk Index']:.2f} | "
            f"Actual vs Expected: {d['Actual vs Expected']:.0f}% | "
            f"Imports: {d['Imports %']:.2f}% | Exports: {d['Exports %']:.2f}%"
        )
    if ctx_policies:
        lines.append("")
        lines.append("Active Trade Policies:")
        for p in ctx_policies:
            lines.append(
                f"  - {p['origin']} → {p['country']} ({p['industry']}) | "
                f"Trade x{p['trade_multiplier']}, Risk x{p['risk_multiplier']}, AE +{p['ae_adjustment']}"
            )
    else:
        lines += ["", "No active trade policies."]

    all_countries_list = sorted(df["country"].unique().tolist())
    all_industries_list = sorted(df["industry"].unique().tolist())
    all_regions_list = sorted(df["region"].unique().tolist())
    lines += [
        "",
        f"Dataset covers {len(df)} rows across {len(all_countries_list)} countries.",
        f"Available industries: {', '.join(all_industries_list)}",
        f"Available regions: {', '.join(all_regions_list)}",
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a knowledgeable Trade Assistant embedded inside the Singapore Trade Opportunity Dashboard.
Your role is to help users understand trade data, interpret risk indices, compare trading partners, \
and reason about policy simulation outcomes.

You have access to a real-time snapshot of the current dashboard state (origin country, filters, \
top-ranked partners and their metrics, and any active trade policies). Use this data to give \
precise, insightful answers.

Guidelines:
- Be concise but thorough. Use bullet points when listing multiple items.
- When discussing risk, remind users that Risk Index 0–30 = low (green), 31–70 = medium (yellow), 71–100 = high (red).
- Actual vs Expected Trade <100% means untapped trade opportunity; >100% means potential overtrading.
- If asked about something outside the dataset, say so clearly and suggest what the user could explore on the dashboard.
- Do not make up data. Only use figures provided in the dashboard context below.

Current Dashboard State:
{context}
"""


def get_assistant_response(messages_history: list) -> str:
    context = build_dashboard_context()
    system = SYSTEM_PROMPT.format(context=context)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system,
        messages=messages_history,
    )
    return response.content[0].text


# -------------------------------
# Layout: right column opens when chat is toggled
# -------------------------------
if st.session_state.show_chat:
    col_main, col_chat = st.columns([2.5, 1])
else:
    col_main = st.container()
    col_chat = None

with col_main:
    tab1, tab2, tab3 = st.tabs(["Map & Charts", "Indicators", "Trade Policies"])

# -------------------------------
# Preparation for Indicators Tab
# -------------------------------
# Dictionary of Indicators
INDICATORS = {
    "Transport Cost": "transptCost_weighted",
    "COUNTERPART/REF Exchange Pct Change":"fxChange_weighted",
    "Ideal Point distance": "IdealPointDistance_weighted",
    "Origin Country fatalities": "repFatalities_weighted",
    "Partner Country fatalities":"partFatalities_weighted",
    "Origin Country Violent events": "repEvents_weighted",
    "Partner Country Violent events": "partEvents_weighted",
    "Total FDI": "totalFdi_weighted",
    "State Visits": "stateVisits_weighted"
}

risk_col = st.session_state.get("risk_index", "risk_index")

# Custom risk index calculation based on selected indicators in the Indicators tab
if risk_col == "custom_risk_index":
    selected_cols = st.session_state.get("selected_cols", [])

    if len(selected_cols) > 0:
        df["custom_risk_index"] = 0

        for col in selected_cols:
            if col not in df.columns:
                continue 

        
            else:
                score = df[col]

            df["custom_risk_index"] += score
   
        ci_max = df["custom_risk_index"].max()
        ci_min = df["custom_risk_index"].min()
        df["custom_risk_index"] = 100 * ((df["custom_risk_index"] - ci_min) / (ci_max - ci_min))
    
def apply_policies(df, policies):
    df_sim = df.copy()

    for policy in policies:
        condition = (
            (df_sim["origin"] == policy["origin"]) &
            (df_sim["country"] == policy["country"])
        )

        if policy["industry"] != "All":
            condition &= (df_sim["industry"] == policy["industry"])

        if policy["trade_multiplier"] != 1.0:
            df_sim.loc[condition, "trade_value"] *= policy["trade_multiplier"]

        if policy["risk_multiplier"] != 1.0:
            df_sim.loc[condition, "risk_index"] *= policy["risk_multiplier"]

        if policy["ae_adjustment"] != 0:
            df_sim.loc[condition, "actual_vs_expected"] += policy["ae_adjustment"]

    # recompute weights AFTER simulation
    df_sim["total_export"] = df_sim.groupby(["origin", "country"])["exports_vol"].transform("sum")
    df_sim["industry_weight"] = df_sim["exports_vol"] / df_sim["total_export"]

    return df_sim
# -------------------------------
# Map & Charts Tab
# -------------------------------
with tab1:
    # Filters
    col1, col2, col3 = st.columns(3) 

    # Add origin selector
    with col1:
        origin_options = sorted(df["origin"].unique())
        default_origin_idx = origin_options.index("Singapore") if "Singapore" in origin_options else 0
        origin = st.selectbox(
            "Origin Country",
            origin_options,
            index=default_origin_idx
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
    countries = sorted(df["country_display"].unique())

    # Row 2: Trading partners spanning same width as the 3 filters above
    col4, colspacer = st.columns([3,1])

    # -------------------------------
    # Fixed-position legend/info over map
    # -------------------------------
    st.markdown(
        """
        <style>
        #legend-details[open] .legend-arrow::after { content: "▼"; }
        #legend-details:not([open]) .legend-arrow::after { content: "▲"; }
        #legend-details summary { list-style: none; }
        #legend-details summary::-webkit-details-marker { display: none; }
        </style>
        <div style="position:fixed;bottom:10px;right:10px;z-index:1000;
                    background-color:#f0f2f6;color:#111827;
                    padding:12px;border-radius:8px;font-size:13px;
                    border:1px solid #E5E7EB;box-shadow:0 2px 6px rgba(0,0,0,0.25);
                    max-width:260px;">
            <details id="legend-details" open>
                <summary style="cursor:pointer;font-weight:bold;user-select:none;
                                display:flex;justify-content:space-between;align-items:center;gap:16px;">
                    <span>Legend / Info</span><span class="legend-arrow" style="font-size:11px;"></span>
                </summary>
                <hr style="margin:6px 0;">
                <div><b>Risk Index:</b> 0–100 (lower = better)</div>
                <div><b>Marker Color:</b> Green = low risk, Yellow = medium risk, Red = high risk</div>
                <div><b>Actual vs Expected Trade:</b> &lt;100% = trade opportunities present, &gt;100% = potentially overtrading</div>
                <div><b>Arrow Width:</b> Proportional to trade with Origin Country (% of OC GDP)</div>
                <div>Click on markers for more trade information</div>
            </details>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------
    # Apply policies (outside tabs so tab1 map always reflects active policies)
    # -------------------------------
    df_sim = apply_policies(df, st.session_state.policies)

    #--------------------------------
    # Multiselect trading partners
    #--------------------------------
    #funtion to find list of countries in region
    def filter_region(x):
        condition=df['region']==x
        a=df[condition]
        return sorted(a['country_display'].unique())
    
    #Reducing the number of countries that can be selected by region
    Clist= sorted(df["country_display"].unique())
    
    if region != "All":
            Clist= filter_region(region)

    #function to find top5 countries for default
    def find_5(data):
        scores = data.groupby("country").apply(
            lambda x: (x[risk_col] * x["industry_weight"]).sum()
        )
        return scores.sort_values(ascending=True).head(5).index
    
    #Getting list of top 5 countires for multiselect default
    df_default = df_sim[df_sim["origin"] == origin].copy()

    if region != "All":
        df_region = df_default[df_default["region"] == region].copy()
        top5_countries = find_5(df_region)
        base_df = df_region
    else:
        top5_countries = find_5(df_default)
        base_df = df_default

    # Convert to display names
    default_list = (
        base_df[base_df["country"].isin(top5_countries)]
        .drop_duplicates("country")["country_display"]
        .tolist()
    )

    with col4:
        selected_countries= st.multiselect("Trading Partners",Clist,default=default_list)
        
    # -------------------------------
    # Filtering results
    # -------------------------------
    filtered = df_sim[df_sim["origin"] == origin].copy()

    if selected_industry != "All": # filter by industry
        filtered = filtered[filtered["industry"] == selected_industry]

    if region != "All": # filter by region
        filtered = filtered[filtered["region"] == region]


    if selected_countries != []:
        filtered=filtered[filtered["country_display"].isin(selected_countries)]

    else: # if no country selected, show top 5 countries by risk index
        top5_scores = filtered.groupby("country").apply(
            lambda x: (x[risk_col] * x["industry_weight"]).sum()
        )

        top5 = (
            top5_scores
            .sort_values(ascending=True)
            .head(5)
            .index
            .tolist()
        )
        filtered = filtered[filtered["country"].isin(top5_countries)]

    

    # -------------------------------
    # Top 5 countries
    # -------------------------------
    df_filtered = filtered[filtered["country"] != origin]

    top5 = (
        df_filtered.groupby("country")[risk_col]
        .mean()
        .sort_values(ascending=True)   # lower = better
        .head(5)
        .index
        .tolist()
    )
    
    country_scores = (
        df_filtered.groupby("country").apply(
            lambda x: (x[risk_col] * x["industry_weight"]).sum()
        ).to_dict()
    )

    # imports/exports over gdp
    country_totals = df_filtered.groupby('country').apply(
        lambda x: pd.Series({
            'imports_pct': ((x['imports_vol'] * x['trade_pct_gdp'])/ x['trade_value']).sum()*100,
            'exports_pct': ((x['exports_vol'] * x['trade_pct_gdp'])/ x['trade_value']).sum()*100,
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
    st.markdown("### Global Trade Network")
        
    # Origin coordinates
    ORIGIN_COORDS = {
        "Singapore": (1.3521, 103.8198),
        "USA": (37.09, -95.71),
        "China": (35.86, 104.19),
        "Japan": (36,138),
        "Germany": (51,9)
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
        display_country = row["country_display"]
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
          
        # flag_url
        iso3 = row["partner_iso"]
        try:
            iso2 = pycountry.countries.get(alpha_3=iso3).alpha_2
            flag_url = f"https://flagcdn.com/w40/{iso2.lower()}.png"
        except:
            flag_url = ""

        # Industry info section for popup
        if selected_industry == "All":
            country_industry_vols = (
                df_filtered[df_filtered["country"] == country]
                .groupby("industry")["trade_value"].sum()
                .sort_values(ascending=False)
                .head(3)
            )
            top3_rows = "".join([
                f"<div style='margin-left:8px;'>• {ind}: {vol:,.0f}</div>"
                for ind, vol in country_industry_vols.items()
            ])
            industry_html = f"<div style='margin-top:4px;'><b>Top 3 Industries:</b></div>{top3_rows}"
        else:
            all_industry_vols = (
                df_sim[(df_sim["origin"] == origin) & (df_sim["country"] == country)]
                .groupby("industry")["trade_value"].sum()
                .sort_values(ascending=False)
            )
            if selected_industry in all_industry_vols.index:
                ind_vol = all_industry_vols[selected_industry]
                industry_html = f"<div style='margin-top:4px;'><b>{selected_industry}</b>: {ind_vol:,.0f}</div>"
            else:
                industry_html = f"<div style='margin-top:4px;'><b>{selected_industry}</b>: N/A</div>"

        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; padding: 8px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <img src="{flag_url}" style="width:24px;">
                <span style="font-size:14px; font-weight:600;">{display_country}</span>
            </div>
            
            <hr style="margin:6px 0;">

            <div>Rank: <b>#{rank}</b></div>
            <div>Risk Index: <b>{row[risk_col]:.2f}</b></div>
            <div>Actual vs Expected: <b>{weighted_ae:.0f}%</b></div>

            <div><b>Imports</b>: {imports_vol:.2f}%</div>
            <div><b>Exports</b>: {exports_vol:.2f}%</div>
            {industry_html}
        </div>
        """

        tooltip = Tooltip(
            f"<b>{display_country}</b>",
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
            tooltip=display_country
        ).add_to(m)

        comparison_data.append({
            "Rank": rank,
            "Country": row["country_display"],
            "ISO3": row["partner_iso"],
            "Risk Index": row[risk_col],
            "Actual vs Expected": weighted_ae,
            "Imports %": imports_vol,
            "Exports %": exports_vol,
            "industry_html": industry_html
        })

    # -------------------------------
    # GeoJSON
    # -------------------------------
    geojson_path = BASE_DIR / "world_countries.json"

    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)

    # Countries where GeoJSON uses -99 instead of proper ISO3
    # Keys must match the GeoJSON "name" field exactly
    GEOJSON_ISO_OVERRIDES = {
        "France":                              "FRA",
        "Norway":                              "NOR",
        "Dominica":                            "DMA",
        "Seychelles":                          "SYC",
        "Saint Kitts and Nevis":               "KNA",
        "Bahrain":                             "BHR",
        "Saint Lucia":                         "LCA",
        "Grenada":                             "GRD",
        "Saint Vincent and the Grenadines":    "VCT",
        "Malta":                               "MLT",
        "Mauritius":                           "MUS",
        }
                    
    # Build ISO list for top5
    top5_iso3 = (
        df_filtered[df_filtered["country"].isin(top5)]
        .drop_duplicates("country")["partner_iso"]
        .tolist()
    )

    # Add fall back matching by country name
    name_to_iso3 = df.drop_duplicates("country").set_index("country")["partner_iso"].to_dict()
    # Also add display names
    display_to_iso3 = df.drop_duplicates("country_display").set_index("country_display")["partner_iso"].to_dict()
    name_to_iso3.update(display_to_iso3)

    def style_function(feature):
        iso = feature['properties'].get('iso_a3')
        name = feature["properties"].get("name", "")
           
        # Use override if available, else use GeoJSON iso, else fallback to name lookup
        corrected_iso = GEOJSON_ISO_OVERRIDES.get(name, iso)
        matched = (
            corrected_iso in top5_iso3
            or name_to_iso3.get(name) in top5_iso3
        )

        if matched:
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
    # origin_iso and origin_flag_url
    try:
        origin_iso2 = pycountry.countries.get(alpha_3=df[df["origin"] == origin]["origin_iso"].iloc[0]).alpha_2
        origin_flag_url = f"https://flagcdn.com/w40/{origin_iso2.lower()}.png"
    except:
        origin_flag_url = ""

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
            # flag_url
            try:
                iso2 = pycountry.countries.get(alpha_3=data["ISO3"]).alpha_2
                flag_url = f"https://flagcdn.com/w40/{iso2.lower()}.png"
            except:
                flag_url = ""
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
                    <div style="margin-bottom: 3px;">Exports: {data['Exports %']:.2f}%</div>
                    {data['industry_html']}
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------
    # Charts section
    # -------------------------------
    with st.expander("", expanded=True):        
        st.markdown("### Trade Insights")
        st.markdown(
            '<div class="subtitle"> Scatter Plot of risk vs. trade volume by partner country — points in the green quadrant signal untapped low-risk opportunities, red quadrant warrants caution</div>',
            unsafe_allow_html=True
        )

        # -------------------------------
        # Scatter Plot: Risk vs Trade Volume
        # -------------------------------

        # Use df_sim (policy-applied) and map display names back for labeling
        scatter_base = df_sim[df_sim["origin"] == origin].copy()

        # Apply filters
        if region != "All":
            scatter_base = scatter_base[scatter_base["region"] == region]
        if selected_industry != "All":
            scatter_base = scatter_base[scatter_base["industry"] == selected_industry]
        if selected_countries:
            scatter_base = scatter_base[scatter_base["country_display"].isin(selected_countries)]

        # Aggregate by country (raw name), then attach display name
        scatter_df = (
            scatter_base.groupby("country")
            .agg(
                risk_value=(risk_col, "mean"),
                trade_value=("trade_value", "sum"),
            )
            .reset_index()
        )

        # Attach display names for labels
        display_map = df_sim.drop_duplicates("country").set_index("country")["country_display"]
        scatter_df["display_name"] = scatter_df["country"].map(display_map).fillna(scatter_df["country"])

        # -------------------------------
        # Reference mean: same region & industry as current selection
        # (uses full df, not just selected countries)
        # -------------------------------
        reference_df = df_sim[df_sim["origin"] == origin].copy()
        if region != "All":
            reference_df = reference_df[reference_df["region"] == region]
        if selected_industry != "All":
            reference_df = reference_df[reference_df["industry"] == selected_industry]

        reference_agg = (
            reference_df.groupby("country")
            .agg(
                risk_value=(risk_col, "mean"),
                trade_value=("trade_value", "sum"),
            )
        )

        x_mean = reference_agg["trade_value"].mean()
        y_mean = reference_agg["risk_value"].mean()

        # -------------------------------
        # Axis ranges: centre the intersection
        # -------------------------------
        x_min = scatter_df["trade_value"].min()
        x_max = scatter_df["trade_value"].max()
        y_min = scatter_df["risk_value"].min()
        y_max = scatter_df["risk_value"].max()

        x_pad = (x_max - x_min) * 0.15
        y_pad = (y_max - y_min) * 0.15

        # Distance from mean to each edge, take the larger so mean lands in the centre
        x_half = max(x_mean - x_min, x_max - x_mean) + x_pad
        y_half = max(y_mean - y_min, y_max - y_mean) + y_pad

        x_range = [x_mean - x_half, x_mean + x_half]
        y_range = [y_mean - y_half, y_mean + y_half]

        # Plot
        industry_label = "All Industries" if selected_industry == "All" else selected_industry
        region_label = "All Regions" if region == "All" else region
        x_label = f"Trade Volume — {region_label} | {industry_label}"
        fig = px.scatter(
            scatter_df,
            x="trade_value",
            y="risk_value",
            text="display_name",
            labels={
                "trade_value": x_label,
                "risk_value": "Risk Index"
            }
        )
        
        # -------------------------------
        # Quadrant shading (added before traces so points render on top)
        # -------------------------------
        # Bottom-left: low trade vol, low risk → green (opportunity)
        fig.add_shape(type="rect",
            x0=x_range[0], x1=x_mean,
            y0=y_range[0], y1=y_mean,
            fillcolor="rgba(39, 174, 96, 0.12)",
            line_width=0, layer="below"
        )
        # Top-right: high trade vol, high risk → red (caution)
        fig.add_shape(type="rect",
            x0=x_mean, x1=x_range[1],
            y0=y_mean, y1=y_range[1],
            fillcolor="rgba(231, 76, 60, 0.12)",
            line_width=0, layer="below"
        )

        # Quadrant labels
        fig.add_annotation(x=x_range[0], y=y_range[0], text="🟢 Opportunity",
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(size=11, color="rgba(39,174,96,0.8)"))
        fig.add_annotation(x=x_range[1], y=y_range[1], text="🔴 Caution",
            showarrow=False, xanchor="right", yanchor="top",
            font=dict(size=11, color="rgba(231,76,60,0.8)"))
        

        fig.add_vline(x=x_mean, line_dash="dash", line_color="gray", 
                      annotation_text="Region and Industry average", annotation_position="top right")
        fig.add_hline(y=y_mean, line_dash="dash", line_color="gray", 
                      annotation_text="Region and Industry average", annotation_position="top right")

        fig.update_traces(textposition="top center", marker=dict(size=9))
        fig.update_layout(template="plotly_white", height=500, showlegend=False, xaxis=dict(range=x_range), yaxis=dict(range=y_range))

        st.plotly_chart(fig, use_container_width=True)


        # -------------------------------
        # Interpretation
        # -------------------------------
        if scatter_df.empty: # no data for filtering
            st.info("No data for selected filters. Adjust your filters to see trade insights.")
        else:
        
            # Classify each country into quadrants
            opportunity = scatter_df[(scatter_df["trade_value"] < x_mean) & (scatter_df["risk_value"] < y_mean)]
            caution = scatter_df[(scatter_df["trade_value"] >= x_mean) & (scatter_df["risk_value"] >= y_mean)]
    
            best_opportunity = (
                opportunity.sort_values("risk_value").iloc[0]["display_name"]
                if not opportunity.empty else None
             )
    
            industry_str = "across all industries" if selected_industry == "All" else f"in {selected_industry}"
            region_str   = "globally" if region == "All" else f"in {region}"

            lines = []

            if best_opportunity:
                lines.append(
                    f"**{best_opportunity}** stands out as the top untapped opportunity — "
                    f"low risk and below-average trade volume {industry_str} {region_str}, "
                    f"suggesting room to grow the relationship."
                )
            
            if not caution.empty:
                caution_names = ", ".join(f"**{n}**" for n in caution["display_name"].tolist())
                lines.append(
                    f"{caution_names} "
                    f"{'falls' if len(caution) == 1 else 'fall'} in the caution zone — "
                    f"high trade volume paired with elevated risk warrants closer monitoring."
                )
    
            if lines:
                st.info("  \n".join(lines))
 
# -------------------------------
# Indicators Tab
# -------------------------------
with tab2:
    st.markdown("### Customise Risk Index Indicators")
    st.write("Create your own risk index by selecting which indicators to include in the calculation. You may observe how the risk index and partner rankings change in the Map & Charts tab.")

    user_selections = {}

    for label, col in INDICATORS.items():
        user_selections[col] = st.checkbox(label, value=True, key=f"check_{col}")

    if st.button("Generate Custom Risk Index"):
        selected_cols = [col for col, selected in user_selections.items() if selected]

        if len(selected_cols) == 0:
            st.error("Select at least one indicator.")
        else:
            # store only selection (NOT computation)
            st.session_state.selected_cols = selected_cols
            st.session_state.risk_index = "custom_risk_index"
            st.success("Custom Risk Index Generated! Check the Map & Charts tab.")

# -------------------------------
# Trade Policy Tab
# -------------------------------
DEFAULT_TRADE = 1.0
DEFAULT_RISK  = 1.0
DEFAULT_AE    = 0.0

if "trade_mult" not in st.session_state:
    st.session_state["trade_mult"] = DEFAULT_TRADE

if "risk_mult" not in st.session_state:
    st.session_state["risk_mult"] = DEFAULT_RISK

if "ae_adj" not in st.session_state:
    st.session_state["ae_adj"] = DEFAULT_AE

with tab3:
    
    st.markdown("### Trade Policy Simulation")
    st.write("Simulate the effect of trade policies on risk index, trade volume, and actual vs expected trade. Policies applied here will update the Map & Charts tab.")

    if st.session_state.last_news_policy:
        news_title = st.session_state.last_news_policy
        short_title = news_title[:60] + "…" if len(news_title) > 60 else news_title
        st.success(f"📰 Last added from news: *{short_title}*")
        if st.button("Dismiss", key="dismiss_news_banner"):
            st.session_state.last_news_policy = None
            st.rerun()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Add New Policy")
        policy_origin = st.selectbox("Origin", origin_options, key="origin")
        policy_country_display  = st.selectbox("Partner Country", sorted(df["country_display"].unique()), key="policy_country")
        # Convert back to actual country name
        policy_country = display_to_country[policy_country_display]
        policy_industry = st.selectbox("Industry", ["All"] + sorted(df["industry"].unique()), key="policy_industry")

        if "last_origin" not in st.session_state:
            st.session_state.last_origin = policy_origin

        if policy_origin != st.session_state.last_origin:
            st.session_state.trade_mult = 1.0
            st.session_state.risk_mult  = 1.0
            st.session_state.ae_adj     = 0.0

            st.session_state.last_origin = policy_origin

        sl1, sl2, sl3 = st.columns(3)
        with sl1:
            trade_mult = st.slider("Trade Multiplier", 0.1, 5.0, step=0.1, key="trade_mult")
        with sl2:
            risk_mult  = st.slider("Risk Multiplier",  0.1, 5.0, step=0.1, key="risk_mult")
        with sl3:
            ae_adj     = st.slider("AE Adjustment",   -20.0,   20.0, step=0.5, key="ae_adj")

        
        
        if st.button("Launch New Policy", use_container_width=True):
            st.session_state.policies.append({
                "origin": policy_origin,
                "country": policy_country,
                "industry": policy_industry,
                "trade_multiplier": trade_mult,
                "risk_multiplier": risk_mult,
                "ae_adjustment": ae_adj,
            })
            st.success(
                f"✅ {len(st.session_state.policies)} "
                f"{'policy' if len(st.session_state.policies) == 1 else 'policies'} currently active — "
                "refer to the **Map & Charts** tab to view the changes."
            )

    with col_b:
        n_policies = len(st.session_state.policies)
        st.markdown(f"#### {n_policies} Active Policies")
        
        if not st.session_state.policies:
            st.info("No active policies yet.")
        else:
            for i, p in enumerate(st.session_state.policies):
                display_country = country_to_display.get(p["country"], p["country"])
                news_tag = " 📰" if p.get("from_news") else ""

                col1, col2 = st.columns([5, 1])

                with col1:
                    st.markdown(f"""
                    <div style="padding:10px; margin-bottom:8px; border-radius:8px;
                                border:1px solid rgba(128,128,128,0.3);
                                background-color:var(--secondary-background-color);">
                        <b>{p['origin']}</b> → <b>{display_country}</b> | {p['industry']}{news_tag}<br>
                        <span style="font-size:12px; color:gray;">
                            Trade ×{p['trade_multiplier']} &nbsp;|&nbsp;
                            Risk ×{p['risk_multiplier']} &nbsp;|&nbsp;
                            AE {'+' if p['ae_adjustment'] >= 0 else ''}{p['ae_adjustment']}
                        </span>
                    </div>
                """, unsafe_allow_html=True)

                with col2:
                    if st.button("❌", key=f"delete_policy_{i}"):
                        st.session_state.policies.pop(i)
                        st.rerun()

        if st.button("Clear All Policies", use_container_width=True):
            st.session_state.policies = []
            st.rerun()
    
# -------------------------------
# Chat Panel (right column, visible when show_chat=True)
# -------------------------------
if st.session_state.show_chat and col_chat is not None:
    with col_chat:
        st.markdown("#### Trade Assistant")
        st.markdown(
            '<div class="subtitle" style="font-size:11px;">Ask me anything about the '
            'trade data, risk indices, partner rankings, or active policies.</div>',
            unsafe_allow_html=True
        )

        for i, prompt in enumerate([
            "Which country do I have the lowest risk of trading with?",
            "Which countries do I have untapped trade opportunities with?",
            "Summarise my top 5 trading partners",
            "How do my active policies affect my trade?",
        ]):
            if st.button(prompt, key=f"suggest_{i}", use_container_width=True):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.spinner("Thinking…"):
                    reply = get_assistant_response(st.session_state.chat_messages)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                st.rerun()

        st.divider()

        if not st.session_state.chat_messages:
            st.markdown(
                "<div style='color:#9CA3AF; font-size:12px; text-align:center; padding:20px 0;'>"
                "Type a question below to get started.</div>",
                unsafe_allow_html=True
            )
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-end; margin:6px 0;">'
                    f'<div class="chat-user">{msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="display:flex; justify-content:flex-start; margin:6px 0;">'
                    f'<div class="chat-assistant">{msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )

        st.divider()

        user_input = st.chat_input(
            "e.g. Which country offers the safest trade opportunity?",
            key="chat_input"
        )

        if user_input and user_input.strip():
            st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking…"):
                reply = get_assistant_response(st.session_state.chat_messages)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.rerun()
        
        if st.session_state.chat_messages:
            if st.button("Clear Chat", use_container_width=True, key="clear_chat"):
                st.session_state.chat_messages = []
                st.rerun()


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

        # Sort news so articles mentioning selected origin/partners/region appear first
        try:
            filter_terms = [origin.lower()]
            if country:
                filter_terms += [c.lower() for c in country]
            if region != "All":
                filter_terms.append(region.lower())
            def news_relevance(a):
                text = (a["title"] + " " + a.get("summary", "")).lower()
                return -sum(term in text for term in filter_terms)
            display_news = sorted(news, key=news_relevance)
        except NameError:
            display_news = news

        for i, article in enumerate(display_news):
            date_str = format_date(article["published"])
            title_lower = article["title"].lower()
            # detected_origins = [c for c in all_origins if c.lower() in title_lower]
            # detected_partners = [c for c in all_countries if c.lower() in title_lower]
            # Need at least one origin AND one distinct partner to enable the button
            detected_countries = [c for c in all_countries if c.lower() in title_lower]
            # partner_only = [c for c in detected_par if c not in detected_origins]
            has_countries = bool(detected_countries) & (len(detected_countries) > 1)

            st.markdown(f"""
<div style="margin-bottom:6px;">
  <div style="font-size:11px; font-weight:600; opacity:0.5; color:var(--text-color); margin-bottom:2px;">
    {article['source']}
  </div>
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


