# -------------------------------
# Import libraries
# -------------------------------
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_folium import st_folium
import folium
from folium.plugins import AntPath
from folium import DivIcon, Tooltip
from folium.features import GeoJsonTooltip

import json
import re
import pycountry
import plotly.express as px
import feedparser
from pathlib import Path
import anthropic
import os

#BASE_DIR = Path(__file__).resolve().parent
#file_path = BASE_DIR / "dummy_dataset_global_indicators.csv"

def get_client():
    if "DSE3101_KEY" not in st.secrets:
        st.error("API key not configured.")
        st.stop()
    return anthropic.Anthropic(api_key=st.secrets["DSE3101_KEY"])
client = get_client()
# client = anthropic.Anthropic(api_key=st.secrets["DSE3101_KEY"])

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


# Keyword → industry display name (must match values in industry_mapping exactly)
INDUSTRY_KEYWORD_MAP = {
    # Electrical Machinery & Electronics
    "chip": "Electrical Machinery & Electronics",
    "semiconductor": "Electrical Machinery & Electronics",
    "electronic": "Electrical Machinery & Electronics",
    "circuit": "Electrical Machinery & Electronics",
    "computer": "Electrical Machinery & Electronics",
    "battery": "Electrical Machinery & Electronics",
    "data center": "Electrical Machinery & Electronics",
    # Mineral Fuels & Oils
    "oil": "Mineral Fuels & Oils",
    "gas": "Mineral Fuels & Oils",
    "fuel": "Mineral Fuels & Oils",
    "opec": "Mineral Fuels & Oils",
    "lng": "Mineral Fuels & Oils",
    "crude": "Mineral Fuels & Oils",
    "petroleum": "Mineral Fuels & Oils",
    "coal": "Mineral Fuels & Oils",
    "refinery": "Mineral Fuels & Oils",
    "energy": "Mineral Fuels & Oils",
    # Vehicles & Parts
    "vehicle": "Vehicles & Parts",
    "automobile": "Vehicles & Parts",
    "automotive": "Vehicles & Parts",
    "electric vehicle": "Vehicles & Parts",
    "truck": "Vehicles & Parts",
    # Pharmaceuticals
    "pharmaceutical": "Pharmaceuticals",
    "drug": "Pharmaceuticals",
    "vaccine": "Pharmaceuticals",
    "medicine": "Pharmaceuticals",
    "biotech": "Pharmaceuticals",
    # Organic Chemicals
    "organic chemical": "Organic Chemicals",
    "petrochemical": "Organic Chemicals",
    # Inorganic Chemicals
    "inorganic chemical": "Inorganic Chemicals",
    "rare earth": "Inorganic Chemicals",
    # Fertilizers
    "fertilizer": "Fertilizers",
    "potash": "Fertilizers",
    "phosphate": "Fertilizers",
    # Machinery & Boilers
    "machinery": "Machinery & Boilers",
    "boiler": "Machinery & Boilers",
    "turbine": "Machinery & Boilers",
    "industrial equipment": "Machinery & Boilers",
    # Nuclear Reactors & Machinery
    "nuclear reactor": "Nuclear Reactors & Machinery",
    "nuclear plant": "Nuclear Reactors & Machinery",
    # Optical & Medical Instruments
    "medical device": "Optical & Medical Instruments",
    "optical instrument": "Optical & Medical Instruments",
    "surgical": "Optical & Medical Instruments",
    # Aircraft & Spacecraft
    "aircraft": "Aircraft & Spacecraft",
    "aerospace": "Aircraft & Spacecraft",
    "aviation": "Aircraft & Spacecraft",
    "spacecraft": "Aircraft & Spacecraft",
    "satellite": "Aircraft & Spacecraft",
    "boeing": "Aircraft & Spacecraft",
    "airbus": "Aircraft & Spacecraft",
    # Ships & Boats
    "ship": "Ships & Boats",
    "vessel": "Ships & Boats",
    "maritime": "Ships & Boats",
    "naval": "Ships & Boats",
    "shipbuilding": "Ships & Boats",
    # Railway Equipment
    "railway": "Railway Equipment",
    "railroad": "Railway Equipment",
    "locomotive": "Railway Equipment",
    # Iron & Steel
    "steel": "Iron & Steel",
    "iron ore": "Iron & Steel",
    # Aluminium
    "aluminium": "Aluminium",
    "aluminum": "Aluminium",
    # Copper
    "copper": "Copper",
    # Nickel
    "nickel": "Nickel",
    # Rubber
    "rubber": "Rubber",
    # Plastics
    "plastic": "Plastics",
    "polymer": "Plastics",
    # Jewellery & Precious Metals
    "gold": "Jewellery & Precious Metals",
    "silver": "Jewellery & Precious Metals",
    "jewellery": "Jewellery & Precious Metals",
    "jewelry": "Jewellery & Precious Metals",
    "precious metal": "Jewellery & Precious Metals",
    "diamond": "Jewellery & Precious Metals",
    # Arms & Ammunition
    "ammunition": "Arms & Ammunition",
    "weapons": "Arms & Ammunition",
    "defence export": "Arms & Ammunition",
    "defense export": "Arms & Ammunition",
    # Cereals
    "wheat": "Cereals",
    "rice": "Cereals",
    "corn": "Cereals",
    "grain": "Cereals",
    "cereal": "Cereals",
    # Oil Seeds & Grains
    "soybean": "Oil Seeds & Grains",
    "soy": "Oil Seeds & Grains",
    "palm oil": "Oil Seeds & Grains",
    "oilseed": "Oil Seeds & Grains",
    # Meat & Offal
    "beef": "Meat & Offal",
    "pork": "Meat & Offal",
    "poultry": "Meat & Offal",
    "meat": "Meat & Offal",
    # Fish & Seafood
    "fish": "Fish & Seafood",
    "seafood": "Fish & Seafood",
    "shrimp": "Fish & Seafood",
    # Sugar & Confectionery
    "sugar": "Sugar & Confectionery",
    # Coffee, Tea & Spices
    "coffee": "Coffee, Tea & Spices",
    "spice": "Coffee, Tea & Spices",
    # Cotton
    "cotton": "Cotton",
    # Apparel & Clothing
    "textile": "Apparel & Clothing",
    "apparel": "Apparel & Clothing",
    "garment": "Apparel & Clothing",
    "fashion": "Apparel & Clothing",
    # Wood & Charcoal
    "timber": "Wood & Charcoal",
    "lumber": "Wood & Charcoal",
    # Paper & Paperboard
    "paper": "Paper & Paperboard",
    # Wood Pulp & Waste Paper
    "pulp": "Wood Pulp & Waste Paper",
    # Tobacco
    "tobacco": "Tobacco",
    "cigarette": "Tobacco",
    # Ores, Slag & Ash
    "ore": "Ores, Slag & Ash",
    "mining": "Ores, Slag & Ash",
    "lithium": "Ores, Slag & Ash",
    "cobalt": "Ores, Slag & Ash",
}

# Short-form aliases → exact dataset country name (verified against df["origin"] and df["country"])
COUNTRY_ALIASES = {
    # USA  (dataset uses "USA" for both reporter and partner)
    "us": "USA", "u.s.": "USA", "america": "USA", "american": "USA", "washington": "USA",
    # United Kingdom  (dataset: "United Kingdom")
    "uk": "United Kingdom", "u.k.": "United Kingdom", "britain": "United Kingdom",
    "british": "United Kingdom", "england": "United Kingdom", "london": "United Kingdom",
    # China  (dataset: "China" for origin; check partner)
    "chinese": "China", "beijing": "China", "prc": "China",
    # Russian Federation  (dataset: "Russian Federation", not "Russia")
    "russia": "Russian Federation", "russian": "Russian Federation",
    "moscow": "Russian Federation", "kremlin": "Russian Federation",
    # Rep. of Korea  (dataset: "Rep. of Korea", not "South Korea")
    "korea": "Rep. of Korea", "korean": "Rep. of Korea", "seoul": "Rep. of Korea",
    "south korea": "Rep. of Korea",
    # Saudi Arabia  (dataset: "Saudi Arabia")
    "saudi": "Saudi Arabia", "riyadh": "Saudi Arabia",
    # United Arab Emirates  (dataset: "United Arab Emirates")
    "uae": "United Arab Emirates", "dubai": "United Arab Emirates",
    # Japan  (dataset: "Japan" for origin)
    "japanese": "Japan", "tokyo": "Japan",
    # Germany  (dataset: "Germany" for origin)
    "german": "Germany", "berlin": "Germany",
    # France  (dataset: "France")
    "french": "France", "paris": "France",
    # India  (dataset: "India")
    "indian": "India", "delhi": "India", "new delhi": "India",
    # Brazil  (dataset: "Brazil")
    "brazilian": "Brazil",
    # Australia  (dataset: "Australia")
    "australian": "Australia",
    # Canada  (dataset: "Canada")
    "canadian": "Canada", "ottawa": "Canada",
    # Mexico  (dataset: "Mexico")
    "mexican": "Mexico",
    # Turkey  (dataset: "Turkey" or "Türkiye" — kept as "Turkey")
    "turkish": "Turkey", "ankara": "Turkey",
    # Indonesia  (dataset: "Indonesia")
    "indonesian": "Indonesia", "jakarta": "Indonesia",
    # Vietnam  (dataset: "Viet Nam" likely — add both)
    "vietnamese": "Viet Nam", "hanoi": "Viet Nam", "vietnam": "Viet Nam",
    # Thailand  (dataset: "Thailand")
    "thai": "Thailand", "bangkok": "Thailand",
    # Malaysia  (dataset: "Malaysia")
    "malaysian": "Malaysia", "kuala lumpur": "Malaysia",
    # Philippines  (dataset: "Philippines")
    "philippine": "Philippines", "filipino": "Philippines", "manila": "Philippines",
    # Myanmar  (dataset: "Myanmar")
    "burma": "Myanmar", "burmese": "Myanmar",
    # Pakistan  (dataset: "Pakistan")
    "pakistani": "Pakistan", "islamabad": "Pakistan",
    # Singapore  (dataset: "Singapore" for origin)
    "singaporean": "Singapore",
}

def detect_countries_in_text(text, all_countries):
    """
    Detect dataset country names in text using both direct substring match and
    alias/short-form match (with word boundaries for short aliases).
    """
    text_lower = text.lower()
    found = set()
    # Direct match
    for c in all_countries:
        if c.lower() in text_lower:
            found.add(c)
    # Alias match with word boundaries to avoid partial-word false positives
    for alias, country in COUNTRY_ALIASES.items():
        if country in all_countries:
            if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
                found.add(country)
    return list(found)


def extract_policy_from_article(title, all_origins, all_countries, all_industries):
    """Heuristically extract a trade policy from a news article title."""
    text = title.lower()

    # --- Country detection (uses aliases for short forms like US, UK, etc.) ---
    found_origins = [c for c in detect_countries_in_text(text, all_origins) if c in all_origins]
    found_countries = detect_countries_in_text(text, all_countries)

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

_POLICY_LEVERS = [
    "ln_tariff",
    "ln_ideal_point_distance",
    "ln_distcap",
    "ln_reporter_gdp_per_capita",
    "ln_partner_gdp_per_capita",
]

def get_claude_policy_suggestion(title, summary, origin, partner, industry):
    """
    Call Claude Haiku to suggest lever % changes for a trade policy based on
    a news article. Returns a dict with lever keys + 'reasoning', or None on failure.
    """
    lever_list = "\n".join(f"- {lv}" for lv in _POLICY_LEVERS)
    prompt = f"""You are a trade policy analyst. Given the news article below, suggest percentage
changes for each gravity-model lever to simulate the article's likely trade impact.

Article title: {title}
Article summary: {summary}
Detected origin country: {origin}
Detected partner country: {partner}
Detected industry: {industry}

Return ONLY a JSON object (no markdown, no explanation outside the JSON) with exactly these keys:
{lever_list}
- reasoning

Rules:
- Each lever value must be an integer between -50 and 100.
- 0 means no change.
- ln_tariff: positive = tariff hike (more trade friction), negative = tariff cut.
- ln_ideal_point_distance: positive = greater political divergence, negative = closer alignment.
- ln_distcap: rarely changes; use 0 unless the article implies a major geographic/logistics shift.
- ln_reporter_gdp_per_capita and ln_partner_gdp_per_capita: reflect expected GDP growth/contraction.
- reasoning: 1-2 sentences explaining your choices.

Example (do not copy values blindly):
{{"ln_tariff": 25, "ln_ideal_point_distance": 10, "ln_distcap": 0, "ln_reporter_gdp_per_capita": -5, "ln_partner_gdp_per_capita": -5, "reasoning": "The tariff hike increases trade friction while political tensions worsen bilateral alignment."}}"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        import json as _json
        data = _json.loads(raw)
        result = {lv: int(max(-50, min(100, data.get(lv, 0)))) for lv in _POLICY_LEVERS}
        result["reasoning"] = str(data.get("reasoning", ""))
        return result
    except Exception:
        return None


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
                    "sentiment": get_sentiment(title + " " + summary),
                    "summary": summary[:300],
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
     "predicted_exportFlow_geoPol": "predicted_exports",
     "predicted_exportFlow_base": "baseline_exports",
     "tradeRatio": "actual_vs_expected",
     "reporterTradePctGdp": "origin_trade_pct_gdp",
     "partnerTradePctGdp": "trade_pct_gdp",
     "Risk_Index_Raw": "risk_index_raw"
})

# Load coefficient data set for Trade Policy simulation
file_path_coef = BASE_DIR.parent / "backend" / "temp_df" / "df_coef.parquet"
df_coef = pd.read_parquet(file_path_coef, engine="fastparquet")
coef_map = dict(zip(df_coef["variable"], df_coef["coef"]))

#setting year for the data
df=df[df["year"] ==2021] 
#Normalising the risk
risk_max= df["risk_index_raw"].max()
risk_min= df["risk_index_raw"].min()
df["risk_index"]=100* ((df["risk_index_raw"]- risk_min)/(risk_max-risk_min))

#Only countries with risk_index are shown
df=df[df["risk_index"]>=0]

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

# Rename industries
industry_mapping = {
    'Electrical machinery and equipment and parts thereof; sound recorders and reproducers; television image and sound recorders and reproducers, parts and accessories of such articles': 'Electrical Machinery & Electronics',
    'Nuclear reactors, boilers, machinery and mechanical appliances; parts thereof': 'Nuclear Reactors & Machinery',
    'Optical, photographic, cinematographic, measuring, checking, medical or surgical instruments and apparatus; parts and accessories': 'Optical & Medical Instruments',
    'Vehicles; other than railway or tramway rolling stock, and parts and accessories thereof': 'Vehicles & Parts',
    'Rubber and articles thereof': 'Rubber',
    'Plastics and articles thereof': 'Plastics',
    'Printed books, newspapers, pictures and other products of the printing industry; manuscripts, typescripts and plans': 'Printed Media',
    'Tools, implements, cutlery, spoons and forks, of base metal; parts thereof, of base metal': 'Tools & Cutlery',
    'Furniture; bedding, mattresses, mattress supports, cushions and similar stuffed furnishings; lamps and lighting fittings, n.e.c.; illuminated signs, illuminated name-plates and the like; prefabricated buildings': 'Furniture & Lighting',
    'Miscellaneous manufactured articles': 'Misc. Manufactured Articles',
    'Metal; miscellaneous products of base metal': 'Misc. Base Metal Products',
    'Pharmaceutical products': 'Pharmaceuticals',
    'Apparel and clothing accessories; knitted or crocheted': 'Knitwear & Clothing',
    'Paper and paperboard; articles of paper pulp, of paper or paperboard': 'Paper & Paperboard',
    'Apparel and clothing accessories; not knitted or crocheted': 'Apparel & Clothing',
    'Inorganic chemicals; organic and inorganic compounds of precious metals; of rare earth metals, of radio-active elements and of isotopes': 'Inorganic Chemicals',
    'Photographic or cinematographic goods': 'Photographic Goods',
    'Organic chemicals': 'Organic Chemicals',
    'Tanning or dyeing extracts; tannins and their derivatives; dyes, pigments and other colouring matter; paints, varnishes; putty, other mastics; inks': 'Dyes, Paints & Inks',
    'Essential oils and resinoids; perfumery, cosmetic or toilet preparations': 'Cosmetics & Perfumery',
    'Articles of leather; saddlery and harness; travel goods, handbags and similar containers; articles of animal gut (other than silk-worm gut)': 'Leather Goods',
    'Furskins and artificial fur; manufactures thereof': 'Fur & Furskins',
    'Wood and articles of wood; wood charcoal': 'Wood & Charcoal',
    'Silk': 'Silk',
    'Cotton': 'Cotton',
    'Vegetable textile fibres; paper yarn and woven fabrics of paper yarn': 'Vegetable Textile Fibres',
    'Man-made filaments; strip and the like of man-made textile materials': 'Synthetic Filaments',
    'Man-made staple fibres': 'Synthetic Staple Fibres',
    'Wadding, felt and nonwovens, special yarns; twine, cordage, ropes and cables and articles thereof': 'Wadding, Rope & Cables',
    'Carpets and other textile floor coverings': 'Carpets & Floor Coverings',
    'Fabrics; special woven fabrics, tufted textile fabrics, lace, tapestries, trimmings, embroidery': 'Woven Fabrics & Lace',
    'Textile fabrics; impregnated, coated, covered or laminated; textile articles of a kind suitable for industrial use': 'Industrial Textile Fabrics',
    'Fabrics; knitted or crocheted': 'Knitted Fabrics',
    'Textiles, made up articles; sets; worn clothing and worn textile articles; rags': 'Made-up Textiles & Rags',
    'Footwear; gaiters and the like; parts of such articles': 'Footwear',
    'Headgear and parts thereof': 'Headgear',
    'Umbrellas, sun umbrellas, walking-sticks, seat sticks, whips, riding crops; and parts thereof': 'Umbrellas & Walking Sticks',
    'Stone, plaster, cement, asbestos, mica or similar materials; articles thereof': 'Stone, Cement & Plaster',
    'Ceramic products': 'Ceramics',
    'Glass and glassware': 'Glass & Glassware',
    'Natural, cultured pearls; precious, semi-precious stones; precious metals, metals clad with precious metal, and articles thereof; imitation jewellery; coin': 'Jewellery & Precious Metals',
    'Iron or steel articles': 'Iron & Steel Articles',
    'Copper and articles thereof': 'Copper',
    'Nickel and articles thereof': 'Nickel',
    'Aluminium and articles thereof': 'Aluminium',
    'Lead and articles thereof': 'Lead',
    'Zinc and articles thereof': 'Zinc',
    'Ships, boats and floating structures': 'Ships & Boats',
    'Clocks and watches and parts thereof': 'Clocks & Watches',
    'Musical instruments; parts and accessories of such articles': 'Musical Instruments',
    'Arms and ammunition; parts and accessories thereof': 'Arms & Ammunition',
    'Toys, games and sports requisites; parts and accessories thereof': 'Toys, Games & Sports',
    'Railway, tramway locomotives, rolling-stock and parts thereof; railway or tramway track fixtures and fittings and parts thereof; mechanical (including electro-mechanical) traffic signalling equipment of all kinds': 'Railway Equipment',
    'Trees and other plants, live; bulbs, roots and the like; cut flowers and ornamental foliage': 'Live Plants & Cut Flowers',
    'Fish and crustaceans, molluscs and other aquatic invertebrates': 'Fish & Seafood',
    'Aircraft, spacecraft and parts thereof': 'Aircraft & Spacecraft',
    "Works of art; collectors' pieces and antiques": 'Art & Antiques',
    'Manufactures of straw, esparto or other plaiting materials; basketware and wickerwork': 'Basketware & Wickerwork',
    'Explosives; pyrotechnic products; matches; pyrophoric alloys; certain combustible preparations': 'Explosives & Pyrotechnics',
    'Soap, organic surface-active agents; washing, lubricating, polishing or scouring preparations; artificial or prepared waxes, candles and similar articles, modelling pastes, dental waxes and dental preparations with a basis of plaster': 'Soap, Waxes & Cleaning Products',
    'Animals; live': 'Live Animals',
    'Fruit and nuts, edible; peel of citrus fruit or melons': 'Fruit & Nuts',
    "Dairy produce; birds' eggs; natural honey; edible products of animal origin, not elsewhere specified or included": 'Dairy, Eggs & Honey',
    'Mineral fuels, mineral oils and products of their distillation; bituminous substances; mineral waxes': 'Mineral Fuels & Oils',
    'Chemical products n.e.c.': 'Misc. Chemical Products',
    'Wool, fine or coarse animal hair; horsehair yarn and woven fabric': 'Wool & Animal Hair',
    'Feathers and down, prepared; and articles made of feather or of down; artificial flowers; articles of human hair': 'Feathers, Down & Artificial Flowers',
    'Tin; articles thereof': 'Tin',
    'Metals; n.e.c., cermets and articles thereof': 'Misc. Metals & Cermets',
    'Albuminoidal substances; modified starches; glues; enzymes': 'Starches, Glues & Enzymes',
    'Lac; gums, resins and other vegetable saps and extracts': 'Gums & Resins',
    'Cork and articles of cork': 'Cork',
    'Coffee, tea, mate and spices': 'Coffee, Tea & Spices',
    'Beverages, spirits and vinegar': 'Beverages & Spirits',
    'Vegetables and certain roots and tubers; edible': 'Vegetables',
    'Products of the milling industry; malt, starches, inulin, wheat gluten': 'Milling Products & Malt',
    'Oil seeds and oleaginous fruits; miscellaneous grains, seeds and fruit, industrial or medicinal plants; straw and fodder': 'Oil Seeds & Grains',
    'Animal or vegetable fats and oils and their cleavage products; prepared animal fats; animal or vegetable waxes': 'Animal & Vegetable Fats & Oils',
    'Meat, fish or crustaceans, molluscs or other aquatic invertebrates; preparations thereof': 'Meat & Seafood Preparations',
    'Sugars and sugar confectionery': 'Sugar & Confectionery',
    'Salt; sulphur; earths, stone; plastering materials, lime and cement': 'Salt, Sulphur & Cement',
    'Preparations of vegetables, fruit, nuts or other parts of plants': 'Preserved Vegetables & Fruit',
    "Preparations of cereals, flour, starch or milk; pastrycooks' products": 'Cereal & Flour Preparations',
    'Cocoa and cocoa preparations': 'Cocoa',
    'Raw hides and skins (other than furskins) and leather': 'Hides, Skins & Leather',
    'Iron and steel': 'Iron & Steel',
    'Tobacco and manufactured tobacco substitutes': 'Tobacco',
    'Cereals': 'Cereals',
    'Vegetable plaiting materials; vegetable products not elsewhere specified or included': 'Misc. Vegetable Products',
    'Meat and edible meat offal': 'Meat & Offal',
    'Animal originated products; not elsewhere specified or included': 'Misc. Animal Products',
    'Food industries, residues and wastes thereof; prepared animal fodder': 'Animal Feed & Food Waste',
    'Ores, slag and ash': 'Ores, Slag & Ash',
    'Fertilizers': 'Fertilizers',
    'Pulp of wood or other fibrous cellulosic material; recovered (waste and scrap) paper or paperboard': 'Wood Pulp & Waste Paper',
    'Miscellaneous edible preparations': 'Misc. Edible Preparations',
    'Machinery and mechanical appliances, boilers, nuclear reactors; parts thereof': 'Machinery & Boilers',
    'Aircraft, spacecraft, and parts thereof': 'Aircraft & Spacecraft',
    'Animal, vegetable or microbial fats and oils and their cleavage products; prepared edible fats; animal or vegetable waxes': 'Fats, Oils & Waxes',
    'Meat, fish, crustaceans, molluscs or other aquatic invertebrates, or insects; preparations thereof': 'Meat, Fish & Insect Preparations',
    'Tobacco and manufactured tobacco substitutes; products, whether or not containing nicotine, intended for inhalation without combustion; other nicotine containing products intended for the intake of nicotine into the human body': 'Tobacco & Nicotine Products',
}
df["industry"] = df["industry"].map(industry_mapping).fillna(df["industry"].str.split(';').str[0])

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

if "pending_news_policy" not in st.session_state:
    st.session_state.pending_news_policy = {}

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
/* TAB STYLING */
/* Reset tab background */
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #61656C; /* default grey */
}
/* Remove hover background */
.stTabs [data-baseweb="tab"]:hover {
    background-color: transparent !important;
}

/* Hide real underline */
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* TAB 1 */
/* Hover */
.stTabs [data-baseweb="tab"]:nth-of-type(1):hover {
    color: #F92828 !important;
}

/* Active tab text */
.stTabs [data-baseweb="tab"]:nth-of-type(1)[aria-selected="true"] {
    color: #F92828 !important;
}
            
/* Colour line below */
.stTabs [data-baseweb="tab"]:nth-of-type(1)[aria-selected="true"] {
    border-bottom: 3px solid #F92828;
}
                 
/* TAB 2 */
/* Hover */
.stTabs [data-baseweb="tab"]:nth-of-type(2):hover {
    color: #225CD0 !important;
}

/* Active tab text */
.stTabs [data-baseweb="tab"]:nth-of-type(2)[aria-selected="true"] {
    color: #225CD0 !important;
}
            
/* Colour line below */
.stTabs [data-baseweb="tab"]:nth-of-type(2)[aria-selected="true"] {
    border-bottom: 3px solid #225CD0;
}

            
/* TAB 3 */
/* Hover */
.stTabs [data-baseweb="tab"]:nth-of-type(3):hover {
    color: #0F9C51 !important;
}

/* Active tab text */
.stTabs [data-baseweb="tab"]:nth-of-type(3)[aria-selected="true"] {
    color: #0F9C51 !important;
}
            
/* Colour line below */
.stTabs [data-baseweb="tab"]:nth-of-type(3)[aria-selected="true"] {
    border-bottom: 3px solid #0F9C51;
}
            
/* TAB 4 */
/* Hover */
.stTabs [data-baseweb="tab"]:nth-of-type(4):hover {
    color: #8D51E8 !important;
}

/* Active tab text */
.stTabs [data-baseweb="tab"]:nth-of-type(4)[aria-selected="true"] {
    color: #8D51E8 !important;
}
            
/* Colour line below */
.stTabs [data-baseweb="tab"]:nth-of-type(4)[aria-selected="true"] {
    border-bottom: 3px solid #8D51E8;
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
    <h2>Trade Opportunity Dashboard</h2>
    <div class="subtitle">
        Identify high-potential trade partners based on risk and sector strength
    </div>
</div>
""", unsafe_allow_html=True)
with _fab_col:
    st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
    _chat_lbl = "Close" if st.session_state.show_chat else "Chat"
    if st.button(_chat_lbl, key="chat_fab", use_container_width=True):
        st.session_state.show_chat = not st.session_state.show_chat
        st.rerun()

# -------------------------------
# Chatbot helpers
# -------------------------------
def build_dashboard_context():
    try:
        ctx_origin = origin
    except NameError:
        ctx_origin = "Unknown"

    try:
        ctx_regions = selected_regions
    except NameError:
        ctx_regions = ["All"]

    try:
        ctx_industries = selected_industries
    except NameError:
        ctx_industries = ["All"]

    try:
        ctx_comparison = comparison_data
    except NameError:
        ctx_comparison = []

    ctx_policies = st.session_state.policies

    lines = [
        f"Origin Country: {ctx_origin}",
        f"Selected Regions: {', '.join(ctx_regions)}",
        f"Selected Industries: {', '.join(ctx_industries)}",
        "",
        "Top Trading Partners currently displayed on the map:",
    ]

    if ctx_comparison:
        for d in ctx_comparison:
            lines.append(
                f"  - {d['Country']} | Rank #{d['Rank']} | Risk Index: {d['Risk Index']:.2f} | "
                f"Actual vs Expected: {d['Actual over Expected']:.0f}% | "
                f"Imports: {d['Imports %']:.2f}% | Exports: {d['Exports %']:.2f}%"
            )
    else:
        lines.append("  (No partner data available yet — user may not have opened the Map & Charts tab.)")

    if ctx_policies:
        lines.append("")
        lines.append("Active Trade Policies:")
        for p in ctx_policies:
            # Handle both policy formats (manual levers vs legacy multipliers)
            if "policy_vars" in p:
                lever_summary = ", ".join(
                    f"{k.replace('ln_', '').replace('_', ' ').title()}: {'+' if v > 0 else ''}{v}%"
                    for k, v in p["policy_vars"].items() if v != 0
                ) or "no lever changes"
                lines.append(
                    f"  - {p['origin']} → {p['country']} ({p['industry']}) | "
                    f"Levers: {lever_summary} | Estimated trade impact: {p.get('trade_effect', 0):+.1f}%"
                    + (f" [from news: {p['from_news'][:60]}]" if p.get("from_news") else "")
                )
            else:
                lines.append(
                    f"  - {p['origin']} → {p['country']} ({p['industry']}) | "
                    f"Trade x{p.get('trade_multiplier', 0)}, Risk x{p.get('risk_multiplier', 0)}"
                )
    else:
        lines += ["", "No active trade policies."]

    all_countries_list = sorted(df["country"].unique().tolist())
    all_industries_list = sorted(df["industry"].unique().tolist())
    all_regions_list = sorted(df["region"].unique().tolist())
    lines += [
        "",
        f"Dataset covers {len(df)} rows across {len(all_countries_list)} partner countries.",
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
        model="claude-haiku-4-5-20251001",
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

# -------------------------------
# Tab Creation
# -------------------------------
with col_main:
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Map & Charts", "Indicators", "Trade Policies"])

# -------------------------------
# Preparation for Indicators Tab
# -------------------------------
# Dictionary of Indicators
INDICATORS = {
    "Transport Cost": "transptCost_weighted",
    "Exchange Rate Change (%)":"fxChange_weighted",
    "Ideal Point Distance": "IdealPointDistance_weighted",
    "Origin Country Fatalities": "repFatalities_weighted",
    "Partner Country Fatalities":"partFatalities_weighted",
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
    
# -------------------------------
# Helper function to simulate policy setting
# -------------------------------    
def apply_policies(df, policies, coef_map):
    df_sim = df.copy()

    # reset to baseline
    df_sim["predicted_exports"] = df["predicted_exports"].copy()

    # Apply gravity model effects in log space
    for i, row in df_sim.iterrows():

        total_log_effect = 0

        for policy in policies:

            # Only check Origin and Country
            # Ensures the effect applies to EVERY industry row for this pair
            if row["origin"] == policy["origin"] and row["country"] == policy["country"]:
                for var, pct_change in policy["policy_vars"].items():

                    coef = coef_map.get(var, 0)

                    multiplier = 1 + pct_change / 100
                    multiplier = max(multiplier, 1e-6)

                    total_log_effect += coef * np.log(multiplier)

        # apply the combined effect of all matching policies to this industry row
        df_sim.at[i, "exports_vol"] *= np.exp(total_log_effect)

    # Recompute actual vs expected
    df_sim["actual_vs_expected"] = (
        df_sim["exports_vol"] / df_sim["predicted_exports"]
    )

    # Weights 
    df_sim["total_export"] = df_sim.groupby(
        ["origin", "country"]
    )["exports_vol"].transform("sum")

    df_sim["industry_weight"] = (
        df_sim["exports_vol"] / df_sim["total_export"]
    )
    # Recompute trade_value
    df_sim["trade_value"] = (
        df_sim["exports_vol"] + df_sim["imports_vol"]
    )

    # Trade as % of GDP
    df_sim["origin_trade_pct_gdp"] = (
        df_sim["trade_value"] / df_sim["origin_gdp"]
    )

    return df_sim


# -------------------------------
# Overview Tab
# -------------------------------
with tab1:
    # Introduction
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        This application is designed for government analysts and policymakers focused on international trade.
                    
        Assess the influence of geopolitical tensions on global trade by:
        - Visualising bilateral trade flows
        - Identifying potential trade opportunities and risks
        - Simulating trade policies and scenarios
        - Supporting analysis at both the national and sectoral level
        """)

    with col2:
        st.markdown("""
        <div style="
        background: var(--color-background-secondary);
        color: var(--color-text-primary);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        ">

        <div style="
        font-size: 13px;
        color: var(--color-text-secondary);
        margin-bottom: 6px;
        letter-spacing: 0.5px;
        ">
        YEAR OF DATA
        </div>

        <div style="
        font-size: 34px;
        font-weight: 600;
        ">
        2021
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        st.write("<i>Note: Currently, all data is based on figures in the latest available year with sufficient data for comprehensive trade analysis.<i>", unsafe_allow_html=True)

    st.markdown("---")

    # Instructions
    st.markdown("### How to Use")
    col1, col2, col3 = st.columns(3)

    def info_card(title, content, color):
        return f"""
        <div style="
        background: var(--color-background-secondary);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        display: flex;
        gap: 12px;
        ">

        <div style="
        width: 6px;
        border-radius: 6px;
        background: {color};
        "></div>

        <div>
            <div style="font-weight: 600; margin-bottom: 6px;">
                {title}
            </div>
            <div style="font-size: 14px; color: var(--color-text-secondary);">
                {content}
            </div>
        </div>

        </div>
        """

    with col1:
        st.markdown(info_card("Map & Charts", 
        """
        - Explore global trade network
        - Select an origin country (Singapore/China/Germany/Japan/USA)
        - Filter by industry, region, partners  
        - Compare trade metrics and risk indices
        """, "#7DA8FF"), unsafe_allow_html=True)

    with col2:
        st.markdown(info_card("Indicators",
        """
        - Customise Risk Index  
        - Select relevant indicators  
        - Observe ranking changes in Map & Charts 
        """, "#7ED6A7"), unsafe_allow_html=True)

    with col3:
        st.markdown(info_card("Trade Policies",
        """
        - Simulate trade scenarios by adjusting policy sliders
        - Select % changes in trade variables 
        - Estimate % change in export trade flows
        - Observe policy effects in Map & Charts
        """, "#C6A0FF"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    **World News:** View the latest world news most relevant to the selected origin country. Click on each headline to be directed to original article.
                  
    **Trade Assistant:** Chat with the trade assistant (top right) to ask specific questions and generate additional trade insights. <br> 
    """, unsafe_allow_html=True)

    st.markdown("<i>For more details, please check out the videos below!<i>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    st.markdown("---")

    # Gravity Model info
    st.markdown("### The Gravity Model of Trade")
    st.markdown("""
    Inspired by Newton’s law of gravitation, the gravity model estimates trade flows from one country to another based on their respective economic sizes and distance between them. In its simplest theoretical form, the model relates trade volume to the Gross Domestic Product (GDP), geographical distance and an error term. \n
    In this application, the baseline gravity model incorporates additional determinants of trade relationships, including population and import tariffs. These factors enhance the gravity model to highlight the impact of geopolitical tensions on bilateral trade. \n
    Our modified gravity model focuses on geopolitical alignment, which we quantify based on the voting patterns in the United Nations General Assembly (UNGA). We employ ideal point estimates to derive the ideal point distance between countries, where a smaller distance indicates closer geopolitical alignment.\n  
    This application presents three measures of export volumes.
    
    - **Actual Trade Volume:** The observed total value (in USD) of real-world export trade from one country to another.
    - **Baseline Trade Volume:** The predicted export trading volume from one country to another based on the baseline gravity model using GDP per capita, population, geographical distance and export tariffs, excluding geopolitical distance.
    - **Expected Trade Volume:** The predicted export trading volume from one country to another based on the modified gravity model, including geopolitical distance along with the other determinants in the baseline model.
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Grouped Bar Chart
    # preparing data
    agg_tradevol = (
        df.groupby("origin")[["exports_vol", "baseline_exports", "predicted_exports"]]
        .sum()
        .reset_index()
        .melt(id_vars = "origin", value_vars = ["exports_vol", "baseline_exports", "predicted_exports"],
              var_name = "type", value_name = "value")
    )

    # rename labels
    agg_tradevol["type"] = agg_tradevol["type"].map({
        "exports_vol": "Actual",
        "baseline_exports": "Baseline",
        "predicted_exports": "Expected"
    })
    
    bar = px.bar(agg_tradevol, x="origin", y="value", color="type", barmode = "group",
                title = "Comparison of Export Volumes (2015-2024)",
                labels = {"origin": "Origin Country", "value": "Total Export Volume (USD) (log scale)", "type": "Trade Volume Type"},
                color_discrete_map={
                    "Actual": "#7DA8FF",
                    "Baseline": "#F9D97A",
                    "Expected": "#C6A0FF"
                }
    )
    
    bar.update_layout(
        margin = dict(l=0, r=0, t=60, b=20), 
        legend=dict(title_text = None, orientation="h", yanchor="bottom", y=0.91, x=0.465, xanchor="center", font=dict(size=15), itemwidth=50),
        title = dict(font_size=20, x=0.5, xanchor="center", y=0.97, yanchor="top")
    )

    bar.update_yaxes(type = "log", tickvals=[1e11, 1e12], ticktext=["100B", "1T"])

    st.plotly_chart(bar, use_container_width=True)

    st.info("""
            - Both China and the USA significantly under-trade relative to model expectations.
            - Singapore shows the largest proportional gap relative to its trade size.
            """)

    st.markdown("---")

    # Risk Index info
    st.markdown("### The Risk Index")

    # Compute quartile thresholds from full dataset
    q25_global = np.percentile(df["risk_index"].dropna(), 25)
    q75_global = np.percentile(df["risk_index"].dropna(), 75)

    st.markdown(f"""
    The risk index is adapted from the Geopolitical Annual Trade Risk Index (GATRI), a metric that integrates geopolitical tensions and global trade dynamics to quantify global trade risk. In this application, the risk index is constructed at the bilateral level, indicating the level of trade risk between two countries. \n
    The risk index incorporates economic, political and security-related indicators in its calculation, such as transport costs, ideal point distance, violent events and more. To customise these indicators, navigate to the **Indicators tab**.\n
    The value of the risk index ranges from 0 to 100. A lower risk index value suggests greater trade compatibility, while a higher value reflects a higher probability of exposure to trade-related risks. Risk thresholds are quartile-based: Low Risk (≤ {q25_global:.0f}), Medium Risk ({q25_global:.0f}–{q75_global:.0f}), High Risk (> {q75_global:.0f}).\n
    - **0 – {q25_global:.0f} (bottom 25%):** Low Risk
    - **{q25_global:.0f} – {q75_global:.0f} (middle 50%):** Medium Risk
    - **{q75_global:.0f} – 100 (top 25%):** High Risk
    """, unsafe_allow_html=True)

    hist = px.histogram(
       df,
       x="risk_index",
       title="Distribution of Risk Index Values (2021)"
    )

    hist.update_layout(
        xaxis_title = "Risk Index",
        yaxis_title = "Number of Bilateral Trade Relationships",
        margin = dict(l=0, r=0, t=40, b=0),
        title = dict(font_size=20, x=0.5, xanchor="center", y=0.97, yanchor="top")
    )

    hist.update_traces(
        marker_color = "#7DA8FF",
        opacity = 1,
        marker_line_width = 1,
        marker_line_color = "#2E2E2E",
        xbins=dict(start=0, end=100, size=5)
    )

    # Quartile-based vlines and shading
    hist.add_vline(x=q25_global, line_dash="dash", line_color="grey",
                   annotation_text=f"Q1 ({q25_global:.0f})", annotation_position="top right")
    hist.add_vline(x=q75_global, line_dash="dash", line_color="grey",
                   annotation_text=f"Q3 ({q75_global:.0f})", annotation_position="top right")
    hist.add_vrect(x0=0, x1=q25_global, fillcolor="rgba(39, 174, 96, 0.12)", line_width=0, layer="below")
    hist.add_vrect(x0=q25_global, x1=q75_global, fillcolor="yellow", opacity=0.1, line_width=0, layer="below")
    hist.add_vrect(x0=q75_global, x1=100, fillcolor="rgba(231, 76, 60, 0.12)", line_width=0, layer="below")

    hist.update_xaxes(dtick=10)

    st.plotly_chart(hist, use_container_width=True)

    st.info("The distribution is left-skewed, with most bilateral trade relationships falling in higher risk ratings.")

    st.markdown("---")
    
    # Data info
    st.markdown("### About the Data")
    st.markdown("This application utlises data from 2015 to 2024 for the building of models. For consistency and data completeness, the visualisations and analysis presented in this dashboard are based only on 2021 data.")

    st.markdown("#### Data Sources")

    st.markdown("""
    | Indicator | Specific Aspect Measured | Source |
    |----------|------------------------|--------|
    | GDP | GDP in USD (2015 prices) | [World Bank](https://data.worldbank.org/indicator/NY.GDP.MKTP.KD) |
    | Population | Yearly population count | World Bank | 
    | Tariff |  | [WTO](https://ttd.wto.org/en/profiles/singapore) | 
    | Exports | FOB export value | [UN Comtrade](https://comtradeplus.un.org/) |
    | Geographical Distance | Distance between capitals | CEPII |
    | Geopolitical Distance | UNGA voting-based ideal point distance | [Erik Voeten Dataverse](https://doi.org/10.7910/DVN/LEJUQZ) | 
    | Foreign Direct Investment | Net direct investment | IMF | 
    | Exchange Rate | End-of-period LCU/USD | [IMF](https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.STA:ER(4.0.1)) |
    | Political Violence Events | Yearly event count | [ACLED](https://acleddata.com/) |
    | Fatalities | Yearly fatalities | [ACLED](https://acleddata.com/) | 
    | State Visits | Bilateral visits per year | [COLT](https://doi.org/10.7910/DVN/HJK7DN) | 
    """)

# -------------------------------
# Map & Charts Tab
# -------------------------------
with tab2:
    # Filters
    col1, col2, col3 = st.columns(3) 

    # Add origin selector
    with col1:
        origin_options = sorted(df["origin"].unique())
        default_origin_idx = origin_options.index("Singapore") if "Singapore" in origin_options else 0
        origin = st.selectbox(
            "Origin Country",
            origin_options,
            index=default_origin_idx,
            key="selected_origin"
        )

    # Region Searchbox
    regions = ["All"] + sorted(df["region"].unique()) # list of regions

    # Region multiselect
    with col2:
        selected_regions = st.multiselect(
            "Region",
            options=regions,
            default=["All"]  # empty = "All"
        )

    # Industry multiselect
    industries = ["All"] + sorted(df["industry"].dropna().unique())

    with col3:
        selected_industries = st.multiselect(
            "Industry",
            options=industries,
            default=["All"]  # empty = "All"
        )
    
    # Show friendly message and stop the script
    if not selected_regions or not selected_industries:
        st.warning("Please select at least one region and one industry.")
        st.stop()
    
    # In case user selectes "All", "Asia", ...
    # Keep just "All" for consistent selection later
    def clean_selection(selection):
        if "All" in selection:
            return ["All"]
        return selection

    # Clean region selection
    selected_regions = clean_selection(selected_regions)

    # Clean industry selection
    selected_industries = clean_selection(selected_industries)

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
                    <span>Legend (Map)</span><span class="legend-arrow" style="font-size:11px;"></span>
                </summary>
                <hr style="margin:6px 0;">
                <div><b>Risk Index:</b> 0–100 (lower = better)</div>
                <div><b>Marker Color:</b> Green = low risk, Yellow = medium risk, Red = high risk</div>
                <div><b>Actual over Expected Export Trade:</b> &lt;100% = untapped trade opportunities, &gt;100% = successful specialisation but may suggest neglect in other markets</div>
                <div><b>Arrow Width:</b> Proportional to trade with Origin Country (% of OC GDP)</div>
                <div>Click on markers for more information</div>
            </details>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------
    # Apply policies (outside tabs so tab1 map always reflects active policies)
    # -------------------------------
    df_sim = apply_policies(df, st.session_state.policies, coef_map)

    #--------------------------------
    # Active risk index indicator
    #--------------------------------
    if risk_col == "custom_risk_index":
        st.info("Custom Risk Index active — rankings based on your selected indicators. Go to the Indicators tab to change selection.")

    #--------------------------------
    # Multiselect trading partners
    #--------------------------------

    # -------------------------------
    # Helpers
    # -------------------------------
    # For UI
    def get_country_list(df, selected_regions,origin):
        df1=df[df["origin"]==origin]
        if "All" in selected_regions:
            return sorted(df1["country_display"].unique())
        return sorted(df1[df1["region"].isin(selected_regions)]["country_display"].unique())
    
    # a weighted average of only the industries that have data
    def compute_scores(df, risk_col):
        df = df.copy()
        df_valid = df.dropna(subset=[risk_col])

        # Recompute weights within the filtered set only
        df_valid = df_valid.copy()
        df_valid["industry_weight"] = (
            df_valid.groupby("country")["exports_vol"]
            .transform(lambda x: x / x.sum())
        )

        grouped = df_valid.groupby("country").apply(
            lambda x: (x[risk_col] * x["industry_weight"]).sum() / x["industry_weight"].sum()
        )

        return grouped.sort_values()

    # Ranking
    def get_top_n(df, risk_col, n=5):
        
        return compute_scores(df, risk_col).head(n).index.tolist()
    
    # -------------------------------
    # Country list + defaults to display on Trading Partners filter
    # -------------------------------
    # controls only countries in user selected regions shows up for user to select
    Clist = get_country_list(df, selected_regions,origin)

    # ensure default is specific to user selected origin country
    # consistent with selected region
    base_df = df_sim[df_sim["origin"] == origin]
    if "All" not in selected_regions:
        base_df = base_df[base_df["region"].isin(selected_regions)]
    
    # best trading partners (lowest risk score)
    top5_countries = get_top_n(base_df, risk_col)
    
    # convert to display names
    # multiselect uses display names
    default_list = (
        base_df.loc[base_df["country"].isin(top5_countries), ["country", "country_display"]]
        .drop_duplicates("country")["country_display"]
        .tolist()
    )
    
    # User sees a list of countries
    # Top 5 are pre-selected
    # User can override
    with col4:
        selected_countries = st.multiselect(
            "Trading Partners", Clist, default=default_list,
            key=f"selected_countries_{risk_col}_{origin}"
        )
            
    # -------------------------------
    # Filtering results
    # -------------------------------
    # Start with origin country
    filtered = df_sim[df_sim["origin"] == origin]
    
    # Filters that user has selected
    # Region filter
    if "All" not in selected_regions:
        filtered = filtered[filtered["region"].isin(selected_regions)]

    # Industry filter
    if "All" not in selected_industries:
        filtered = filtered[filtered["industry"].isin(selected_industries)]

    # User selected countries OR fallback to top 5
    if selected_countries:
        filtered = filtered[filtered["country_display"].isin(selected_countries)]
    else:
        top5 = get_top_n(filtered, risk_col)
        filtered = filtered[filtered["country"].isin(top5)]
        
    
    # -------------------------------
    # Final computation (information to be displayed on map etc)
    # -------------------------------
    # Remove self trade
    df_filtered = filtered[filtered["country"] != origin]

    # Standardise Top 5 to use weighted risk by industry (in line with UI, filtering above)
    top5 = compute_scores(df_filtered, risk_col).head(5).index.tolist()

    # Weighted scores
    country_scores = compute_scores(df_filtered, risk_col).to_dict()

    # Trade metrics
    country_totals = (
        df_filtered.groupby("country")
        .apply(lambda x: pd.Series({
            "imports_pct": ((x["imports_vol"] * x["origin_trade_pct_gdp"]) / x["trade_value"]).sum() * 100,
            "exports_pct": ((x["exports_vol"] * x["origin_trade_pct_gdp"]) / x["trade_value"]).sum() * 100,
            "arrow_width_factor": x["origin_trade_pct_gdp"].sum()
        }), include_groups = False)
        .to_dict("index")
    )

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
    # Compute thresholds from the filtered data BEFORE the marker loop
    scores = list(country_scores.values())
    q25 = np.percentile(scores, 25)
    q75 = np.percentile(scores, 75)

    def get_color(score, q25=q25, q75=q75):
        if score <= q25:
            return "#269E54"   # green - bottom 25% = lowest risk
        elif score <= q75:
            return "#E8BE3F"   # yellow
        else:
            return "#EE4A4D"   # red - top 25% = highest risk
        

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
        tiles="Esri.WorldStreetMap"
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
        risk_message=""
        if country_scores[country]>0:
            risk_message=""
        else:
            risk_message="No risk data found"
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

        # Initialize trade message as empty
        trade_message = ""

        # Case 1: "All" selected → show top 3 industries
        if "All" in selected_industries:
            country_industry_vols = (
                df_filtered[df_filtered["country"] == country]
                .groupby("industry")["trade_value"].sum()
                .sort_values(ascending=False)
                .head(3)
            )

            # Check if there's actually any data in the resulting series
            if country_industry_vols.empty or country_industry_vols.sum() == 0:
                trade_message = "No trade data found"
                top3_rows = ""

            else: 
                top3_rows = "".join([
                    f"<div style='margin-left:8px;'>• {ind}: {vol:,.0f}</div>"
                    for ind, vol in country_industry_vols.items()
                ])
            industry_html = f"<div style='margin-top:4px;'><b>Top 3 Industries (by Trade Value):</b></div>{top3_rows}"

        # Case 2: Specific industries selected → show those industries    
        else:
            all_industry_vols = (
                df_sim[(df_sim["origin"] == origin) & (df_sim["country"] == country)]
                .groupby("industry")["trade_value"].sum()
                .sort_values(ascending=False)
            )

            # Check if these specific industries have any data
            if all_industry_vols.empty or all_industry_vols.sum() == 0:
                trade_message = "No trade data found"

            rows = []
            for ind in selected_industries:
                if ind in all_industry_vols.index:
                    ind_vol = all_industry_vols[ind]
                    rows.append(f"<div style='margin-left:8px;'>• {ind}: {ind_vol:,.0f}</div>")
                else:
                    rows.append(f"<div style='margin-left:8px;'>• {ind}: N/A</div>")
            
            industry_html = (
                f"<div style='margin-top:4px;'><b>Selected Industries:</b></div>"
                + "".join(rows)
                )
               
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; padding: 8px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <img src="{flag_url}" style="width:24px;">
                <span style="font-size:14px; font-weight:600;">{display_country}</span>
            </div>
            
            <hr style="margin:6px 0;">

            <div>Rank: <b>#{rank}</b></div>
            <div>Risk Index: <b>{country_scores[country]:.2f}</b></div>
            <div style="margin-bottom: 3px;"><span style="color:Red;"><b>{risk_message:.30s}</b></div>
            <div>Actual over Expected: <b>{weighted_ae:.0f}%</b></div>

            <div>Imports: <b>{imports_vol:.2f}%</b></div>
            <div>Exports: <b>{exports_vol:.2f}%</b></div>
            <div style="margin-bottom: 3px;"><span style="color:Red;"><b>{trade_message:.30s}</b></div>
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
            "Risk Index": country_scores[country],
            "Actual over Expected": weighted_ae,
            "Imports %": imports_vol,
            "Exports %": exports_vol,
            "industry_html": industry_html,
            "risk_message" : risk_message,
            "trade_message" : trade_message

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
                    <div style="margin-bottom: 3px;"><span style="color:Red;"><b>{data['risk_message']:.30s}</b></div>
                    <div style="margin-bottom: 3px;"> Actual over Expected: <b>{data['Actual over Expected']:.0f}%</b></div>
                    <div style="margin-bottom: 3px;"> Imports: {data['Imports %']:.2f}%</div>
                    <div style="margin-bottom: 3px;">Exports: {data['Exports %']:.2f}%</div>
                    <div style="margin-bottom: 3px;"><span style="color:Red;"><b>{data['trade_message']:.30s}</b></div>
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
            '<div class="subtitle"> Scatter Plot of risk vs trade volume by partner country — points in the green quadrant signal untapped low-risk opportunities, red quadrant warrants caution</div>',
            unsafe_allow_html=True
        )

        # -------------------------------
        # Scatter Plot: Risk vs Trade Volume
        # -------------------------------

        # Use df_sim (policy-applied) and map display names back for labeling
        base = df_sim[df_sim["origin"] == origin].copy()

        # Apply filters
        # Region filter
        if "All" not in selected_regions:
            base = base[base["region"].isin(selected_regions)]

        # Industry filter
        if "All" not in selected_industries:
            base = base[base["industry"].isin(selected_industries)]
        
        # Split usage
        scatter_base = base.copy()

        reference_df = base.copy()  # same filters, just no country restriction

        # Trading Partner filter
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
        # Industry label
        if "All" in selected_industries:
            industry_label = "All Industries"
        elif len(selected_industries) <= 3:
            industry_label = ", ".join(selected_industries)
        else:
            industry_label = f"{len(selected_industries)} Industries"

        # Region label
        if "All" in selected_regions:
            region_label = "All Regions"
        elif len(selected_regions) <= 2:
            region_label = ", ".join(selected_regions)
        else:
            region_label = f"{len(selected_regions)} Regions"

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
    
            # Priotise best opportunity by lowest risk first then lowest trade (true “low-risk opportunity”)
            best_opportunity = None

            if not opportunity.empty:
                best_opportunity = (
                    opportunity.sort_values(["risk_value", "trade_value"])
                    .iloc[0]["display_name"]
                    )
            
            # Industry text
            if "All" in selected_industries:
                industry_str = "across all industries"
            elif len(selected_industries) <= 3:
                industry_str = f"in {', '.join(selected_industries)}"
            else:
                industry_str = f"in {len(selected_industries)} industries"

            # Region text
            if "All" in selected_regions:
                region_str = "globally"
            elif len(selected_regions) <= 3:
                region_str = f"in {', '.join(selected_regions)}"
            else:
                region_str = f"in {len(selected_regions)} regions"
            
            lines = []

            if best_opportunity:
                lines.append(
                    f"**{best_opportunity}** stands out as the top untapped opportunity — "
                    f"below-average risk and trade volume as compared to other partners {industry_str} {region_str}, "
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
with tab3:
    st.markdown("### Customise Risk Index Indicators")
    st.write("Create your own risk index by selecting which indicators to include in the calculation. You may observe how the risk index and partner rankings change in the Map & Charts tab.")
    st.caption("Tip: With all indicators selected, rankings will look similar to the default index (both use the same inputs). Uncheck some indicators to see different results.")

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
DEFAULT_POLICY_PCT = 0.0 # 0% change

policy_vars = [
    "ln_reporter_gdp_per_capita",
    "ln_partner_gdp_per_capita",
    "ln_distcap",
    "ln_tariff",
    "ln_ideal_point_distance"
]

# Initialize session state as % changes
for var in policy_vars:
    if var not in st.session_state:
        st.session_state[var] = DEFAULT_POLICY_PCT

with tab4:
    
    st.markdown("### Trade Policy Simulation")
    st.write("""
    Test how changes in trade conditions affect flows between countries by **adjusting the policy sliders and launching a scenario**. You can add multiple policies to see combined effects.
    The % impact on export trade will be estimated based on the modified gravity model.
 
    What are the variables in the modified gravity model that you can adjust and what do they mean for trade?
    """)
    
    cols = st.columns(5)

    titles = ["Origin GDP per capita", "Partner GDP per capita", "Trade Distance", "Export Tariffs", "Geopolitical Distance"]

    descriptions = [
        "Exporting country’s economic strength and production capacity.",
        "Importing country’s income level and demand for goods.",
        "Proxy for transport costs and geographic barriers.",
        "Taxes that reduce international competitiveness.",
        "Political differences that weaken trade ties."
    ]

    for col, title, desc in zip(cols, titles, descriptions):
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #C6A0FF, #E6D8FF);
                padding: 12px;
                border-radius: 12px;
                font-size: 12px;
                line-height: 1.4;
                color: #1F1F1F;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                height: 100%;
            ">
                <div style="font-weight:700; font-size:13px; margin-bottom:6px;">
                    {title}
                </div>
                {desc}
            </div>
            """, unsafe_allow_html=True)
    st.write("")  # one line gap
    st.write("Results will update in the Map & Charts tab, where you can compare updates to trade, export as % of GDP, and Actual over Expected export flows.")

    if st.session_state.last_news_policy:
        news_title = st.session_state.last_news_policy
        short_title = news_title[:60] + "…" if len(news_title) > 60 else news_title
        st.success(f"📰 Last added from news: *{short_title}*")
        if st.button("Dismiss", key="dismiss_news_banner"):
            st.session_state.last_news_policy = None
            st.rerun()

    col_a, col_b = st.columns(2)
    
    policy_origin = st.session_state.get("selected_origin")
    with col_a:
        st.markdown("#### Add New Policy")
        # Origin comes from Tab 1 selection
        
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox(
                "Origin",
                origin_options,
                index=default_origin_idx,
                key="policy_origin"
            )
        with c2:
            policy_country_display  = st.selectbox("Partner Country", sorted(df["country_display"].unique()), key="policy_country")
            # Convert back to actual country name
            policy_country = display_to_country[policy_country_display]
                
        st.markdown("##### Policy Levers")

        # Row 1 (3 sliders)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.slider("Origin GDP per capita", -50, 100, value=0, step=1, key="ln_reporter_gdp_per_capita")
        
        with r1c2:
            st.slider("Partner GDP per capita", -50, 100, value=0, step=1, key="ln_partner_gdp_per_capita")
            
        with r1c3:
            st.slider("Trade Distance", -50, 100, value=0, step=1, key="ln_distcap")

        # Row 2 (2 sliders)
        r2c1, r2c2, r2c3 = st.columns(3)

        with r2c1:
            st.slider("Export Tariffs", -50, 100, value=0, step=1, key="ln_tariff")

        with r2c2:
            st.slider("Geopolitical Distance", -50, 100, value=0, step=1, key="ln_ideal_point_distance")
        
        with r2c3:
            st.empty() # consistent layout of sliders

        # For easier interpretation for user
        st.markdown("""
        **How to interpret:**

        - `+10%` → 10% increase in explanatory variable 
        - `-20%` → 20% decrease in explanatory variable 
        - `0%` → no change  
        """)

        # Compute log-change using coefficients
        log_effect = 0

        for var in policy_vars:
            pct_change = st.session_state[var]

            # convert to multiplier
            multiplier = 1 + pct_change / 100

            # safety (avoid log(0))
            multiplier = max(multiplier, 1e-6)

            coef = coef_map.get(var, 0)

            log_effect += coef * np.log(multiplier)

        # Convert to % change
        trade_effect = (np.exp(log_effect) - 1) * 100
        
        if st.button("Launch New Policy", use_container_width=True):
                    
            st.session_state.policies.append({
                "origin": policy_origin,
                "country": policy_country,
                "industry": ["All"],
                "policy_vars": {var: st.session_state[var] for var in policy_vars},
                "trade_effect": trade_effect
            })
            st.success(
                f"{len(st.session_state.policies)} "
                f"{'policy' if len(st.session_state.policies) == 1 else 'policies'} currently active. "
                "Refer to the Map & Charts tab to view the changes."
            )

    with col_b:
        n_policies = len(st.session_state.policies)
        st.markdown(f"#### {n_policies} {'Active Policy' if n_policies == 1 else 'Active Policies'}")
        
        if not st.session_state.policies:
            st.info("No active policies yet.")
        else:
            for i, p in enumerate(st.session_state.policies):
                display_country = country_to_display.get(p["country"], p["country"])
                news_tag = " 📰" if p.get("from_news") else ""

                col1, col2 = st.columns([10, 1])

                with col1:
                    origin_display = p['origin'] if p['origin'] is not None else "Unknown"

                    changes = ", ".join([
                        f"{k.replace('ln_', '').replace('_', ' ').title()}: {'+' if v > 0 else ''}{v}%"
                        for k, v in p["policy_vars"].items() if v != 0
                        ])
                    if not changes:
                        changes = "No changes"

                    st.markdown(f"""
                                <div style="padding:10px; margin-bottom:8px; border-radius:8px;
                                    border:1px solid rgba(128,128,128,0.3);
                                    background-color:var(--secondary-background-color);">
                                <b>{origin_display}</b> → <b>{display_country}</b>{news_tag}<br>
                                <span style="font-size:12px; color:gray;">
                                {changes}
                                </span>
                                <br/>
                                <span style="font-size:13px;">
                                Estimated Export Trade Impact: 
                                <b>{p['trade_effect']:+.2f}%</b>
                                </span>
                                </div>
                                """, unsafe_allow_html=True)

                with col2:
                    if st.button("✕", key=f"delete_policy_{i}", help="Remove policy"):
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
                st.markdown(msg["content"])

        st.divider()

        user_input = st.chat_input(
            "e.g. Which country offers the safest trade opportunity?",
            key="chat_input"
        )

        if user_input and user_input.strip():
            st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})
            with st.spinner("Thinking…"):
                try:
                    reply = get_assistant_response(st.session_state.chat_messages)
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state.chat_messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
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

        _lever_labels = {
            "ln_tariff": "Tariff",
            "ln_ideal_point_distance": "Political Distance",
            "ln_distcap": "Distance/Logistics",
            "ln_reporter_gdp_per_capita": "Origin GDP/cap",
            "ln_partner_gdp_per_capita": "Partner GDP/cap",
        }

        for i, article in enumerate(display_news):
            date_str = format_date(article["published"])
            title_lower = article["title"].lower()
            detected_in_origins = set(detect_countries_in_text(title_lower, all_origins))
            detected_in_partners = set(detect_countries_in_text(title_lower, all_countries))
            all_detected = detected_in_origins | detected_in_partners
            partner_candidates = detected_in_partners - detected_in_origins
            has_countries = len(detected_in_origins) >= 1 and len(partner_candidates) >= 1

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

            # --- Suggest Policy card ---
            if has_countries:
                pending = st.session_state.pending_news_policy.get(i)

                if pending is None:
                    if st.button("Suggested Policy", key=f"news_suggest_{i}", use_container_width=True):
                        base = extract_policy_from_article(
                            article["title"], all_origins, all_countries, all_industries
                        )
                        with st.spinner("Formulating policy..."):
                            suggestion = get_claude_policy_suggestion(
                                title=article["title"],
                                summary=article.get("summary", ""),
                                origin=base["origin"],
                                partner=base["country"],
                                industry=base["industry"],
                            )
                        if suggestion is not None:
                            st.session_state.pending_news_policy[i] = {
                                "origin": base["origin"],
                                "partner": base["country"],
                                "industry": base["industry"],
                                "levers": {lv: suggestion[lv] for lv in _POLICY_LEVERS},
                                "reasoning": suggestion.get("reasoning", ""),
                                "article_title": article["title"],
                            }
                        else:
                            st.session_state.pending_news_policy[i] = "failed"
                        st.rerun()

                elif pending == "failed":
                    st.warning("Claude could not generate a suggestion for this article.")
                    if st.button("Dismiss", key=f"news_dismiss_fail_{i}", use_container_width=True):
                        del st.session_state.pending_news_policy[i]
                        st.rerun()

                else:
                    lever_lines = "&nbsp;&nbsp;".join(
                        f"<b>{_lever_labels[lv]}</b>: {'+' if v > 0 else ''}{v}%"
                        for lv, v in pending["levers"].items()
                        if v != 0
                    ) or "No policies suggested"

                    st.markdown(f"""
<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.15);
            border-radius:8px;padding:10px;margin-bottom:6px;font-size:12px;">
  <div style="font-weight:700;margin-bottom:4px;">
    📋 {pending['origin']} → {pending['partner']} | {pending['industry']}
  </div>
  <div style="opacity:0.75;margin-bottom:6px;font-style:italic;">{pending['reasoning']}</div>
  <div style="opacity:0.85;">{lever_lines}</div>
</div>
""", unsafe_allow_html=True)

                    col_apply, col_dismiss = st.columns(2)
                    with col_apply:
                        if st.button("Apply", key=f"news_apply_{i}", use_container_width=True):
                            levers = pending["levers"]
                            log_effect = sum(
                                coef_map.get(lv, 0) * np.log(max(1 + pct / 100, 1e-6))
                                for lv, pct in levers.items()
                            )
                            trade_effect = (np.exp(log_effect) - 1) * 100
                            st.session_state.policies.append({
                                "origin": pending["origin"],
                                "country": pending["partner"],
                                "industry": pending["industry"],
                                "policy_vars": levers,
                                "trade_effect": trade_effect,
                                "from_news": pending["article_title"],
                            })
                            st.session_state.last_news_policy = pending["article_title"]
                            del st.session_state.pending_news_policy[i]
                            try:
                                st.toast(
                                    f"Policy applied: {pending['origin']} → {pending['partner']} | {pending['industry']}"
                                    
                                )
                            except Exception:
                                pass
                            st.rerun()

                    with col_dismiss:
                        if st.button("Dismiss", key=f"news_dismiss_{i}", use_container_width=True):
                            del st.session_state.pending_news_policy[i]
                            st.rerun()

            st.markdown('<hr style="margin:6px 0; opacity:0.2;">', unsafe_allow_html=True)


