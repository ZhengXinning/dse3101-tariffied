import pandas as pd
import numpy as np

np.random.seed(42)

# -------------------------------
# GDP (in millions USD)
# -------------------------------
GDP = {
    "Singapore": 574_000,
    "United States of America": 27_000_000,
    "China": 18_000_000
}

# -------------------------------
# Countries list
# -------------------------------
countries = [
"China","United States of America","Malaysia","Indonesia","Japan","South Korea","India","Germany","France",
"United Kingdom","Australia","Thailand","Vietnam","Philippines","Netherlands","Italy",
"Spain","Brazil","Mexico","Canada","Turkey","Saudi Arabia","United Arab Emirates","South Africa", "Singapore",
"Russia","Poland","Sweden","Norway","Denmark","Finland","Switzerland","Austria",
"Belgium","Ireland","Portugal","Chile","Argentina","Colombia","Peru","New Zealand"
]

# -------------------------------
# Regions
# -------------------------------
regions = {
"China":"APAC","United States of America":"NA","Malaysia":"APAC","Indonesia":"APAC","Japan":"APAC",
"South Korea":"APAC","India":"APAC","Germany":"EU","France":"EU","United Kingdom":"EU",
"Australia":"APAC","Thailand":"APAC","Vietnam":"APAC","Philippines":"APAC",
"Netherlands":"EU","Italy":"EU","Spain":"EU","Brazil":"LATAM","Mexico":"LATAM",
"Canada":"NA","Turkey":"EU","Saudi Arabia":"ME","United Arab Emirates":"ME","South Africa":"AF", "Singapore":"APAC",
"Russia":"EU","Poland":"EU","Sweden":"EU","Norway":"EU","Denmark":"EU","Finland":"EU",
"Switzerland":"EU","Austria":"EU","Belgium":"EU","Ireland":"EU","Portugal":"EU",
"Chile":"LATAM","Argentina":"LATAM","Colombia":"LATAM","Peru":"LATAM","New Zealand":"APAC"
}

# -------------------------------
# Coordinates
# -------------------------------
coordinates = {
"China":(35.86,104.19),"United States of America":(37.09,-95.71),"Malaysia":(4.21,101.97),
"Indonesia":(-0.79,113.92),"Japan":(36.20,138.25),"South Korea":(35.90,127.77),
"India":(20.59,78.96),"Germany":(51.16,10.45),"France":(46.22,2.21),
"United Kingdom":(55.37,-3.43),"Australia":(-25.27,133.77),"Thailand":(15.87,100.99),
"Vietnam":(14.06,108.28),"Philippines":(12.88,121.77),"Netherlands":(52.13,5.29),
"Italy":(41.87,12.57),"Spain":(40.46,-3.75),"Brazil":(-14.23,-51.92),
"Mexico":(23.63,-102.55),"Canada":(56.13,-106.35),"Turkey":(38.96,35.24),
"Saudi Arabia":(23.88,45.08),"United Arab Emirates":(23.42,53.85),"South Africa":(-30.56,22.94), "Singapore": (1.3521, 103.8198),
"Russia":(61.52,105.31),"Poland":(51.92,19.15),"Sweden":(60.13,18.64),
"Norway":(60.47,8.47),"Denmark":(56.26,9.50),"Finland":(61.92,25.75),
"Switzerland":(46.82,8.23),"Austria":(47.52,14.55),"Belgium":(50.50,4.47),
"Ireland":(53.41,-8.24),"Portugal":(39.40,-8.22),"Chile":(-35.68,-71.54),
"Argentina":(-38.42,-63.62),"Colombia":(4.57,-74.30),"Peru":(-9.19,-75.02),
"New Zealand":(-40.90,174.88)
}

# -------------------------------
# Industries
# -------------------------------
industries = [
"Electronics","Automotive","Chemicals","Agriculture",
"Energy","Machinery","Pharmaceuticals","Financial Services"
]

# -------------------------------
# Combine different origin countries into one csv
# -------------------------------
all_rows = []

for origin in ["Singapore", "United States of America", "China"]:
    
    origin_gdp = GDP[origin]
    
    for country in countries:
        
        if country == origin:
            continue
        
        base_trade = np.random.uniform(80_000, 300_000)
        risk = np.random.uniform(1, 90)
        
        for industry in industries:
            
            industry_weight = np.random.uniform(0.05, 0.25)
            trade_value = base_trade * industry_weight
            
            exports = trade_value * np.random.uniform(0.4, 0.7)
            imports = trade_value - exports
            
            trade_pct_gdp = trade_value / origin_gdp
            AE = np.random.uniform(50, 150)
            
            lat, lon = coordinates[country]
            
            all_rows.append({
                "origin": origin,
                "country": country,
                "region": regions[country],
                "industry": industry,
                "industry_weight": industry_weight,
                "exports_vol": round(exports,2),
                "imports_vol": round(imports,2),
                "trade_value": round(trade_value,2),
                "trade_pct_gdp": round(trade_pct_gdp,5),
                "actual_vs_expected": round(AE, 3),
                "risk_index": round(risk,3),
                "latitude": lat,
                "longitude": lon,
                "year": 2025
            })

df = pd.DataFrame(all_rows)

df.to_csv("dummy_dataset_global.csv", index=False)

print("Dataset created:", df.shape)