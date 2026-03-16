import pandas as pd
import numpy as np

np.random.seed(42)

# Singapore GDP (roughly, for % calculations)
SG_GDP = 574_000  # in millions

countries = [
"China","USA","Malaysia","Indonesia","Japan","South Korea","India","Germany","France",
"United Kingdom","Australia","Thailand","Vietnam","Philippines","Netherlands","Italy",
"Spain","Brazil","Mexico","Canada","Turkey","Saudi Arabia","UAE","South Africa",
"Russia","Poland","Sweden","Norway","Denmark","Finland","Switzerland","Austria",
"Belgium","Ireland","Portugal","Chile","Argentina","Colombia","Peru","New Zealand"
]

regions = {
"China":"APAC","USA":"NA","Malaysia":"APAC","Indonesia":"APAC","Japan":"APAC",
"South Korea":"APAC","India":"APAC","Germany":"EU","France":"EU","United Kingdom":"EU",
"Australia":"APAC","Thailand":"APAC","Vietnam":"APAC","Philippines":"APAC",
"Netherlands":"EU","Italy":"EU","Spain":"EU","Brazil":"LATAM","Mexico":"LATAM",
"Canada":"NA","Turkey":"EU","Saudi Arabia":"ME","UAE":"ME","South Africa":"AF",
"Russia":"EU","Poland":"EU","Sweden":"EU","Norway":"EU","Denmark":"EU","Finland":"EU",
"Switzerland":"EU","Austria":"EU","Belgium":"EU","Ireland":"EU","Portugal":"EU",
"Chile":"LATAM","Argentina":"LATAM","Colombia":"LATAM","Peru":"LATAM","New Zealand":"APAC"
}

coordinates = {
"China":(35.86,104.19),"USA":(37.09,-95.71),"Malaysia":(4.21,101.97),
"Indonesia":(-0.79,113.92),"Japan":(36.20,138.25),"South Korea":(35.90,127.77),
"India":(20.59,78.96),"Germany":(51.16,10.45),"France":(46.22,2.21),
"United Kingdom":(55.37,-3.43),"Australia":(-25.27,133.77),"Thailand":(15.87,100.99),
"Vietnam":(14.06,108.28),"Philippines":(12.88,121.77),"Netherlands":(52.13,5.29),
"Italy":(41.87,12.57),"Spain":(40.46,-3.75),"Brazil":(-14.23,-51.92),
"Mexico":(23.63,-102.55),"Canada":(56.13,-106.35),"Turkey":(38.96,35.24),
"Saudi Arabia":(23.88,45.08),"UAE":(23.42,53.85),"South Africa":(-30.56,22.94),
"Russia":(61.52,105.31),"Poland":(51.92,19.15),"Sweden":(60.13,18.64),
"Norway":(60.47,8.47),"Denmark":(56.26,9.50),"Finland":(61.92,25.75),
"Switzerland":(46.82,8.23),"Austria":(47.52,14.55),"Belgium":(50.50,4.47),
"Ireland":(53.41,-8.24),"Portugal":(39.40,-8.22),"Chile":(-35.68,-71.54),
"Argentina":(-38.42,-63.62),"Colombia":(4.57,-74.30),"Peru":(-9.19,-75.02),
"New Zealand":(-40.90,174.88)
}

industries = [
"Electronics",
"Automotive",
"Chemicals",
"Agriculture",
"Energy",
"Machinery",
"Pharmaceuticals",
"Financial Services"
]

rows = []

for country in countries:
    
    base_trade = np.random.uniform(5_000, 80_000)
    compatibility = np.random.uniform(0.4, 0.95)
    
    for industry in industries:
        
        industry_weight = np.random.uniform(0.05, 0.25)
        trade_value = base_trade * industry_weight
        
        exports = trade_value * np.random.uniform(0.4, 0.7)
        imports = trade_value - exports
        
        trade_pct_gdp = trade_value / SG_GDP
        
        lat, lon = coordinates[country]
        
        rows.append({
            "country": country,
            "region": regions[country],
            "industry": industry,
            "exports_sg": round(exports,2),
            "imports_sg": round(imports,2),
            "trade_value": round(trade_value,2),
            "trade_pct_gdp": round(trade_pct_gdp,5),
            "compatibility_score": round(compatibility,3),
            "latitude": lat,
            "longitude": lon,
            "year": 2025
        })

df = pd.DataFrame(rows)
# print(df.head())

df.to_csv("dummy_dataset.csv", index=False)

print("Dataset created:", df.shape)