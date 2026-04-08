# Geopolitical Trade Analytics: Gravity Modeling and PCA-Based Risk Index
## Overview
Global trade has become increasingly shaped by geopolitical tensions and shifting alliances. This project extends the gravity model of trade by incorporating geopolitical alignment and other risk factors to better explain bilateral trade relationships.

An interactive dashboard is developed to help users:
- Visualise trade risk
- Identify vulnerabilities and opportunities
- Support trade policy decisions

## Setup & Run
The interactive dashboard can be accessed directly via this [link](https://tariffied.streamlit.app/).

To clone the repository and run the application on your machine:
```bash
git clone https://github.com/ZhengXinning/dse3101-tariffied.git
cd <your-directory-path>\dse3101-tariffied
pip install -r requirements.txt
streamlit run "C:\Users\<your-directory-path>\dse3101-tariffied\frontend\map_with_features.py"
```

## Repository Structure
.
├── data/
│   └── (raw datasets)
├── frontend/
│   ├── Map.py          
│   ├── chart + filters.py     
│   ├── check_geojson.py
│   ├── filters.py
│   ├── map_with_features.py     # Dashboard / entry point
│   └── world_countries.json    
├── backend/
│   ├── prepare_data.py          # Data cleaning and transformations
│   ├── gravity_model.py         # Gravity model
│   ├── gravity_model_pred.py    # Predictions based on trained model
│   ├── risk_index.py            # PCA risk index function
│   ├── final_df.py              # Outputs comprehensive df for frontend processing
│   └── temp_df/                 # Stores all intermediate and final df as parquet    
├── .devcontainer/                   
├── .gitignore                 
├── requirements.txt
└── README.md

Users could directly run `map_with_features.py` to access dashboard on localhost as per instructions above, using our team's final models and dataframes.

Should users wish to include modifications and rerun on local machine, the order of execution of files is as follow:
- `prepare_data.py` ensures correct data handling, cleaning and transformations.
- `gravity_model.py` runs gravity model and outputs model coefficients `df_coef.parquet` to frontend.
- `gravity_model_pred.py` generates prediction.
- `risk_index.py` runs pca to generate risk index.
- `final_df.py` combines updated results from above and outputs updated `df_final.parquet` to frontend.
- `map_with_features.py` entry point of dashboard on localhost.

## The Gravity Model
Our project builds on the gravity model of trade, which estimates trade flows between two countries based on their respective economic sizes and distance between them in its simplest theoretical form. We extend this gravity model by incorporating additional determinants of trade relationships, with a particular focus on geopolitical alignment. 

Export flows are estimated using a log-linear OLS regression based on a standard gravity model framework. The dependent variable is bilateral export flow between countries. Following the baseline gravity model, our independent variables are GDP per capita, population, export tariffs, and geographical distance. We also incorporate sector and year fixed effects to control for time-specific shocks and industry-level differences. 

To construct the augmented gravity model, geopolitical distance is included as a variable by using UN General Assembly voting patterns to calculate Ideal Point Estimates. This quantifies the similarity of geopolitical alignment between countries, with a smaller ideal point distance indicating similar interests. 

`gravity_model.py`
```python
formula = (
        "ln_exportflow ~ ln_reporter_gdp_per_capita + ln_partner_gdp_per_capita + ln_distcap + "
        "ln_tariff + ln_repPop + ln_partPop + ln_ideal_point_distance"
        "+ C(cmdCode) + C(refYear) + C(reporterISO) + C(partnerISO)"
    )
```

The augmented gravity model is used to generate fitted values of bilateral export flows by refitting raw data `df_comb.parquet` as data for prediction in `gravity_model_pred.py`. Predicted trade is obtained by exponentiating the estimated log-linear model. We then compared actual over predicted trade flows to construct a trade gap measure (‘tradeRatio’). Values above 1 indicate “overtrading,” where observed actual trade exceeds model predictions, while values below 1 indicate “undertrading.” These values are calculated and stored together with raw data as a new `df_gravity.parquet` file. The ratio allows us to identify country pairs whose trade relationships are stronger or weaker, thereby highlighting potential inefficiencies and areas where trade relationships could be further developed. 

## The Risk Index
The risk index, which measures the trade risk between country pairs is a composite indicator derived from economic, political, and military variables, constructed using Principal Component Analysis to derive objective weights based on component variance. We have engineered this to be user-centric, so users can customize the index by toggling variables based on specific trade policy priorities in the dashboard.

PCA finds the direction (principal component) that captures the most variation across all variables. A variable with a higher absolute weight in this component contributes more to that main pattern of variation. Therefore, a higher weight means changes in that variable have a stronger effect on the risk index, as it drives more of the overall variation that the index represents.

The details of PCA construction can be located in `risk_index.py`. Each indicator is first normalised, with PCA conducted on the entire dataset to identify weights for each indicator. The signs of the weights are adjusted to meet empirical expectations. The composite index is then computed as a weighted sum. The resulting index is stored in `df_pca_risk.parquet`, together with the individual weighted component values and the associated dataset. This dataset is then merged with `df_gravity.parquet` in `final_df.py`, alongside other transformations, and output to the frontend as `df_final.parquet`.



