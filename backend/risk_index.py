import pandas as pd
import requests
from io import StringIO, BytesIO
from easyDataverse import Dataverse
import os
import time
import io
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from prepare_data import *

risk_columns = [
    "transptCost",
    "fxChange",
    "IdealPointDistance",
    "stateVisits",
    "repFatalities",
    "repEvents",
    "partFatalities",
    "partEvents",
    "totalFdi"
]

direction_map = {
    "stateVisits": -1,   
}

log_cols = [
    "repFatalities",
    "repEvents",
    "partFatalities",
    "partEvents"
]


def pca_risk_index(
    df,
    risk_columns,
    direction_map=None,
    log_transform_cols=None,
    anchor_variable=None,
    return_all=True
):

    df_out = df[risk_columns].copy()
    # Log transform 
    if log_transform_cols is not None:
        for col in log_transform_cols:
            if col in df_out.columns:
                df_out[col] = np.log1p(df_out[col])

    # Align direction 
    if direction_map is not None:
        for col, sign in direction_map.items():
            if col in df_out.columns:
                df_out[col] = df_out[col] * sign

    # Drop missing
    df_out = df_out.dropna()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_out)

    # PCA
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(X_scaled).flatten()
    loadings = pca.components_[0]

    # Fix PCA sign 
    if anchor_variable is not None and anchor_variable in df_out.columns:
        anchor_idx = df_out.columns.get_loc(anchor_variable)
        if loadings[anchor_idx] < 0:
            pc1_scores *= -1
            loadings *= -1
    else:
        # fallback: majority sign
        if np.sum(loadings) < 0:
            pc1_scores *= -1
            loadings *= -1

    # Weighted indexes
    X_scaled_df = pd.DataFrame(
        X_scaled, columns=df_out.columns, index=df_out.index
    )

    weighted_df = X_scaled_df.mul(loadings, axis=1)
    risk_index_raw = weighted_df.sum(axis=1)

    # Normalize to 0-100
    risk_index = 100 * (
        (risk_index_raw - risk_index_raw.min()) /        
        (risk_index_raw.max() - risk_index_raw.min())
    )

    # Output
    df_result = df.loc[df_out.index].copy()

    # Add weighted columns
    for col in weighted_df.columns:
        df_result[f"{col}_weighted"] = weighted_df[col]

    df_result["Risk_Index_Raw"] = risk_index_raw
    df_result["Risk_Index_Normalized"] = risk_index

    weights = pd.Series(loadings, index=df_out.columns)
    explained_var = pca.explained_variance_ratio_[0]

    if return_all:
        return df_result, weights, explained_var
    else:
        return df_result

def equal_weight_risk_index(
    df,
    risk_columns,
    direction_map=None,
    log_transform_cols=None,
    normalize=True
):

    df_out = df[risk_columns].copy()

    # Log transform 
    if log_transform_cols is not None:
        for col in log_transform_cols:
            if col in df_out.columns:
                df_out[col] = np.log1p(df_out[col])

    # Align direction 
    if direction_map is not None:
        for col, sign in direction_map.items():
            if col in df_out.columns:
                df_out[col] = df_out[col] * sign

    # Drop missing 
    df_out = df_out.dropna()

    # Standardize 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_out)

    # Equal-weight index
    index = X_scaled.mean(axis=1)

    # Normalize
    if normalize:
        index = 100 * (
            (index - index.min()) /
            (index.max() - index.min())
        )

    # Output
    df_result = df.loc[df_out.index].copy()
    df_result["Equal_Weight_Risk_Index"] = index

    return df_result

if __name__ == "__main__":
    df_exchange_rate = df_exchange_rate_clean()
    df_geopolitical_dist = df_geopolitical_dist_clean()
    df_visits_count = df_state_visits_clean()
    df_fdi = df_fdi_clean()


    df_transport_costs_sg = df_transport_costs_clean("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/OECD.SDD.TPS%2CDSD_ITIC%40DF_ITIC%2C1.1%2BSGP...._T.A..csv")
    df_transport_costs_sg

    df_transport_costs_chn = df_transport_costs_clean("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/OECD.SDD.TPS%2CDSD_ITIC%40DF_ITIC%2C1.1%2BCHN...._T.A..csv")
    df_transport_costs_chn

    df_transport_costs_us = df_transport_costs_clean('https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/OECD.SDD.TPS%2CDSD_ITIC%40DF_ITIC%2C1.1%2BUSA...._T.A..csv')
    df_transport_costs_us

    df_transport_costs_jpn = df_transport_costs_clean("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/OECD.SDD.TPS%2CDSD_ITIC%40DF_ITIC%2C1.1%2BJPN...._T.A..csv")
    df_transport_costs_jpn

    df_transport_costs_deu = df_transport_costs_clean("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/OECD.SDD.TPS%2CDSD_ITIC%40DF_ITIC%2C1.1%2BDEU...._T.A..csv")
    df_transport_costs_deu

    df_transport_costs = pd.concat([
        df_transport_costs_sg,
        df_transport_costs_chn,
        df_transport_costs_us,
        df_transport_costs_jpn,
        df_transport_costs_deu
    ]).reset_index(drop=True)
    df_transport_costs

    # Merge transport costs df base with exchange rates  
    df_0 = df_transport_costs.merge(df_exchange_rate, left_on=['REF_AREA', 'TIME_PERIOD'], right_on=['COUNTRY ISO', 'TIME_PERIOD'], how='left')
    df_0 = df_0.merge(df_exchange_rate, left_on=['COUNTERPART_AREA', 'TIME_PERIOD'], right_on=['COUNTRY ISO', 'TIME_PERIOD'], how='left')
    df_0 = df_0.rename(columns={'Exchange_Rate_Pct_Change_x': 'REF Exchange Rate/USD Pct Change','Exchange_Rate_Pct_Change_y':'Counterpart Exchange Rate/USD Pct Change'})
    df_0['COUNTERPART/REF Exchange Pct Change'] = df_0['Counterpart Exchange Rate/USD Pct Change']-df_0['REF Exchange Rate/USD Pct Change']
    df_0 = df_0.drop(columns=['COUNTRY_y','COUNTRY_x','COUNTRY ISO_y','COUNTRY ISO_x','Exchange Rate_x','Exchange Rate_y','REF Exchange Rate/USD Pct Change','Counterpart Exchange Rate/USD Pct Change'])

    # Merge in geopolitical distance
    df_1 = df_0.merge(df_geopolitical_dist, left_on=['REF_AREA', 'COUNTERPART_AREA','TIME_PERIOD'],right_on = [ 'iso3_country1', 'iso3_country2','year'], how='left')
    df_1 = df_1.drop(columns=['iso3_country1',	'cname_country1',	'iso3_country2',	'cname_country2',	'year','session'])
    df_1 = df_1.dropna()

    # Merge in state visits count
    df_2 = df_1.merge(df_visits_count, left_on=['REF_AREA', 'COUNTERPART_AREA','TIME_PERIOD'], right_on=['Country_ISO_1', 'Country_ISO_2','TripYear'], how='left')
    df_2 = df_2.drop(columns=['Country_ISO_1','Country_ISO_2','TripYear'])
    df_2['Total_Visits'] = df_2['Total_Visits'].fillna(0)

    # Merge in military conflict data for reporter and partner
    df_3 = df_2.merge(df_military_clean(), left_on=['REF_AREA','TIME_PERIOD'], right_on=['country_ISO', 'YEAR'], how='left').rename(columns={'total_fatalities': 'reporter_fatalities','total_Events':'reporter_events'})
    df_3 = df_3.drop(columns=['country_ISO','YEAR'])
    df_3 = df_3.sort_values(by="TIME_PERIOD")

    df_4 = df_3.merge(df_military_clean(), left_on=['COUNTERPART_AREA','TIME_PERIOD'], right_on=['country_ISO', 'YEAR'], how='left').rename(columns={'total_fatalities': 'partner_fatalities','total_Events':'partner_events'})
    df_4 = df_4.drop(columns=['country_ISO','YEAR'])
    df_4 = df_4.dropna()

    # Merge in FDI data (total inward + outward)
    df_5 = df_4.merge(df_fdi, left_on=['REF_AREA', 'COUNTERPART_AREA','TIME_PERIOD'], right_on=['COUNTRY.ID', 'COUNTERPART_COUNTRY.ID','TIME_PERIOD'], how='left')
    df_5 = df_5.drop(columns=['COUNTRY.ID' , 'COUNTERPART_COUNTRY.ID' , 	'Inward FDI' ,	'Outward FDI'])

    df_final = df_5.dropna()
    df_final = df_final.rename(columns={
        'REF_AREA': 'reporter_iso',
        'COUNTERPART_AREA': 'partner_iso',
        'TIME_PERIOD': 'year',
        'Transportation Cost' : 'transptCost',
        'COUNTERPART/REF Exchange Pct Change': 'fxChange',
        'Total_Visits': 'stateVisits',
        'reporter_fatalities': 'repFatalities',
        'reporter_events': 'repEvents',
        'partner_fatalities': 'partFatalities',
        'partner_events': 'partEvents',
        "total_fdi": "totalFdi"
    })

    df_equal_weight_risk = equal_weight_risk_index(
    df_final,
    risk_columns=risk_columns,
    direction_map=direction_map,
    log_transform_cols=log_cols)
    
    df_pca_risk, weights, explained_var  = pca_risk_index(
    df_final,
    risk_columns=risk_columns,
    direction_map=direction_map,
    log_transform_cols=log_cols,
    anchor_variable="partner_fatalities")

    print(df_equal_weight_risk)
    print(weights)
    print(explained_var)
    print(df_pca_risk)

    os.makedirs("./backend/temp_df", exist_ok=True)
    df_pca_risk.to_parquet("./backend/temp_df/df_risk.parquet")




