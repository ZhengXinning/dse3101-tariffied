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

from prepare_data import *
import os
import pickle
from datetime import datetime

def df_merge(df_trade, df_tariff):
    df_gdp = df_gdp_clean()
    df_population = df_pop_clean()
    df_geogdist = df_geogdist_clean()
    df_geopoldist = df_geopolitical_dist_clean()
    df_coords = df_centroid_coords_clean()

    # Explicitly convert 'refYear' to int and ISO codes to string for consistency
    try:
        df_trade['refYear'] = df_trade['refYear'].astype(int)
        df_trade['reporterISO'] = df_trade['reporterISO'].astype(str)
        df_trade['partnerISO'] = df_trade['partnerISO'].astype(str)
    except ValueError as e:
        print(f"Error converting initial df_trade types: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure

    common_columns_to_drop = ['Country Name', 'Country Code', 'year']

    # Prepare df_gdp and df_population for merging by ensuring consistent types
    try:
        df_gdp['year'] = df_gdp['year'].astype(int)
        df_gdp['Country Code'] = df_gdp['Country Code'].astype(str)
        df_population['year'] = df_population['year'].astype(int)
        df_population['Country Code'] = df_population['Country Code'].astype(str)
    except ValueError as e:
        print(f"Error converting df_gdp or df_population types: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure

    # Merge Reporter trade and gdp
    try:
        df_merge_trade_repgdp = df_trade.merge(df_gdp,
                                              left_on=['reporterISO', 'refYear'],
                                              right_on=['Country Code', 'year'],
                                              how='left').rename(columns={'gdp': 'reporterGdp'})

        df_merge_trade_repgdp = df_merge_trade_repgdp[df_merge_trade_repgdp['partnerCode']!=0]
        df_merge_trade_repgdp=df_merge_trade_repgdp.drop(columns=common_columns_to_drop)
    except Exception as e:
        print(f"Error merging reporter GDP: {e}")
        return pd.DataFrame()

    # Merge reporter population
    try:
        df_merge_trade_reppop= df_merge_trade_repgdp.merge(df_population,
                                                           left_on=['reporterISO', 'refYear'],
                                                           right_on=['Country Code', 'year'],
                                                           how='left').rename(columns={'population': 'reporterPopulation'})
        df_merge_trade_reppop= df_merge_trade_reppop.drop(columns=common_columns_to_drop)
    except Exception as e:
        print(f"Error merging reporter population: {e}")
        return pd.DataFrame()
    
    # Merge reporter coord
    try:
        df_merge_trade_reppopcoord= df_merge_trade_reppop.merge(df_coords,
                          left_on=['reporterISO'],
                          right_on=['Alpha-3 code'],
                          how='left').rename(columns={'Latitude (average)': 'reporterlat',
                                                      'Longitude (average)': 'reporterlong'})
        df_merge_trade_reppopcoord = df_merge_trade_reppopcoord.drop(columns=['Alpha-3 code'])  
    except Exception as e:
        print(f"Error merging reporter coordinates: {e}")
        return pd.DataFrame()
    

    # Merge partner gdp
    try:
        df_merge_trade_partgdp= df_merge_trade_reppopcoord.merge(df_gdp,
                                              left_on=['partnerISO', 'refYear'],
                                              right_on=['Country Code', 'year'],
                                              how='left').rename(columns={'gdp': 'partnerGdp'})
        df_merge_trade_partgdp= df_merge_trade_partgdp.drop(columns=common_columns_to_drop)
    except Exception as e:
        print(f"Error merging partner GDP: {e}")
        return pd.DataFrame()

    #Merge partner population
    try:
        df_merge_trade_partgdp['refYear'] = df_merge_trade_partgdp['refYear'].astype(int)
        df_merge_trade_partpop= df_merge_trade_partgdp.merge(df_population,
                                                              left_on=['partnerISO', 'refYear'],
                                                              right_on=['Country Code', 'year'],
                                                              how='left').rename(columns={'population': 'partnerPopulation'})
        df_merge_trade_partpop= df_merge_trade_partpop.drop(columns=common_columns_to_drop)
    except ValueError as e:
        print(f"Error converting refYear to int before partner population merge: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error merging partner population: {e}")
        return pd.DataFrame()


    #Merge partner coordinates
    try:
        df_merge_trade_partpopcoord= df_merge_trade_partpop.merge(df_coords,
                          left_on=['partnerISO'],
                          right_on=['Alpha-3 code'],
                          how='left').rename(columns={'Latitude (average)': 'partnerlat',
                                                      'Longitude (average)': 'partnerlong'})
        df_merge_trade_partpopcoord = df_merge_trade_partpopcoord.drop(columns=['Alpha-3 code'])            
   
    except Exception as e:
        print(f"Error merging partner coordinates: {e}")
        return pd.DataFrame()

    #Merge geographical distance
    try:
        df_geogdist['iso_o'] = df_geogdist['iso_o'].astype(str)
        df_geogdist['iso_d'] = df_geogdist['iso_d'].astype(str)
        df_merge_trade_geogdist = df_merge_trade_partpopcoord.merge(df_geogdist,
                                                              left_on=['reporterISO','partnerISO'],
                                                              right_on=['iso_o', 'iso_d'],
                                                              how='left')
        df_merge_trade_geogdist = df_merge_trade_geogdist.drop(columns=['iso_o', 'iso_d'])
    except ValueError as e:
        print(f"Error converting geographical distance ISO types: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error merging geographical distance: {e}")
        return pd.DataFrame()

    #Merge geopolitical distance
    try:
        df_geopoldist['iso3_country1'] = df_geopoldist['iso3_country1'].astype(str)
        df_geopoldist['iso3_country2'] = df_geopoldist['iso3_country2'].astype(str)
        df_geopoldist['year'] = df_geopoldist['year'].astype(int)

        df_merge_trade_geogpol= df_merge_trade_geogdist.merge(df_geopoldist,
                                                              left_on=['reporterISO','partnerISO','refYear'],
                                                              right_on=['iso3_country1', 'iso3_country2','year'],
                                                              how='left')
        df_merge_trade_geogpol = df_merge_trade_geogpol.drop(columns=['iso3_country1',
                                                                      'cname_country1',
                                                                      'iso3_country2',
                                                                      'cname_country2',
                                                                      'year',
                                                                      'session'])
    except ValueError as e:
        print(f"Error converting geopolitical distance types: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error merging geopolitical distance: {e}")
        return pd.DataFrame()

    # Merge tariff data
    try:
        df_tariff['Reporting Economy ISO3A Code'] = df_tariff['Reporting Economy ISO3A Code'].astype(str)
        df_tariff['Partner Economy ISO3A Code'] = df_tariff['Partner Economy ISO3A Code'].astype(str)
        df_tariff['Year'] = df_tariff['Year'].astype(int)
        df_tariff['HS2'] = df_tariff['HS2'].astype(int)

        df_merge_trade_tarr= df_merge_trade_geogpol.merge(df_tariff,
                                                          left_on=['reporterISO','partnerISO','refYear','cmdCode'],
                                                          right_on=['Partner Economy ISO3A Code', 'Reporting Economy ISO3A Code','Year','HS2'],
                                                          how='left').rename(columns={'Value': 'Tariff'})

        df_merge_trade_tarr= df_merge_trade_tarr.drop(columns=['Reporting Economy ISO3A Code',
                                                                'Reporting Economy',
                                                                'Reporting Economy Code',
                                                                'Partner Economy Code',
                                                                'Partner Economy ISO3A Code',
                                                                'Partner Economy',
                                                                'Year',
                                                                'HS2'])
    except ValueError as e:
        print(f"Error converting tariff data types: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error merging tariff data: {e}")
        return pd.DataFrame()

    df_merge_trade = df_merge_trade_tarr[~df_merge_trade_tarr['Tariff'].isna()].copy().reset_index().dropna().drop(columns='index')

    # New column for gdp per capita
    df_merge_trade['reporter_gdp/capita'] = df_merge_trade['reporterGdp'] / df_merge_trade['reporterPopulation']
    df_merge_trade['partner_gdp/capita'] = df_merge_trade['partnerGdp'] / df_merge_trade['partnerPopulation']

    return df_merge_trade


def gravity_model(df_input, predict=False, test_size=0.2, random_state=42):

    # Create a working copy of the dataframe
    model_df = df_input.copy()

    # Drop rows with NaN in IdealPointDistance as it's a key variable for the model
    model_df.dropna(subset=['IdealPointDistance'], inplace=True)

    # Filter out any zero or negative values before taking logs
    model_df = model_df[model_df['exportFlow'] > 0]
    model_df = model_df[model_df['distcap'] > 0] # Using distcap as per description
    model_df = model_df[model_df['reporter_gdp/capita'] > 0]
    model_df = model_df[model_df['partner_gdp/capita'] > 0]
    model_df = model_df[model_df['reporterPopulation'] > 0]
    model_df = model_df[model_df['partnerPopulation'] > 0]

    # Ensure Tariff is non-negative before taking log
    model_df = model_df[model_df['Tariff'] >= 0]

    # Apply logarithmic transformations
    model_df['ln_exportflow'] = np.log(model_df['exportFlow'])
    model_df['ln_reporter_gdp_per_capita'] = np.log(model_df['reporter_gdp/capita'])
    model_df['ln_partner_gdp_per_capita'] = np.log(model_df['partner_gdp/capita'])
    model_df['ln_distcap'] = np.log(model_df['distcap'])
    model_df['ln_repPop'] = np.log(model_df['reporterPopulation'])
    model_df['ln_partPop'] = np.log(model_df['partnerPopulation'])

    # Add small constant to avoid log(0) if IdealPointDistance or Tariff can be 0 or very small
    model_df['ln_ideal_point_distance'] = np.log(model_df['IdealPointDistance'])
    model_df['ln_tariff'] = np.log(model_df['Tariff'] + 0.0001)

    # Split the data into training and testing sets


    # Define the regression formula based on the gravity model
    formula = (
        "ln_exportflow ~ ln_reporter_gdp_per_capita + ln_partner_gdp_per_capita + ln_distcap + "
        "ln_tariff + C(cmdCode) + C(refYear) + ln_ideal_point_distance + ln_repPop + ln_partPop"
    )

    # Build and fit the OLS model on the training data
    if predict:
      train_df, test_df = train_test_split(
          model_df, test_size=test_size, random_state=random_state
      )

      model = smf.ols(formula, data=train_df)
      results = model.fit()

      # Predict on test set (log scale)
      test_df['predicted_ln_exportflow'] = results.predict(test_df)

      # Convert back to original scale (trade flow)
      test_df['predicted_exportflow'] = np.exp(test_df['predicted_ln_exportflow'])

      return results, test_df


    else:
      model = smf.ols(formula, data=model_df)
      results = model.fit()
      return results

#Baseline gravity model WITHOUT geopolitical distance
def base_gravity_model(df_input, predict=False, test_size=0.2, random_state=42):

    # Create a working copy of the dataframe
    model_df = df_input.copy()

    # Filter out any zero or negative values before taking logs
    model_df = model_df[model_df['exportFlow'] > 0]
    model_df = model_df[model_df['distcap'] > 0] # Using distcap as per description
    model_df = model_df[model_df['reporter_gdp/capita'] > 0]
    model_df = model_df[model_df['partner_gdp/capita'] > 0]
    model_df = model_df[model_df['reporterPopulation'] > 0]
    model_df = model_df[model_df['partnerPopulation'] > 0]

    # Ensure Tariff is non-negative before taking log
    model_df = model_df[model_df['Tariff'] >= 0]

    # Apply logarithmic transformations
    model_df['ln_exportflow'] = np.log(model_df['exportFlow'])
    model_df['ln_reporter_gdp_per_capita'] = np.log(model_df['reporter_gdp/capita'])
    model_df['ln_partner_gdp_per_capita'] = np.log(model_df['partner_gdp/capita'])
    model_df['ln_distcap'] = np.log(model_df['distcap'])
    model_df['ln_repPop'] = np.log(model_df['reporterPopulation'])
    model_df['ln_partPop'] = np.log(model_df['partnerPopulation'])

    # Add small constant to avoid log(0) if Tariff can be 0 or very small
    model_df['ln_tariff'] = np.log(model_df['Tariff'] + 0.0001)

    # Split the data into training and testing sets


    # Define the regression formula based on the gravity model
    formula = (
        "ln_exportflow ~ ln_reporter_gdp_per_capita + ln_partner_gdp_per_capita + ln_distcap + "
        "ln_tariff + C(cmdCode) + C(refYear) + ln_repPop + ln_partPop"
    )

    # Build and fit the OLS model on the training data
    if predict:
      train_df, test_df = train_test_split(
          model_df, test_size=test_size, random_state=random_state
      )

      model = smf.ols(formula, data=train_df)
      results = model.fit()

      # Predict on test set (log scale)
      test_df['predicted_ln_exportflow'] = results.predict(test_df)

      # Convert back to original scale (trade flow)
      test_df['predicted_exportflow'] = np.exp(test_df['predicted_ln_exportflow'])

      return results, test_df


    else:
      model = smf.ols(formula, data=model_df)
      results = model.fit()
      return results

# Configuration
countries_config = {
    'SG': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/SGP_ExporttoWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstSGPClean.csv"
    },
    'CHN': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/CHN_ExporttoWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstCHNClean.csv"
    },
    'DEU': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/DEU_ExporttoWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstDEUClean.csv"
    },
    'USA': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/USA_ExporttoWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstUSAClean.csv"
    },
    'JPN': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/JPN_ExporttoWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstJPNClean.csv"
    }
}

# Run baseline model for individual countries at once
def run_baseline_gravity_model(countries_config, save_dir="./backend/gravity_models", predict=False):
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    for country_code, urls in countries_config.items():
        
        try:
            # Load and merge data
            df_export = clean_export(exporturl = urls['export'])
            df_tariff = df_tariff_clean(url = urls['tariff'])
            cleaned_df = df_merge(df_export, df_tariff)
            
            # Run gravity model
            if predict:
                model, test_df = base_gravity_model(cleaned_df, predict=True, test_size=0.3, random_state=42)
            else:
                model = base_gravity_model(cleaned_df, predict=False)
                test_df = None
            
            # Save results
            model_path = os.path.join(save_dir, f"{country_code}_base_model.pickle")
            model.save(model_path)
            
            results[country_code] = {
                'model': model,
                'test_df': test_df,
                'cleaned_df': cleaned_df,
                'model_path': model_path
            }
            
            print(f"{country_code} completed. Model saved to {model_path}")
            
        except Exception as e:
            print(f"Error processing {country_code}: {e}")
            results[country_code] = None
    
    return results

# Run gravity model for individual countries at once
def run_gravity_model(countries_config, save_dir="./backend/gravity_models", predict=False):
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    for country_code, urls in countries_config.items():
        
        try:
            # Load and merge data
            df_export = clean_export(exporturl = urls['export'])
            df_tariff = df_tariff_clean(url = urls['tariff'])
            cleaned_df = df_merge(df_export, df_tariff)
            
            # Run gravity model
            if predict:
                model, test_df = gravity_model(cleaned_df, predict=True, test_size=0.3, random_state=42)
            else:
                model = gravity_model(cleaned_df, predict=False)
                test_df = None
            
            # Save results
            model_path = os.path.join(save_dir, f"{country_code}_model.pickle")
            model.save(model_path)

            results[country_code] = {
                'model': model,
                'test_df': test_df,
                'cleaned_df': cleaned_df,
                'model_path': model_path
            }
            
            print(f"{country_code} completed. Model saved to {model_path}")
            
        except Exception as e:
            print(f"Error processing {country_code}: {e}")
            results[country_code] = None
    
    return results

if __name__ == "__main__":
    all_results_baseline = run_baseline_gravity_model(
        countries_config, save_dir="./backend/gravity_models", predict=False
    )

    all_results = run_gravity_model(
        countries_config, save_dir="./backend/gravity_models", predict=False
    )

    # Combine all cleaned dfs
    combined_dfs = [v['cleaned_df'] for v in all_results.values() if v is not None]
    df_comb = pd.concat(combined_dfs).reset_index(drop=True)

    os.makedirs("./backend/temp_df", exist_ok=True)
    df_comb.to_parquet("./backend/temp_df/df_comb.parquet")

    # Run combined model
    comb_model = gravity_model(df_comb, predict=False)
    comb_model_base = base_gravity_model(df_comb, predict=False)

    print(comb_model_base.summary())
    print(comb_model.summary())

    # Save models
    comb_model.save('./backend/gravity_models/combined_model.pickle')
    comb_model_base.save('./backend/gravity_models/combined_model_base.pickle')