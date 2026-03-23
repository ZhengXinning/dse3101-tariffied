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

# fetch data from a given URL, specifically handling GitHub raw file URLs. 
def fetch_data(url):
    url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data from {url}. Status code: {response.status_code}")
    return BytesIO(response.content)


# clean export data
def clean_export(exporturl,
          relevant_columns=['refYear', 'reporterCode', 'reporterISO', 'reporterDesc', 'partnerCode', 'partnerISO', 'partnerDesc', 'cmdCode', 'cmdDesc'],
          export_val='fobvalue'):

    # Fetch export and import data from the provided URL
    export_data = fetch_data(exporturl)

    # Read export and import data into a pandas DataFrame, selecting relevant columns and the export value column
    df_export = pd.read_csv(export_data, encoding = 'latin1', index_col = 0)[relevant_columns + [export_val]]

    # Drop rows that contain any NaN values, which typically result from unmatched entries in the outer merge
    df_export.dropna(inplace = True)

    # Reset the DataFrame index after dropping rows, ensuring a clean sequential index
    df_export.reset_index(drop = True, inplace = True)

    # Rename column
    df_export = df_export.rename(columns={'fobvalue':'exportFlow'})

    return df_export

# clean gdp data
def df_gdp_clean():
  df_gdp = pd.read_excel(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/GDP%20data.xls"),sheet_name="Data", skiprows=3, usecols= [0,1] +list(range(59,69)), index_col=0)
  df_gdp = df_gdp.reset_index()
  df_gdp = df_gdp.melt(id_vars=['Country Name', 'Country Code'], var_name='year', value_name='gdp')
  df_gdp.dropna(inplace = True)
  df_gdp['year'] = df_gdp['year'].astype(int)
  return df_gdp

# clean population data
def df_pop_clean():
  df_population = pd.read_excel(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/Population.xls"),skiprows=3, usecols= [0,1] +list(range(59,69)))
  df_population = df_population.melt(id_vars=['Country Name', 'Country Code'], var_name='year', value_name='population')
  df_population.dropna(inplace = True)
  df_population['year'] = df_population['year'].astype(int)
  return df_population

# clean geographical distance data
def df_geogdist_clean():
  df_geogdist = pd.read_excel(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/Geographical%20Distance%20(dist_cepii).xls"),
                            engine='calamine',usecols= [0,1]+list(range(10,14)), na_values='.', index_col=[0,1])
  cols = df_geogdist.select_dtypes(include='object').columns
  for a in cols:
    if a not in ['iso_d','iso_o']:
      print(a)
      df_geogdist[a] = df_geogdist[a].astype('float64')
  df_geogdist.reset_index(inplace = True)
  df_geogdist.dropna(inplace = True)
  return df_geogdist

# clean tariff data
def df_tariff_clean(url):
  df_tariff = pd.read_csv(fetch_data(url), encoding = 'latin1', index_col = 0)
  df_tariff = df_tariff.groupby(['Reporting Economy Code', 'Reporting Economy ISO3A Code', 'Reporting Economy', 'Partner Economy Code', 'Partner Economy ISO3A Code', 'Partner Economy', 'Year', 'HS2'])['Value'].mean().reset_index()
  df_tariff.dropna(inplace = True)
  return df_tariff

# clean centroid coordinates data
def df_centroid_coords_clean():
  df_coords = pd.read_csv(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/Country%20ISO%20Codes%20and%20Centroid%20Coordinates.csv"))
  df_coords['Latitude (average)'] = df_coords['Latitude (average)'].str.replace('"', '').str.strip().astype(float)
  df_coords['Longitude (average)'] = df_coords['Longitude (average)'].str.replace('"', '').str.strip().astype(float)
  df_coords['Alpha-3 code'] = df_coords['Alpha-3 code'].str.replace('"', '').str.strip().astype(str)
  df_coords['Alpha-2 code'] = df_coords['Alpha-2 code'].str.replace('"', '').str.strip().astype(str)
  df_coords['Numeric code'] = df_coords['Numeric code'].str.replace('"', '').str.strip().astype(int)

  return df_coords[['Alpha-3 code','Latitude (average)','Longitude (average)']]

# clean geopolitical distance
def df_geopolitical_dist_clean():

    df_isocow_ccode = pd.read_csv(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/cow2iso.csv"), encoding='latin1')
    df_isocow_ccode = df_isocow_ccode[(df_isocow_ccode['valid_until'].isna())| (df_isocow_ccode['valid_until'] >= 2015)]

    df_isocow_ccode['cow_id'] = df_isocow_ccode['cow_id'].astype('Int64')
    df_isocow_ccode['iso_id'] = df_isocow_ccode['iso_id'].astype('Int64')

    #Setting the index to cow_id and cow3 for easier merging later on
    df_isocow_ccode.set_index(["cow_id", "cow3"], inplace=True)
    #print("COW to ISO code mapping DataFrame of shape "+str(df_isocow_ccode.shape)+" has been loaded successfully!")


    #using harvard dataverse
    #geopo_dist_agreement_scores_df= fetch_geopolitical_dist_dataverse()
    geopo_dist_agreement_scores_df = pd.read_csv(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/agreementnewfiltered.csv"))

    geopo_dist_agreement_scores_df = geopo_dist_agreement_scores_df[geopo_dist_agreement_scores_df['year'] >= 2015]



    geopo_dist_agreement_scores_df = geopo_dist_agreement_scores_df.merge(df_isocow_ccode,
                                     left_on=["ccode1"],
                                    right_on=["cow_id"],
                                    how="left",
                                    suffixes=("", "_country1")
                                    ).merge(df_isocow_ccode,
                                     left_on=["ccode2"],
                                    right_on=["cow_id"],
                                    how="left",
                                    suffixes=("", "_country2")
                                    )
    # Only using relevant columns
    columns_geopo_agreement=[
        'session.x',
            'iso3',
            'cname',
            'iso3_country2',
            'cname_country2',
            'year',
            'IdealPointDistance'
    ]
    cleaned_geopo_dist_agreement_scores_df = geopo_dist_agreement_scores_df[columns_geopo_agreement].copy()


    # Rename Columns
    cleaned_geopo_dist_agreement_scores_df.rename(columns={'iso3': 'iso3_country1',
                                                           'cname': 'cname_country1',
                                                           'session.x': 'session'
                                                           }, inplace=True)

    # Filtering out na values
    cleaned_geopo_dist_agreement_scores_df.dropna(inplace=True)

    return cleaned_geopo_dist_agreement_scores_df

