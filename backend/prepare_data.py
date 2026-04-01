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

# clean import data
def clean_import(importurl,
          relevant_columns=['refYear', 'reporterCode', 'reporterISO', 'reporterDesc', 'partnerCode', 'partnerISO', 'partnerDesc', 'cmdCode', 'cmdDesc'],
          import_val='cifvalue'):

    # Fetch export and import data from the provided URL
    import_data = fetch_data(importurl)

    # Read export and import data into a pandas DataFrame, selecting relevant columns and the export value column
    df_import = pd.read_csv(import_data, encoding = 'latin1', index_col = 0)[relevant_columns + [import_val]]

    # Drop rows that contain any NaN values, which typically result from unmatched entries in the outer merge
    df_import.dropna(inplace = True)

    # Reset the DataFrame index after dropping rows, ensuring a clean sequential index
    df_import.reset_index(drop = True, inplace = True)

    # Rename column
    df_import = df_import.rename(columns={'cifvalue':'importFlow'})

    return df_import

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

def df_tariff_clean(url):
    df_tariff = pd.read_csv(fetch_data(url), encoding='latin1', index_col=0)

    df_tariff = (
        df_tariff
        .groupby([
            'Reporting Economy Code',
            'Reporting Economy ISO3A Code',
            'Reporting Economy',
            'Partner Economy Code',
            'Partner Economy ISO3A Code',
            'Partner Economy',
            'Year',
            'HS2'
        ])['Value']
        .mean()
        .reset_index()
    )

    df_tariff.dropna(inplace=True)

    # Create a DataFrame with country names, ISO codes, and ISO codes. 
    # Note that United Kingdom is included here as well, but will be filtered out for years after 2019.
    eu_map = pd.DataFrame({
        "country": [
            "Austria","Belgium","Bulgaria","Croatia","Cyprus","Czechia","Denmark",
            "Estonia","Finland","France","Germany","Greece","Hungary","Ireland",
            "Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands",
            "Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden",
            "United Kingdom"
        ],
        "iso": [
            "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC",
            "HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK",
            "SVN","ESP","SWE","GBR"
        ],
        "iso_code": [
            40, 56, 100, 191, 196, 203, 208, 
            233, 246, 250, 276, 300, 348, 372, 
            380, 428, 440, 442, 470, 528, 
            616, 620, 642, 703, 705, 724, 752,
            826
        ]
    })

    # Split EU vs non-EU (assuming EU appears in Partner Economy)
    df_eu = df_tariff[df_tariff['Reporting Economy'] == 'European Union']
    df_non_eu = df_tariff[df_tariff['Reporting Economy'] != 'European Union']
    
    # Expand EU rows
    df_eu_expanded = df_eu.merge(eu_map, how='cross')

    # Replace with actual countries
    df_eu_expanded['Reporting Economy'] = df_eu_expanded['country']
    df_eu_expanded['Reporting Economy ISO3A Code'] = df_eu_expanded['iso']
    df_eu_expanded['Reporting Economy Code'] = df_eu_expanded['iso_code']

    # Filter out UK for years after 2019 (Brexit)
    df_eu_expanded = df_eu_expanded[~((df_eu_expanded['Reporting Economy'] == 'United Kingdom') & (df_eu_expanded['Year'] > 2019))]

    # Drop helper columns
    df_eu_expanded = df_eu_expanded.drop(columns=['country', 'iso', 'iso_code'])

    # Combine back
    df_tariff = pd.concat([df_non_eu, df_eu_expanded], ignore_index=True)
    
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

# clean iso w region code
def df_isoregion_clean():
    df_isoregion = pd.read_csv(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/iso_country_codes_with_regions.csv"), encoding='latin1')
    df_isoregion = df_isoregion[['alpha-3', 'region']]
    return df_isoregion

# clean iso to cow code
def df_isocow_ccode_clean():
    df_isocow_ccode = pd.read_csv(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/cow2iso.csv"), encoding='latin1')
    df_isocow_ccode = df_isocow_ccode[(df_isocow_ccode['valid_until'].isna())| (df_isocow_ccode['valid_until'] >= 2015)]

    df_isocow_ccode['cow_id'] = df_isocow_ccode['cow_id'].astype('Int64')
    df_isocow_ccode['iso_id'] = df_isocow_ccode['iso_id'].astype('Int64')

    return df_isocow_ccode

# clean geopolitical distance
def df_geopolitical_dist_clean():
    
    df_isocow_ccode = df_isocow_ccode_clean()

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

# clean transport cost data
def df_transport_costs_clean(url):
  df = pd.read_csv(fetch_data(url), encoding = 'latin1', usecols = [4,5,6,7,18,20]).rename(columns={'OBS_VALUE': 'Transportation Cost'})
  df = df.sort_values(by='TIME_PERIOD').reset_index(drop=True)
  return df

# clean exchange rate data
def euro_area_expansion(df):
  eu_map = pd.DataFrame({
      #austria, belgium, finland, france, germany, ireland, italy,
      #luxembourg, netherland, portugal, spain, greece, slovenia, cyprus, malta, slovakia, estonia, latvia, lithuania, croatia
      "country": [
          "Austria","Belgium","Croatia","Cyprus",
          "Estonia","Finland","France","Germany","Greece","Ireland",
          "Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands",
          "Portugal","Slovakia","Slovenia","Spain"
      ],
      "iso": [
          "AUT","BEL","HRV","CYP","EST","FIN","FRA","DEU","GRC",
          "IRL","ITA","LVA","LTU","LUX","MLT","NLD","PRT","SVK",
          "SVN","ESP"
      ]
  })

  # Split EU vs non-EU (assuming EU appears in Partner Economy)
  df_eu = df[df['COUNTRY'] == 'Euro Area (EA)']
  df_non_eu = df[df['COUNTRY'] != 'Euro Area (EA)']

  # Expand EU rows
  df_eu_expanded = df_eu.merge(eu_map, how='cross')

  # Replace with actual countries
  df_eu_expanded['COUNTRY'] = df_eu_expanded['country']
  df_eu_expanded['COUNTRY ISO'] = df_eu_expanded['iso']
  #df_eu_expanded['Reporting Economy Code'] = df_eu_expanded['iso_code']

  # Filter out UK for years after 2019 (Brexit)
  df_eu_expanded = df_eu_expanded[~((df_eu_expanded['COUNTRY ISO'] == 'GBR') & (df_eu_expanded['TIME_PERIOD'] > 2019))] #UK

  #Filter out for countries joining European union
  df_eu_expanded = df_eu_expanded[~((df_eu_expanded['COUNTRY ISO'] == 'HRV') & (df_eu_expanded['TIME_PERIOD'] <= 2022))] #Croatia

  # Drop helper columns
  df_eu_expanded = df_eu_expanded.drop(columns=['country', 'iso'])

  # Combine back
  df = pd.concat([df_non_eu, df_eu_expanded], ignore_index=True)
  return df

def df_exchange_rate_clean():
    df_exchange_rate = pd.read_csv(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/Exchange%20Rate%20per%20USD%202014_2024.csv"), encoding='latin1')
    df_exchange_rate = df_exchange_rate.rename(columns={'OBS_VALUE': 'Exchange Rate','ï»¿"COUNTRY.ID"':'COUNTRY ISO'})
    df_exchange_rate = df_exchange_rate.dropna()
    df_exchange_rate['TIME_PERIOD'] = df_exchange_rate['TIME_PERIOD'].astype(int)
    df_exchange_rate['Exchange_Rate_Pct_Change'] = (
	df_exchange_rate.groupby(['COUNTRY ISO'])['Exchange Rate'].pct_change() * 100)
    df_exchange_rate=df_exchange_rate[df_exchange_rate['TIME_PERIOD']>=2015]
    df_exchange_rate= euro_area_expansion(df_exchange_rate)
    return df_exchange_rate

# clean state visit data
def df_state_visits_clean():
  df_visits = pd.read_excel(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/Diplometrics_COLT_Travel_Dataset_Primary-HOGS-1990-2024_20250317%20(1).xlsx"),sheet_name = "Master")
  df_visits = df_visits.reset_index()
  df_visits = df_visits[(df_visits['TripYear'] >= 2015) & (df_visits['TripYear'] <= 2024)]

  # Counting for number of visits between each pair of countries
  df_visits['standardized_pair'] = df_visits.apply(
      lambda row: tuple(sorted([row['LeaderCountryISO'], row['CountryVisitedISO']])),
      axis=1
  )
  individual_pairs = df_visits.groupby(['TripYear', 'standardized_pair']).size().reset_index(name='Total_Visits')
  individual_pairs[['Country_ISO_1', 'Country_ISO_2']] = pd.DataFrame(individual_pairs['standardized_pair'].tolist(), index=individual_pairs.index)
  individual_pairs = individual_pairs.drop(columns=['standardized_pair'])

  # Recoding for asymmetric pairs
  asymmetric_pairs = individual_pairs[individual_pairs['Country_ISO_1'] != individual_pairs['Country_ISO_2']].copy()
  asymmetric_pairs['Country_ISO_1'], asymmetric_pairs['Country_ISO_2'] = asymmetric_pairs['Country_ISO_2'], asymmetric_pairs['Country_ISO_1']
  df_visits_count = pd.concat([individual_pairs, asymmetric_pairs], ignore_index=True)
  df_visits_count = df_visits_count.sort_values(by=['TripYear', 'Country_ISO_1', 'Country_ISO_2']).reset_index(drop=True)
  return df_visits_count

# clean military data
def df_military_clean():
  #read data from github
  df_civftl = pd.read_excel(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/number_of_political_violence_events_by_country-year_as-of-06Mar2026.xlsx"))
  df_polvol = pd.read_excel(fetch_data("https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/number_of_reported_fatalities_by_country-year_as-of-06Mar2026.xlsx"))

   # Reset index of df_isocow_ccode to make 'cname' accessible for merging
  df_isocow_ccode=df_isocow_ccode_clean()
  df_isocow_ccode = df_isocow_ccode.reset_index()


  df_military = pd.merge(df_civftl, df_polvol, on=['COUNTRY', 'YEAR'], how='inner')



  # Create a unique mapping from country name to iso3 code
  # If a country name maps to multiple iso3 codes, keep the first one encountered
  unique_iso_mapping = df_isocow_ccode.drop_duplicates(subset=['cname'], keep='first')[['cname', 'iso3']]

  # Merge df_military with the unique ISO mapping
  df_military = pd.merge(df_military,
                        unique_iso_mapping,
                        left_on='COUNTRY',
                        right_on='cname',
                        how='left')

  # Rename the 'iso3' column to 'country_ISO' for clarity
  df_military.rename(columns={'iso3': 'country_ISO'}, inplace=True)

  # Drop the redundant 'cname' column from the merge
  df_military.drop(columns=['cname'], inplace=True)

  # Ensure the index is unique after all merges and manipulations
  df_military = df_military.reset_index(drop=True)

  #Some country has missing or non corresponding names they were manually matched, those countries can be found by running code below
  #df_military_missing_iso = df_military[df_military['country_ISO'].isna()]
  #df_military_missing_iso["COUNTRY"].unique()

  country_name_mapping = {
    'Akrotiri and Dhekelia': 'GBR',
    'Bailiwick of Guernsey': 'GBR',
    'Bailiwick of Jersey': 'GBR',
    'Bolivia': 'BOL',
    'Bosnia and Herzegovina': 'BIH',
    'British Indian Ocean Territory': 'GBR',
    'British Virgin Islands': 'VGB',
    'Brunei': 'BRN',
    'Cape Verde': 'CPV',
    'Caribbean Netherlands': 'NLD',
    'Cayman Islands': 'CYM',
    'Central African Republic': 'CAF',
    'Christmas Island': 'CXR',
    'Cocos (Keeling) Islands': 'CCK',
    'Cook Islands': 'COK',
    'Curacao': 'CUW',
    'Czech Republic': 'CZE',
    'Democratic Republic of Congo': 'COD',
    'Dominican Republic': 'DOM',
    'East Timor': 'TLS',
    'Falkland Islands': 'FLK',
    'Faroe Islands': 'FRO',
    'French Guiana': 'GUF',
    'French Southern and Antarctic Lands': 'ATF',
    'Guadeloupe': 'GLP',
    'Isle of Man': 'IMN',
    'Ivory Coast': 'CIV',
    'Kosovo': 'XKX',
    'Laos': 'LAO',
    'Liechtenstein': 'LIE',
    'Marshall Islands': 'MHL',
    'Martinique': 'MTQ',
    'Micronesia': 'FSM',
    'Moldova': 'MDA',
    'Monaco': 'MCO',
    'Norfolk Island': 'NFK',
    'North Korea': 'PRK',
    'North Macedonia': 'MKD',
    'Northern Mariana Islands': 'MNP',
    'Palestine': 'PSE',
    'Puerto Rico': 'PRI',
    'Republic of Congo': 'COG',
    'Reunion': 'REU',
    'Russia': 'RUS',
    'Saint Helena, Ascension and Tristan da Cunha': 'SHN',
    'Saint-Barthelemy': 'BLM',
    'Saint-Martin': 'MAF',
    'Sint Maarten': 'SXM',
    'Solomon Islands': 'SLB',
    'South Korea': 'KOR',
    'Taiwan': 'TWN',
    'Tanzania': 'TZA',
    'Turks and Caicos Islands': 'TCA',
    'United States': 'USA',
    'Vatican City': 'VAT',
    'Vietnam': 'VNM',
    'Virgin Islands, U.S.': 'USA',
    'Wallis and Futuna': 'WLF',
    'eSwatini': 'SWZ'
  }

  df_military['country_ISO'] = df_military.apply(
    lambda row: country_name_mapping[row['COUNTRY']] if pd.isna(row['country_ISO']) and row['COUNTRY'] in country_name_mapping else row['country_ISO'],
    axis=1
  )

  df_military = df_military.groupby(['country_ISO', 'YEAR'])[['FATALITIES', 'EVENTS']].sum().reset_index()
  df_military.rename(columns={'FATALITIES': 'total_fatalities', "EVENTS":"total_Events"}, inplace=True)

  return df_military

# clean fdi data
def df_fdi_clean():

    def normalize_columns(df):
        df = df.copy()
        df.columns = [
            col.replace('\ufeff', '')
               .replace('ï»¿', '')
               .strip()
               .strip('"')
            for col in df.columns
        ]
        return df

    # -----------------------
    # INWARD FDI
    # -----------------------
    df_inward_fdi = pd.read_csv(
        fetch_data(
            "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/"
            "dataset_2026-03-29T08_13_34.426354506Z_DEFAULT_INTEGRATION_IMF.STA_DIP_12.0.1.csv"
        ),
        encoding='latin1',
        usecols=[0, 1, 2, 6, 7, 10, 11]
    )

    df_inward_fdi = normalize_columns(df_inward_fdi)
    df_inward_fdi = df_inward_fdi.dropna(subset=['TIME_PERIOD'])
    df_inward_fdi['TIME_PERIOD'] = df_inward_fdi['TIME_PERIOD'].astype(int)
    df_inward_fdi['OBS_VALUE'] = pd.to_numeric(df_inward_fdi['OBS_VALUE'], errors='coerce')

    keys = ['COUNTRY.ID', 'COUNTERPART_COUNTRY.ID', 'TIME_PERIOD']

    df_o_in = (
        df_inward_fdi[df_inward_fdi['DV_TYPE.ID'] == 'O']
        .groupby(keys, as_index=False)['OBS_VALUE']
        .sum()
        .rename(columns={'OBS_VALUE': 'O'})
    )

    df_scc_in = (
        df_inward_fdi[df_inward_fdi['DV_TYPE.ID'] == 'SCC']
        .groupby(keys, as_index=False)['OBS_VALUE']
        .sum()
        .rename(columns={'OBS_VALUE': 'SCC'})
    )

    df_inward = pd.merge(df_o_in, df_scc_in, on=keys, how='outer')
    df_inward['Inward FDI'] = df_inward['O'].fillna(df_inward['SCC'])
    df_inward = df_inward[keys + ['Inward FDI']]

    # -----------------------
    # OUTWARD FDI
    # -----------------------
    df_outward_fdi = pd.read_csv(
        fetch_data(
            "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/"
            "dataset_2026-03-29T08_21_04.640359868Z_DEFAULT_INTEGRATION_IMF.STA_DIP_12.0.1.csv"
        ),
        encoding='latin1',
        usecols=[0, 1, 2, 6, 7, 10, 11]
    )

    df_outward_fdi = normalize_columns(df_outward_fdi)
    df_outward_fdi = df_outward_fdi.dropna(subset=['TIME_PERIOD'])
    df_outward_fdi['TIME_PERIOD'] = df_outward_fdi['TIME_PERIOD'].astype(int)
    df_outward_fdi['OBS_VALUE'] = pd.to_numeric(df_outward_fdi['OBS_VALUE'], errors='coerce')

    keys = ['COUNTRY.ID', 'COUNTERPART_COUNTRY.ID', 'TIME_PERIOD']

    df_o = (
        df_outward_fdi[df_outward_fdi['DV_TYPE.ID'] == 'O']
        .groupby(keys, as_index=False)['OBS_VALUE']
        .sum()
        .rename(columns={'OBS_VALUE': 'O'})
    )

    df_scc = (
        df_outward_fdi[df_outward_fdi['DV_TYPE.ID'] == 'SCC']
        .groupby(keys, as_index=False)['OBS_VALUE']
        .sum()
        .rename(columns={'OBS_VALUE': 'SCC'})
    )

    df_outward = pd.merge(df_o, df_scc, on=keys, how='outer')
    df_outward['Outward FDI'] = df_outward['O'].fillna(df_outward['SCC'])
    df_outward = df_outward[keys + ['Outward FDI']]

    df_fdi_merged = pd.merge(
        df_inward,
        df_outward,
        on=['COUNTRY.ID', 'COUNTERPART_COUNTRY.ID', 'TIME_PERIOD'],
        how='outer'
    )

    df_fdi_merged = df_fdi_merged.dropna(subset=['Inward FDI', 'Outward FDI'])
    df_fdi_merged['total_fdi'] = df_fdi_merged['Inward FDI'] + df_fdi_merged['Outward FDI']
    return df_fdi_merged