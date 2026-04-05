import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from prepare_data import *

countries_config = {
    'SG': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/SGP_ExporttoWorld_2015-2024.csv",
        'import': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/SGP_ImportfromWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstSGPClean.csv"
    },
    'CHN': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/CHN_ExporttoWorld_2015-2024.csv",
        'import': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/CHN_ImportfromWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstCHNClean.csv"
    },
    'DEU': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/DEU_ExporttoWorld_2015-2024.csv",
        'import': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/DEU_ImportfromWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstDEUClean.csv"
    },
    'USA': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/USA_ExporttoWorld_2015-2024.csv",
        'import': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/USA_ImportfromWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstUSAClean.csv"
    },
    'JPN': {
        'export': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/JPN_ExporttoWorld_2015-2024.csv",
        'import': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/JPN_ImportfromWorld_2015-2024.csv",
        'tariff': "https://github.com/ZhengXinning/dse3101-tariffied/blob/main/data/WorldReportedTariffAgainstJPNClean.csv"
    }
}

if __name__ == "__main__":
    gravity = pd.read_parquet('./backend/temp_df/df_gravity.parquet')
    df_gravity = gravity.copy()
    print(df_gravity.shape)
    """
    df_gravity
    Index columns: refYear, reporterCode, reporterISO, reporterDesc, partnerCode,
    partnerISO, partnerDesc, cmdCode, cmdDesc, exportFlow,
    reporterGdp, reporterPopulation, reporterlat, reporterlong,
    partnerGdp, partnerPopulation, partnerlat, partnerlong, dist,
    distcap, distw, distwces, IdealPointDistance, Tariff,
    reporter_gdp/capita, partner_gdp/capita,
    ln_reporter_gdp_per_capita, ln_partner_gdp_per_capita, ln_distcap,
    ln_ideal_point_distance, ln_tariff, ln_repPop, ln_partPop,
    predicted_exportFlow, predicted_exportFlow_base, tradeRatio
    """
    
    risk = pd.read_parquet('./backend/temp_df/df_pca_risk.parquet')
    df_risk = risk.copy()
    print(df_risk.shape)
    """
    Index(['reporter_iso', 'Reference area', 'partner_iso', 'Counterpart area',    
       'year', 'transptCost', 'fxChange', 'IdealPointDistance', 'stateVisits', 
       'repFatalities', 'repEvents', 'partFatalities', 'partEvents',
       'totalFdi', 'transptCost_weighted', 'fxChange_weighted',
       'IdealPointDistance_weighted', 'stateVisits_weighted',
       'repFatalities_weighted', 'repEvents_weighted',
       'partFatalities_weighted', 'partEvents_weighted', 'totalFdi_weighted',  
       'Risk_Index_Raw'],
      dtype='str')
    """

    # Merge import data
    df_import_comb = []
    for country_code, urls in countries_config.items():
        # Load and merge data
        df_import = clean_import(importurl=urls['import'])
        df_import_comb.append(df_import)

    df_import_comb = pd.concat(df_import_comb, ignore_index=True)
    df_final = df_gravity.merge(df_import_comb, how='left', on=['refYear', 'reporterCode', 'reporterISO', 'reporterDesc', 'partnerCode', 'partnerISO', 'partnerDesc', 'cmdCode', 'cmdDesc'])
    
    # Create total flow col
    df_final['totalFlow'] = df_final['exportFlow'] + df_final['importFlow']
    
    # Create total trade to GDP ratio col
    df_final['reporterTradePctGdp'] = df_final['totalFlow'] / df_final['reporterGdp']
    df_final['partnerTradePctGdp'] = df_final['totalFlow'] / df_final['partnerGdp']
    
    # Create region cols
    df_iso = df_isoregion_clean()
    df_final = df_final.merge(df_iso, how='left', left_on='reporterISO', right_on='alpha-3').rename(columns={'region':'reporterRegion'})
    df_final = df_final.merge(df_iso, how='left', left_on='partnerISO', right_on='alpha-3').rename(columns={'region':'partnerRegion'})
    
    # Weighted risk index cols
    df_final = df_final.merge(df_risk, how='left', left_on=['refYear', 'reporterISO', 'partnerISO'], right_on=['year', 'reporter_iso', 'partner_iso'])
    df_final = df_final[['refYear', 'cmdCode', 'cmdDesc',
                        'reporterCode', 'reporterISO', 'reporterDesc', 'reporterRegion',
                        'reporterGdp', 'reporterPopulation', 'reporter_gdp/capita',
                        'reporterlat', 'reporterlong',
                        'partnerCode', 'partnerISO', 'partnerDesc', 'partnerRegion',
                        'partnerGdp', 'partnerPopulation', 'partner_gdp/capita',
                        'partnerlat', 'partnerlong',
                        'exportFlow', 'importFlow', 'totalFlow', 'predicted_exportFlow_base', 'predicted_exportFlow_geoPol', 'tradeRatio', 
                        'reporterTradePctGdp', 'partnerTradePctGdp',
                        'transptCost_weighted', 'fxChange_weighted',
                        'IdealPointDistance_weighted', 'stateVisits_weighted',
                        'repFatalities_weighted', 'repEvents_weighted',
                        'partFatalities_weighted', 'partEvents_weighted', 'totalFdi_weighted',  
                        'Risk_Index_Raw']]
    print(df_final.shape)
    print(df_final.columns)
    print(df_final['reporterRegion'])
    print(df_final['partnerRegion'])
    print(df_final['tradeRatio'])
    print(df_final.isna().sum())
    print(df_final) 

    os.makedirs("./backend/temp_df", exist_ok=True)
    df_final.to_parquet("./backend/temp_df/df_final.parquet")


