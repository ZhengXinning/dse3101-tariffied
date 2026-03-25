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
    predicted_exportFlow, tradeRatio
    """
    df_import_comb = []
    for country_code, urls in countries_config.items():
        # Load and merge data
        df_import = clean_import(importurl=urls['import'])
        df_import_comb.append(df_import)

    df_import_comb = pd.concat(df_import_comb, ignore_index=True)
    df_final = df_gravity.merge(df_import_comb, how='left', on=['refYear', 'reporterCode', 'reporterISO', 'reporterDesc', 'partnerCode', 'partnerISO', 'partnerDesc', 'cmdCode', 'cmdDesc'])
    df_final['totalFlow'] = df_final['exportFlow'] + df_final['importFlow']
    df_final['reporterTradePctGdp'] = df_final['totalFlow'] / df_final['reporterGdp']
    df_final['partnerTradePctGdp'] = df_final['totalFlow'] / df_final['partnerGdp']
    df_iso = df_isoregion_clean()
    df_final = df_final.merge(df_iso, how='left', left_on='reporterISO', right_on='alpha-3')
    df_final['riskIndex'] = 100
    df_final = df_final[['refYear', 'cmdCode', 'cmdDesc',
                        'reporterCode', 'reporterISO', 'reporterDesc', 
                        'reporterGdp', 'reporterPopulation', 'reporter_gdp/capita',
                        'reporterlat', 'reporterlong',
                        'partnerCode', 'partnerISO', 'partnerDesc', 
                        'partnerGdp', 'partnerPopulation', 'partner_gdp/capita',
                        'partnerlat', 'partnerlong',
                        'exportFlow', 'importFlow', 'totalFlow', 'predicted_exportFlow', 'tradeRatio', 
                        'riskIndex',
                        'reporterTradePctGdp', 'partnerTradePctGdp',
                        'IdealPointDistance', 'Tariff']]
    print(df_final.shape)
    print(df_final.columns)
    print(df_final) #59177
    # print(df_final.dropna().shape) 48737

    os.makedirs("./backend/temp_df", exist_ok=True)
    df_final.to_parquet("./backend/temp_df/df_final.parquet")


