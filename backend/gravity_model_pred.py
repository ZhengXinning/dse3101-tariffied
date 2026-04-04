import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

if __name__ == "__main__":
    # Load the trained gravity model (model.fit())
    comb_model = sm.load('./backend/gravity_models/combined_model.pickle')

    # Load the combined dataframe
    df_comb = pd.read_parquet('./backend/temp_df/df_comb.parquet')

    # Prepare data for prediction
    df_gravity = df_comb.copy()

    # # Ensure IdealPointDistance is not NaN before prediction
    # df_gravity.dropna(subset=['IdealPointDistance'], inplace=True)

    # # Filter out any zero or negative values before taking logs, consistent with training
    df_gravity = df_gravity[df_gravity['exportFlow'] > 0]
    df_gravity = df_gravity[df_gravity['distcap'] > 0]
    df_gravity = df_gravity[df_gravity['reporter_gdp/capita'] > 0]
    df_gravity = df_gravity[df_gravity['partner_gdp/capita'] > 0]
    df_gravity = df_gravity[df_gravity['reporterPopulation'] > 0]
    df_gravity = df_gravity[df_gravity['partnerPopulation'] > 0]

    # Ensure Tariff is non-negative before taking log
    df_gravity = df_gravity[df_gravity['Tariff'] >= 0]

    # Apply logarithmic transformations
    df_gravity['ln_reporter_gdp_per_capita'] = np.log(df_gravity['reporter_gdp/capita'])
    df_gravity['ln_partner_gdp_per_capita'] = np.log(df_gravity['partner_gdp/capita'])
    df_gravity['ln_distcap'] = np.log(df_gravity['distcap'])
    df_gravity['ln_ideal_point_distance'] = np.log(df_gravity['IdealPointDistance'])
    df_gravity['ln_tariff'] = np.log(df_gravity['Tariff'] + 0.0001)
    df_gravity['ln_repPop'] = np.log(df_gravity['reporterPopulation'])
    df_gravity['ln_partPop'] = np.log(df_gravity['partnerPopulation'])

    # Predict the log of export flow
    predicted_ln_exportflow = comb_model.predict(df_gravity, True)

    # Exponentiate to get the predicted actual export flow
    df_gravity['predicted_exportFlow'] = np.exp(predicted_ln_exportflow)

    # Ratio of predicted to actual export flow
    df_gravity['tradeRatio'] = df_gravity['exportFlow']/df_gravity['predicted_exportFlow']

    # Display results
    print(df_gravity)

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

    # Save results
    os.makedirs("./backend/temp_df", exist_ok=True)
    df_gravity.to_parquet("./backend/temp_df/df_gravity.parquet")

