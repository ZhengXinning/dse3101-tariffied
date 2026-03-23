import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load the trained gravity model
comb_model = sm.load('./combined_model.pickle')

# Load the combined dataframe
df_comb = pd.read_parquet('./backend/temp_df/df_comb.parquet')

# Prepare data for prediction
model_df_comb = df_comb.copy()

# Ensure IdealPointDistance is not NaN before prediction
model_df_comb.dropna(subset=['IdealPointDistance'], inplace=True)

# Filter out any zero or negative values before taking logs, consistent with training
model_df_comb = model_df_comb[model_df_comb['exportFlow'] > 0]
model_df_comb = model_df_comb[model_df_comb['distcap'] > 0]
model_df_comb = model_df_comb[model_df_comb['reporter_gdp/capita'] > 0]
model_df_comb = model_df_comb[model_df_comb['partner_gdp/capita'] > 0]
model_df_comb = model_df_comb[model_df_comb['reporterPopulation'] > 0]
model_df_comb = model_df_comb[model_df_comb['partnerPopulation'] > 0]

# Ensure Tariff is non-negative before taking log
model_df_comb = model_df_comb[model_df_comb['Tariff'] >= 0]

# Apply logarithmic transformations
model_df_comb['ln_reporter_gdp_per_capita'] = np.log(model_df_comb['reporter_gdp/capita'])
model_df_comb['ln_partner_gdp_per_capita'] = np.log(model_df_comb['partner_gdp/capita'])
model_df_comb['ln_distcap'] = np.log(model_df_comb['distcap'])
model_df_comb['ln_ideal_point_distance'] = np.log(model_df_comb['IdealPointDistance'])
model_df_comb['ln_tariff'] = np.log(model_df_comb['Tariff'] + 0.0001)
model_df_comb['ln_repPop'] = np.log(model_df_comb['reporterPopulation'])
model_df_comb['ln_partPop'] = np.log(model_df_comb['partnerPopulation'])

# Predict the log of export flow
predicted_ln_exportflow = comb_model.predict(model_df_comb, True)

# Exponentiate to get the predicted actual export flow
model_df_comb['predicted_exportFlow'] = np.exp(predicted_ln_exportflow)

# Display results
results = model_df_comb[['exportFlow', 'predicted_exportFlow', 'reporterISO', 'reporterDesc', 'partnerISO', 'partnerDesc']]
print(results)