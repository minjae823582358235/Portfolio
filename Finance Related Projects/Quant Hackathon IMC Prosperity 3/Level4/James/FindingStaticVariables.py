import pandas as pd
import numpy as np

# File paths for the observation and price CSVs
observation_files = [
    '/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level4/round-4-island-data-bottle/observations_round_4_day_1.csv',
    '/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level4/round-4-island-data-bottle/observations_round_4_day_2.csv',
    '/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level4/round-4-island-data-bottle/observations_round_4_day_3.csv'
]

price_files = [
    '/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level4/round-4-island-data-bottle/prices_round_4_day_1.csv',
    '/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level4/round-4-island-data-bottle/prices_round_4_day_2.csv',
    '/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level4/round-4-island-data-bottle/prices_round_4_day_3.csv'
]

# Load and concatenate observation data (using the default comma delimiter)
obs_list = [pd.read_csv(f) for f in observation_files]
obs_data = pd.concat(obs_list, ignore_index=True)

# Load and concatenate price data (using semicolon as delimiter)
price_list = [pd.read_csv(f, sep=';') for f in price_files]
price_data = pd.concat(price_list, ignore_index=True)

# Print column names for debugging
print("Observation Data Columns:", obs_data.columns.tolist())
print("Price Data Columns:", price_data.columns.tolist())

# Check if the time column needs renaming in either DataFrame.
# For example, if one DataFrame uses "time" but the other uses "timestamp".
if 'time' in obs_data.columns and 'timestamp' not in obs_data.columns:
    obs_data.rename(columns={"time": "timestamp"}, inplace=True)

if 'time' in price_data.columns and 'timestamp' not in price_data.columns:
    price_data.rename(columns={"time": "timestamp"}, inplace=True)

# Filter the price data for "MAGNIFICENT_MACARONS"
# It is assumed that there is a 'product' column in the price files.
magma_data = price_data[price_data['product'] == "MAGNIFICENT_MACARONS"]

# Merge the observation and price data on the 'timestamp' column.
# (If the timestamps don’t exactly match, consider using pd.merge_asof.)
merged_data = pd.merge(magma_data, obs_data, on='timestamp', how='inner')

# Print the merged DataFrame (first 5 rows) for verification
print("Merged Data (first 5 rows):")
print(merged_data.head())

# Prepare the variables for regression:
# - 'sugarPrice' (from observations) is the independent variable.
# - 'mid_price' (from price data) is the dependent variable.
X = merged_data['sugarPrice'].values   # Independent variable
y = merged_data['mid_price'].values      # Dependent variable

# Compute the regression coefficients (slope and intercept) using numpy's polyfit
m, c = np.polyfit(X, y, 1)
print(f"\nRegression Coefficients:\n  Slope (m) = {m}\n  Intercept (c) = {c}")

# Calculate the expected mid-price and the residuals
merged_data['expected_mid'] = m * merged_data['sugarPrice'] + c
merged_data['residual'] = merged_data['mid_price'] - merged_data['expected_mid']

# Compute the mean and standard deviation of the residuals
mean_residual = merged_data['residual'].mean()
std_residual = merged_data['residual'].std()
print(f"\nResidual Statistics:\n  Mean = {mean_residual}\n  Standard Deviation = {std_residual}")

# Set a z-score threshold (commonly ±2 standard deviations)
z_threshold = 2
print(f"\nZ-score Threshold: ±{z_threshold}")
