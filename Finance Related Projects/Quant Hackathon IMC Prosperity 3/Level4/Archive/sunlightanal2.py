import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

days = ['_day_1.csv', '_day_2.csv', '_day_3.csv']
timearr = []
deltaSunlightArr = []
pristinePriceArr = []
sugarPriceArr = []
MAGNIFICENT_MACARONS = 'MAGNIFICENT_MACARONS'
constant = 0

def kalman_filter_1d(z, Q=1e-5, R=4):
    """
    1D Kalman filter using numpy.
    z: observed time series (e.g., mid-prices)
    Q: process variance (model noise)
    R: measurement variance (observation noise)
    Returns: filtered estimates
    """
    n = len(z)
    x_hat = np.zeros(n)      # filtered state estimate
    P = np.zeros(n)          # error covariance
    x_hat[0] = z[0]          # initial state
    P[0] = 1.0               # initial covariance

    for k in range(1, n):
        # Predict
        x_hat_minus = x_hat[k - 1]
        P_minus = P[k - 1] + Q

        # Update
        K = P_minus / (P_minus + R)
        x_hat[k] = x_hat_minus + K * (z[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus

    return x_hat

for day in days:
    # Read the observation and price data
    ObservationDF = pd.read_csv(f'Level4\\round-4-island-data-bottle\\observations_round_4{day}', sep=',')
    PricesDF = pd.read_csv(f'Level4\\round-4-island-data-bottle\\prices_round_4{day}', sep=';')
    PricesDF = PricesDF[PricesDF['product'] == MAGNIFICENT_MACARONS]

    for time in PricesDF['timestamp']:
        if time == 0:
            continue

        # Retrieve current and previous timeframes
        ObservationTimeFrame = ObservationDF[ObservationDF['timestamp'] == time]
        TimeFrameonestepBack = ObservationDF[ObservationDF['timestamp'] == time - 100]

        # Skip if data is missing
        if ObservationTimeFrame.empty or TimeFrameonestepBack.empty:
            continue

        # Calculate delta sunlight
        deltaSun = ObservationTimeFrame['sunlightIndex'].iloc[0] - TimeFrameonestepBack['sunlightIndex'].iloc[0]
        PristineMidPrice = np.mean([ObservationTimeFrame['bidPrice'].iloc[0], ObservationTimeFrame['askPrice'].iloc[0]])
        SugarPrice = ObservationTimeFrame['sugarPrice'].iloc[0]

        # Append data to arrays
        deltaSunlightArr.append(deltaSun)
        pristinePriceArr.append(PristineMidPrice)
        sugarPriceArr.append(SugarPrice)
        timearr.append(time / 100 + constant)

    constant += 10000

# Ensure arrays are numpy arrays for easier manipulation
timearr = np.array(timearr)
pristinePriceArr = np.array(pristinePriceArr)
deltaSunlightArr = np.array(deltaSunlightArr)
sugarPriceArr = np.array(sugarPriceArr)

# Apply Kalman smoothing to the arrays
smoothed_pristine = kalman_filter_1d(pristinePriceArr)
smoothed_sugar = kalman_filter_1d(sugarPriceArr)

# 1. Time-Lagged Correlation Analysis
print("\nTime-Lagged Correlation Analysis:")
for lag in range(1, 11):  # Test lags from 1 to 10 time steps
    lagged_sunlight = np.roll(deltaSunlightArr, lag)
    lagged_sugar = np.roll(smoothed_sugar, lag)
    corr_lag_sunlight, _ = pearsonr(smoothed_pristine[lag:], lagged_sunlight[lag:])
    corr_lag_sugar, _ = pearsonr(smoothed_pristine[lag:], lagged_sugar[lag:])
    print(f"Lag {lag}: Correlation with Delta Sunlight Index = {corr_lag_sunlight:.2f}, "
          f"Correlation with Sugar Prices = {corr_lag_sugar:.2f}")

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Extract additional features
transportFeesArr = []
exportTariffArr = []
importTariffArr = []

for day in days:
    ObservationDF = pd.read_csv(f'Level4\\round-4-island-data-bottle\\observations_round_4{day}', sep=',')
    for time in PricesDF['timestamp']:
        if time == 0:
            continue
        ObservationTimeFrame = ObservationDF[ObservationDF['timestamp'] == time]
        if ObservationTimeFrame.empty:
            continue
        transportFeesArr.append(ObservationTimeFrame['transportFees'].iloc[0])
        exportTariffArr.append(ObservationTimeFrame['exportTariff'].iloc[0])
        importTariffArr.append(ObservationTimeFrame['importTariff'].iloc[0])

# Convert to numpy arrays
transportFeesArr = np.array(transportFeesArr)
exportTariffArr = np.array(exportTariffArr)
importTariffArr = np.array(importTariffArr)

# Prepare data for regression
X = np.column_stack((deltaSunlightArr, smoothed_sugar, transportFeesArr, exportTariffArr, importTariffArr))  # Independent variables
y = smoothed_pristine  # Dependent variable

# Decision tree regressor
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)  # Limit depth to prevent overfitting
tree_model.fit(X, y)
y_tree_pred = tree_model.predict(X)

# Model Performance
mae = mean_absolute_error(y, y_tree_pred)
rmse = np.sqrt(mean_squared_error(y, y_tree_pred))
print("\nDecision Tree Model Performance with Additional Features:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Feature Importance
feature_importances = tree_model.feature_importances_
print("\nFeature Importances:")
print(f"Delta Sunlight Index: {feature_importances[0]:.2f}")
print(f"Sugar Prices: {feature_importances[1]:.2f}")
print(f"Transport Fees: {feature_importances[2]:.2f}")
print(f"Export Tariff: {feature_importances[3]:.2f}")
print(f"Import Tariff: {feature_importances[4]:.2f}")

# Residual Plot
residuals = y - y_tree_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_tree_pred, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Pristine Mid Price")
plt.ylabel("Residuals")
plt.title("Residual Plot (Decision Tree with Additional Features)")
plt.show()

# Visualization
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# Plot 1: Pristine Mid Price vs Delta Sunlight Index
axs[0].scatter(deltaSunlightArr, smoothed_pristine, color='orange', alpha=0.6, label='Data Points')
axs[0].set_xlabel("Delta Sunlight Index")
axs[0].set_ylabel("Pristine Mid Price")
axs[0].set_title("Pristine Mid Price vs Delta Sunlight Index")
axs[0].legend()

# Plot 2: Pristine Mid Price vs Sugar Prices
axs[1].scatter(smoothed_sugar, smoothed_pristine, color='green', alpha=0.6, label='Data Points')
axs[1].set_xlabel("Sugar Prices")
axs[1].set_ylabel("Pristine Mid Price")
axs[1].set_title("Pristine Mid Price vs Sugar Prices")
axs[1].legend()

# Plot 3: Actual vs Predicted Pristine Mid Price
axs[2].plot(timearr, smoothed_pristine, color='blue', label='Actual Pristine Mid Price')
axs[2].plot(timearr, y_tree_pred, color='red', linestyle='--', label='Predicted Pristine Mid Price (Decision Tree with Additional Features)')
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Pristine Mid Price")
axs[2].set_title("Actual vs Predicted Pristine Mid Price (Decision Tree with Additional Features)")
axs[2].legend()

plt.tight_layout()
plt.show()