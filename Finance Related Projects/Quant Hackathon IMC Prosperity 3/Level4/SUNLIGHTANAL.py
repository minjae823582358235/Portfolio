import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Constants and data structures
days = ['_day_2.csv', '_day_3.csv', '_day_4.csv']
timearr = []
sunlightIndexArr = []
pristinePriceArr = []
MAGNIFICENT_MACARONS = 'MAGNIFICENT_MACARONS'
constant = 0

# Kalman filter for smoothing
def kalman_filter_1d(z, Q=1e-5, R=4):
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

# Process data for each day
for day in days:
    # Read the observation and price data
    ObservationDF = pd.read_csv(f'Level4\\round-4-island-data-bottle\\observations_round_5{day}', sep=',')
    PricesDF = pd.read_csv(f'Level4\\round-4-island-data-bottle\\prices_round_5{day}', sep=',')
    PricesDF = PricesDF[PricesDF['product'] == MAGNIFICENT_MACARONS]

    for time in PricesDF['timestamp']:
        if time == 0:
            continue

        # Retrieve current and previous timeframes
        ObservationTimeFrame = ObservationDF[ObservationDF['timestamp'] == time]

        # Skip if data is missing
        if ObservationTimeFrame.empty:
            continue

        # Extract sunlight index and pristine mid-price
        sunlightIndex = ObservationTimeFrame['sunlightIndex'].iloc[0]
        PristineMidPrice = np.mean([ObservationTimeFrame['bidPrice'].iloc[0], ObservationTimeFrame['askPrice'].iloc[0]])

        # Append data to arrays
        sunlightIndexArr.append(sunlightIndex)
        pristinePriceArr.append(PristineMidPrice)
        timearr.append(time / 100 + constant)

    constant += 10000

# Smooth the pristine price using a Kalman filter
smoothedPristinePrice = kalman_filter_1d(pristinePriceArr)

# Use the raw sunlight index (no smoothing)
rawSunlightIndex = sunlightIndexArr

# Adjust the time array to match the smoothed data
smoothedTimeArr = timearr

# Calculate the rate of change of the smoothed pristine mid-price
pristinePriceChange = np.diff(smoothedPristinePrice)

# Classify turning points into bullish and bearish trends
bullish_turning_points = [] 
bearish_turning_points = []

# Calculate the rate of change of the smoothed pristine mid-price
pristinePriceChange = np.diff(smoothedPristinePrice)

# Identify turning points (where the rate of change crosses zero)
turning_points = []
for i in range(1, len(pristinePriceChange)):
    if pristinePriceChange[i - 1] > 0 and pristinePriceChange[i] < 0:  # Local maxima
        turning_points.append(i)
    elif pristinePriceChange[i - 1] < 0 and pristinePriceChange[i] > 0:  # Local minima
        turning_points.append(i)

# Debugging: Log the identified turning points
print("Turning Points (Indices):", turning_points)
print("Turning Points (Values):", [smoothedPristinePrice[i] for i in turning_points])

for i in range(1, len(turning_points) - 1):
    if smoothedPristinePrice[turning_points[i]] < smoothedPristinePrice[turning_points[i + 1]]:  # Bullish trend
        bullish_turning_points.append(turning_points[i])
    elif smoothedPristinePrice[turning_points[i]] > smoothedPristinePrice[turning_points[i + 1]]:  # Bearish trend
        bearish_turning_points.append(turning_points[i])

# Extract sunlight index values for bullish and bearish trends
bullish_sunlight_values = [rawSunlightIndex[i] for i in bullish_turning_points]
bearish_sunlight_values = [rawSunlightIndex[i] for i in bearish_turning_points]

# Calculate critical sunlight levels
critical_bullish_sunlight = np.mean(bullish_sunlight_values) if bullish_sunlight_values else None
critical_bearish_sunlight = np.mean(bearish_sunlight_values) if bearish_sunlight_values else None

# Define the triangular wave function
def triangular_wave(time, period=10000, min_val=0, max_val=100):
    """
    Generate a triangular wave for a given time array.
    Args:
        time (float): The time value.
        period (int): The period of the triangular wave.
        min_val (float): The minimum value of the wave.
        max_val (float): The maximum value of the wave.
    Returns:
        float: The value of the triangular wave at the given time.
    """
    half_period = period / 2
    amplitude = (max_val - min_val) / 2
    offset = min_val + amplitude

    # Calculate the position within the period
    t_mod = time % period

    # Ascending part of the wave
    if t_mod <= half_period:
        return min_val + (t_mod / half_period) * (max_val - min_val)
    # Descending part of the wave
    else:
        return max_val - ((t_mod - half_period) / half_period) * (max_val - min_val)


# Generate the triangular wave for the entire time array
triangular_wave_values = [triangular_wave(t) for t in timearr]

# Subtract the triangular wave from the raw sunlight index
adjusted_sunlight_index = [raw - tri for raw, tri in zip(rawSunlightIndex, triangular_wave_values)]

# Plot the results
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot smoothed pristine prices on the primary y-axis
ax1.plot(smoothedTimeArr, smoothedPristinePrice, color='blue', label='Smoothed Pristine Mid Price')
ax1.scatter([smoothedTimeArr[i] for i in bullish_turning_points], [smoothedPristinePrice[i] for i in bullish_turning_points],
            color='green', label='Bullish Turning Points', zorder=5)
ax1.scatter([smoothedTimeArr[i] for i in bearish_turning_points], [smoothedPristinePrice[i] for i in bearish_turning_points],
            color='red', label='Bearish Turning Points', zorder=5)
ax1.set_xlabel("Time")
ax1.set_ylabel("Pristine Mid Price", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for adjusted sunlight index
ax2 = ax1.twinx()
ax2.plot(smoothedTimeArr, adjusted_sunlight_index, color='orange', label='Adjusted Sunlight Index', alpha=0.6)
ax2.plot(smoothedTimeArr, triangular_wave_values, color='gray', linestyle='dotted', label='Triangular Wave')
ax2.set_ylabel("Sunlight Index", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add a title and legends
plt.title("Smoothed Pristine Mid Price and Adjusted Sunlight Index with Bullish and Bearish Turning Points")
ax1.legend(loc='upper left')  # Legend for the primary y-axis
ax2.legend(loc='upper right')  # Legend for the secondary y-axis
fig.tight_layout()
plt.show()