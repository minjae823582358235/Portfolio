import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
days = ['_day_1.csv', '_day_2.csv', '_day_3.csv']
MAGNIFICENT_MACARONS = 'MAGNIFICENT_MACARONS'

# Data storage
X_all = []
Y_all = []

# Load and merge data
for day in days:
    obs_df = pd.read_csv(f'Level4/round-4-island-data-bottle/observations_round_4{day}')
    price_df = pd.read_csv(f'Level4/round-4-island-data-bottle/prices_round_4{day}', sep=';')
    price_df = price_df[price_df['product'] == MAGNIFICENT_MACARONS]

    for time in price_df['timestamp']:
        obs_row = obs_df[obs_df['timestamp'] == time]
        if obs_row.empty:
            continue

        # Pristine mid-price from observations
        pristine_mid = np.mean([
            obs_row['bidPrice'].iloc[0],
            obs_row['askPrice'].iloc[0]
        ])
        shift=0
        if time+shift>=999900:
            continue
        # Build feature row
        row = [
            obs_row['transportFees'].iloc[0],
            obs_row['exportTariff'].iloc[0],
            obs_row['importTariff'].iloc[0],
            obs_row['sugarPrice'].iloc[0],
            obs_row['sunlightIndex'].iloc[0],
            # pristine_mid
        ]

        X_all.append(row)
        Y_all.append(price_df[price_df['timestamp'] == time+shift]['mid_price'].iloc[0])

X_all = np.array(X_all)
Y_all = np.array(Y_all)

# --- Kalman Filter Setup ---
n_factors = 5  # T, E, I, Sug, Sun, PristineMid
x = np.zeros((n_factors, 1))          # Initial weights
P = np.eye(n_factors) * 1000          # Initial uncertainty
Q = np.eye(n_factors) * 1e-5          # Process noise
R = 4                                 # Observation noise

H_history = []

# --- Kalman Filter Loop ---
for t in range(len(Y_all)):
    z = Y_all[t]
    H = X_all[t].reshape(1, -1)  # Measurement matrix (1x6)

    # Predict
    x_prior = x
    P_prior = P + Q

    # Kalman Gain
    S_k = H @ P_prior @ H.T + R
    K = P_prior @ H.T / S_k

    # Update
    y_tilde = z - H @ x_prior
    x = x_prior + K * y_tilde
    P = (np.eye(n_factors) - K @ H) @ P_prior

    H_history.append(x.flatten())

# --- Output final smoothed weights ---
labels = ["Transport", "Export", "Import", "Sugar", "Sunlight"]
final_weights = x.flatten()

print("\n📈 Final Kalman-Smoothed Coefficient Estimates:")
for name, val in zip(labels, final_weights):
    print(f"{name:15}: {val:.4f}")

# --- Visualization ---
H_array = np.array(H_history)

# --- Visualization ---
H_array = np.array(H_history)

plt.figure(figsize=(12, 6))
for i in range(n_factors):
    plt.plot(H_array[:, i], label=labels[i])
plt.title("Kalman Filter – Coefficient Estimates Over Time")
plt.xlabel("Time Step")
plt.ylabel("Weight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#movement of the prices in day 2 is just noise? Extreme slopes trigger the change of the prices
# mean reversion + shit hit the fan strat?