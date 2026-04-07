import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import Counter

import subprocess

# --- Step 1: Load price data ---
data_dir_round1 = "Level1/round-1-island-data-bottle/"
data_dir_round2 = "Level2/data/"

day1_files = ["prices_round_1_day_-1.csv", "prices_round_1_day_0.csv", "prices_round_1_day_-2.csv"]
day2_files = ["prices_round_2_day_-1.csv", "prices_round_2_day_0.csv", "prices_round_2_day_1.csv"]

# Combine all file paths with their respective directories
all_files = [os.path.join(data_dir_round1, f) for f in day1_files] + \
            [os.path.join(data_dir_round2, f) for f in day2_files]

# Read all dataframes
DFArr = [pd.read_csv(f, sep=",") for f in all_files]

commodities = ["KELP"]
dictofDF = {}

for i, df in enumerate(DFArr):
    for product in commodities:
        productDF = df[df["product"] == product].copy()
        productDF["timestamp"] += i * 1_000_000

        if product not in dictofDF:
            dictofDF[product] = productDF.copy()
        else:
            dictofDF[product] = pd.concat([dictofDF[product], productDF], ignore_index=True)

kelp = dictofDF["KELP"]
kelp_mid = kelp["mid_price"].reset_index(drop=True)
kelpReturns = [(kelp_mid[i] - kelp_mid[i - 1]) / kelp_mid[i - 1] for i in range(1, len(kelp_mid))]

kelplines = np.sort([
    0.000494, -0.000247, 0.0, -0.000494, 0.000247, 0.000741,
    -0.000741, 0.000989, -0.000988, -0.00123, 0.00148, -0.00148, 0.00124
])

# --- SMA/LMA Settings ---######################################################################################################################################################################################
sma_window = 1
lma_window = 5
neutral_window = 0
######################################################################################################################################################################################

# --- Trend Classification ---
sma = pd.Series(kelp_mid).rolling(window=sma_window).mean()
lma = pd.Series(kelp_mid).rolling(window=lma_window).mean()
crossover_points = np.where(np.sign(sma - lma).diff().fillna(0) != 0)[0]

def classify_trend(idx):
    if np.any(np.abs(crossover_points - idx) <= neutral_window):
        return "Neutral"
    return "Uptrend" if sma[idx] > lma[idx] else "Downtrend"

def find_nearest_index(array, value):
    tol = 0.0015
    if abs(value) > tol:
        return "PlusAnomaly" if value > tol else "MinusAnomaly"
    return (np.abs(array - value)).argmin()

def find_nearest_value(array, value):
    tol = 0.0015
    if abs(value) > tol:
        return value
    return array[(np.abs(array - value)).argmin()]

# --- Matrix Initialization ---
rows = len(kelplines) + 2
pMatrix = np.empty((rows,), dtype=object)
trendMatrix = np.empty((rows,), dtype=object)
for i in range(rows):
    pMatrix[i] = []
    trendMatrix[i] = []

# --- Horizon setting ---##################################################################################################################################
future_return_horizon = 3  # adjustable to 5, 10, etc.
######################################################################################################################################################

# --- Fill Matrix with: [return_t+1, cumulative_t+1→t+horizon], and trend label
for i in range(len(kelpReturns) - future_return_horizon):
    kelp_ret = kelpReturns[i]                      # return at t
    kelp_ret_t1 = kelpReturns[i + 1]               # return at t+1
    kelp_ret_thorizon = sum(kelpReturns[i + 1 : i + 1 + future_return_horizon])  # cumulative return
    trend = classify_trend(i + future_return_horizon)

    KIndex = find_nearest_index(kelplines, kelp_ret)

    if KIndex == "MinusAnomaly":
        row = 0
    elif KIndex == "PlusAnomaly":
        row = rows - 1
    else:
        row = int(KIndex) + 1

    pMatrix[row].append((kelp_ret_t1, kelp_ret_thorizon))
    trendMatrix[row].append(trend)

# --- Build Final Matrix: [t+1 return, cumulative return, trend label, dominance %]
trend_output_matrix = []
for i in range(rows):
    cell_data = pMatrix[i]
    trend_data = trendMatrix[i]

    if not cell_data:
        trend_output_matrix.append([None, None, "Neutral", 0.0])
    else:
        t1_vals = [x[0] for x in cell_data]
        th_vals = [x[1] for x in cell_data]

        avg_t1 = np.mean(t1_vals)
        avg_th = np.mean(th_vals)

        trend_counts = Counter(trend_data)
        dominant = max(trend_counts, key=trend_counts.get)
        total = sum(trend_counts.values())
        dominant_pct = trend_counts[dominant] / total if total > 0 else 0.0

        trend_output_matrix.append([
            float(f"{avg_t1:.6g}"),
            float(f"{avg_th:.6g}"),
            dominant,
            float(f"{dominant_pct:.6f}")
        ])

# --- Write Matrix to File ---
def format_trend_matrix(matrix):
    lines = ["TREND_MATRIX = ["]
    for cell in matrix:
        val1 = "None" if cell[0] is None or np.isnan(cell[0]) else f"{cell[0]:.6g}"
        val2 = "None" if cell[1] is None or np.isnan(cell[1]) else f"{cell[1]:.6g}"
        label = cell[2]
        pct = cell[3]
        lines.append(f"  [{val1}, {val2}, \"{label}\", {pct}],")
    lines.append("]")
    return "\n".join(lines)

formatted_output = format_trend_matrix(trend_output_matrix)
output_path = "/Users/jameszhao/Documents/IMC3Round1/Kelp/solo_kelp_trend_matrix_output.txt"

# --- Delete file if exists ---
if os.path.exists(output_path):
    os.remove(output_path)

# --- Write new output ---
with open(output_path, "w") as f:
    f.write(formatted_output)

print("Saved trend matrix to:", output_path)

# --- Open the file in TextEdit ---
subprocess.run(["open", "-a", "TextEdit", output_path])
