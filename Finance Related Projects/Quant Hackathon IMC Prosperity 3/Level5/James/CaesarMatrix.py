import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from collections import Counter

import subprocess

# --- Step 1: Load price data ---
data_dir_round1 = "Level5/r5data/"

price_files = ["prices_round_5_day_2.csv", "prices_round_5_day_3.csv", "prices_round_5_day_4.csv"]
trade_files = ["trades_round_5_day_2.csv", "trades_round_5_day_3.csv", "trades_round_5_day_4.csv"]

# Combine all file paths with their respective directories
all_files = os.path.join(data_dir_round1)
      
# Read all dataframes
DFArr = [pd.read_csv(f, sep=",") for f in all_files]

commodities = ["VOLCANIC_ROCK_VOUCHER_10500"]
dictofDF = {}

for i, df in enumerate(DFArr):
    for product in commodities:
        productDF = df[df["product"] == product].copy()
        productDF["timestamp"] += i * 1_000_000

        if product not in dictofDF:
            dictofDF[product] = productDF.copy()
        else:
            dictofDF[product] = pd.concat([dictofDF[product], productDF], ignore_index=True)

ink = dictofDF["VOLCANIC_ROCK_VOUCHER_10500"]
ink_mid = ink["mid_price"].reset_index(drop=True)
inkReturns = [(ink_mid[i] - ink_mid[i - 1]) / ink_mid[i - 1] for i in range(1, len(ink_mid))]

inklines = np.sort([
    0.000494, -0.000247, 0.0, -0.000494, 0.000247, 0.000741,
    -0.000741, 0.000989, -0.000988, -0.00123, 0.00148, -0.00148, 0.00124
])

# --- SMA/LMA Settings ---######################################################################################################################################################################################
sma_window = 2
lma_window = 15
neutral_window = 1
######################################################################################################################################################################################

# --- Trend Classification ---
sma = pd.Series(ink_mid).rolling(window=sma_window).mean()
lma = pd.Series(ink_mid).rolling(window=lma_window).mean()
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
rows = len(inklines) + 2
pMatrix = np.empty((rows,), dtype=object)
trendMatrix = np.empty((rows,), dtype=object)
for i in range(rows):
    pMatrix[i] = []
    trendMatrix[i] = []

# --- Horizon setting ---##################################################################################################################################
future_return_horizon = 5  # adjustable to 5, 10, etc.
######################################################################################################################################################

# --- Fill Matrix with: [return_t+1, cumulative_t+1→t+horizon], and trend label
for i in range(len(inkReturns) - future_return_horizon):
    ink_ret = inkReturns[i]                      # return at t
    ink_ret_t1 = inkReturns[i + 1]               # return at t+1
    ink_ret_thorizon = sum(inkReturns[i + 1 : i + 1 + future_return_horizon])  # cumulative return
    trend = classify_trend(i + future_return_horizon)

    KIndex = find_nearest_index(inklines, ink_ret)

    if KIndex == "MinusAnomaly":
        row = 0
    elif KIndex == "PlusAnomaly":
        row = rows - 1
    else:
        row = int(KIndex) + 1

    pMatrix[row].append((ink_ret_t1, ink_ret_thorizon))
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
output_path = "/Users/jameszhao/Documents.txt"

# --- Delete file if exists ---
if os.path.exists(output_path):
    os.remove(output_path)

# --- Write new output ---
with open(output_path, "w") as f:
    f.write(formatted_output)

print("Saved trend matrix to:", output_path)

# --- Open the file in TextEdit ---
subprocess.run(["open", "-a", "TextEdit", output_path])
"""
Here is my code below : Currently, this takes a given benchmark to map the next current return. 

The price files contain the bid and ask prices of the 10500 volcanic rock vouchers. And also the trade files contain tradePrice of Caesar. Essentially I want instead to generate a optimal matrix for each return of 


"""