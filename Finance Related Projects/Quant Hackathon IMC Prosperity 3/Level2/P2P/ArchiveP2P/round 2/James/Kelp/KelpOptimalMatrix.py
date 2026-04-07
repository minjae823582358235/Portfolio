import pandas as pd
import numpy as np
import os

# File paths
data_dir = "/Users/jameszhao/Documents/IMC3Round1"


day2_file = os.path.join(data_dir, "day2.csv")
day1_file = os.path.join(data_dir, "day1.csv")
day0_file = os.path.join(data_dir, "day0.csv")

# Load the provided CSVs
day2 = pd.read_csv(day2_file, sep=";")
day1 = pd.read_csv(day1_file, sep=";")
day0 = pd.read_csv(day0_file, sep=";")

# Combine and align timestamps
DFArr = [day2, day1, day0]
commodities = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]
dictofDF = {}

for i, df in enumerate(DFArr):
    for product in commodities:
        df_product = df[df["product"] == product].copy()
        df_product["timestamp"] += i * 1_000_000
        if i == 0:
            dictofDF[product] = df_product
        else:
            dictofDF[product] = pd.concat([dictofDF[product], df_product], ignore_index=True)

# Extract mid_prices
kelp = dictofDF["KELP"]
ink = dictofDF["SQUID_INK"]
kelp_mid = kelp["mid_price"].reset_index(drop=True)
ink_mid = ink["mid_price"].reset_index(drop=True)

kelpReturns = [(kelp_mid[i] - kelp_mid[i - 1]) / kelp_mid[i - 1] for i in range(1, len(kelp_mid))]
inkReturns = [(ink_mid[i] - ink_mid[i - 1]) / ink_mid[i - 1] for i in range(1, len(ink_mid))]

# Return bins
kelplines = np.sort([
    0.000494, -0.000247, 0.0, -0.000494, 0.000247, 0.000741, -0.000741,
    0.000989, -0.000988, -0.00123, 0.00148, -0.00148, 0.00124
])
inklines = np.round([
    -4.959e-03, -4.706e-03, -4.520e-03, -4.263e-03, -3.991e-03, -3.776e-03,
    -3.246e-03, -3.014e-03, -2.744e-03, -2.512e-03, -2.239e-03, -2.008e-03,
    -1.748e-03, -1.511e-03, -1.258e-03, -1.013e-03, -7.590e-04, -5.100e-04,
    -2.540e-04, 0.000e00, 3.000e-06, 9.000e-06, 2.540e-04, 5.090e-04,
    7.600e-04, 1.014e-03, 1.262e-03, 1.504e-03, 1.736e-03, 2.021e-03,
    2.243e-03, 2.513e-03, 2.754e-03, 3.012e-03, 3.251e-03, 3.536e-03,
    3.743e-03, 4.014e-03, 4.229e-03, 4.528e-03, 4.685e-03, 4.913e-03
], decimals=6)

def find_bin_index(arr, val):
    return int((np.abs(arr - val)).argmin())

# Matrix initialization
rows, cols = len(inklines) + 2, len(kelplines) + 2
pMatrix = np.empty((rows, cols), dtype=object)
for i in range(rows):
    for j in range(cols):
        pMatrix[i, j] = []

# Fill pMatrix
for i in range(len(kelpReturns) - 1):
    ink_ret = inkReturns[i]
    kelp_ret = kelpReturns[i]
    kelp_next = kelpReturns[i + 1]

    i_idx = find_bin_index(inklines, ink_ret) + 1
    k_idx = find_bin_index(kelplines, kelp_ret) + 1

    pMatrix[i_idx][k_idx].append(kelp_next)

# SMA/LMA Trend Classification
####################################################################################







sma_window = 2
lma_window = 15
neutral_window = 1
#Best: 2, 15, 1
#the bigger the difference between SMA, LMA, Trades become less likely. However in downtrend there trades well? 
#fix uptrend trader. 
#Higher the neutralwindow, less trades occur. 
#If you put netralwindow = 0: Dont do that 


#Careful: 106 and 107 are commented now 




####################################################################################
sma = pd.Series(kelp_mid).rolling(window=sma_window).mean()
lma = pd.Series(kelp_mid).rolling(window=lma_window).mean()
crossover_points = np.where(np.sign(sma - lma).diff().fillna(0) != 0)[0]

def classify_trend(idx):
    #if idx >= len(sma) or pd.isna(sma[idx]) or pd.isna(lma[idx]):
        #return "Neutral"
    if np.any(np.abs(crossover_points - idx) <= neutral_window):
        return "Neutral"
    return "Uptrend" if sma[idx] > lma[idx] else "Downtrend"

# Construct trend matrix
trend_matrix = np.empty((rows, cols), dtype=object)

for i in range(rows):
    for j in range(cols):
        cell_data = pMatrix[i, j]
        if not cell_data:
            trend_matrix[i, j] = [None, "Neutral"]
        else:
            mean_val = np.mean(cell_data)
            trend_votes = {"Uptrend": 0, "Downtrend": 0, "Neutral": 0}
            for t in range(len(cell_data)):
                trend = classify_trend(t + 1)
                trend_votes[trend] += 1
            dominant = max(trend_votes, key=trend_votes.get)
            trend_matrix[i, j] = [float(f"{mean_val:.6g}"), dominant]

# Write TREND_MATRIX to file
def format_trend_matrix(matrix):
    lines = ["TREND_MATRIX = ["]
    for row in matrix:
        lines.append("  [")
        for cell in row:
            val_str = "None" if cell[0] is None or np.isnan(cell[0]) else f"{cell[0]:.6g}"
            lines.append(f"    [{val_str}, \"{cell[1]}\"],")
        lines.append("  ],")
    lines.append("]")
    return "\n".join(lines)

formatted_output = format_trend_matrix(trend_matrix)
output_path =  "/Users/jameszhao/Documents/IMC3Round1/Kelp/kelp_trend_matrix_output.txt"

with open(output_path, "w") as f:
    f.write(formatted_output)

output_path

