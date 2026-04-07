import pandas as pd
import numpy as np
import os
from collections import Counter

# --- 1. File setup ---
data_dir = "Level5/r5data/"
price_files = [
    "prices_round_5_day_2.csv",
    "prices_round_5_day_3.csv",
    "prices_round_5_day_4.csv"
]
trade_files = [
    "trades_round_5_day_2.csv",
    "trades_round_5_day_3.csv",
    "trades_round_5_day_4.csv"
]

# --- 2. Load & concatenate price data ---
price_dfs = []
for fn in price_files:
    df = pd.read_csv(os.path.join(data_dir, fn))
    price_dfs.append(df)
prices = pd.concat(price_dfs, ignore_index=True)
prices = prices.sort_values("timestamp").reset_index(drop=True)

# --- 2a. Ensure unique timestamps ---
prices_unique = prices.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

# Add positional index for trend lookup
prices_unique['pos'] = np.arange(len(prices_unique))

# --- 3. Compute moving averages for trend classification ---
sma_window = 15    # short-term moving average window (in ticks)
lma_window = 40   # long-term moving average window (in ticks)
neutral_window = 0  # number of ticks around crossover considered neutral

prices_unique['sma'] = prices_unique['mid_price'].rolling(window=sma_window).mean()
prices_unique['lma'] = prices_unique['mid_price'].rolling(window=lma_window).mean()

# Store arrays for classification to avoid index confusion
sma_arr = prices_unique['sma'].to_numpy()
lma_arr = prices_unique['lma'].to_numpy()

# Precompute crossover points (where sign(sma - lma) changes)
diff = np.nan_to_num(sma_arr - lma_arr)
crossover_points = np.where(np.sign(diff).astype(int)[1:] != np.sign(diff).astype(int)[:-1])[0] + 1

# Trend classifier by position
def classify_trend(idx):
    # Neutral if within 'neutral_window' ticks of a crossover
    # if np.any(np.abs(crossover_points - idx) <= neutral_window):
    #     return 'Neutral'
    return 'Uptrend' if sma_arr[idx] > lma_arr[idx] else 'Downtrend'

# --- 4. Build lookup by timestamp ---
# Set index to timestamp for join, keep 'pos'
lookup = prices_unique.set_index('timestamp')[['pos']]

# --- 5. Load & concatenate trade data ---
trade_dfs = []
for fn in trade_files:
    df = pd.read_csv(os.path.join(data_dir, fn))
    trade_dfs.append(df)
trades = pd.concat(trade_dfs, ignore_index=True)

# --- 6. Filter for Caesar's buys ---
caesar_trades = trades[trades['buyer'] == 'Gina'].copy()

# --- 7. Merge trades with price info ---
caesar_trades = caesar_trades.join(lookup, on='timestamp', how='inner')

# --- 8. Classify trend at each trade based on 'pos' ---
caesar_trades['trend'] = caesar_trades['pos'].apply(classify_trend)

# --- 9. Count trades by trend ---
trend_counts = Counter(caesar_trades['trend'])
total = sum(trend_counts.values())

print("Total Caesar buys analyzed:", total)
for trend, count in trend_counts.items():
    pct = count / total * 100 if total else 0
    print(f"{trend}: {count} trades ({pct:.2f}%)")

# --- 10. (Optional) Save detailed output ---
out_path = "/Users/jameszhao/Downloads/caesar_buys_trend_summary.csv"
caesar_trades.to_csv(out_path, index=False)
print(f"Detailed trades with trend labels saved to {out_path}")
