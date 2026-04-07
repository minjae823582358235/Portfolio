import pandas as pd
import numpy as np
import os
from collections import Counter

# --- PARAMETERS ---
buyer_name = "Penelope"       # pick your buyer
horizon = 20            # ticks in advance to classify trend

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
price_dfs = [pd.read_csv(os.path.join(data_dir, fn)) for fn in price_files]
prices = pd.concat(price_dfs, ignore_index=True)
prices = prices.sort_values("timestamp").reset_index(drop=True)

# --- 2a. Ensure unique timestamps and add positional index ---
prices_unique = prices.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
prices_unique['pos'] = np.arange(len(prices_unique))

# --- 3. Compute moving averages ---
sma_window = 7    # short-term window
lma_window = 20    # long-term window
prices_unique['sma'] = prices_unique['mid_price'].rolling(window=sma_window).mean()
prices_unique['lma'] = prices_unique['mid_price'].rolling(window=lma_window).mean()

# Convert to numpy arrays for fast classification
sma_arr = prices_unique['sma'].to_numpy()
lma_arr = prices_unique['lma'].to_numpy()

# Trend classification function

def classify_trend(idx):
    if idx < 0 or idx >= len(sma_arr):
        return np.nan
    return 'Uptrend' if sma_arr[idx] > lma_arr[idx] else 'Downtrend'

# --- 4. Build timestamp-to-position lookup ---
lookup = prices_unique.set_index('timestamp')['pos']

# --- 5. Load & concatenate trade data ---
trade_dfs = [pd.read_csv(os.path.join(data_dir, fn)) for fn in trade_files]
trades = pd.concat(trade_dfs, ignore_index=True)

# --- 6. Filter for buyer ---
buyer_trades = trades[trades['buyer'] == buyer_name].copy()

# --- 7. Annotate with current position and future position ---
buyer_trades['pos'] = buyer_trades['timestamp'].map(lookup)
buyer_trades = buyer_trades.dropna(subset=['pos']).copy()
buyer_trades['pos'] = buyer_trades['pos'].astype(int)
buyer_trades['future_pos'] = buyer_trades['pos'] + horizon

# --- 8. Classify future trend for each trade ---
buyer_trades['future_trend'] = buyer_trades['future_pos'].apply(classify_trend)

# --- 9. Count trends ---
trend_counts = Counter(buyer_trades['future_trend'].dropna())
total = sum(trend_counts.values())

print(f"Buyer: {buyer_name}, Horizon: {horizon} ticks")
print(f"Total trades analyzed: {total}")
for trend, count in trend_counts.items():
    pct = count / total * 100 if total else 0
    print(f"{trend}: {count} ({pct:.2f}%)")

# --- 10. Save detailed results ---
out_path = os.path.join(os.getcwd(), f"{buyer_name}_trend_{horizon}ticks.csv")
buyer_trades[['timestamp','pos','future_pos','future_trend']].to_csv(out_path, index=False)
print(f"Results saved to {out_path}")
