import pandas as pd
import numpy as np
import os

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

# --- 2a. Ensure unique timestamps for lookup ---
#    If the same timestamp appears multiple times, take the first mid_price
prices_unique = prices.drop_duplicates(subset="timestamp", keep="first")

# --- 3. Build a price lookup keyed by timestamp ---
price_lookup = prices_unique.set_index("timestamp")["mid_price"]

# --- 4. Load & concatenate trade data ---
trade_dfs = []
for fn in trade_files:
    df = pd.read_csv(os.path.join(data_dir, fn))
    trade_dfs.append(df)
trades = pd.concat(trade_dfs, ignore_index=True)

# --- 5. Filter for Caesar's buys of VOLCANIC_ROCK_VOUCHER_10500 ---
mask = (
    (trades["buyer"] == "Pablo") &
    (trades["symbol"] == "PICNIC_BASKET1") 
)
caesar_trades = trades.loc[mask, ["timestamp", "price"]].copy()

# --- 6. Compute the +100‑tick return for each trade ---
returns = []
for t in caesar_trades["timestamp"]:
    if t in price_lookup.index and (t + 100) in price_lookup.index:
        p0 = price_lookup.at[t]
        p1 = price_lookup.at[t + 100]
        # simple return
        returns.append((p1 - p0) / p0)
    # missing ticks are silently skipped

# --- 7. Tally positive / negative / zero returns ---
total = len(returns)
if total > 0:
    positives = sum(1 for r in returns if r > 0)
    negatives = sum(1 for r in returns if r < 0)
    zeros     = sum(1 for r in returns if r == 0)

    pos_pct = positives / total * 100
    neg_pct = negatives / total * 100
    zero_pct= zeros     / total * 100

    print(f"Total trades used: {total}")
    print(f"Positive returns: {positives} ({pos_pct:.2f}%)")
    print(f"Negative returns: {negatives} ({neg_pct:.2f}%)")
    print(f"Zero returns:      {zeros} ({zero_pct:.2f}%)")
else:
    print("No valid +100‑tick returns found.")

# --- 8. Write detailed returns to a txt file ---
out_df = pd.DataFrame({
    "trade_timestamp": caesar_trades["timestamp"].iloc[:total],
    "return_100": returns
})

output_path = "/Users/jameszhao/Downloads/CaesarPro.txt"
out_df.to_csv(output_path, sep="\t", index=False)
print(f"Detailed returns written to {output_path}")
