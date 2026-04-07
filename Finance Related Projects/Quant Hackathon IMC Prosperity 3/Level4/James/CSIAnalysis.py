"""

ΔMacaron_t = β₀ + β₁·ΔSugar_t + β₂·I(Sunlight_t < k) + ε_t

"""
#!/usr/bin/env python3
"""
CSIAnalysis.py  – critical‑sunlight‑index estimator (and visualiser)

Example:
    python CSIAnalysis.py \
        --data-dir "/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level4/round-4-island-data-bottle" \
        --plot
"""

import argparse, glob, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

# ------------- command‑line args -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=".", help="folder with CSVs")
parser.add_argument("--plot", action="store_true",
                    help="draw & save SSR‑vs‑k chart (matplotlib)")
args = parser.parse_args()
DATA_DIR = Path(args.data_dir).expanduser()

# ------------------- settings --------------------
OBS_GLOB   = DATA_DIR / "observations_round_4_day_*.csv"
PRICE_GLOB = DATA_DIR / "prices_round_4_day_*.csv"
MAC_NAME   = "MAGNIFICENT_MACARONS"
RET_LAG    = 1
ROLLING_Z  = 50
GRID_MIN, GRID_MAX, GRID_STEP = 20.0, 65.0, 0.05
# -------------------------------------------------

# ===============  data loaders  ==================
def load_observations():
    frames = []
    for fp in sorted(glob.glob(str(OBS_GLOB))):
        day = int(Path(fp).stem.split('_')[-1])   # …day_2.csv → 2
        df  = pd.read_csv(fp,
                          usecols=["timestamp", "sugarPrice", "sunlightIndex"])
        df["day"]  = day
        df["time"] = day * 1_000_000 + df["timestamp"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[
        ["time", "sugarPrice", "sunlightIndex"]
    ]

def load_prices():
    frames = []
    for fp in sorted(glob.glob(str(PRICE_GLOB))):
        day = int(Path(fp).stem.split('_')[-1])
        df  = pd.read_csv(fp, sep=";")
        mac = df.loc[df["product"] == MAC_NAME, ["timestamp", "mid_price"]]
        mac["day"]  = day
        mac["time"] = day * 1_000_000 + mac["timestamp"]
        frames.append(mac.rename(columns={"mid_price": "mac_mid"})
                        [["time", "mac_mid"]])
    return pd.concat(frames, ignore_index=True)

def make_panel():
    panel = load_prices().merge(load_observations(), on="time").sort_values("time")
    # k‑step log‑returns
    panel["mac_ret"]   = np.log(panel["mac_mid"]).diff(RET_LAG)
    panel["sugar_ret"] = np.log(panel["sugarPrice"]).diff(RET_LAG)
    panel.dropna(inplace=True)

    # rolling Z‑score (helps numeric conditioning, optional)
    for col in ("mac_ret", "sugar_ret"):
        panel[col] = (panel[col] - panel[col].rolling(ROLLING_Z).mean()) / \
                     panel[col].rolling(ROLLING_Z).std()
    panel.dropna(inplace=True)
    return panel

# ===============  estimation utils  ===============
def ssr_for_k(df: pd.DataFrame, k: float) -> float:
    """Return sum of squared residuals for a given candidate break k."""
    X = np.column_stack([
        df["sugar_ret"].values,
        (df["sunlightIndex"].values < k).astype(int)
    ])
    X = sm.add_constant(X)
    y = df["mac_ret"].values
    return sm.OLS(y, X).fit().ssr

def estimate_csi(df: pd.DataFrame):
    grid = np.arange(GRID_MIN, GRID_MAX + GRID_STEP / 2, GRID_STEP)
    ssr  = [ssr_for_k(df, k) for k in tqdm(grid, desc="grid‑search")]
    best_k = grid[int(np.argmin(ssr))]
    return best_k, grid, ssr

# ==================== main ========================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    panel = make_panel()
    csi, grid, ssr = estimate_csi(panel)

    print(f"\nEstimated CSI (all 3 days) = {csi:.2f}\n")

    # ---------- optional plotting -----------------
    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(grid, ssr, label="SSR(k)")
        plt.axvline(csi, linestyle="--", label=f"CSI = {csi:.2f}")
        plt.xlabel("Sunlight‑index candidate k")
        plt.ylabel("Sum of squared residuals")
        plt.title("Threshold‑regression score vs. candidate CSI")
        plt.legend()
        plt.tight_layout()

        out_png = DATA_DIR / "CSI_SSR_curve.png"
        plt.savefig(out_png, dpi=150)
        print(f"Chart saved → {out_png}")
        plt.show()
