import json
from typing import Any
import numpy as np

# ----------------- Bayesian Optimization Implementation -----------------
import os

# Read parameters from environment variables with defaults if not set

# Resin
strong = int(os.environ.get("strong", "1"))
weak = int(os.environ.get("weak", "8"))

# ------------------------------------------------------------------------

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
TREND_MATRIX = [
  [0.000288804, 0.000392748, "Downtrend", 0.605155],
  [0.00024443, 0.000173474, "Downtrend", 0.552716],
  [0.000241727, 0.000161882, "Downtrend", 0.579588],
  [0.000190424, 0.000178946, "Downtrend", 0.533973],
  [0.000212563, 0.000202431, "Downtrend", 0.526407],
  [8.66403e-05, 5.60804e-05, "Downtrend", 0.500864],
  [9.11103e-05, 4.61627e-05, "Downtrend", 0.486336],
  [-2.32912e-05, -7.54664e-05, "Uptrend", 0.488874],
  [-5.39797e-05, -6.53021e-05, "Uptrend", 0.492974],
  [-0.000137702, -0.000123757, "Uptrend", 0.517963],
  [-0.000136467, -2.81602e-05, "Uptrend", 0.545606],
  [-0.000238301, -0.000159826, "Uptrend", 0.54728],
  [-0.000258421, -0.000222583, "Uptrend", 0.486204],
  [-0.000263593, -6.96923e-05, "Uptrend", 0.576052],
  [-0.000170038, -0.000109998, "Uptrend", 0.554697],
]

INK_LINES = np.sort([
    0.000494, -0.000247, 0.0, -0.000494, 0.000247, 0.000741,
    -0.000741, 0.000989, -0.000988, -0.00123, 0.00148, -0.00148, 0.00124
])

def find_nearest_index(array, value):
    tol = 0.0015
    if abs(value) > tol:
        return 0 if value < 0 else len(array) + 1
    return (np.abs(array - value)).argmin() + 1

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750
        
    

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            ""
        ]))
        remaining_length = self.max_log_length - base_length
        self.logs = self.logs[-remaining_length:]
        print(self.to_json([
            self.compress_state(state, trader_data),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> Any:
        return {
            "traderData": trader_data,
            "orders": {},
            "marketState": {s: None for s in state.order_depths},
            "ownTrades": {},
            "marketTrades": {},
            "position": state.position,
            "observations": {},
        }

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> Any:
        return {s: [o.__dict__ for o in lst] for s, lst in orders.items()}

    def to_json(self, obj: Any) -> str:
        return json.dumps(obj, cls=ProsperityEncoder, separators=(",", ":"))

logger = Logger()

class Trader:
    def __init__(self):
        self.prev_mid = {}
        self.strong = strong
        self.weak = weak 
  

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0

        # Loop over products in the current state
        for product in state.order_depths:
            if product != "SQUID_INK":
                continue

            order_depth = state.order_depths[product]
            inkorder = []

            # Skip if one side of the order book is empty.
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2

            ink_ret = 0.0
            if product in self.prev_mid and self.prev_mid[product] != 0:
                ink_ret = (mid_price - self.prev_mid[product]) / self.prev_mid[product]
            self.prev_mid[product] = mid_price

            index = find_nearest_index(INK_LINES, ink_ret)

            try:
                cell = TREND_MATRIX[index]
                if isinstance(cell, list) and len(cell) == 4:
                    return_t1, return_th, trend, confidence = cell
                else:
                    logger.print(f"Bad matrix entry at {index}: {cell}")
                    return_t1, return_th, trend, confidence = 0.0, 0.0, "Neutral", 0.0
            except Exception as e:
                logger.print(f"Matrix index error {index}: {e}")
                return_t1, return_th, trend, confidence = 0.0, 0.0, "Neutral", 0.0

            logger.print(
                f"INK return: {ink_ret:.6f} → t+1: {return_t1:.6f}, t+h: {return_th:.6f}, Trend: {trend}, Confidence: {confidence:.2%}"
            )

            # --- Fixed Quantity for Grid Testing ---
            if return_t1 > 0 and trend == "Uptrend":
                inkorder.append(Order(product, best_ask, self.strong))  # Strong bullish signal
            elif return_t1 < 0 and trend == "Uptrend":
                inkorder.append(Order(product, best_ask, -self.weak))  # Mid signal

            if return_t1 < 0 and trend == "Downtrend":
                inkorder.append(Order(product, best_bid, -self.strong))  # Strong bearish signal
            if return_t1 > 0 and trend == "Downtrend":
                inkorder.append(Order(product, best_bid, self.weak))  # Mid bearish signa

            if trend == "Uptrend":
                inkorder.append(Order(product, best_ask, 0))  # Additional weak uptrend order
            elif trend == "Downtrend":
                inkorder.append(Order(product, best_bid, 0))  # Addsitional weak downtrend order

            result[product] = inkorder

        logger.flush(state, result, conversions, "")
        return result, conversions, ""
    

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  
#         prosperity3bt "/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level2/James/Ink/SoloJamesInkOnly.py" 0 1 2 --no-out
#         prosperity3bt "/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level2/James/Ink/SoloJamesInkOnly.py" 0 1 2  --vis



