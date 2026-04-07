# import json
# from typing import Any
# import numpy as np

# from datamodel import (
#     Listing,
#     Observation,
#     Order,
#     OrderDepth,
#     ProsperityEncoder,
#     Symbol,
#     Trade,
#     TradingState,
# )

# TREND_MATRIX = [
#   [4.96202e-06, -0.000978138, "Downtrend", 0.5],
#   [0.000677672, 0.00103415, "Downtrend", 0.481481],
#   [0.00057431, 0.000268251, "Downtrend", 0.407407],
#   [0.000565665, 0.000579415, "Uptrend", 0.338753],
#   [0.000392119, 0.000379537, "Downtrend", 0.358779],
#   [0.000190713, 0.00021264, "Uptrend", 0.343913],
#   [0.000138015, 0.00018287, "Uptrend", 0.357816],
#   [-6.97943e-06, 8.61103e-06, "Uptrend", 0.352023],
#   [-8.62839e-05, -4.00512e-05, "Uptrend", 0.362555],
#   [-0.000289016, -0.000272709, "Uptrend", 0.341913],
#   [-0.000340186, -0.000331858, "Downtrend", 0.353039],
#   [-0.000352443, -0.000449086, "Downtrend", 0.356589],
#   [-0.000598895, -0.000675809, "Downtrend", 0.448276],
#   [-0.000629367, -0.000590215, "Downtrend", 0.483871],
#   [-0.00195647, -0.00341906, "Neutral", 1.0],
# ]
# KELP_LINES = np.sort([
#     0.000494, -0.000247, 0.0, -0.000494, 0.000247, 0.000741, -0.000741,
#     0.000989, -0.000988, -0.00123, 0.00148, -0.00148, 0.00124
# ])

# def find_nearest_index(array, value):
#     tol = 0.0015
#     if abs(value) > tol:
#         return 0 if value < 0 else len(array) + 1
#     return (np.abs(array - value)).argmin() + 1

# class Logger:
#     def __init__(self) -> None:
#         self.logs = ""
#         self.max_log_length = 3750

#     def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
#         self.logs += sep.join(map(str, objects)) + end

#     def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
#         base_length = len(self.to_json([
#             self.compress_state(state, ""),
#             self.compress_orders(orders),
#             conversions,
#             "",
#             ""
#         ]))
#         remaining_length = self.max_log_length - base_length
#         self.logs = self.logs[-remaining_length:]

#         print(self.to_json([self.compress_state(state, trader_data), self.compress_orders(orders), conversions, trader_data, self.logs]))
#         self.logs = ""

#     def compress_state(self, state: TradingState, trader_data: str) -> Any:
#         return {
#             "traderData": trader_data,
#             "orders": {},
#             "marketState": {s: None for s in state.order_depths},
#             "ownTrades": {},
#             "marketTrades": {},
#             "position": state.position,
#             "observations": {},
#         }

#     def compress_orders(self, orders: dict[Symbol, list[Order]]) -> Any:
#         return {s: [o.__dict__ for o in lst] for s, lst in orders.items()}

#     def to_json(self, obj: Any) -> str:
#         return json.dumps(obj, cls=ProsperityEncoder, separators=(",", ":"))

# logger = Logger()

# class Trader:
#     def __init__(self):
#         self.prev_mid = {}

#     def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
#         result = {}
#         conversions = 0

#         for product in state.order_depths:
#             order_depth = state.order_depths[product]
#             kelporder = []

#             if product != "KELP":
#                 continue

#             if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
#                 continue

#             best_bid = max(order_depth.buy_orders.keys())
#             best_ask = min(order_depth.sell_orders.keys())
#             mid_price = (best_bid + best_ask) / 2

#             # Compute return
#             kelp_ret = 0.0
#             if product in self.prev_mid:
#                 kelp_ret = (mid_price - self.prev_mid[product]) / self.prev_mid[product]
#             self.prev_mid[product] = mid_price

#             index = find_nearest_index(KELP_LINES, kelp_ret)

#             try:
#                 cell = TREND_MATRIX[index]
#                 if isinstance(cell, list) and len(cell) == 3:
#                     predicted_return, trend, confidence = cell
#                 else:
#                     logger.print(f"Unexpected format in TREND_MATRIX[{index}]: {cell} → Fallback to HOLD")
#                     predicted_return, trend, confidence = 0.0, "Neutral", 0.0
#             except Exception as e:
#                 logger.print(f"Index error at {index}: {e} → Fallback to HOLD")
#                 predicted_return, trend, confidence = 0.0, "Neutral", 0.0

#             logger.print(f"KELP return: {kelp_ret:.6f} → Trend: {trend}, Expected: {predicted_return:.6f}, Confidence: {confidence:.2%}")

#             ###################################################################################################################################################





#             parameter = 10






#             ###################################################################################################################################################
#             max_quantity = max(1, int(confidence * parameter))
#             pos = state.position.get(product, 0)
#             CheapestPrice = best_ask
#             HighestPrice = best_bid

#             if trend == "Uptrend":
#                 if predicted_return > 0:
#                     kelporder.append(Order("KELP", CheapestPrice, max_quantity))
#                 else:
#                     kelporder.append(Order("KELP", CheapestPrice, -max_quantity))
#             elif trend == "Downtrend":
#                 if predicted_return < 0:
#                     kelporder.append(Order("KELP", HighestPrice, -max_quantity))
#                 else:
#                     kelporder.append(Order("KELP", CheapestPrice, max_quantity))
#             elif trend == "Neutral":
#                 if predicted_return > 0:
#                     kelporder.append(Order("KELP", CheapestPrice, 0))
#                 elif predicted_return < 0:
#                     kelporder.append(Order("KELP", HighestPrice, 0))

#             result[product] = kelporder

#         logger.flush(state, result, conversions, "")
#         return result, conversions, ""


import json
from typing import Any
import numpy as np

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
  [-0.000160051, -0.000652604, "Uptrend", 0.666667],
  [0.000691861, 0.000827619, "Downtrend", 0.627451],
  [0.000579909, 0.000547219, "Downtrend", 0.517073],
  [0.000562811, 0.000598851, "Downtrend", 0.496608],
  [0.000388259, 0.000385049, "Downtrend", 0.45707],
  [0.000191409, 0.000196477, "Uptrend", 0.440905],
  [0.000143105, 0.000145572, "Downtrend", 0.420711],
  [-6.85009e-06, -8.65492e-06, "Uptrend", 0.42934],
  [-8.87463e-05, -9.14426e-05, "Uptrend", 0.437835],
  [-0.000289262, -0.000283709, "Uptrend", 0.446344],
  [-0.000340774, -0.000307287, "Uptrend", 0.449239],
  [-0.000366881, -0.000361857, "Downtrend", 0.435897],
  [-0.000609036, -0.000574268, "Uptrend", 0.494505],
  [-0.000633429, -0.000546761, "Downtrend", 0.539683],
  [-0.00195647, -0.0024454, "Uptrend", 1.0],
]

INK_LINES = np.sort([
    0.000494, -0.000247, 0.0, -0.000494, 0.000247, 0.000741,
    -0.000741, 0.000989, -0.000988, -0.00123, 0.00148, -0.00148, 0.00124
])

def find_nearest_index(array, value):
    tol = 0.0015
    if abs(value) > tol:
        # If value is outside the tolerance, return an extreme index
        return 0 if value < 0 else len(array) + 1
    # Otherwise, return the index + 1 (to match how TREND_MATRIX is structured)
    return (np.abs(array - value)).argmin() + 1

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            [], # self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

class Trader:
    def __init__(self):
        self.prev_mid = {}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0

        # Process only the product "SQUID_INK"
        for product in state.order_depths:
            if product != "SQUID_INK":
                continue

            order_depth = state.order_depths[product]
            inkorder = []

            # Skip if there are no orders on one side
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
            if return_t1 > 0 and return_th > 0 and trend == "Uptrend":
                inkorder.append(Order(product, best_ask, 2))  # Strong bullish signal
            elif return_t1 < 0 and return_th > 0 and trend == "Uptrend":
                inkorder.append(Order(product, best_ask, -1))  # Adjustment signal
            elif return_t1 > 0 and return_th > 0 and trend == "Downtrend":
                inkorder.append(Order(product, best_ask, 0))  # Weak bullish signal

            if return_t1 < 0 and return_th < 0 and trend == "Downtrend":
                inkorder.append(Order(product, best_bid, -2))  # Strong bearish signal
            elif return_t1 > 0 and return_th < 0 and trend == "Downtrend":
                inkorder.append(Order(product, best_bid, 1))  # Strong bearish signal
            elif return_t1 < 0 and return_th < 0 and trend == "Uptrend":
                inkorder.append(Order(product, best_bid, 0))  # Weak bearish signal

            if trend == "Uptrend":
                inkorder.append(Order(product, best_ask, 0))  # Additional weak uptrend order
            elif trend == "Downtrend":
                inkorder.append(Order(product, best_bid, 0))  # Additional weak downtrend order

            result[product] = inkorder

        logger.flush(state, result, conversions, "")
        return result, conversions, ""
    

# #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  
# #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  
# #
# #         prosperity3bt "/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level2/James/Kelp/SoloJamesKelpOnly.py" 0 1 2 --no-out
# #         prosperity3bt "/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level2/James/Kelp/SoloJamesKelpOnly.py" 0 1 2  --vis