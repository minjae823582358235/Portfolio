import json
from typing import Any
import numpy as np
import os

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

# ----------------- Bayesian Optimization Implementation -----------------
# Read parameters from environment variables with defaults if not set
global_window = int(os.environ.get("global_window", "20"))
window_multiplier = float(os.environ.get("window_multiplier", "0.75"))
threshold = float(os.environ.get("threshold", "1"))
volume = int(os.environ.get("volume", "50"))

# ------------------------------------------------------------------------

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
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
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
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
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
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
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()



####################################################################################

RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = 'KELP'
SQUID_INK = 'SQUID_INK'
CROISSANTS = 'CROISSANTS'
PICNIC_BASKET1='PICNIC_BASKET1'
PICNIC_BASKET2='PICNIC_BASKET2'
JAMS='JAMS'
DJEMBES='DJEMBES'
VOLCANIC_ROCK='VOLCANIC_ROCK'
VOLCANIC_ROCK_VOUCHER_9500='VOLCANIC_ROCK_VOUCHER_9500'
VOLCANIC_ROCK_VOUCHER_9750='VOLCANIC_ROCK_VOUCHER_9750'
VOLCANIC_ROCK_VOUCHER_10000='VOLCANIC_ROCK_VOUCHER_10000'
VOLCANIC_ROCK_VOUCHER_10250='VOLCANIC_ROCK_VOUCHER_10250'
VOLCANIC_ROCK_VOUCHER_10500='VOLCANIC_ROCK_VOUCHER_10500'

LIMIT = {
    RAINFOREST_RESIN: 50,
    KELP: 50,
    SQUID_INK: 50,
    CROISSANTS: 250,
    PICNIC_BASKET1: 60,
    PICNIC_BASKET2: 100,
    JAMS: 350,
    DJEMBES: 60,
    VOLCANIC_ROCK: 400,
    VOLCANIC_ROCK_VOUCHER_9500: 200,
    VOLCANIC_ROCK_VOUCHER_9750: 200,
    VOLCANIC_ROCK_VOUCHER_10000: 200,
    VOLCANIC_ROCK_VOUCHER_10250: 200,
    VOLCANIC_ROCK_VOUCHER_10500: 200,
}

## GENERAL FUNCTIONS #######################
def sortDict(dictionary:dict):
    return {key: dictionary[key] for key in sorted(dictionary)}

def VolumeCapability(product, mode,state:TradingState):
    if mode == "buy":
        return LIMIT[product] - state.position[product]
    if mode == "sell":
        return state.position[product] + LIMIT[product]

def vwap(product: str,state:TradingState) -> float:
    vwap = 0
    total_amt = 0

    for prc, amt in state.order_depths[product].buy_orders.items():
        vwap += prc * amt
        total_amt += amt

    for prc, amt in state.order_depths[product].sell_orders.items():
        vwap += prc * abs(amt)
        total_amt += abs(amt)

    vwap /= total_amt
    return np.round(vwap,decimals=5)

def mid_price(order_depth:OrderDepth) -> float:
    # Compute a mid-price
    if len(order_depth.sell_orders) != 0:
        m1 = 0
        n1 = 0
        for best_ask, best_ask_amount in order_depth.sell_orders.items():
            m1 += best_ask * best_ask_amount
            n1 += best_ask_amount
        m1 = m1 / n1
    else:
        m1 = 0

    if len(order_depth.buy_orders) != 0:
        m2 = 0
        n2 = 0
        for best_bid, best_bid_amount in order_depth.buy_orders.items():
            m2 += best_bid * best_bid_amount
            n2 += best_bid_amount
        m2 = m2 / n2
    else:
        m2 = 0
        
    # Use whichever is available, or the average if both are
    return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)

def AskPrice(product, mode,state:TradingState):  # how much a seller is willing to sell for
    if mode == "max":
        if product not in set(state.order_depths.keys()):
            return 0  # FREAKY
        return max(set(state.order_depths[product].sell_orders.keys()))
    if mode == "min":
        if product not in set(state.order_depths.keys()):
            return 0  # FREAKY
        return min(set(state.order_depths[product].sell_orders.keys()))


def BidPrice(product, mode,state):  # how much a buyer is willing to buy for
    if mode == "max":
        if product not in set(state.order_depths.keys()):  # FREAKY
            return 1000000  # FREAKY
        return max(set(state.order_depths[product].buy_orders.keys()))
    if mode == "min":
        if product not in set(state.order_depths.keys()):  # FREAKY
            return 1000000  # FREAKY
        return min(set(state.order_depths[product].buy_orders.keys()))

def AskVolume(
    product, mode,state):  # ITS FOR THE HIGHEST/LOWEST PRICE NOT VOLUME!!
    if product not in set(state.order_depths.keys()):  # FREAKY
        return 100
    if mode == "max":
        return abs(
            state.order_depths[product].sell_orders[
                AskPrice(product, mode="max",state=state)
            ]
        )
    if mode == "min":
        return abs(
            state.order_depths[product].sell_orders[
                AskPrice(product, mode="min",state=state)
            ]
        )

def BidVolume(
    product, mode,state):  # ITS FOR THE HIGHEST/LOWEST PRICE NOT VOLUME!!
    if product not in set(state.order_depths.keys()):
        return 100  # FREAKY
    if mode == "max":
        return abs(
            state.order_depths[product].buy_orders[
                BidPrice(product, mode="max",state=state)
            ]
        )
    if mode == "min":
        return abs(
            state.order_depths[product].buy_orders[
                BidPrice(product, mode="min",state=state)
            ]
        )




###############################################################
# ---------------------- Trading Logic ----------------------##
###############################################################



class Trader:
    def __init__(
            self, 
            global_window = 48.3266, 
            window_multiplier = 0.5385, 
            threshold = 1.9651, 
            volume = 8.7873 
            ):
    
        self.global_window = global_window
        self.window_multiplier = window_multiplier
        self.threshold = threshold 
        self.volume = int(volume)
        self.midprice_window = []
        self.bid_volume_window = []
        self.ask_volume_window = []


    def mean_reversion_signal(self, midprices, current_mid):
        if len(midprices) < self.global_window:
            return None
        mean = np.mean(midprices)
        std = np.std(midprices)
        if std == 0:
            return None
        z = (current_mid - mean) / std
        if z < -self.threshold:
            return 1
        elif z > self.threshold:
            return -1
        return 0

    # def momentum_signal(self, midprices):
    #     window = int(self.global_window * self.window_multiplier)
    #     if len(midprices) < window:
    #         return None
    #     past = midprices[-window]
    #     current = midprices[-1]
    #     if current > past:
    #         return 1
    #     elif current < past:
    #         return -1
    #     return 0
    

    def momentum_signal(self, midprices):
        SmoothingWindow = 5 
        window = int(self.global_window * self.window_multiplier)
        if len(midprices) < window:
            return None
        past = midprices[-window]
        past = np.mean(midprices[-window - SmoothingWindow: -window])
        current = midprices[-1]
        if current > past:
            return 1
        elif current < past:
            return -1
        return 0

    def volume_pressure_signal(self, order_depth: OrderDepth):
        current_bid_volume = sum(order_depth.buy_orders.values())
        current_ask_volume = sum(order_depth.sell_orders.values())

        self.bid_volume_window.append(current_bid_volume)
        self.ask_volume_window.append(current_ask_volume)

        if len(self.bid_volume_window) > int(self.global_window):
            self.bid_volume_window.pop(0)
        if len(self.ask_volume_window) > int(self.global_window):
            self.ask_volume_window.pop(0)

        mean_bid = np.mean(self.bid_volume_window)
        mean_ask = np.mean(self.ask_volume_window)

        if current_bid_volume > mean_bid:
            return 1  # Buy signal
        elif current_ask_volume > mean_ask:
            return -1  # Sell signal
        return 0  # Hold

    def combined_signal(self, midprices, current_mid, order_depth):
        mr = self.mean_reversion_signal(midprices, current_mid)
        mo = self.momentum_signal(midprices)
        vp = self.volume_pressure_signal(order_depth)


      
        if mr == 1:
            return "MR BUY"
        elif mr == -1:
            return "MR SELL"
        elif mr == 0 and mo == 1 and vp == 1:
            return "BUY"
        elif mr == 0 and mo == -1 and vp == -1:
            return "SELL"
        else: 
            return "HOLD"
    



###############################################################
##---------------------- Buying Logic -----------------------##
###############################################################

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        symbol = "KELP"
        trader_data = ""

        if symbol in state.order_depths:
            order_depth = state.order_depths[symbol]
            current_mid = mid_price(order_depth)

            if current_mid is not None:
                self.midprice_window.append(current_mid)
                if len(self.midprice_window) > int(self.global_window):
                    self.midprice_window.pop(0)

        signal = self.combined_signal(self.midprice_window, current_mid, order_depth)
        orders = []

        if signal == "MR BUY":
            best_ask = AskPrice(KELP, mode="max", state=state) if order_depth.sell_orders else None
            if best_ask:
                buy_volume = VolumeCapability(KELP, mode="buy", state=state) 
                orders.append(Order(symbol, best_ask, buy_volume))

        elif signal == "MR SELL":
            best_bid = BidPrice(KELP, mode="min", state=state) if order_depth.buy_orders else None
            if best_bid:
                sell_volume = VolumeCapability(KELP, mode="sell", state=state) 
                orders.append(Order(symbol, best_bid, -sell_volume))

        elif signal == "BUY":
            best_ask = AskPrice(KELP, mode="max", state=state) if order_depth.sell_orders else None
            if best_ask:
                buy_volume = min(AskVolume(KELP, mode="max", state=state), self.volume)
                orders.append(Order(symbol, best_ask, buy_volume))

        elif signal == "SELL":
            best_bid = BidPrice(KELP, mode="min", state=state) if order_depth.buy_orders else None
            if best_bid:
                sell_volume = min(BidVolume(KELP, mode="min", state=state), self.volume)
                orders.append(Order(symbol, best_bid, -sell_volume))

        result[symbol] = orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data 

