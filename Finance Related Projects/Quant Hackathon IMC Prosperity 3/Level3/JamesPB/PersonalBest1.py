import json
import jsonpickle
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque


################################################################################################################
###----------------------------------      Logger                  --------------------------------------###
################################################################################################################

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
            self.compress_trades(state.market_trades),
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
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])
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
        return json.dumps(value, cls=ProsperityEncoder, separators=(',', ':'))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


################################################################################################################
###----------------------------------      Constants                --------------------------------------###
################################################################################################################

PICNIC_BASKET1 = 'PICNIC_BASKET1'
PICNIC_BASKET2 = 'PICNIC_BASKET2'
JAMS = 'JAMS'
DJEMBES = 'DJEMBES'
CROISSANTS = 'CROISSANTS'

LIMIT = {
    PICNIC_BASKET1: 60,
    PICNIC_BASKET2: 100,
    CROISSANTS: 250,
    JAMS: 350,
    DJEMBES: 60,
}

croissanthistorylength = 83.8247
croissantzthreshold = 1.9986

PARAMS = {
    CROISSANTS: {
        "history_length": croissanthistorylength,
        "z_threshold": croissantzthreshold
    }
}

# Static synthetic basket params (will be overridden dynamically)
synth1Mean = -131.606
synth1Sigma = np.round(29.05 // np.sqrt(1000), decimals=5)
s1Zscore = 0.3394
synth2Mean = 105.417
synth2Sigma = np.round(27.166 // np.sqrt(1000), decimals=5)
s2Zscore = 1.4417
diff_threshold_b1_b2 = 176.8118

# Dynamic inventory factor params
pb1high = 3.8116
pb1mid = 37.2380
pb1low = -9.0092
pb1neg = 7.9516
pb2high = 39.2463
pb2mid = 26.8949
pb2low = 5.3674
pb2neg = -26.1430

djembeshigh = 22.4723
djembesmid = 16.7552
djembeslow = -4.4401
djembesneg = -9.0611

# djemb eHold = 1.7
pb1Hold = 38.6961
pb2Hold = 3.9930


def sortDict(dictionary: dict):
    return {key: dictionary[key] for key in sorted(dictionary)}

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



def get_best_asks_to_fill_WITH_LEVELS(product, state, LeastVolIWant):
    order_depth = state.order_depths[product]
    asks = order_depth.sell_orders  # Dict of price: volume

    # Sort ask prices from best (lowest) to worst (highest)
    sorted_asks = sorted(asks.items())

    # Use at most best and second-best asks
    max_levels = 3
    orders_to_place = []
    filled_volume = 0

    for price, volume in sorted_asks[:max_levels]:
        if filled_volume >= LeastVolIWant:
            break
        volume_to_use = min(volume, LeastVolIWant - filled_volume)
        orders_to_place.append((price, volume_to_use))
        filled_volume += volume_to_use

    return orders_to_place


def get_best_bids_to_fill_WITH_LEVELS(product, state, LeastVolIWant):
    order_depth = state.order_depths[product]
    bids = order_depth.buy_orders  # Dict of price: volume

    # Sort bid prices from best (highest) to worst (lowest)
    sorted_bids = sorted(bids.items(), reverse=True)

    # Use at most best and second-best bids
    max_levels = 3
    orders_to_place = []
    filled_volume = 0

    for price, volume in sorted_bids[:max_levels]:
        if filled_volume >= LeastVolIWant:
            break
        volume_to_use = min(volume, LeastVolIWant - filled_volume)
        orders_to_place.append((price, volume_to_use))
        filled_volume += volume_to_use

    return orders_to_place

# [Other utility functions remain unchanged: VolumeCapability, vwap, mid_price, AskPrice, BidPrice, AskVolume, BidVolume,
# get_best_asks_to_fill, get_best_bids_to_fill, ... ]

################################################################################################################
###----------------------------------    Trader Class               --------------------------------------###
################################################################################################################

class Trader:
    def __init__(self, window_size: int = 1000) -> None:
        self.logger = Logger()
        self.logger.print("Initialized trader with dynamic parameters.")
        self.pos_limits = LIMIT
        self.diff_threshold_b1_b2 = diff_threshold_b1_b2
        self.lot_size = 1
        # Position tracking
        self.previousposition = {p: 0 for p in LIMIT}
        self.position = {p: 0 for p in LIMIT}
        self.positionCounter = {p: 0 for p in LIMIT}
        # Dynamic parameter histories
        self.window_size = window_size
        self.spread1_history = deque(maxlen=self.window_size)
        self.spread2_history = deque(maxlen=self.window_size)
        self.diff_comp_history = deque(maxlen=self.window_size)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in LIMIT.keys()}
        conversions = 0
        trader_data = ""
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        # Update positions and counters
        for product in state.position:
            if state.position[product] == self.previousposition[product]:
                self.positionCounter[product] += 1
            else:
                self.positionCounter[product] = 0
            self.previousposition[product] = state.position[product]
            self.position[product] = state.position[product]

        # Extract best bids, asks, mid_prices
        relevant = [
            CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, PICNIC_BASKET2
        ]
        best_bid = {}
        best_ask = {}
        mid_price = {}
        for prod in relevant:
            od = state.order_depths.get(prod)
            if od:
                best_bid[prod] = max(od.buy_orders.keys()) if od.buy_orders else None
                best_ask[prod] = min(od.sell_orders.keys()) if od.sell_orders else None
                if best_bid[prod] is not None and best_ask[prod] is not None:
                    mid_price[prod] = 0.5 * (best_bid[prod] + best_ask[prod])
                elif best_bid[prod] is not None:
                    mid_price[prod] = best_bid[prod]
                elif best_ask[prod] is not None:
                    mid_price[prod] = best_ask[prod]
                else:
                    mid_price[prod] = None
            else:
                best_bid[prod] = best_ask[prod] = mid_price[prod] = None

        # Composition signal
        if all(mid_price[p] is not None for p in relevant):
            b1 = mid_price[PICNIC_BASKET1]
            b2 = mid_price[PICNIC_BASKET2]
            c = mid_price[CROISSANTS]
            j = mid_price[JAMS]
            d = mid_price[DJEMBES]
            implied_b1 = b2 + 2 * c + j + d
            diff_comp = b1 - implied_b1
            self.logger.print(f"Composition signal: diff_comp={diff_comp:.2f}")
            # Update dynamic histories
            self.spread1_history.append(
                vwap(PICNIC_BASKET1, state) - 1.5 * vwap(PICNIC_BASKET2, state) - vwap(DJEMBES, state)
            )
            self.spread2_history.append(
                vwap(PICNIC_BASKET2, state) - 4 * vwap(CROISSANTS, state) - 2 * vwap(JAMS, state)
            )
            self.diff_comp_history.append(diff_comp)
            # Compute dynamic parameters
            if len(self.spread1_history) > 1:
                synth1Mean = np.mean(self.spread1_history)
                synth1Sigma = np.std(self.spread1_history, ddof=1)
                synth2Mean = np.mean(self.spread2_history)
                synth2Sigma = np.std(self.spread2_history, ddof=1)
                # dynamic threshold = 2 * std of diff_comp
                self.diff_threshold_b1_b2 = np.std(self.diff_comp_history, ddof=1) * 2
                self.logger.print(
                    f"Dynamic synth1Mean={synth1Mean:.2f}, synth1Sigma={synth1Sigma:.2f}; "
                    f"synth2Mean={synth2Mean:.2f}, synth2Sigma={synth2Sigma:.2f}; "
                    f"diff_thresh={self.diff_threshold_b1_b2:.2f}"
                )
            # Signal decision
            signal = 0
            if diff_comp > self.diff_threshold_b1_b2:
                signal = +1
            elif diff_comp < -self.diff_threshold_b1_b2:
                signal = -1
            self.logger.print(f"Signal: {signal}")
        else:
            self.logger.print("Insufficient data for composition signal.")
            signal = 0

        # PICNIC BASKET trading logic remains unchanged
        if PICNIC_BASKET1 in state.order_depths:
            IWantThisMuch = 50
            if diff_comp < self.diff_threshold_b1_b2:
                for price, volume in get_best_bids_to_fill_WITH_LEVELS(PICNIC_BASKET1, state, IWantThisMuch):
                    result[PICNIC_BASKET1].append(Order(PICNIC_BASKET1, price , volume))
            if diff_comp > self.diff_threshold_b1_b2:
                for price, volume in get_best_asks_to_fill_WITH_LEVELS(PICNIC_BASKET1, state, IWantThisMuch):
                    result[PICNIC_BASKET1].append(Order(PICNIC_BASKET1, price, -volume))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
