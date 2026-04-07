import json
import jsonpickle
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import statistics
from statistics import NormalDist
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

kelptw = 18.7017
kelppositionlimit=50
kelpminvolume = 21.7543
kelpspreadedge = 1.0760
squidz = 1.9572
squidhistorylength = 700.9250
SQrsi_window = 106
SQrsi_overbought = 52
SQrsi_oversold = 41
squidrsitradesize = 36.3892
croissanthistorylength = 83.8247
croissantzthreshold = 1.9986

PARAMS = {
    RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 0.5,
        "join_edge": 2,
        "default_edge": 2,
        "soft_position_limit": 45,
    },
    KELP: {
        "take_width": kelptw,
        "position_limit": kelppositionlimit,
        "min_volume_filter": kelpminvolume,
        "spread_edge": kelpspreadedge,
        "default_fair_method": "vwap_with_vol_filter",
    },
    # New parameters for the SQUID_INK mean reversion strategy
    SQUID_INK: {
        "rsi_window": SQrsi_window,
        "rsi_overbought": SQrsi_overbought,
        "rsi_oversold": SQrsi_oversold,
    },
    CROISSANTS: {
        "history_length": croissanthistorylength,  # Number of mid-price datapoints to use for z-score calculation.
        "z_threshold": croissantzthreshold        # Threshold for trading.
    },
    VOLCANIC_ROCK: {
        "starting_time_to_expiry": 8 / 365,
        "std_window": 10,
    },
    VOLCANIC_ROCK_VOUCHER_9500: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 9500,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_9750: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 9750,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_10000: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 10000,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_10250: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 10250,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_10500: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 10500,
        "std_window": 10,
        "implied_volatility": 0.16,
    }
}



### PB Synthetic Basket Parameters #######################
synth1Mean = -131.606  # PB1 is usually cheaper than PB2
synth1Sigma = np.round(29.05 // np.sqrt(1000), decimals=5)
s1Zscore = 0.3394  # TODO OPTIMISE!!!!!! 1 works pretty well
synth2Mean = 105.417
synth2Sigma = np.round(27.166 // np.sqrt(1000), decimals=5)
s2Zscore = 1.4417  # TODO OPTIMISE!!!!!!
diff_threshold_b1_b2 = 176.8118
diff_threshold_b1 = 176.8118
#### Kelp Squink Pairs Trade Parameters

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

def get_best_asks_to_fill_WITH_LEVELS(product, state, LeastVolIWant):
    order_depth = state.order_depths[product]
    asks = order_depth.sell_orders  # price: negative volume (since it's the ask side)

    sorted_asks = sorted(asks.items())  # Lowest price first

    max_levels = 2
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
    bids = order_depth.buy_orders  # price: positive volume

    sorted_bids = sorted(bids.items(), reverse=True)  # Highest price first

    max_levels = 2
    orders_to_place = []
    filled_volume = 0

    for price, volume in sorted_bids[:max_levels]:
        if filled_volume >= LeastVolIWant:
            break
        volume_to_use = min(volume, LeastVolIWant - filled_volume)
        orders_to_place.append((price, volume_to_use))
        filled_volume += volume_to_use

    return orders_to_place

def desired_volume(product, state):
        pos = state.position.get(product, 0)
        limit = LIMIT[product]
        # The closer to the limit, the smaller the volume
        max_vol = 50
        scale = max(1, int(max_vol * (1 - abs(pos) / limit)))
        return scale
##############################################################


# ---------------------------
# RAINFOREST_RESIN Strategy Functions (from round1rsi)
# ---------------------------
def resin_take_orders(
    order_depth: OrderDepth, fair_value: float, position: int, position_limit: int
) -> Tuple[List[Order], int, int]:
    orders = []
    buy_order_volume = 0
    sell_order_volume = 0
    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask < fair_value:
            quantity = min(best_ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order(RAINFOREST_RESIN, best_ask, quantity))
                buy_order_volume += quantity
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        if best_bid > fair_value:
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order(RAINFOREST_RESIN, best_bid, -quantity))
                sell_order_volume += quantity
    return orders, buy_order_volume, sell_order_volume

def resin_clear_orders(
    order_depth: OrderDepth,
    position: int,
    fair_value: float,
    position_limit: int,
    buy_order_volume: int,
    sell_order_volume: int,
) -> Tuple[List[Order], int, int]:
    orders = []
    position_after_take = position + buy_order_volume - sell_order_volume
    fair_for_bid = math.floor(fair_value)
    fair_for_ask = math.ceil(fair_value)
    buy_quantity = position_limit - (position + buy_order_volume)
    sell_quantity = position_limit + (position - sell_order_volume)
    if position_after_take > 0:
        if fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(RAINFOREST_RESIN, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(RAINFOREST_RESIN, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)
    return orders, buy_order_volume, sell_order_volume

def resin_make_orders(
    order_depth: OrderDepth,
    fair_value: float,
    position: int,
    position_limit: int,
    buy_order_volume: int,
    sell_order_volume: int,
) -> List[Order]:
    orders = []
    aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
    bbbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
    baaf = min(aaf) if aaf else fair_value + 2
    bbbf_val = max(bbbf) if bbbf else fair_value - 2
    buy_quantity = position_limit - (position + buy_order_volume)
    if buy_quantity > 0:
        orders.append(Order(RAINFOREST_RESIN, bbbf_val + 1, buy_quantity))
    sell_quantity = position_limit + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(RAINFOREST_RESIN, baaf - 1, -sell_quantity))
    return orders

# ---------------------------
# KELP Strategy Functions (from round1rsi)
# ---------------------------
def kelp_fair_value(
    order_depth: OrderDepth, method: str = "vwap_with_vol_filter", min_vol: int = 20
) -> float:
    if method == "mid_price":
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2
    elif method == "mid_price_with_vol_filter":
        sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
        buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
        if not sell_orders or not buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
        else:
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
        return (best_ask + best_bid) / 2
    elif method == "vwap":
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
        if volume == 0:
            return (best_ask + best_bid) / 2
        return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
    elif method == "vwap_with_vol_filter":
        sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
        buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
        if not sell_orders or not buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            if volume == 0:
                return (best_ask + best_bid) / 2
            return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
        else:
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            if volume == 0:
                return (best_ask + best_bid) / 2
            return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
    else:
        raise ValueError("Unknown fair value method specified.")

def kelp_take_orders(
    order_depth: OrderDepth, fair_value: float, params: dict, position: int
) -> Tuple[List[Order], int, int]:
    orders = []
    buy_order_volume = 0
    sell_order_volume = 0
    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask <= fair_value - params["take_width"] and ask_amount <= 50:
            quantity = min(ask_amount, params["position_limit"] - position)
            if quantity > 0:
                orders.append(Order(KELP, int(best_ask), quantity))
                buy_order_volume += quantity
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        bid_amount = order_depth.buy_orders[best_bid]
        if best_bid >= fair_value + params["take_width"] and bid_amount <= 50:
            quantity = min(bid_amount, params["position_limit"] + position)
            if quantity > 0:
                orders.append(Order(KELP, int(best_bid), -quantity))
                sell_order_volume += quantity
    return orders, buy_order_volume, sell_order_volume

def kelp_clear_orders(
    order_depth: OrderDepth,
    position: int,
    params: dict,
    fair_value: float,
    buy_order_volume: int,
    sell_order_volume: int,
) -> Tuple[List[Order], int, int]:
    orders = []
    position_after_take = position + buy_order_volume - sell_order_volume
    fair_for_bid = math.floor(fair_value)
    fair_for_ask = math.ceil(fair_value)
    buy_quantity = params["position_limit"] - (position + buy_order_volume)
    sell_quantity = params["position_limit"] + (position - sell_order_volume)
    if position_after_take > 0:
        if fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(KELP, int(fair_for_ask), -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(KELP, int(fair_for_bid), abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)
    return orders, buy_order_volume, sell_order_volume

def kelp_make_orders(
    order_depth: OrderDepth,
    fair_value: float,
    position: int,
    params: dict,
    buy_order_volume: int,
    sell_order_volume: int,
) -> List[Order]:
    orders = []
    edge = params["spread_edge"]
    aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + edge]
    bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - edge]
    baaf = min(aaf) if aaf else fair_value + edge + 1
    bbbf = max(bbf) if bbf else fair_value - edge - 1
    buy_quantity = params["position_limit"] - (position + buy_order_volume)
    if buy_quantity > 0:
        orders.append(Order(KELP, int(bbbf + 1), buy_quantity))
    sell_quantity = params["position_limit"] + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(KELP, int(baaf - 1), -sell_quantity))
    return orders
# ------------------------------------------------------------


class Trader: ##from here

    def __init__(self) -> None:
        self.logger = Logger()
        self.logger.print("Initialized trader for JAMS and DJEMBES only.")
        self.pos_limits = LIMIT
        self.params = PARAMS
        self.diff_threshold_b1_b2 = diff_threshold_b1_b2
        self.lot_size = 1

        self.previousposition = {params:0 for params in LIMIT}
        self.position = {params:0 for params in LIMIT}
        self.positionCounter = {params:0 for params in LIMIT}

        self.resin_timestamps = []
        self.resin_mid_prices = []
        self.kelp_timestamps = []
        self.kelp_mid_prices = []
        self.ink_timestamps = []
        self.ink_mid_prices = []
        self.rock_timestamps = []
        self.rock_mid_prices = []
        self.rock9500_timestamps = []
        self.rock9500_mid_prices = []
        self.rock9750_timestamps = []
        self.rock9750_mid_prices = []
        self.rock10000_timestamps = []
        self.rock10000_mid_prices = []
        self.rock10250_timestamps = []
        self.rock10250_mid_prices = []
        self.rock10500_timestamps = []
        self.rock10500_mid_prices = []
        self.diff_threshold_b1 = diff_threshold_b1 

    def update_market_data(self, product, state):
        # Store current timestamp and mid-price for all assets
        order_depth = state.order_depths[product]
        mid = mid_price(order_depth)
        if product == RAINFOREST_RESIN:
            self.resin_timestamps.append(state.timestamp)
            self.resin_mid_prices.append(mid)
        elif product == KELP:
            self.kelp_timestamps.append(state.timestamp)
            self.kelp_mid_prices.append(mid)
        elif product == SQUID_INK:
            self.ink_timestamps.append(state.timestamp)
            self.ink_mid_prices.append(mid)
        elif product == VOLCANIC_ROCK:
            self.rock_timestamps.append(state.timestamp)
            self.rock_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_9500":
            self.rock9500_timestamps.append(state.timestamp)
            self.rock9500_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_9750":
            self.rock9750_timestamps.append(state.timestamp)
            self.rock9750_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_10000":
            self.rock10000_timestamps.append(state.timestamp)
            self.rock10000_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_10250":
            self.rock10250_timestamps.append(state.timestamp)
            self.rock10250_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_10500":
            self.rock10500_timestamps.append(state.timestamp)
            self.rock10500_mid_prices.append(mid)

    def black_scholes_call(self, S, K, vol, T, r = 0, q = 0):
        """
        Black Scholes formula for European call option pricing - CHECK IF YOU WANT TO RETURN DELTA
        """
        # S = Current stock price
        # K = Option striking price
        # r = risk free interest
        # q = dividend yield - JUST 0 FOR NOW, MAYBE LATER IDK
        # vol = standard deviation of stock (volatility)
        # T = time until option expiration
        # Black Scholes - calculates theoretical call premium

        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)

        norm_dist = NormalDist().cdf
        norm_pdf = NormalDist().pdf
        N1 = norm_dist(d1)
        N2 = norm_dist(d2)

        # Call option price
        call_price = S*N1 - K*np.exp((q-r) * T)*N2

        # Delta - how much option price changes per $1 change in stock price
        delta = norm_dist(d1)

        # Gamma - how much delta changes per $1 change in stock price - like acceleration - if we know distribution we can use this to capture 
        gamma = norm_pdf(d1) / (S * vol * np.sqrt(T))
        
        # Vega - how much option price changes per 1% change in volatility - like sensitivity
        vega = S * norm_pdf(d1) * np.sqrt(T)

        return call_price, delta, gamma, vega

    def calc_fair_price(self, T) -> Dict[str, Tuple[int, int]]:
        """
        Calculate fair call option prices for the volcanic rock vouchers using the Black–Scholes formula.
        """
        if not self.rock_mid_prices:
            return {}

        S = self.rock_mid_prices[-1]  # most recent mid-price for VOLCANIC_ROCK
        fair_prices = {}

        for voucher in [VOLCANIC_ROCK_VOUCHER_9500,
                        VOLCANIC_ROCK_VOUCHER_9750,
                        VOLCANIC_ROCK_VOUCHER_10000,
                        VOLCANIC_ROCK_VOUCHER_10250,
                        VOLCANIC_ROCK_VOUCHER_10500]:

            # Get mid price for the specific voucher
            if voucher == VOLCANIC_ROCK_VOUCHER_9750:
                mid_price = self.rock9750_mid_prices[-1]
            elif voucher == VOLCANIC_ROCK_VOUCHER_9500:
                mid_price = self.rock9500_mid_prices[-1]
            elif voucher == VOLCANIC_ROCK_VOUCHER_10000:
                mid_price = self.rock10000_mid_prices[-1]
            elif voucher == VOLCANIC_ROCK_VOUCHER_10250:
                mid_price = self.rock10250_mid_prices[-1]
            elif voucher == VOLCANIC_ROCK_VOUCHER_10500:
                mid_price = self.rock10500_mid_prices[-1]
            # Get the parameters for the specific voucher
            vol = self.params[voucher]["implied_volatility"]
            strike = int(voucher.split("_")[-1])

            call_price, delta, gamma, vega = self.black_scholes_call(S, strike, vol, T)
            fair_prices[voucher] = (strike, mid_price, call_price, delta, gamma, vega)
        return fair_prices

    def OrderOptimised(self, product: str, size: int, mode: str, state: TradingState) -> list[Order]:
            orders = []
            VolTarget = size
            depth = state.order_depths[product]

            if mode == 'buy':
                # Ensure there are sell orders available; if not, return an empty list.
                if not depth.sell_orders:
                    return orders
                sell_orders = depth.sell_orders  # Use the correct attribute name
                # Get the prices sorted in ascending order (lowest offers first)
                sorted_prices = sorted(sell_orders.keys())
                for price in sorted_prices:
                    # Use the minimum between the target volume and what is available
                    volume_to_take = min(VolTarget, abs(sell_orders[price]))
                    orders.append(Order(product, price, volume_to_take))
                    VolTarget -= volume_to_take
                    if VolTarget <= 0:
                        break

            elif mode == 'sell':
                # Ensure there are buy orders available; if not, return an empty list.
                if not depth.buy_orders:
                    return orders
                buy_orders = depth.buy_orders  # Use the correct attribute name
                # Get the prices sorted in descending order (best bids first)
                sorted_prices = sorted(buy_orders.keys(), reverse=True)
                for price in sorted_prices:
                    volume_to_take = min(VolTarget, abs(buy_orders[price]))
                    # For selling, we send a negative quantity
                    orders.append(Order(product, price, -volume_to_take))
                    VolTarget -= volume_to_take
                    if VolTarget <= 0:
                        break

            return orders

    def PositionFraction(self, product: str, state: TradingState) -> float:
        """
        Calculate the fraction of the position for a given product.
        """
        position = state.position.get(product, 0)
        return np.round(position/LIMIT[product], decimals=3)

    def Profitable(self, product: str, state: TradingState, greed: float, mode: str) -> bool:
        """
        Assess whether taking a buy or sell action is profitable for the given product,
        based on historical position, market conditions, and the specified mode.
        
        Args:
            product (str): The product to evaluate.
            state (TradingState): The current trading state.
            greed (float): The minimum profit margin required to consider the action profitable.
            mode (str): The mode of the action, either 'buy' or 'sell'.
        
        Returns:
            bool: True if the action is profitable, False otherwise.
        """
        # Retrieve the order depth and current position for the product
        order_depth = state.order_depths.get(product)
        current_position = state.position.get(product, 0)
        if not order_depth:
            return False  # No market data available

        # Calculate the mid-price
        mid_price = self.mid_price(order_depth)

        # Get the best bid and ask prices
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        # Retrieve historical trades for the product
        own_trades = state.own_trades.get(product, [])
        if not own_trades:
            return False  # No historical trade data available

        # Calculate the average entry price based on historical trades
        total_quantity = sum(trade.quantity for trade in own_trades)
        total_cost = sum(trade.price * trade.quantity for trade in own_trades)
        avg_entry_price = total_cost / total_quantity if total_quantity != 0 else mid_price

        # Assess profitability based on the mode
        if mode == 'sell' and current_position > 0 and best_bid is not None:
            # Calculate potential profit from selling at the best bid
            sell_profit = (best_bid - avg_entry_price) * current_position
            if sell_profit >= greed * abs(current_position):
                return True

        elif mode == 'buy' and current_position < self.pos_limits[product] and best_ask is not None:
            # Calculate potential profit from buying at the best ask
            buy_profit = (avg_entry_price - best_ask) * (self.pos_limits[product] - current_position)
            if buy_profit >= greed * abs(self.pos_limits[product] - current_position):
                return True

        return False
    
    def mid_price(self, order_depth: OrderDepth) -> float:
        # Compute a mid-price using available order depth information
        if order_depth.sell_orders:
            total_ask = sum(price * quantity for price, quantity in order_depth.sell_orders.items())
            total_qty = sum(quantity for quantity in order_depth.sell_orders.values())
            if total_qty != 0:
                m1 = total_ask / total_qty
            else:
                m1 = None
        else:
            m1 = 0

        if order_depth.buy_orders:
            total_bid = sum(price * quantity for price, quantity in order_depth.buy_orders.items())
            total_qty = sum(quantity for quantity in order_depth.buy_orders.values())
            if total_qty != 0:
                m2 = total_bid / total_qty
            else:
                m2 = None
        else:
            m2 = 0
            
        return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)
    
    def mean_reversion_trade(self, product: str, mid_prices: List[float],
                            order_depth: OrderDepth, current_position: int,
                            position_limit: int, state: TradingState,
                            window: int = 50, z_score_thresh: float = 2.0) :
        # Only run if we have enough price history
        orders=[]
        if len(mid_prices) < window:
            return

        recent_prices = mid_prices[-window:]
        sma = statistics.mean(recent_prices)
        std_dev = statistics.stdev(recent_prices)

        # Calculate Bollinger Bands
        lower_band = sma - (z_score_thresh * std_dev)
        upper_band = sma + (z_score_thresh * std_dev)
        current_mid = mid_prices[-1]

        # Ensure there are available order levels
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        buy_volume = position_limit - current_position
        sell_volume = -position_limit - current_position

        # Buy when price is below the lower band - undervalued
        if current_mid < lower_band:
            # return Order(product, best_ask, buy_volume)
            return self.OrderOptimised(product, buy_volume, mode='buy', state=state) ## SWITCHED CHANGE IF SHIT
        # Sell when price is above the upper band - overvalued
        elif current_mid > upper_band:
            # return Order(product, best_bid, sell_volume)
            return self.OrderOptimised(product, sell_volume, mode='sell', state=state) ## SWITCHED CHANGE IF SHIT

    def UpdatePreviousPositionCounter(self,product,state:TradingState) -> None:
            if product not in set(state.position.keys()):
                return None
            if (
                state.position[product] == self.previousposition[product]
            ):  # Updates previouspositionCounter
                self.positionCounter[product] += 1
            else:
                self.positionCounter[product] = 0

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in LIMIT}
        conversions = 0
        trader_data = ""
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        # Updating Position and Position Counters
        for product in self.pos_limits.keys():
            self.UpdatePreviousPositionCounter(product,state)
            self.position[product] = state.position.get(product,0)
            try:
                self.update_market_data(product, state)
            except Exception:
                pass

# -------- Volcanic Rock: Options Trading with Black-Scholes and mean reversion for underlying: MAX STUFF WE CAN BUY SELL --------
        if VOLCANIC_ROCK in state.order_depths:
            rock_position = state.position.get(VOLCANIC_ROCK, 0)
            rock_params = PARAMS[VOLCANIC_ROCK]
            rock_order_depth = state.order_depths[VOLCANIC_ROCK]

            # Calculate time to expiry
            time_expiry = (PARAMS[VOLCANIC_ROCK]["starting_time_to_expiry"] - (state.timestamp) / 1000000 / 365)

            # Retrieve fair call option prices for volcanic rock vouchers
            fair_prices = self.calc_fair_price(T=time_expiry)
            undercut = 2.6667  # undercut for triggering orders more aggressively - OPTIMISE

        for voucher, (strike, mid_price, bs_price, bs_delta, bs_gamma, bs_vega) in fair_prices.items():
            if voucher in state.order_depths:
                voucher_order_depth = state.order_depths[voucher]

                buy_order_depth = voucher_order_depth.buy_orders
                sell_order_depth = voucher_order_depth.sell_orders

                # Skip if no orders exist
                if not buy_order_depth or not sell_order_depth:
                    continue
                else:

                    # Calculate allowed trading volumes for voucher
                    voucher_position = self.position[voucher]
                    voucher_buy_volume = self.pos_limits[voucher] - voucher_position
                    voucher_sell_volume = self.pos_limits[voucher] - voucher_position

                    # Fill orders more aggressively
                    if mid_price < bs_price - undercut:
                        # For sell side: get orders to sell the voucher
                        sell_orders = self.OrderOptimised(voucher, voucher_sell_volume, mode='sell', state=state)
                        for order in sell_orders:
                            result[voucher].append(order)
                    elif mid_price > bs_price + undercut:
                        # For buy side: get orders to buy the voucher
                        buy_orders = self.OrderOptimised(voucher, voucher_buy_volume, mode='buy', state=state)
                        for order in buy_orders:
                            result[voucher].append(order)

            rock_price_dict = {
                VOLCANIC_ROCK: (self.rock_mid_prices,
                                state.order_depths[VOLCANIC_ROCK],
                                state.position.get(VOLCANIC_ROCK, 0),
                                self.pos_limits[VOLCANIC_ROCK]),
                VOLCANIC_ROCK_VOUCHER_9500: (self.rock9500_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_9500],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_9500, 0),
                                            self.pos_limits[VOLCANIC_ROCK_VOUCHER_9500]),
                VOLCANIC_ROCK_VOUCHER_9750: (self.rock9750_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_9750],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_9750, 0),
                                            self.pos_limits[VOLCANIC_ROCK_VOUCHER_9750]),
                VOLCANIC_ROCK_VOUCHER_10000: (self.rock10000_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_10000],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_10000, 0),
                                            self.pos_limits[VOLCANIC_ROCK_VOUCHER_10000]),
                VOLCANIC_ROCK_VOUCHER_10250: (self.rock10250_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_10250],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_10250, 0),
                                            self.pos_limits[VOLCANIC_ROCK_VOUCHER_10250]),
                VOLCANIC_ROCK_VOUCHER_10500: (self.rock10500_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_10500],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_10500, 0),
                                            self.pos_limits[VOLCANIC_ROCK_VOUCHER_10500]),
            }

            # Loop over each product and apply the mean reversion strategy.
            selloffAggro = 0.7804
            selloffThreshold = 0.0724
            zscoreThresh = 2.4661
            greed = 5.8066 #How much return we want before dumping
            for product, (mid_prices, o_depth, pos, pos_limit) in rock_price_dict.items():
                PositionFraction = self.PositionFraction(product, state)
                if PositionFraction > selloffThreshold and self.Profitable(product, state,greed=greed,mode='sell'):
                    result[product]+=self.OrderOptimised(product,mode='sell',size=int(selloffAggro*pos),state=state)
                if PositionFraction <-selloffThreshold and self.Profitable(product, state,greed=greed,mode='buy'):
                    result[product]+=self.OrderOptimised(product,mode='buy',size=int(selloffAggro*pos),state=state)
                    order=self.mean_reversion_trade(product=product, mid_prices=mid_prices, order_depth=o_depth,
                                                    current_position= pos, position_limit=pos_limit,window=50, z_score_thresh=int(zscoreThresh), state=state)
                    if order:
                        result[product]+=order



        # RESIN MARKET MAKING
        # -------- RESIN: Simple Market Making Assuming Constant Fair Value --------
        if RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(RAINFOREST_RESIN, 0)
            resin_params = PARAMS[RAINFOREST_RESIN]
            resin_order_depth = state.order_depths[RAINFOREST_RESIN]
            resin_fair_value = resin_params["fair_value"]
            orders_take, bo, so = resin_take_orders(
                resin_order_depth, resin_fair_value, resin_position, LIMIT[RAINFOREST_RESIN]
            )
            orders_clear, bo, so = resin_clear_orders(
                resin_order_depth, resin_position, resin_fair_value, LIMIT[RAINFOREST_RESIN], bo, so
            )
            orders_make = resin_make_orders(
                resin_order_depth, resin_fair_value, resin_position, LIMIT[RAINFOREST_RESIN], bo, so
            )
            result[RAINFOREST_RESIN] += orders_take + orders_clear + orders_make
        # -------------------------------------------------------------------------



                    
        relevant = [
                    CROISSANTS,
                    JAMS,
                    DJEMBES,
                    PICNIC_BASKET1,
                    PICNIC_BASKET2
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
                best_bid[prod] = None
                best_ask[prod] = None
                mid_price[prod] = None

        # Compute the composition signal exactly as in the original:
        #    implied_BASKET1 = BASKET2 + 2 * CROISSANTS + 1 * JAMS + 1 * DJEMBES
        #    diff_comp = BASKET1 - implied_BASKET1
        if (mid_price[PICNIC_BASKET1] is not None and
            mid_price[PICNIC_BASKET2] is not None and
            mid_price[CROISSANTS] is not None and
            mid_price[JAMS] is not None and
            mid_price[DJEMBES] is not None):
            b1 = mid_price[PICNIC_BASKET1]
            b2 = mid_price[PICNIC_BASKET2]
            c = mid_price[CROISSANTS]
            j = mid_price[JAMS]
            d = mid_price[DJEMBES]
            implied_b1 = b2 + 2 * c + j + d
            diff_comp1 = b1 - implied_b1         
            signal = 0
            if diff_comp1 > self.diff_threshold_b1_b2:
                # When diff is significantly positive, the composite view indicates that
                # the basket is overpriced. In the original, this leads to shorting baskets
                # and buying items. For our version we then wish to buy JAMS and DJEMBES.
                signal = +1  # signal to buy the items
            elif diff_comp1 < -self.diff_threshold_b1_b2:
                # When diff is significantly negative, the items are overpriced relative to the basket.
                # For our version we then signal to sell JAMS and DJEMBES.
                signal = -1  # signal to sell the items
        else:
            signal = 0

        # Only place orders in JAMS and DJEMBES – ignore any orders for other products.
        for prod in [JAMS, DJEMBES]:
            od = state.order_depths.get(prod)
            if not od:
                continue
            current_pos = state.position.get(prod, 0)
            pos_limit = self.pos_limits[prod]
            if signal == +1:
                # Buy signal: check if we have capacity to buy.
                capacity = pos_limit - current_pos
                if capacity <= 0:
                    continue
                trade_qty = min(self.lot_size, capacity)
                # For a buy order, mimic the original synergy logic:
                # use best ask price + 1 as our bid price.
                if od.sell_orders:
                    best_ask_price = min(od.sell_orders.keys())
                    price = best_ask_price + 1
                    result[prod].append(Order(prod, price, trade_qty))
            elif signal == -1:
                # Sell signal: check idiscorf we have capacity to sell.
                capacity = pos_limit + current_pos  # since short positions count against limit
                if capacity <= 0:
                    continue
                trade_qty = min(self.lot_size, capacity)
                if od.buy_orders:
                    best_bid_price = max(od.buy_orders.keys())
                    price = best_bid_price - 1
                    result[prod].append(Order(prod, price, -trade_qty))





#-----------------JAMES BRAINROT ------------------
        IWantThisMuchFor2 = 50
        spread2 = (vwap(PICNIC_BASKET2,state=state) - 4 * vwap(CROISSANTS,state=state) - 2 * vwap(JAMS,state=state))
        normspread2 = spread2 - synth2Mean
        if PICNIC_BASKET1 in state.order_depths:
            if (diff_comp1 < self.diff_threshold_b1): 
                for price, volume in get_best_bids_to_fill_WITH_LEVELS(PICNIC_BASKET1, state, desired_volume(PICNIC_BASKET1, state=state)):
                    result[PICNIC_BASKET1].append(Order(PICNIC_BASKET1, price , volume)) 
                
            elif (diff_comp1 > self.diff_threshold_b1):  
                for price, volume in get_best_asks_to_fill_WITH_LEVELS(PICNIC_BASKET1, state, desired_volume(PICNIC_BASKET1, state=state)):
                    result[PICNIC_BASKET1].append(Order(PICNIC_BASKET1, price, -volume)) 
        if PICNIC_BASKET2 in state.order_depths:  
            if (
                normspread2 < synth2Sigma * s2Zscore
            ):  # assume this means PB2 is overvalued
                    for price, volume in get_best_asks_to_fill_WITH_LEVELS(PICNIC_BASKET2, state, IWantThisMuchFor2):
                        result[PICNIC_BASKET2].append(Order(PICNIC_BASKET2, price, -volume))   
            elif (
                normspread2 > synth2Sigma * s2Zscore
            ):  # assume this means PB2 is the one that is overvalued
                    for price, volume in get_best_bids_to_fill_WITH_LEVELS(PICNIC_BASKET2, state, IWantThisMuchFor2):
                        result[PICNIC_BASKET2].append(Order(PICNIC_BASKET2, price , volume))        



        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data