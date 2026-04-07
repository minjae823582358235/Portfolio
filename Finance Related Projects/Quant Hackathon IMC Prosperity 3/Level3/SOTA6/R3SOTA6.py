import json
import jsonpickle
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional, Deque
from abc import abstractmethod
from collections import deque, defaultdict
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

STRIKES = {
    VOLCANIC_ROCK_VOUCHER_9500: 9500,
    VOLCANIC_ROCK_VOUCHER_9750: 9750,
    VOLCANIC_ROCK_VOUCHER_10000: 10000,
    VOLCANIC_ROCK_VOUCHER_10250: 10250,
    VOLCANIC_ROCK_VOUCHER_10500: 10500,
}

B1B2_THEORETICAL_COMPONENTS = {
    CROISSANTS: 2,
    JAMS: 1,
    DJEMBES: 1
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

def calculate_option_price(S, K, T, r, sigma):
    """Black-Scholes formula for call option price."""
    if sigma <= 0 or T <= 0:
        return max(0, S - K)
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def norm_pdf(x):
    """Standard normal probability density function."""
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def calculate_implied_volatility(option_price, S, K, T, r=0, initial_vol=0.3, max_iterations=50, precision=0.0001):
    """Newton-Raphson method to find implied volatility - optimized version."""
    if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0
    
    intrinsic_value = max(0, S - K)
    if option_price <= intrinsic_value:
        return 0.0
    
    vol = initial_vol
    for i in range(max_iterations):
        price = calculate_option_price(S, K, T, r, vol)
        diff = option_price - price
        
        if abs(diff) < precision:
            return vol
        
        d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
        vega = S * math.sqrt(T) * norm_pdf(d1)
        
        if vega == 0:
            return vol
     
        vol = vol + diff / vega
        
        if vol <= 0:
            vol = 0.0001
        elif vol > 5:  
            vol = 5.0
    
    return vol

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

# --- Base Strategy Class ---
class Strategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders: List[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = [] 
        self.act(state)
        return self.orders

    def _place_buy_order(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <=0: return
        self.orders.append(Order(self.symbol, price, quantity))
        logger.print(f"PLACE {self.symbol} BUY {quantity}x{price}") # Keep logs minimal

    def _place_sell_order(self, price: float, quantity: float) -> None:
        if quantity <= 0: return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <=0: return
        self.orders.append(Order(self.symbol, price, -quantity))
        logger.print(f"PLACE {self.symbol} SELL {quantity}x{price}") # Keep logs minimal

    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass

    def _get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        order_depth = state.order_depths.get(symbol)
        if not order_depth: return None

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        elif best_bid is not None: return best_bid 
        elif best_ask is not None: return best_ask 
        else: return None

class RsiStrategy(Strategy):
    """A generic RSI strategy applicable to different products."""
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = PARAMS.get(self.symbol, {}) # Load params specific to self.symbol
        if not self.params:
            logger.print(f"ERROR: Parameters for RSI strategy on {self.symbol} not found in PARAMS. Using defaults.")
            self.params = {"rsi_window": 85, "rsi_overbought": 52.0, "rsi_oversold": 42.0, "price_offset": 0} # Fallback defaults

        # Load RSI parameters from self.params
        self.window = self.params.get("rsi_window", 14)
        if self.window < 2:
            logger.print(f"Warning: RSI window {self.window} too small for {self.symbol}, setting to 2.")
            self.window = 2
        self.overbought_threshold = self.params.get("rsi_overbought", 70.0)
        self.oversold_threshold = self.params.get("rsi_oversold", 30.0)
        self.price_offset = self.params.get("price_offset", 0)  # New parameter with default 0 (no offset)

        # State variables for RSI calculation
        self.mid_price_history: deque[float] = deque(maxlen=self.window + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_initialized: bool = False

        logger.print(f"Initialized RsiStrategy for {self.symbol}: Win={self.window}, OB={self.overbought_threshold}, OS={self.oversold_threshold}, Offset={self.price_offset}")

    def _calculate_rsi(self, current_mid_price: float) -> Optional[float]:
        """Calculates RSI using Wilder's smoothing method."""
        self.mid_price_history.append(current_mid_price)

        if len(self.mid_price_history) < self.window + 1:
            return None # Need enough data points

        prices = list(self.mid_price_history)
        # Need at least 2 prices to calculate 1 change
        if len(prices) < 2: 
            return None 
        
        changes = np.diff(prices) # Use numpy.diff for efficient calculation
        
        # Ensure changes array is not empty
        if changes.size == 0: 
            return None

        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))

        # Ensure we have enough data points for the initial calculation
        if len(gains) < self.window: 
            return None 

        if not self.rsi_initialized or self.avg_gain is None or self.avg_loss is None:
            # First calculation: Use simple average over the window
            # Slice to get exactly 'window' number of changes
            self.avg_gain = np.mean(gains[-self.window:]) 
            self.avg_loss = np.mean(losses[-self.window:])
            self.rsi_initialized = True
            # logger.print(f" {self.symbol} (RSI): Initialized avg_gain={self.avg_gain:.4f}, avg_loss={self.avg_loss:.4f}")
        else:
            # Subsequent calculations: Use Wilder's smoothing
            current_gain = gains[-1]
            current_loss = losses[-1]
            self.avg_gain = ((self.avg_gain * (self.window - 1)) + current_gain) / self.window
            self.avg_loss = ((self.avg_loss * (self.window - 1)) + current_loss) / self.window

        if self.avg_loss is not None and self.avg_loss < 1e-9: # Check for near-zero loss
             # Avoid division by zero or extreme RSI; RSI is 100 if avg_loss is 0
             return 100.0
        elif self.avg_gain is None or self.avg_loss is None:
             # Should not happen if initialized correctly, but safety check
             return None 
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)
        if not order_depth: 
            return

        # Use the base class helper to get mid price
        current_mid_price = self._get_mid_price(self.symbol, state)
        if current_mid_price is None:
             
             return

        # Calculate RSI
        rsi_value = self._calculate_rsi(current_mid_price)
        if rsi_value is None:
            
            return
        

        # Generate Signal & Trade
        to_buy_capacity = self.position_limit - position
        to_sell_capacity = self.position_limit + position

        best_bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        # Signal: Sell when RSI is overbought
        if rsi_value > self.overbought_threshold and to_sell_capacity > 0:
            if best_bid_price is not None: # Need a bid to hit
                size_to_sell = to_sell_capacity # Sell max capacity
                # Apply price offset for selling (negative direction)
                aggressive_sell_price = best_bid_price - self.price_offset
                if aggressive_sell_price <= 0: # Ensure price is positive
                    aggressive_sell_price = best_bid_price
                
                self._place_sell_order(aggressive_sell_price, size_to_sell)

        # Signal: Buy when RSI is oversold
        elif rsi_value < self.oversold_threshold and to_buy_capacity > 0:
            if best_ask_price is not None: # Need an ask to hit
                size_to_buy = to_buy_capacity # Buy max capacity
                # Apply price offset for buying (positive direction)
                aggressive_buy_price = best_ask_price + self.price_offset
                
                self._place_buy_order(aggressive_buy_price, size_to_buy)

    def save(self) -> dict:
        # Save strategy state - include RSI state
        return {
            "mid_price_history": list(self.mid_price_history),
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "rsi_initialized": self.rsi_initialized
        }

    def load(self, data: dict) -> None:
        # Load strategy state - include RSI state
        loaded_history = data.get("mid_price_history", [])
        if isinstance(loaded_history, list):
             # Ensure history respects maxlen on load
             start_index = max(0, len(loaded_history) - (self.window + 1))
             self.mid_price_history = deque(loaded_history[start_index:], maxlen=self.window + 1)
        else:
             self.mid_price_history = deque(maxlen=self.window + 1) # Reset if invalid

        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        # Ensure avg_gain/loss are floats if loaded
        if self.avg_gain is not None:
            try: 
                self.avg_gain = float(self.avg_gain)
            except (ValueError, TypeError): 
                self.avg_gain = None
        if self.avg_loss is not None:
            try: 
                self.avg_loss = float(self.avg_loss)
            except (ValueError, TypeError): 
                self.avg_loss = None

        self.rsi_initialized = data.get("rsi_initialized", False)
        if not isinstance(self.rsi_initialized, bool): 
            self.rsi_initialized = False

        # Reset if essential state is inconsistent
        if self.rsi_initialized and (self.avg_gain is None or self.avg_loss is None):
            
            self.rsi_initialized = False
            self.avg_gain = None
            self.avg_loss = None
            # History might be okay, don't clear it unless needed

class B1B2DeviationStrategy(Strategy):
    """Trades the deviation between the actual B1-B2 spread and the theoretical B1-B2 spread."""
    def __init__(self, symbol: str, position_limits: Dict[str, int]) -> None:
        # Use the B1B2_DEVIATION symbol for the strategy key, position limit is not directly used here
        super().__init__(symbol, 0)
        self.pos_limits = position_limits # Store limits for all relevant products
        self.params = PARAMS.get(self.symbol, {})
        if not self.params:
            self.params = { # Fallback defaults
                "deviation_mean": 0, "deviation_std_window": 500,
                "zscore_threshold_entry": 2.0, "zscore_threshold_exit": 0.5,
                "target_deviation_spread_size": 10
            }
            logger.print(f"Warning: {self.symbol} params not found, using defaults.")

        # Load parameters
        self.deviation_mean = self.params.get("deviation_mean", 0)
        self.deviation_std_window = self.params.get("deviation_std_window", 500)
        if self.deviation_std_window < 10: self.deviation_std_window = 10
        self.zscore_entry = abs(self.params.get("zscore_threshold_entry", 2.0))
        self.zscore_exit = abs(self.params.get("zscore_threshold_exit", 0.5))
        self.target_size = abs(self.params.get("target_deviation_spread_size", 10))

        # State variables
        self.deviation_history: Deque[float] = deque(maxlen=self.deviation_std_window)
        # Represents the net number of deviation units we are holding
        # +ve means: Long B1, Short B2, Short 2C, Short 1J, Short 1D
        # -ve means: Short B1, Long B2, Long 2C, Long 1J, Long 1D
        self.current_effective_deviation_pos: int = 0

    # Override _place_buy/sell_order to ensure they operate on self.orders directly
    def _place_order(self, product_symbol: Symbol, price: int, quantity: int) -> None:
         if quantity == 0: return
         if price <= 0: return
         self.orders.append(Order(product_symbol, price, quantity))
         

    def _get_mid_price_safe(self, product: Symbol, state: TradingState) -> Optional[float]:
        """Safely gets mid price, returning None if unavailable."""
        return super()._get_mid_price(product, state) # Use base class method

    def act(self, state: TradingState) -> None:
        self.orders = [] # Crucial: Clear orders at the start of each act call for THIS strategy

        # 1. Calculate required mid-prices
        mid_b1 = self._get_mid_price_safe(PICNIC_BASKET1, state)
        mid_b2 = self._get_mid_price_safe(PICNIC_BASKET2, state)
        mid_c = self._get_mid_price_safe(CROISSANTS, state)
        mid_j = self._get_mid_price_safe(JAMS, state)
        mid_d = self._get_mid_price_safe(DJEMBES, state)

        required_products_present = [PICNIC_BASKET1, PICNIC_BASKET2,
                                     CROISSANTS, JAMS, DJEMBES]
        if any(prod not in state.order_depths for prod in required_products_present):
             logger.print(f"{self.symbol}: Missing order depth for one or more products. Skipping.")
             return

        if None in [mid_b1, mid_b2, mid_c, mid_j, mid_d]:
            logger.print(f"{self.symbol}: Missing mid-price for one or more components/baskets. Skipping.")
            return # Cannot calculate deviation

        # 2. Calculate actual and theoretical spreads, and the deviation
        actual_spread = mid_b1 - mid_b2
        theoretical_spread = (B1B2_THEORETICAL_COMPONENTS[CROISSANTS] * mid_c +
                              B1B2_THEORETICAL_COMPONENTS[JAMS] * mid_j +
                              B1B2_THEORETICAL_COMPONENTS[DJEMBES] * mid_d)
        deviation = actual_spread - theoretical_spread
        self.deviation_history.append(deviation)

        # 3. Check if history is sufficient for Z-score calculation
        # Use a smaller fraction for min_periods check to start trading sooner
        if len(self.deviation_history) < self.deviation_std_window // 4:
             
             return

        # 4. Calculate Z-score
        current_deviation_history = list(self.deviation_history)
        deviation_std = np.std(current_deviation_history)
        if deviation_std < 1e-6: # Avoid division by zero
            logger.print(f"{self.symbol}: Deviation std dev too low ({deviation_std:.4f}). Skipping.")
            return

        z_score = (deviation - self.deviation_mean) / deviation_std
        logger.print(f"{self.symbol}: Dev={deviation:.2f}, Mean={self.deviation_mean:.2f}, Std={deviation_std:.2f}, Z={z_score:.2f}, CurrEffPos={self.current_effective_deviation_pos}")

        # 5. Determine desired position based on Z-score
        desired_effective_deviation_pos = self.current_effective_deviation_pos # Default: no change

        if z_score >= self.zscore_entry:
            # Deviation is high -> Sell Deviation (-target_size)
            desired_effective_deviation_pos = -self.target_size
        elif z_score <= -self.zscore_entry:
            # Deviation is low -> Buy Deviation (+target_size)
            desired_effective_deviation_pos = self.target_size
        else:
            # Check for exit signal ONLY if we hold a position
            if self.current_effective_deviation_pos > 0 and z_score >= -self.zscore_exit:
                # Holding Long, Z moved back up -> Close
                desired_effective_deviation_pos = 0
                logger.print(f"{self.symbol}: Exit Long Deviation signal (Z={z_score:.2f} >= {-self.zscore_exit:.2f})")
            elif self.current_effective_deviation_pos < 0 and z_score <= self.zscore_exit:
                # Holding Short, Z moved back down -> Close
                desired_effective_deviation_pos = 0
                logger.print(f"{self.symbol}: Exit Short Deviation signal (Z={z_score:.2f} <= {self.zscore_exit:.2f})")
            # Otherwise, stay in current position if between exit and entry thresholds

        # 6. Execute trades if desired position changed
        if desired_effective_deviation_pos != self.current_effective_deviation_pos:
            
            self._execute_deviation_trade(state, desired_effective_deviation_pos)
        

    def _calculate_max_deviation_spread_size(self, state: TradingState, direction: int) -> int:
        """Calculates max units of deviation spread tradeable based on all 5 product limits."""
        if direction == 0: return 0

        # Define quantity changes PER UNIT of deviation spread trade
        # direction > 0 (Buy Deviation): +B1, -B2, -2C, -1J, -1D
        # direction < 0 (Sell Deviation): -B1, +B2, +2C, +1J, +1D
        qty_changes_per_unit = {
            PICNIC_BASKET1: +direction,
            PICNIC_BASKET2: -direction,
            CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[CROISSANTS] * direction,
            JAMS: -B1B2_THEORETICAL_COMPONENTS[JAMS] * direction,
            DJEMBES: -B1B2_THEORETICAL_COMPONENTS[DJEMBES] * direction,
        }

        max_units = float('inf')

        for product, qty_change in qty_changes_per_unit.items():
            if qty_change == 0: continue # Should not happen with current def

            current_pos = state.position.get(product, 0)
            limit = self.pos_limits.get(product)
            if limit is None: # Should have limit defined
                logger.print(f"Error: Missing position limit for {product}")
                return 0
            if limit == 0: # Safety check
                logger.print(f"Warning: Limit for {product} is 0. Cannot trade deviation.")
                return 0

            if qty_change > 0: # Need to BUY this product
                capacity = limit - current_pos
            else: # Need to SELL this product
                capacity = limit + current_pos # Capacity is positive number

            if capacity < 0: capacity = 0 # Already over limit in the wrong direction

            # How many units can we trade based on this product's capacity?
            max_units_for_product = capacity // abs(qty_change)
            max_units = min(max_units, max_units_for_product)

        final_max = max(0, int(max_units))
      
        return final_max

    def _calculate_market_liquidity_limit(self, state: TradingState, direction: int) -> int:
        """Calculates max units of deviation spread tradeable based on TOTAL market liquidity for each leg."""
        if direction == 0: return 0

        order_depths = state.order_depths
        max_units = float('inf')

        # Define quantity changes PER UNIT of deviation spread trade
        qty_changes_per_unit = {
            PICNIC_BASKET1: +direction,
            PICNIC_BASKET2: -direction,
            CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[CROISSANTS] * direction,
            JAMS: -B1B2_THEORETICAL_COMPONENTS[JAMS] * direction,
            DJEMBES: -B1B2_THEORETICAL_COMPONENTS[DJEMBES] * direction,
        }

        for product, qty_change in qty_changes_per_unit.items():
            if qty_change == 0: continue

            od = order_depths.get(product)
            if not od: return 0 # Cannot trade if any order book is missing

            total_available_volume = 0
            if qty_change > 0: # Need to BUY this product -> sum volume from SELL side
                if not od.sell_orders: return 0 # No liquidity
                total_available_volume = sum(abs(vol) for vol in od.sell_orders.values())
            else: # Need to SELL this product -> sum volume from BUY side
                if not od.buy_orders: return 0 # No liquidity
                total_available_volume = sum(abs(vol) for vol in od.buy_orders.values())

            if total_available_volume <= 0:
                return 0 # One leg has no liquidity

            units_fillable_for_product = total_available_volume // abs(qty_change)
            max_units = min(max_units, units_fillable_for_product)
            

        final_max_liq = max(0, int(max_units))
        
        return final_max_liq

    def _place_aggressive_orders_for_leg(self, product_symbol: Symbol, total_quantity_needed: int, order_depth: OrderDepth):
        """Places orders for one leg, consuming book levels until quantity is met or liquidity runs out."""
        if total_quantity_needed == 0: return

        orders_to_place = []
        remaining_qty = abs(total_quantity_needed)

        if total_quantity_needed > 0: # Need to BUY
            if not order_depth.sell_orders: return # No liquidity
            sorted_levels = sorted(order_depth.sell_orders.items()) # Sort asks by price ascending
            for price, volume_at_level in sorted_levels:
                vol = abs(volume_at_level)
                qty_at_this_level = min(remaining_qty, vol)
                if qty_at_this_level > 0:
                    orders_to_place.append(Order(product_symbol, price, int(qty_at_this_level)))
                    remaining_qty -= qty_at_this_level
                if remaining_qty <= 0: break

        else: # Need to SELL
            if not order_depth.buy_orders: return # No liquidity
            sorted_levels = sorted(order_depth.buy_orders.items(), reverse=True) # Sort bids by price descending
            for price, volume_at_level in sorted_levels:
                vol = abs(volume_at_level)
                qty_at_this_level = min(remaining_qty, vol)
                if qty_at_this_level > 0:
                    orders_to_place.append(Order(product_symbol, price, -int(qty_at_this_level)))
                    remaining_qty -= qty_at_this_level
                if remaining_qty <= 0: break

        # Add collected orders to the strategy's main list
        self.orders.extend(orders_to_place)

    def _execute_deviation_trade(self, state: TradingState, target_effective_pos: int):
        """Calculates executable size considering limits & liquidity, then places aggressive orders."""

        qty_units_to_trade = target_effective_pos - self.current_effective_deviation_pos
        if qty_units_to_trade == 0:
            return

        direction = 1 if qty_units_to_trade > 0 else -1

        # 1. Check Position Limit Constraint
        max_units_pos = self._calculate_max_deviation_spread_size(state, direction)
        if max_units_pos <= 0:
            logger.print(f"Execute: Cannot trade {direction} unit(s), blocked by position limit.")
            return

        # 2. Check Market Liquidity Constraint
        max_units_liq = self._calculate_market_liquidity_limit(state, direction)
        if max_units_liq <= 0:
            logger.print(f"Execute: Cannot trade {direction} unit(s), blocked by market liquidity.")
            return

        # 3. Determine Actual Executable Units
        actual_units_to_trade = direction * min(abs(qty_units_to_trade), max_units_pos, max_units_liq)

        if actual_units_to_trade == 0:
            # This case should ideally be caught by the checks above, but added for safety
            logger.print(f"Execute: Calculated 0 actual units to trade (Target: {target_effective_pos}, Current: {self.current_effective_deviation_pos}, MaxPos: {max_units_pos}, MaxLiq: {max_units_liq}).")
            return

        logger.print(f"Execute: Attempting to trade {actual_units_to_trade} deviation units (LimitPos: {max_units_pos}, LimitLiq: {max_units_liq}).")

        # 4. Define quantity changes based on the ACTUAL units we will trade
        final_qty_changes = {
            PICNIC_BASKET1: +actual_units_to_trade,
            PICNIC_BASKET2: -actual_units_to_trade,
            CROISSANTS: -B1B2_THEORETICAL_COMPONENTS[CROISSANTS] * actual_units_to_trade,
            JAMS: -B1B2_THEORETICAL_COMPONENTS[JAMS] * actual_units_to_trade,
            DJEMBES: -B1B2_THEORETICAL_COMPONENTS[DJEMBES] * actual_units_to_trade,
        }

        # 5. Place aggressive orders for each leg
        order_depths = state.order_depths
        for product, final_qty_int in final_qty_changes.items():
            final_qty = int(round(final_qty_int)) # Ensure integer
            if final_qty == 0: continue

            od = order_depths.get(product)
            if not od:
                logger.print(f"Error: Order depth for {product} disappeared before placing orders!")
                # Note: This might mean the overall trade isn't perfectly hedged anymore.
                # Could potentially cancel already added orders, but becomes complex.
                continue # Skip this leg

            self._place_aggressive_orders_for_leg(product, final_qty, od)

        # 6. Update internal state *after* attempting to place all orders
        # Assume the orders will fill aggressivley up to the calculated limit
        self.current_effective_deviation_pos += actual_units_to_trade
        logger.print(f"Execute: Aggressive orders placed. New effective pos: {self.current_effective_deviation_pos}")

    def save(self) -> dict:
        # Need to save history and current effective position
        return {
            "deviation_history": list(self.deviation_history),
            "current_effective_deviation_pos": self.current_effective_deviation_pos
        }

    def load(self, data: dict) -> None:
        # Load history
        loaded_history = data.get("deviation_history", [])
        if isinstance(loaded_history, list):
             # Ensure deque uses correct maxlen from params
             self.deviation_history = deque(loaded_history, maxlen=self.deviation_std_window)
        else:
             self.deviation_history = deque(maxlen=self.deviation_std_window)

        # Load effective position
        loaded_pos = data.get("current_effective_deviation_pos", 0)
        if isinstance(loaded_pos, (int, float)):
            self.current_effective_deviation_pos = int(loaded_pos)
        else:
            self.current_effective_deviation_pos = 0

class VolatilitySmileStrategy:
    def __init__(self) -> None:
        """Initialize the Volatility Smile trader for all vouchers"""
        self.pos_limits = {
            VOLCANIC_ROCK: 400,
            VOLCANIC_ROCK_VOUCHER_9500: 200,
            VOLCANIC_ROCK_VOUCHER_9750: 200,
            VOLCANIC_ROCK_VOUCHER_10000: 200,
            VOLCANIC_ROCK_VOUCHER_10250: 200,
            VOLCANIC_ROCK_VOUCHER_10500: 200,
        }
        
        # Voucher symbols
        self.voucher_symbols = [
            VOLCANIC_ROCK_VOUCHER_9500, 
            VOLCANIC_ROCK_VOUCHER_9750, 
            VOLCANIC_ROCK_VOUCHER_10000, 
            VOLCANIC_ROCK_VOUCHER_10250, 
            VOLCANIC_ROCK_VOUCHER_10500
        ]
        
        # Strategy parameters for base IV mean reversion
        self.short_ewma_span = 37  # First level EWMA span for Base IV
        self.long_ewma_span = 68  # Second level EWMA span for double EWMA
        self.rolling_window = 48   # Window for rolling standard deviation
        
        # Z-score thresholds for trading signals
        self.zscore_upper_threshold = 0.5  # Z-score threshold for sell signals
        self.zscore_lower_threshold = -2.8  # Z-score threshold for buy signals
        
        self.trade_size = 22
        
        self.base_iv_history = deque(maxlen=200)
        
        self.short_ewma_base_iv = None
        self.long_ewma_first = None  
        self.long_ewma_base_iv = None  
        
        self.ewma_diff_history = deque(maxlen=200)
        
        self.zscore_history = deque(maxlen=100)
        
        self.day = 3  # set to 3 when submitting
        self.last_timestamp = None
        
        # Store orders
        self.orders = {}
    
    def update_time_to_expiry(self, timestamp):
        """Calculate time to expiry based on the current timestamp."""
        base_tte = 8 - self.day
        iteration = (timestamp % 1000000) // 100
        iteration_adjustment = iteration / 10000
        tte = (base_tte - iteration_adjustment) / 365
        return max(0.0001, tte)  
    
    def get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
        """Get the mid price for a given symbol from the order depth."""
        order_depth = state.order_depths.get(symbol)
        if not order_depth:
            return None
        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            return best_bid
        elif best_ask is not None:
            return best_ask
        else:
            return None
    
    def calculate_order_size(self, symbol: Symbol, zscore: float, state: TradingState) -> int:
        """Calculate order size based on z-score sign and current position."""
        current_position = state.position.get(symbol, 0)
        position_limit = self.pos_limits.get(symbol, 0)
        
        fixed_size = self.trade_size
        
        if zscore > 0:  
            if current_position - fixed_size >= -position_limit:
                return -fixed_size  
            else:
                return 0  
        else: 
            if current_position + fixed_size <= position_limit:
                return fixed_size 
            else:
                return 0  
    
    def place_order(self, orders_dict, symbol, price, quantity):
        """Add an order to the orders dictionary."""
        if quantity == 0:
            return
        
        if symbol not in orders_dict:
            orders_dict[symbol] = []
        
        orders_dict[symbol].append(Order(symbol, price, quantity))
        logger.print(f"PLACE {symbol} {'BUY' if quantity > 0 else 'SELL'} {abs(quantity)}x{price}")
    
    def update_ewma(self, current_value, previous_ewma, span):
        """Calculate EWMA (Exponentially Weighted Moving Average)."""
        if previous_ewma is None:
            return current_value
        alpha = 2 / (span + 1)
        return alpha * current_value + (1 - alpha) * previous_ewma
        
    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        """Execute the volatility smile trading strategy using base IV mean reversion signals."""
        orders_dict = {}
        
        self.last_timestamp = state.timestamp
        time_to_expiry = self.update_time_to_expiry(state.timestamp)
        
        rock_price = self.get_mid_price(VOLCANIC_ROCK, state)
        if not rock_price:
            logger.print("No price available for VOLCANIC_ROCK, skipping iteration")
            return orders_dict
        
        # Add risk management check - if deep out of the money, go flat
        for voucher in self.voucher_symbols:
            strike = STRIKES[voucher]
            # Check if underlying - strike is <= -250 (deep out of the money)
            if rock_price - strike <= -250:
                current_position = state.position.get(voucher, 0)
                if current_position != 0:
                    voucher_price = self.get_mid_price(voucher, state)
                    if voucher_price and current_position > 0:
                        # Have long position, need to sell to go flat
                        order_depth = state.order_depths.get(voucher)
                        if order_depth and order_depth.buy_orders:
                            best_bid = max(order_depth.buy_orders.keys())
                            # Place sell order at best bid to flatten position
                            self.place_order(orders_dict, voucher, best_bid, -current_position)
                            
                    elif voucher_price and current_position < 0:
                        # Have short position, need to buy to go flat
                        order_depth = state.order_depths.get(voucher)
                        if order_depth and order_depth.sell_orders:
                            best_ask = min(order_depth.sell_orders.keys())
                            # Place buy order at best ask to flatten position
                            self.place_order(orders_dict, voucher, best_ask, -current_position)
                            
        
        moneyness_values = []
        iv_values = []
        voucher_data = {}
        
        for voucher in self.voucher_symbols:
            voucher_price = self.get_mid_price(voucher, state)
            if not voucher_price:
                continue
                
            strike = STRIKES[voucher]
            
            moneyness = math.log(strike / rock_price) / math.sqrt(time_to_expiry)
            
            impl_vol = calculate_implied_volatility(voucher_price, rock_price, strike, time_to_expiry)
            
            if impl_vol > 0: 
                moneyness_values.append(moneyness)
                iv_values.append(impl_vol)
                voucher_data[voucher] = {'moneyness': moneyness, 'iv': impl_vol}
        
        if len(moneyness_values) >= 3:
            try:
                coeffs = np.polyfit(moneyness_values, iv_values, 2)
                a, b, c = coeffs
                
                base_iv = c
                logger.print(f"Base IV (ATM): {base_iv:.6f}")
                
                self.base_iv_history.append(base_iv)
                
                self.short_ewma_base_iv = self.update_ewma(
                    base_iv, 
                    self.short_ewma_base_iv, 
                    self.short_ewma_span
                )
                
                # Update first-level EWMA
                self.long_ewma_first = self.update_ewma(
                    base_iv,
                    self.long_ewma_first,
                    self.long_ewma_span
                )
                
                # Update second-level (double) EWMA
                self.long_ewma_base_iv = self.update_ewma(
                    self.long_ewma_first,
                    self.long_ewma_base_iv,
                    self.long_ewma_span
                )
                
                if len(self.base_iv_history) >= self.rolling_window and self.short_ewma_base_iv is not None and self.long_ewma_base_iv is not None:
                    
                    ewma_diff = self.short_ewma_base_iv - self.long_ewma_base_iv
                    
                    if not hasattr(self, 'ewma_diff_history'):
                        self.ewma_diff_history = deque(maxlen=200)
                    self.ewma_diff_history.append(ewma_diff)
                    
                    if len(self.ewma_diff_history) >= self.rolling_window:
                        recent_ewma_diffs = list(self.ewma_diff_history)[-self.rolling_window:]
                        rolling_std = np.std(recent_ewma_diffs)
                    else:
                        rolling_std = np.std(list(self.ewma_diff_history))
                    
                    if rolling_std > 0:
                        zscore = ewma_diff / rolling_std
                    else:
                        zscore = 0
                    
                    self.zscore_history.append(zscore)
                    
                    logger.print(f"Base IV: {base_iv:.6f}, Short EWMA: {self.short_ewma_base_iv:.6f}, Long EWMA: {self.long_ewma_base_iv:.6f}")
                    logger.print(f"EWMA Diff: {ewma_diff:.6f}, Rolling StdDev: {rolling_std:.6f}")
                    logger.print(f"Z-score: {zscore:.4f}, Upper Threshold: {self.zscore_upper_threshold}, Lower Threshold: {self.zscore_lower_threshold}")
                    
                    if zscore > self.zscore_upper_threshold:
                        logger.print(f"SELL SIGNAL - Z-score: {zscore:.4f} > {self.zscore_upper_threshold}")
                        
                        for voucher in self.voucher_symbols:
                            # Check if deep out of the money before placing orders
                            strike = STRIKES[voucher]
                            if rock_price - strike <= -250:
                                logger.print(f"SKIPPING {voucher} - Too far out of money: rock:{rock_price}, strike:{strike}")
                                continue
                                
                            order_depth = state.order_depths.get(voucher)
                            # Ensure we have an order depth and buy orders to determine the best bid
                            if order_depth and order_depth.buy_orders:
                                best_bid = max(order_depth.buy_orders.keys())
                                price_to_sell = best_bid # Place sell order at best bid
                            
                                order_size = self.calculate_order_size(voucher, zscore, state)
                                if order_size != 0:
                                    self.place_order(orders_dict, voucher, price_to_sell, order_size)
                            else:
                                logger.print(f"No best bid found for {voucher} to place sell order.") # Log if no bid exists
                    
                    elif zscore < self.zscore_lower_threshold:
                        logger.print(f"BUY SIGNAL - Z-score: {zscore:.4f} < {self.zscore_lower_threshold}")
                        
                        for voucher in self.voucher_symbols:
                            # Check if deep out of the money before placing orders
                            strike = STRIKES[voucher]
                            if rock_price - strike <= -250:
                                logger.print(f"SKIPPING {voucher} - Too far out of money: rock:{rock_price}, strike:{strike}")
                                continue
                                
                            order_depth = state.order_depths.get(voucher)
                            # Ensure we have an order depth and sell orders to determine the best ask
                            if order_depth and order_depth.sell_orders:
                                best_ask = min(order_depth.sell_orders.keys())
                                price_to_buy = best_ask # Place buy order at best ask
                                
                                order_size = self.calculate_order_size(voucher, zscore, state)
                                if order_size != 0:
                                    self.place_order(orders_dict, voucher, price_to_buy, order_size)
                            else:
                                logger.print(f"No best ask found for {voucher} to place buy order.") # Log if no ask exists
            
            except Exception as e:
                logger.print(f"Error: {e}")
                import traceback
                logger.print(traceback.format_exc())
        
        self.orders = orders_dict
        return orders_dict
    
    def save_state(self) -> dict:
        """Save the current state of the trader."""
        return {
            "day": self.day,
            "base_iv_history": list(self.base_iv_history),
            "short_ewma_base_iv": self.short_ewma_base_iv,
            "long_ewma_first": self.long_ewma_first,
            "long_ewma_base_iv": self.long_ewma_base_iv,
            "ewma_diff_history": list(self.ewma_diff_history) if hasattr(self, 'ewma_diff_history') else [],
            "zscore_history": list(self.zscore_history),
            "zscore_upper_threshold": self.zscore_upper_threshold,
            "zscore_lower_threshold": self.zscore_lower_threshold,
            "last_timestamp": self.last_timestamp
        }
    
    def load_state(self, data: dict) -> None:
        """Load the state of the trader."""
        if not data:
            return
            
        self.day = data.get("day", self.day)
        
        base_iv_history = data.get("base_iv_history", [])
        self.base_iv_history = deque(base_iv_history, maxlen=200)
        
        self.short_ewma_base_iv = data.get("short_ewma_base_iv")
        self.long_ewma_first = data.get("long_ewma_first")
        self.long_ewma_base_iv = data.get("long_ewma_base_iv")
        
        ewma_diff_history = data.get("ewma_diff_history", [])
        self.ewma_diff_history = deque(ewma_diff_history, maxlen=200)
        
        zscore_history = data.get("zscore_history", [])
        self.zscore_history = deque(zscore_history, maxlen=100)
        
        self.zscore_upper_threshold = data.get("zscore_upper_threshold", self.zscore_upper_threshold)
        self.zscore_lower_threshold = data.get("zscore_lower_threshold", self.zscore_lower_threshold)
        
        self.last_timestamp = data.get("last_timestamp")



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

        self.volcanicposition_limits={
            VOLCANIC_ROCK: 400,
            VOLCANIC_ROCK_VOUCHER_9500: 200, 
            VOLCANIC_ROCK_VOUCHER_9750: 200, 
            VOLCANIC_ROCK_VOUCHER_10000: 200, 
            VOLCANIC_ROCK_VOUCHER_10250: 200, 
            VOLCANIC_ROCK_VOUCHER_10500: 200
        }

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
        self.strategies = {
            VOLCANIC_ROCK: RsiStrategy(VOLCANIC_ROCK, self.volcanicposition_limits[VOLCANIC_ROCK]),
            "VOLATILITY_SMILE": VolatilitySmileStrategy()
        }
        self.position_limits = LIMIT

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
        Calculate the fraction of the position for a given 
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
        all_orders: List[Order] = []
        result = {product: [] for product in LIMIT}
        conversions = 0
        trader_data = ""
        trader_data_for_next_round = {}  
        # Updating Position and Position Counters
        for product in self.pos_limits.keys():
            self.UpdatePreviousPositionCounter(product,state)
            self.position[product] = state.position.get(product,0)
            try:
                self.update_market_data(product, state)
            except Exception:
                pass
        try:
            loaded_data = json.loads(state.traderData) if state.traderData and state.traderData != '""' else {}
            if not isinstance(loaded_data, dict): loaded_data = {}
        except Exception as e:
            logger.print(f"Error loading traderData: {e}")
            loaded_data = {}

        for strategy_key, strategy in self.strategies.items():
            # Load state for this specific strategy
            strategy_state = loaded_data.get(str(strategy_key), {}) # Use str(key) just in case
            if isinstance(strategy_state, dict):
                try: strategy.load(strategy_state)
                except Exception as e: logger.print(f"Error loading state for {strategy_key}: {e}")
            else: strategy.load({}) # Load empty state if not dict

            # Check if market data is available for the strategy
            market_data_available = True
            required_products = []
            if isinstance(strategy, B1B2DeviationStrategy):
                required_products = [PICNIC_BASKET1, PICNIC_BASKET2, CROISSANTS, JAMS, DJEMBES]
            elif isinstance(strategy, VolatilitySmileStrategy):
                # Check for Volcanic Rock and at least one voucher
                required_products = [VOLCANIC_ROCK]
                voucher_data_available = any(voucher in state.order_depths for voucher in 
                    [VOLCANIC_ROCK_VOUCHER_9500, VOLCANIC_ROCK_VOUCHER_9750, VOLCANIC_ROCK_VOUCHER_10000, VOLCANIC_ROCK_VOUCHER_10250, VOLCANIC_ROCK_VOUCHER_10500])
                if not voucher_data_available:
                    market_data_available = False
            elif str(strategy_key) in self.volcanicposition_limits: # Assume other keys are single products if in limits
                required_products = [str(strategy_key)]
            else:
                
                logger.print(f"Warning: Unsure how to check market data availability for strategy key: {strategy_key}")
                # market_data_available = False # Be conservative?

            if any(prod not in state.order_depths for prod in required_products):
                if required_products: # Only log if we actually knew which products were needed
                    logger.print(f"Strategy {strategy_key}: Market data missing for required products ({[p for p in required_products if p not in state.order_depths]}). Skipping run.")
                market_data_available = False

            if market_data_available:
                try:
                    if isinstance(strategy, VolatilitySmileStrategy):
                        # For VolatilitySmileStrategy, we call run with a different interface
                        orders_from_strategy = strategy.run(state)
                        for symbol, orders in orders_from_strategy.items():
                            all_orders.extend(orders)
                    else:
                        # For other strategies, use the original approach
                        strategy.run(state) # This calls the strategy's act method
                        all_orders.extend(strategy.orders) # Add this strategy's orders to the main list
                except Exception as e:
                    logger.print(f"*** ERROR running {strategy_key} strategy: {e} ***");
                    import traceback; logger.print(traceback.format_exc())

            # Save state for this strategy
            try: 
                if isinstance(strategy, VolatilitySmileStrategy):
                    trader_data_for_next_round[str(strategy_key)] = strategy.save_state()
                else:
                    trader_data_for_next_round[str(strategy_key)] = strategy.save()
            except Exception as e:
                logger.print(f"Error saving state for {strategy_key}: {e}")
                trader_data_for_next_round[str(strategy_key)] = {}


        for order in all_orders:
            # --- Ensure quantity is integer ---
            if not isinstance(order.quantity, int):
                logger.print(f"Warning: Order quantity was not int for {order.symbol}: {order.quantity}. Rounding.")
                order.quantity = int(round(order.quantity))
            # --- Ensure price is integer ---
            if not isinstance(order.price, int):
                logger.print(f"Warning: Order price was not int for {order.symbol}: {order.price}. Rounding.")
                order.price = int(round(order.price))

            if order.quantity != 0: # Don't submit zero quantity orders
                result[order.symbol].append(order)

        # if VOLCANIC_ROCK in state.order_depths:
        #     rock_position = state.position.get(VOLCANIC_ROCK, 0)
        #     rock_params = PARAMS[VOLCANIC_ROCK]
        #     rock_order_depth = state.order_depths[VOLCANIC_ROCK]

        #     # Calculate time to expiry
        #     time_expiry = (PARAMS[VOLCANIC_ROCK]["starting_time_to_expiry"] - (state.timestamp) / 1000000 / 365)

        #     # Retrieve fair call option prices for volcanic rock vouchers
        #     fair_prices = self.calc_fair_price(T=time_expiry)
        #     undercut = 1.3228


        # for voucher, (strike, mid_price, bs_price, bs_delta, bs_gamma, bs_vega) in fair_prices.items():
        #     if voucher in state.order_depths:
        #         voucher_order_depth = state.order_depths[voucher]

        #         buy_order_depth = voucher_order_depth.buy_orders
        #         sell_order_depth = voucher_order_depth.sell_orders

        #         # Skip if no orders exist
        #         if not buy_order_depth or not sell_order_depth:
        #             continue
        #         else:

        #             # Calculate allowed trading volumes for voucher
        #             voucher_position = self.position[voucher]
        #             voucher_buy_volume = self.pos_limits[voucher] - voucher_position
        #             voucher_sell_volume = self.pos_limits[voucher] - voucher_position

        #             # Fill orders more aggressively
        #             if mid_price < bs_price - undercut:
        #                 # For sell side: get orders to sell the voucher
        #                 sell_orders = self.OrderOptimised(voucher, voucher_sell_volume, mode='sell', state=state)
        #                 for order in sell_orders:
        #                     result[voucher].append(order)
        #             elif mid_price > bs_price + undercut:
        #                 # For buy side: get orders to buy the voucher
        #                 buy_orders = self.OrderOptimised(voucher, voucher_buy_volume, mode='buy', state=state)
        #                 for order in buy_orders:
        #                     result[voucher].append(order)

        #     rock_price_dict = {
        #         # VOLCANIC_ROCK: (self.rock_mid_prices,
        #         #                 state.order_depths[VOLCANIC_ROCK],
        #         #                 state.position.get(VOLCANIC_ROCK, 0),
        #         #                 self.pos_limits[VOLCANIC_ROCK]),
        #         VOLCANIC_ROCK_VOUCHER_9500: (self.rock9500_mid_prices,
        #                                     state.order_depths[VOLCANIC_ROCK_VOUCHER_9500],
        #                                     state.position.get(VOLCANIC_ROCK_VOUCHER_9500, 0),
        #                                     self.pos_limits[VOLCANIC_ROCK_VOUCHER_9500]),
        #         VOLCANIC_ROCK_VOUCHER_9750: (self.rock9750_mid_prices,
        #                                     state.order_depths[VOLCANIC_ROCK_VOUCHER_9750],
        #                                     state.position.get(VOLCANIC_ROCK_VOUCHER_9750, 0),
        #                                     self.pos_limits[VOLCANIC_ROCK_VOUCHER_9750]),
        #         VOLCANIC_ROCK_VOUCHER_10000: (self.rock10000_mid_prices,
        #                                     state.order_depths[VOLCANIC_ROCK_VOUCHER_10000],
        #                                     state.position.get(VOLCANIC_ROCK_VOUCHER_10000, 0),
        #                                     self.pos_limits[VOLCANIC_ROCK_VOUCHER_10000]),
        #         VOLCANIC_ROCK_VOUCHER_10250: (self.rock10250_mid_prices,
        #                                     state.order_depths[VOLCANIC_ROCK_VOUCHER_10250],
        #                                     state.position.get(VOLCANIC_ROCK_VOUCHER_10250, 0),
        #                                     self.pos_limits[VOLCANIC_ROCK_VOUCHER_10250]),
        #         VOLCANIC_ROCK_VOUCHER_10500: (self.rock10500_mid_prices,
        #                                     state.order_depths[VOLCANIC_ROCK_VOUCHER_10500],
        #                                     state.position.get(VOLCANIC_ROCK_VOUCHER_10500, 0),
        #                                     self.pos_limits[VOLCANIC_ROCK_VOUCHER_10500]),
        #     }

        #     # Loop over each product and apply the mean reversion strategy.
        #     selloffAggro = 0.4195
        #     selloffThreshold = 0.0213
        #     zscoreThresh = 1.5334
        #     greed = 5.7 #How much return we want before dumping 1.1744
        #     for product, (mid_prices, o_depth, pos, pos_limit) in rock_price_dict.items():
        #         PositionFraction = self.PositionFraction(product, state)
        #         if PositionFraction > selloffThreshold and self.Profitable(product, state,greed=greed,mode='sell'):
        #             result[product]+=self.OrderOptimised(product,mode='sell',size=int(selloffAggro*pos),state=state)
        #         if PositionFraction <-selloffThreshold and self.Profitable(product, state,greed=greed,mode='buy'):
        #             result[product]+=self.OrderOptimised(product,mode='buy',size=int(selloffAggro*pos),state=state)
        #             order=self.mean_reversion_trade(product=product, mid_prices=mid_prices, order_depth=o_depth,
        #                                             current_position= pos, position_limit=pos_limit,window=50, z_score_thresh=int(zscoreThresh), state=state)
        #             if order:
        #                 result[product]+=order



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

        try:
            # Ensure keys are strings for JSON
            trader_data_to_encode = {str(k): v for k, v in trader_data_for_next_round.items()}
            traderData_encoded = json.dumps(trader_data_to_encode, separators=(",", ":"), cls=ProsperityEncoder)
        except Exception as e:
            logger.print(f"Error encoding traderData: {e}")
            traderData_encoded = "{}"

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data