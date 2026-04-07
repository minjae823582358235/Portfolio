import json
import jsonpickle
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque

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
    }
}



### PB Synthetic Basket Parameters #######################
synth1Mean = -131.606  # PB1 is usually cheaper than PB2
synth1Sigma = np.round(29.05 // np.sqrt(1000), decimals=5)
s1Zscore = 0.1260  # TODO OPTIMISE!!!!!! 1 works pretty well
synth2Mean = 105.417
synth2Sigma = np.round(27.166 // np.sqrt(1000), decimals=5)
s2Zscore = 1.4417  # TODO OPTIMISE!!!!!!
diff_threshold_b1_b2 = 176.8118
#### Kelp Squink Pairs Trade Parameters
KSZscore = 0.190327



#### DYNAMIC INVENTORY FACTOR PARAMS#### TODO APPLY PARAMS TO ALL PRODUCTS
pb1high = 22.0129
pb1mid = 6.1441
pb1low = -9.9265
pb1neg = 8.9899

pb2high = 39.2463
pb2mid = 26.8949
pb2low = 5.3674
pb2neg = -26.1430

djembeshigh = 22.4723 #not being optimised
djembesmid = 16.7552 #not being optimised
djembeslow = -4.4401 #not being optimised 
djembesneg = -9.0611 #not being optimised

squidinkhigh = 26.244 #not being optimised
squidinkmid = 3.57077 #not being optimised
squidinklow = 1.0171 #not being optimised 
squidinkneg = -19.168886 #not being optimised

kelphigh = 35.561048 #not being optimised
kelpmid = 34.49889 #not being optimised
kelplow = 9.11303 #not being optimised
kelpneg = 1.2853198 #not being optimised
##############################################

#### HOLD FACTOR PARAMS ####
pb1Hold = 6.7626
pb2Hold = 3.9930
djembeHold = 1.7 #not being optimised
squidinkHold = 28.964 #not being optimised
kelpHold = 28.964 #not being optimised
###############################################

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

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def _place_buy_order(self, price: float, quantity: float) -> None:
        if quantity <= 0:
            return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <= 0:
            return
        self.orders.append(Order(self.symbol, price, quantity))

    def _place_sell_order(self, price: float, quantity: float) -> None:
        if quantity <= 0:
            return
        price = int(round(price))
        quantity = int(math.floor(quantity))
        if quantity <= 0:
            return
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass

    def _get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
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

class SquidInkRsiStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        self.params = PARAMS.get(self.symbol, {})
        if not self.params:
            self.params = {"rsi_window": 14, "rsi_overbought": 70.0, "rsi_oversold": 30.0}
        self.window = self.params.get("rsi_window", 14)
        if self.window < 2:
            self.window = 2
        self.overbought_threshold = self.params.get("rsi_overbought", 70.0)
        self.oversold_threshold = self.params.get("rsi_oversold", 30.0)
        self.mid_price_history: deque[float] = deque(maxlen=self.window + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_initialized: bool = False

    def _calculate_rsi(self, current_mid_price: float) -> Optional[float]:
        self.mid_price_history.append(current_mid_price)
        if len(self.mid_price_history) < self.window + 1:
            return None
        prices = list(self.mid_price_history)
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [max(change, 0) for change in changes]
        losses = [abs(min(change, 0)) for change in changes]

        if not self.rsi_initialized:
            self.avg_gain = sum(gains) / self.window
            self.avg_loss = sum(losses) / self.window
            self.rsi_initialized = True
        else:
            current_gain = gains[-1]
            current_loss = losses[-1]
            self.avg_gain = ((self.avg_gain * (self.window - 1)) + current_gain) / self.window
            self.avg_loss = ((self.avg_loss * (self.window - 1)) + current_loss) / self.window

        if self.avg_loss == 0:
            return 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)
        if not order_depth:
            return

        best_bid_price = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask_price = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid_price is None or best_ask_price is None:
            return

        current_mid_price = (best_bid_price + best_ask_price) / 2.0
        rsi_value = self._calculate_rsi(current_mid_price)
        if rsi_value is None:
            return

        to_buy_capacity = self.position_limit - position
        to_sell_capacity = self.position_limit + position

        # If RSI is too high, sell; if too low, buy.
        if rsi_value > self.overbought_threshold and to_sell_capacity > 0:
            size_to_sell = to_sell_capacity
            aggressive_sell_price = best_bid_price - 1
            self._place_sell_order(aggressive_sell_price, size_to_sell)
        elif rsi_value < self.oversold_threshold and to_buy_capacity > 0:
            size_to_buy = to_buy_capacity
            aggressive_buy_price = best_ask_price + 1
            self._place_buy_order(aggressive_buy_price, size_to_buy)
    def save(self) -> dict:
        return {
            "mid_price_history": list(self.mid_price_history),
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "rsi_initialized": self.rsi_initialized,
        }

    def load(self, data: dict) -> None:
        loaded_history = data.get("mid_price_history", [])
        if isinstance(loaded_history, list):
            self.mid_price_history = deque(loaded_history, maxlen=self.window + 1)
        else:
            self.mid_price_history = deque(maxlen=self.window + 1)
        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        self.rsi_initialized = data.get("rsi_initialized", False)
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
        if not isinstance(self.rsi_initialized, bool):
            self.rsi_initialized = False
        if self.rsi_initialized and (self.avg_gain is None or self.avg_loss is None):
            self.rsi_initialized = False
            self.avg_gain = None
            self.avg_loss = None

class Trader: ##from here

    def __init__(self) -> None:
        self.logger = Logger()
        self.logger.print("Initialized trader for JAMS and DJEMBES only.")
        self.pos_limits = LIMIT
        self.diff_threshold_b1_b2 = diff_threshold_b1_b2
        self.lot_size = 1

        self.previousposition = {params:0 for params in LIMIT.keys()}
        self.position = {params:0 for params in LIMIT.keys()}
        self.positionCounter = {params:0 for params in LIMIT.keys()}

        self.resin_timestamps = []
        self.resin_mid_prices = []
        self.kelp_timestamps = []
        self.kelp_mid_prices = []
        self.ink_timestamps = []
        self.ink_mid_prices = []

        # pairs trading strategy for ink and kelp
        self.m = None
        self.c = None
        self.mean = None
        self.stddev = None
        self.lookbackwindow = 45
        self.strategy = SquidInkRsiStrategy(SQUID_INK, LIMIT[SQUID_INK])
    
    def update_market_data(self, product, state):
        # Store current timestamp and mid-price for all assets
        order_depth = state.order_depths[product]
        mid = mid_price(order_depth)
        if product == "RAINFOREST_RESIN":
            self.resin_timestamps.append(state.timestamp)
            self.resin_mid_prices.append(mid)
        elif product == KELP:
            self.kelp_timestamps.append(state.timestamp)
            self.kelp_mid_prices.append(mid)
        elif product == SQUID_INK:
            self.ink_timestamps.append(state.timestamp)
            self.ink_mid_prices.append(mid)

    def UpdatePreviousPositionCounter(self,product,state:TradingState) -> None:
            if product not in set(state.position.keys()):
                return None
            if (
                state.position[product] == self.previousposition[product]
            ):  # Updates previouspositionCounter
                self.positionCounter[product] += 1
            else:
                self.positionCounter[product] = 0

    def PriceAdjustment(self,product, mode=None,state=TradingState):
        if product not in set(state.position.keys()):
            return 0
        AdjustmentDict = {
            PICNIC_BASKET1: {
                "high": pb1high,
                "mid": pb1mid,
                "low": pb1low,
                "neg": pb1neg,
            },
            PICNIC_BASKET2: {
                "high": pb2high,
                "mid": pb2mid,
                "low": pb2low,
                "neg": pb2neg,
            },
            DJEMBES: {
                "high": djembeshigh,
                "mid": djembesmid,
                "low": djembeslow,
                "neg": djembesneg,
            },
            SQUID_INK: {
                "high": squidinkhigh,
                "mid": squidinkmid,
                "low": squidinklow,
                "neg": squidinkneg,
            },
            KELP: {
                "high": kelphigh,
                "mid": kelpmid,
                "low": kelplow,
                "neg": kelpneg,
            },
        }

        holdFactorDict = {
            PICNIC_BASKET1: pb1Hold,
            PICNIC_BASKET2: pb2Hold,
            DJEMBES: djembeHold,
            SQUID_INK: squidinkHold,
            KELP: kelpHold,
        }

        VolumeFraction = VolumeCapability(product, mode=mode,state=state) / LIMIT[product]
        holdPremium = int(holdFactorDict[product] * self.positionCounter[product])
        high = AdjustmentDict[product]["high"] + holdPremium
        mid = AdjustmentDict[product]["mid"] + holdPremium
        low = AdjustmentDict[product]["low"] + holdPremium
        neg = AdjustmentDict[product]["neg"] + holdPremium

        if product == PICNIC_BASKET1 or product == PICNIC_BASKET2:
            if mode == "buy":
                factor = 1
            if mode == "sell":
                factor = -1
            if VolumeFraction <= 0.1:
                return int(
                    factor * (high + 3)
                )  # FOR SOME REASON PICNIC BASKET 1 LIKES THIS
            if VolumeFraction > 0.1 and VolumeFraction <= 0.2:
                return int(factor * high)
            if VolumeFraction > 0.2 and VolumeFraction < 0.5:
                return int(factor * mid)
            if VolumeFraction >= 0.5 and VolumeFraction < 1:
                return int(factor * low)
            if VolumeFraction >= 1:
                return int(factor * neg)

        if product == DJEMBES:
            if mode == "buy":
                factor = 1
            if mode == "sell":
                factor = -1
            if VolumeFraction <= 0.1:
                return int(factor * high)
            if VolumeFraction > 0.1 and VolumeFraction < 0.5:
                return int(factor * mid)
            if VolumeFraction >= 0.5 and VolumeFraction < 1:
                return int(factor * low)
            if VolumeFraction >= 1:
                return int(factor * neg)

        if product == SQUID_INK or product == KELP:
            if mode == "buy":
                factor = 1
            if mode == "sell":
                factor = -1
            if VolumeFraction <= 0.1:
                return int(factor * high)
            if VolumeFraction > 0.1 and VolumeFraction < 0.5:
                return int(factor * mid)
            if VolumeFraction >= 0.5 and VolumeFraction < 1:
                return int(factor * low)
            if VolumeFraction >= 1:
                return int(factor * neg)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in LIMIT.keys()}
        conversions = 0
        trader_data = ""
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        # Updating Position and Position Counters
        for product in state.position:
            self.UpdatePreviousPositionCounter(product,state)
            self.position[product] = state.position[product]
            self.update_market_data(product, state)
        
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


        ##KELP Trading Method
        if KELP in state.order_depths:
            kelp_position = state.position.get(KELP, 0)
            kelp_params = PARAMS[KELP]
            kelp_order_depth = state.order_depths[KELP]
            kelp_fair = kelp_fair_value(
                kelp_order_depth, kelp_params["default_fair_method"], kelp_params["min_volume_filter"]
            )
            kelp_take, bo, so = kelp_take_orders(kelp_order_depth, kelp_fair, kelp_params, kelp_position)
            kelp_clear, bo, so = kelp_clear_orders(kelp_order_depth, kelp_position, kelp_params, kelp_fair, bo, so)
            kelp_make = kelp_make_orders(kelp_order_depth, kelp_fair, kelp_position, kelp_params, bo, so)
            result[KELP] = kelp_take + kelp_clear + kelp_make

        ## SQUINK!
        try:
            if state.traderData:
                loaded_data = json.loads(state.traderData)
                data_for_squid = loaded_data.get(SQUID_INK, {})
                self.strategy.load(data_for_squid)
            else:
                self.strategy.load({})
        except Exception:
            self.strategy.load({})

        # Run the SQUID_INK strategy
        self.strategy.run(state)
        if self.strategy.orders:
            result[SQUID_INK] = self.strategy.orders

        trader_data_for_next_round = {SQUID_INK: self.strategy.save()}
        trader_data_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"))



        #=======PICNIC BASKET (WHY DOES IT SOUND LIKE CHICKEN JOCKEY)
        if PICNIC_BASKET1 in state.order_depths:
            spread1 = (
                                vwap(PICNIC_BASKET1,state=state)  # FIXME MAYBE THE LOGIC IS WRONG
                                - 1.5 * vwap(PICNIC_BASKET2,state=state)
                                - vwap(DJEMBES,state=state)
                            )
            normspread1 = spread1 - synth1Mean
            spread2 = (
                vwap(PICNIC_BASKET2,state=state) - 4 * vwap(CROISSANTS,state=state) - 2 * vwap(JAMS,state=state)
            )
            normspread2 = spread2 - synth2Mean
            if (
                normspread1
                > synth1Sigma
                * s1Zscore  ##Picnic Basket 1 is overvalued or PB2 OR Djembe is undervalued
            ):  ## sell at the worst bid(cheapest) ##TODO maybe be more aggressive?
                HighestBid = BidPrice(PICNIC_BASKET1, mode="max",state=state)
                HighestVolume = BidVolume(PICNIC_BASKET1, mode="max",state=state)
                result[PICNIC_BASKET1].append(Order(PICNIC_BASKET1,HighestBid + self.PriceAdjustment(PICNIC_BASKET1, mode="sell",state=state),-HighestVolume))

                if (
                    normspread2 > synth2Sigma * s1Zscore
                ):  # assume this means PB2 is overvalued
                    result[PICNIC_BASKET2].append(Order(
                            PICNIC_BASKET2,
                            AskPrice(PICNIC_BASKET2, mode="min",state=state)
                            + self.PriceAdjustment(PICNIC_BASKET2, mode="buy",state=state),
                            AskVolume(PICNIC_BASKET2, mode="min",state=state),
                        ))

                # else:  # DJEMBE is undervalued
                #     # result[DJEMBES].append(Order(
                #     #         DJEMBES,
                #     #         AskPrice(DJEMBES, mode="min",state=state)
                #     #         + self.PriceAdjustment(DJEMBES, mode="buy",state=state),
                #     #         AskVolume(DJEMBES, mode="min",state=state),
                #     #     ))

            if (
                normspread1
                < -synth1Sigma
                * s1Zscore  ##Picnic Basket 1 is undervalued or PB2 OR Djembe is overvalued
            ):  ## buy at the worst ask(most expensive) ##TODO maybe be more aggressive?
                CheapestAsk = AskPrice(PICNIC_BASKET1, mode="min",state=state)
                CheapestVolume = AskVolume(PICNIC_BASKET1, mode="min",state=state)
                result[PICNIC_BASKET1].append(Order(
                        PICNIC_BASKET1,
                        CheapestAsk + self.PriceAdjustment(PICNIC_BASKET1, mode="buy",state=state),
                        CheapestVolume,
                    ))

                if (
                    normspread2 > synth2Sigma * s2Zscore
                ):  # assume this means PB2 is the one that is overvalued
                    result[PICNIC_BASKET2].append(Order(
                            PICNIC_BASKET2,
                            BidPrice(PICNIC_BASKET2, mode="max",state=state)
                            + self.PriceAdjustment(PICNIC_BASKET2, mode="sell",state=state),
                            -BidVolume(PICNIC_BASKET2, mode="max",state=state),
                        ))

                # else:  # Djembe is overvalued
                #     # result[DJEMBES].append(Order(
                #     #         DJEMBES,
                #     #         BidPrice(DJEMBES, mode="max",state=state)
                #     #         + self.PriceAdjustment(DJEMBES, mode="sell",state=state),
                #     #         -BidVolume(DJEMBES, mode="max",state=state),
                #     #     ))
                    
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
            diff_comp = b1 - implied_b1
            self.logger.print(f"Composition signal: BASKET1={b1:.1f}, Implied_BASKET1={implied_b1:.1f}, diff={diff_comp:.1f}")
            signal = 0
            if diff_comp > self.diff_threshold_b1_b2:
                # When diff is significantly positive, the composite view indicates that
                # the basket is overpriced. In the original, this leads to shorting baskets
                # and buying items. For our version we then wish to buy JAMS and DJEMBES.
                signal = +1  # signal to buy the items
            elif diff_comp < -self.diff_threshold_b1_b2:
                # When diff is significantly negative, the items are overpriced relative to the basket.
                # For our version we then signal to sell JAMS and DJEMBES.
                signal = -1  # signal to sell the items
            self.logger.print(f"Signal determined (for JAMS and DJEMBES): {signal}")
        else:
            self.logger.print("Insufficient market data to compute composition signal.")
            signal = 0

        # Only place orders in JAMS and DJEMBES – ignore any orders for other products.
        for prod in [JAMS, DJEMBES]:
            od = state.order_depths.get(prod)
            if not od:
                self.logger.print(f"No order depth for {prod}, skipping.")
                continue
            current_pos = state.position.get(prod, 0)
            pos_limit = self.pos_limits[prod]
            if signal == +1:
                # Buy signal: check if we have capacity to buy.
                capacity = pos_limit - current_pos
                if capacity <= 0:
                    self.logger.print(f"No buying capacity for {prod} (current position: {current_pos}).")
                    continue
                trade_qty = min(self.lot_size, capacity)
                # For a buy order, mimic the original synergy logic:
                # use best ask price + 1 as our bid price.
                if od.sell_orders:
                    best_ask_price = min(od.sell_orders.keys())
                    price = best_ask_price + 1
                    result[prod].append(Order(prod, price, trade_qty))
                    self.logger.print(f"Placing BUY order for {prod}: {trade_qty}x{price}")
                else:
                    self.logger.print(f"No sell orders for {prod}, cannot place BUY order.")
            elif signal == -1:
                # Sell signal: check if we have capacity to sell.
                capacity = pos_limit + current_pos  # since short positions count against limit
                if capacity <= 0:
                    self.logger.print(f"No selling capacity for {prod} (current position: {current_pos}).")
                    continue
                trade_qty = min(self.lot_size, capacity)
                if od.buy_orders:
                    best_bid_price = max(od.buy_orders.keys())
                    price = best_bid_price - 1
                    result[prod].append(Order(prod, price, -trade_qty))
                    self.logger.print(f"Placing SELL order for {prod}: {trade_qty}x{price}")
                else:
                    self.logger.print(f"No buy orders for {prod}, cannot place SELL order.")
            else:
                self.logger.print(f"No clear signal for {prod} – no order placed.")



        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data