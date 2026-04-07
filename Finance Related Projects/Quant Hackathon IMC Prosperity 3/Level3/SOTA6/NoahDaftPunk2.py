import json
import jsonpickle
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod
from collections import deque, defaultdict

# Import data model types (assumed to be defined in your environment)
from datamodel import Listing, Observation, OrderDepth, Order, Symbol, Trade, TradingState, ProsperityEncoder


# =============================================================================
# Logger – common to both parts
# =============================================================================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_trades(self, trades: dict[Symbol, List[Trade]]) -> list[list[Any]]:
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
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, List[Order]]) -> list[list[Any]]:
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


# =============================================================================
# Product Definitions and Basket Info
# =============================================================================
class Product:
    KELP = "KELP"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    B1B2_DEVIATION = "B1B2_DEVIATION"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


BASKET_COMPONENTS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2
    }
}

B1B2_THEORETICAL_COMPONENTS = {
    Product.CROISSANTS: 2,
    Product.JAMS: 1,
    Product.DJEMBES: 1
}


# =============================================================================
# Combined Parameters and Limits
# =============================================================================
# For non-volcanic products we use parameters taken from Round2SOTA4.
# For volcanic rock we keep the unchanged parameters from our previous code.
COMBINED_PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 0.5,
        "join_edge": 2,
        "default_edge": 2,
        "soft_position_limit": 45,
    },
    Product.KELP: {
        "take_width": 18.7017,
        "position_limit": 50,
        "min_volume_filter": 21.7543,
        "spread_edge": 1.0760,
        "default_fair_method": "vwap_with_vol_filter",
    },
    Product.SQUID_INK: {
        "rsi_window": 106,
        "rsi_overbought": 52,
        "rsi_oversold": 41,
    },
    Product.CROISSANTS: {
        "history_length": 83.8247,
        "z_threshold": 1.9986
    },
    # Volcanic rock parameters (kept unchanged)
    Product.VOLCANIC_ROCK: {
        "rsi_window": 85,
        "rsi_overbought": 52,
        "rsi_oversold": 42,
        "price_offset": 0
    }
}

LIMIT = {
    Product.RAINFOREST_RESIN: 50,
    Product.KELP: 50,
    Product.SQUID_INK: 50,
    Product.CROISSANTS: 250,
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 100,
    Product.JAMS: 350,
    Product.DJEMBES: 60,
    Product.VOLCANIC_ROCK: 400,
    Product.VOUCHER_9500: 200,
    Product.VOUCHER_9750: 200,
    Product.VOUCHER_10000: 200,
    Product.VOUCHER_10250: 200,
    Product.VOUCHER_10500: 200,
}


# =============================================================================
# Black-Scholes and Option Functions (used for volcanic voucher pricing)
# =============================================================================
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def calculate_option_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def calculate_implied_volatility(option_price, S, K, T, r=0, initial_vol=0.3, max_iterations=50, precision=0.0001):
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
        d1 = (math.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
        vega = S * math.sqrt(T) * norm_pdf(d1)
        if vega == 0:
            return vol
        vol = vol + diff / vega
        if vol <= 0:
            vol = 0.0001
        elif vol > 5:
            vol = 5.0
    return vol


# =============================================================================
# Volcanic Rock Trading Strategies (unchanged)
# =============================================================================
class RsiStrategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.params = COMBINED_PARAMS.get(symbol, {"rsi_window": 85, "rsi_overbought": 52, "rsi_oversold": 42, "price_offset": 0})
        self.window = self.params.get("rsi_window", 14)
        if self.window < 2:
            self.window = 2
        self.overbought_threshold = self.params.get("rsi_overbought", 70.0)
        self.oversold_threshold = self.params.get("rsi_oversold", 30.0)
        self.price_offset = self.params.get("price_offset", 0)
        self.mid_price_history: deque[float] = deque(maxlen=self.window + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_initialized: bool = False
        logger.print(f"Initialized RsiStrategy for {self.symbol}")

    def _calculate_rsi(self, current_mid_price: float) -> Optional[float]:
        self.mid_price_history.append(current_mid_price)
        if len(self.mid_price_history) < self.window + 1:
            return None
        prices = list(self.mid_price_history)
        changes = np.diff(prices)
        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))
        if not self.rsi_initialized:
            self.avg_gain = np.mean(gains[-self.window:])
            self.avg_loss = np.mean(losses[-self.window:])
            self.rsi_initialized = True
        else:
            current_gain = gains[-1]
            current_loss = losses[-1]
            self.avg_gain = ((self.avg_gain * (self.window - 1)) + current_gain) / self.window
            self.avg_loss = ((self.avg_loss * (self.window - 1)) + current_loss) / self.window
        if self.avg_loss < 1e-9:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)
        if not order_depth:
            return
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return
        current_mid_price = (best_bid + best_ask) / 2.0
        rsi_value = self._calculate_rsi(current_mid_price)
        if rsi_value is None:
            return
        to_buy_capacity = self.position_limit - position
        to_sell_capacity = self.position_limit + position
        if rsi_value > self.overbought_threshold and to_sell_capacity > 0:
            self._place_sell_order(best_bid - self.price_offset, to_sell_capacity)
        elif rsi_value < self.oversold_threshold and to_buy_capacity > 0:
            self._place_buy_order(best_ask + self.price_offset, to_buy_capacity)

    def _place_buy_order(self, price: float, quantity: int) -> None:
        if quantity <= 0:
            return
        # Here we simply store the order in an internal list.
        self.order_append(Order(self.symbol, int(round(price)), int(math.floor(quantity))))

    def _place_sell_order(self, price: float, quantity: int) -> None:
        if quantity <= 0:
            return
        self.order_append(Order(self.symbol, int(round(price)), -int(math.floor(quantity))))

    def order_append(self, order: Order) -> None:
        if not hasattr(self, "orders"):
            self.orders = []
        self.orders.append(order)

    def save(self) -> dict:
        return {
            "mid_price_history": list(self.mid_price_history),
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "rsi_initialized": self.rsi_initialized
        }

    def load(self, data: dict) -> None:
        loaded_history = data.get("mid_price_history", [])
        self.mid_price_history = deque(loaded_history, maxlen=self.window + 1)
        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        self.rsi_initialized = data.get("rsi_initialized", False)


class VolatilitySmileStrategy:
    def __init__(self) -> None:
        self.position_limits = {
            Product.VOLCANIC_ROCK: LIMIT[Product.VOLCANIC_ROCK],
            Product.VOUCHER_9500: LIMIT[Product.VOUCHER_9500],
            Product.VOUCHER_9750: LIMIT[Product.VOUCHER_9750],
            Product.VOUCHER_10000: LIMIT[Product.VOUCHER_10000],
            Product.VOUCHER_10250: LIMIT[Product.VOUCHER_10250],
            Product.VOUCHER_10500: LIMIT[Product.VOUCHER_10500],
        }
        self.voucher_symbols = [
            Product.VOUCHER_9500,
            Product.VOUCHER_9750,
            Product.VOUCHER_10000,
            Product.VOUCHER_10250,
            Product.VOUCHER_10500
        ]
        self.short_ewma_span = 37
        self.long_ewma_span = 68
        self.rolling_window = 48
        self.zscore_upper_threshold = 0.5
        self.zscore_lower_threshold = -2.8
        self.trade_size = 22
        self.base_iv_history = deque(maxlen=200)
        self.short_ewma_base_iv = None
        self.long_ewma_first = None
        self.long_ewma_base_iv = None
        self.ewma_diff_history = deque(maxlen=200)
        self.zscore_history = deque(maxlen=100)
        self.day = 3
        self.last_timestamp = None
        self.orders = {}

    def update_time_to_expiry(self, timestamp):
        base_tte = 8 - self.day
        iteration = (timestamp % 1000000) // 100
        iteration_adjustment = iteration / 10000
        tte = (base_tte - iteration_adjustment) / 365
        return max(0.0001, tte)

    def get_mid_price(self, symbol: Symbol, state: TradingState) -> Optional[float]:
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
        current_position = state.position.get(symbol, 0)
        position_limit = self.position_limits.get(symbol, 0)
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
        if quantity == 0:
            return
        if symbol not in orders_dict:
            orders_dict[symbol] = []
        orders_dict[symbol].append(Order(symbol, price, quantity))
        logger.print(f"PLACE {symbol} {'BUY' if quantity > 0 else 'SELL'} {abs(quantity)}x{price}")

    def update_ewma(self, current_value, previous_ewma, span):
        if previous_ewma is None:
            return current_value
        alpha = 2 / (span + 1)
        return alpha * current_value + (1 - alpha) * previous_ewma

    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        orders_dict = {}
        self.last_timestamp = state.timestamp
        time_to_expiry = self.update_time_to_expiry(state.timestamp)
        rock_price = self.get_mid_price(Product.VOLCANIC_ROCK, state)
        if not rock_price:
            logger.print("No price available for VOLCANIC_ROCK, skipping iteration")
            return orders_dict

        # Risk management: flatten voucher positions if deep OTM.
        for voucher in self.voucher_symbols:
            strike = {Product.VOUCHER_9500: 9500, Product.VOUCHER_9750: 9750,
                      Product.VOUCHER_10000: 10000, Product.VOUCHER_10250: 10250,
                      Product.VOUCHER_10500: 10500}[voucher]
            if rock_price - strike <= -250:
                current_position = state.position.get(voucher, 0)
                if current_position != 0:
                    voucher_price = self.get_mid_price(voucher, state)
                    if voucher_price and current_position > 0:
                        od = state.order_depths.get(voucher)
                        if od and od.buy_orders:
                            best_bid = max(od.buy_orders.keys())
                            self.place_order(orders_dict, voucher, best_bid, -current_position)
                    elif voucher_price and current_position < 0:
                        od = state.order_depths.get(voucher)
                        if od and od.sell_orders:
                            best_ask = min(od.sell_orders.keys())
                            self.place_order(orders_dict, voucher, best_ask, -current_position)
        moneyness_values = []
        iv_values = []
        for voucher in self.voucher_symbols:
            voucher_price = self.get_mid_price(voucher, state)
            if not voucher_price:
                continue
            strike = {Product.VOUCHER_9500: 9500, Product.VOUCHER_9750: 9750,
                      Product.VOUCHER_10000: 10000, Product.VOUCHER_10250: 10250,
                      Product.VOUCHER_10500: 10500}[voucher]
            moneyness = math.log(strike / rock_price) / math.sqrt(time_to_expiry)
            impl_vol = calculate_implied_volatility(voucher_price, rock_price, strike, time_to_expiry)
            if impl_vol > 0:
                moneyness_values.append(moneyness)
                iv_values.append(impl_vol)
        if len(moneyness_values) >= 3:
            try:
                coeffs = np.polyfit(moneyness_values, iv_values, 2)
                a, b, c = coeffs
                base_iv = c
                logger.print(f"Base IV (ATM): {base_iv:.6f}")
                self.base_iv_history.append(base_iv)
                self.short_ewma_base_iv = self.update_ewma(base_iv, self.short_ewma_base_iv, self.short_ewma_span)
                self.long_ewma_first = self.update_ewma(base_iv, self.long_ewma_first, self.long_ewma_span)
                self.long_ewma_base_iv = self.update_ewma(self.long_ewma_first, self.long_ewma_base_iv, self.long_ewma_span)
                if len(self.base_iv_history) >= self.rolling_window and self.short_ewma_base_iv is not None and self.long_ewma_base_iv is not None:
                    ewma_diff = self.short_ewma_base_iv - self.long_ewma_base_iv
                    if not hasattr(self, 'ewma_diff_history'):
                        self.ewma_diff_history = deque(maxlen=200)
                    self.ewma_diff_history.append(ewma_diff)
                    if len(self.ewma_diff_history) >= self.rolling_window:
                        recent = list(self.ewma_diff_history)[-self.rolling_window:]
                        rolling_std = np.std(recent)
                    else:
                        rolling_std = np.std(list(self.ewma_diff_history))
                    zscore = ewma_diff / rolling_std if rolling_std > 0 else 0
                    logger.print(f"Z-score: {zscore:.4f}")
                    if zscore > self.zscore_upper_threshold:
                        for voucher in self.voucher_symbols:
                            strike = {Product.VOUCHER_9500: 9500, Product.VOUCHER_9750: 9750,
                                      Product.VOUCHER_10000: 10000, Product.VOUCHER_10250: 10250,
                                      Product.VOUCHER_10500: 10500}[voucher]
                            if rock_price - strike <= -250:
                                continue
                            od = state.order_depths.get(voucher)
                            if od and od.buy_orders:
                                best_bid = max(od.buy_orders.keys())
                                order_size = self.calculate_order_size(voucher, zscore, state)
                                if order_size:
                                    self.place_order(orders_dict, voucher, best_bid, order_size)
                    elif zscore < self.zscore_lower_threshold:
                        for voucher in self.voucher_symbols:
                            strike = {Product.VOUCHER_9500: 9500, Product.VOUCHER_9750: 9750,
                                      Product.VOUCHER_10000: 10000, Product.VOUCHER_10250: 10250,
                                      Product.VOUCHER_10500: 10500}[voucher]
                            if rock_price - strike <= -250:
                                continue
                            od = state.order_depths.get(voucher)
                            if od and od.sell_orders:
                                best_ask = min(od.sell_orders.keys())
                                order_size = self.calculate_order_size(voucher, zscore, state)
                                if order_size:
                                    self.place_order(orders_dict, voucher, best_ask, order_size)
            except Exception as e:
                logger.print(f"Error: {e}")
        self.orders = orders_dict
        return orders_dict

    def save_state(self) -> dict:
        return {
            "day": self.day,
            "base_iv_history": list(self.base_iv_history),
            "short_ewma_base_iv": self.short_ewma_base_iv,
            "long_ewma_first": self.long_ewma_first,
            "long_ewma_base_iv": self.long_ewma_base_iv,
            "ewma_diff_history": list(self.ewma_diff_history),
            "zscore_history": list(self.zscore_history),
            "zscore_upper_threshold": self.zscore_upper_threshold,
            "zscore_lower_threshold": self.zscore_lower_threshold,
            "last_timestamp": self.last_timestamp
        }

    def load_state(self, data: dict) -> None:
        if not data:
            return
        self.day = data.get("day", self.day)
        self.base_iv_history = deque(data.get("base_iv_history", []), maxlen=200)
        self.short_ewma_base_iv = data.get("short_ewma_base_iv")
        self.long_ewma_first = data.get("long_ewma_first")
        self.long_ewma_base_iv = data.get("long_ewma_base_iv")
        self.ewma_diff_history = deque(data.get("ewma_diff_history", []), maxlen=200)
        self.zscore_history = deque(data.get("zscore_history", []), maxlen=100)
        self.zscore_upper_threshold = data.get("zscore_upper_threshold", self.zscore_upper_threshold)
        self.zscore_lower_threshold = data.get("zscore_lower_threshold", self.zscore_lower_threshold)
        self.last_timestamp = data.get("last_timestamp")


# =============================================================================
# Functions for Other Products (copied verbatim from Round2SOTA4)
# =============================================================================
def sortDict(dictionary: dict):
    return {key: dictionary[key] for key in sorted(dictionary)}

def VolumeCapability(product, mode, state: TradingState):
    if mode == "buy":
        return LIMIT[product] - state.position[product]
    if mode == "sell":
        return state.position[product] + LIMIT[product]

def vwap(product: str, state: TradingState) -> float:
    total_amt = 0
    total_val = 0
    for prc, amt in state.order_depths[product].buy_orders.items():
        total_val += prc * amt
        total_amt += amt
    for prc, amt in state.order_depths[product].sell_orders.items():
        total_val += prc * abs(amt)
        total_amt += abs(amt)
    return np.round(total_val / total_amt, decimals=5)

def mid_price(order_depth: OrderDepth) -> float:
    if order_depth.sell_orders:
        m1, n1 = 0, 0
        for price, amt in order_depth.sell_orders.items():
            m1 += price * amt
            n1 += amt
        m1 = m1 / n1
    else:
        m1 = 0
    if order_depth.buy_orders:
        m2, n2 = 0, 0
        for price, amt in order_depth.buy_orders.items():
            m2 += price * amt
            n2 += amt
        m2 = m2 / n2
    else:
        m2 = 0
    return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)

def AskPrice(product, mode, state: TradingState):
    if product not in state.order_depths:
        return 0
    return max(state.order_depths[product].sell_orders.keys()) if mode=="max" else min(state.order_depths[product].sell_orders.keys())

def BidPrice(product, mode, state: TradingState):
    if product not in state.order_depths:
        return 1000000
    return max(state.order_depths[product].buy_orders.keys()) if mode=="max" else min(state.order_depths[product].buy_orders.keys())

def AskVolume(product, mode, state: TradingState):
    if product not in state.order_depths:
        return 100
    price = AskPrice(product, "max" if mode=="max" else "min", state)
    return abs(state.order_depths[product].sell_orders[price])

def BidVolume(product, mode, state: TradingState):
    if product not in state.order_depths:
        return 100
    price = BidPrice(product, "max" if mode=="max" else "min", state)
    return abs(state.order_depths[product].buy_orders[price])

def resin_take_orders(order_depth: OrderDepth, fair_value: float, position: int, position_limit: int) -> Tuple[List[Order], int, int]:
    orders = []
    buy_order_volume = 0
    sell_order_volume = 0
    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask < fair_value:
            quantity = min(best_ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, best_ask, quantity))
                buy_order_volume += quantity
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        if best_bid > fair_value:
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, best_bid, -quantity))
                sell_order_volume += quantity
    return orders, buy_order_volume, sell_order_volume

def resin_clear_orders(order_depth: OrderDepth, position: int, fair_value: float, position_limit: int,
                         buy_order_volume: int, sell_order_volume: int) -> Tuple[List[Order], int, int]:
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
            orders.append(Order(Product.RAINFOREST_RESIN, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(Product.RAINFOREST_RESIN, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)
    return orders, buy_order_volume, sell_order_volume

def resin_make_orders(order_depth: OrderDepth, fair_value: float, position: int, position_limit: int,
                      buy_order_volume: int, sell_order_volume: int) -> List[Order]:
    orders = []
    aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value]
    bbbf = [price for price in order_depth.buy_orders.keys() if price < fair_value]
    baaf = min(aaf) if aaf else fair_value + 2
    bbbf_val = max(bbbf) if bbbf else fair_value - 2
    buy_quantity = position_limit - (position + buy_order_volume)
    if buy_quantity > 0:
        orders.append(Order(Product.RAINFOREST_RESIN, bbbf_val + 1, buy_quantity))
    sell_quantity = position_limit + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(Product.RAINFOREST_RESIN, baaf - 1, -sell_quantity))
    return orders

def kelp_fair_value(order_depth: OrderDepth, method: str = "vwap_with_vol_filter", min_vol: int = 20) -> float:
    if method == "mid_price":
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2
    elif method == "vwap_with_vol_filter":
        sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
        buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
        if not sell_orders or not buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume if volume != 0 else (best_ask+best_bid)/2
        else:
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume if volume != 0 else (best_ask+best_bid)/2
    else:
        raise ValueError("Unknown fair value method specified.")

def kelp_take_orders(order_depth: OrderDepth, fair_value: float, params: dict, position: int) -> Tuple[List[Order], int, int]:
    orders = []
    buy_order_volume = 0
    sell_order_volume = 0
    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask <= fair_value - params["take_width"] and ask_amount <= 50:
            quantity = min(ask_amount, params["position_limit"] - position)
            if quantity > 0:
                orders.append(Order(Product.KELP, int(best_ask), quantity))
                buy_order_volume += quantity
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        bid_amount = order_depth.buy_orders[best_bid]
        if best_bid >= fair_value + params["take_width"] and bid_amount <= 50:
            quantity = min(bid_amount, params["position_limit"] + position)
            if quantity > 0:
                orders.append(Order(Product.KELP, int(best_bid), -quantity))
                sell_order_volume += quantity
    return orders, buy_order_volume, sell_order_volume

def kelp_clear_orders(order_depth: OrderDepth, position: int, params: dict, fair_value: float,
                        buy_order_volume: int, sell_order_volume: int) -> Tuple[List[Order], int, int]:
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
            orders.append(Order(Product.KELP, int(fair_for_ask), -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(Product.KELP, int(fair_for_bid), abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)
    return orders, buy_order_volume, sell_order_volume

def kelp_make_orders(order_depth: OrderDepth, fair_value: float, position: int, params: dict,
                     buy_order_volume: int, sell_order_volume: int) -> List[Order]:
    orders = []
    edge = params["spread_edge"]
    aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + edge]
    bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - edge]
    baaf = min(aaf) if aaf else fair_value + edge + 1
    bbbf = max(bbf) if bbf else fair_value - edge - 1
    buy_quantity = params["position_limit"] - (position + buy_order_volume)
    if buy_quantity > 0:
        orders.append(Order(Product.KELP, int(bbbf + 1), buy_quantity))
    sell_quantity = params["position_limit"] + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(Product.KELP, int(baaf - 1), -sell_quantity))
    return orders

class SquidInkRsiStrategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.params = COMBINED_PARAMS.get(symbol, {"rsi_window": 106, "rsi_overbought": 52, "rsi_oversold": 41})
        self.window = self.params.get("rsi_window", 14)
        if self.window < 2:
            self.window = 2
        self.overbought_threshold = self.params.get("rsi_overbought", 70.0)
        self.oversold_threshold = self.params.get("rsi_oversold", 30.0)
        self.mid_price_history: deque[float] = deque(maxlen=self.window + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_initialized: bool = False
        self.orders: List[Order] = []

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
        rs = self.avg_gain / self.avg_loss
        return 100 - (100 / (1 + rs))

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths.get(self.symbol)
        position = state.position.get(self.symbol, 0)
        if not order_depth:
            return
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return
        current_mid_price = (best_bid + best_ask) / 2.0
        rsi_value = self._calculate_rsi(current_mid_price)
        if rsi_value is None:
            return
        to_buy_capacity = self.position_limit - position
        to_sell_capacity = self.position_limit + position
        if rsi_value > self.overbought_threshold and to_sell_capacity > 0:
            trade_qty = to_sell_capacity
            price = best_bid - 1
            self.orders.append(Order(self.symbol, price, -trade_qty))
        elif rsi_value < self.oversold_threshold and to_buy_capacity > 0:
            trade_qty = to_buy_capacity
            price = best_ask + 1
            self.orders.append(Order(self.symbol, price, trade_qty))

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def save(self) -> dict:
        return {"mid_price_history": list(self.mid_price_history), "avg_gain": self.avg_gain, "avg_loss": self.avg_loss, "rsi_initialized": self.rsi_initialized}

    def load(self, data: dict) -> None:
        loaded_history = data.get("mid_price_history", [])
        self.mid_price_history = deque(loaded_history, maxlen=self.window + 1)
        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        self.rsi_initialized = data.get("rsi_initialized", False)


class OtherProductsTrader:
    def __init__(self) -> None:
        self.logger = Logger()
        self.pos_limits = LIMIT
        self.diff_threshold_b1_b2 = 176.8118
        self.lot_size = 1
        self.previousposition = {p: 0 for p in LIMIT.keys()}
        self.position = {p: 0 for p in LIMIT.keys()}
        self.positionCounter = {p: 0 for p in LIMIT.keys()}
        self.resin_timestamps = []
        self.resin_mid_prices = []
        self.kelp_timestamps = []
        self.kelp_mid_prices = []
        self.ink_timestamps = []
        self.ink_mid_prices = []
        self.strategy = SquidInkRsiStrategy(Product.SQUID_INK, LIMIT[Product.SQUID_INK])

    def update_market_data(self, product, state: TradingState):
        od = state.order_depths[product]
        mid = mid_price(od)
        if product == Product.RAINFOREST_RESIN:
            self.resin_timestamps.append(state.timestamp)
            self.resin_mid_prices.append(mid)
        elif product == Product.KELP:
            self.kelp_timestamps.append(state.timestamp)
            self.kelp_mid_prices.append(mid)
        elif product == Product.SQUID_INK:
            self.ink_timestamps.append(state.timestamp)
            self.ink_mid_prices.append(mid)

    def UpdatePreviousPositionCounter(self, product, state: TradingState) -> None:
        if product not in state.position:
            return
        if state.position[product] == self.previousposition[product]:
            self.positionCounter[product] += 1
        else:
            self.positionCounter[product] = 0

    def PriceAdjustment(self, product, mode, state: TradingState):
        if product not in state.position:
            return 0
        AdjustmentDict = {
            Product.PICNIC_BASKET1: {"high": 3.8116, "mid": 37.2380, "low": -9.0092, "neg": 7.9516},
            Product.PICNIC_BASKET2: {"high": 39.2463, "mid": 26.8949, "low": 5.3674, "neg": -26.1430},
            Product.DJEMBES: {"high": 22.4723, "mid": 16.7552, "low": -4.4401, "neg": -9.0611},
            Product.SQUID_INK: {"high": 26.244, "mid": 3.57077, "low": 1.0171, "neg": -19.168886},
            Product.KELP: {"high": 35.561048, "mid": 34.49889, "low": 9.11303, "neg": 1.2853198},
        }
        holdFactorDict = {
            Product.PICNIC_BASKET1: 38.6961,
            Product.PICNIC_BASKET2: 3.9930,
            Product.DJEMBES: 1.7,
            Product.SQUID_INK: 28.964,
            Product.KELP: 28.964,
        }
        VolumeFraction = VolumeCapability(product, mode, state) / LIMIT[product]
        holdPremium = int(holdFactorDict[product] * self.positionCounter[product])
        high = AdjustmentDict[product]["high"] + holdPremium
        mid = AdjustmentDict[product]["mid"] + holdPremium
        low = AdjustmentDict[product]["low"] + holdPremium
        neg = AdjustmentDict[product]["neg"] + holdPremium
        if product in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
            factor = 1 if mode=="buy" else -1
            if VolumeFraction <= 0.1:
                return int(factor * (high + 3))
            if 0.1 < VolumeFraction <= 0.2:
                return int(factor * high)
            if 0.2 < VolumeFraction < 0.5:
                return int(factor * mid)
            if 0.5 <= VolumeFraction < 1:
                return int(factor * low)
            if VolumeFraction >= 1:
                return int(factor * neg)
        if product in [Product.DJEMBES]:
            factor = 1 if mode=="buy" else -1
            if VolumeFraction <= 0.1:
                return int(factor * high)
            if 0.1 < VolumeFraction < 0.5:
                return int(factor * mid)
            if 0.5 <= VolumeFraction < 1:
                return int(factor * low)
            if VolumeFraction >= 1:
                return int(factor * neg)
        if product in [Product.SQUID_INK, Product.KELP]:
            factor = 1 if mode=="buy" else -1
            if VolumeFraction <= 0.1:
                return int(factor * high)
            if 0.1 < VolumeFraction < 0.5:
                return int(factor * mid)
            if 0.5 <= VolumeFraction < 1:
                return int(factor * low)
            if VolumeFraction >= 1:
                return int(factor * neg)
        return 0

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result = {p: [] for p in LIMIT.keys() if p not in [Product.VOLCANIC_ROCK, Product.VOUCHER_9500, Product.VOUCHER_9750, Product.VOUCHER_10000, Product.VOUCHER_10250, Product.VOUCHER_10500]}
        conversions = 0
        for product in state.position:
            if product in result:
                self.UpdatePreviousPositionCounter(product, state)
                self.update_market_data(product, state)
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_params = COMBINED_PARAMS[Product.RAINFOREST_RESIN]
            resin_order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            resin_fair_value = resin_params["fair_value"]
            orders_take, bo, so = resin_take_orders(resin_order_depth, resin_fair_value, resin_position, LIMIT[Product.RAINFOREST_RESIN])
            orders_clear, bo, so = resin_clear_orders(resin_order_depth, resin_position, resin_fair_value, LIMIT[Product.RAINFOREST_RESIN], bo, so)
            orders_make = resin_make_orders(resin_order_depth, resin_fair_value, resin_position, LIMIT[Product.RAINFOREST_RESIN], bo, so)
            result[Product.RAINFOREST_RESIN] += orders_take + orders_clear + orders_make
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_params = COMBINED_PARAMS[Product.KELP]
            kelp_order_depth = state.order_depths[Product.KELP]
            kelp_fair = kelp_fair_value(kelp_order_depth, kelp_params["default_fair_method"], kelp_params["min_volume_filter"])
            kelp_take, bo, so = kelp_take_orders(kelp_order_depth, kelp_fair, kelp_params, kelp_position)
            kelp_clear, bo, so = kelp_clear_orders(kelp_order_depth, kelp_position, kelp_params, kelp_fair, bo, so)
            kelp_make = kelp_make_orders(kelp_order_depth, kelp_fair, kelp_position, kelp_params, bo, so)
            result[Product.KELP] = kelp_take + kelp_clear + kelp_make
        try:
            if state.traderData:
                loaded_data = json.loads(state.traderData)
                data_for_squid = loaded_data.get(Product.SQUID_INK, {})
                self.strategy.load(data_for_squid)
            else:
                self.strategy.load({})
        except Exception:
            self.strategy.load({})
        self.strategy.run(state)
        if self.strategy.orders:
            result[Product.SQUID_INK] = self.strategy.orders
        squid_data = {Product.SQUID_INK: self.strategy.save()}
        relevant = [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]
        best_bid, best_ask, mid_vals = {}, {}, {}
        for prod in relevant:
            od = state.order_depths.get(prod)
            if od:
                best_bid[prod] = max(od.buy_orders.keys()) if od.buy_orders else None
                best_ask[prod] = min(od.sell_orders.keys()) if od.sell_orders else None
                if best_bid[prod] is not None and best_ask[prod] is not None:
                    mid_vals[prod] = 0.5 * (best_bid[prod] + best_ask[prod])
                elif best_bid[prod] is not None:
                    mid_vals[prod] = best_bid[prod]
                elif best_ask[prod] is not None:
                    mid_vals[prod] = best_ask[prod]
                else:
                    mid_vals[prod] = None
            else:
                best_bid[prod] = None
                best_ask[prod] = None
                mid_vals[prod] = None
        if (mid_vals.get(Product.PICNIC_BASKET1) is not None and mid_vals.get(Product.PICNIC_BASKET2) is not None
            and mid_vals.get(Product.CROISSANTS) is not None and mid_vals.get(Product.JAMS) is not None
            and mid_vals.get(Product.DJEMBES) is not None):
            b1 = mid_vals[Product.PICNIC_BASKET1]
            b2 = mid_vals[Product.PICNIC_BASKET2]
            c = mid_vals[Product.CROISSANTS]
            j = mid_vals[Product.JAMS]
            d = mid_vals[Product.DJEMBES]
            implied_b1 = b2 + 2 * c + j + d
            diff_comp = b1 - implied_b1
            self.logger.print(f"Composition signal: BASKET1={b1:.1f}, Implied_BASKET1={implied_b1:.1f}, diff={diff_comp:.1f}")
            signal = 1 if diff_comp > self.diff_threshold_b1_b2 else (-1 if diff_comp < -self.diff_threshold_b1_b2 else 0)
            self.logger.print(f"Signal determined (for JAMS and DJEMBES): {signal}")
        else:
            self.logger.print("Insufficient market data to compute composition signal.")
            signal = 0
        for prod in [Product.JAMS, Product.DJEMBES]:
            od = state.order_depths.get(prod)
            if not od:
                self.logger.print(f"No order depth for {prod}, skipping.")
                continue
            current_pos = state.position.get(prod, 0)
            pos_limit = self.pos_limits[prod]
            if signal == 1:
                capacity = pos_limit - current_pos
                if capacity > 0 and od.sell_orders:
                    price = min(od.sell_orders.keys()) + self.PriceAdjustment(prod, "buy", state)
                    result[prod].append(Order(prod, price, capacity))
                    self.logger.print(f"Placing BUY order for {prod}: {capacity}x{price}")
            elif signal == -1:
                capacity = pos_limit + current_pos
                if capacity > 0 and od.buy_orders:
                    price = max(od.buy_orders.keys()) - self.PriceAdjustment(prod, "sell", state)
                    result[prod].append(Order(prod, price, -capacity))
                    self.logger.print(f"Placing SELL order for {prod}: {capacity}x{price}")
            else:
                self.logger.print(f"No clear signal for {prod} – no order placed.")
        trader_data_encoded = json.dumps(squid_data, separators=(",", ":"))
        return result, conversions, trader_data_encoded


# =============================================================================
# VolcanicTrader – handles VOLCANIC_ROCK and its vouchers only
# =============================================================================
class VolcanicTrader:
    def __init__(self) -> None:
        self.position_limits = {
            Product.VOLCANIC_ROCK: LIMIT[Product.VOLCANIC_ROCK],
            Product.VOUCHER_9500: LIMIT[Product.VOUCHER_9500],
            Product.VOUCHER_9750: LIMIT[Product.VOUCHER_9750],
            Product.VOUCHER_10000: LIMIT[Product.VOUCHER_10000],
            Product.VOUCHER_10250: LIMIT[Product.VOUCHER_10250],
            Product.VOUCHER_10500: LIMIT[Product.VOUCHER_10500],
        }
        self.strategies = {
            Product.VOLCANIC_ROCK: RsiStrategy(Product.VOLCANIC_ROCK, self.position_limits[Product.VOLCANIC_ROCK]),
            "VOLATILITY_SMILE": VolatilitySmileStrategy()
        }

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        all_orders: List[Order] = []
        conversions = 0
        trader_data_for_next_round = {}
        for key, strat in self.strategies.items():
            if isinstance(strat, VolatilitySmileStrategy):
                orders = strat.run(state)
                for sym, ords in orders.items():
                    all_orders.extend(ords)
                trader_data_for_next_round[key] = strat.save_state()
            else:
                if hasattr(strat, 'act'):
                    strat.act(state)
                    all_orders.extend(strat.orders)
                    trader_data_for_next_round[key] = strat.save()
        final_result: Dict[Symbol, List[Order]] = defaultdict(list)
        for order in all_orders:
            if order.quantity != 0:
                final_result[order.symbol].append(order)
        trader_data_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"), cls=ProsperityEncoder)
        logger.flush(state, dict(final_result), conversions, trader_data_encoded)
        return dict(final_result), conversions, trader_data_encoded


# =============================================================================
# CombinedTrader – merges volcanic and non-volcanic subsystems
# =============================================================================
class Trader:
    def __init__(self) -> None:
        self.volcanic_trader = VolcanicTrader()
        self.other_trader = OtherProductsTrader()

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        orders_volcanic, conv_v, data_v = self.volcanic_trader.run(state)
        orders_other, conv_o, data_o = self.other_trader.run(state)
        combined_orders: Dict[Symbol, List[Order]] = defaultdict(list)
        for sym, ords in orders_volcanic.items():
            combined_orders[sym].extend(ords)
        for sym, ords in orders_other.items():
            combined_orders[sym].extend(ords)
        conversions = conv_v + conv_o
        combined_data = json.dumps({"volcanic": json.loads(data_v), "other": json.loads(data_o)})
        return dict(combined_orders), conversions, combined_data


# =============================================================================
# The main Trader entry point
# =============================================================================
    def run(state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        TraderInstance = Trader()
        return TraderInstance.run(state)
