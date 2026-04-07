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
        # self.max_log_length = 3750
        self.max_log_length = 1000

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
            # UNCOMMENT TO ENABLE MARKET TRADES
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
MAGNIFICENT_MACARONS = 'MAGNIFICENT_MACARONS'

ETARIFF='export_tariff'
ITARIFF='import_tariff'
SUGAR='sugar_price'
TRANSPORT='transport_fee'
SUNLIGHT='sunlight_index'
Kz=1
Sz=1
Jz=1
Dz=1
Cz=1
KTrigger=5*Kz
STrigger=26*Sz
JTrigger=9*Jz
DTrigger=9*Dz
CTrigger=19*Cz

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
    MAGNIFICENT_MACARONS :{'limit':75,'conversion':10} 
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
    KELP: {
        "take_width": 1,
        "position_limit": 50,
        "min_volume_filter": 20,
        "spread_edge": 1,
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
        "rsi_window": 85,
        "rsi_overbought": 52,
        "rsi_oversold": 42,
        "price_offset": 0,
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
    # VOLCANIC_ROCK_VOUCHER_10500: 10500,  # CHECK IF YOU WANT TO UNCOMMENT
}

POSITION_LIMITS = {
    VOLCANIC_ROCK: 400,
    VOLCANIC_ROCK_VOUCHER_9500:  200,
    VOLCANIC_ROCK_VOUCHER_9750:  200,
    VOLCANIC_ROCK_VOUCHER_10000: 200,
    VOLCANIC_ROCK_VOUCHER_10250: 200,
    VOLCANIC_ROCK_VOUCHER_10500: 200,
}

B1B2_THEORETICAL_COMPONENTS = {
    CROISSANTS: 2,
    JAMS: 1,
    DJEMBES: 1
}

DAYS_LEFT = 3  # constant used to compute days‑to‑expiry
VOUCHERS: List[str] = [
    "VOLCANIC_ROCK_VOUCHER_9500",
    "VOLCANIC_ROCK_VOUCHER_9750",
    "VOLCANIC_ROCK_VOUCHER_10000",
    # "VOLCANIC_ROCK_VOUCHER_10500",   # <-- trading disabled
]

STRIKES: Dict[str, int] = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    # "VOLCANIC_ROCK_VOUCHER_10500": 10500,  # not used anymore
}

# Coefficients of the quadratic IV smile (base, linear, squared) per voucher
COEFFICIENTS: Dict[str, Tuple[float, float, float]] = {
    "VOLCANIC_ROCK_VOUCHER_9500": (0.264416, 0.010031, 0.147604),
    "VOLCANIC_ROCK_VOUCHER_9750": (0.264416, 0.010031, 0.147604),
    "VOLCANIC_ROCK_VOUCHER_10000": (0.14786181, 0.00099561, 0.23544086),
    # "VOLCANIC_ROCK_VOUCHER_10500": (0.264416, 0.010031, 0.147604),  # unused
}

# Absolute diff thresholds that trigger a trade
THRESHOLDS: Dict[str, float] = {
    "VOLCANIC_ROCK_VOUCHER_9500": 0.0005,
    "VOLCANIC_ROCK_VOUCHER_9750": 0.0055,
    "VOLCANIC_ROCK_VOUCHER_10000": 0.0035,
    # "VOLCANIC_ROCK_VOUCHER_10500": 0.001,  # unused
}

BID='bid'
ASK='ask'
BUY='buy'
SELL='sell'

POSITION_LIMIT = 200  # same for every voucher in original code
TRADE_SIZE = 10       # fixed clip size per decision

# --- Resin Market Making ---
def resin_take_orders(
    order_depth: OrderDepth, fair_value: float, position: int, position_limit: int, product
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
                orders.append(Order(product, best_ask, quantity))
                buy_order_volume += quantity
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        if best_bid > fair_value:
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order(product, best_bid, -quantity))
                sell_order_volume += quantity
    return orders, buy_order_volume, sell_order_volume

def resin_clear_orders(
    order_depth: OrderDepth,
    position: int,
    fair_value: float,
    position_limit: int,
    buy_order_volume: int,
    sell_order_volume: int,
    product
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
            orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)
    return orders, buy_order_volume, sell_order_volume

def resin_make_orders(
    order_depth: OrderDepth,
    fair_value: float,
    position: int,
    position_limit: int,
    buy_order_volume: int,
    sell_order_volume: int,
    product
) -> List[Order]:
    orders = []
    aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
    bbbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
    baaf = min(aaf) if aaf else fair_value + 2
    bbbf_val = max(bbbf) if bbbf else fair_value - 2
    buy_quantity = position_limit - (position + buy_order_volume)
    if buy_quantity > 0:
        orders.append(Order(product, bbbf_val + 1, buy_quantity))
    sell_quantity = position_limit + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(product, baaf - 1, -sell_quantity))
    return orders
# -----------------------------------------------------


# --- For Volcanic Rocks ---
# ---------- Black‑Scholes helpers ---------- #
def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x: float) -> float:
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)


def calculate_option_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def calculate_implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    initial_vol: float = 0.3,
    max_iterations: int = 50,
    precision: float = 1e-4,
) -> float:
    if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0
    intrinsic = max(0, S - K)
    if option_price <= intrinsic:
        return 0.0

    vol = initial_vol
    for _ in range(max_iterations):
        price = calculate_option_price(S, K, T, r, vol)
        diff = option_price - price
        if abs(diff) < precision:
            return vol
        d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
        vega = S * math.sqrt(T) * norm_pdf(d1)
        if vega == 0:
            return vol
        vol = max(0.0001, min(5.0, vol + diff / vega))
    return vol


# ---------- Base Strategy ---------- #
class Strategy:
    def __init__(self, symbol: str, position_limit: int) -> None:
        self.symbol = symbol
        self.position_limit = position_limit
        self.orders: List[Order] = []

    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    # helpers
    def _get_mid_price(self, symbol: str, state: TradingState) -> Optional[float]:
        od = state.order_depths.get(symbol)
        if not od:
            return None
        bids = od.buy_orders.keys()
        asks = od.sell_orders.keys()
        if bids and asks:
            return (max(bids) + min(asks)) / 2.0
        return max(bids) if bids else (min(asks) if asks else None)

    # persistence (no‑op unless overridden)
    def save(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass


# ---------- RSI strategy for VOLCANIC_ROCK ---------- #
class RsiStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int) -> None:
        super().__init__(symbol, position_limit)
        p = PARAMS[symbol]
        self.window = p["rsi_window"]
        self.overbought_threshold = p["rsi_overbought"]
        self.oversold_threshold = p["rsi_oversold"]
        self.price_offset = p["price_offset"]
        self.mid_price_history: Deque[float] = deque(maxlen=self.window + 1)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.rsi_initialized = False

    def _calculate_rsi(self, price: float) -> Optional[float]:
        self.mid_price_history.append(price)
        if len(self.mid_price_history) < self.window + 1:
            return None
        changes = np.diff(self.mid_price_history)
        gains = np.maximum(changes, 0)
        losses = -np.minimum(changes, 0)
        if not self.rsi_initialized:
            self.avg_gain = float(np.mean(gains[-self.window:]))
            self.avg_loss = float(np.mean(losses[-self.window:]))
            self.rsi_initialized = True
        else:
            self.avg_gain = (self.avg_gain * (self.window - 1) + gains[-1]) / self.window
            self.avg_loss = (self.avg_loss * (self.window - 1) + losses[-1]) / self.window
        if self.avg_loss < 1e-9:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100 - (100 / (1 + rs))

    def act(self, state: TradingState) -> None:
        od = state.order_depths.get(self.symbol)
        if not od:
            return
        mid = self._get_mid_price(self.symbol, state)
        if mid is None:
            return
        rsi = self._calculate_rsi(mid)
        if rsi is None:
            return

        position = state.position.get(self.symbol, 0)
        buy_cap = self.position_limit - position
        sell_cap = self.position_limit + position
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if rsi > self.overbought_threshold and sell_cap > 0 and best_bid is not None:
            price = best_bid - self.price_offset
            self.orders.append(Order(self.symbol, int(price), -sell_cap))
        elif rsi < self.oversold_threshold and buy_cap > 0 and best_ask is not None:
            price = best_ask + self.price_offset
            self.orders.append(Order(self.symbol, int(price), buy_cap))

    # ----- persistence ----- #
    def save(self) -> dict:
        return {
            "mid_price_history": list(self.mid_price_history),
            "avg_gain": self.avg_gain,
            "avg_loss": self.avg_loss,
            "rsi_initialized": self.rsi_initialized,
        }

    def load(self, data: dict) -> None:
        hist = data.get("mid_price_history", [])
        self.mid_price_history = deque(hist[-(self.window + 1) :], maxlen=self.window + 1)
        self.avg_gain = data.get("avg_gain")
        self.avg_loss = data.get("avg_loss")
        self.rsi_initialized = data.get("rsi_initialized", False)


# ---------- Volatility‑Smile strategy for vouchers ---------- #
class VolatilitySmileStrategy:
    def __init__(self) -> None:
        self.position_limits = {
            VOLCANIC_ROCK: 400,
            **{v: 200 for v in STRIKES},
        }
        self.voucher_symbols = list(STRIKES.keys())
        self.short_ewma_span = 10
        self.long_ewma_span = 51
        self.rolling_window = 30
        self.zscore_upper_threshold = 0.5
        self.zscore_lower_threshold = -2.8
        self.trade_size = 11

        self.base_iv_history: Deque[float] = deque(maxlen=200)
        self.short_ewma_base_iv: Optional[float] = None
        self.long_ewma_first: Optional[float] = None
        self.long_ewma_base_iv: Optional[float] = None
        self.ewma_diff_history: Deque[float] = deque(maxlen=200)
        self.zscore_history: Deque[float] = deque(maxlen=100)

        self.day = 4  # simulation day
        self.last_timestamp: Optional[int] = None

    # ----- helpers ----- #
    def update_time_to_expiry(self, ts: int) -> float:
        base = 8 - self.day
        iter_no = (ts % 1_000_000) // 100
        adj = iter_no / 10_000
        return max(0.0001, (base - adj) / 365)

    def get_mid(self, symbol: str, state: TradingState) -> Optional[float]:
        od = state.order_depths.get(symbol)
        if not od:
            return None
        bids = od.buy_orders.keys()
        asks = od.sell_orders.keys()
        if bids and asks:
            return (max(bids) + min(asks)) / 2.0
        return max(bids) if bids else (min(asks) if asks else None)

    def place(
        self, orders: Dict[str, List[Order]], sym: str, price: float, qty: int
    ) -> None:
        if qty == 0:
            return
        orders.setdefault(sym, []).append(Order(sym, int(round(price)), qty))

    def ewma(self, val: float, prev: Optional[float], span: int) -> float:
        return val if prev is None else (2 / (span + 1)) * val + (1 - 2 / (span + 1)) * prev

    # ----- core run ----- #
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        orders: Dict[str, List[Order]] = {}
        self.last_timestamp = state.timestamp

        rock_mid = self.get_mid(VOLCANIC_ROCK, state)
        if rock_mid is None:
            return orders
        tte = self.update_time_to_expiry(state.timestamp)

        # flatten deep OTM
        for v in self.voucher_symbols:
            if rock_mid - STRIKES[v] <= -250:
                pos = state.position.get(v, 0)
                if pos != 0:
                    od = state.order_depths.get(v)
                    if od:
                        if pos > 0 and od.buy_orders:
                            self.place(orders, v, max(od.buy_orders), -pos)
                        elif pos < 0 and od.sell_orders:
                            self.place(orders, v, min(od.sell_orders), -pos)

        # build smile
        moneyness, ivs = [], []
        for v in self.voucher_symbols:
            mid = self.get_mid(v, state)
            if mid is None or tte <= 0:
                continue
            try:
                m = math.log(STRIKES[v] / rock_mid) / math.sqrt(tte)
                iv = calculate_implied_volatility(mid, rock_mid, STRIKES[v], tte)
                if iv > 0:
                    moneyness.append(m)
                    ivs.append(iv)
            except Exception:
                continue

        if len(moneyness) >= 3:
            a, b, c = np.polyfit(moneyness, ivs, 2)  # noqa: F841
            base_iv = c
            self.base_iv_history.append(base_iv)
            self.short_ewma_base_iv = self.ewma(
                base_iv, self.short_ewma_base_iv, self.short_ewma_span
            )
            self.long_ewma_first = self.ewma(
                base_iv, self.long_ewma_first, self.long_ewma_span
            )
            self.long_ewma_base_iv = self.ewma(
                self.long_ewma_first, self.long_ewma_base_iv, self.long_ewma_span
            )

            if (
                len(self.base_iv_history) >= self.rolling_window
                and self.short_ewma_base_iv is not None
                and self.long_ewma_base_iv is not None
            ):
                diff = self.short_ewma_base_iv - self.long_ewma_base_iv
                self.ewma_diff_history.append(diff)
                std = (
                    np.std(list(self.ewma_diff_history)[-self.rolling_window :])
                    if len(self.ewma_diff_history) >= self.rolling_window
                    else np.std(self.ewma_diff_history)
                )
                z = diff / std if std > 1e-9 else 0
                self.zscore_history.append(z)

                for v in self.voucher_symbols:
                    if rock_mid - STRIKES[v] <= -250:
                        continue
                    od = state.order_depths.get(v)
                    if not od:
                        continue
                    size = self.trade_size
                    pos = state.position.get(v, 0)
                    if z > self.zscore_upper_threshold and od.buy_orders and pos + size <= 200:
                        self.place(orders, v, max(od.buy_orders), -size)
                    elif (
                        z < self.zscore_lower_threshold and od.sell_orders and pos - size >= -200
                    ):
                        self.place(orders, v, min(od.sell_orders), size)

        return orders

    # ----- persistence ----- #
    def save_state(self) -> dict:
        return {
            "day": self.day,
            "base_iv_history": list(self.base_iv_history),
            "short_ewma_base_iv": self.short_ewma_base_iv,
            "long_ewma_first": self.long_ewma_first,
            "long_ewma_base_iv": self.long_ewma_base_iv,
            "ewma_diff_history": list(self.ewma_diff_history),
            "zscore_history": list(self.zscore_history),
            "last_timestamp": self.last_timestamp,
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
        self.last_timestamp = data.get("last_timestamp")
# ----------------------------------------------------------



# --- Black‑Scholes helper functions (bisection IV) ---

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes_call(S: float, K: float, T_days: float, r: float, sigma: float) -> float:
    T = T_days / 365.0
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def implied_vol_call(market_price: float, S: float, K: float, T_days: float, r: float = 0.0,
                      tol: float = 1e-8, max_iter: int = 100) -> float:
    low, high = 0.01, 1.0  # reasonable bounds
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = black_scholes_call(S, K, T_days, r, mid)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            high = mid
        else:
            low = mid
    return mid  # best estimate after iterations

# --- Strategy class ---
class VoucherOnlyTrader:
    """Algorithm that trades the volcanic‑rock vouchers (9500/9750/10000)."""

    def __init__(self):
        # short diff history per voucher to de‑mean like original
        self.diff_history: Dict[str, deque] = {v: deque(maxlen=20) for v in VOUCHERS}

    # ----- utility helpers -----
    @staticmethod
    def _best_bid(order_depth: OrderDepth):
        return max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

    @staticmethod
    def _best_ask(order_depth: OrderDepth):
        return min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

    @staticmethod
    def _mid_price(order_depth: OrderDepth):
        bid = VoucherOnlyTrader._best_bid(order_depth)
        ask = VoucherOnlyTrader._best_ask(order_depth)
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return bid if bid is not None else ask

    # ----- core logic for one voucher -----
    def _trade_one(self, voucher: str, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        voucher_depth = state.order_depths.get(voucher)
        rock_depth = state.order_depths.get("VOLCANIC_ROCK")
        if not voucher_depth or not rock_depth:
            return orders  # need both books for pricing

        mid_voucher = self._mid_price(voucher_depth)
        mid_rock = self._mid_price(rock_depth)
        if mid_voucher is None or mid_rock is None:
            return orders

        # calculate DTE (days to expiry)
        dte = max(1e-4, DAYS_LEFT - state.timestamp / 1_000_000)

        # implied vol from market price
        iv_market = implied_vol_call(mid_voucher, mid_rock, STRIKES[voucher], dte, 0.0)

        # theoretical IV from quadratic smile
        base, linear, quad = COEFFICIENTS[voucher]
        m_t = math.log(STRIKES[voucher] / mid_rock) / math.sqrt(dte / 365)
        iv_fair = base + linear * m_t + quad * (m_t ** 2)

        diff = iv_market - iv_fair
        hist = self.diff_history[voucher]
        hist.append(diff)
        adj_diff = diff - (sum(hist) / len(hist)) if len(hist) == hist.maxlen else diff

        threshold = THRESHOLDS[voucher]
        position = state.position.get(voucher, 0)

        # decide direction
        qty = 0
        if adj_diff > threshold and position > -POSITION_LIMIT:      # implied vol rich -> SELL
            qty = -min(TRADE_SIZE, POSITION_LIMIT + position)
            price = self._best_bid(voucher_depth)
        elif adj_diff < -threshold and position < POSITION_LIMIT:    # implied vol cheap -> BUY
            qty = min(TRADE_SIZE, POSITION_LIMIT - position)
            price = self._best_ask(voucher_depth)
        else:
            return orders  # no trade

        if price is not None and qty != 0:
            orders.append(Order(voucher, int(price), qty))
        return orders

    # ----- main entry point -----
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        all_orders: Dict[Symbol, List[Order]] = {}
        for v in VOUCHERS:
            orders = self._trade_one(v, state)
            if orders:
                all_orders[v] = orders
        next_state = json.dumps({k: list(self.diff_history[k]) for k in VOUCHERS})
        return all_orders, 0, next_state

    # ----- state persistence -----
    def load_state(self, data: str):
        try:
            d = json.loads(data)
            for v in VOUCHERS:
                if v in d and isinstance(d[v], list):
                    self.diff_history[v] = deque(d[v], maxlen=self.diff_history[v].maxlen)
        except Exception:
            pass

    def save_state(self) -> str:
        return json.dumps({k: list(self.diff_history[k]) for k in VOUCHERS})

# -------- General Functions --------
def kalman_filter_1d(z, Q=1e-5, R=4):
    """
    1D Kalman filter using numpy.
    z: observed time series (e.g., mid-prices)
    Q: process variance (model noise)
    R: measurement variance (observation noise)
    Returns: filtered estimates
    """
    n = len(z)
    x_hat = np.zeros(n)      # filtered state estimate
    P = np.zeros(n)          # error covariance
    x_hat[0] = z[0]          # initial state
    P[0] = 1.0               # initial covariance

    for k in range(1, n):
        # Predict
        x_hat_minus = x_hat[k - 1]
        P_minus = P[k - 1] + Q

        # Update
        K = P_minus / (P_minus + R)
        x_hat[k] = x_hat_minus + K * (z[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus

    return x_hat

def project_next(smoothed, window=10, correlation_threshold=0.8):
    """
    Project the next point using linear or quadratic fit based on Pearson correlation.
    
    Args:
        smoothed (list[float]): The smoothed time series data.
        window (int): The number of recent points to use for the fit.
        order (int): The default order of the polynomial fit (1 for linear, 2 for quadratic).
        correlation_threshold (float): The minimum Pearson correlation for a linear fit to be acceptable.
    
    Returns:
        float: The projected next value.
    """
    # Ensure there are enough elements in `smoothed`
    if len(smoothed) < window:
        window = len(smoothed)  # Use all available elements if fewer than `window`

    recent = smoothed[-window:]
    x = np.arange(len(recent))  # Match the length of `recent`

    # Validate the `recent` array
    if np.any(np.isnan(recent)) or np.any(np.isinf(recent)):
        return int(smoothed[-1])  # Return the last valid value as a fallback

    try:
        # Attempt a linear fit (order=1)
        a, b = np.polyfit(x, recent, 1)  # y = ax + b
        linear_fit = a * x + b
        # Calculate Pearson correlation for the linear fit
        correlation = np.corrcoef(recent, linear_fit)[0, 1]

        if correlation >= correlation_threshold:
            # Linear fit is good, project the next value
            next_value = a * len(recent) + b
        else:
            # Linear fit is poor, attempt a quadratic fit (order=2)
            a, b, c = np.polyfit(x, recent, 2)  # y = ax^2 + bx + c
            next_value = a * (len(recent) ** 2) + b * len(recent) + c
    except np.linalg.LinAlgError as e:
        return int(smoothed[-1])  # Return the last valid value as a fallback

    return int(next_value)

## GENERAL FUNCTIONS #######################
def sortDict(dictionary:dict):
    return {key: dictionary[key] for key in sorted(dictionary)}

def VolumeCapability(product, mode, state: TradingState):
    if product not in set(state.position.keys()):
        position = 0
    else:
        position = state.position[product]

    if mode == "buy":
        limit = LIMIT[product]['limit'] if isinstance(LIMIT[product], dict) else LIMIT[product]
        return limit - position
    elif mode == "sell":
        limit = LIMIT[product]['limit'] if isinstance(LIMIT[product], dict) else LIMIT[product]
        return position + limit

def mid_price(order_depth: OrderDepth) -> float:
    if order_depth.sell_orders and order_depth.buy_orders:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2
    elif order_depth.sell_orders:
        return min(order_depth.sell_orders.keys())
    elif order_depth.buy_orders:
        return max(order_depth.buy_orders.keys())
    return 0.0

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
    if not state.order_depths[product].sell_orders:  # FREAKY
        return 0
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
    if not state.order_depths[product].buy_orders:
        return 0  # FREAKY
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
# ---------------------------------------------

# ---------------------------------------------------------------------------
NEUTRAL = [None, "Neutral"]
NEUTRAL_ROW = [NEUTRAL] * 15

TREND_MATRIX = [
    NEUTRAL_ROW,
    [
        NEUTRAL, NEUTRAL, NEUTRAL, [-0.000932618, "Downtrend"], [0.00286761, "Downtrend"],
        [-0.00286832, "Downtrend"], [0.000307658, "Downtrend"], [0.00231439, "Downtrend"],
        [0.000630606, "Downtrend"], [0.00101488, "Downtrend"], NEUTRAL, [0.000253872, "Downtrend"],
        NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [-0.000773395, "Downtrend"],
        [-0.00364109, "Downtrend"], [0.00413116, "Downtrend"], [-0.000256013, "Downtrend"],
        [-0.00214848, "Downtrend"], [0.00939212, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [0.000246348, "Downtrend"],
        [-0.00681597, "Downtrend"], [0.00211427, "Downtrend"], [0.00216763, "Downtrend"],
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [-0.00105904, "Downtrend"], [0.00025355, "Downtrend"],
        [0.00151439, "Downtrend"], [-0.000944908, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL,
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [-0.000371379, "Downtrend"],
        [0.000346696, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [-0.00151553, "Downtrend"], [0.00123701, "Downtrend"],
        [-2.98354e-05, "Downtrend"], [-0.000127546, "Downtrend"], [-8.99841e-05, "Downtrend"],
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [-0.00049838, "Downtrend"], [0.00064273, "Downtrend"],
        [-0.000898603, "Downtrend"], [0.000210548, "Downtrend"], [0.000384589, "Downtrend"],
        [-0.00106795, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, [0.0054567, "Downtrend"], [0.000609756, "Downtrend"],
        [0.000438983, "Downtrend"], [-1.48817e-05, "Downtrend"], [0.000305021, "Downtrend"],
        [-0.00245242, "Downtrend"], NEUTRAL, [0.00316372, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, [0.00204708, "Downtrend"], [-0.00107325, "Downtrend"], [0.000310385, "Downtrend"],
        [0.000799627, "Downtrend"], [-0.000408405, "Downtrend"], [0.000542229, "Downtrend"],
        [0.000493705, "Downtrend"], [-0.000313869, "Downtrend"], [-0.000811249, "Downtrend"],
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, [0.001531, "Downtrend"], [0.0010924, "Downtrend"], [0.000339091, "Downtrend"],
        [0.000187681, "Downtrend"], [0.00080719, "Downtrend"], [-0.000206106, "Downtrend"],
        [-0.000662753, "Downtrend"], [-0.0028472, "Downtrend"], [-0.006189, "Downtrend"],
        [-0.000239866, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, [0.00154123, "Downtrend"], [7.40966e-07, "Downtrend"], [1.22694e-05, "Downtrend"],
        [-0.000383621, "Downtrend"], [0.000223367, "Downtrend"], [0.000498693, "Downtrend"],
        [-0.000536222, "Downtrend"], [-0.000724988, "Downtrend"], [0.000599611, "Downtrend"],
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, [0.000936988, "Downtrend"], [0.000117, "Downtrend"], [0.00018827, "Downtrend"],
        [0.00034544, "Downtrend"], [0.00062788, "Downtrend"], [0.00021003, "Downtrend"],
        [0.000201746, "Uptrend"], [0.000736537, "Downtrend"], [-0.000237816, "Downtrend"],
        [0.00111013, "Downtrend"], [0, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, [0, "Downtrend"], [8.44342e-05, "Downtrend"], [0.000851939, "Downtrend"],
        [0.000936586, "Uptrend"], [0.000135677, "Downtrend"], [3.992e-05, "Downtrend"],
        [0.000126552, "Uptrend"], [0.000307895, "Downtrend"], [-0.000370192, "Downtrend"],
        [-0.00108618, "Downtrend"], [-0.000549621, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, [0.000833216, "Downtrend"], [0.000520585, "Downtrend"], [0.000803972, "Downtrend"],
        [0.000641393, "Downtrend"], [0.000144998, "Uptrend"], [0.000320204, "Uptrend"],
        [0.000251086, "Uptrend"], [8.27495e-05, "Uptrend"], [-0.000141575, "Downtrend"],
        [-0.000218485, "Downtrend"], [-0.000347464, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, [-0.000254347, "Downtrend"], [0.000219446, "Downtrend"], [0.000418931, "Downtrend"],
        [0.000444139, "Uptrend"], [0.000123094, "Uptrend"], [0.000309518, "Downtrend"],
        [0.000339389, "Uptrend"], [0.000173049, "Uptrend"], [-0.000177855, "Downtrend"],
        [-0.000537388, "Downtrend"], [-0.000514531, "Downtrend"], [-0.00212114, "Downtrend"],
        NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, [0.000405762, "Downtrend"], [0.000487896, "Downtrend"], [0.000414984, "Uptrend"],
        [0.000418312, "Uptrend"], [0.000189863, "Downtrend"], [0.000270396, "Downtrend"],
        [0.000129039, "Downtrend"], [0.000217029, "Uptrend"], [-0.000436228, "Uptrend"],
        [0.000258372, "Downtrend"], [-1.1972e-06, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, [-0.000237514, "Downtrend"], [0.000256085, "Uptrend"], [0.000170757, "Uptrend"],
        [0.000269943, "Uptrend"], [0.000195313, "Downtrend"], [0.00037907, "Downtrend"],
        [0.000110317, "Downtrend"], [-1.98557e-05, "Uptrend"], [3.04163e-05, "Uptrend"],
        [0.000404166, "Downtrend"], [-0.000256094, "Downtrend"], NEUTRAL, [0.00170094, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, [0, "Downtrend"], [7.2e-05, "Downtrend"], [0.000358886, "Uptrend"],
        [0.000298484, "Uptrend"], [0.000119994, "Downtrend"], [0.00019274, "Downtrend"],
        [2.86651e-05, "Downtrend"], [4.94614e-05, "Downtrend"], [-0.000103853, "Uptrend"],
        [2.69493e-05, "Uptrend"], [-5.0743e-06, "Downtrend"], NEUTRAL, [-0.00116194, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, [8.52297e-05, "Downtrend"], [0.000593037, "Downtrend"], [0.00032515, "Downtrend"],
        [0.000255625, "Downtrend"], [0.000179259, "Downtrend"], [0.000218987, "Downtrend"],
        [2.71452e-05, "Uptrend"], [2.35453e-06, "Downtrend"], [-0.000173662, "Uptrend"],
        [-3.68312e-05, "Uptrend"], [-0.000658588, "Downtrend"], [-0.000267451, "Downtrend"],
        [0.000987167, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, [-0.000128436, "Downtrend"], [-7.92472e-05, "Downtrend"], [6.78453e-05, "Uptrend"],
        [0.000171872, "Uptrend"], [3.52364e-05, "Downtrend"], [5.35117e-05, "Downtrend"],
        [-5.6795e-06, "Uptrend"], [-7.39083e-05, "Downtrend"], [-0.000209475, "Downtrend"],
        [-0.0001944, "Uptrend"], [-0.000245462, "Uptrend"], [-0.00049078, "Downtrend"],
        [0.000250452, "Downtrend"], NEUTRAL,
    ],
    NEUTRAL_ROW,
    NEUTRAL_ROW,
    [
        NEUTRAL, [-0.00048071, "Downtrend"], [-0.000765111, "Downtrend"], [0.00160138, "Downtrend"],
        [0.000158684, "Uptrend"], [0.00012919, "Downtrend"], [7.43274e-05, "Downtrend"],
        [-8.52621e-05, "Uptrend"], [-6.69961e-05, "Downtrend"], [-0.000191414, "Downtrend"],
        [-0.000183853, "Uptrend"], [-0.000303255, "Uptrend"], [-6.73477e-05, "Downtrend"],
        [-0.000361928, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, [0.000650208, "Downtrend"], [0.00116288, "Downtrend"], [-0.000103262, "Uptrend"],
        [-1.07901e-05, "Uptrend"], [2.01524e-05, "Downtrend"], [-8.56006e-05, "Downtrend"],
        [-0.000112532, "Downtrend"], [-0.000368102, "Downtrend"], [-0.000254923, "Downtrend"],
        [-0.000154989, "Uptrend"], [-0.000487287, "Downtrend"], [-0.000259866, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, [0.000502828, "Downtrend"], [0.000701761, "Downtrend"], [2.35097e-05, "Downtrend"],
        [0.000189085, "Uptrend"], [2.64355e-05, "Downtrend"], [-0.000104333, "Downtrend"],
        [-0.000179226, "Downtrend"], [-0.000253123, "Downtrend"], [-0.000292559, "Downtrend"],
        [-0.000303344, "Uptrend"], [-0.000654007, "Downtrend"], [-0.00074869, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, [-2.07428e-05, "Downtrend"], [0.000154398, "Downtrend"],
        [-1.39036e-05, "Uptrend"], [-7.11308e-05, "Uptrend"], [-0.000130136, "Downtrend"],
        [-0.00028713, "Uptrend"], [-0.000442084, "Downtrend"], [-0.000347562, "Uptrend"],
        [-0.000321924, "Uptrend"], [-0.000374148, "Downtrend"], [-0.000345818, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, [-0.000786576, "Downtrend"], [0.000218405, "Downtrend"],
        [9.99765e-06, "Downtrend"], [-7.0041e-05, "Uptrend"], [-0.000156338, "Uptrend"],
        [-0.00015327, "Downtrend"], [-0.000350301, "Uptrend"], [-0.000468537, "Uptrend"],
        [-0.000671593, "Uptrend"], [-0.000593592, "Downtrend"], [-0.00110844, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, [0.0023175, "Downtrend"], NEUTRAL, NEUTRAL, [1.0665e-06, "Downtrend"],
        [-0.000110806, "Downtrend"], [-3.59815e-05, "Uptrend"], [-0.000119161, "Uptrend"],
        [-7.81739e-05, "Uptrend"], [-0.000325136, "Uptrend"], [-0.000285624, "Uptrend"],
        [-0.000497746, "Downtrend"], [-0.000899304, "Downtrend"], [-0.00127453, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [0.00384153, "Downtrend"], [0.0011557, "Downtrend"],
        [5.83122e-05, "Downtrend"], [-5.82568e-05, "Uptrend"], [-0.000284314, "Downtrend"],
        [-0.000122249, "Uptrend"], [-0.000590547, "Downtrend"], [4.76666e-05, "Downtrend"],
        [-0.000335026, "Downtrend"], [-0.00151248, "Downtrend"], NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, [-0.000341775, "Downtrend"], [-0.000381534, "Downtrend"],
        [0.000965569, "Downtrend"], [3.58653e-05, "Downtrend"], [-0.000182158, "Uptrend"],
        [-0.000271138, "Uptrend"], [-0.000543722, "Uptrend"], [0.000363693, "Downtrend"],
        [-0.000507462, "Downtrend"], [-0.00352379, "Downtrend"], NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, [0.000809581, "Downtrend"], [-0.00025641, "Downtrend"],
        [6.72894e-05, "Downtrend"], [0.00012609, "Downtrend"], [0.000194114, "Downtrend"],
        [-0.000135022, "Downtrend"], [-0.000410526, "Downtrend"], [-0.000511875, "Downtrend"],
        [-0.000179136, "Downtrend"], [-0.000462428, "Downtrend"], NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, [-0.000488759, "Downtrend"], [0.00023245, "Downtrend"],
        [-6.72876e-05, "Downtrend"], [0.000272331, "Downtrend"], [0.000164826, "Downtrend"],
        [0.00022396, "Downtrend"], [-0.000955015, "Downtrend"], [-0.000644157, "Downtrend"],
        [-0.000982942, "Downtrend"], [-0.000972053, "Downtrend"], NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, [-0.00140482, "Downtrend"], NEUTRAL, [-0.00055609, "Downtrend"], [-0.00357228, "Downtrend"],
        [-0.000128988, "Downtrend"], [-0.00100045, "Downtrend"], [0.000214974, "Downtrend"],
        [-0.000652395, "Downtrend"], [-0.000369478, "Downtrend"], [-0.000393777, "Downtrend"],
        [0.000336804, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, [0.0014595, "Downtrend"], NEUTRAL, [-0.000255428, "Downtrend"],
        [-0.00142045, "Downtrend"], [-0.000670484, "Downtrend"], [0.000311752, "Downtrend"],
        [-0.00109476, "Downtrend"], [-0.0026538, "Downtrend"], NEUTRAL, [0, "Downtrend"], NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [0.0020926, "Downtrend"],
        [0.00275551, "Downtrend"], [-0.000271791, "Downtrend"], [-0.000117178, "Downtrend"],
        [-0.000355585, "Downtrend"], [-0.000122877, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [-0.000127, "Downtrend"],
        [0.000435839, "Downtrend"], [1.00582e-05, "Downtrend"], [-0.00102119, "Downtrend"],
        [-0.000239292, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [0.000508259, "Downtrend"],
        [-0.000195879, "Downtrend"], [-0.00161004, "Downtrend"], [0.000256148, "Downtrend"],
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [-0.00262784, "Downtrend"],
        [-0.00408677, "Downtrend"], NEUTRAL, [-0.00178526, "Downtrend"], NEUTRAL, NEUTRAL,
        NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [0.00286601, "Downtrend"],
        [0.00232812, "Downtrend"], NEUTRAL, [-0.00143096, "Downtrend"], NEUTRAL, NEUTRAL,
        NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, [-0.00154281, "Downtrend"],
        NEUTRAL, [0.00133941, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL,
        [-0.00196168, "Downtrend"], NEUTRAL, [-0.00121007, "Downtrend"], NEUTRAL, NEUTRAL, NEUTRAL,
    ],
    [
        NEUTRAL, [0, "Downtrend"], NEUTRAL, NEUTRAL, [0.0085531, "Downtrend"],
        [-3.22681e-05, "Downtrend"], [0.00240971, "Downtrend"], [0.000874922, "Downtrend"],
        [0.0015495, "Downtrend"], [0.000478354, "Downtrend"], [-0.00590987, "Downtrend"],
        [-0.00023596, "Downtrend"], [0.00283505, "Downtrend"], [-0.00075662, "Downtrend"], NEUTRAL,
    ],
    NEUTRAL_ROW,
    NEUTRAL_ROW,
    NEUTRAL_ROW,
    NEUTRAL_ROW,
]


KELP_TREND_MATRIX = [
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
    # Otherwise, return the index + 1 (to match how KELP_TREND_MATRIX is structured)
    return (np.abs(array - value)).argmin() + 1

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
                orders.append(Order(KELP, best_ask, quantity))
                buy_order_volume += quantity
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        bid_amount = order_depth.buy_orders[best_bid]
        if best_bid >= fair_value + params["take_width"] and bid_amount <= 50:
            quantity = min(bid_amount, params["position_limit"] + position)
            if quantity > 0:
                orders.append(Order(KELP, best_bid, -quantity))
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
            orders.append(Order(KELP, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(KELP, fair_for_bid, abs(sent_quantity)))
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
        orders.append(Order(KELP, bbbf + 1, buy_quantity))
    sell_quantity = params["position_limit"] + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(KELP, baaf - 1, -sell_quantity))
    return orders

# ---------------------------------------------------------------------------



class Trader: ##from here

    def __init__(self) -> None:
        self.logger = Logger()
        self.pos_limits = LIMIT
        self.params = PARAMS
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
        self.position_limits = LIMIT

        # -------- CHECK IF WE NEED ALL THIS FOR FINAL STRATEGY --------
        self.jam_mid_prices=[]
        self.djembe_mid_prices=[]
        self.croissant_mid_prices=[]
        self.macaron_mid_prices=[]
        self.pb1_mid_prices=[]
        self.pb2_mid_prices=[]
        self.S1Arr=[]
        self.S2Arr=[]
        self.sunpricearray=[]
        self.conversionMemory=0
        self.position_limits = LIMIT
        self.historical_portfolio_value={params:0 for params in LIMIT}
        self.signalDict={
                    KELP:0,
                    SQUID_INK:0,
                    JAMS:0,
                    DJEMBES:0,
                    CROISSANTS:0
                }
        self.signalTrigger={
            KELP:KTrigger,
            SQUID_INK:STrigger,
            JAMS:JTrigger,
            DJEMBES:DTrigger,
            CROISSANTS:CTrigger
        }
        self.triggerlist={
            product:[] for product in self.signalTrigger
        }

        self.strategy = VoucherOnlyTrader()
    
        self.order_params = {
                        "RAINFOREST_RESIN": {"max_order_AS_size": 40, "buffer": 2},
                        "KELP": {"max_order_AS_size": 30, "buffer": 2},
                        "SQUID_INK": {"max_order_AS_size": 30, "buffer": 2},
                        "VOLCANIC_ROCK_VOUCHER_10250": {"max_order_AS_size": 40, "buffer": 1},
                        "VOLCANIC_ROCK_VOUCHER_10500": {"max_order_AS_size": 40, "buffer": 1},
                    }
        self.gamma = 0.666162234599478
        self.sigma = 0.38344697684007834
        self.k = 0.38711859174678465
        self.T = 1.0
        self.dt = 1.0

        self.prev_mid = {}

        self.rsi = RsiStrategy(VOLCANIC_ROCK, 400)
        self.sunlightHistory=[]
        self.macstateHistory='undefined'
        self.macaronState='undefined'
        self.orangeMacCounter=0
    # --------------------------------------------------------------------------------------------

        self.major = 3
        self.minor = 3

        self.InkMeanState = None
        self.InkSmaArr = []
        self.InkLmaArr = []
        N = float('nan')
        NAN_ROW = [N] * 15

        self.meanMatrix = [
            [N, N, N, -0.001013, 0.002865, N, 0.00017, 0.005352, 0.000894, 0.000125, N, N, N, N, N],
            [N, N, N, N, N, -0.002744, 0.000507, -0.002302, 0.0, 0.002754, N, N, N, N, N],
            [N, N, N, N, N, -0.000759, -0.003776, 0.004229, -0.000252, -0.002239, 0.009392, N, N, N, N],
            [N, N, N, N, N, 0.000245, -0.006816, 0.002132, 0.002243, N, N, N, N, N, N],
            [N, N, N, N, -0.001013, 0.000254, 0.001504, -0.001011, N, N, N, N, N, N, N],
            [N, N, N, N, N, N, -0.00038, 0.000338, N, N, N, N, N, N, N],
            [N, N, N, N, -0.001511, 0.001262, 1e-06, -0.000138, -8.3e-05, N, N, N, N, N, N],
            [N, N, N, N, -0.00051, 0.000601, -0.000907, 0.000164, 0.000382, -0.001011, N, N, N, N, N],
            [N, N, N, 0.004913, 0.000631, 0.000423, 3e-06, 0.000319, -0.002352, N, 0.003251, N, N, N, N],
            [N, N, 0.002021, -0.001013, 0.000334, 0.000795, -0.000396, 0.000563, 0.000507, -0.000321, -0.000759, N, N, N, N],
            [N, N, 0.001504, 0.001091, 0.000337, 0.000195, 0.000803, -0.000172, -0.000695, -0.002879, -0.006189, -0.000254, N, N, N],
            [N, N, 0.001504, 0.0, 4e-06, -0.000388, 0.000202, 0.000501, -0.000555, -0.000756, 0.000586, N, N, N, N],
            [N, 0.001014, 0.000127, 0.000195, 0.000351, 0.000621, 0.000208, 0.000196, 0.000719, -0.000256, 0.001089, 0.0, N, N, N],
            [N, 0.0, 0.000107, 0.000841, 0.000941, 0.000143, 4.1e-05, 0.000118, 0.000314, -0.000352, -0.001084, -0.000494, N, N, N],
            [N, 0.000845, 0.000508, 0.000791, 0.000641, 0.000145, 0.000316, 0.000245, 7.3e-05, -0.000147, -0.000219, -0.00038, N, N, N],
            [N, -0.000255, 0.00022, 0.000423, 0.000444, 0.000122, 0.000309, 0.000341, 0.00017, -0.000148, -0.000522, -0.000506, -0.002008, N, N],
            [N, 0.000423, 0.000475, 0.000409, 0.000415, 0.000189, 0.000267, 0.00013, 0.000219, -0.000436, 0.000254, -0.0, N, N, N],
            [N, -0.000253, 0.000253, 0.000172, 0.000272, 0.000195, 0.000351, 0.000112, -1.4e-05, 2.9e-05, 0.000399, -0.000251, N, 0.001752, N],
            [N, 0.0, 5.2e-05, 0.00036, 0.000297, 0.000119, 0.000192, 2.8e-05, 5e-05, -0.000107, 2.9e-05, 1e-06, N, -0.001136, N],
            [N, 8.5e-05, 0.000598, 0.000325, 0.000252, 0.000178, 0.000217, 2.7e-05, 4e-06, -0.000169, -3.7e-05, -0.000664, -0.000255, 0.001014, N],
            [0.0, -0.000254, -8.3e-05, 6.9e-05, 0.000171, 3.5e-05, 5.4e-05, -6e-06, -7.4e-05, -0.000207, -0.000186, -0.000243, -0.000504, 0.000253, N],
            NAN_ROW,
            NAN_ROW,
            [N, -0.000501, -0.000759, 0.001605, 0.000162, 0.00013, 7.3e-05, -8.5e-05, -6.7e-05, -0.00019, -0.000183, -0.000317, -7e-05, -0.000451, 0.0],
            [N, N, 0.000663, 0.001167, -9.9e-05, -7e-06, 2e-05, -8.5e-05, -0.000111, -0.000339, -0.000255, -0.000152, -0.000473, -0.000254, N],
            [N, N, 0.000507, 0.00075, 2.8e-05, 0.000189, 2.4e-05, -0.000104, -0.000178, -0.00025, -0.000289, -0.000302, -0.000659, -0.000759, N],
            [N, N, N, -2.7e-05, 0.000157, -1e-05, -7e-05, -0.000131, -0.000285, -0.000441, -0.000343, -0.000318, -0.000367, -0.000356, N],
            [N, N, N, -0.000759, 0.000198, -2e-06, -7.1e-05, -0.000163, -0.000154, -0.000356, -0.000467, -0.000672, -0.000589, -0.001089, N],
            [N, 0.002243, N, N, -1e-06, -0.000115, -2.9e-05, -0.000117, -7.5e-05, -0.000328, -0.000292, -0.000505, -0.000884, -0.001258, N],
            [N, N, N, N, 0.003824, 0.001148, 5.8e-05, -5.1e-05, -0.000296, -0.000127, -0.000598, 3.1e-05, -0.000316, -0.001511, N],
            [N, N, N, -0.000334, -0.000413, 0.000986, 5.3e-05, -0.000171, -0.000281, -0.00055, 0.000374, -0.000506, -0.003776, N, N],
            [N, N, N, 0.000764, -0.000254, 6.6e-05, 0.000146, 0.00019, -0.000146, -0.000421, -0.000527, -0.00017, -0.00051, N, N],
            [N, N, N, -0.000506, 0.000254, -8.3e-05, 0.000254, 0.000167, 0.000214, -0.00095, -0.000629, -0.001003, -0.001013, N, N],
            [N, -0.001511, N, -0.000625, -0.003776, -0.00017, -0.001006, 0.000202, -0.000666, -0.000381, -0.000379, 0.000338, N, N, N],
            [N, N, N, 0.001504, N, -0.000255, -0.00142, -0.000676, 0.00029, -0.001108, -0.002744, N, 0.0, N, N],
            [N, N, N, N, N, 0.002021, 0.002754, -0.000283, -0.000127, -0.000411, -0.000126, N, N, N, N],
            [N, N, N, N, N, N, -0.000127, 0.000381, 1e-06, -0.001004, -0.000254, N, N, N, N],
            [N, N, N, N, N, N, 0.000509, -0.000162, -0.001601, 0.000254, N, N, N, N, N],
            [N, N, N, N, N, N, -0.002628, -0.003599, N, -0.001748, N, N, N, N, N],
            [N, N, N, N, N, N, 0.002754, 0.00233, N, -0.001511, N, N, N, N, N],
            [N, N, N, N, N, N, -0.001511, N, 0.001262, N, N, N, N, N, N],
            [N, N, N, N, N, N, N, N, N, -0.002006, N, -0.001258, N, N, N],
            [N, N, N, N, N, N, N, 0.00139, N, N, N, -0.000254, N, N, N],
            [N, N, N, N, 0.008553, -4e-06, 0.002405, 0.000698, 0.00162, 0.000509, -0.004959, N, 0.002754, -0.000759, N],
        ]

        self.inklines = [
            -4.959e-03, -4.706e-03, -4.520e-03, -4.263e-03, -3.991e-03, -3.776e-03,
            -3.246e-03, -3.014e-03, -2.744e-03, -2.512e-03, -2.239e-03, -2.008e-03,
            -1.748e-03, -1.511e-03, -1.258e-03, -1.013e-03, -7.590e-04, -5.100e-04,
            -2.540e-04, 0.000e00, 3.000e-06, 9.000e-06, 2.540e-04, 5.090e-04,
            7.600e-04, 1.014e-03, 1.262e-03, 1.504e-03, 1.736e-03, 2.021e-03,
            2.243e-03, 2.513e-03, 2.754e-03, 3.012e-03, 3.251e-03, 3.536e-03,
            3.743e-03, 4.014e-03, 4.229e-03, 4.528e-03, 4.685e-03, 4.913e-03,
        ]

        self.kelplines = [
            0.000494, -0.000247, 0.0, -0.000494, 0.000247, 0.000741, -0.000741,
            0.000989, -0.000988, -0.00123, 0.00148, -0.00148, 0.00124,
        ]

        self.previousInkprice = None
        self.previousKelpprice = None

    # --------------------------------------------------------------------------------------------



    def vwapOD(self,product: str, orderdepth:OrderDepth, mode: Optional[str] = None) -> float:
        depth = orderdepth
        if product not in depth.buy_orders or depth.sell_orders:
            return 0.0  # Product doesn't exist

        vwap = 0
        total_amt = 0

        if mode == 'bid' and depth.buy_orders:
            for prc, amt in depth.buy_orders.items():
                vwap += prc * amt
                total_amt += amt
        elif mode == 'ask' and depth.sell_orders:
            for prc, amt in depth.sell_orders.items():
                vwap += prc * amt
                total_amt += amt
        elif mode is None:
            for prc, amt in depth.buy_orders.items():
                vwap += prc * amt
                total_amt += amt
            for prc, amt in depth.sell_orders.items():
                vwap += prc * abs(amt)
                total_amt += abs(amt)

        if total_amt == 0:
            return 0.0  # Avoid division by zero

        return np.round(vwap / total_amt, decimals=5)

    def Viable(self,price,feedict:dict,mode:str) -> bool:
        if mode =='import':
            if price-self.ImportDiff<0:
                return False
            if price-self.ImportDiff-minimumMacOrder*(feedict[ITARIFF]+feedict[TRANSPORT]+0.1*self.averageHold)-MacShipThreshold<self.ImportSigma:
                return False #SHOULD BE FALSE BUT FOR TESTS ONLY
        elif mode == 'export':
            if price-self.ExportDiff>0:
                return False
            if price-self.ExportDiff+minimumMacOrder*(feedict[ETARIFF]+feedict[TRANSPORT]+0.1*self.averageHold)+MacShipThreshold>-self.ExportSigma:
                return False #SHOULD BE FALSE BUT FOR TESTS ONLY
        return True
    
    def OrderOptimised(self, product: str, size: int, mode: str, state: TradingState) -> list[Order]:
            orders = []
            VolTarget = size
            if size==0:
                return []
            depth = state.order_depths[product]

            if mode == 'buy':
                # Ensure there are sell orders available; if not, return an empty list.
                if not depth.sell_orders:
                    return orders
                sell_orders = depth.sell_orders  # Use the correct attribute name
                # Get the prices sorted in ascending order (lowest offers first)
                sorted_prices = sorted(sell_orders.keys(), reverse= False)
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
        position = state.position.get(product, 0)
        return np.round(position/LIMIT[product], decimals=3)
    
    def mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.sell_orders:
            total_ask = sum(price * quantity for price, quantity in order_depth.sell_orders.items())
            total_qty = sum(quantity for quantity in order_depth.sell_orders.values())
            m1 = total_ask / total_qty if total_qty != 0 else 0
        else:
            m1 = 0

        if order_depth.buy_orders:
            total_bid = sum(price * quantity for price, quantity in order_depth.buy_orders.items())
            total_qty = sum(quantity for quantity in order_depth.buy_orders.values())
            m2 = total_bid / total_qty if total_qty != 0 else 0
        else:
            m2 = 0

        return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)
    
    def calculate_realized_returns(self, product: str, state: TradingState) -> float:
        # Retrieve historical trades for the product
        own_trades = state.own_trades.get(product, [])
        if not own_trades:
            return 0.0  # No trades executed, so no realized profit or loss

        # Calculate the total cost and total quantity of executed trades
        total_cost = 0.0
        total_quantity = 0
        for trade in own_trades:
            if trade.price <= 0 or abs(trade.quantity) > LIMIT[product]:
                continue
            total_cost += trade.price * trade.quantity
            total_quantity += abs(trade.quantity)

        # If no quantity has been traded, return 0
        if total_quantity == 0:
            return 0.0

        # Calculate the average entry price
        avg_entry_price = np.round(total_cost / total_quantity,decimals=1)

        # Calculate the current portfolio value for the product
        current_position = self.position.get(product, 0)
        if current_position == 0:
            return 0.0  # No position, so no realized profit or loss

        # Retrieve the current mid-price
        order_depth = state.order_depths.get(product)
        if not order_depth:
            return 0.0  # No market data available

        mid_price = self.mid_price(order_depth)

        # Calculate the realized profit or loss as a quantity
        # realized_quantity = (mid_price - avg_entry_price) * current_position
        realized_quantity=(mid_price)*current_position-total_cost

        return realized_quantity
    def currentPNL(self, product: str, state: TradingState) -> float:
        # Retrieve historical trades for the product
        own_trades = state.own_trades.get(product, [])
        if not own_trades:
            return 0.0  # No trades executed, so no realized profit or loss

        # Calculate the total cost and total quantity of executed trades
        total_cost = 0.0
        total_quantity = 0
        for trade in own_trades:
            if trade.price <= 0 or abs(trade.quantity) > LIMIT[product]:
                continue
            total_cost += trade.price * trade.quantity
            total_quantity += abs(trade.quantity)

        return total_cost
    
    def UpdatePreviousPositionCounter(self,product,state:TradingState) -> None:
            if product not in set(state.position.keys()):
                return None
            if (
                state.position[product] == self.previousposition[product]
            ):  # Updates previouspositionCounter
                self.positionCounter[product] += 1
            else:
                self.positionCounter[product] = 0
    def martingaleOrders(self,product,price,orderDepth:OrderDepth,mode:str,volume=None)->list[Order]:
        midprice=mid_price(orderDepth)
        vwap=self.vwapOD(product,orderDepth)
        askGaps=[askPrice-midprice for askPrice in orderDepth.sell_orders]
        bidGaps=[midprice-bidPrice for bidPrice in orderDepth.buy_orders]
        if vwap>midprice:
            momentum=1
        elif vwap<midprice:
            momentum=-1
        else:
            momentum=0
        if volume==None:
            if mode == 'buy':
                return [Order(product,int(price+abs(gap)+momentum),abs(orderDepth.sell_orders[_])) for gap,_ in zip(askGaps,orderDepth.sell_orders)]  
            elif mode == 'sell':
                return [Order(product,int(price-abs(gap)+momentum),-abs(orderDepth.buy_orders[_])) for gap,_ in zip(bidGaps,orderDepth.buy_orders)]
        else:
            if mode =='buy':
                volumeincrement=int(volume/len(askGaps))
                return [Order(product,int(price+abs(gap)+momentum),abs(volumeincrement)) for gap,_ in zip(askGaps,orderDepth.sell_orders)]  
            elif mode == 'sell':
                volumeincrement=int(volume/len(bidGaps))
                return [Order(product,int(price-abs(gap)+momentum),-abs(volumeincrement)) for gap,_ in zip(bidGaps,orderDepth.buy_orders)]
    def PBPricing(self,base,synth,od,product,state)->list[Order]:
        if base>synth+PBtolerance:
            return self.martingaleOrders(product=product,price=int(base),orderDepth=od,mode='sell')
        elif synth-PBtolerance>base:
            return self.martingaleOrders(product=product,price=int(base),orderDepth=od,mode='buy')
        else:
            return []

    def updateSunlightTrend(self,state:TradingState)->str:
        self.sunlightHistory=self.sunlightHistory[-20:]
        diffs = np.diff(self.sunlightHistory[-2:], n=1)
        if diffs.size and (diffs > 0).all():
            return 'uptrend'
        elif diffs.size and (diffs < 0).all():
            return 'downtrend'
        else:
            return 'neutral'
            
    def mean_reversion_trade(self, product: str, mid_prices: List[float],
                             order_depth: OrderDepth, current_position: int,
                             position_limit: int, result: Dict[str, List[Order]], state: TradingState,
                             window: int = 50, z_score_thresh: float = 2.0) -> None:
        # Only run if we have enough price history
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
            result[product].append(Order(product, best_ask, buy_volume))
        # Sell when price is above the upper band - overvalued
        elif current_mid > upper_band:
            result[product].append(Order(product, best_bid, sell_volume))

    def avellaneda_stoikov(self, product, mid, inventory):
        reservation_price = mid - inventory * self.gamma * (self.sigma ** 2) * self.T
        optimal_spread = (2 / self.gamma) * math.log(1 + self.gamma / self.k)
        bid = reservation_price - optimal_spread / 2
        ask = reservation_price + optimal_spread / 2

        max_order_size = self.order_params[product]["max_order_AS_size"]
        buffer = self.order_params[product]["buffer"]
        limit = self.position_limits[product]
        order_size = max(1, min(max_order_size, (limit - abs(inventory)) // buffer))
        return [Order(product, int(round(bid)), order_size),
                Order(product, int(round(ask)), -order_size)]
    # --------------------------------------------------------------------------------------------            



    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        def sortDict(dictionary):
            return {key: dictionary[key] for key in sorted(dictionary)}

        def find_matrix_index(array, value, mode):
            if mode == "KELP":
                tol = 0.0015  # TODO slightly skewed
            if mode == "INK":
                tol = 0.0058
            if abs(value) > tol:
                if value > tol:
                    return -1  # positive anomalous
                else:
                    return 0  # negative anomalous
            return (np.abs(array - value)).argmin() + 1

        def find_nearest_value(array, value, mode):
            if mode == "KELP":
                tol = 0.0015
            if mode == "INK":
                tol = 0.006
            if abs(value) > tol:
                return value
            return array[(np.abs(array - value)).argmin()]

        def current_price(bid: dict, ask: dict):
            return np.round(
                (max(set(bid.keys())) + min(set(ask.keys()))) / 2, decimals=1
            )

        all_orders: List[Order] = []
        result = {product: [] for product in LIMIT}
        conversions = 0
        trader_data = ""
        kelporder = []
        inkorder = []
        resinorder = []
        OrderbookDict = state.order_depths

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
            # logger.print(f"Error loading traderData: {e}")
            loaded_data = {}
        # ------------------------------------------------------------------------



        # -------- RESIN: Simple Market Making Assuming Constant Fair Value --------
        if RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(RAINFOREST_RESIN, 0)
            resin_params = PARAMS[RAINFOREST_RESIN]
            resin_order_depth = state.order_depths[RAINFOREST_RESIN]
            resin_fair_value = resin_params["fair_value"]
            orders_take, bo, so = resin_take_orders(
                resin_order_depth, resin_fair_value, resin_position, LIMIT[RAINFOREST_RESIN], product=RAINFOREST_RESIN
            )
            orders_clear, bo, so = resin_clear_orders(
                resin_order_depth, resin_position, resin_fair_value, LIMIT[RAINFOREST_RESIN], bo, so, product=RAINFOREST_RESIN
            )
            orders_make = resin_make_orders(
                resin_order_depth, resin_fair_value, resin_position, LIMIT[RAINFOREST_RESIN], bo, so, product=RAINFOREST_RESIN
            )
            result[RAINFOREST_RESIN] += orders_take + orders_clear + orders_make
        # -------------------------------------------------------------------------



        # ----------------------------- BASKET TRADING ----------------------------
        futurePriceDict={product:None for product in [KELP,SQUID_INK,JAMS,DJEMBES,CROISSANTS]}
        for product in [KELP,SQUID_INK,JAMS,DJEMBES,CROISSANTS]:
            if state.order_depths[product].buy_orders or state.order_depths[product].sell_orders:
                ProdOD=state.order_depths[product]
                midP=mid_price(ProdOD)
                if product==KELP:
                    midParr =self.kelp_mid_prices
                    Korder=1
                    order=Korder
                elif product==SQUID_INK:
                    midParr=self.ink_mid_prices
                    Sorder=2
                    order=Sorder
                elif product==JAMS:
                    midParr=self.jam_mid_prices
                    Jorder=2
                    order=Jorder
                elif product == DJEMBES:
                    midParr=self.djembe_mid_prices
                    Dorder=2
                    order=Dorder
                elif product == CROISSANTS:
                    midParr=self.croissant_mid_prices
                    Dorder=2
                    order=Dorder
                midParr.append(midP)
                midParr=midParr[-101:]
            if state.timestamp >= 200:
                if midParr[-1]>midParr[-2]:
                    self.signalDict[product]+=1
                elif midParr[-1]<midParr[-2]:
                    self.signalDict[product]-=1
                else:
                    pass
                self.triggerlist[product].append(self.signalDict[product])
                if state.timestamp>=100*50:
                    smoothedmid=kalman_filter_1d(midParr[-100:])
                    nextPrice=project_next(smoothedmid,window=30,correlation_threshold=0.7)
                    Kmode=None
                    futurePriceDict[product]=nextPrice
        if state.timestamp >= 100*50:
            pbs=[PICNIC_BASKET1,PICNIC_BASKET2]
            comps=[CROISSANTS,DJEMBES]
            PB1OD=state.order_depths[PICNIC_BASKET1]
            PB2OD=state.order_depths[PICNIC_BASKET2]
            self.pb1_mid_prices.append(mid_price(PB1OD))
            self.pb1_mid_prices=self.pb1_mid_prices[-200:]
            self.pb2_mid_prices.append(mid_price(PB2OD))
            self.pb2_mid_prices=self.pb2_mid_prices[-200:]
            for pb in pbs:
                synthDict={PICNIC_BASKET1:[6,3,1],PICNIC_BASKET2:[4,2,0]}
                SArr={PICNIC_BASKET1:self.S1Arr,PICNIC_BASKET2:self.S2Arr}
                pbMidDict={PICNIC_BASKET1:self.pb1_mid_prices[-1],PICNIC_BASKET2:self.pb2_mid_prices[-1]}
                compPricearr=[self.croissant_mid_prices[-100:][-1],self.jam_mid_prices[-100:][-1],self.djembe_mid_prices[-100:][-1]]
                coeffs=synthDict[pb]
                synthVal=0
                pbVal=pbMidDict[pb]
                synthVal+=pbVal
                for coeff,price in zip(coeffs,compPricearr):
                    synthVal-=coeff*price
                SArr[pb].append(synthVal)
                SArr[pb]=SArr[pb][-200:]
                if self.positionCounter[pb]>5 and self.calculate_realized_returns(pb,state)>0:
                    if self.position[pb]>0:
                        result[pb]=self.martingaleOrders(pb,pbVal-2,state.order_depths[pb],mode='sell',volume=abs(self.position[pb]))
                    elif self.position[pb]<0:
                        result[pb]=self.martingaleOrders(pb,pbVal+2,state.order_depths[pb],mode='buy',volume=abs(self.position[pb]))
                else:
                    zscoredict={PICNIC_BASKET1:3,PICNIC_BASKET2:2.5}
                    currentZ=np.round((synthVal-np.mean(SArr[pb][-30:]))/np.sqrt(np.var(SArr[pb][-30:])),decimals=2)
                    if currentZ>=zscoredict[pb]:
                        result[pb]+=self.martingaleOrders(pb,pbVal,state.order_depths[pb],mode='sell')
                        for product,comp in zip(comps,compPricearr):
                            result[product]=self.martingaleOrders(product,comp,state.order_depths[product],'buy')
                    elif currentZ<=-zscoredict[pb]:
                        result[pb]+=self.martingaleOrders(pb,pbVal,state.order_depths[pb],mode='buy')
                        for product,comp in zip(comps,compPricearr):
                            result[product]=self.martingaleOrders(product,comp,state.order_depths[product],'sell')
                for product in comps:
                    if self.positionCounter[product]>5 and self.calculate_realized_returns(product,state)>0:
                        if self.position[product]>0:
                            result[product]=self.martingaleOrders(product,compPricearr[product]-2,state.order_depths[pb],mode='sell',volume=abs(self.position[product]))
                        elif self.position[pb]<0:
                            result[product]=self.martingaleOrders(product,compPricearr[product]+2,state.order_depths[pb],mode='buy',volume=abs(self.position[product])) 
        # -------------------------------------------------------------------------



        # --------------------------------- KELP ----------------------------------
        if KELP in state.order_depths:
            kelp_position = state.position.get(KELP, 0)
            kelp_params = self.params[KELP]
            kelp_order_depth = state.order_depths[KELP]
            kelp_fair = kelp_fair_value(
                kelp_order_depth, kelp_params["default_fair_method"], kelp_params["min_volume_filter"]
            )
            kelp_take, bo, so = kelp_take_orders(kelp_order_depth, int(kelp_fair), kelp_params, kelp_position)
            kelp_clear, bo, so = kelp_clear_orders(kelp_order_depth, kelp_position, kelp_params, int(kelp_fair), bo, so)
            kelp_make = kelp_make_orders(kelp_order_depth, int(kelp_fair), kelp_position, kelp_params, bo, so)
            result[KELP] = kelp_take + kelp_clear + kelp_make
        # -------------------------------------------------------------------------



        # ------------------------------ SQUID INK --------------------------------
        for product in OrderbookDict:
            if product == "SQUID_INK":
                InkOrderbookDepth = OrderbookDict[product]
                InkbidSpread = sortDict(InkOrderbookDepth.buy_orders)
                InkaskSpread = sortDict(InkOrderbookDepth.sell_orders)
                KelpOrderbookDepth = OrderbookDict["KELP"]
                KelpbidSpread = sortDict(KelpOrderbookDepth.buy_orders)
                KelpaskSpread = sortDict(KelpOrderbookDepth.sell_orders)
                CurrentInkPrice = current_price(InkbidSpread, InkaskSpread)
                CurrentKelpPrice = current_price(KelpbidSpread, KelpaskSpread)
                self.InkLmaArr.append(CurrentInkPrice)
                self.InkSmaArr.append(CurrentInkPrice)
                if len(self.InkLmaArr) > 40:
                    self.InkLmaArr = self.InkLmaArr[1:41]
                if len(self.InkSmaArr) > 10:
                    self.InkSmaArr = self.InkSmaArr[1:11]
                if state.timestamp > 5000:
                    InkLma = np.mean(self.InkLmaArr)
                    InkSma = np.mean(self.InkSmaArr)
                    if self.InkMeanState is None:
                        if InkLma > InkSma:
                            self.InkMeanState = "BEAR"
                        if InkLma < InkSma:
                            self.InkMeanState = "BULL"
                if self.previousInkprice is None and self.previousKelpprice is None:
                    self.previousInkprice = CurrentInkPrice
                    self.previousKelpprice = CurrentKelpPrice
                    continue
                else:
                    InkReturn = np.round(
                        (
                            current_price(InkbidSpread, InkaskSpread)
                            - self.previousInkprice
                        )
                        / self.previousInkprice,
                        decimals=6,
                    )
                    KelpReturn = np.round(
                        (
                            current_price(KelpbidSpread, KelpaskSpread)
                            - self.previousKelpprice
                        )
                        / self.previousKelpprice,
                        decimals=6,
                    )
                    InkMatrixLoc = find_matrix_index(
                        array=self.inklines, value=InkReturn, mode="INK"
                    )
                    KelpMatrixLoc = find_matrix_index(
                        array=self.kelplines, value=KelpReturn, mode="KELP"
                    )
                    trend = TREND_MATRIX[InkMatrixLoc][KelpMatrixLoc][1]
                    meanReturn = self.meanMatrix[InkMatrixLoc][KelpMatrixLoc]
                    if meanReturn == np.nan:
                        meanReturn = 0
                    tplus1InkPrice = meanReturn * CurrentInkPrice + CurrentInkPrice
                    predicted_return = meanReturn

                    # Trend-based trading logic
                    CheapestPrice = int(round(min(InkaskSpread.keys())))
                    HighestPrice = int(round(max(InkbidSpread.keys()))) 
                    
                    if trend == "Uptrend":
                        if predicted_return > 0: 
                            # Big Bull & Small Bull Prediction
                            inkorder.append(Order("SQUID_INK", CheapestPrice, self.major)) #0
                        else:
                            #Big Bull & Small Bear Prediction
                            inkorder.append(Order("SQUID_INK", CheapestPrice, -self.minor)) #0

                    elif trend == "Downtrend": #Bullish
                        if predicted_return < 0:
                            # Big Bear & Small Bear Prediction
                            inkorder.append(Order("SQUID_INK", HighestPrice, -self.major)) #-4
                        else:
                            # Big Bear & Small Bull Prediction
                            inkorder.append(Order("SQUID_INK", CheapestPrice, self.minor)) #14

                    elif trend == "Neutral":
                        if predicted_return > 0:
                            inkorder.append(Order("SQUID_INK", CheapestPrice, 0)) #0
                        elif predicted_return < 0:
                            inkorder.append(Order("SQUID_INK", HighestPrice, 0)) #0
                    else:
                        continue
            if product == "KELP":
                pass
        result["SQUID_INK"] = inkorder
        # -------------------------------------------------------------------------



        # --------------------- VOLCANIC ROCK OPTIONS STRATEGY ---------------------
        if state.traderData:
            self.strategy.load_state(state.traderData)
        options_results, conversions, trader_data = self.strategy.run(state)
        # Add options orders to the result
        for product, orders in options_results.items():
            result[product].extend(orders)
        # --------------------------------------------------------------------------



        # -------------------------- VOLCANIC ROCK STRATEGY ------------------------
        saved = {}
        if state.traderData and state.traderData != '""':
            try:
                saved = json.loads(state.traderData)
            except Exception:
                saved = {}

        self.rsi.load(saved.get("RSI", {}))

        # ----- generate orders ----- #
        rock_orders = self.rsi.run(state)
        if rock_orders:
            result[VOLCANIC_ROCK].extend(rock_orders)

        next_state = {
            "RSI": self.rsi.save(),
        }
        trader_data = json.dumps(next_state, cls=ProsperityEncoder)
        # --------------------------------------------------------------------------



        # ------------------------- MACARONS STRATEGY ------------------------------
        # Update position tracking
        for product in self.pos_limits.keys():
            self.UpdatePreviousPositionCounter(product, state)
            self.position[product] = state.position.get(product, 0)
            try:
                self.update_market_data(product, state)
            except Exception:
                pass
        # Load prior trader data (optional if you're passing internal state)
        try:
            loaded_data = json.loads(state.traderData) if state.traderData and state.traderData != '""' else {}
            if not isinstance(loaded_data, dict): loaded_data = {}
        except Exception as e:
            logger.print(f"Error loading traderData: {e}")
            loaded_data = {}
        obvs = state.observations.conversionObservations[MAGNIFICENT_MACARONS]
        outsideMacaronDict = {
            BID: obvs.bidPrice,
            ASK: obvs.askPrice
        }
        outsideMid=np.mean([outsideMacaronDict[BID],outsideMacaronDict[ASK]])
        feeDict = {
            ETARIFF: abs(obvs.exportTariff),
            ITARIFF: abs(obvs.importTariff),
            SUNLIGHT: obvs.sunlightIndex,
            SUGAR: obvs.sugarPrice,
            TRANSPORT: obvs.transportFees
        }
        if state.timestamp<500000:
            sunlightAdjusted=feeDict[SUNLIGHT]-0.0002*state.timestamp
        else:
            sunlightAdjusted=feeDict[SUNLIGHT]-((-0.0002*state.timestamp)+200)
        self.sunlightHistory.append(sunlightAdjusted)
        self.macaronState=self.updateSunlightTrend(state) #trims sunlight history size too for memory management
        macMid = mid_price(state.order_depths[MAGNIFICENT_MACARONS])    
        
        if self.macstateHistory=='RedBuyExecuted':
            if self.macaronState=='uptrend' and self.sunlightHistory[-1]>(-30-0.05) and self.sunlightHistory[-1]<(-30+0.05):
                self.macstateHistory='OrangeSellTrigger'
                self.orangeMacCounter=0
                result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='sell',volume=abs(self.position[MAGNIFICENT_MACARONS]))
                self.macstateHistory='OrangeSellExecuted'
                ### ORANGE LOGIC
            else:
                result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='buy',volume=1)
        elif self.macstateHistory=='OrangeSellExecuted':
            if self.macaronState=='uptrend' and self.sunlightHistory[-1]>49-0.05 and self.sunlightHistory[-1]<49+0.05:
                self.macstateHistory='GreenSellTrigger'
                result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='sell')
                self.macstateHistory='GreenSellExecuted'
            else:
                pass
                if self.orangeMacCounter<100:
                    result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='sell',volume=1)
                ### ORANGE MAIN LOGIC
                self.orangeMacCounter+=1
        elif self.macstateHistory=='GreenSellExecuted':
            if self.macaronState=='downtrend' and self.sunlightHistory[-1]>25-0.05 and self.sunlightHistory[-1]<25+0.05:
                self.macstateHistory='RedBuyTrigger'
                result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='buy')
                self.macstateHistory='RedBuyExectued'
            else:
                result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='sell',volume=1)
        elif self.macstateHistory=='undefined': #macstatehistory is undefined
            if self.sunlightHistory[-1]>49: #awaiting green
                self.macstateHistory='waitingGreen'
            elif self.sunlightHistory[-1]>25 and self.sunlightHistory[-1]<49:
                if self.macaronState=='downtrend': #waiting red
                    self.macstateHistory='waitingRed'
                elif self.macaronState=='uptrend': #waiting green
                    self.macstateHistory='waitingGreen'
            elif self.sunlightHistory[-1]<25 and self.sunlightHistory[-1]>-30:
                if self.macaronState=='downtrend': #waiting orange
                    self.macstateHistory='waitingOrange'
                elif self.macaronState=='uptrend': #waiting red
                    self.macstateHistory='waitingRed'
            elif self.sunlightHistory[-1]<-30: #waiting orange
                self.macstateHistory='waitingOrange'
        elif self.macstateHistory=='waitingRed':
            if self.macaronState=='downtrend' and self.sunlightHistory[-1]>25-0.05 and self.sunlightHistory[-1]<25+0.05:
                    self.macstateHistory='RedBuyTrigger'
                    result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='buy')
                    self.macstateHistory='RedBuyExecuted'
        elif self.macstateHistory=='waitingGreen':
            if self.macaronState=='uptrend' and self.sunlightHistory[-1]>49-0.05 and self.sunlightHistory[-1]<49+0.05:
                self.macstateHistory='GreenSellTrigger'
                result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='sell')
                self.macstateHistory='GreenSellExecuted'
            elif self.macaronState=='downtrend' and self.sunlightHistory[-1]>25-0.05 and self.sunlightHistory[-1]<25+0.05:
                self.macstateHistory='RedBuyTrigger'
                result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='buy')
                self.macstateHistory='RedBuyExecuted'
        elif self.macstateHistory=='waitingOrange':
            if self.macaronState=='uptrend' and self.sunlightHistory[-1]>(-30-0.05) and self.sunlightHistory[-1]<(-30+0.05):
                self.macstateHistory='OrangeSellTrigger'
                self.orangeMacCounter=0
                result[MAGNIFICENT_MACARONS]+=self.martingaleOrders(MAGNIFICENT_MACARONS,macMid,orderDepth=state.order_depths[MAGNIFICENT_MACARONS],mode='sell',volume=abs(self.position[MAGNIFICENT_MACARONS]))
                self.macstateHistory='OrangeSellExecuted'
        # -------------------------------------------------------------------------


        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data