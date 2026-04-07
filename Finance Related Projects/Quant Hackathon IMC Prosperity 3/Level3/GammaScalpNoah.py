import math
import statistics
import json
import numpy as np
import jsonpickle
from statistics import NormalDist
from typing import Any, Dict, List, Tuple
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
# ------------------ End Logger ------------------

# ------------------ Constants & Parameters ------------------
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# We now evaluate time-to-expiry on 365 trading days.
# Update starting_time_to_expiry accordingly.
TOTAL_DATE=250
PARAMS = {
    VOLCANIC_ROCK: {
        "starting_time_to_expiry": np.round(245 / TOTAL_DATE,decimals=4),  # adjust for 365 days/year
        "std_window": 10,
        "hedge_threshold": 0.1           # Parameterized underlying hedge threshold.
    },
    VOLCANIC_ROCK_VOUCHER_9500: {
        "starting_time_to_expiry": np.round(245 / TOTAL_DATE,decimals=4),
        "strike": 9500,
        "std_window": 10,
        "implied_volatility": 0.15,
        "MISPRICING_THRESHOLD": 30,      # Minimum mispricing to trigger trade.
        "REDUCED_SIZE_THRESHOLD": 60     # Below this, scale size to 0.8.
    },
    VOLCANIC_ROCK_VOUCHER_9750: {
        "starting_time_to_expiry": np.round(245 / TOTAL_DATE,decimals=4),
        "strike": 9750,
        "std_window": 10,
        "implied_volatility": 0.15,
        "MISPRICING_THRESHOLD": 30,
        "REDUCED_SIZE_THRESHOLD": 60
    },
    VOLCANIC_ROCK_VOUCHER_10000: {
        "starting_time_to_expiry": np.round(245 / TOTAL_DATE,decimals=4),
        "strike": 10000,
        "std_window": 10,
        "implied_volatility": 0.15,
        "MISPRICING_THRESHOLD": 30,
        "REDUCED_SIZE_THRESHOLD": 60
    },
    VOLCANIC_ROCK_VOUCHER_10250: {
        "starting_time_to_expiry": np.round(245 / TOTAL_DATE,decimals=4),
        "strike": 10250,
        "std_window": 10,
        "implied_volatility": 0.15,
        "MISPRICING_THRESHOLD": 30,
        "REDUCED_SIZE_THRESHOLD": 60
    },
    VOLCANIC_ROCK_VOUCHER_10500: {
        "starting_time_to_expiry": np.round(245 / TOTAL_DATE,decimals=4),
        "strike": 10500,
        "std_window": 10,
        "implied_volatility": 0.15,
        "MISPRICING_THRESHOLD": 30,
        "REDUCED_SIZE_THRESHOLD": 60
    },
}

POSITION_LIMITS = {
    VOLCANIC_ROCK: 400,
    VOLCANIC_ROCK_VOUCHER_9500: 200,
    VOLCANIC_ROCK_VOUCHER_9750: 200,
    VOLCANIC_ROCK_VOUCHER_10000: 200,
    VOLCANIC_ROCK_VOUCHER_10250: 200,
    VOLCANIC_ROCK_VOUCHER_10500: 200,
}
# ------------------ End Constants ------------------

# ------------------ Trader Class -----------------------
class Trader:
    def __init__(self, params=None):
        self.position = {
            VOLCANIC_ROCK: 0,
            VOLCANIC_ROCK_VOUCHER_9500: 0,
            VOLCANIC_ROCK_VOUCHER_9750: 0,
            VOLCANIC_ROCK_VOUCHER_10000: 0,
            VOLCANIC_ROCK_VOUCHER_10250: 0,
            VOLCANIC_ROCK_VOUCHER_10500: 0,
        }
        self.position_limits = POSITION_LIMITS
        self.params = PARAMS

        # Arrays to store mid-prices.
        self.rock_mid_prices = []
        self.rock9500_mid_prices = []
        self.rock9750_mid_prices = []
        self.rock10000_mid_prices = []
        self.rock10250_mid_prices = []
        self.rock10500_mid_prices = []

    def mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.sell_orders:
            total_ask = sum(price * quantity for price, quantity in order_depth.sell_orders.items())
            total_qty = sum(quantity for quantity in order_depth.sell_orders.values())
            m1 = total_ask / total_qty
        else:
            m1 = 0
        if order_depth.buy_orders:
            total_bid = sum(price * quantity for price, quantity in order_depth.buy_orders.items())
            total_qty = sum(quantity for quantity in order_depth.buy_orders.values())
            m2 = total_bid / total_qty
        else:
            m2 = 0
        return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)
    
    def norm_pdf(self,x):
        """Standard normal probability density function (PDF)."""
        return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

    def norm_cdf(self,x):
        """Standard normal cumulative distribution function (CDF) using an approximation."""
        # Abramowitz and Stegun approximation (good enough for IV estimation)
        k = 1.0 / (1.0 + 0.2316419 * abs(x))
        a = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
        poly = sum([a[i] * k**(i + 1) for i in range(5)])
        approx = 1.0 - self.norm_pdf(x) * poly
        return approx if x >= 0 else 1.0 - approx

    def update_market_data(self, product: str, state: TradingState) -> None:
        order_depth = state.order_depths.get(product)
        if order_depth is None:
            return
        mid = self.mid_price(order_depth)
        if product == VOLCANIC_ROCK:
            self.rock_mid_prices.append(mid)
        elif product == VOLCANIC_ROCK_VOUCHER_9500:
            self.rock9500_mid_prices.append(mid)
        elif product == VOLCANIC_ROCK_VOUCHER_9750:
            self.rock9750_mid_prices.append(mid)
        elif product == VOLCANIC_ROCK_VOUCHER_10000:
            self.rock10000_mid_prices.append(mid)
        elif product == VOLCANIC_ROCK_VOUCHER_10250:
            self.rock10250_mid_prices.append(mid)
        elif product == VOLCANIC_ROCK_VOUCHER_10500:
            self.rock10500_mid_prices.append(mid)

    def black_scholes_call(self, S: float, K: float, vol: float, T: float, r: float = 0, q: float = 0) -> Tuple[float, float, float, float]:
        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        norm_pdf = NormalDist().pdf
        N1 = self.norm_cdf(d1)
        N2 = self.norm_cdf(d2)
        call_price = S * N1 - K * np.exp((q - r) * T) * N2
        delta = N1
        gamma = norm_pdf(d1) / (S * vol * np.sqrt(T))
        vega = S * norm_pdf(d1) * np.sqrt(T)
        return call_price, delta, gamma, vega

    def calc_implied_vol(self, S: float, K: float, T: float, market_call: float, r: float = 0, q: float = 0, initial_vol: float = 0.15) -> float:
        vol = initial_vol
        epsilon = 1e-6
        max_iter = 100
        for _ in range(max_iter):
            price, delta, gamma, vega = self.black_scholes_call(S, K, vol, T, r, q)
            diff = price - market_call
            if abs(diff) < epsilon:
                break
            if abs(vega) < 1e-8:
                break
            vol = vol - diff / vega
            if vol < 0:
                vol = epsilon
        return vol

    def calc_fair_price(self, T: float) -> Dict[str, Tuple[int, float, float, float, float, float, float]]:
        if not self.rock_mid_prices:
            return {}

        S = self.rock_mid_prices[-1]
        fair_prices = {}

        voucher_to_prices = {
            VOLCANIC_ROCK_VOUCHER_9500: self.rock9500_mid_prices,
            VOLCANIC_ROCK_VOUCHER_9750: self.rock9750_mid_prices,
            VOLCANIC_ROCK_VOUCHER_10000: self.rock10000_mid_prices,
            VOLCANIC_ROCK_VOUCHER_10250: self.rock10250_mid_prices,
            VOLCANIC_ROCK_VOUCHER_10500: self.rock10500_mid_prices,
        }

        for voucher, price_list in voucher_to_prices.items():
            if not price_list:
                continue

            mid_price = price_list[-1]
            strike = self.params[voucher]["strike"]

            dynamic_vol = self.calc_implied_vol(
                S, strike, T, market_call=mid_price, r=0, q=0,
                initial_vol=self.params[voucher]["implied_volatility"]
            )

            bs_price, delta, gamma, vega = self.black_scholes_call(S, strike, dynamic_vol, T)

            fair_prices[voucher] = (strike, mid_price, bs_price, delta, gamma, vega, dynamic_vol)

        return fair_prices


    def delta_aggregation_and_hedge(self, state: TradingState, T: float) -> List[Order]:
        orders: List[Order] = []
        if not self.rock_mid_prices:
            return orders
        
 
        
        S = self.rock_mid_prices[-1]
        fair_prices = self.calc_fair_price(T)

        net_option_delta = sum(
            self.position.get(voucher, 0) * fair_prices[voucher][3]
            for voucher in fair_prices
            if self.position.get(voucher, 0) != 0
        )

        underlying_delta = self.position.get(VOLCANIC_ROCK, 0)
        net_delta = net_option_delta + underlying_delta

        hedge_order_size = -int(round(net_delta))
        hedge_threshold = self.params[VOLCANIC_ROCK].get("hedge_threshold", 0.1)
        if abs(net_delta) <= hedge_threshold:
            return orders  # No hedge needed
        rock_depth = state.order_depths.get(VOLCANIC_ROCK)
        if hedge_order_size > 0 and rock_depth and rock_depth.sell_orders:
            price = min(rock_depth.sell_orders.keys())
        elif hedge_order_size < 0 and rock_depth and rock_depth.buy_orders:
            price = max(rock_depth.buy_orders.keys())
        else:
            price = S  # fallback to mid price

        orders.append(Order(VOLCANIC_ROCK, price, hedge_order_size))
        return orders


    def trade_itm_options(self, state: TradingState, T: float) -> List[Order]:
        if not self.rock_mid_prices:
            return []

        orders: List[Order] = []
        S = self.rock_mid_prices[-1]
        fair_prices = self.calc_fair_price(T)

        for voucher, fair in fair_prices.items():
            strike, market_mid_price, bs_price, *_ , dynamic_vol = fair

            if S <= strike:
                continue  # Only trade in-the-money calls

            price_diff = bs_price - market_mid_price
            MISPRICING_THRESHOLD = self.params[voucher].get("MISPRICING_THRESHOLD", 30)
            REDUCED_SIZE_THRESHOLD = self.params[voucher].get("REDUCED_SIZE_THRESHOLD", 60)

            if abs(price_diff) < MISPRICING_THRESHOLD:
                continue  # Not enough mispricing

            depth = state.order_depths.get(voucher)
            if not depth:
                continue

            if price_diff > 0:
                if not depth.sell_orders:
                    continue
                exec_price = min(depth.sell_orders)
                direction = 1  # buy undervalued
            else:
                if not depth.buy_orders:
                    continue
                exec_price = max(depth.buy_orders)
                direction = -1  # sell overvalued

            pos = self.position.get(voucher, 0)
            limit = self.position_limits.get(voucher, 200)
            remaining = limit - abs(pos)
            if remaining <= 0:
                continue

            size_factor = 0.8 if abs(price_diff) < REDUCED_SIZE_THRESHOLD else 1.0
            order_size = max(int(size_factor * remaining), 1)

            orders.append(Order(voucher, exec_price, direction * order_size))

        return orders


    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result = {product: [] for product in POSITION_LIMITS}
        conversions = 0
        trader_data = ""
        
        # Update exchange positions.
        for product in state.position:
            self.position[product] = state.position[product]

        # Update market data for our products.
        for product in state.order_depths:
            if product in (
                VOLCANIC_ROCK,
                VOLCANIC_ROCK_VOUCHER_9500,
                VOLCANIC_ROCK_VOUCHER_9750,
                VOLCANIC_ROCK_VOUCHER_10000,
                VOLCANIC_ROCK_VOUCHER_10250,
                VOLCANIC_ROCK_VOUCHER_10500,
            ):
                try:
                    self.update_market_data(product, state)
                except Exception:
                    pass

        # Calculate time to expiry using TOTAL_DAT trading days.
        time_expiry = self.params[VOLCANIC_ROCK]["starting_time_to_expiry"] - (state.timestamp / 1000000 / TOTAL_DATE)
        
        option_orders = self.trade_itm_options(state, time_expiry)
        hedge_orders = self.delta_aggregation_and_hedge(state, time_expiry)
        for optionorder,hedgeorder in zip(option_orders,hedge_orders):
            result[optionorder.symbol].append(optionorder)
            result[hedgeorder.symbol].append(hedgeorder)
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
