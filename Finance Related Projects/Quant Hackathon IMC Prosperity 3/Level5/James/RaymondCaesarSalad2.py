import math
import statistics
import json
import numpy as np
import jsonpickle
from statistics import NormalDist
from typing import Any, Dict, List, Tuple
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

# ───────────────────────── Logger for Visualiser ─────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[Symbol, List[Order]],
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
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state,
                        self.truncate(state.traderData, max_item_length)
                        if hasattr(state, "traderData")
                        else "",
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    # ------------------- compression helpers (unchanged) -------------------
    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
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

    def compress_listings(
        self, listings: Dict[Symbol, Listing]
    ) -> List[List[Any]]:
        return [
            [listing.symbol, listing.product, listing.denomination]
            for listing in listings.values()
        ]

    def compress_order_depths(
        self, order_depths: Dict[Symbol, OrderDepth]
    ) -> Dict[Symbol, List[Any]]:
        return {
            symbol: [od.buy_orders, od.sell_orders]
            for symbol, od in order_depths.items()
        }

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        return [
            [
                tr.symbol,
                tr.price,
                tr.quantity,
                tr.buyer,
                tr.seller,
                tr.timestamp,
            ]
            for trade_list in trades.values()
            for tr in trade_list
        ]

    def compress_observations(self, obs: Observation) -> List[Any]:
        conv_obs = {
            prod: [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
            for prod, o in obs.conversionObservations.items()
        }
        return [obs.plainValueObservations, conv_obs]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        return [
            [o.symbol, o.price, o.quantity]
            for order_list in orders.values()
            for o in order_list
        ]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()
# ──────────────────────────────────────────────────────────────────────────


# -------------------------- product constants ----------------------------
RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
CROISSANTS = "CROISSANTS"
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"
JAMS = "JAMS"
DJEMBES = "DJEMBES"
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

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
    MAGNIFICENT_MACARONS: {"limit": 75, "conversion": 10},
}

# ----------------------------- parameters -------------------------------
PARAMS = {
    RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 25,
    },
    # … (unchanged PARAMS for other products) …
    MAGNIFICENT_MACARONS: {"FILL": 1},
}

# -------------------------- helper: book VWAP ---------------------------
def vwap(product: str, state: TradingState) -> float:
    total, vol = 0.0, 0
    for p, q in state.order_depths[product].buy_orders.items():
        total += p * q
        vol += q
    for p, q in state.order_depths[product].sell_orders.items():
        total += p * abs(q)
        vol += abs(q)
    return np.round(total / vol, 5) if vol else 0.0


# ------------------------------- Trader ---------------------------------
class Trader:
    def __init__(self, params=None):
        self.position = {p: 0 for p in LIMIT}
        self.position_limits = {
            p: LIMIT[p] if isinstance(LIMIT[p], int) else LIMIT[p]["limit"]
            for p in LIMIT
            if p != MAGNIFICENT_MACARONS
        }
        self.params = PARAMS

        # price‑history arrays (truncated list of attributes kept)
        self.rock10000_mid_prices = []
        self.rock10000_timestamps = []

    # ---------- tiny helpers ----------
    @staticmethod
    def mid_price(depth: OrderDepth) -> float:
        m_ask = (
            sum(p * q for p, q in depth.sell_orders.items())
            / sum(depth.sell_orders.values())
            if depth.sell_orders
            else 0
        )
        m_bid = (
            sum(p * q for p, q in depth.buy_orders.items())
            / sum(depth.buy_orders.values())
            if depth.buy_orders
            else 0
        )
        return (m_ask + m_bid) / 2 if (m_ask and m_bid) else m_ask or m_bid

    def update_market_data(self, product: str, state: TradingState) -> None:
        mid = self.mid_price(state.order_depths[product])
        if product == VOLCANIC_ROCK_VOUCHER_10000:
            self.rock10000_mid_prices.append(mid)
            self.rock10000_timestamps.append(state.timestamp)

    # ---------------- optimised sweeping ----------------
    def OrderOptimised(
        self, product: str, size: int, mode: str, state: TradingState
    ) -> List[Order]:
        orders, target = [], size
        depth = state.order_depths[product]
        if mode == "buy":
            for px in sorted(depth.sell_orders):
                take = min(target, abs(depth.sell_orders[px]))
                orders.append(Order(product, px, take))
                target -= take
                if target <= 0:
                    break
        elif mode == "sell":
            for px in sorted(depth.buy_orders, reverse=True):
                take = min(target, abs(depth.buy_orders[px]))
                orders.append(Order(product, px, -take))
                target -= take
                if target <= 0:
                    break
        return orders

    # ---------------- main entry point ------------------
    def run(self, state: TradingState):
        # ── EARLY‑EXIT ─ trade only when *no* bids exist anywhere ─────
        if any(od.buy_orders for od in state.order_depths.values()):
            empty = {p: [] for p in state.order_depths}
            logger.flush(state, empty, conversions=0, trader_data="")
            return empty, 0, ""
        # ──────────────────────────────────────────────────────────────

        result = {prod: [] for prod in state.order_depths}
        conversions, trader_data = 0, ""

        # reload any pickled data
        if state.traderData:
            try:
                trader_object = jsonpickle.decode(state.traderData)
            except Exception:
                trader_object = {}
        else:
            trader_object = {}

        # update internal positions
        self.position.update(state.position)

        # record latest mid‑prices
        for prod in state.order_depths:
            self.update_market_data(prod, state)

        # --------------- "copy Caesar" strategy ----------------
        if VOLCANIC_ROCK_VOUCHER_10000 in state.order_depths:
            product = VOLCANIC_ROCK_VOUCHER_10000
            pos = self.position.get(product, 0)
            limit = LIMIT[product]
            if product in state.market_trades:
                for tr in state.market_trades[product]:
                    if tr.buyer == "Caesar" and pos < limit:
                        buy_qty = min(limit - pos, tr.quantity)
                        result[product].append(Order(product, tr.price, buy_qty))
                        pos += buy_qty
                    elif tr.seller == "Caesar" and pos > -limit:
                        sell_qty = min(pos + limit, tr.quantity)
                        result[product].append(Order(product, tr.price, -sell_qty))
                        pos -= sell_qty

        # (other strategies were removed for brevity but can be re‑inserted)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    

#############################################################################################################################
"""



prosperity3bt "Level5/James/RaymondCaesarSalad2.py" 5 --no-out




"""
#############################################################################################################################
