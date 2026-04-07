import json
import jsonpickle
import numpy as np
import math
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

################################################################################################################
###----------------------------------            Logger                --------------------------------------###
################################################################################################################

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(
        self, order_depths: Dict[Symbol, OrderDepth]
    ) -> Dict[Symbol, List[Any]]:
        return {
            sym: [od.buy_orders, od.sell_orders]
            for sym, od in order_depths.items()
        }

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        out: List[List[Any]] = []
        for arr in trades.values():
            for t in arr:
                out.append(
                    [
                        t.symbol,
                        t.price,
                        t.quantity,
                        t.buyer,
                        t.seller,
                        t.timestamp,
                    ]
                )
        return out

    def compress_observations(self, obs: Observation) -> List[Any]:
        conv = {
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
        return [obs.plainValueObservations, conv]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        out: List[List[Any]] = []
        for arr in orders.values():
            for o in arr:
                out.append([o.symbol, o.price, o.quantity])
        return out

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()

################################################################################################################
###----------------------------------          Settings & Params      --------------------------------------###
################################################################################################################

JAMS = "JAMS"
LIMIT = {JAMS: 350}
ASPARAMS = {
    JAMS: {
        "gamma": 2.98,
        "sigma": 55.53,
        "k": 4.37,
        "max_order_size": 5,
        "T": 1.0,
        "limit": 350,
        "buffer": 2,
    }
}

################################################################################################################
###----------------------------------         General Functions        --------------------------------------###
################################################################################################################

def avellaneda_stoikov_delta(product: str, mid: float, inventory: int) -> float:
    p = ASPARAMS[product]
    reservation = mid - inventory * p["gamma"] * (p["sigma"] ** 2) * p["T"]
    spread = (2 / p["gamma"]) * math.log(1 + p["gamma"] / p["k"])
    return abs(reservation - mid) + spread / 2

################################################################################################################
###----------------------------------            Trader Class          --------------------------------------###
################################################################################################################

class Trader:
    def __init__(self) -> None:
        self.lot_size = 10
        self.logger = Logger()
        self.pos_limits = LIMIT
        self.position = {params:0 for params in LIMIT.keys()}


    
    def mid_price(self, order_depth: OrderDepth) -> float:
        def weighted_avg(d: Dict[float, int]) -> float:
            num = sum(p * v for p, v in d.items())
            den = sum(v for v in d.values())
            return num / den if den else 0.0

        m1 = weighted_avg(order_depth.sell_orders)
        m2 = weighted_avg(order_depth.buy_orders)
        if m1 and m2:
            return (m1 + m2) / 2
        return m1 or m2

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        conversions = 0
        result = {JAMS: []}
        logger.logs = ""
        trader_data = ""

        # current mid‑price for JAMS
        od = state.order_depths[JAMS]
        mid = self.mid_price(od)
        logger.print(f"Mid price: {mid:.2f}")

        # compute AS‑spread delta
        delta = avellaneda_stoikov_delta(JAMS, mid, self.position[JAMS])
        logger.print(f"AS delta: {delta:.2f}")

        # quotes at mid ± delta
        bid_price = mid - delta
        ask_price = mid + delta

        # respect position limits
        buy_size  = min(self.lot_size, LIMIT[JAMS] - self.position[JAMS])
        sell_size = min(self.lot_size, LIMIT[JAMS] + self.position[JAMS])
        if JAMS in state.order_depths:
            if buy_size > 0:
                result[JAMS].append(Order(JAMS, round(bid_price), +buy_size))
                
            if sell_size > 0:
                result[JAMS].append(Order(JAMS, round(ask_price), -sell_size))
                

        # flush logs, no conversions
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  

#         prosperity3bt "Level5/James/JamJames1.py" 5 --no-out

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  # 
