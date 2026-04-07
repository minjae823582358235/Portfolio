import json
from typing import Any
import numpy as np
from NaiveCompare.datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


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


class Trader:
    def __init__(self):
        self.InkMeanState = None
        self.InkPreviousPriceArr = []
        self.KelpPreviousPriceArr = []
        self.m = None
        self.c = None
        self.mean = None
        self.stddev = None
        self.lookbackwindow = 5
        self.previousInkprice = None
        self.previousKelpprice = None
        self.spread1arr = []
        self.spread2arr = []
        return None

    # def previous_price(product:str,state:TradingState):
    #     MarkettradeHistory=state.market_trades[product]
    #     previousTimestepMarketTrade=[trade for trade in MarkettradeHistory if trade.timestamp==int(state.timestamp-100)]
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        positionlimit = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }

        def vwap(product: str) -> float:
            vwap = 0
            total_amt = 0

            for prc, amt in state.order_depths[product].buy_orders.items():
                vwap += prc * amt
                total_amt += amt

            for prc, amt in state.order_depths[product].sell_orders.items():
                vwap += prc * abs(amt)
                total_amt += abs(amt)

            vwap /= total_amt
            return vwap

        def mid_price(product):
            orderbook = state.order_depths[product]
            bid = orderbook.buy_orders
            ask = orderbook.sell_orders
            return np.round(
                (max(set(bid.keys())) + min(set(ask.keys()))) / 2, decimals=5
            )

        def VolumeCapability(product, mode=None):
            if mode == "buy":
                return positionlimit[product] - state.position[product]
            if mode == "sell":
                return state.position[product] + positionlimit[product]

        def AskPrice(product, mode=None):
            if mode == "max":
                return max(set(state.order_depths[product].buy_orders.keys()))
            if mode == "min":
                return min(set(state.order_depths[product].buy_orders.keys()))

        def BidPrice(product, mode=None):
            if mode == "max":
                return max(set(state.order_depths[product].sell_orders.keys()))
            if mode == "min":
                return min(set(state.order_depths[product].sell_orders.keys()))

        def AskVolume(product, mode=None):
            if mode == "max":
                return state.order_depths[product].buy_orders[
                    AskPrice(product, mode="max")
                ]
            if mode == "min":
                return state.order_depths[product].buy_orders[
                    AskPrice(product, mode="min")
                ]

        def BidVolume(product, mode=None):
            if mode == "max":
                return state.order_depths[product].sell_orders[
                    BidPrice(product, mode="max")
                ]
            if mode == "min":
                return state.order_depths[product].sell_orders[
                    BidPrice(product, mode="min")
                ]

        result = {}
        kelporder = []
        inkorder = []
        resinorder = []
        croissantorder = []
        jamorder = []
        djembeorder = []
        p1order = []
        p2order = []
        offset = 0
        conversions = 0
        trader_data = ""
        OrderbookDict = state.order_depths
        # TODO: Add logic
        for product in OrderbookDict:
            if product == "PICNIC_BASKET1":  # 6 croissants 3 jams 1 djembe
                s1offset = 26.51911544355958
                stdev1 = 27.060413698355 // np.sqrt(1000)
                zfactor1 = 2
                ##############################
                spread1 = (
                    vwap("PICNIC_BASKET1")
                    - 6 * vwap("CROISSANTS")
                    - 3 * vwap("JAMS")
                    - vwap("DJEMBES")
                )
                normspread1 = spread1 - s1offset
                self.spread1arr.append(spread1)
                if (
                    normspread1 > stdev1 * zfactor1
                ):  ## sell at the worst bid(cheapest) ##TODO maybe be more aggressive?
                    cheapestbid = min(
                        set(state.order_depths[product].sell_orders.keys())
                    )
                    cheapestVolume = state.order_depths[product].sell_orders[
                        cheapestbid
                    ]
                    p1order.append(
                        Order(
                            "PICNIC_BASKET1",
                            cheapestbid,
                            -cheapestVolume,
                        )
                    )
                    croissantorder.append(
                        Order(
                            "CROISSANTS",
                            AskPrice("CROISSANTS", mode="min"),
                            AskVolume("CROISSANTS", mode="min"),
                        )
                    )
                    jamorder.append(
                        Order(
                            "JAMS",
                            AskPrice("JAMS", mode="min"),
                            AskVolume("JAMS", mode="min"),
                        )
                    )
                    djembeorder.append(
                        Order(
                            "DJEMBES",
                            AskPrice("DJEMBES", mode="min"),
                            AskVolume("DJEMBES", mode="min"),
                        )
                    )
                if (
                    normspread1 < -stdev1 * zfactor1
                ):  ## buy at the worst ask(most expensive) ##TODO maybe be more aggressive?
                    expensiveask = max(
                        set(state.order_depths[product].buy_orders.keys())
                    )
                    expensiveVolume = state.order_depths[product].buy_orders[
                        expensiveask
                    ]
                    p1order.append(
                        Order(
                            "PICNIC_BASKET1",
                            expensiveask,
                            expensiveVolume,
                        )
                    )
                    croissantorder.append(
                        Order(
                            "CROISSANTS",
                            BidPrice("CROISSANTS", mode="min"),
                            -BidVolume("CROISSANTS", mode="min"),
                        )
                    )
                    jamorder.append(
                        Order(
                            "JAMS",
                            BidPrice("JAMS", mode="min"),
                            -BidVolume("JAMS", mode="min"),
                        )
                    )
                    djembeorder.append(
                        Order(
                            "DJEMBES",
                            BidPrice("DJEMBES", mode="min"),
                            -BidVolume("DJEMBES", mode="min"),
                        )
                    )
            if product == "PICNIC_BASKET2":  # 4 croissants 2 jams
                s2offset = 105.41726
                stdev2 = 27.1663 / np.sqrt(1000)
                zfactor2 = 2
                ##############################
                spread2 = (
                    vwap("PICNIC_BASKET2") - 4 * vwap("CROISSANTS") - 2 * vwap("JAMS")
                )
                normspread2 = spread2 - s2offset
                self.spread2arr.append(spread2)
                if (
                    normspread2 > stdev2 * zfactor2
                ):  ## sell at the worst bid(cheapest) ##TODO maybe be more aggressive?
                    cheapestbid = min(
                        set(state.order_depths[product].sell_orders.keys())
                    )
                    cheapestVolume = state.order_depths[product].sell_orders[
                        cheapestbid
                    ]
                    p1order.append(
                        Order(
                            "PICNIC_BASKET2",
                            cheapestbid,
                            -cheapestVolume,
                        )
                    )
                if (
                    normspread2 < -stdev2 * zfactor2
                ):  ## buy at the worst ask(most expensive) ##TODO maybe be more aggressive?
                    expensiveask = max(
                        set(state.order_depths[product].buy_orders.keys())
                    )
                    expensiveVolume = state.order_depths[product].buy_orders[
                        expensiveask
                    ]
                    p2order.append(
                        Order(
                            "PICNIC_BASKET2",
                            expensiveask,
                            expensiveVolume,
                        )
                    )
        logger.print("Spread 1 average: " + str(np.mean(self.spread1arr)))
        logger.print("Spread 1 stddev: " + str(np.sqrt(np.var(self.spread1arr))))
        logger.print("Spread 2 average: " + str(np.mean(self.spread2arr)))
        logger.print("Spread 2 stddev: " + str(np.sqrt(np.var(self.spread2arr))))
        logger.flush(state, result, conversions, trader_data)
        result["PICNIC_BASKET1"] = p1order
        result["PICNIC_BASKET2"] = p2order
        result["CROISSANTS"] = croissantorder
        result["JAMS"] = jamorder
        result["DJEMBES"] = djembeorder
        return result, conversions, trader_data
PicnicandUnderlying.py
15 KB