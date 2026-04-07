import json
from typing import Any
import numpy as np
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
        self.InkReturns = []
        self.KelpReturns = []
        self.m = None
        self.c = None
        self.mean = None
        self.stddev = None
        self.lookbackwindow = 10
        self.previousInkprice = None
        self.previousKelpprice = None

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        def sortDict(dictionary):
            return {key: dictionary[key] for key in sorted(dictionary)}

        def MultipleCheck(m, n):
            return m % n == 0

        def current_price(bid: dict, ask: dict):
            return np.round((max(bid.keys()) + min(ask.keys())) / 2, decimals=1)

        def calibrateM(kelp_returns, ink_returns):
            a = np.vstack([kelp_returns, np.ones(len(ink_returns))]).T
            m, c = np.linalg.lstsq(a, ink_returns, rcond=None)[0]
            return m, c

        def BasketAnalyse(coeff, const, kelp_returns, ink_returns):
            basket = np.array(ink_returns) - coeff * np.array(kelp_returns) - const
            return np.round(np.mean(basket), decimals=5), np.round(np.var(basket), decimals=5)

        result = {}
        kelporder = []
        inkorder = []
        conversions = 0
        trader_data = ""
        OrderbookDict = state.order_depths

        for product in OrderbookDict:
            if product in ["KELP", "SQUID_INK"]:
                OrderbookDepth = OrderbookDict[product]
                bidSpread = sortDict(OrderbookDepth.buy_orders)
                askSpread = sortDict(OrderbookDepth.sell_orders)
                currentPrice = current_price(bidSpread, askSpread)

                if product == "SQUID_INK":
                    if self.previousInkprice is not None:
                        ink_return = np.log(currentPrice) - np.log(self.previousInkprice)
                        self.InkReturns.append(ink_return)
                        if len(self.InkReturns) > self.lookbackwindow:
                            self.InkReturns = self.InkReturns[-self.lookbackwindow:]
                    self.previousInkprice = currentPrice
                    CheapestInkPrice = min(askSpread.keys())
                    HighestInkPrice = max(bidSpread.keys())

                if product == "KELP":
                    if self.previousKelpprice is not None:
                        kelp_return = np.log(currentPrice) - np.log(self.previousKelpprice)
                        self.KelpReturns.append(kelp_return)
                        if len(self.KelpReturns) > self.lookbackwindow:
                            self.KelpReturns = self.KelpReturns[-self.lookbackwindow:]
                    self.previousKelpprice = currentPrice
                    CheapestKelpPrice = min(askSpread.keys())
                    HighestKelpPrice = max(bidSpread.keys())

        if len(self.InkReturns) == self.lookbackwindow and len(self.KelpReturns) == self.lookbackwindow:
            if MultipleCheck(state.timestamp, 100 * self.lookbackwindow):
                self.m, self.c = calibrateM(self.KelpReturns, self.InkReturns)
                self.mean, var = BasketAnalyse(self.m, self.c, self.KelpReturns, self.InkReturns)
                self.stddev = np.round(np.sqrt(var), 5)

            latest_ink_return = self.InkReturns[-1]
            latest_kelp_return = self.KelpReturns[-1]
            X = latest_ink_return - self.c - self.m * latest_kelp_return
            zscore = np.round((X - self.mean) / self.stddev, decimals=5)

            if zscore >= 1.5:  # Sell ink, buy kelp
                inkorder.append(Order("SQUID_INK", HighestInkPrice, -OrderbookDict["SQUID_INK"].buy_orders[HighestInkPrice]))
                kelporder.append(Order("KELP", CheapestKelpPrice, OrderbookDict["KELP"].sell_orders[CheapestKelpPrice]))
            elif zscore <= -1.5:  # Buy ink, sell kelp
                inkorder.append(Order("SQUID_INK", CheapestInkPrice, OrderbookDict["SQUID_INK"].sell_orders[CheapestInkPrice]))
                kelporder.append(Order("KELP", HighestKelpPrice, -OrderbookDict["KELP"].buy_orders[HighestKelpPrice]))

        result["SQUID_INK"] = inkorder
        result["KELP"] = kelporder
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
