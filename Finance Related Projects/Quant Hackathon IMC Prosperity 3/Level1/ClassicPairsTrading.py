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

import os

# Read parameters from environment variables with defaults if not set

# Resin #TODO jansdcljkasd
gamma = float(os.environ.get("gamma", "0.1"))
sigma = float(os.environ.get("sigma", "2"))
k = float(os.environ.get("k", "1.5"))
max_order_AS_size = int(os.environ.get("max_order_AS_size", "5"))
buffer = int(os.environ.get("buffer", "1"))


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
        self.lookbackwindow = 28.2097
        self.gamma = 2.3430  # Risk aversion parameter
        self.sigma = 1.6219  # Volatility parameter
        self.k = 3.5896  # Market order arrival parameter
        self.max_order_AS_size = 210.9072  # Maximum order size for market making
        self.buffer = 1.0564  # Buffer to avoid exceeding position limits
        self.T = 1.0  # Total trading horizon - stable so assume 1
        self.dt = 1.0
        self.previousposition = {
            "KELP": 0,
            "RAINFOREST_RESIN": 0,
            "SQUID_INK": 0,
            "CROISSANTS": 0,
            "JAMS": 0,
            "DJEMBES": 0,
            "PICNIC_BASKET1": 0,
            "PICNIC_BASKET2": 0,
        }
        self.previouspositionCounter = {
            "KELP": 0,
            "RAINFOREST_RESIN": 0,
            "SQUID_INK": 0,
            "CROISSANTS": 0,
            "JAMS": 0,
            "DJEMBES": 0,
            "PICNIC_BASKET1": 0,
            "PICNIC_BASKET2": 0,
        }
        return None

    # def previous_price(product:str,state:TradingState):
    #     MarkettradeHistory=state.market_trades[product]
    #     previousTimestepMarketTrade=[trade for trade in MarkettradeHistory if trade.timestamp==int(state.timestamp-100)]
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        lookbackwindow = 28.2097
        gamma = 2.3430
        sigma = 1.6219
        k = 3.5896
        max_order_AS_size = 210.9072
        buffer = 1.0564
        positionlimit = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }
        self.lookbackwindow = int(lookbackwindow)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.k = float(k)
        self.max_order_AS_size = int(max_order_AS_size)
        self.buffer = int(buffer)

        def sortDict(dictionary):
            return {key: dictionary[key] for key in sorted(dictionary)}

        def MultipleCheck(m, n):
            return True if m % n == 0 else False

        def current_price(bid: dict, ask: dict):
            return np.round(
                (max(set(bid.keys())) + min(set(ask.keys()))) / 2, decimals=1
            )

        def calibrateM(kelparr, inkarr):
            a = np.vstack([np.log(kelparr), np.ones(len(inkarr))]).T
            m, c = np.linalg.lstsq(a, np.log(inkarr), rcond=None)[0]
            return m, c

        def BasketAnalyse(coeff, const, kelp, ink):
            basket = np.array(ink) - coeff * np.array(kelp) - const
            return np.round(np.mean(basket), decimals=5), np.round(
                np.var(basket), decimals=5
            )

        def UpdatePreviousPosition(state) -> None:
            for product in set(state.position.keys()):
                if product not in set(self.previousposition.keys()):
                    self.previousposition[product] = 0
                if state.position[product] != self.previousposition[product]:
                    self.previousposition[product] = state.position[product]

        def UpdatePreviousPositionCounter(product) -> None:
            if product not in set(state.position.keys()):
                return None
            if (
                state.position[product] == self.previousposition[product]
            ):  # Updates previouspositionCounter
                self.previouspositionCounter[product] += 1
            else:
                self.previouspositionCounter[product] = 0

        def VolumeCapability(product, mode=None):
            if mode == "buy":
                return positionlimit[product] - state.position[product]
            if mode == "sell":
                return state.position[product] + positionlimit[product]

        def AskPrice(product, mode=None):  # how much a seller is willing to sell for
            if mode == "max":
                if product not in set(state.order_depths.keys()):
                    return 0  # FREAKY
                return max(set(state.order_depths[product].sell_orders.keys()))
            if mode == "min":
                if product not in set(state.order_depths.keys()):
                    return 0  # FREAKY
                return min(set(state.order_depths[product].sell_orders.keys()))

        def BidPrice(product, mode=None):  # how much a buyer is willing to buy for
            if mode == "max":
                if product not in set(state.order_depths.keys()):  # FREAKY
                    return 1000000  # FREAKY
                return max(set(state.order_depths[product].buy_orders.keys()))
            if mode == "min":
                if product not in set(state.order_depths.keys()):  # FREAKY
                    return 1000000  # FREAKY
                return min(set(state.order_depths[product].buy_orders.keys()))

        def AskVolume(
            product, mode=None
        ):  # ITS FOR THE HIGHEST/LOWEST PRICE NOT VOLUME!!
            if product not in set(state.order_depths.keys()):  # FREAKY
                return 100
            if mode == "max":
                return abs(
                    state.order_depths[product].sell_orders[
                        AskPrice(product, mode="max")
                    ]
                )
            if mode == "min":
                return abs(
                    state.order_depths[product].sell_orders[
                        AskPrice(product, mode="min")
                    ]
                )

        def BidVolume(
            product, mode=None
        ):  # ITS FOR THE HIGHEST/LOWEST PRICE NOT VOLUME!!
            if product not in set(state.order_depths.keys()):
                return 100  # FREAKY
            if mode == "max":
                return abs(
                    state.order_depths[product].buy_orders[
                        BidPrice(product, mode="max")
                    ]
                )
            if mode == "min":
                return abs(
                    state.order_depths[product].buy_orders[
                        BidPrice(product, mode="min")
                    ]
                )

        def PriceAdjustment(product, mode=None):
            holdFactor = 14.8618  ## TODO OPTIMISE
            holdPremium = int(holdFactor * self.previouspositionCounter[product])
            if product not in set(state.position.keys()):
                return 0
            VolumeFraction = (
                VolumeCapability(product, mode=mode) / positionlimit[product]
            )
            if product == "PICNIC_BASKET1":
                PB1_high = 29.6905 + holdPremium  # TODO OPTIMISE
                PB1_mid = 26.0245 + holdPremium  # TODO OPTIMISE
                PB1_low = 0.6551 + holdPremium  # TODO OPTIMISE
                PB1_neg = -17.0629 + holdPremium  # TODO OPTIMISE
                if mode == "buy":
                    factor = 1
                if mode == "sell":
                    factor = -1
                if VolumeFraction <= 0.1:
                    return int(
                        factor * (PB1_high + 3)
                    )  # FOR SOME REASON PICNIC BASKET 1 LIKES THIS
                if VolumeFraction > 0.1 and VolumeFraction <= 0.2:
                    return int(factor * PB1_high)
                if VolumeFraction > 0.2 and VolumeFraction < 0.5:
                    return int(factor * PB1_mid)
                if VolumeFraction >= 0.5 and VolumeFraction < 1:
                    return int(factor * PB1_low)
                if VolumeFraction >= 1:
                    return int(factor * PB1_neg)
            if product == "PICNIC_BASKET2":
                PB2_high = 15.3932 + holdPremium  # TODO OPTIMISE
                PB2_mid = 9.5059 + holdPremium  # TODO OPTIMISE
                PB2_low = -2.1304 + holdPremium  # TODO OPTIMISE
                PB2_neg = -34.8136 + holdPremium  # TODO OPTIMISE
                if mode == "buy":
                    factor = 1
                if mode == "sell":
                    factor = -1
                if VolumeFraction <= 0.1:
                    return int(
                        factor * (PB2_high + 3)
                    )  # FOR SOME REASON PICNIC BASKET 1 LIKES THIS
                if VolumeFraction > 0.1 and VolumeFraction <= 0.2:
                    return int(factor * PB2_high)
                if VolumeFraction > 0.2 and VolumeFraction < 0.5:
                    return int(factor * PB2_mid)
                if VolumeFraction >= 0.5 and VolumeFraction < 1:
                    return int(factor * PB2_low)
                if VolumeFraction >= 1:
                    return int(factor * PB2_neg)

            if product == "DJEMBES":
                DJ_high = 22.4723 + holdPremium  # TODO OPTIMISE
                DJ_mid = 16.7552 + holdPremium  # TODO OPTIMISE
                DJ_low = -4.4401 + holdPremium  # TODO OPTIMISE
                DJ_neg = -9.0611 + holdPremium  # TODO OPTIMISE
                if mode == "buy":
                    factor = 1
                if mode == "sell":
                    factor = -1
                if VolumeFraction <= 0.1:
                    return int(factor * DJ_high)
                if VolumeFraction > 0.1 and VolumeFraction < 0.5:
                    return int(factor * DJ_mid)
                if VolumeFraction >= 0.5 and VolumeFraction < 1:
                    return int(factor * DJ_low)
                if VolumeFraction >= 1:
                    return int(factor * DJ_neg)
            if product == "SQUID_INK":
                S_high = 31.3431 + holdPremium  # TODO OPTIMISE
                S_mid = 30.5972 + holdPremium  # TODO OPTIMISE
                S_low = -7.0549 + holdPremium  # TODO OPTIMISE
                S_neg = -14.8013 + holdPremium  # TODO OPTIMISE
                if mode == "buy":
                    factor = 1
                if mode == "sell":
                    factor = -1
                if VolumeFraction <= 0.1:
                    return int(factor * S_high)
                if VolumeFraction > 0.1 and VolumeFraction < 0.5:
                    return int(factor * S_mid)
                if VolumeFraction >= 0.5 and VolumeFraction < 1:
                    return int(factor * S_low)
                if VolumeFraction >= 1:
                    return int(factor * S_neg)
            if product == "KELP":
                K_high = 24.0361 + holdPremium  # TODO OPTIMISE
                K_mid = 12.8753 + holdPremium  # TODO OPTIMISE
                K_low = 8.5749 + holdPremium  # TODO OPTIMISE
                K_neg = -7.6168 + holdPremium  # TODO OPTIMISE
                if mode == "buy":
                    factor = 1
                if mode == "sell":
                    factor = -1
                if VolumeFraction <= 0.1:
                    return int(factor * K_high)
                if VolumeFraction > 0.1 and VolumeFraction < 0.5:
                    return int(factor * K_mid)
                if VolumeFraction >= 0.5 and VolumeFraction < 1:
                    return int(factor * K_low)
                if VolumeFraction >= 1:
                    return int(factor * K_neg)

        def avellaneda_stoikov(product, mid, inventory):
            """
            Calculate bid and ask prices around the mid-price/reservation price whilst adjusting based on current inventory.
            """
            # Reservation price - adjust the mid price based on inventory risk
            reservation_price = mid - inventory * self.gamma * (self.sigma**2) * 1

            # Optimal spread - adjusted based on the risk aversion parameter and order intensity
            optimal_spread = (2 / self.gamma) * np.log(1 + self.gamma / self.k)

            # Set final bid and ask around the reservation price.
            bid = reservation_price - optimal_spread / 2
            ask = reservation_price + optimal_spread / 2

            # Dynamic order sizing - reduce order size if close to limit to avoid exceeding position limits
            max_order_size = self.max_order_AS_size
            buffer = 1.0564
            limit = 50
            order_size = max(1, min(max_order_size, (limit - abs(inventory)) // buffer))

            return [
                Order(product, int(round(bid)), order_size),
                Order(product, int(round(ask)), -order_size),
            ]

        result = {}
        kelporder = []
        inkorder = []
        resinorder = []
        conversions = 0
        trader_data = ""
        OrderbookDict = state.order_depths
        zscorethreshold = 0.9165
        # TODO: Add logic
        for product in OrderbookDict:
            UpdatePreviousPositionCounter(product)
            if product == "RAINFOREST_RESIN":
                continue
            if product == "SQUID_INK":
                InkOrderbookDepth = OrderbookDict[product]
                InkbidSpread = sortDict(InkOrderbookDepth.buy_orders)
                InkaskSpread = sortDict(InkOrderbookDepth.sell_orders)
                KelpOrderbookDepth = OrderbookDict["KELP"]
                KelpbidSpread = sortDict(KelpOrderbookDepth.buy_orders)
                KelpaskSpread = sortDict(KelpOrderbookDepth.sell_orders)
                CurrentInkPrice = current_price(InkbidSpread, InkaskSpread)
                CurrentKelpPrice = current_price(KelpbidSpread, KelpaskSpread)
                self.InkPreviousPriceArr.append(CurrentInkPrice)
                self.KelpPreviousPriceArr.append(CurrentKelpPrice)
                CheapestInkPrice = min(InkaskSpread.keys())
                CheapestKelpPrice = min(KelpaskSpread.keys())
                HighestInkPrice = max(InkbidSpread.keys())
                HighestKelpPrice = max(KelpbidSpread.keys())

                if (
                    len(self.InkPreviousPriceArr) > self.lookbackwindow
                ):  ## constantly adjusting the lookback window array
                    self.InkPreviousPriceArr = self.InkPreviousPriceArr[
                        1 : self.lookbackwindow + 1
                    ]
                if (
                    len(self.KelpPreviousPriceArr) > self.lookbackwindow
                ):  ## constantly adjusting the lookback window array
                    self.KelpPreviousPriceArr = self.KelpPreviousPriceArr[
                        1 : self.lookbackwindow + 1
                    ]
                if state.timestamp >= 100 * self.lookbackwindow:
                    continue  ##test
                threshold = 0.7
                if (
                    np.corrcoef(
                        np.log(self.KelpPreviousPriceArr),
                        np.log(self.InkPreviousPriceArr),
                    )[0, 1]
                    < threshold
                ):
                    ink_mid = CurrentInkPrice
                    ink_orders = avellaneda_stoikov(
                        "SQUID_INK", ink_mid, state.position["SQUID_INK"]
                    )
                    result["SQUID_INK"] = ink_orders
                else:
                    if MultipleCheck(state.timestamp, 100 * self.lookbackwindow):
                        # considers if we are in the lookback window area
                        self.m, self.c = calibrateM(
                            self.KelpPreviousPriceArr, self.InkPreviousPriceArr
                        )
                        self.mean, var = BasketAnalyse(
                            self.m,
                            self.c,
                            self.KelpPreviousPriceArr,
                            self.InkPreviousPriceArr,
                        )
                        self.stddev = np.round(np.sqrt(var), 5)
                    X = np.round(
                        np.log(CurrentInkPrice)
                        - self.c
                        - self.m * np.log(CurrentKelpPrice),
                        decimals=5,
                    )
                    zscore = np.round((X - self.mean) / self.stddev, decimals=5)
                    if zscore >= zscorethreshold:  # sell ink buy kelp
                        inkorder.append(
                            Order(
                                "SQUID_INK",
                                BidPrice("SQUID_INK", mode="max")
                                + PriceAdjustment("SQUID_INK", mode="sell"),
                                -InkbidSpread[HighestInkPrice],
                            )
                        )
                        kelporder.append(
                            Order(
                                "KELP",
                                AskPrice("KELP", mode="min")
                                + PriceAdjustment("KELP", mode="buy"),
                                KelpaskSpread[CheapestKelpPrice],
                            )
                        )
                    if zscore <= -zscorethreshold:  # buy ink sell kelp
                        inkorder.append(
                            Order(
                                "SQUID_INK",
                                AskPrice("SQUID_INK", mode="min")
                                + PriceAdjustment("SQUID_INK", mode="buy"),
                                InkaskSpread[CheapestInkPrice],
                            )
                        )
                        kelporder.append(
                            Order(
                                "KELP",
                                BidPrice("KELP", mode="max")
                                + PriceAdjustment("KELP", mode="sell"),
                                -KelpbidSpread[HighestKelpPrice],
                            )
                        )
            if product == "KELP":
                pass
        result["SQUID_INK"] = inkorder
        result["KELP"] = kelporder
        UpdatePreviousPosition(state)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
