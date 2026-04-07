########## P1-1.5P2-D=S
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
        self.InkMeanState = None
        self.InkPreviousPriceArr = []
        self.KelpPreviousPriceArr = []
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
        self.m = None
        self.c = None
        self.mean = None
        self.stddev = None
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
            "DJEMBES": 60,
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
            holdFactor = 1.7  ## TODO OPTIMISE
            holdPremium = int(holdFactor * self.previouspositionCounter[product])
            if product not in set(state.position.keys()):
                return 0
            VolumeFraction = (
                VolumeCapability(product, mode=mode) / positionlimit[product]
            )
            if product == "PICNIC_BASKET1":
                PB1_high = 38.0160 + holdPremium  # TODO OPTIMISE
                PB1_mid = 32.7509 + holdPremium  # TODO OPTIMISE
                PB1_low = -8.9425 + holdPremium  # TODO OPTIMISE
                PB1_neg = -33.6974 + holdPremium  # TODO OPTIMISE
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
                PB2_high = 4.7662 + holdPremium  # TODO OPTIMISE
                PB2_mid = 3.6735 + holdPremium  # TODO OPTIMISE
                PB2_low = 1.7025 + holdPremium  # TODO OPTIMISE
                PB2_neg = 0.7633 + holdPremium  # TODO OPTIMISE
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
                DJ_high = 35.2641 + holdPremium  # TODO OPTIMISE
                DJ_mid = 16.2827 + holdPremium  # TODO OPTIMISE
                DJ_low = -6.7623 + holdPremium  # TODO OPTIMISE
                DJ_neg = -8.3677 + holdPremium  # TODO OPTIMISE
                if mode == "buy":
                    factor = 1
                if mode == "sell":
                    factor = -1
                if VolumeFraction <= 0.1:
                    return factor * DJ_high
                if VolumeFraction > 0.1 and VolumeFraction < 0.5:
                    return factor * DJ_mid
                if VolumeFraction >= 0.5 and VolumeFraction < 1:
                    return factor * DJ_low
                if VolumeFraction >= 1:
                    return factor * DJ_neg

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
        s1offset = -131.606  # PB1 is usually cheaper than PB2
        stdev1 = np.round(29.05 // np.sqrt(1000), decimals=5)
        zfactor1 = 0.3093  # TODO OPTIMISE!!!!!! 1 works pretty well
        s2offset = 105.417
        stdev2 = np.round(27.166 // np.sqrt(1000), decimals=5)
        zfactor2 = 1.5261  # TODO OPTIMISE!!!!!!
        # MAIN CHAIN OF LOGIC ##################################################

        for product in OrderbookDict:
            UpdatePreviousPositionCounter(product)
            if product == "PICNIC_BASKET1":  # 6 croissants 3 jams 1 djembe
                ##############################
                spread1 = (
                    vwap("PICNIC_BASKET1")  # FIXME MAYBE THE LOGIC IS WRONG
                    - 1.5 * vwap("PICNIC_BASKET2")
                    - vwap("DJEMBES")
                )
                normspread1 = spread1 - s1offset
                spread2 = (
                    vwap("PICNIC_BASKET2") - 4 * vwap("CROISSANTS") - 2 * vwap("JAMS")
                )
                normspread2 = spread2 - s2offset
                if (
                    normspread1
                    > stdev1
                    * zfactor1  ##Picnic Basket 1 is overvalued or PB2 OR Djembe is undervalued
                ):  ## sell at the worst bid(cheapest) ##TODO maybe be more aggressive?
                    HighestBid = BidPrice("PICNIC_BASKET1", mode="max")
                    HighestVolume = BidVolume("PICNIC_BASKET1", mode="max")
                    p1order.append(
                        Order(
                            "PICNIC_BASKET1",
                            HighestBid + PriceAdjustment("PICNIC_BASKET1", mode="sell"),
                            -HighestVolume,
                        )
                    )
                    if (
                        normspread2 > stdev2 * zfactor2
                    ):  # assume this means PB2 is overvalued
                        p2order.append(
                            Order(
                                "PICNIC_BASKET2",
                                AskPrice("PICNIC_BASKET2", mode="min")
                                + PriceAdjustment("PICNIC_BASKET2", mode="buy"),
                                AskVolume("PICNIC_BASKET2", mode="min"),
                            )
                        )
                    else:  # DJEMBE is undervalued
                        djembeorder.append(
                            Order(
                                "DJEMBES",
                                AskPrice("DJEMBES", mode="min")
                                + PriceAdjustment("DJEMBES", mode="buy"),
                                AskVolume("DJEMBES", mode="min"),
                            )
                        )

                if (
                    normspread1
                    < -stdev1
                    * zfactor1  ##Picnic Basket 1 is undervalued or PB2 OR Djembe is overvalued
                ):  ## buy at the worst ask(most expensive) ##TODO maybe be more aggressive?
                    CheapestAsk = AskPrice("PICNIC_BASKET1", mode="min")
                    CheapestVolume = AskVolume("PICNIC_BASKET1", mode="min")
                    p1order.append(
                        Order(
                            "PICNIC_BASKET1",
                            CheapestAsk + PriceAdjustment("PICNIC_BASKET1", mode="buy"),
                            CheapestVolume,
                        )
                    )
                    if (
                        normspread2 > stdev2 * zfactor2
                    ):  # assume this means PB2 is the one that is overvalued
                        p2order.append(
                            Order(
                                "PICNIC_BASKET2",
                                BidPrice("PICNIC_BASKET2", mode="max")
                                + PriceAdjustment("PICNIC_BASKET2", mode="sell"),
                                -BidVolume("PICNIC_BASKET2", mode="max"),
                            )
                        )
                    else:  # Djembe is overvalued
                        djembeorder.append(
                            Order(
                                "DJEMBES",
                                BidPrice("DJEMBES", mode="max")
                                + PriceAdjustment("DJEMBES", mode="sell"),
                                -BidVolume("DJEMBES", mode="max"),
                            )
                        )
        UpdatePreviousPosition(state)
        logger.print("Spread 1 average: " + str(np.mean(self.spread1arr)))
        logger.print("Spread 1 stddev: " + str(np.sqrt(np.var(self.spread1arr))))
        logger.print("Spread 2 average: " + str(np.mean(self.spread2arr)))
        logger.print("Spread 2 stddev: " + str(np.sqrt(np.var(self.spread2arr))))
        result["PICNIC_BASKET1"] = p1order
        result["PICNIC_BASKET2"] = p2order
        result["CROISSANTS"] = croissantorder
        result["JAMS"] = jamorder
        result["DJEMBES"] = djembeorder
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
