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
        nan = np.nan
        self.InkMeanState = None
        self.InkSmaArr = []
        self.InkLmaArr = []
        self.meanMatrix = [
            [
                nan,
                nan,
                nan,
                -0.001013,
                0.002865,
                nan,
                0.00017,
                0.005352,
                0.000894,
                0.000125,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                -0.002744,
                0.000507,
                -0.002302,
                0.0,
                0.002754,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                -0.000759,
                -0.003776,
                0.004229,
                -0.000252,
                -0.002239,
                0.009392,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                0.000245,
                -0.006816,
                0.002132,
                0.002243,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                -0.001013,
                0.000254,
                0.001504,
                -0.001011,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                -0.00038,
                0.000338,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                -0.001511,
                0.001262,
                1e-06,
                -0.000138,
                -8.3e-05,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                -0.00051,
                0.000601,
                -0.000907,
                0.000164,
                0.000382,
                -0.001011,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                0.004913,
                0.000631,
                0.000423,
                3e-06,
                0.000319,
                -0.002352,
                nan,
                0.003251,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                0.002021,
                -0.001013,
                0.000334,
                0.000795,
                -0.000396,
                0.000563,
                0.000507,
                -0.000321,
                -0.000759,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                0.001504,
                0.001091,
                0.000337,
                0.000195,
                0.000803,
                -0.000172,
                -0.000695,
                -0.002879,
                -0.006189,
                -0.000254,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                0.001504,
                0.0,
                4e-06,
                -0.000388,
                0.000202,
                0.000501,
                -0.000555,
                -0.000756,
                0.000586,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                0.001014,
                0.000127,
                0.000195,
                0.000351,
                0.000621,
                0.000208,
                0.000196,
                0.000719,
                -0.000256,
                0.001089,
                0.0,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                0.0,
                0.000107,
                0.000841,
                0.000941,
                0.000143,
                4.1e-05,
                0.000118,
                0.000314,
                -0.000352,
                -0.001084,
                -0.000494,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                0.000845,
                0.000508,
                0.000791,
                0.000641,
                0.000145,
                0.000316,
                0.000245,
                7.3e-05,
                -0.000147,
                -0.000219,
                -0.00038,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                -0.000255,
                0.00022,
                0.000423,
                0.000444,
                0.000122,
                0.000309,
                0.000341,
                0.00017,
                -0.000148,
                -0.000522,
                -0.000506,
                -0.002008,
                nan,
                nan,
            ],
            [
                nan,
                0.000423,
                0.000475,
                0.000409,
                0.000415,
                0.000189,
                0.000267,
                0.00013,
                0.000219,
                -0.000436,
                0.000254,
                -0.0,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                -0.000253,
                0.000253,
                0.000172,
                0.000272,
                0.000195,
                0.000351,
                0.000112,
                -1.4e-05,
                2.9e-05,
                0.000399,
                -0.000251,
                nan,
                0.001752,
                nan,
            ],
            [
                nan,
                0.0,
                5.2e-05,
                0.00036,
                0.000297,
                0.000119,
                0.000192,
                2.8e-05,
                5e-05,
                -0.000107,
                2.9e-05,
                1e-06,
                nan,
                -0.001136,
                nan,
            ],
            [
                nan,
                8.5e-05,
                0.000598,
                0.000325,
                0.000252,
                0.000178,
                0.000217,
                2.7e-05,
                4e-06,
                -0.000169,
                -3.7e-05,
                -0.000664,
                -0.000255,
                0.001014,
                nan,
            ],
            [
                0.0,
                -0.000254,
                -8.3e-05,
                6.9e-05,
                0.000171,
                3.5e-05,
                5.4e-05,
                -6e-06,
                -7.4e-05,
                -0.000207,
                -0.000186,
                -0.000243,
                -0.000504,
                0.000253,
                nan,
            ],
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
            [
                nan,
                -0.000501,
                -0.000759,
                0.001605,
                0.000162,
                0.00013,
                7.3e-05,
                -8.5e-05,
                -6.7e-05,
                -0.00019,
                -0.000183,
                -0.000317,
                -7e-05,
                -0.000451,
                0.0,
            ],
            [
                nan,
                nan,
                0.000663,
                0.001167,
                -9.9e-05,
                -7e-06,
                2e-05,
                -8.5e-05,
                -0.000111,
                -0.000339,
                -0.000255,
                -0.000152,
                -0.000473,
                -0.000254,
                nan,
            ],
            [
                nan,
                nan,
                0.000507,
                0.00075,
                2.8e-05,
                0.000189,
                2.4e-05,
                -0.000104,
                -0.000178,
                -0.00025,
                -0.000289,
                -0.000302,
                -0.000659,
                -0.000759,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                -2.7e-05,
                0.000157,
                -1e-05,
                -7e-05,
                -0.000131,
                -0.000285,
                -0.000441,
                -0.000343,
                -0.000318,
                -0.000367,
                -0.000356,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                -0.000759,
                0.000198,
                -2e-06,
                -7.1e-05,
                -0.000163,
                -0.000154,
                -0.000356,
                -0.000467,
                -0.000672,
                -0.000589,
                -0.001089,
                nan,
            ],
            [
                nan,
                0.002243,
                nan,
                nan,
                -1e-06,
                -0.000115,
                -2.9e-05,
                -0.000117,
                -7.5e-05,
                -0.000328,
                -0.000292,
                -0.000505,
                -0.000884,
                -0.001258,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                0.003824,
                0.001148,
                5.8e-05,
                -5.1e-05,
                -0.000296,
                -0.000127,
                -0.000598,
                3.1e-05,
                -0.000316,
                -0.001511,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                -0.000334,
                -0.000413,
                0.000986,
                5.3e-05,
                -0.000171,
                -0.000281,
                -0.00055,
                0.000374,
                -0.000506,
                -0.003776,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                0.000764,
                -0.000254,
                6.6e-05,
                0.000146,
                0.00019,
                -0.000146,
                -0.000421,
                -0.000527,
                -0.00017,
                -0.00051,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                -0.000506,
                0.000254,
                -8.3e-05,
                0.000254,
                0.000167,
                0.000214,
                -0.00095,
                -0.000629,
                -0.001003,
                -0.001013,
                nan,
                nan,
            ],
            [
                nan,
                -0.001511,
                nan,
                -0.000625,
                -0.003776,
                -0.00017,
                -0.001006,
                0.000202,
                -0.000666,
                -0.000381,
                -0.000379,
                0.000338,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                0.001504,
                nan,
                -0.000255,
                -0.00142,
                -0.000676,
                0.00029,
                -0.001108,
                -0.002744,
                nan,
                0.0,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                0.002021,
                0.002754,
                -0.000283,
                -0.000127,
                -0.000411,
                -0.000126,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                -0.000127,
                0.000381,
                1e-06,
                -0.001004,
                -0.000254,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                0.000509,
                -0.000162,
                -0.001601,
                0.000254,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                -0.002628,
                -0.003599,
                nan,
                -0.001748,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                0.002754,
                0.00233,
                nan,
                -0.001511,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                -0.001511,
                nan,
                0.001262,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                -0.002006,
                nan,
                -0.001258,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                0.00139,
                nan,
                nan,
                nan,
                -0.000254,
                nan,
                nan,
                nan,
            ],
            [
                nan,
                nan,
                nan,
                nan,
                0.008553,
                -4e-06,
                0.002405,
                0.000698,
                0.00162,
                0.000509,
                -0.004959,
                nan,
                0.002754,
                -0.000759,
                nan,
            ],
        ]
        self.inklines = [  ## atol plus minus 0.006
            -4.959e-03,
            -4.706e-03,
            -4.520e-03,
            -4.263e-03,
            -3.991e-03,
            -3.776e-03,
            -3.246e-03,
            -3.014e-03,
            -2.744e-03,
            -2.512e-03,
            -2.239e-03,
            -2.008e-03,
            -1.748e-03,
            -1.511e-03,
            -1.258e-03,
            -1.013e-03,
            -7.590e-04,
            -5.100e-04,
            -2.540e-04,
            0.000e00,
            3.000e-06,
            9.000e-06,
            2.540e-04,
            5.090e-04,
            7.600e-04,
            1.014e-03,
            1.262e-03,
            1.504e-03,
            1.736e-03,
            2.021e-03,
            2.243e-03,
            2.513e-03,
            2.754e-03,
            3.012e-03,
            3.251e-03,
            3.536e-03,
            3.743e-03,
            4.014e-03,
            4.229e-03,
            4.528e-03,
            4.685e-03,
            4.913e-03,
        ]
        self.kelplines = [
            0.000494,
            -0.000247,
            0.0,
            -0.000494,
            0.000247,
            0.000741,
            -0.000741,
            0.000989,
            -0.000988,
            -0.00123,
            0.00148,
            -0.00148,
            0.00124,
        ]
        self.previousInkprice = None
        self.previousKelpprice = None
        return None

    # def previous_price(product:str,state:TradingState):
    #     MarkettradeHistory=state.market_trades[product]
    #     previousTimestepMarketTrade=[trade for trade in MarkettradeHistory if trade.timestamp==int(state.timestamp-100)]
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

        result = {}
        kelporder = []
        inkorder = []
        resinorder = []
        conversions = 0
        trader_data = ""
        OrderbookDict = state.order_depths
        # TODO: Add logic
        for product in OrderbookDict:
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
                    meanReturn = self.meanMatrix[InkMatrixLoc][KelpMatrixLoc]
                    if meanReturn == np.nan:
                        meanReturn = 0
                    tplus1InkPrice = meanReturn * CurrentInkPrice + CurrentInkPrice
                    if self.InkMeanState == "BULL":
                        if tplus1InkPrice > CurrentInkPrice:  # bullish
                            CheapestPrice = min(InkaskSpread.keys())
                            inkorder.append(Order("SQUID_INK", CheapestPrice, 1))
                        if tplus1InkPrice < CurrentInkPrice:  # bearish
                            HighestPrice = max(InkbidSpread.keys())
                            inkorder.append(Order("SQUID_INK", HighestPrice, +1))
                        if tplus1InkPrice == CurrentInkPrice:  # Neutral
                            inkorder.append(
                                Order("SQUID_INK", CurrentInkPrice - 1, 0)
                            )
                            inkorder.append(Order("SQUID_INK", CurrentInkPrice + 1, 1))
                    if self.InkMeanState == "BEAR":
                        if tplus1InkPrice > CurrentInkPrice:  # bullish
                            CheapestPrice = min(InkaskSpread.keys())
                            inkorder.append(Order("SQUID_INK", CheapestPrice, -1))
                        if tplus1InkPrice < CurrentInkPrice:  # bearish
                            HighestPrice = max(InkbidSpread.keys())
                            inkorder.append(Order("SQUID_INK", HighestPrice, 1))
                        if tplus1InkPrice == CurrentInkPrice:  # Neutral
                            inkorder.append(
                                Order("SQUID_INK", CurrentInkPrice - 1, 0)
                            )
                            inkorder.append(Order("SQUID_INK", CurrentInkPrice + 1, 15))
                    else:
                        continue
            if product == "KELP":
                pass
        result["SQUID_INK"] = inkorder
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
