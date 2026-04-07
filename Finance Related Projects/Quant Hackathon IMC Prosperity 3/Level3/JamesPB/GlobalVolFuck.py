################################################################################################################
###----------------------------------      Yes                         --------------------------------------###
################################################################################################################

import json
import jsonpickle
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque








################################################################################################################
###----------------------------------          Logger                  --------------------------------------###
################################################################################################################


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
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()






################################################################################################################
###----------------------------------             Defining             --------------------------------------###
################################################################################################################




PICNIC_BASKET1='PICNIC_BASKET1'
PICNIC_BASKET2='PICNIC_BASKET2'
JAMS='JAMS'
DJEMBES='DJEMBES'
CROISSANTS = 'CROISSANTS'


LIMIT = {
    
    PICNIC_BASKET1: 60,
    PICNIC_BASKET2: 100,
    CROISSANTS: 250,
    JAMS: 350,
    DJEMBES: 60,
}

croissanthistorylength = 83.8247
croissantzthreshold = 1.9986


PARAMS = {
    
 CROISSANTS: {
        "history_length": croissanthistorylength,  # Number of mid-price datapoints to use for z-score calculation.
        "z_threshold": croissantzthreshold        # Threshold for trading.
    }

}







################################################################################################################
###----------------------------------          Parameters              --------------------------------------###
################################################################################################################


### PB Synthetic Basket Parameters #######################
synth1Mean = -131.606  # PB1 is usually cheaper than PB2
synth1Sigma = np.round(29.05 // np.sqrt(1000), decimals=5)
s1Zscore = 0.3394  # TODO OPTIMISE!!!!!! 1 works pretty well
synth2Mean = 105.417
synth2Sigma = np.round(27.166 // np.sqrt(1000), decimals=5)
s2Zscore = 1 # 1.4417 # TODO OPTIMISE!!!!!!
diff_threshold_b1 = 176.8118 #176.8118
diff_threshold_b2 = 170 #needs to be lowerr than diff_threshold_b1. 




#### DYNAMIC INVENTORY FACTOR PARAMS#### TODO APPLY PARAMS TO ALL PRODUCTS
pb1high = 3.8116
pb1mid = 37.2380
pb1low = -9.0092
pb1neg = 7.9516

pb2high = 39.2463
pb2mid = 26.8949
pb2low = 5.3674
pb2neg = -26.1430

djembeshigh = 22.4723 #not being optimised
djembesmid = 16.7552 #not being optimised
djembeslow = -4.4401 #not being optimised 
djembesneg = -9.0611 #not being optimised

##############################################

#### HOLD FACTOR PARAMS ####
pb1Hold = 38.6961
pb2Hold = 3.9930
djembeHold = 1.7 #not being optimised
###############################################











################################################################################################################
###----------------------------------          General Functions       --------------------------------------###
################################################################################################################


## GENERAL FUNCTIONS #######################
def sortDict(dictionary:dict):
    return {key: dictionary[key] for key in sorted(dictionary)}

def VolumeCapability(product, mode,state:TradingState):
    if mode == "buy":
        return LIMIT[product] - state.position[product]
    if mode == "sell":
        return state.position[product] + LIMIT[product]

def vwap(product: str,state:TradingState) -> float:
    vwap = 0
    total_amt = 0

    for prc, amt in state.order_depths[product].buy_orders.items():
        vwap += prc * amt
        total_amt += amt

    for prc, amt in state.order_depths[product].sell_orders.items():
        vwap += prc * abs(amt)
        total_amt += abs(amt)

    vwap /= total_amt
    return np.round(vwap,decimals=5)

def mid_price(order_depth:OrderDepth) -> float:
    # Compute a mid-price
    if len(order_depth.sell_orders) != 0:
        m1 = 0
        n1 = 0
        for best_ask, best_ask_amount in order_depth.sell_orders.items():
            m1 += best_ask * best_ask_amount
            n1 += best_ask_amount
        m1 = m1 / n1
    else:
        m1 = 0

    if len(order_depth.buy_orders) != 0:
        m2 = 0
        n2 = 0
        for best_bid, best_bid_amount in order_depth.buy_orders.items():
            m2 += best_bid * best_bid_amount
            n2 += best_bid_amount
        m2 = m2 / n2
    else:
        m2 = 0
        
    # Use whichever is available, or the average if both are
    return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)

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
    if product not in set(state.order_depths.keys()):  # FREAKY
        return 100
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
    if product not in set(state.order_depths.keys()):
        return 100  # FREAKY
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


def get_best_asks_to_fill(
    product, state, LeastVolIWant):
    order_depth = state.order_depths[product]
    asks = order_depth.sell_orders  # Dict of price: volume

    # Sort ask prices from best (lowest) to worst (highest)
    sorted_asks = sorted(asks.items())  # List of (price, volume)

    volume = abs(volume)

    orders_to_place = []
    filled_volume = 0

    for price, volume in sorted_asks:
        if filled_volume >= LeastVolIWant:
            break
        volume_to_use = min(volume, LeastVolIWant - filled_volume)
        orders_to_place.append((price, volume_to_use))
        filled_volume += volume_to_use

    return orders_to_place


def get_best_bids_to_fill(product, state, LeastVolIWant):
    order_depth = state.order_depths[product]
    bids = order_depth.buy_orders  # Dict of price: volume

    # Sort bid prices from best (highest) to worst (lowest)
    sorted_bids = sorted(bids.items(), reverse=True)  # List of (price, volume)

    orders_to_place = []
    filled_volume = 0

    for price, volume in sorted_bids:
        if filled_volume >= LeastVolIWant:
            break
        volume_to_use = min(volume, LeastVolIWant - filled_volume)
        orders_to_place.append((price, volume_to_use))
        filled_volume += volume_to_use

    return orders_to_place


def get_best_asks_to_fill_WITH_LEVELS(product, state, LeastVolIWant):
    order_depth = state.order_depths[product]
    asks = order_depth.sell_orders  # price: negative volume (since it's the ask side)

    sorted_asks = sorted(asks.items())  # Lowest price first

    max_levels = 3
    orders_to_place = []
    filled_volume = 0

    for price, volume in sorted_asks[:max_levels]: 
        if filled_volume >= LeastVolIWant:
            break
        volume_to_use = min(volume, LeastVolIWant - filled_volume)
        orders_to_place.append((price, volume_to_use))
        filled_volume += volume_to_use

    return orders_to_place



def get_best_bids_to_fill_WITH_LEVELS(product, state, LeastVolIWant):
    order_depth = state.order_depths[product]
    bids = order_depth.buy_orders  # price: positive volume

    sorted_bids = sorted(bids.items(), reverse=True)  # Highest price first

    max_levels = 3
    orders_to_place = []
    filled_volume = 0

    for price, volume in sorted_bids[:max_levels]:
        if filled_volume >= LeastVolIWant:
            break
        volume_to_use = min(volume, LeastVolIWant - filled_volume)
        orders_to_place.append((price, volume_to_use))
        filled_volume += volume_to_use

    return orders_to_place



##############################################################











################################################################################################################
###----------------------------------            Class Trader          --------------------------------------###
################################################################################################################



class Trader: ##from here

    def __init__(self) -> None:
        self.logger = Logger()
        self.logger.print("Initialized trader for JAMS and DJEMBES only.")
        self.pos_limits = LIMIT
        self.diff_threshold_b1 = diff_threshold_b1
        self.diff_threshold_b2 = diff_threshold_b2
        self.lot_size = 1

        self.previousposition = {params:0 for params in LIMIT.keys()}
        self.position = {params:0 for params in LIMIT.keys()}
        self.positionCounter = {params:0 for params in LIMIT.keys()}


    def update_market_data(self, product, state):
        # Store current timestamp and mid-price for all assets
        order_depth = state.order_depths[product]
        mid = mid_price(order_depth)

    def UpdatePreviousPositionCounter(self,product,state:TradingState) -> None:
            if product not in set(state.position.keys()):
                return None
            if (
                state.position[product] == self.previousposition[product]
            ):  # Updates previouspositionCounter
                self.positionCounter[product] += 1
            else:
                self.positionCounter[product] = 0

    def PriceAdjustment(self,product, mode=None,state=TradingState):
        if product not in set(state.position.keys()):
            return 0
        AdjustmentDict = {
            PICNIC_BASKET1: {
                "high": pb1high,
                "mid": pb1mid,
                "low": pb1low,
                "neg": pb1neg,
            },
            PICNIC_BASKET2: {
                "high": pb2high,
                "mid": pb2mid,
                "low": pb2low,
                "neg": pb2neg,
            },
            DJEMBES: {
                "high": djembeshigh,
                "mid": djembesmid,
                "low": djembeslow,
                "neg": djembesneg,
            },
        }

        holdFactorDict = {
            PICNIC_BASKET1: pb1Hold,
            PICNIC_BASKET2: pb2Hold,
            DJEMBES: djembeHold,
        }

        VolumeFraction = VolumeCapability(product, mode=mode,state=state) / LIMIT[product]
        holdPremium = int(holdFactorDict[product] * self.positionCounter[product])
        high = AdjustmentDict[product]["high"] + holdPremium
        mid = AdjustmentDict[product]["mid"] + holdPremium
        low = AdjustmentDict[product]["low"] + holdPremium
        neg = AdjustmentDict[product]["neg"] + holdPremium

        if product == PICNIC_BASKET1 or product == PICNIC_BASKET2:
            if mode == "buy":
                factor = 1
            if mode == "sell":
                factor = -1
            if VolumeFraction <= 0.1:
                return int(
                    factor * (high + 3)
                )  # FOR SOME REASON PICNIC BASKET 1 LIKES THIS
            if VolumeFraction > 0.1 and VolumeFraction <= 0.2:
                return int(factor * high)
            if VolumeFraction > 0.2 and VolumeFraction < 0.5:
                return int(factor * mid)
            if VolumeFraction >= 0.5 and VolumeFraction < 1:
                return int(factor * low)
            if VolumeFraction >= 1:
                return int(factor * neg)

        if product == DJEMBES:
            if mode == "buy":
                factor = 1
            if mode == "sell":
                factor = -1
            if VolumeFraction <= 0.1:
                return int(factor * high)
            if VolumeFraction > 0.1 and VolumeFraction < 0.5:
                return int(factor * mid)
            if VolumeFraction >= 0.5 and VolumeFraction < 1:
                return int(factor * low)
            if VolumeFraction >= 1:
                return int(factor * neg)
            








################################################################################################################
###----------------------------------             Run Function         --------------------------------------###
################################################################################################################



    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in LIMIT.keys()}
        conversions = 0
        trader_data = ""
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        # Updating Position and Position Counters
        for product in state.position:
            self.UpdatePreviousPositionCounter(product,state)
            self.position[product] = state.position[product]
            self.update_market_data(product, state)
        
        # -------------------------------------------------------------------------










################################################################################################################
###----------------------------------          Order Logic             --------------------------------------###
################################################################################################################
        
        relevant = [
                    CROISSANTS,
                    JAMS,
                    DJEMBES,
                    PICNIC_BASKET1,
                    PICNIC_BASKET2
                ]
        best_bid = {}
        best_ask = {}
        mid_price = {}
        for prod in relevant:
            od = state.order_depths.get(prod)
            if od:
                best_bid[prod] = max(od.buy_orders.keys()) if od.buy_orders else None
                best_ask[prod] = min(od.sell_orders.keys()) if od.sell_orders else None
                if best_bid[prod] is not None and best_ask[prod] is not None:
                    mid_price[prod] = 0.5 * (best_bid[prod] + best_ask[prod])
                elif best_bid[prod] is not None:
                    mid_price[prod] = best_bid[prod]
                elif best_ask[prod] is not None:
                    mid_price[prod] = best_ask[prod]
                else:
                    mid_price[prod] = None
            else:
                best_bid[prod] = None
                best_ask[prod] = None
                mid_price[prod] = None

        # Compute the composition signal exactly as in the original:
        #    implied_BASKET1 = BASKET2 + 2 * CROISSANTS + 1 * JAMS + 1 * DJEMBES
        #    diff_comp = BASKET1 - implied_BASKET1
        if (mid_price[PICNIC_BASKET1] is not None and
            mid_price[PICNIC_BASKET2] is not None and
            mid_price[CROISSANTS] is not None and
            mid_price[JAMS] is not None and
            mid_price[DJEMBES] is not None):
            
            b1 = mid_price[PICNIC_BASKET1]
            b2 = mid_price[PICNIC_BASKET2]
            c = mid_price[CROISSANTS]
            j = mid_price[JAMS]
            d = mid_price[DJEMBES]

            implied_b1 = b2 + 2 * c + j + d
            implied_b2 = b1 - 2 * c - j - d 
            diff_comp1 = b1 - implied_b1
            diff_comp2 = b2 - implied_b2
            self.logger.print(f"Composition signal: BASKET1={b1:.1f}, Implied_BASKET1={implied_b1:.1f}, diff={diff_comp1:.1f}")
            self.logger.print(f"Composition signal: BASKET2={b2:.1f}, Implied_BASKET1={implied_b2:.1f}, diff={diff_comp2:.1f}")
            signal = 0
        else:
            self.logger.print("Insufficient market data to compute composition signal.")
            signal = 0



        IWantThisMuchFor1 = 50 # Orders that we want to spray on the best price and second best price. 
        IWantThisMuchFor2 = 50



        spread1 = (
                                vwap(PICNIC_BASKET1,state=state)  # FIXME MAYBE THE LOGIC IS WRONG
                                - 1.5 * vwap(PICNIC_BASKET2,state=state)
                                - vwap(DJEMBES,state=state)
                            )
        normspread1 = spread1 - synth1Mean
        spread2 = (
                vwap(PICNIC_BASKET2,state=state) - 4 * vwap(CROISSANTS,state=state) - 2 * vwap(JAMS,state=state)
            )
        normspread2 = spread2 - synth2Mean



        #=======PICNIC BASKET (WHY DOES IT SOUND LIKE CHICKEN JOCKEY)
        if PICNIC_BASKET1 in state.order_depths:

            # spread1 = (
            #                     vwap(PICNIC_BASKET1,state=state)  # FIXME MAYBE THE LOGIC IS WRONG
            #                     - 1.5 * vwap(PICNIC_BASKET2,state=state)
            #                     - vwap(DJEMBES,state=state)
            #                 )
            # normspread1 = spread1 - synth1Mean
            # spread2 = (
            #     vwap(PICNIC_BASKET2,state=state) - 4 * vwap(CROISSANTS,state=state) - 2 * vwap(JAMS,state=state)
            # )
            # normspread2 = spread2 - synth2Mean
            
            
            if (
                # normspread1
                # > synth1Sigma
                # * s1Zscore  ##Picnic Basket 1 is overvalued or PB2 OR Djembe is undervalued

                
                diff_comp1 < self.diff_threshold_b1

            ):  ## sell at the worst bid(cheapest) ##TODO maybe be more aggressive?
                # LowestBid = BidPrice(PICNIC_BASKET1, mode="min",state=state)
                # LowestVolume = BidVolume(PICNIC_BASKET1, mode="min",state=state)   
                orders_bid_b1 = []
                for price, volume in get_best_bids_to_fill_WITH_LEVELS(PICNIC_BASKET1, state, IWantThisMuchFor1):
                    result[PICNIC_BASKET1].append(Order(PICNIC_BASKET1, price , volume)) 
                # orders_bid_b2 = []
                # for price, volume in get_best_asks_to_fill_WITH_LEVELS(PICNIC_BASKET2, state, IWantThisMuchFor2):
                #     result[PICNIC_BASKET2].append(Order(PICNIC_BASKET2, price , volume))       
                        
            if (
                # normspread1
                # < -synth1Sigma
                # * s1Zscore  ##Picnic Basket 1 is undervalued or PB2 OR Djembe is overvalued

                 diff_comp1 > self.diff_threshold_b1

            ):  ## buy at the worst ask(most expensive) ##TODO maybe be more aggressive?
                # HighestAsk = AskPrice(PICNIC_BASKET1, mode="max",state=state)
                # HighestVolume = AskVolume(PICNIC_BASKET1, mode="max",state=state)
                orders_ask_b1 = []
                for price, volume in get_best_asks_to_fill_WITH_LEVELS(PICNIC_BASKET1, state, IWantThisMuchFor1):
                    result[PICNIC_BASKET1].append(Order(PICNIC_BASKET1, price, -volume)) 
                # orders_ask_b2 = []
                # for price, volume in get_best_bids_to_fill_WITH_LEVELS(PICNIC_BASKET2, state, IWantThisMuchFor2):
                #     result[PICNIC_BASKET2].append(Order(PICNIC_BASKET2, price, volume))    

            
        if PICNIC_BASKET2 in state.order_depths:  

            # if (
            #     normspread2 < synth2Sigma * s2Zscore
            # ):  # assume this means PB2 is overvalued
                    # orders_ask_b2 = []
                    # for price, volume in get_best_asks_to_fill_WITH_LEVELS(PICNIC_BASKET2, state, IWantThisMuchFor2):
                    #     result[PICNIC_BASKET2].append(Order(PICNIC_BASKET2, price, -volume))   

            # elif (
            #     normspread2 > synth2Sigma * s2Zscore
            # ):  # assume this means PB2 is the one that is overvalued
                    # orders_bid_b2 = []
                    # for price, volume in get_best_bids_to_fill_WITH_LEVELS(PICNIC_BASKET2, state, IWantThisMuchFor2):
                    #     result[PICNIC_BASKET2].append(Order(PICNIC_BASKET2, price , volume))        

            logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  

#         prosperity3bt "Level3/James/MeanJaymes/PB1+PB2.py" 3 --no-out

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  # 