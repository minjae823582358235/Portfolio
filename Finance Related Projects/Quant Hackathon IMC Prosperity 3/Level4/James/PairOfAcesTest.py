import json
import jsonpickle
import numpy as np
import math
from typing import Any, Dict, List, Tuple

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState




################################################################################################################
###----------------------------------            Logger                --------------------------------------###
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
###----------------------------------            Defining              --------------------------------------###
################################################################################################################


MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
ETARIFF='export_tariff'
ITARIFF='import_tariff'
SUGAR='sugar_price'
TRANSPORT='transport_fee'
SUNLIGHT='sunlight_index'

LIMIT = {

    MAGNIFICENT_MACARONS: 75, 

}

PARAMS = {
         
    MAGNIFICENT_MACARONS: {
            "z_threshold": 2,         # The z-score threshold
            "history_length": 50,      # Number of historical mid–prices to use
            "default_fair_method": "vwap_with_vol_filter", #NOT SURE ABOUT THIS ONE. SHOULD BE USED FOR SUGAR 
            "min_volume_filter": 20 #AlsoNOT SURE ABOUT THIS ONE. SHOULD BE USED FOR SUGAR 
            }

}


ASPARAMS={MAGNIFICENT_MACARONS: {
                "gamma": 1.3074794082080743,
                "sigma": 1.1391336142619657,
                "k": 2.65639863217165,
                "max_order_size": 10,
                "T": 1.0,
                "limit": 50,
                "buffer": 2,
            }}



################################################################################################################
###----------------------------------            Params                --------------------------------------###
################################################################################################################


#### Kelp Squink Pairs Trade Parameters
MARZscore = 0.190327

magmahigh = 26.244
magmamid = 3.57077
magmalow = 1.0171
magmaneg = -19.168886

magmaHold = 28.964






################################################################################################################
###----------------------------------         General Functions        --------------------------------------###
################################################################################################################
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
    

##############################################################












################################################################################################################
###----------------------------------         Strategy Functions       --------------------------------------###
################################################################################################################


# mid_price(MAGNIFICENT_MACARONS) = self.m * Sugma + currentresidul + self.c 


# min of last 45 timesteps ((MagmaMid - previousMagmaMid) -  (self.m * (sugma - previoussugma) + self.c)^2 

# current_residual = current_magma_mid - (self.m * sugma + self.c)

# z = currentresidual - self.mean / self.stddev 

# if z > + z_threshold: 
#     sell Magma

# if z < - zthreshold: 
#     buy Magma


def trade_magma_signal(
    self,
    current_magma_mid: float,
    current_sugma: float,
    magma_mid_history: list[float],
    sugma_history: list[float],
    z_threshold: float,
) -> tuple[str, float, float, float]:
    """
    Determine trade signal for MAGNIFICENT_MACARONS based on the regression model against Sugma.
    
    Args:
        current_magma_mid: Current mid-price for MAGNIFICENT_MACARONS.
        current_sugma: Current price (mid–price) for Sugma.
        magma_mid_history: List of previous MAGNIFICENT_MACARONS mid–prices (ideally the last 45 timesteps).
        sugma_history: List of previous Sugma prices (aligned with magma_mid_history).
        z_threshold: The threshold for the z–score at which to trigger trades.
        
    Returns:
        A tuple containing:
            - action: "buy", "sell", or "hold".
            - current_residual: The current residual value.
            - min_squared_diff: Minimum squared difference over the history (used for monitoring).
            - z: The computed z–score.
    """
    # Calculate the expected mid–price from the regression:
    expected_mid = self.m * current_sugma + self.c

    # According to the equation:
    # mid_price(MAGNIFICENT_MACARONS) = self.m * Sugma + current_residual + self.c
    # we solve for the current residual:
    current_residual = current_magma_mid - expected_mid

    # Compute the squared differences over the history (for each timestep difference)
    squared_diffs = []
    # Need at least two history points to compute a difference.
    if len(magma_mid_history) >= 2 and len(sugma_history) >= 2:
        # Zip the history in pairs: (previous, current) for both MAGMA and SUGMA
        for m_prev, m_curr, s_prev, s_curr in zip(
            magma_mid_history[:-1],
            magma_mid_history[1:],
            sugma_history[:-1],
            sugma_history[1:],
        ):
            # Compute the difference in MAGMA mid–price and the predicted difference from Sugma:
            # Note: The equation is interpreted as:  
            # ((MagmaMid - previousMagmaMid) - (self.m * (sugma - previoussugma) + self.c))**2
            diff = (m_curr - m_prev) - (self.m * (s_curr - s_prev) + self.c)
            squared_diffs.append(diff ** 2)
        min_squared_diff = min(squared_diffs)
    else:
        min_squared_diff = float('inf')  # or set to None if insufficient data

    # Compute the z–score for the current residual.
    # Here we assume self.stddev is nonzero; in production code you should check for zero.
    if self.stddev and self.stddev != 0:
        z = (current_residual - self.mean) / self.stddev
    else:
        z = 0  # fallback in case stddev is 0

    # Determine trade signal:
    if z > z_threshold:
        action = "sell"  # Signal: the residual is too high, so MAGNIFICENT_MACARONS is overpriced
    elif z < -z_threshold:
        action = "buy"   # Signal: the residual is too low, so MAGNIFICENT_MACARONS is underpriced
    else:
        action = "hold"  # No clear signal

    # For debugging/logging, one might print the computed values:
    # print(f"Expected mid: {expected_mid}, Residual: {current_residual}, z-score: {z}, Min Squared Diff: {min_squared_diff}")

    return action, current_residual, min_squared_diff, z


# def magma_fair_value(
#     order_depth: OrderDepth, method: str = "vwap_with_vol_filter", min_vol: int = 20
# ) -> float:
#     if method == "mid_price":
#         best_ask = min(order_depth.sell_orders.keys())
#         best_bid = max(order_depth.buy_orders.keys())
#         return (best_ask + best_bid) / 2
#     elif method == "mid_price_with_vol_filter":
#         sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
#         buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
#         if not sell_orders or not buy_orders:
#             best_ask = min(order_depth.sell_orders.keys())
#             best_bid = max(order_depth.buy_orders.keys())
#         else:
#             best_ask = min(sell_orders)
#             best_bid = max(buy_orders)
#         return (best_ask + best_bid) / 2
#     elif method == "vwap":
#         best_ask = min(order_depth.sell_orders.keys())
#         best_bid = max(order_depth.buy_orders.keys())
#         volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
#         if volume == 0:
#             return (best_ask + best_bid) / 2
#         return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
#     elif method == "vwap_with_vol_filter":
#         sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
#         buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
#         if not sell_orders or not buy_orders:
#             best_ask = min(order_depth.sell_orders.keys())
#             best_bid = max(order_depth.buy_orders.keys())
#             volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
#             if volume == 0:
#                 return (best_ask + best_bid) / 2
#             return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
#         else:
#             best_ask = min(sell_orders)
#             best_bid = max(buy_orders)
#             volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
#             if volume == 0:
#                 return (best_ask + best_bid) / 2
#             return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
#     else:
#         raise ValueError("Unknown fair value method specified.")

# def magma_take_orders(
#     order_depth: OrderDepth, fair_value: float, params: dict, position: int
# ) -> Tuple[List[Order], int, int]:
#     orders = []
#     buy_order_volume = 0
#     sell_order_volume = 0
#     if order_depth.sell_orders:
#         best_ask = min(order_depth.sell_orders.keys())
#         ask_amount = -order_depth.sell_orders[best_ask]
#         if best_ask <= fair_value - params["take_width"] and ask_amount <= 50:
#             quantity = min(ask_amount, params["position_limit"] - position)
#             if quantity > 0:
#                 orders.append(Order(MAGNIFICENT_MACARONS, int(best_ask), -quantity))
#                 buy_order_volume += quantity
#     if order_depth.buy_orders:
#         best_bid = max(order_depth.buy_orders.keys())
#         bid_amount = order_depth.buy_orders[best_bid]
#         if best_bid >= fair_value + params["take_width"] and bid_amount <= 50:
#             quantity = min(bid_amount, params["position_limit"] + position)
#             if quantity > 0:
#                 orders.append(Order(MAGNIFICENT_MACARONS, int(best_bid), +quantity))
#                 sell_order_volume += quantity
#     return orders, buy_order_volume, sell_order_volume

# def magma_clear_orders(
#     order_depth: OrderDepth,
#     position: int,
#     params: dict,
#     fair_value: float,
#     buy_order_volume: int,
#     sell_order_volume: int,
# ) -> Tuple[List[Order], int, int]:
#     orders = []
#     position_after_take = position + buy_order_volume - sell_order_volume
#     fair_for_bid = math.floor(fair_value)
#     fair_for_ask = math.ceil(fair_value)
#     buy_quantity = params["position_limit"] - (position + buy_order_volume)
#     sell_quantity = params["position_limit"] + (position - sell_order_volume)
#     if position_after_take > 0:
#         if fair_for_ask in order_depth.buy_orders:
#             clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
#             sent_quantity = min(sell_quantity, clear_quantity)
#             orders.append(Order(MAGNIFICENT_MACARONS, int(fair_for_ask), abs(sent_quantity)))
#             sell_order_volume += abs(sent_quantity)
#     if position_after_take < 0:
#         if fair_for_bid in order_depth.sell_orders:
#             clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
#             sent_quantity = min(buy_quantity, clear_quantity)
#             orders.append(Order(MAGNIFICENT_MACARONS, int(fair_for_bid), -abs(sent_quantity)))
#             buy_order_volume += abs(sent_quantity)
#     return orders, buy_order_volume, sell_order_volume

# def magma_make_orders(
#     order_depth: OrderDepth,
#     fair_value: float,
#     position: int,
#     params: dict,
#     buy_order_volume: int,
#     sell_order_volume: int,
# ) -> List[Order]:
#     orders = []
#     edge = params["spread_edge"]
#     aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + edge]
#     bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - edge]
#     baaf = min(aaf) if aaf else fair_value + edge + 1
#     bbbf = max(bbf) if bbf else fair_value - edge - 1
#     buy_quantity = params["position_limit"] - (position + buy_order_volume)
#     if buy_quantity > 0:
#         orders.append(Order(MAGNIFICENT_MACARONS, int(bbbf + 1), -buy_quantity))
#     sell_quantity = params["position_limit"] + (position - sell_order_volume)
#     if sell_quantity > 0:
#         orders.append(Order(MAGNIFICENT_MACARONS, int(baaf - 1), sell_quantity))
#     return orders










################################################################################################################
###----------------------------------            Trader                --------------------------------------###
################################################################################################################

class Trader:

    def __init__(self) -> None:
        self.logger = Logger()
        self.logger.print("Initialized trader for MACARONS only.")
        self.pos_limits = LIMIT
        self.lot_size = 1

        self.previousposition = {params:0 for params in LIMIT.keys()}
        self.position = {params:0 for params in LIMIT.keys()}
        self.positionCounter = {params:0 for params in LIMIT.keys()}

        self.magma_timestamps = []
        self.magma_mid_prices = []
        self.sugma_timestamps = []
        self.sugma_prices = []
        


        # pairs trading strategy for ink and kelp
        self.m = None
        self.c = None
        self.mean = None
        self.stddev = None
        self.lookbackwindow = 45
        
    def update_market_data(self, product, state):
        # Store current timestamp and mid-price for all assets
        order_depth = state.order_depths[product]
        mid = mid_price(order_depth)
        if product == MAGNIFICENT_MACARONS:
            self.magma_timestamps.append(state.timestamp)
            self.magma_mid_prices.append(mid)
        elif product == SUGAR:
            self.sugma_timestamps.append(state.timestamp)
            self.sugma_prices.append(mid)

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

            MAGNIFICENT_MACARONS: {
                "high": magmahigh,
                "mid": magmamid,
                "low": magmalow,
                "neg": magmaneg,
            
            }
        }
        
        

        holdFactorDict = {
            MAGNIFICENT_MACARONS: magmaHold
        }



        VolumeFraction = VolumeCapability(product, mode=mode,state=state) / LIMIT[product]
        holdPremium = int(holdFactorDict[product] * self.positionCounter[product])
        high = AdjustmentDict[product]["high"] + holdPremium
        mid = AdjustmentDict[product]["mid"] + holdPremium
        low = AdjustmentDict[product]["low"] + holdPremium
        neg = AdjustmentDict[product]["neg"] + holdPremium

        if product == MAGNIFICENT_MACARONS: 
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
###----------------------------------            Runnin                --------------------------------------###
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
 
       
        obvs=state.observations.conversionObservations[MAGNIFICENT_MACARONS]
        feeDict={ETARIFF:abs(obvs.exportTariff),ITARIFF:abs(obvs.importTariff),SUNLIGHT:obvs.sunlightIndex,SUGAR:obvs.sugarPrice,TRANSPORT:obvs.transportFees}

        current_magma_mid = mid_price(state.order_depths[MAGNIFICENT_MACARONS])
        current_sugma = obvs.sugarPrice

        z_threshold = 2.0

        # Call the trading signal function:
        action, current_residual, min_squared_diff, z = self.trade_magma_signal(
            current_magma_mid,
            current_sugma,
            self.magma_mid_prices,  # historical prices for MAGNIFICENT_MACARONS (length ideally 45)
            self.sugma_prices,      # historical prices for SUGMA (length ideally 45)
            z_threshold
        )
        
        
        if action == "sell":

            order_price = current_magma_mid - 1  
            order_quantity = 10  

            result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, order_price, -order_quantity))
       
        elif action == "buy":
  
            order_price = current_magma_mid + 1  
            order_quantity = 10  

            result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, order_price, order_quantity))
            
        else:
            self.logger.print(f"No trade signal triggered: z = {z:.2f} within threshold ±{z_threshold}.")

        
        # Finally, flush the logs and return the result.
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data


################################################################################################################
###----------------------------------            Finish                --------------------------------------###
################################################################################################################


        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    

    

################################################################################################################
###-----------         prosperity3bt "Level4/James/PairOfAces1.py" 4 --no-out               -----------------###
################################################################################################################

 