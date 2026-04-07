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
###----------------------------------          Current Logic           --------------------------------------###
################################################################################################################
"""



"""
################################################################################################################
###----------------------------------            Defining              --------------------------------------###
################################################################################################################


MAGNIFICENT_MACARONS = 'MAGNIFICENT_MACARONS'
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
  
            }

}

ASPARAMS = {


}


################################################################################################################
###----------------------------------            Params                --------------------------------------###
################################################################################################################





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



def get_best_asks_to_fill_WITH_LEVELS(product, state, LeastVolIWant):
    order_depth = state.order_depths[product]
    asks = order_depth.sell_orders  # Dict of price: volume

    # Sort ask prices from best (lowest) to worst (highest)
    sorted_asks = sorted(asks.items())

    # Use at most best and second-best asks
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
    bids = order_depth.buy_orders  # Dict of price: volume

    # Sort bid prices from best (highest) to worst (lowest)
    sorted_bids = sorted(bids.items(), reverse=True)

    # Use at most best and second-best bids
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

def avellaneda_stoikov_delta(product, mid, inventory):
    '''
    Return the delta (price adjustment) using Avellaneda-Stoikov model.
    Includes reservation price shift from inventory risk and optimal half-spread.
    '''
    params = ASPARAMS[product]
    gamma = params["gamma"]
    sigma = params["sigma"]
    k_val = params["k"]
    T = params["T"]

    reservation_price = mid - inventory * gamma * (sigma ** 2) * T
    optimal_spread = (2 / gamma) * math.log(1 + gamma / k_val)

    delta = abs(reservation_price - mid) + optimal_spread / 2
    return delta

def net_profit_importing_from_island2(product, state):
    obvs = state.observations.conversionObservations[product]
    island2_ask = obvs.askPrice + obvs.transportFees + obvs.exportTariff
    island1_bid = max(state.order_depths[product].buy_orders.keys())

    return island1_bid - island2_ask

##############################################################
#####################################################################






################################################################################################################
###----------------------------------            Trader Class          --------------------------------------###
################################################################################################################
class Trader:
    def __init__(self) -> None:
        self.lot_size = 10

################################################################################################################
###---------------------     updating the fucking counter im so mad right now   -----------------------------###
################################################################################################################


###############################################################################################################
###----------------------------------      Trader -> All Calcs        --------------------------------------###
###############################################################################################################

    



################################################################################################################
###----------------------------------            Run Function          --------------------------------------###
################################################################################################################
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        # Prepare orders container
        result: Dict[Symbol, List[Order]] = {MAGNIFICENT_MACARONS: []}
        conversions = 0
        trader_data = ""

        # Get island conversion observation
        obvs = state.observations.conversionObservations[MAGNIFICENT_MACARONS]
        island_buy_price  = obvs.askPrice + obvs.transportFees + obvs.importTariff
        island_sell_price = obvs.bidPrice - obvs.transportFees - obvs.exportTariff

        # Local orderbook
        depth: OrderDepth = state.order_depths[MAGNIFICENT_MACARONS]
        local_best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else 0
        local_best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else float('inf')



################################################################################################################
###----------------------------------          Run - Place Trades      --------------------------------------###
################################################################################################################


        # Buy from island (import) and sell locally if profitable
        if island_buy_price < local_best_bid:
            logger.print(f"Arb SELL: island_buy={island_buy_price:.2f} < local_bid={local_best_bid:.2f}")
            result[MAGNIFICENT_MACARONS].append(
                Order(MAGNIFICENT_MACARONS, local_best_bid, -self.lot_size)
            )

        # Buy locally and sell to island if profitable
        # Buy locally and sell to island if profitable
        if island_sell_price > local_best_ask:
            logger.print(f"Arb BUY: island_sell={island_sell_price:.2f} > local_ask={local_best_ask:.2f}")
            result[MAGNIFICENT_MACARONS].append(
                Order(MAGNIFICENT_MACARONS, local_best_ask, +self.lot_size)
            )
    

################################################################################################################
###----------------------------------            Finish                --------------------------------------###
################################################################################################################


        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data


################################################################################################################
###-----------         prosperity3bt "Level4/James/WasteOfFuckingTime.py" 4 --no-out        -----------------###
################################################################################################################

