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


1. Calculate the AS Delta using AS Params of Magma
    -> These still need to be found offline. 
2. Kalman filter: FV Initialisation -> Prediction -> FV (X) Update with Kalman Gain Weighting. 
    -> 
3. Fit Regression using Kalman X, M, C, Sugma
4. Z-score calculations using : mean, std, current residual 
5. Pearson Correlation -> Cock Block if too low. 
6. Z-Score Action -> Buy/Sell Signals
7. Microprice -> Finds out Buy/Sell Price adjusted with AS Delta
8. Order Size -> Dynamic Limit 
9. Execute Orders. 



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
    SUGAR: 69

}

PARAMS = {
         
    MAGNIFICENT_MACARONS: {
            "z_threshold": 0.05,         # The z-score threshold
            "history_length": 45,      # Number of historical mid–prices to use
            "default_fair_method": "vwap_with_vol_filter", #NOT SURE ABOUT THIS ONE. SHOULD BE USED FOR SUGAR 
            "min_volume_filter": 20, #AlsoNOT SURE ABOUT THIS ONE. SHOULD BE USED FOR SUGAR 
            "min_corr_threshold": 0, 
            }

}



ASPARAMS={MAGNIFICENT_MACARONS: {
                "gamma": 1.3,
                "sigma": 80,
                "k": 2,
                "max_order_size": 10,
                "T": 1.0,
                "limit": 75,
                "buffer": 2,
            }}



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
        self.residual_history = []
        self.logger = Logger()
        self.pos_limits = LIMIT[MAGNIFICENT_MACARONS]
        self.lot_size = 10

        self.previousposition = {params:0 for params in LIMIT.keys()}
        self.position = {params:0 for params in LIMIT.keys()}
        self.positionCounter = {params:0 for params in LIMIT.keys()}

        self.magma_timestamps = []
        self.magma_mid_prices = []
        self.sugma_timestamps = []
        self.sugma_prices = []

        self.lookbackwindow = 45  # Define the lookback window here

        self.m = 5.067721425995204
        self.c = -361.0670085205476
        self.mean = -1.96e-13
        self.stddev = 66.35232979828858


        # kalman filter state
        self.kf_x: float = None
        self.kf_P: float = 1.0
        self.Q:   float = 0.0001
        self.R:   float = 0.01




################################################################################################################
###---------------------     updating the fucking counter im so mad right now   -----------------------------###
################################################################################################################


    def update_market_data(self, product, state):
        order_depth = state.order_depths[product]
        obvs = state.observations.conversionObservations[MAGNIFICENT_MACARONS]
        mid = mid_price(order_depth)

        if product == MAGNIFICENT_MACARONS:
            self.magma_timestamps.append(state.timestamp)
            self.magma_mid_prices.append(mid)
            # Truncate to last 45 steps
            self.magma_mid_prices = self.magma_mid_prices[-self.lookbackwindow:]
            self.magma_timestamps = self.magma_timestamps[-self.lookbackwindow:]

        elif product == SUGAR:
            self.sugma_timestamps.append(state.timestamp)
            self.sugma_prices.append(obvs.sugarPrice)
            # Truncate to last 45 steps
            self.sugma_prices = self.sugma_prices[-self.lookbackwindow:]
            self.sugma_timestamps = self.sugma_timestamps[-self.lookbackwindow:]

    def UpdatePreviousPositionCounter(self,product,state:TradingState) -> None:
            if product not in set(state.position.keys()):
                return None
            if (
                state.position[product] == self.previousposition[product]
            ):  # Updates previouspositionCounter
                self.positionCounter[product] += 1
            else:
                self.positionCounter[product] = 0


###############################################################################################################
###----------------------------------      Trader -> All Calcs        --------------------------------------###
###############################################################################################################

    def kalman_filter(self, mid: float) -> float:
        if self.kf_x is None:
            self.kf_x = mid
        self.kf_P += self.Q
        K = self.kf_P / (self.kf_P + self.R)
        self.kf_x += K * (mid - self.kf_x)
        self.kf_P *= (1 - K)
        return self.kf_x

    def microprice(self, depth: OrderDepth) -> float:
        bb = max(depth.buy_orders)
        ba = min(depth.sell_orders)
        return (ba*depth.buy_orders[bb] + bb*depth.sell_orders[ba]) / \
            (depth.buy_orders[bb] + depth.sell_orders[ba])
        
    def compute_pearson_correlation(self) -> float:
        if len(self.magma_mid_prices) < 2 or len(self.sugma_prices) < 2:
            return 0.0  # Not enough data

        # Ensure aligned lengths
        min_len = min(len(self.magma_mid_prices), len(self.sugma_prices))
        magma = self.magma_mid_prices[-min_len:]
        sugma = self.sugma_prices[-min_len:]

        return np.corrcoef(magma, sugma)[0, 1]

    def calculate_zscore_action(
        self,
        current_residual,
        residual_history,
        lookback_window,
        z_threshold,
    ):

        # Update residual history
        residual_history.append(current_residual)
        if len(residual_history) > lookback_window:
            del residual_history[:-lookback_window]
            # residual_history = residual_history[-lookback_window:]
            
        # Compute z-score
        if len(residual_history) >= 2:
            mean = np.mean(residual_history)
            stddev = np.std(residual_history)
            z = (current_residual - mean) / stddev if stddev != 0 else 0
        else:
            z = 0

        # Determine trading action
        if z > z_threshold:
            action = "sell"
        elif z < -z_threshold:
            action = "buy"
        else:
            action = "hold"

        return action, current_residual, z








################################################################################################################
###----------------------------------            Run Function          --------------------------------------###
################################################################################################################

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        result = {product: [] for product in LIMIT.keys()}
        can_convert = min(abs(self.position[MAGNIFICENT_MACARONS]), ASPARAMS[MAGNIFICENT_MACARONS]["max_order_size"])
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

        island2_ask = obvs.askPrice + obvs.transportFees + obvs.importTariff
        island2_bid = obvs.bidPrice - obvs.transportFees - obvs.exportTariff
        local_best_bid = max(state.order_depths[MAGNIFICENT_MACARONS].buy_orders.keys())
        local_best_ask = min(state.order_depths[MAGNIFICENT_MACARONS].sell_orders.keys())

            
        self.sugma_prices.append(obvs.sugarPrice)
        self.sugma_prices = self.sugma_prices[-self.lookbackwindow:]


        island2_mid = (obvs.bidPrice + obvs.askPrice) / 2
        fv = self.kalman_filter(island2_mid)
        sugar = obvs.sugarPrice
        current_residual = fv - (self.m * sugar + self.c)
        # mp = self.microprice(state.order_depths[MAGNIFICENT_MACARONS])
        delta = avellaneda_stoikov_delta(MAGNIFICENT_MACARONS, fv, self.position[MAGNIFICENT_MACARONS])


        correlation = self.compute_pearson_correlation()
        
        if abs(correlation) < PARAMS[MAGNIFICENT_MACARONS]["min_corr_threshold"]:
            self.logger.print(f"Skipped trading: Pearson correlation too low ({correlation:.2f}).")
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data
        else:
            self.logger.print(f"Pearson correlation OK: {correlation:.2f}")





################################################################################################################
###----------------------------------          Run - Place Trades      --------------------------------------###
###############################################################################################################


        action, current_residual, z = self.calculate_zscore_action(
            current_residual=current_residual,
            residual_history=self.residual_history,
            lookback_window=self.lookbackwindow,
            z_threshold=PARAMS[MAGNIFICENT_MACARONS]["z_threshold"],
        )


        if action == "buy" and island2_ask < fv:
            result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, fv, +self.lot_size))
            if self.position[MAGNIFICENT_MACARONS] > 0:
                conversions = -min(self.lot_size, can_convert)
            else:
                conversions = 0
        elif action == "sell" and island2_bid > fv:
            result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, fv, -self.lot_size))
            if self.position[MAGNIFICENT_MACARONS] < 0:
                conversions = +min(self.lot_size, can_convert)
            else:
                conversions = 0

        


################################################################################################################
###----------------------------------            Finish                --------------------------------------###
################################################################################################################

        logger.print(f"Z-score: {z:.4f} | Action: {action}")
        logger.print(f"FV: {fv:.2f}, Island2 Ask: {island2_ask:.2f}, Island2 Bid: {island2_bid:.2f}")
        logger.print(f"Trade Conditions — BUY? {action == 'buy' and island2_ask < fv}, SELL? {action == 'sell' and island2_bid > fv}")
        logger.print(f"[DEBUG] z={z:.2f}, act={action}, fv={fv:.2f}, i2ask={island2_ask:.2f}, i2bid={island2_bid:.2f}, pos={self.position[MAGNIFICENT_MACARONS]}")


        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data


################################################################################################################
###-----------         prosperity3bt "Level4/James/PaVelaKalMicro.py" 4 --no-out           -----------------###
################################################################################################################

