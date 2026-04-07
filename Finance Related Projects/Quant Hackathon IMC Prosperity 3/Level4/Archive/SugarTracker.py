import json
import jsonpickle
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional, Deque
from abc import abstractmethod
from collections import deque, defaultdict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import statistics
from statistics import NormalDist
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

BID='bid'
ASK='ask'
BUY='buy'
SELL='sell'

RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = 'KELP'
SQUID_INK = 'SQUID_INK'
CROISSANTS = 'CROISSANTS'
PICNIC_BASKET1='PICNIC_BASKET1'
PICNIC_BASKET2='PICNIC_BASKET2'
JAMS='JAMS'
DJEMBES='DJEMBES'
VOLCANIC_ROCK='VOLCANIC_ROCK'
VOLCANIC_ROCK_VOUCHER_9500='VOLCANIC_ROCK_VOUCHER_9500'
VOLCANIC_ROCK_VOUCHER_9750='VOLCANIC_ROCK_VOUCHER_9750'
VOLCANIC_ROCK_VOUCHER_10000='VOLCANIC_ROCK_VOUCHER_10000'
VOLCANIC_ROCK_VOUCHER_10250='VOLCANIC_ROCK_VOUCHER_10250'
VOLCANIC_ROCK_VOUCHER_10500='VOLCANIC_ROCK_VOUCHER_10500'
MAGNIFICENT_MACARONS ='MAGNIFICENT_MACARONS'

ETARIFF='export_tariff'
ITARIFF='import_tariff'
SUGAR='sugar_price'
TRANSPORT='transport_fee'
SUNLIGHT='sunlight_index'

ImportDiff=-9.91
ExportDiff=15.936
ImportSigma=0
ExportSigma=0

LIMIT = {
    RAINFOREST_RESIN: 50,
    KELP: 50,
    SQUID_INK: 50,
    CROISSANTS: 250,
    PICNIC_BASKET1: 60,
    PICNIC_BASKET2: 100,
    JAMS: 350,
    DJEMBES: 60,
    VOLCANIC_ROCK: 400,
    VOLCANIC_ROCK_VOUCHER_9500: 200,
    VOLCANIC_ROCK_VOUCHER_9750: 200,
    VOLCANIC_ROCK_VOUCHER_10000: 200,
    VOLCANIC_ROCK_VOUCHER_10250: 200,
    VOLCANIC_ROCK_VOUCHER_10500: 200,
    MAGNIFICENT_MACARONS :{'limit':75,'conversion':10} 
}

kelptw = 18.7017
kelppositionlimit=50
kelpminvolume = 21.7543
kelpspreadedge = 1.0760
squidz = 1.9572
squidhistorylength = 700.9250
SQrsi_window = 106
SQrsi_overbought = 52
SQrsi_oversold = 41
squidrsitradesize = 36.3892
croissanthistorylength = 83.8247
croissantzthreshold = 1.9986
minimumMacOrder=1
PARAMS = {
    RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 0.5,
        "join_edge": 2,
        "default_edge": 2,
        "soft_position_limit": 45,
    },
    KELP: {
        "take_width": kelptw,
        "position_limit": kelppositionlimit,
        "min_volume_filter": kelpminvolume,
        "spread_edge": kelpspreadedge,
        "default_fair_method": "vwap_with_vol_filter",
    },
    # New parameters for the SQUID_INK mean reversion strategy
    SQUID_INK: {
        "rsi_window": SQrsi_window,
        "rsi_overbought": SQrsi_overbought,
        "rsi_oversold": SQrsi_oversold,
    },
    CROISSANTS: {
        "history_length": croissanthistorylength,  # Number of mid-price datapoints to use for z-score calculation.
        "z_threshold": croissantzthreshold        # Threshold for trading.
    },
    VOLCANIC_ROCK: {
        "starting_time_to_expiry": 8 / 365,
        "std_window": 10,
    },
    VOLCANIC_ROCK_VOUCHER_9500: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 9500,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_9750: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 9750,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_10000: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 10000,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_10250: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 10250,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_10500: {
        "starting_time_to_expiry": 8 / 365,
        "strike": 10500,
        "std_window": 10,
        "implied_volatility": 0.16,
    }
}

STRIKES = {
    VOLCANIC_ROCK_VOUCHER_9500: 9500,
    VOLCANIC_ROCK_VOUCHER_9750: 9750,
    VOLCANIC_ROCK_VOUCHER_10000: 10000,
    VOLCANIC_ROCK_VOUCHER_10250: 10250,
    VOLCANIC_ROCK_VOUCHER_10500: 10500,
}

B1B2_THEORETICAL_COMPONENTS = {
    CROISSANTS: 2,
    JAMS: 1,
    DJEMBES: 1
}

### PB Synthetic Basket Parameters #######################
synth1Mean = -131.606  # PB1 is usually cheaper than PB2
synth1Sigma = np.round(29.05 // np.sqrt(1000), decimals=5)
s1Zscore = 0.3394  # TODO OPTIMISE!!!!!! 1 works pretty well
synth2Mean = 105.417
synth2Sigma = np.round(27.166 // np.sqrt(1000), decimals=5)
s2Zscore = 1.4417  # TODO OPTIMISE!!!!!!
diff_threshold_b1_b2 = 176.8118
diff_threshold_b1 = 176.8118
MacShipThreshold=0
#### Kelp Squink Pairs Trade Parameters

## GENERAL FUNCTIONS #######################
def sortDict(dictionary:dict):
    return {key: dictionary[key] for key in sorted(dictionary)}

def VolumeCapability(product, mode,state:TradingState):
    if product not in set(state.position.keys()):
        output2=0
    else:
        output2=state.position[product]
    if mode == "buy":
        output=LIMIT[product]
        if product==MAGNIFICENT_MACARONS:
            output=output['limit']
        return output - output2
    elif mode == "sell":
        output=LIMIT[product]
        if product==MAGNIFICENT_MACARONS:
            output=output['limit']
        return output2 + output
def vwap(product: str, state: TradingState, mode: Optional[str] = None) -> float:
    """
    Calculate volume-weighted average price (VWAP) for a given product.

    If `mode` is 'bid', uses buy orders only.
    If `mode` is 'ask', uses sell orders only.
    If `mode` is None, calculates full book VWAP.

    Returns 0.0 if no valid orders are available.
    """
    if product not in state.order_depths:
        return 0.0  # Product doesn't exist

    depth = state.order_depths[product]
    vwap = 0
    total_amt = 0

    if mode == 'bid' and depth.buy_orders:
        for prc, amt in depth.buy_orders.items():
            vwap += prc * amt
            total_amt += amt
    elif mode == 'ask' and depth.sell_orders:
        for prc, amt in depth.sell_orders.items():
            vwap += prc * amt
            total_amt += amt
    elif mode is None:
        for prc, amt in depth.buy_orders.items():
            vwap += prc * amt
            total_amt += amt
        for prc, amt in depth.sell_orders.items():
            vwap += prc * abs(amt)
            total_amt += abs(amt)

    if total_amt == 0:
        return 0.0  # Avoid division by zero

    return np.round(vwap / total_amt, decimals=5)


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
    if not state.order_depths[product].sell_orders:  # FREAKY
        return 0
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
    if not state.order_depths[product].buy_orders:
        return 0  # FREAKY
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



class Trader: ##from here

    def __init__(self) -> None:
        self.logger = Logger()
        self.logger.print("Initialized trader for JAMS and DJEMBES only.")
        self.pos_limits = LIMIT
        self.params = PARAMS
        self.diff_threshold_b1_b2 = diff_threshold_b1_b2
        self.lot_size = 1

        self.previousposition = {params:0 for params in LIMIT}
        self.position = {params:0 for params in LIMIT}
        self.positionCounter = {params:0 for params in LIMIT}
        self.importArr=[]
        self.exportArr=[]
        self.resin_timestamps = []
        self.resin_mid_prices = []
        self.kelp_timestamps = []
        self.kelp_mid_prices = []
        self.ink_timestamps = []
        self.ink_mid_prices = []
        self.rock_timestamps = []
        self.rock_mid_prices = []
        self.rock9500_timestamps = []
        self.rock9500_mid_prices = []
        self.rock9750_timestamps = []
        self.rock9750_mid_prices = []
        self.rock10000_timestamps = []
        self.rock10000_mid_prices = []
        self.rock10250_timestamps = []
        self.rock10250_mid_prices = []
        self.rock10500_timestamps = []
        self.rock10500_mid_prices = []
        self.diff_threshold_b1 = diff_threshold_b1 
        self.conversionMemory=0
        self.position_limits = LIMIT
        self.averageHold=1
        self.ExportDiff=ExportDiff
        self.ImportDiff=ImportDiff
        self.ExportSigma=ExportSigma
        self.ImportSigma=ImportSigma
    def update_market_data(self, product, state):
        # Store current timestamp and mid-price for all assets
        order_depth = state.order_depths[product]
        mid = mid_price(order_depth)
        if product == RAINFOREST_RESIN:
            self.resin_timestamps.append(state.timestamp)
            self.resin_mid_prices.append(mid)
        elif product == KELP:
            self.kelp_timestamps.append(state.timestamp)
            self.kelp_mid_prices.append(mid)
        elif product == SQUID_INK:
            self.ink_timestamps.append(state.timestamp)
            self.ink_mid_prices.append(mid)
        elif product == VOLCANIC_ROCK:
            self.rock_timestamps.append(state.timestamp)
            self.rock_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_9500":
            self.rock9500_timestamps.append(state.timestamp)
            self.rock9500_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_9750":
            self.rock9750_timestamps.append(state.timestamp)
            self.rock9750_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_10000":
            self.rock10000_timestamps.append(state.timestamp)
            self.rock10000_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_10250":
            self.rock10250_timestamps.append(state.timestamp)
            self.rock10250_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_10500":
            self.rock10500_timestamps.append(state.timestamp)
            self.rock10500_mid_prices.append(mid)

    def Viable(self,price,feedict:dict,macPriceDict:dict,mode:str) -> bool:
        if mode =='import':
            if price-self.ImportDiff<0:
                return False
            # if price-self.ImportDiff-minimumMacOrder*(feedict[ITARIFF]+feedict[TRANSPORT]+0.1*self.averageHold)-MacShipThreshold<0:
            if price-self.ImportDiff-feedict[ITARIFF]*macPriceDict[ASK]/100-minimumMacOrder*(feedict[TRANSPORT]+0.1*self.averageHold)-MacShipThreshold<0:
                return False #SHOULD BE FALSE BUT FOR TESTS ONLY
        elif mode == 'export':
            if price-self.ExportDiff>0:
                return False
            # if price-self.ExportDiff+minimumMacOrder*(feedict[ETARIFF]+feedict[TRANSPORT]+0.1*self.averageHold)+MacShipThreshold>0:
            if price-self.ImportDiff+feedict[ETARIFF]*macPriceDict[BID]/100+minimumMacOrder*(feedict[TRANSPORT]+0.1*self.averageHold)+MacShipThreshold>0:
                return False #SHOULD BE FALSE BUT FOR TESTS ONLY
        return True
    
    def OrderOptimised(self, product: str, size: int, mode: str, state: TradingState) -> list[Order]:
            orders = []
            VolTarget = size
            if size==0:
                return
            depth = state.order_depths[product]

            if mode == 'buy':
                # Ensure there are sell orders available; if not, return an empty list.
                if not depth.sell_orders:
                    return orders
                sell_orders = depth.sell_orders  # Use the correct attribute name
                # Get the prices sorted in ascending order (lowest offers first)
                sorted_prices = sorted(sell_orders.keys(), reverse= True)
                for price in sorted_prices:
                    # Use the minimum between the target volume and what is available
                    volume_to_take = min(VolTarget, abs(sell_orders[price]))
                    orders.append(Order(product, price, volume_to_take))
                    VolTarget -= volume_to_take
                    if VolTarget <= 0:
                        break

            elif mode == 'sell':
                # Ensure there are buy orders available; if not, return an empty list.
                if not depth.buy_orders:
                    return orders
                buy_orders = depth.buy_orders  # Use the correct attribute name
                # Get the prices sorted in descending order (best bids first)
                sorted_prices = sorted(buy_orders.keys(), reverse=False)
                for price in sorted_prices:
                    volume_to_take = min(VolTarget, abs(buy_orders[price]))
                    # For selling, we send a negative quantity
                    orders.append(Order(product, price, -volume_to_take))
                    VolTarget -= volume_to_take
                    if VolTarget <= 0:
                        break

            return orders

    def PositionFraction(self, product: str, state: TradingState) -> float:
        """
        Calculate the fraction of the position for a given 
        """
        position = state.position.get(product, 0)
        return np.round(position/LIMIT[product], decimals=3)
    
    def mid_price(self, order_depth: OrderDepth) -> float:
        # Compute a mid-price using available order depth information
        if order_depth.sell_orders:
            total_ask = sum(price * quantity for price, quantity in order_depth.sell_orders.items())
            total_qty = sum(quantity for quantity in order_depth.sell_orders.values())
            if total_qty != 0:
                m1 = total_ask / total_qty
            else:
                m1 = None
        else:
            m1 = 0

        if order_depth.buy_orders:
            total_bid = sum(price * quantity for price, quantity in order_depth.buy_orders.items())
            total_qty = sum(quantity for quantity in order_depth.buy_orders.values())
            if total_qty != 0:
                m2 = total_bid / total_qty
            else:
                m2 = None
        else:
            m2 = 0
            
        return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)
    
    def UpdatePreviousPositionCounter(self,product,state:TradingState) -> None:
            if product not in set(state.position.keys()):
                return None
            if (
                state.position[product] == self.previousposition[product]
            ):  # Updates previouspositionCounter
                self.positionCounter[product] += 1
            else:
                self.positionCounter[product] = 0


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        all_orders: List[Order] = []
        result = {product: [] for product in LIMIT}
        conversions = 0
        trader_data = ""
        trader_data_for_next_round = {}  
        # Updating Position and Position Counters
        for product in self.pos_limits.keys():
            self.UpdatePreviousPositionCounter(product,state)
            self.position[product] = state.position.get(product,0)
            try:
                self.update_market_data(product, state)
            except Exception:
                pass
        try:
            loaded_data = json.loads(state.traderData) if state.traderData and state.traderData != '""' else {}
            if not isinstance(loaded_data, dict): loaded_data = {}
        except Exception as e:
            logger.print(f"Error loading traderData: {e}")
            loaded_data = {}
        
        #### Strats for the macaron
        if MAGNIFICENT_MACARONS in state.order_depths:
            marketMacaronDict={BID:vwap(MAGNIFICENT_MACARONS,state,BID),ASK:vwap(MAGNIFICENT_MACARONS,state,ASK)}
            obvs=state.observations.conversionObservations[MAGNIFICENT_MACARONS]
            outsideMacaronDict={BID:obvs.bidPrice,ASK:obvs.askPrice}
            feeDict={ETARIFF:abs(obvs.exportTariff),ITARIFF:abs(obvs.importTariff),SUNLIGHT:obvs.sunlightIndex,SUGAR:obvs.sugarPrice,TRANSPORT:obvs.transportFees}
            ImportOpportunity=marketMacaronDict[BID]-outsideMacaronDict[ASK] # +ve means there is a chance to buy from island
            ExportOpportunity=marketMacaronDict[ASK]-outsideMacaronDict[BID] # -ve means there is a chance to sell to the island
            self.importArr.append(ImportOpportunity)
            self.exportArr.append(ExportOpportunity)
            ImportVol=0
            ExportVol=0
            if self.Viable(ImportOpportunity,feeDict,outsideMacaronDict,mode='import'):
                ImportVol=max(BidVolume(MAGNIFICENT_MACARONS,mode='max',state=state),VolumeCapability(MAGNIFICENT_MACARONS,mode='sell',state=state))
                result[MAGNIFICENT_MACARONS]+=self.OrderOptimised(MAGNIFICENT_MACARONS,ImportVol,mode='sell',state=state)
            if self.Viable(ExportOpportunity,feeDict,outsideMacaronDict,mode='export'):
                ExportVol=max(AskVolume(MAGNIFICENT_MACARONS,mode='min',state=state),VolumeCapability(MAGNIFICENT_MACARONS,mode='buy',state=state))
                result[MAGNIFICENT_MACARONS]+=self.OrderOptimised(MAGNIFICENT_MACARONS,ExportVol,mode='buy',state=state)
            logger.print('ImportVol:'+str(ImportVol))
            logger.print('ExportVol:'+str(ExportVol))
            logger.print('ImportOpportunity'+str(ImportOpportunity))
            logger.print('ExportOpportunity'+str(ExportOpportunity))
            logger.print('Importmean'+str(np.mean(self.importArr)))
            logger.print('Exportmean'+str(np.mean(self.exportArr)))
            logger.print('ImportSigma'+str(np.sqrt(np.var(self.importArr))))
            logger.print('ExportSigma'+str(np.sqrt(np.var(self.exportArr))))
            conversions=self.conversionMemory#ImportVol-ExportVol
            self.conversionMemory=ImportVol-ExportVol
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data