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

MacShipThreshold=0
MAGNIFICENT_MACARONS ='MAGNIFICENT_MACARONS'
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
ETARIFF='export_tariff'
ITARIFF='import_tariff'
SUGAR='sugar_price'
TRANSPORT='transport_fee'
SUNLIGHT='sunlight_index'
Kz=1
Sz=1
Jz=1
Dz=1
Cz=1
KTrigger=5*Kz
STrigger=26*Sz
JTrigger=9*Jz
DTrigger=9*Dz
CTrigger=19*Cz
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
PBtolerance = 3
PBWindow = 10

def kalman_filter_1d(z, Q=1e-5, R=4):
    """
    1D Kalman filter using numpy.
    z: observed time series (e.g., mid-prices)
    Q: process variance (model noise)
    R: measurement variance (observation noise)
    Returns: filtered estimates
    """
    n = len(z)
    x_hat = np.zeros(n)      # filtered state estimate
    P = np.zeros(n)          # error covariance
    x_hat[0] = z[0]          # initial state
    P[0] = 1.0               # initial covariance

    for k in range(1, n):
        # Predict
        x_hat_minus = x_hat[k - 1]
        P_minus = P[k - 1] + Q

        # Update
        K = P_minus / (P_minus + R)
        x_hat[k] = x_hat_minus + K * (z[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus

    return x_hat

def project_next(smoothed, window=10, correlation_threshold=0.8):
    """
    Project the next point using linear or quadratic fit based on Pearson correlation.
    
    Args:
        smoothed (list[float]): The smoothed time series data.
        window (int): The number of recent points to use for the fit.
        order (int): The default order of the polynomial fit (1 for linear, 2 for quadratic).
        correlation_threshold (float): The minimum Pearson correlation for a linear fit to be acceptable.
    
    Returns:
        float: The projected next value.
    """
    # Ensure there are enough elements in `smoothed`
    if len(smoothed) < window:
        window = len(smoothed)  # Use all available elements if fewer than `window`

    recent = smoothed[-window:]
    x = np.arange(len(recent))  # Match the length of `recent`

    # Validate the `recent` array
    if np.any(np.isnan(recent)) or np.any(np.isinf(recent)):
        return int(smoothed[-1])  # Return the last valid value as a fallback

    try:
        # Attempt a linear fit (order=1)
        a, b = np.polyfit(x, recent, 1)  # y = ax + b
        linear_fit = a * x + b
        # Calculate Pearson correlation for the linear fit
        correlation = np.corrcoef(recent, linear_fit)[0, 1]

        if correlation >= correlation_threshold:
            # Linear fit is good, project the next value
            next_value = a * len(recent) + b
        else:
            # Linear fit is poor, attempt a quadratic fit (order=2)
            a, b, c = np.polyfit(x, recent, 2)  # y = ax^2 + bx + c
            next_value = a * (len(recent) ** 2) + b * len(recent) + c
    except np.linalg.LinAlgError as e:
        return int(smoothed[-1])  # Return the last valid value as a fallback

    return int(next_value)

## GENERAL FUNCTIONS #######################
def sortDict(dictionary:dict):
    return {key: dictionary[key] for key in sorted(dictionary)}

def VolumeCapability(product, mode, state: TradingState):
    if product not in set(state.position.keys()):
        position = 0
    else:
        position = state.position[product]

    if mode == "buy":
        limit = LIMIT[product]['limit'] if isinstance(LIMIT[product], dict) else LIMIT[product]
        return limit - position
    elif mode == "sell":
        limit = LIMIT[product]['limit'] if isinstance(LIMIT[product], dict) else LIMIT[product]
        return position + limit

def mid_price(order_depth: OrderDepth) -> float:
    if order_depth.sell_orders and order_depth.buy_orders:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2
    elif order_depth.sell_orders:
        return min(order_depth.sell_orders.keys())
    elif order_depth.buy_orders:
        return max(order_depth.buy_orders.keys())
    return 0.0

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
        self.pos_limits = LIMIT
        self.lot_size = 1
        self.factor_labels = ["sugarfactor", "exportfactor", "importfactor", "transportfactor", "sunfactor"]
        self.previousposition = {params:0 for params in LIMIT}
        self.position = {params:0 for params in LIMIT}
        self.positionCounter = {params:0 for params in LIMIT}
        self.importArr=[]
        self.exportArr=[]
        self.resin_mid_prices = []
        self.kelp_mid_prices = []
        self.ink_mid_prices = []
        self.rock_mid_prices = []
        self.jam_mid_prices=[]
        self.djembe_mid_prices=[]
        self.croissant_mid_prices=[]
        self.macaron_mid_prices=[]
        self.pb1_mid_prices=[]
        self.pb2_mid_prices=[]
        self.S1Arr=[]
        self.S2Arr=[]
        self.sunpricearray=[]
        self.conversionMemory=0
        self.position_limits = LIMIT
        self.historical_portfolio_value={params:0 for params in LIMIT}
        self.signalDict={
                    KELP:0,
                    SQUID_INK:0,
                    JAMS:0,
                    DJEMBES:0,
                    CROISSANTS:0
                }
        self.signalTrigger={
            KELP:KTrigger,
            SQUID_INK:STrigger,
            JAMS:JTrigger,
            DJEMBES:DTrigger,
            CROISSANTS:CTrigger
        }
        self.triggerlist={
            product:[] for product in self.signalTrigger
        }

    def vwapOD(self,product: str, orderdepth:OrderDepth, mode: Optional[str] = None) -> float:
        """
        Calculate volume-weighted average price (VWAP) for a given product.

        If `mode` is 'bid', uses buy orders only.
        If `mode` is 'ask', uses sell orders only.
        If `mode` is None, calculates full book VWAP.

        Returns 0.0 if no valid orders are available.
        """
        depth = orderdepth
        if product not in depth.buy_orders or depth.sell_orders:
            return 0.0  # Product doesn't exist

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

    def Viable(self,price,feedict:dict,mode:str) -> bool:
        if mode =='import':
            if price-self.ImportDiff<0:
                return False
            if price-self.ImportDiff-minimumMacOrder*(feedict[ITARIFF]+feedict[TRANSPORT]+0.1*self.averageHold)-MacShipThreshold<self.ImportSigma:
                return False #SHOULD BE FALSE BUT FOR TESTS ONLY
        elif mode == 'export':
            if price-self.ExportDiff>0:
                return False
            if price-self.ExportDiff+minimumMacOrder*(feedict[ETARIFF]+feedict[TRANSPORT]+0.1*self.averageHold)+MacShipThreshold>-self.ExportSigma:
                return False #SHOULD BE FALSE BUT FOR TESTS ONLY
        return True
    
    def OrderOptimised(self, product: str, size: int, mode: str, state: TradingState) -> list[Order]:
            orders = []
            VolTarget = size
            if size==0:
                return []
            depth = state.order_depths[product]

            if mode == 'buy':
                # Ensure there are sell orders available; if not, return an empty list.
                if not depth.sell_orders:
                    return orders
                sell_orders = depth.sell_orders  # Use the correct attribute name
                # Get the prices sorted in ascending order (lowest offers first)
                sorted_prices = sorted(sell_orders.keys(), reverse= False)
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
                sorted_prices = sorted(buy_orders.keys(), reverse=True)
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
        """
        Calculate the mid-price using the order depth.
        """
        if order_depth.sell_orders:
            total_ask = sum(price * quantity for price, quantity in order_depth.sell_orders.items())
            total_qty = sum(quantity for quantity in order_depth.sell_orders.values())
            m1 = total_ask / total_qty if total_qty != 0 else 0
        else:
            m1 = 0

        if order_depth.buy_orders:
            total_bid = sum(price * quantity for price, quantity in order_depth.buy_orders.items())
            total_qty = sum(quantity for quantity in order_depth.buy_orders.values())
            m2 = total_bid / total_qty if total_qty != 0 else 0
        else:
            m2 = 0

        return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)
    
    def calculate_realized_returns(self, product: str, state: TradingState) -> float:
        """
        Calculate the realized profit or loss as a quantity for a given product,
        accounting for large spikes in position changes caused by recent trades.

        Args:
            product (str): The product to calculate realized profit or loss for.
            state (TradingState): The current trading state containing trade history and positions.

        Returns:
            float: The realized profit or loss as a quantity.
        """
        # Retrieve historical trades for the product
        own_trades = state.own_trades.get(product, [])
        if not own_trades:
            return 0.0  # No trades executed, so no realized profit or loss

        # Calculate the total cost and total quantity of executed trades
        total_cost = 0.0
        total_quantity = 0
        for trade in own_trades:
            if trade.price <= 0 or abs(trade.quantity) > LIMIT[product]:
                continue
            total_cost += trade.price * trade.quantity
            total_quantity += abs(trade.quantity)

        # If no quantity has been traded, return 0
        if total_quantity == 0:
            return 0.0

        # Calculate the average entry price
        avg_entry_price = np.round(total_cost / total_quantity,decimals=1)

        # Calculate the current portfolio value for the product
        current_position = self.position.get(product, 0)
        if current_position == 0:
            return 0.0  # No position, so no realized profit or loss

        # Retrieve the current mid-price
        order_depth = state.order_depths.get(product)
        if not order_depth:
            return 0.0  # No market data available

        mid_price = self.mid_price(order_depth)

        # Calculate the realized profit or loss as a quantity
        # realized_quantity = (mid_price - avg_entry_price) * current_position
        realized_quantity=(mid_price)*current_position-total_cost

        return realized_quantity
    def currentPNL(self, product: str, state: TradingState) -> float:
        """
        Calculate the realized profit or loss as a quantity for a given product,
        accounting for large spikes in position changes caused by recent trades.

        Args:
            product (str): The product to calculate realized profit or loss for.
            state (TradingState): The current trading state containing trade history and positions.

        Returns:
            float: The realized profit or loss as a quantity.
        """
        # Retrieve historical trades for the product
        own_trades = state.own_trades.get(product, [])
        if not own_trades:
            return 0.0  # No trades executed, so no realized profit or loss

        # Calculate the total cost and total quantity of executed trades
        total_cost = 0.0
        total_quantity = 0
        for trade in own_trades:
            if trade.price <= 0 or abs(trade.quantity) > LIMIT[product]:
                continue
            total_cost += trade.price * trade.quantity
            total_quantity += abs(trade.quantity)

        return total_cost
    
    def UpdatePreviousPositionCounter(self,product,state:TradingState) -> None:
            if product not in set(state.position.keys()):
                return None
            if (
                state.position[product] == self.previousposition[product]
            ):  # Updates previouspositionCounter
                self.positionCounter[product] += 1
            else:
                self.positionCounter[product] = 0
    def martingaleOrders(self,product,price,orderDepth:OrderDepth,mode:str,volume=None)->list[Order]:
        midprice=mid_price(orderDepth)
        vwap=self.vwapOD(product,orderDepth)
        askGaps=[askPrice-midprice for askPrice in orderDepth.sell_orders]
        bidGaps=[midprice-bidPrice for bidPrice in orderDepth.buy_orders]
        if vwap>midprice:
            momentum=1
        elif vwap<midprice:
            momentum=-1
        else:
            momentum=0
        if volume==None:
            if mode == 'buy':
                return [Order(product,int(price+abs(gap)+momentum),abs(orderDepth.sell_orders[_])) for gap,_ in zip(askGaps,orderDepth.sell_orders)]  
            elif mode == 'sell':
                return [Order(product,int(price-abs(gap)+momentum),-abs(orderDepth.buy_orders[_])) for gap,_ in zip(bidGaps,orderDepth.buy_orders)]
        else:
            if mode =='buy':
                volumeincrement=int(volume/len(askGaps))
                return [Order(product,int(price+abs(gap)+momentum),abs(volumeincrement)) for gap,_ in zip(askGaps,orderDepth.sell_orders)]  
            elif mode == 'sell':
                volumeincrement=int(volume/len(bidGaps))
                return [Order(product,int(price-abs(gap)+momentum),-abs(volumeincrement)) for gap,_ in zip(bidGaps,orderDepth.buy_orders)]
    def PBPricing(self,base,synth,od,product,state)->list[Order]:
        if base>synth+PBtolerance:
            return self.martingaleOrders(product=product,price=int(base),orderDepth=od,mode='sell')
        elif synth-PBtolerance>base:
            return self.martingaleOrders(product=product,price=int(base),orderDepth=od,mode='buy')
        else:
            return []

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {product: [] for product in LIMIT}
        conversions = 0
        trader_data = ""

        # Update position tracking
        for product in self.pos_limits.keys():
            self.UpdatePreviousPositionCounter(product, state)
            self.position[product] = state.position.get(product, 0)
            try:
                self.update_market_data(product, state)
            except Exception:
                pass
        # Load prior trader data (optional if you're passing internal state)
        try:
            loaded_data = json.loads(state.traderData) if state.traderData and state.traderData != '""' else {}
            if not isinstance(loaded_data, dict): loaded_data = {}
        except Exception as e:
            logger.print(f"Error loading traderData: {e}")
            loaded_data = {}
        
        futurePriceDict={product:None for product in [KELP,SQUID_INK,JAMS,DJEMBES,CROISSANTS]}
        for product in [KELP,SQUID_INK,JAMS,DJEMBES,CROISSANTS]:
            if state.order_depths[product].buy_orders or state.order_depths[product].sell_orders:
                ProdOD=state.order_depths[product]
                midP=mid_price(ProdOD)
                if product==KELP:
                    midParr =self.kelp_mid_prices
                    Korder=1
                    order=Korder
                elif product==SQUID_INK:
                    midParr=self.ink_mid_prices
                    Sorder=2
                    order=Sorder
                elif product==JAMS:
                    midParr=self.jam_mid_prices
                    Jorder=2
                    order=Jorder
                elif product == DJEMBES:
                    midParr=self.djembe_mid_prices
                    Dorder=2
                    order=Dorder
                elif product == CROISSANTS:
                    midParr=self.croissant_mid_prices
                    Dorder=2
                    order=Dorder
                midParr.append(midP)
                midParr=midParr[-101:]
            if state.timestamp >= 200:
                if midParr[-1]>midParr[-2]:
                    self.signalDict[product]+=1
                elif midParr[-1]<midParr[-2]:
                    self.signalDict[product]-=1
                else:
                    # if self.position[product]>0: Commented out due to severe losses
                    #     result[product]+=self.martingaleOrders(product,midP,ProdOD,'sell')
                    #     # logger.print(f'sell neutral triggered')
                    # elif self.position[product]<0:
                    #     result[product]+=self.martingaleOrders(product,midP,ProdOD,'buy')
                    #     # logger.print(f'buy neutral triggered')
                    pass
                # logger.print(product+' signal: '+ str(self.signalDict[product]))
                self.triggerlist[product].append(self.signalDict[product])
                if state.timestamp>=100*50:
                    smoothedmid=kalman_filter_1d(midParr[-100:])
                    nextPrice=project_next(smoothedmid,window=30,correlation_threshold=0.7)
                    Kmode=None
                    futurePriceDict[product]=nextPrice
        if state.timestamp >= 100*50:
            
            PB1OD=state.order_depths[PICNIC_BASKET1]
            PB2OD=state.order_depths[PICNIC_BASKET2]
            self.pb1_mid_prices.append(mid_price(PB1OD))
            self.pb1_mid_prices=self.pb1_mid_prices[-200:]
            self.pb2_mid_prices.append(mid_price(PB2OD))
            self.pb2_mid_prices=self.pb2_mid_prices[-200:]
            for pb in [PICNIC_BASKET1,PICNIC_BASKET2]:
                synthDict={PICNIC_BASKET1:[6,3,1],PICNIC_BASKET2:[4,2,0]}
                SArr={PICNIC_BASKET1:self.S1Arr,PICNIC_BASKET2:self.S2Arr}
                pbMidDict={PICNIC_BASKET1:self.pb1_mid_prices[-1],PICNIC_BASKET2:self.pb2_mid_prices[-1]}
                compPricearr=[self.croissant_mid_prices[-100:][-1],self.jam_mid_prices[-100:][-1],self.djembe_mid_prices[-100:][-1]]
                coeffs=synthDict[pb]
                synthVal=0
                pbVal=pbMidDict[pb]
                synthVal+=pbVal
                for coeff,price in zip(coeffs,compPricearr):
                    synthVal-=coeff*price
                SArr[pb].append(synthVal)
                SArr[pb]=SArr[pb][-200:]
                if self.positionCounter[pb]>10 and self.calculate_realized_returns(pb,state)>0:
                    if self.position[pb]>0:
                        result[pb]=self.martingaleOrders(pb,pbVal,state.order_depths[pb],mode='sell',volume=self.position[pb])
                    elif self.position[pb]<0:
                        result[pb]=self.martingaleOrders(pb,pbVal,state.order_depths[pb],mode='buy',volume=self.position[pb])
                else:
                    zscoredict={PICNIC_BASKET1:3,PICNIC_BASKET2:2}
                    currentZ=np.round((synthVal-np.mean(SArr[pb][-30:]))/np.sqrt(np.var(SArr[pb][-30:])),decimals=2)
                    if currentZ>=zscoredict[pb]:
                        result[pb]+=self.martingaleOrders(pb,pbVal,state.order_depths[pb],mode='buy')
                    elif currentZ<=-zscoredict[pb]:
                        result[pb]+=self.martingaleOrders(pb,pbVal,state.order_depths[pb],mode='sell')
        logger.flush(state, result, conversions, trader_data)   
        return result, conversions, trader_data
