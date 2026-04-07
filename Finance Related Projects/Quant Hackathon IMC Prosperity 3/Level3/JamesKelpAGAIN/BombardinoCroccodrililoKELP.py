import json
import jsonpickle
from typing import Any
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque
from typing import Any, TypeAlias
import numpy as np




################################################################################################################
###----------------------------------          Logger                  --------------------------------------###
################################################################################################################



#this sets JSON as the type alias for everything that can be a json
JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
#this class is written by jmerle and needed for using the visualizer and backtester (just ignore)
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
KELP = 'KELP'

LIMIT = {
    
    PICNIC_BASKET1: 60,
    PICNIC_BASKET2: 100,
    CROISSANTS: 250,
    JAMS: 350,
    DJEMBES: 60,
    KELP: 50,  
}


PARAMS = {
    

}



################################################################################################################
###----------------------------------            Strategy              --------------------------------------###
################################################################################################################


# class Strategy:
#     def __init__(self, product: str, limit: int):
#         self.product = product
#         self.limit = limit

#     @abstractmethod
#     def act(self, state: TradingState) -> None:
#         raise NotImplementedError()

#     def run(self, state: TradingState) -> list[Order]:
#         self.orders = []
#         self.act(state)
#         return self.orders

#     def buy(self, price: int, quantity: int) -> None:
#         self.orders.append(Order(self.product, int(price), quantity))

#     def sell(self, price: int, quantity: int) -> None:
#         self.orders.append(Order(self.product, int(price), -quantity))

#     #this is for transferring data from one trader to the next
#     def save(self) -> JSON:
#         return None

#     def load(self, data: JSON) -> None:
#         pass





################################################################################################################
###----------------------------------          Market Making           --------------------------------------###
################################################################################################################

# class MarketMakingStrategy(Strategy):
#     def __init__(self, 
#                  product: str, limit: int, 
#                  strategy_args):
#         super().__init__(product, limit)

#         self.history = deque()
#         self.mid_price_history = deque(maxlen=10)

#         self.EMA_alpha = strategy_args.get("EMA_alpha", 0.32)
#         self.history_size = strategy_args.get("history_size", 10)
#         self.soft_liquidate_thresh = strategy_args.get("soft_liquidation_tresh", 0.5)
#         self.volatility_multiplier = strategy_args.get("volatility_multiplier", 1.0)

#         self.beta_reversion = strategy_args.get("beta_reversion", 0.369)
#         self.volume_threshold = strategy_args.get("volume_threshold", 12) #for indentifying the market maker
#         self.last_mm_mid_price = None #in this parameter we store the last mid_price if no new midprice can be calculated
#         self.EMA = None

#     def get_popular_average(self, state : TradingState) -> int:
#         #calculate the average between the most popular buy and sell price
#         order_depths = state.order_depths[self.product]
#         sell_orders = order_depths.sell_orders.items()
#         buy_orders = order_depths.buy_orders.items()

#         most_popular_sell_price = min(sell_orders, key = lambda item : item[1])[0]
#         most_popular_buy_price = max(buy_orders, key = lambda item : item[1])[0]
        
#         #calculate average of those prices
#         return (most_popular_buy_price + most_popular_sell_price)//2

#     #returns and updates the current EMA
#     def get_EMA(self, state : TradingState) -> float:

#         average_price = self.get_popular_average(state)

#         alpha = self.EMA_alpha

#         if self.EMA == None:
#             self.EMA = average_price
#         else:
#             self.EMA = average_price * alpha + (1 - alpha) * self.EMA

#         return self.EMA
    
#     #estimates volatility as std from the last 20 mid prices
#     def estimate_volatility(self):
#         if len(self.mid_price_history)<5:
#             return 0
#         return np.std(self.mid_price_history)
    

#     def act(self, state: TradingState) -> None:
#         ##Logic
#         #sort buy and sell orders on the market for later use
#         #this priotizes the cheapest sell offer and the highest buy offers
#         buy_orders = sorted(state.order_depths[self.product].buy_orders.items(), reverse = True)
#         sell_orders = sorted(state.order_depths[self.product].sell_orders.items())

#         #use the mid_price to calculate a volatility
#         mid_price = self.get_popular_average(state)
#         self.mid_price_history.append(mid_price)
#         volatility = self.estimate_volatility()

#         #how much we can buy/sell of this specific product
#         position = state.position.get(self.product, 0)
#         to_buy = self.limit - position
#         to_sell = self.limit + position

#         default_price = self.get_default_price(state)

#         #calculate a spread_corr depending on the volatility
#         spread_corr = round(self.volatility_multiplier * volatility)

#         #if we are completely short or long than add a true to the history
#         self.history.append(abs(position) == self.limit)
        
#         #if history is full remove first element
#         if len(self.history) > self.history_size:
#             self.history.popleft()

#         #define if we want to hard or soft liquidate (default to false if the history isnt full)
#         #soft: if more than half of the history is true and the last one is true
#         soft_liquidate = len(self.history) == self.history_size and sum(self.history) >= self.history_size * self.soft_liquidate_thresh and self.history[-1]
#         #hard: if all of the history is true
#         hard_liquidate = len(self.history) == self.history_size and all(self.history)

#         #now we want to define if we want to buy or sell more depending on how full our position is
#         #we can regulate the prob. of buying and selling by increasing/decreasing the max_buy_price/min_sell_price

#         #buy less likely if position to long
#         max_buy_price = default_price - 1 - spread_corr if position > self.limit * 0.5 else default_price - spread_corr

#         #sell less likely if position is to short
#         min_sell_price = default_price + 1 + spread_corr if position < self.limit * -0.5 else default_price + spread_corr

#         #buy as much as possible below the max_buy_price starting with cheaper offers first
#         for price, volume in sell_orders:
#             if price <= max_buy_price and to_buy > 0:
#                 #buy as much as possible, either restricted by limit or by order volume
#                 quantity = min(-volume, to_buy)
#                 self.buy(price, quantity)
#                 to_buy -= quantity
                
#         #do the hard liquidation for buying --> we are very short
#         if to_buy > 0 and hard_liquidate:
#             #buy half of whats possible for the default price of that product
#             quantity = to_buy // 2
#             self.buy(default_price, quantity)
#             to_buy -= quantity

#         #do the soft liquidation for buying --> we are short but not that short
#         if to_buy > 0 and soft_liquidate:
#             #buy half of whats possible for the default price of that product - some value
#             #this makes buying less likely
#             quantity = to_buy //2
#             self.buy(default_price - 2, quantity)
#             to_buy -= quantity

#         #do the normal buying without hard or soft liquidation
#         #for this we need to know the most popular buy price
#         if to_buy > 0:
#             #this finds the buy price of the highest volume
#             most_popular_price = max(buy_orders, key = lambda item: item[1])[0]
#             #now we either buy at this price + 1 or the max_buy_price depending on whats smaller
#             #--> we never buy over the max_buy_price
#             price = min(max_buy_price, most_popular_price + 1)
#             #now we buy as much as possible at this price
#             self.buy(price, to_buy)

#         #sell as much as possible above the min_sell_price starting with the highest offer first
#         for price, volume in buy_orders:
#             if price >= min_sell_price and to_sell > 0:
#                 #sell as much as possible, either restriced by limit or by order volume
#                 quantity = min(volume, to_sell)
#                 self.sell(price, quantity)
#                 to_sell -= quantity

#         #now do the hard liquidation for selling --> we are completely long
#         if to_sell > 0 and hard_liquidate:            
#             #sell half of whats possible at default price
#             quantity = to_sell // 2
#             self.sell(default_price, quantity)
#             to_sell -= quantity

#         #now do the soft liquidation for selling --> we are very short
#         if to_sell > 0 and soft_liquidate:
#             #sell half of whats possible at default price + some value making it less likely
#             quantity = to_sell // 2
#             self.sell (default_price + 2, quantity)
#             to_sell -= quantity

#         #now do the normal selling at the most popular selling price
#         if to_sell > 0:
#             #find sell price with highest volume (this means the smallest volume)
#             most_popular_price = min(sell_orders, key = lambda item: item[1])[0]
#             #now we either sell at the most_popular_price or at the min_sell_price what ever is higher
#             #-> we never sell under the min_sell_price
#             price = max(min_sell_price, most_popular_price - 1)
#             #now sell as much as possible for this price
#             self.sell(price, to_sell)

#     @abstractmethod
#     def get_default_price(self, state: TradingState) -> int:
#         raise NotImplementedError()
    
#     #this saves the current history
#     def save(self) -> JSON:
#         return list(self.history)
    
#     #this can load a history
#     def load(self, data : JSON) -> None:
#         self.history = deque(data)




################################################################################################################
###----------------------------------          SQUID INK               --------------------------------------###
################################################################################################################

# class SquidInkStrategy(MarketMakingStrategy):
#     def get_default_price(self, state) -> float:
        
#         #return self.get_popular_average(state)

#         order_depth = state.order_depths[self.product]

#         #this gets the cheapest price we can buy at, and the highest price we can sell at
#         best_ask = min(order_depth.sell_orders.keys())
#         best_bid = max(order_depth.buy_orders.keys())
#         #creates a list of the prices of orders with high volumes --> market maker prices
#         filtered_ask =  [price for price in order_depth.sell_orders.keys()
#                          if abs(order_depth.sell_orders[price])
#                          >= self.volume_threshold]

#         filtered_bid = [price for price in order_depth.buy_orders.keys()
#                          if abs(order_depth.buy_orders[price])
#                          >= self.volume_threshold]

#         #defines the ask and bid of the market maker as the best ask and bid of those prices        
#         mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
#         mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

#         #if there is no market maker
#         if mm_ask == None or mm_bid == None:
#             if self.last_mm_mid_price == None:
#                 mm_mid_price = (best_ask + best_bid)/2
#             else:
#                 mm_mid_price = self.last_mm_mid_price

#         else:
#             mm_mid_price = (mm_ask + mm_bid)/2

#         #mean reversion
#         if self.last_mm_mid_price != None:
#             last_price = self.last_mm_mid_price
#             last_returns = (mm_mid_price - last_price) / last_price
#             pred_returns = ( 
#                 last_returns * -0.369 #this tries to predict how much 
#             )
#             fair = mm_mid_price + (mm_mid_price * pred_returns)
#         else:
#             fair = mm_mid_price

#         self.last_mm_mid_price = mm_mid_price

#         return fair
    









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
    asks = order_depth.sell_orders  # Dict of price: volume

    # Sort ask prices from best (lowest) to worst (highest)
    sorted_asks = sorted(asks.items())

    # Use at most best and second-best asks
    max_levels = 2
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
    max_levels = 2
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
###----------------------------------            Trader                --------------------------------------###
################################################################################################################


class Trader:
    def __init__(self) -> None:
        #Define the limits and goods traded in the given round, this need to be changed for every submission
        self.pos_limits = LIMIT
        self.previousposition = {params:0 for params in LIMIT.keys()}
        self.position = {params:0 for params in LIMIT.keys()}
        self.positionCounter = {params:0 for params in LIMIT.keys()}
        self.volume_threshold = 12 #for indentifying the market maker
        self.last_mm_mid_price = None 

        # self.strategy_args = strategy_args if strategy_args != None else {}
        # #Define a strategy for every product, this is done by creating an instance of a specific strategy for every product
        # self.strategies = { symbol : strategyClass(symbol, limits[symbol], self.strategy_args.get(symbol, {})) for symbol, strategyClass in {
        #     "KELP" : KelpStrategy,
        #     # "SQUID_INK": SquidInkStrategy
        # }.items()}

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













################################################################################################################
###----------------------------------             Run Function         --------------------------------------###
################################################################################################################



    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
     
        order_depths = state.order_depths
        result = {product: [] for product in LIMIT.keys()}
        conversions = 0
        trader_data = ""

        if state.traderData:
            trader_data_all = jsonpickle.decode(state.traderData)
        else:
            trader_data_all = {}

        kelp_data = trader_data_all.get("KELP", {})
        self.last_mm_mid_price = kelp_data.get("last_mm_mid_price", None)

        # ...
        trader_data_dict = {
            "last_mm_mid_price": self.last_mm_mid_price
        }
        trader_data = json.dumps(trader_data_dict, separators=(",", ":"))


        # Updating Position and Position Counters
        for product in state.position:
            self.UpdatePreviousPositionCounter(product,state)
            self.position[product] = state.position[product]
            self.update_market_data(product, state)
        
        # -------------------------------------------------------------------------











        relevant = [
                        KELP
                    ]
        best_bid = {}
        best_ask = {}
        mid_price = {}
        for prod in relevant:
            od = state.order_depths.get(prod)
            if od:
                best_bid[prod] = max(od.buy_orders.keys(), default = None) 
                best_ask[prod] = min(od.sell_orders.keys(), default = None)
                if best_bid[prod] is not None and best_ask[prod] is not None:
                    mid_price[prod] = 0.5 * (best_bid[prod] + best_ask[prod])
                elif best_bid[prod] is not None:
                    mid_price[prod] = best_bid[prod]
                elif best_ask[prod] is not None:
                    mid_price[prod] = best_ask[prod]
                else:
                    mid_price[prod] = None
            else:
                # best_bid[prod] = None
                # best_ask[prod] = None
                mid_price[prod] = None

        




################################################################################################################
###----------------------------------             KELP                 --------------------------------------###
################################################################################################################
        # if KELP in state.order_depths:
        #     order_depth = state.order_depths[KELP]
        #     position = state.position.get(KELP, 0)
        #     to_buy = self.pos_limits[KELP] - position
        #     to_sell = self.pos_limits[KELP] + position

        #     # High-volume market maker detection
        #     filtered_ask = [
        #         price for price in order_depth.sell_orders.keys()
        #         if abs(order_depth.sell_orders[price]) >= self.volume_threshold
        #     ]
        #     filtered_bid = [
        #         price for price in order_depth.buy_orders.keys()
        #         if abs(order_depth.buy_orders[price]) >= self.volume_threshold
        #     ]

        #     mm_ask = min(filtered_ask) if filtered_ask else None
        #     mm_bid = max(filtered_bid) if filtered_bid else None

        #     # Fallback mid price
        #     if mm_ask is None or mm_bid is None:
        #         if self.last_mm_mid_price is None:
        #             best_ask = min(order_depth.sell_orders.keys())
        #             best_bid = max(order_depth.buy_orders.keys())
        #             mm_mid_price = (best_ask + best_bid) / 2
        #         else:
        #             mm_mid_price = self.last_mm_mid_price
        #     else:
        #         mm_mid_price = (mm_ask + mm_bid) / 2

        #     # Mean reversion logic
        #     if self.last_mm_mid_price is not None:
        #         last_price = self.last_mm_mid_price
        #         last_returns = (mm_mid_price - last_price) / last_price
        #         pred_returns = last_returns * -0.5
        #         fair = mm_mid_price + (mm_mid_price * pred_returns)
        #     else:
        #         fair = mm_mid_price

        #     self.last_mm_mid_price = mm_mid_price  # Store for next round

        #     # Buy from market if price < fair
        #     for price, volume in sorted(order_depth.sell_orders.items()):
        #         if price <= fair and to_buy > 0:
        #             quantity = min(-volume, to_buy)
        #             result[KELP].append(Order(KELP, price, quantity))
        #             to_buy -= quantity

        #     # Sell to market if price > fair
        #     for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
        #         if price >= fair and to_sell > 0:
        #             quantity = min(volume, to_sell)
        #             result[KELP].append(Order(KELP, price, -quantity))
        #             to_sell -= quantity

        # if KELP in state.order_depths:
        #     order_depth = state.order_depths[KELP]
        #     position = state.position.get(KELP, 0)
        #     to_buy = self.pos_limits[KELP] - position
        #     to_sell = self.pos_limits[KELP] + position

            # High-volume market maker detection
        if KELP in state.order_depths:
            od = state.order_depths[KELP]
            position = state.position.get(KELP, 0)
            to_buy = self.pos_limits[KELP] - position
            to_sell = self.pos_limits[KELP] + position
        else:
            od = None

        if od:
            # High-volume market maker detection
            filtered_ask = [
                price for price in od.sell_orders.keys()
                if abs(od.sell_orders[price]) >= self.volume_threshold
            ]
            filtered_bid = [
                price for price in od.buy_orders.keys()
                if abs(od.buy_orders[price]) >= self.volume_threshold
            ]

            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None

            # Fallback mid price
            if mm_ask is None or mm_bid is None:
                if self.last_mm_mid_price is None:
                    best_ask = min(od.sell_orders.keys())
                    best_bid = max(od.buy_orders.keys())
                    mm_mid_price = (best_ask + best_bid) / 2
                else:
                    mm_mid_price = self.last_mm_mid_price
            else:
                mm_mid_price = (mm_ask + mm_bid) / 2

            # Mean reversion logic
            if self.last_mm_mid_price is not None:
                last_price = self.last_mm_mid_price
                last_returns = (mm_mid_price - last_price) / last_price
                pred_returns = last_returns * -0.5
                fair = mm_mid_price + (mm_mid_price * pred_returns)
            else:
                fair = mm_mid_price

            self.last_mm_mid_price = mm_mid_price  # Store for next round

            # Buy from market if price < fair
            for price, volume in sorted(od.sell_orders.items()):
                if price <= fair and to_buy > 0:
                    quantity = min(-volume, to_buy)
                    result[KELP].append(Order(KELP, price, quantity))
                    to_buy -= quantity

            # Sell to market if price > fair
            for price, volume in sorted(od.buy_orders.items(), reverse=True):
                if price >= fair and to_sell > 0:
                    quantity = min(volume, to_sell)
                    result[KELP].append(Order(KELP, price, -quantity))
                    to_sell -= quantity

        kelp_data = {
            "last_mm_mid_price": self.last_mm_mid_price
        }

        trader_data_all["KELP"] = kelp_data
        trader_data = jsonpickle.encode(trader_data_all)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

################################################################################################################
###----------- prosperity3bt "Level3/JamesKelpAGAIN/BombardinoCroccodrililoKELP.py" 3 --no-out --------------###
################################################################################################################

 