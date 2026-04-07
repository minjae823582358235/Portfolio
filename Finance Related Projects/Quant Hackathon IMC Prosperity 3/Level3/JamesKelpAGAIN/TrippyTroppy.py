import json
from typing import Any
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque
from enum import IntEnum
from typing import Any, TypeAlias
import numpy as np

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
#super strategy class
class Strategy:
    def __init__(self, product: str, limit: int, strategies):
        self.product = product
        self.limit = limit
        self.strategies = strategies

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.product, int(price), quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.product, int(price), -quantity))
    
    def convert(self, amount: int) -> None:
        self.conversions += amount

    #this is for transferring data from one trader to the next
    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class GeneralStrategy(Strategy):
    def __init__(self, 
                product: str, limit: int,
                strategies, 
                strategy_args):
        super().__init__(product, limit, strategies)

        # Market making attributes:
        self.history = deque()
        self.mid_price_history = deque(maxlen=10)
        self.EMA_alpha = strategy_args.get("EMA_alpha", 0.32)
        self.history_size = strategy_args.get("history_size", 10)
        self.soft_liquidate_thresh = strategy_args.get("soft_liquidation_thresh", 0.5)
        self.volatility_multiplier = strategy_args.get("volatility_multiplier", 1.0)
        self.beta_reversion = strategy_args.get("beta_reversion", 0.369)
        self.volume_threshold = strategy_args.get("volume_threshold", 12)
        self.last_mm_mid_price = None
        self.EMA = None

        # basket arbitrage attributes:
        self.EMA_price_history = deque(maxlen=10)

    
    @abstractmethod
    def get_default_price(self, state: TradingState) -> int:
        raise NotImplementedError()

   #-----helper round1-------- 
    def get_popular_average(self, state: TradingState) -> int:
        """
        Calculate the average between the most popular buy and sell prices.
        """
        order_depths = state.order_depths[self.product]
        sell_orders = order_depths.sell_orders.items()
        buy_orders = order_depths.buy_orders.items()
        most_popular_sell_price = min(sell_orders, key=lambda item: item[1])[0]
        most_popular_buy_price = max(buy_orders, key=lambda item: item[1])[0]
        return (most_popular_buy_price + most_popular_sell_price) // 2

    def get_EMA(self, state: TradingState) -> float:
        """
        Get or update the Exponential Moving Average (EMA) based on the popular average price.
        """
        average_price = self.get_popular_average(state)
        if self.EMA is None:
            self.EMA = average_price
        else:
            self.EMA = average_price * self.EMA_alpha + (1 - self.EMA_alpha) * self.EMA
        return self.EMA

    def estimate_volatility(self) -> float:
        """
        Estimate volatility as the standard deviation of the last few mid prices.
        """
        if len(self.mid_price_history) < 5:
            return 0
        return np.std(self.mid_price_history)

    def estimate_mean(self) -> float:
        """
        Estimate volatility as the standard deviation of the last few mid prices.
        """
        if len(self.mid_price_history) < 3:
            return 0
        return np.mean(self.mid_price_history)

    def estimate_EMA_mean(self) -> float:
        """
        Estimate volatility as the standard deviation of the last few mid prices.
        """
        if len(self.EMA_price_history) < 3:
            return 0
        return np.mean(self.EMA_price_history) 

    #---------helper round2-------------

    def get_mid_price(self, state: TradingState, product: str) -> float:
        depth = state.order_depths[product]
        best_bid = max(depth.buy_orders.keys(), default=None)
        best_ask = min(depth.sell_orders.keys(), default=None)
        if best_bid is None or best_ask is None:
            return 0
        return (best_bid + best_ask) / 2
    
    def get_popular_average_basket(self, state: TradingState, product: str) -> int:
        # Calculate the average between the most popular buy and sell price
        order_depths = state.order_depths[product]
        sell_orders = sorted(order_depths.sell_orders.items())
        buy_orders = sorted(order_depths.buy_orders.items(), reverse=True)

        most_popular_sell_price = min(sell_orders, key=lambda item: item[1])[0]
        most_popular_buy_price = max(buy_orders, key=lambda item: item[1])[0]
        
        # Calculate average of those prices
        return (most_popular_buy_price + most_popular_sell_price) // 2  
    
    #----------bigger strats------------------------
    def market_making_strategy(self, state: TradingState) -> None:
        """
        Implements the market making logic.
        """
        
        # Sort buy and sell orders
        buy_orders = sorted(
            state.order_depths[self.product].buy_orders.items(),
            reverse=True
        )
        sell_orders = sorted(
            state.order_depths[self.product].sell_orders.items()
        )

        # Calculate mid price and volatility
        mid_price = self.get_popular_average(state)
        self.mid_price_history.append(mid_price)
        volatility = self.estimate_volatility()

        # Calculate how much can be bought/sold
        position = state.position.get(self.product, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        default_price = self.get_default_price(state)
        spread_corr = round(self.volatility_multiplier * volatility)

        # Track if we are fully long or short over recent history
        self.history.append(abs(position) == self.limit)
        if len(self.history) > self.history_size:
            self.history.popleft()

        # Determine liquidation conditions
        soft_liquidate = (
            len(self.history) == self.history_size and 
            sum(self.history) >= self.history_size * self.soft_liquidate_thresh and 
            self.history[-1]
        )
        hard_liquidate = len(self.history) == self.history_size and all(self.history)

        # Adjust max_buy_price and min_sell_price based on current position
        max_buy_price = default_price - 1 - spread_corr if position > self.limit * 0.5 else default_price - spread_corr
        min_sell_price = default_price + 1 + spread_corr if position < self.limit * -0.5 else default_price + spread_corr

        # --- BUY LOGIC ---
        # 1. Buy from sell orders if below max_buy_price
        for price, volume in sell_orders:
            if price <= max_buy_price and to_buy > 0:
                quantity = min(-volume, to_buy)
                self.buy(price, quantity)
                to_buy -= quantity

        # 2. Hard liquidation (if fully short)
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(default_price, quantity)
            to_buy -= quantity

        # 3. Soft liquidation (if partially short)
        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(default_price - 2, quantity)
            to_buy -= quantity

        # 4. Normal buying based on the most popular buy price.
        if to_buy > 0:
            most_popular_price = max(buy_orders, key=lambda item: item[1])[0]
            price = min(max_buy_price, most_popular_price + 1)
            self.buy(price, to_buy)

        # --- SELL LOGIC ---
        # 1. Sell into buy orders if above min_sell_price.
        for price, volume in buy_orders:
            if price >= min_sell_price and to_sell > 0:
                quantity = min(volume, to_sell)
                self.sell(price, quantity)
                to_sell -= quantity

        # 2. Hard liquidation (if fully long)
        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(default_price, quantity)
            to_sell -= quantity

        # 3. Soft liquidation (if partially long)
        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(default_price + 2, quantity)
            to_sell -= quantity

        # 4. Normal selling based on the most popular sell price.
        if to_sell > 0:
            most_popular_price = min(sell_orders, key=lambda item: item[1])[0]
            price = max(min_sell_price, most_popular_price - 1)
            self.sell(price, to_sell)

    def trend_following_strategy(self, state: TradingState) -> None:
            """
            Implements the trend following logic based on price and simple moving average (SMA).
            If the price is above the moving average, buy; if below, sell.
            """
            # Get the current price (mid price) and the last 100 prices
            current_price = self.get_popular_average(state) #same results as EMA
            # current_price = self.get_EMA(state)


            # Retrieve the last 100 prices

            # Calculate the Simple Moving Average (SMA) of the last 100 prices
            sma = self.estimate_mean() #better than EMA
            # sma = self.estimate_EMA_mean()

            # Get current position
            position = state.position.get(self.product, 0)

            # If the price is above the moving average, we are in a bullish trend, so buy
            if current_price > sma:
                # Buy if we're not already at the limit position
                if position < self.limit:
                    to_buy = self.limit - position
                    self.buy(current_price, to_buy)

            # If the price is below the moving average, we are in a bearish trend, so sell
            elif current_price < sma:
                # Sell if we're not already at the limit position in the negative
                if position > -self.limit:
                    to_sell = self.limit + position
                    self.sell(current_price, to_sell)

    def mean_reversion_strategy(self, state: TradingState) -> None:
        """
        Implements a mean reversion strategy.
        Goes short if the current price is 1 standard deviation above the mean,
        and goes long if the price is 1 standard deviation below the mean.
        """
        # Get the current price (mid price)
        current_price = self.get_popular_average(state)

        # Estimate the mean of the price history
        mean_price = self.estimate_EMA_mean()

        # Estimate the standard deviation of the price history
        std_dev = self.estimate_volatility()

        # Get current position
        position = state.position.get(self.product, 0)

        upper_threshold = mean_price + std_dev
        lower_threshold = mean_price - std_dev

        # If current price is above (mean + 1 std), sell (go short)
        if current_price > upper_threshold:
            if position > -self.limit:
                to_sell = self.limit + position
                self.sell(current_price, to_sell)

        # If current price is below (mean - 1 std), buy (go long)
        elif current_price < lower_threshold:
            if position < self.limit:
                to_buy = self.limit - position
                self.buy(current_price, to_buy)

    def picnic_basket_strategy_noah(self, state: TradingState) -> None:
        """
        Implements arbitrage strategy based on weighted price difference
        between related products (like Picnic Baskets and components).
        """

        required_products = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
        }

        # If the product is not a basket or components missing — skip
        if self.product not in required_products:
            return

        component_weights = required_products[self.product]

        if any(prod not in state.order_depths for prod in [self.product] + list(component_weights.keys())):
            return

        # Calculate mid-prices
        mp_main = self.get_popular_average_basket(state, self.product)
        mp_components = {prod: self.get_popular_average_basket(state, prod) for prod in component_weights}

        # Calculate theoretical price difference
        theoretical_value = sum(component_weights[prod] * mp_components[prod] for prod in component_weights)
        diff = mp_main - theoretical_value

        # Threshold lookup from shared dictionary
        thresholds = {
            "CROISSANTS": (-50, 100),
            "JAMS": (-50, 100),
            "DJEMBES": (-50, 100),
            "PICNIC_BASKET2": (-50, 100),
            "PICNIC_BASKET1": (-50, 100),  # should add this too!
        }

        if self.product not in thresholds:
            return  # No thresholds for this product — skip

        long_threshold, short_threshold = thresholds[self.product]

        # Sort orders
        order_depth = state.order_depths[self.product]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.product, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # --- BUY LOGIC (if undervalued) ---
        if diff < long_threshold:
            for price, volume in sell_orders:
                if to_buy <= 0:
                    break
                quantity = min(-volume, to_buy)
                self.buy(price, quantity)
                to_buy -= quantity

        # --- SELL LOGIC (if overvalued) ---
        elif diff > short_threshold:
            for price, volume in buy_orders:
                if to_sell <= 0:
                    break
                quantity = min(volume, to_sell)
                self.sell(price, quantity)
                to_sell -= quantity


#-------------------- individual fair prices plus signals which strategy to choose-----------------------------
#
# ---------------------Round3 products--------------------------------    
class VulcanicRockStrategy(GeneralStrategy):
    def get_default_price(self, state: TradingState):
        return 0

    def act(self, state: TradingState):
        pass

class VulcanicRockVoucher(GeneralStrategy):
    def __init__(self, product, limit, strategies, strategy_args):
        super().__init__(product, limit, strategies, strategy_args)

        # default is 0 this needs to be set in every voucher subclass
        self.strike_price = 0


    def get_default_price(self, state: TradingState):
        return 0

    def act(self, state: TradingState):
        #----------------------
        #logic for trading options
        pass

class VulcanicRockVoucher9500(GeneralStrategy):
    def __init__(self, product, limit, strategies, strategy_args):
        super().__init__(product, limit, strategies, strategy_args)

        self.strike_price = 9500

class VulcanicRockVoucher9750(GeneralStrategy):
    def __init__(self, product, limit, strategies, strategy_args):
        super().__init__(product, limit, strategies, strategy_args)

        self.strike_price = 9750

class VulcanicRockVoucher10000(GeneralStrategy):
    def __init__(self, product, limit, strategies, strategy_args):
        super().__init__(product, limit, strategies, strategy_args)

        self.strike_price = 10000

class VulcanicRockVoucher10250(GeneralStrategy):
    def __init__(self, product, limit, strategies, strategy_args):
        super().__init__(product, limit, strategies, strategy_args)

        self.strike_price = 10250

class VulcanicRockVoucher10500(GeneralStrategy):
    def __init__(self, product, limit, strategies, strategy_args):
        super().__init__(product, limit, strategies, strategy_args)

        self.strike_price = 10500


# ---------------------Round2 products--------------------------------    
class CroissantStrategy(GeneralStrategy):
    def __init__(self, product: str, limit: int, strategies, strategy_args):
        super().__init__(product, limit, strategies, strategy_args)
        self.mid_price_history = deque(maxlen=1000)

    def get_default_price(self, state: TradingState) -> int:
        return 0

    def act(self, state: TradingState) -> None:
        self.market_making_strategy(state)


class JamStrategy(GeneralStrategy):
    def get_default_price(self, state: TradingState) -> int:
        return self.get_EMA(state)
    
    def act(self, state: TradingState) -> None:
        self.market_making_strategy(state)
    

class DjembeStrategy(GeneralStrategy):
    def __init__(self, product: str, limit: int, strategies, strategy_args):
        super().__init__(product, limit, strategies, strategy_args)
        self.mid_price_history = deque(maxlen=1000)
    def get_default_price(self, state: TradingState) -> int:
        return self.get_popular_average(state)
    
    def act(self, state: TradingState) -> None:
        self.market_making_strategy(state)


class Picnic_Basket_1_Strategy(GeneralStrategy):
    def get_default_price(self, state: TradingState) -> int:
        return self.get_popular_average(state)
    
    def act(self, state: TradingState) -> None:
        self.market_making_strategy(state)


class Picnic_Basket_2_Strategy(GeneralStrategy):
    def get_default_price(self, state: TradingState) -> int:
        return self.get_popular_average(state)
    
    def act(self, state: TradingState) -> None:
        self.market_making_strategy(state)


#-------------Round1 products-----------------------------

class RainForestResinStrategy(GeneralStrategy):
    def get_default_price(self, state: TradingState) -> int:
        return 10_000
    
    def act(self, state: TradingState) -> None:
        self.market_making_strategy(state)
    
class SquidInkStrategy(GeneralStrategy):
    def get_default_price(self, state) -> float:
        
        #return self.get_popular_average(state)

        order_depth = state.order_depths[self.product]

        #this gets the cheapest price we can buy at, and the highest price we can sell at
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        #creates a list of the prices of orders with high volumes --> market maker prices
        filtered_ask =  [price for price in order_depth.sell_orders.keys()
                         if abs(order_depth.sell_orders[price])
                         >= self.volume_threshold]

        filtered_bid = [price for price in order_depth.buy_orders.keys()
                         if abs(order_depth.buy_orders[price])
                         >= self.volume_threshold]

        #defines the ask and bid of the market maker as the best ask and bid of those prices        
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

        #if there is no market maker
        if mm_ask == None or mm_bid == None:
            if self.last_mm_mid_price == None:
                mm_mid_price = (best_ask + best_bid)/2
            else:
                mm_mid_price = self.last_mm_mid_price

        else:
            mm_mid_price = (mm_ask + mm_bid)/2

        #mean reversion
        if self.last_mm_mid_price != None:
            last_price = self.last_mm_mid_price
            last_returns = (mm_mid_price - last_price) / last_price
            pred_returns = ( 
                last_returns * -0.369 #this tries to predict how much 
            )
            fair = mm_mid_price + (mm_mid_price * pred_returns)
        else:
            fair = mm_mid_price

        self.last_mm_mid_price = mm_mid_price

        return fair

    def act(self, state: TradingState) -> None:
        self.market_making_strategy(state)

class KelpStrategy(GeneralStrategy):

    #for kelp try a marketmaking strategy with a dynamic default price
    def get_default_price(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.product]

        #this gets the cheapest price we can buy at, and the highest price we can sell at
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        #creates a list of the prices of orders with high volumes --> market maker prices
        filtered_ask =  [price for price in order_depth.sell_orders.keys()
                         if abs(order_depth.sell_orders[price])
                         >= self.volume_threshold]

        filtered_bid = [price for price in order_depth.buy_orders.keys()
                         if abs(order_depth.buy_orders[price])
                         >= self.volume_threshold]

        #defines the ask and bid of the market maker as the best ask and bid of those prices        
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

        #if there is no market maker
        if mm_ask == None or mm_bid == None:
            if self.last_mm_mid_price == None:
                mm_mid_price = (best_ask + best_bid)/2
            else:
                mm_mid_price = self.last_mm_mid_price

        else:
            mm_mid_price = (mm_ask + mm_bid)/2

        #mean reversion
        if self.last_mm_mid_price != None:
            last_price = self.last_mm_mid_price
            last_returns = (mm_mid_price - last_price) / last_price
            pred_returns = ( 
                last_returns * -0.5 #this tries to predict how much 
            )
            fair = mm_mid_price + (mm_mid_price * pred_returns)
        else:
            fair = mm_mid_price

        self.last_mm_mid_price = mm_mid_price

        return fair

    def act(self, state: TradingState) -> None:
        self.market_making_strategy(state)

#------- execute general Strategy-------------------
class Trader:
    def __init__(self, strategy_args = None) -> None:
        #Define the limits and goods traded in the given round, this need to be changed for every submission
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP" : 50,
            "SQUID_INK" : 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500" : 200,
            "VOLCANIC_ROCK_VOUCHER_9750" : 200,
            "VOLCANIC_ROCK_VOUCHER_10000" : 200,
            "VOLCANIC_ROCK_VOUCHER_10250" : 200,
            "VOLCANIC_ROCK_VOUCHER_10500" : 200
        }

        self.strategy_args = strategy_args if strategy_args != None else {}
        #Define a strategy for every product, this is done by creating an instance of a specific strategy for every product
        self.strategies = {}

        for symbol, strategyClass in {
            "RAINFOREST_RESIN" : RainForestResinStrategy,
            "KELP" : KelpStrategy,
            "SQUID_INK": SquidInkStrategy,
            "CROISSANTS": CroissantStrategy,
            "JAMS": JamStrategy,
            "DJEMBES": DjembeStrategy,
            "PICNIC_BASKET1": Picnic_Basket_1_Strategy,
            "PICNIC_BASKET2": Picnic_Basket_2_Strategy,
            "VOLCANIC_ROCK": VulcanicRockStrategy,
            "VOLCANIC_ROCK_VOUCHER_9500" : VulcanicRockVoucher9500,
            "VOLCANIC_ROCK_VOUCHER_9750" : VulcanicRockVoucher9750,
            "VOLCANIC_ROCK_VOUCHER_10000" : VulcanicRockVoucher10000,
            "VOLCANIC_ROCK_VOUCHER_10250" : VulcanicRockVoucher10250,
            "VOLCANIC_ROCK_VOUCHER_10500" : VulcanicRockVoucher10500
        }.items():
            self.strategies.update({symbol : strategyClass(symbol, limits[symbol], self.strategies, self.strategy_args.get(symbol, {}))})


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0

        #saves what ever comes from the previous iteration into a dictionary (if empty its just a empty dictionary)
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        #creates empty dictionary
        new_trader_data = {}

        #need to modify the input of parameters into a specific strategy

        #iterate over every product
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                #this just makes sure the current iteration gets the data of the old iteration if important
                #not important in this case right now
                strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                result[symbol] = strategy.run(state)

            #write the data from the current iteration into the new_trader_data dictionary under the current product
            #in the case of the marketmaking strategy this is just the current 
            new_trader_data[symbol] = strategy.save()

        #convert the dictionary with the current trader_data back into a string with the format of a dictionary
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data