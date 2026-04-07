import math
import statistics
import json
import numpy as np
import jsonpickle
from statistics import NormalDist
from typing import Any, Dict, List, Tuple
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# ------------------ Logger for Visualiser ------------------
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
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
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state,
                        self.truncate(state.traderData, max_item_length)
                        if hasattr(state, "traderData")
                        else ""
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        return {symbol: [order_depth.buy_orders, order_depth.sell_orders] for symbol, order_depth in order_depths.items()}

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        return [
            [
                trade.symbol,
                trade.price,
                trade.quantity,
                trade.buyer,
                trade.seller,
                trade.timestamp,
            ]
            for trade_list in trades.values()
            for trade in trade_list
        ]

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {
            product: [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
            for product, observation in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        return [[order.symbol, order.price, order.quantity] for order_list in orders.values() for order in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[: max_length - 3] + "..."


logger = Logger()
# ------------------ Logger for Visualiser ------------------



RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


PARAMS = {
    RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 25,
    },
    KELP: {
        "take_width": 1,
        "clear_width": -0.25,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    SQUID_INK: {
        "take_width": 1,
        "clear_width": -0.25,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    VOLCANIC_ROCK: {
        "starting_time_to_expiry": 245 / 250,
        "std_window": 10,
    },
    VOLCANIC_ROCK_VOUCHER_9500: {
        "starting_time_to_expiry": 245 / 250,
        "strike": 9500,
        "std_window": 10,
        "implied_volatility": 0.15,
    },
    VOLCANIC_ROCK_VOUCHER_9750: {
        "starting_time_to_expiry": 245 / 250,
        "strike": 9500,
        "std_window": 10,
        "implied_volatility": 0.15,
    },
    VOLCANIC_ROCK_VOUCHER_10000: {
        "starting_time_to_expiry": 245 / 250,
        "strike": 10000,
        "std_window": 10,
        "implied_volatility": 0.15,
    },
    VOLCANIC_ROCK_VOUCHER_10250: {
        "starting_time_to_expiry": 245 / 250,
        "strike": 10250,
        "std_window": 10,
        "implied_volatility": 0.15,
    },
    VOLCANIC_ROCK_VOUCHER_10500: {
        "starting_time_to_expiry": 245 / 250,
        "strike": 10500,
        "std_window": 10,
        "implied_volatility": 0.15,
    },
}



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





# ----------------------- Trader Class -----------------------
class Trader:
    
    def __init__(self, params=None):
        # Initialise positions and limits for all assets
        self.position = {
            "RAINFOREST_RESIN": 0,
            "KELP": 0,
            "SQUID_INK": 0,
            "VOLCANIC_ROCK": 0,
            "VOLCANIC_ROCK_VOUCHER_9500": 0,
            "VOLCANIC_ROCK_VOUCHER_9750": 0,
            "VOLCANIC_ROCK_VOUCHER_10000": 0,
            "VOLCANIC_ROCK_VOUCHER_10250": 0,
            "VOLCANIC_ROCK_VOUCHER_10500": 0,
        }
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }

        # Set parameters so that they can be easily accessed via self.params
        self.params = PARAMS

        # Resin parameters - for market making
        self.resin_timestamps = []
        self.resin_mid_prices = []
        
        # Kelp and Ink parameters - for pairs trading
        self.kelp_timestamps = []
        self.kelp_mid_prices = []
        self.ink_timestamps = []
        self.ink_mid_prices = []
        self.rock_timestamps = []
        self.rock_mid_prices = []
        self.rock9750_timestamps = []
        self.rock9750_mid_prices = []
        self.rock10000_timestamps = []
        self.rock10000_mid_prices = []
        self.rock10250_timestamps = []
        self.rock10250_mid_prices = []
        self.rock10500_timestamps = []
        self.rock10500_mid_prices = []

        # Avellaneda–Stoikov parameters
        self.as_params = {
            "RAINFOREST_RESIN": {
                "gamma": 0.666162234599478,
                "sigma": 0.38344697684007834,
                "k": 0.38711859174678465,
                "max_order_size": 40,
                "T": 1,
                "limit": 50,
                "buffer": 2,
            }
        }

    def mid_price(self, order_depth: OrderDepth) -> float:
        # Compute a mid-price using available order depth information
        if order_depth.sell_orders:
            total_ask = sum(price * quantity for price, quantity in order_depth.sell_orders.items())
            total_qty = sum(quantity for quantity in order_depth.sell_orders.values())
            m1 = total_ask / total_qty
        else:
            m1 = 0

        if order_depth.buy_orders:
            total_bid = sum(price * quantity for price, quantity in order_depth.buy_orders.items())
            total_qty = sum(quantity for quantity in order_depth.buy_orders.values())
            m2 = total_bid / total_qty
        else:
            m2 = 0
            
        return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)
    
    def update_market_data(self, product: str, state: TradingState) -> None:
        # Update timestamp and mid-price for each asset
        order_depth = state.order_depths[product]
        mid = self.mid_price(order_depth)
        if product == "RAINFOREST_RESIN":
            self.resin_timestamps.append(state.timestamp)
            self.resin_mid_prices.append(mid)
        elif product == "KELP":
            self.kelp_timestamps.append(state.timestamp)
            self.kelp_mid_prices.append(mid)
        elif product == "SQUID_INK":
            self.ink_timestamps.append(state.timestamp)
            self.ink_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK":
            self.rock_timestamps.append(state.timestamp)
            self.rock_mid_prices.append(mid)
        elif product == "VOLCANIC_ROCK_VOUCHER_9500":
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

    def black_scholes_call(self, S, K, vol, T, r = 0, q = 0):
        """
        Black Scholes formula for European call option pricing - CHECK IF YOU WANT TO RETURN DELTA
        """
        # S = Current stock price
        # K = Option striking price
        # r = risk free interest
        # q = dividend yield - JUST 0 FOR NOW, MAYBE LATER IDK
        # vol = standard deviation of stock (volatility)
        # T = time until option expiration
        # Black Scholes - calculates theoretical call premium

        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)

        norm_dist = NormalDist().cdf
        norm_pdf = NormalDist().pdf
        N1 = norm_dist(d1)
        N2 = norm_dist(d2)

        # Call option price
        call_price = S*N1 - K*np.exp((q-r) * T)*N2

        # Delta - how much option price changes per $1 change in stock price
        delta = norm_dist(d1)

        # Gamma - how much delta changes per $1 change in stock price - like acceleration - if we know distribution we can use this to capture 
        gamma = norm_pdf(d1) / (S * vol * np.sqrt(T))
        
        # Vega - how much option price changes per 1% change in volatility - like sensitivity
        vega = S * norm_pdf(d1) * np.sqrt(T)

        return call_price, delta, gamma, vega

    # def calc_implied_vol(self, initial_vol: float, S: float, K: float, market_call_price: float, T: float, step: float = 0.00001) -> float:
    #     """
    #     Iteratively calculate the implied volatility that matches the market call price.
    #     """
    #     vol = initial_vol
    #     for _ in range(500):
    #         call_price_new, _, _, _ = self.black_scholes_call(S, K, vol, T)
    #         if abs(call_price_new - market_call_price) < 0.001:
    #             break
    #         vol += (market_call_price - call_price_new) * step
    #     return vol

    def calc_fair_price(self, T) -> Dict[str, Tuple[int, int]]:
        """
        Calculate fair call option prices for the volcanic rock vouchers using the Black–Scholes formula.
        """
        if not self.rock_mid_prices:
            return {}

        S = self.rock_mid_prices[-1]  # most recent mid-price for VOLCANIC_ROCK
        fair_prices = {}

        for voucher in [VOLCANIC_ROCK_VOUCHER_9500,
                        VOLCANIC_ROCK_VOUCHER_9750,
                        VOLCANIC_ROCK_VOUCHER_10000,
                        VOLCANIC_ROCK_VOUCHER_10250,
                        VOLCANIC_ROCK_VOUCHER_10500]:

            # Get mid price for the specific voucher
            if voucher == VOLCANIC_ROCK_VOUCHER_9500:
                mid_price = self.rock9750_mid_prices[-1]
            elif voucher == VOLCANIC_ROCK_VOUCHER_10000:
                mid_price = self.rock10000_mid_prices[-1]
            elif voucher == VOLCANIC_ROCK_VOUCHER_10250:
                mid_price = self.rock10250_mid_prices[-1]
            elif voucher == VOLCANIC_ROCK_VOUCHER_10500:
                mid_price = self.rock10500_mid_prices[-1]

            # Get the parameters for the specific voucher
            vol = self.params[voucher]["implied_volatility"]
            strike = int(voucher.split("_")[-1])

            call_price, delta, gamma, vega = self.black_scholes_call(S, strike, vol, T)
            fair_prices[voucher] = (strike, mid_price, call_price, delta, gamma, vega)
        return fair_prices


    def OrderOptimised(self, product: str, size: int, mode: str, state: TradingState) -> list[Order]:
        orders = []
        VolTarget = size
        depth = state.order_depths[product]

        if mode == 'buy':
            # Ensure there are sell orders available; if not, return an empty list.
            if not depth.sell_orders:
                return orders
            sell_orders = depth.sell_orders  # Use the correct attribute name
            # Get the prices sorted in ascending order (lowest offers first)
            sorted_prices = sorted(sell_orders.keys())
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

    
    def mean_reversion_trade(self, product: str, mid_prices: List[float],
                             order_depth: OrderDepth, current_position: int,
                             position_limit: int, result: Dict[str, List[Order]], state:TradingState,
                             window: int = 50, z_score_thresh: float = 2.0) -> None:
        # Only run if we have enough price history
        if len(mid_prices) < window:
            return

        recent_prices = mid_prices[-window:]
        sma = statistics.mean(recent_prices)
        std_dev = statistics.stdev(recent_prices)

        # Calculate Bollinger Bands
        lower_band = sma - (z_score_thresh * std_dev)
        upper_band = sma + (z_score_thresh * std_dev)
        current_mid = mid_prices[-1]

        # Ensure there are available order levels
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        buy_volume = position_limit - current_position
        sell_volume = abs(-position_limit - current_position)

        # Buy when price is below the lower band - undervalued
        if current_mid < lower_band:
            result[product]+=self.OrderOptimised(product, buy_volume, 'buy', state)
        # Sell when price is above the upper band - overvalued
        elif current_mid > upper_band:
            result[product]+=self.OrderOptimised(product, sell_volume, 'sell', state)



    def OrderOptimised(self, product: str, size: int,mode:str, state: TradingState) -> list[Order]:
            orders=[]
            VolTarget=size
            if mode == 'buy':
                if product not in state.order_depths[product].sell_orders:
                    orders.append(Order(product,int(vwap(product,state)),VolTarget))
                    return orders
                #Ascending
                SellOrders=state.order_depths[product].sellorders
                Prices=[key for key, in sorted(SellOrders.items(),reverse=True)]
                for price in Prices:
                    orders.append(Order(product,price,min(VolTarget,abs(SellOrders[price]))))
                    VolTarget-=abs(SellOrders[price])
                    if VolTarget<=0:
                        break
            if mode == 'sell':
                BuyOrders=state.order_depths[product].buy_orders
                if product not in BuyOrders:
                    orders.append(Order(product,int(vwap(product,state)),-VolTarget))
                    return orders
                Prices=[key for key, in sorted(BuyOrders.items(),reverse=True)]
                for price in Prices:
                    orders.append(Order(product,price,-min(VolTarget,abs(BuyOrders[price]))))
                    VolTarget-=abs(BuyOrders[price])
                    if VolTarget<=0:
                        break
                    return orders



    def run(self, state: TradingState):
        # Initialise result dict for orders
        result = {product: [] for product in state.order_depths.keys()}
        conversions = 0
        trader_data = ""  # Placeholder for trader data - for visualisation

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        # Update positions from state.
        for product in state.position:
            self.position[product] = state.position[product]
            
        # Save current timestamp and mid-price for all assets.
        for product in state.order_depths:
            try:
                self.update_market_data(product, state)
            except Exception:
                pass
        


        # -------- Volcanic Rock: Options Trading with Black-Scholes and mean reversion for underlying: MAX STUFF WE CAN BUY SELL --------
        if VOLCANIC_ROCK in state.order_depths:
            rock_position = state.position.get(VOLCANIC_ROCK, 0)
            rock_params = PARAMS[VOLCANIC_ROCK]
            rock_order_depth = state.order_depths[VOLCANIC_ROCK]

            # Calculate time to expiry
            time_expiry = (PARAMS[VOLCANIC_ROCK]["starting_time_to_expiry"] - (state.timestamp) / 1000000 / 250)

            # Retrieve fair call option prices for volcanic rock vouchers
            fair_prices = self.calc_fair_price(T=time_expiry)
            undercut = 75  # undercut for triggering orders more aggressively - OPTIMISE

            for voucher, (strike, mid_price, bs_price, bs_delta, bs_gamma, bs_vega) in fair_prices.items():
                if voucher in state.order_depths:
                    voucher_order_depth = state.order_depths[voucher]

                    buy_order_depth = voucher_order_depth.buy_orders
                    sell_order_depth = voucher_order_depth.sell_orders

                    # Skip if no orders exist
                    if not buy_order_depth or not sell_order_depth:
                        continue
                    else:
                        rock_mid = self.rock_mid_prices[-1]

                        # If orders exist - calculate the best bid and ask prices
                        voucher_bid = max(buy_order_depth.keys())
                        voucher_ask = min(sell_order_depth.keys())

                        # Calculate allowed trading volumes for voucher
                        voucher_position = self.position[voucher]
                        voucher_buy_volume = self.position_limits[voucher] - voucher_position
                        voucher_sell_volume = self.position_limits[voucher] - voucher_position

                        # # Compare the Black–Scholes price to market offers
                        # if mid_price < bs_price - undercut:
                        #     result[voucher].append(Order(voucher, voucher_bid, -abs(voucher_sell_volume)))  # Short
                        # elif mid_price > bs_price + undercut:
                        #     result[voucher].append(Order(voucher, voucher_ask, abs(voucher_buy_volume)))  # Long      


                        # # Fill orders more aggressively
                        # if mid_price < bs_price - undercut:
                        #     # For sell side: get orders to sell the voucher
                        #     sell_orders = self.OrderOptimised(voucher, voucher_sell_volume, mode='sell', state=state)
                        #     for order in sell_orders:
                        #         result[voucher].append(order)
                        # elif mid_price > bs_price + undercut:
                        #     # For buy side: get orders to buy the voucher
                        #     buy_orders = self.OrderOptimised(voucher, voucher_buy_volume, mode='buy', state=state)
                        #     for order in buy_orders:
                        #         result[voucher].append(order)



            # -------- Mean Reversion - Based on historical prices (Bollinger Bands) --------
            # window = 50  # OPTIMISE THIS WINDOW
            # if len(self.rock_mid_prices) >= window:
            #     recent_prices = self.rock_mid_prices[-window:]
            #     sma = statistics.mean(recent_prices)
            #     std_dev = statistics.stdev(recent_prices)
            #     z_score_thresh = 2  # OPTIMISE THIS THRESHOLD

            #     # Calculate Bollinger Bands
            #     lower_band = sma - (z_score_thresh * std_dev)
            #     upper_band = sma + (z_score_thresh * std_dev)

            #     rock_mid = self.rock_mid_prices[-1]  # most recent mid price for VOLCANIC_ROCK
            #     rock_bid = max(rock_order_depth.buy_orders.keys())
            #     rock_ask = min(rock_order_depth.sell_orders.keys())

            #     rock_buy_volume = self.position_limits[VOLCANIC_ROCK] - rock_position
            #     rock_sell_volume = -self.position_limits[VOLCANIC_ROCK] - rock_position

            #     # Buy VOLCANIC_ROCK when price is below the lower band
            #     if rock_mid < lower_band:
            #         result[VOLCANIC_ROCK].append(Order(VOLCANIC_ROCK, rock_ask, rock_buy_volume))
            #     # Sell VOLCANIC_ROCK when price is above the upper band
            #     elif rock_mid > upper_band:
            #         result[VOLCANIC_ROCK].append(Order(VOLCANIC_ROCK, rock_bid, rock_sell_volume))
            # ---------------------------------------------------------------------

            rock_price_dict = {
                VOLCANIC_ROCK: (self.rock_mid_prices,
                                state.order_depths[VOLCANIC_ROCK],
                                state.position.get(VOLCANIC_ROCK, 0),
                                self.position_limits[VOLCANIC_ROCK]),
                VOLCANIC_ROCK_VOUCHER_9500: (self.rock9750_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_9500],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_9500, 0),
                                            self.position_limits[VOLCANIC_ROCK_VOUCHER_9500]),
                VOLCANIC_ROCK_VOUCHER_9750: (self.rock9750_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_9750],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_9750, 0),
                                            self.position_limits[VOLCANIC_ROCK_VOUCHER_9750]),
                VOLCANIC_ROCK_VOUCHER_10000: (self.rock10000_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_10000],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_10000, 0),
                                            self.position_limits[VOLCANIC_ROCK_VOUCHER_10000]),
                VOLCANIC_ROCK_VOUCHER_10250: (self.rock10250_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_10250],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_10250, 0),
                                            self.position_limits[VOLCANIC_ROCK_VOUCHER_10250]),
                VOLCANIC_ROCK_VOUCHER_10500: (self.rock10500_mid_prices,
                                            state.order_depths[VOLCANIC_ROCK_VOUCHER_10500],
                                            state.position.get(VOLCANIC_ROCK_VOUCHER_10500, 0),
                                            self.position_limits[VOLCANIC_ROCK_VOUCHER_10500]),
            }

            # Loop over each product and apply the mean reversion strategy.
            for product, (mid_prices, o_depth, pos, pos_limit) in rock_price_dict.items():
                self.mean_reversion_trade(product, mid_prices, o_depth, pos, pos_limit, result, window=50, z_score_thresh=2.0,state=state)













        # Convert orders to the format expected by the logger.
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
