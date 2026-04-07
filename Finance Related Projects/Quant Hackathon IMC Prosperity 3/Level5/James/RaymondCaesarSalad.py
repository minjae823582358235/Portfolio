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
MAGNIFICENT_MACARONS = 'MAGNIFICENT_MACARONS'

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

# feeDict={OTARIFF:state.observations.exportTariff,
#          ITARIFF:state.observations.importTariff,
#          SUNLIGHT:state.observations.sunlightIndex,
#          SUGAR:state.observations.sugarPrice}

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
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_9750: {
        "starting_time_to_expiry": 245 / 250,
        "strike": 9500,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_10000: {
        "starting_time_to_expiry": 245 / 250,
        "strike": 10000,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_10250: {
        "starting_time_to_expiry": 245 / 250,
        "strike": 10250,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    VOLCANIC_ROCK_VOUCHER_10500: {
        "starting_time_to_expiry": 245 / 250,
        "strike": 10500,
        "std_window": 10,
        "implied_volatility": 0.16,
    },
    MAGNIFICENT_MACARONS: {
        "FILL": 1,
    }
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
            "MAGNIFICENT_MACARONS": 0,
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
        self.macaron_timestamps = []
        self.macaron_mid_prices = []

        self.macaron_short_price = []
        self.macaron_long_price = []
        self.best_internal_ask = 0
        self.best_internal_bid = 0

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
        elif product == "MAGNIFICENT_MACARONS":
            self.macaron_timestamps.append(state.timestamp)
            self.macaron_mid_prices.append(mid)



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
                             position_limit: int, result: Dict[str, List[Order]], state: TradingState,
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
        sell_volume = -position_limit - current_position

        # Buy when price is below the lower band - undervalued
        if current_mid < lower_band:
            result[product].append(Order(product, best_ask, buy_volume))
            # result[product] += self.OrderOptimised(product, buy_volume, mode='buy', state=state)
        # Sell when price is above the upper band - overvalued
        elif current_mid > upper_band:
            result[product].append(Order(product, best_bid, sell_volume))
            # result[product] += self.OrderOptimised(product, sell_volume, mode='sell', state=state)



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



        if VOLCANIC_ROCK_VOUCHER_10000 in state.order_depths:
            product = VOLCANIC_ROCK_VOUCHER_10000
            position = state.position.get(product, 0)
            depth = state.order_depths[product]
            limit = LIMIT[product]

            # Ensure market trades exist for this product
            if product in state.market_trades:
                for trade in state.market_trades[product]:
                    if trade.buyer == "Caesar":
                        best_bid = int(trade.price)
                        buy_volume = int(trade.quantity)
                        # buy_volume = limit - position

                        # If Caesar bought at a high price — follow (buy)
                        if trade.buyer == "Caesar" and trade.price > best_bid and position < limit:
                            result[product].append(Order(product, best_bid, buy_volume))

                    elif trade.seller == "Caesar":
                        best_ask = int(trade.price)
                        sell_volume = int(trade.quantity)
                        # sell_volume = position + limit

                        # If Caesar sold at a low price — follow (sell)
                        if trade.seller == "Caesar" and trade.price <= best_ask and position > -limit:
                            result[product].append(Order(product, best_ask, -sell_volume))



        # -------- Magnificent Macarons: Arbitrage Between Islands --------
        # if MAGNIFICENT_MACARONS in state.order_depths:
        #     product = MAGNIFICENT_MACARONS
        #     macaron_position = state.position.get(product, 0)
        #     macaron_order_depth = state.order_depths[product]
        #     macaron_limit = LIMIT[product]['limit']
        #     macaron_conversion = LIMIT[product]['conversion']
            
        #     # If currently shorted - convert macarons back to avoid holding
        #     if (macaron_position >= -macaron_limit) and (macaron_position < 0):
        #         conversions = -macaron_position

        #     # Intenal prices
        #     mid_price = vwap(product,state)
        #     orders_mac = []
        #     best_bid = max(macaron_order_depth.buy_orders.keys())
        #     best_ask = min(macaron_order_depth.sell_orders.keys())
        #     # Update internal prices
        #     self.macaron_short_price.append(self.best_internal_ask)
        #     self.macaron_long_price.append(self.best_internal_bid)

        #     # Get bid, ask, tariffs, sugar, and sunlight for conversions
        #     observation = state.observations.conversionObservations[product]
        #     bid_price = observation.bidPrice
        #     ask_price = observation.askPrice
        #     transport_fees = observation.transportFees
        #     export_tariff = observation.exportTariff
        #     import_tariff = observation.importTariff
        #     sugar_price = observation.sugarPrice  # THINK WHAT WE CAN DO WITH THIS
        #     sunlight_index = observation.sunlightIndex  # THINK WHAT WE CAN DO WITH THIS

        #     # Calculate total costs including tariffs
        #     total_cost_price = ask_price + transport_fees + import_tariff
        #     total_sale_price = bid_price - export_tariff - transport_fees



        #     # Arbitrage if best internal selling price is higher than the external
        #     if max(self.macaron_short_price) > total_cost_price and total_cost_price > 0:
        #         conversions = -macaron_position
        #         self.macaron_short_price.clear()  # Reset list after conversions

        #     # Update internal prices
        #     self.best_internal_bid = best_ask  # Internal selling price - set threshold based on current cheapest
        #     self.best_internal_ask = best_bid  # Internal buying price - set threshold based on current highest

        #     # Offload inventory if we are long and under mid price
        #     if macaron_position >= -macaron_conversion and macaron_position <= 0:
        #         result[product].append(Order(product, int(mid_price - 1), -macaron_conversion))
        #     elif macaron_position >= -macaron_conversion:
        #         result[product].append(Order(product, int(mid_price - 1), -macaron_conversion))



        # Convert orders to the format expected by the logger.
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data