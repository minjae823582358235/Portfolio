import math
import statistics
import json
import numpy as np
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



# -------- Noah's code --------
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 0.5,
        "join_edge": 2,
        "default_edge": 2,
        "soft_position_limit": 45,
    },
    Product.KELP: {
        "take_width": 1,
        "position_limit": 50,
        "min_volume_filter": 20,
        "spread_edge": 1,
        "default_fair_method": "vwap_with_vol_filter",
    },
    # New parameters for the SQUID_INK mean reversion strategy
    Product.SQUID_INK: {
        "z_threshold": 2,         # The z-score threshold
        "history_length": 50,      # Number of historical mid–prices to use
        # The RSI parameters are no longer used in our mean-reversion strategy
        "rsi_period": 37,
        "rsi_overbought": 65,
        "rsi_oversold": 28,
        "rsi_trade_size": 18,
    },
    Product.CROISSANTS: {
        "history_length": 100,  # Number of mid-price datapoints to use for z-score calculation.
        "z_threshold": 2.5        # Threshold for trading.
    }
}

LIMIT = {
    Product.RAINFOREST_RESIN: 50,
    Product.KELP: 50,
    Product.SQUID_INK: 50,
    Product.CROISSANTS: 250,
}

# ---------------------------
# RAINFOREST_RESIN Strategy Functions (from round1rsi)
# ---------------------------
def resin_take_orders(
    order_depth: OrderDepth, fair_value: float, position: int, position_limit: int
) -> Tuple[List[Order], int, int]:
    orders = []
    buy_order_volume = 0
    sell_order_volume = 0
    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask < fair_value:
            quantity = min(best_ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, best_ask, quantity))
                buy_order_volume += quantity
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        if best_bid > fair_value:
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order(Product.RAINFOREST_RESIN, best_bid, -quantity))
                sell_order_volume += quantity
    return orders, buy_order_volume, sell_order_volume

def resin_clear_orders(
    order_depth: OrderDepth,
    position: int,
    fair_value: float,
    position_limit: int,
    buy_order_volume: int,
    sell_order_volume: int,
) -> Tuple[List[Order], int, int]:
    orders = []
    position_after_take = position + buy_order_volume - sell_order_volume
    fair_for_bid = math.floor(fair_value)
    fair_for_ask = math.ceil(fair_value)
    buy_quantity = position_limit - (position + buy_order_volume)
    sell_quantity = position_limit + (position - sell_order_volume)
    if position_after_take > 0:
        if fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(Product.RAINFOREST_RESIN, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(Product.RAINFOREST_RESIN, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)
    return orders, buy_order_volume, sell_order_volume

def resin_make_orders(
    order_depth: OrderDepth,
    fair_value: float,
    position: int,
    position_limit: int,
    buy_order_volume: int,
    sell_order_volume: int,
) -> List[Order]:
    orders = []
    aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
    bbbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
    baaf = min(aaf) if aaf else fair_value + 2
    bbbf_val = max(bbbf) if bbbf else fair_value - 2
    buy_quantity = position_limit - (position + buy_order_volume)
    if buy_quantity > 0:
        orders.append(Order(Product.RAINFOREST_RESIN, bbbf_val + 1, buy_quantity))
    sell_quantity = position_limit + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(Product.RAINFOREST_RESIN, baaf - 1, -sell_quantity))
    return orders

# ---------------------------
# KELP Strategy Functions (from round1rsi)
# ---------------------------
def kelp_fair_value(
    order_depth: OrderDepth, method: str = "vwap_with_vol_filter", min_vol: int = 20
) -> float:
    if method == "mid_price":
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2
    elif method == "mid_price_with_vol_filter":
        sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
        buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
        if not sell_orders or not buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
        else:
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
        return (best_ask + best_bid) / 2
    elif method == "vwap":
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
        if volume == 0:
            return (best_ask + best_bid) / 2
        return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
    elif method == "vwap_with_vol_filter":
        sell_orders = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]
        buy_orders = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]
        if not sell_orders or not buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            if volume == 0:
                return (best_ask + best_bid) / 2
            return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
        else:
            best_ask = min(sell_orders)
            best_bid = max(buy_orders)
            volume = -order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            if volume == 0:
                return (best_ask + best_bid) / 2
            return ((best_bid * (-order_depth.sell_orders[best_ask])) + (best_ask * order_depth.buy_orders[best_bid])) / volume
    else:
        raise ValueError("Unknown fair value method specified.")

def kelp_take_orders(
    order_depth: OrderDepth, fair_value: float, params: dict, position: int
) -> Tuple[List[Order], int, int]:
    orders = []
    buy_order_volume = 0
    sell_order_volume = 0
    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask <= fair_value - params["take_width"] and ask_amount <= 50:
            quantity = min(ask_amount, params["position_limit"] - position)
            if quantity > 0:
                orders.append(Order(Product.KELP, best_ask, quantity))
                buy_order_volume += quantity
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        bid_amount = order_depth.buy_orders[best_bid]
        if best_bid >= fair_value + params["take_width"] and bid_amount <= 50:
            quantity = min(bid_amount, params["position_limit"] + position)
            if quantity > 0:
                orders.append(Order(Product.KELP, best_bid, -quantity))
                sell_order_volume += quantity
    return orders, buy_order_volume, sell_order_volume

def kelp_clear_orders(
    order_depth: OrderDepth,
    position: int,
    params: dict,
    fair_value: float,
    buy_order_volume: int,
    sell_order_volume: int,
) -> Tuple[List[Order], int, int]:
    orders = []
    position_after_take = position + buy_order_volume - sell_order_volume
    fair_for_bid = math.floor(fair_value)
    fair_for_ask = math.ceil(fair_value)
    buy_quantity = params["position_limit"] - (position + buy_order_volume)
    sell_quantity = params["position_limit"] + (position - sell_order_volume)
    if position_after_take > 0:
        if fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(Product.KELP, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)
    if position_after_take < 0:
        if fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(Product.KELP, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)
    return orders, buy_order_volume, sell_order_volume

def kelp_make_orders(
    order_depth: OrderDepth,
    fair_value: float,
    position: int,
    params: dict,
    buy_order_volume: int,
    sell_order_volume: int,
) -> List[Order]:
    orders = []
    edge = params["spread_edge"]
    aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + edge]
    bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - edge]
    baaf = min(aaf) if aaf else fair_value + edge + 1
    bbbf = max(bbf) if bbf else fair_value - edge - 1
    buy_quantity = params["position_limit"] - (position + buy_order_volume)
    if buy_quantity > 0:
        orders.append(Order(Product.KELP, bbbf + 1, buy_quantity))
    sell_quantity = params["position_limit"] + (position - sell_order_volume)
    if sell_quantity > 0:
        orders.append(Order(Product.KELP, baaf - 1, -sell_quantity))
    return orders
# ------------------------------------------------------------









# ----------------------- Trader Class -----------------------
class Trader:
    
    def __init__(self, params=None):
        # Initialise positions and limits for all assets
        self.position = {"RAINFOREST_RESIN": 0, "KELP": 0, "SQUID_INK": 0}
        self.position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50}

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
            },
            "SQUID_INK": {
                "gamma": 1.3074794082080743,
                "sigma": 1.1391336142619657,
                "k": 2.65639863217165,
                "max_order_size": 10,
                "T": 1.0,
                "limit": 50,
                "buffer": 2,
            }
        }

        # Resin parameters - for market making
        self.resin_timestamps = []
        self.resin_mid_prices = []
        
        # Kelp and Ink parameters - for pairs trading
        self.kelp_timestamps = []
        self.kelp_mid_prices = []
        self.ink_timestamps = []
        self.ink_mid_prices = []



        # Ink and Kelp parameters - for pairs trading
        self.InkPreviousPriceArr = []
        self.KelpPreviousPriceArr = []
        self.m = None
        self.c = None
        self.mean = None
        self.stddev = None
        self.lookbackwindow = 10
        self.previousInkprice = None
        self.previousKelpprice = None


        # For Resin and Kelp formatting - CHANGE LATER
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = LIMIT



        # Picnic Basket parameters
        self.previousposition = {
            "KELP": 0,
            "RAINFOREST_RESIN": 0,
            "SQUID_INK": 0,
            "CROISSANTS": 0,
            "JAMS": 0,
            "DJEMBES": 0,
            "PICNIC_BASKET1": 0,
            "PICNIC_BASKET2": 0,
        }
        self.previouspositionCounter = {
            "KELP": 0,
            "RAINFOREST_RESIN": 0,
            "SQUID_INK": 0,
            "CROISSANTS": 0,
            "JAMS": 0,
            "DJEMBES": 0,
            "PICNIC_BASKET1": 0,
            "PICNIC_BASKET2": 0,
        }
        self.spread1arr = []
        self.spread2arr = []
        return None



    def mid_price(self, order_depth):
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
    
    def update_market_data(self, product, state):
        # Store current timestamp and mid-price for all assets
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



    # --------------------- RESIN LOGIC ---------------------
    def avellaneda_stoikov(self, product, mid, inventory):
        '''
        Calculate bid and ask prices around the mid-price/reservation price whilst adjusting based on current inventory.
        '''
        # Define product specific parameters Avellaneda-Stoikov parameters
        params = self.as_params.get(product, self.as_params["RAINFOREST_RESIN"])
        gamma = params["gamma"]
        sigma = params["sigma"]
        k_val = params["k"]
        max_order_size = params["max_order_size"]
        T = params["T"]
        limit = params["limit"]
        buffer_val = params["buffer"]

        # Reservation price - adjust the mid price based on inventory risk
        reservation_price = mid - inventory * gamma * (sigma ** 2) * T

        # Optimal spread - adjusted based on the risk aversion parameter and order intensity
        optimal_spread = (2 / gamma) * math.log(1 + gamma / k_val)

        # Set final bid and ask around the reservation price
        bid = reservation_price - optimal_spread / 2
        ask = reservation_price + optimal_spread / 2

        # Dynamic order sizing - reduce order size if close to limit to avoid exceeding position limits
        order_size = max(1, min(max_order_size, (limit - abs(inventory)) // buffer_val))

        return [Order(product, int(round(bid)), order_size),
                Order(product, int(round(ask)), -order_size)]
    # ---------------------------------------------------------









    # -------- TIDY ALL CODE AFTER THIS ROUND - LOW PRIORITY --------
    
    # -------- OPTIMIZE ALL PARAMETERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # --------------------- TRADING LOGIC ---------------------
    def run(self, state: TradingState):
        # Initialise result dict for orders
        result = {product: [] for product in state.order_depths.keys()}
        conversions = 0
        trader_data = ""  # Placeholder for trader data - for visualiser

        # TIDY THIS LATER
        kelporder = []
        inkorder = []
        conversions = 0
        OrderbookDict = state.order_depths

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        # --------------------------------



        # -------- UPDATE POSITION AND PRICES ---------
    
        # Update current positions from the state
        for product in state.position:
            self.position[product] = state.position[product]
            
        # Save current timestamp and mid-price for all assets - for algo use
        for product in state.order_depths:
            self.update_market_data(product, state)
        # -----------------------------------------------
        


        # -------- Resin: Avellaneda-Stoikov Market Making --------
        # if "RAINFOREST_RESIN" in state.order_depths:
        #     resin_depth: OrderDepth = state.order_depths["RAINFOREST_RESIN"]
        #     resin_mid = self.mid_price(resin_depth)
        #     resin_orders = self.avellaneda_stoikov("RAINFOREST_RESIN", resin_mid, self.position["RAINFOREST_RESIN"])
        #     result["RAINFOREST_RESIN"] = resin_orders
        # ---------------------------------------------------------

    

        # -------- Resin: Simple Market Making Assuming Constant Fair Value --------
        if Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_params = self.params[Product.RAINFOREST_RESIN]
            resin_order_depth = state.order_depths[Product.RAINFOREST_RESIN]
            resin_fair_value = resin_params["fair_value"]
            orders_take, bo, so = resin_take_orders(
                resin_order_depth, resin_fair_value, resin_position, self.LIMIT[Product.RAINFOREST_RESIN]
            )
            orders_clear, bo, so = resin_clear_orders(
                resin_order_depth, resin_position, resin_fair_value, self.LIMIT[Product.RAINFOREST_RESIN], bo, so
            )
            orders_make = resin_make_orders(
                resin_order_depth, resin_fair_value, resin_position, self.LIMIT[Product.RAINFOREST_RESIN], bo, so
            )
            result[Product.RAINFOREST_RESIN] = orders_take + orders_clear + orders_make
        # -------------------------------------------------------------------------
    


        # ------------- Kelp: Trading Strat -------------
        if Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_params = self.params[Product.KELP]
            kelp_order_depth = state.order_depths[Product.KELP]
            kelp_fair = kelp_fair_value(
                kelp_order_depth, kelp_params["default_fair_method"], kelp_params["min_volume_filter"]
            )
            kelp_take, bo, so = kelp_take_orders(kelp_order_depth, kelp_fair, kelp_params, kelp_position)
            kelp_clear, bo, so = kelp_clear_orders(kelp_order_depth, kelp_position, kelp_params, kelp_fair, bo, so)
            kelp_make = kelp_make_orders(kelp_order_depth, kelp_fair, kelp_position, kelp_params, bo, so)
            # result[Product.KELP] = kelp_take + kelp_clear + kelp_make
        # -----------------------------------------------



        # ----------- Squid Ink: Pairs Trading -----------
        def sortDict(dictionary):
            return {key: dictionary[key] for key in sorted(dictionary)}

        def MultipleCheck(m, n):
            return True if m % n == 0 else False

        def current_price(bid: dict, ask: dict):
            return np.round(
                (max(set(bid.keys())) + min(set(ask.keys()))) / 2, decimals=1
            )

        def calibrateM(kelparr, inkarr):
            a = np.vstack([np.log(kelparr), np.ones(len(inkarr))]).T
            m, c = np.linalg.lstsq(a, np.log(inkarr), rcond=None)[0]
            return m, c

        def BasketAnalyse(coeff, const, kelp, ink):
            basket = np.array(ink) - coeff * np.array(kelp) - const
            return np.round(np.mean(basket), decimals=5), np.round(np.var(basket), decimals=5)



        for product in OrderbookDict:
            if product == "SQUID_INK":
                InkOrderbookDepth = OrderbookDict[product]
                InkbidSpread = sortDict(InkOrderbookDepth.buy_orders)
                InkaskSpread = sortDict(InkOrderbookDepth.sell_orders)
                KelpOrderbookDepth = OrderbookDict["KELP"]
                KelpbidSpread = sortDict(KelpOrderbookDepth.buy_orders)
                KelpaskSpread = sortDict(KelpOrderbookDepth.sell_orders)
                CurrentInkPrice = current_price(InkbidSpread, InkaskSpread)
                CurrentKelpPrice = current_price(KelpbidSpread, KelpaskSpread)
                self.InkPreviousPriceArr.append(CurrentInkPrice)
                self.KelpPreviousPriceArr.append(CurrentKelpPrice)
                CheapestInkPrice = min(InkaskSpread.keys())
                CheapestKelpPrice = min(KelpaskSpread.keys())
                HighestInkPrice = max(InkbidSpread.keys())
                HighestKelpPrice = max(KelpbidSpread.keys())

                if (
                    len(self.InkPreviousPriceArr) > self.lookbackwindow
                ):  ## constantly adjusting the lookback window array
                    self.InkPreviousPriceArr = self.InkPreviousPriceArr[
                        1 : self.lookbackwindow + 1
                    ]
                if (
                    len(self.KelpPreviousPriceArr) > self.lookbackwindow
                ):  ## constantly adjusting the lookback window array
                    self.KelpPreviousPriceArr = self.KelpPreviousPriceArr[
                        1 : self.lookbackwindow + 1
                    ]
                if state.timestamp >= 100 * self.lookbackwindow:
                    # return result, conversions, trader_data
                    continue
                threshold = 0.7
                if (np.corrcoef(np.log(self.KelpPreviousPriceArr), np.log(self.InkPreviousPriceArr))[0, 1] < threshold):
                    ink_depth: OrderDepth = InkOrderbookDepth
                    ink_mid = CurrentInkPrice
                    ink_orders = self.avellaneda_stoikov("SQUID_INK", ink_mid, state.position["SQUID_INK"])
                    result["SQUID_INK"] = ink_orders
                else:
                    if MultipleCheck(state.timestamp, 100 * self.lookbackwindow):
                        # considers if we are in the lookback window area
                        self.m, self.c = calibrateM(
                            self.KelpPreviousPriceArr, self.InkPreviousPriceArr
                        )
                        self.mean, var = BasketAnalyse(
                            self.m,
                            self.c,
                            self.KelpPreviousPriceArr,
                            self.InkPreviousPriceArr,
                        )
                        self.stddev = np.round(np.sqrt(var), 5)
                    X = np.round(
                        np.log(CurrentInkPrice)
                        - self.c
                        - self.m * np.log(CurrentKelpPrice),
                        decimals=5,
                    )
                    zscore = np.round((X - self.mean) / self.stddev, decimals=5)
                    if zscore >= 1:  # sell ink buy kelp
                        inkorder.append(
                            Order(
                                "SQUID_INK",
                                HighestInkPrice,
                                -InkbidSpread[HighestInkPrice],
                            )
                        )
                        kelporder.append(
                            Order(
                                "KELP",
                                CheapestKelpPrice,
                                KelpaskSpread[CheapestKelpPrice],
                            )
                        )
                    if zscore <= -1:  # buy ink sell kelp
                        inkorder.append(
                            Order(
                                "SQUID_INK",
                                CheapestInkPrice,
                                InkaskSpread[CheapestInkPrice],
                            )
                        )
                        kelporder.append(
                            Order(
                                "KELP",
                                HighestKelpPrice,
                                -KelpbidSpread[HighestKelpPrice],
                            )
                        )
            else:
                continue
        result["SQUID_INK"] = inkorder
        # result["KELP"] = kelporder
        # ------------------------------------------------



        # ---------------- Picnic Baskets: Based on Fair Value Relation ----------------
        positionlimit = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }

        def vwap(product: str) -> float:
            vwap = 0
            total_amt = 0

            for prc, amt in state.order_depths[product].buy_orders.items():
                vwap += prc * amt
                total_amt += amt

            for prc, amt in state.order_depths[product].sell_orders.items():
                vwap += prc * abs(amt)
                total_amt += abs(amt)

            vwap /= total_amt
            return vwap

        def mid_price(product):
            orderbook = state.order_depths[product]
            bid = orderbook.buy_orders
            ask = orderbook.sell_orders
            return np.round(
                (max(set(bid.keys())) + min(set(ask.keys()))) / 2, decimals=5
            )

        def UpdatePreviousPosition(state) -> None:
            for product in set(state.position.keys()):
                if product not in set(self.previousposition.keys()):
                    self.previousposition[product] = 0
                if state.position[product] != self.previousposition[product]:
                    self.previousposition[product] = state.position[product]

        def UpdatePreviousPositionCounter(product) -> None:
            if product not in set(state.position.keys()):
                return None
            if (
                state.position[product] == self.previousposition[product]
            ):  # Updates previouspositionCounter
                self.previouspositionCounter[product] += 1
            else:
                self.previouspositionCounter[product] = 0

        def VolumeCapability(product, mode=None):
            if mode == "buy":
                return positionlimit[product] - state.position[product]
            if mode == "sell":
                return state.position[product] + positionlimit[product]

        def AskPrice(product, mode=None):  # how much a seller is willing to sell for
            if mode == "max":
                if product not in set(state.order_depths.keys()):
                    return 0  # FREAKY
                return max(set(state.order_depths[product].sell_orders.keys()))
            if mode == "min":
                if product not in set(state.order_depths.keys()):
                    return 0  # FREAKY
                return min(set(state.order_depths[product].sell_orders.keys()))

        def BidPrice(product, mode=None):  # how much a buyer is willing to buy for
            if mode == "max":
                if product not in set(state.order_depths.keys()):  # FREAKY
                    return 1000000  # FREAKY
                return max(set(state.order_depths[product].buy_orders.keys()))
            if mode == "min":
                if product not in set(state.order_depths.keys()):  # FREAKY
                    return 1000000  # FREAKY
                return min(set(state.order_depths[product].buy_orders.keys()))

        def AskVolume(
            product, mode=None
        ):  # ITS FOR THE HIGHEST/LOWEST PRICE NOT VOLUME!!
            if product not in set(state.order_depths.keys()):  # FREAKY
                return 100
            if mode == "max":
                return abs(
                    state.order_depths[product].sell_orders[
                        AskPrice(product, mode="max")
                    ]
                )
            if mode == "min":
                return abs(
                    state.order_depths[product].sell_orders[
                        AskPrice(product, mode="min")
                    ]
                )

        def BidVolume(
            product, mode=None
        ):  # ITS FOR THE HIGHEST/LOWEST PRICE NOT VOLUME!!
            if product not in set(state.order_depths.keys()):
                return 100  # FREAKY
            if mode == "max":
                return abs(
                    state.order_depths[product].buy_orders[
                        BidPrice(product, mode="max")
                    ]
                )
            if mode == "min":
                return abs(
                    state.order_depths[product].buy_orders[
                        BidPrice(product, mode="min")
                    ]
                )

        def PriceAdjustment(product, mode=None):
            holdFactor = 1.7  ## TODO OPTIMISE
            holdPremium = int(holdFactor * self.previouspositionCounter[product])
            if product not in set(state.position.keys()):
                return 0
            VolumeFraction = (
                VolumeCapability(product, mode=mode) / positionlimit[product]
            )
            if product == "PICNIC_BASKET1":
                PB1_high = 29.6905 + holdPremium  # TODO OPTIMISE
                PB1_mid = 26.0245 + holdPremium  # TODO OPTIMISE
                PB1_low = 0.6551 + holdPremium  # TODO OPTIMISE
                PB1_neg = -17.0629 + holdPremium  # TODO OPTIMISE
                if mode == "buy":
                    factor = 1
                if mode == "sell":
                    factor = -1
                if VolumeFraction <= 0.1:
                    return int(
                        factor * (PB1_high + 3)
                    )  # FOR SOME REASON PICNIC BASKET 1 LIKES THIS
                if VolumeFraction > 0.1 and VolumeFraction <= 0.2:
                    return int(factor * PB1_high)
                if VolumeFraction > 0.2 and VolumeFraction < 0.5:
                    return int(factor * PB1_mid)
                if VolumeFraction >= 0.5 and VolumeFraction < 1:
                    return int(factor * PB1_low)
                if VolumeFraction >= 1:
                    return int(factor * PB1_neg)
            if product == "PICNIC_BASKET2":
                PB2_high = 15.3932 + holdPremium  # TODO OPTIMISE
                PB2_mid = 9.5059 + holdPremium  # TODO OPTIMISE
                PB2_low = -2.1304 + holdPremium  # TODO OPTIMISE
                PB2_neg = -34.8136 + holdPremium  # TODO OPTIMISE
                if mode == "buy":
                    factor = 1
                if mode == "sell":
                    factor = -1
                if VolumeFraction <= 0.1:
                    return int(
                        factor * (PB2_high + 3)
                    )  # FOR SOME REASON PICNIC BASKET 1 LIKES THIS
                if VolumeFraction > 0.1 and VolumeFraction <= 0.2:
                    return int(factor * PB2_high)
                if VolumeFraction > 0.2 and VolumeFraction < 0.5:
                    return int(factor * PB2_mid)
                if VolumeFraction >= 0.5 and VolumeFraction < 1:
                    return int(factor * PB2_low)
                if VolumeFraction >= 1:
                    return int(factor * PB2_neg)

            if product == "DJEMBES":
                DJ_high = 22.4723 + holdPremium  # TODO OPTIMISE
                DJ_mid = 16.7552 + holdPremium  # TODO OPTIMISE
                DJ_low = -4.4401 + holdPremium  # TODO OPTIMISE
                DJ_neg = -9.0611 + holdPremium  # TODO OPTIMISE
                if mode == "buy":
                    factor = 1
                if mode == "sell":
                    factor = -1
                if VolumeFraction <= 0.1:
                    return int(factor * DJ_high)
                if VolumeFraction > 0.1 and VolumeFraction < 0.5:
                    return int(factor * DJ_mid)
                if VolumeFraction >= 0.5 and VolumeFraction < 1:
                    return int(factor * DJ_low)
                if VolumeFraction >= 1:
                    return int(factor * DJ_neg)

        kelporder = []
        inkorder = []
        resinorder = []
        croissantorder = []
        jamorder = []
        djembeorder = []
        p1order = []
        p2order = []
        offset = 0
        conversions = 0
        trader_data = ""
        # OrderbookDict = state.order_depths
        s1offset = -131.606  # PB1 is usually cheaper than PB2
        stdev1 = np.round(29.05 // np.sqrt(1000), decimals=5)
        zfactor1 = 0.5858  # TODO OPTIMISE!!!!!! 1 works pretty well
        s2offset = 105.417
        stdev2 = np.round(27.166 // np.sqrt(1000), decimals=5)
        zfactor2 = 0.1743  # TODO OPTIMISE!!!!!!
        # MAIN CHAIN OF LOGIC ##################################################

        for product in OrderbookDict:
            UpdatePreviousPositionCounter(product)
            if product == "PICNIC_BASKET1":  # 6 croissants 3 jams 1 djembe
                ##############################
                spread1 = (
                    vwap("PICNIC_BASKET1")  # FIXME MAYBE THE LOGIC IS WRONG
                    - 1.5 * vwap("PICNIC_BASKET2")
                    - vwap("DJEMBES")
                )
                normspread1 = spread1 - s1offset
                spread2 = (
                    vwap("PICNIC_BASKET2") - 4 * vwap("CROISSANTS") - 2 * vwap("JAMS")
                )
                normspread2 = spread2 - s2offset
                if (
                    normspread1
                    > stdev1
                    * zfactor1  ##Picnic Basket 1 is overvalued or PB2 OR Djembe is undervalued
                ):  ## sell at the worst bid(cheapest) ##TODO maybe be more aggressive?
                    HighestBid = BidPrice("PICNIC_BASKET1", mode="max")
                    HighestVolume = BidVolume("PICNIC_BASKET1", mode="max")
                    p1order.append(
                        Order(
                            "PICNIC_BASKET1",
                            HighestBid + PriceAdjustment("PICNIC_BASKET1", mode="sell"),
                            -HighestVolume,
                        )
                    )
                    if (
                        normspread2 > stdev2 * zfactor2
                    ):  # assume this means PB2 is overvalued
                        p2order.append(
                            Order(
                                "PICNIC_BASKET2",
                                AskPrice("PICNIC_BASKET2", mode="min")
                                + PriceAdjustment("PICNIC_BASKET2", mode="buy"),
                                AskVolume("PICNIC_BASKET2", mode="min"),
                            )
                        )
                    else:  # DJEMBE is undervalued
                        djembeorder.append(
                            Order(
                                "DJEMBES",
                                AskPrice("DJEMBES", mode="min")
                                + PriceAdjustment("DJEMBES", mode="buy"),
                                AskVolume("DJEMBES", mode="min"),
                            )
                        )

                if (
                    normspread1
                    < -stdev1
                    * zfactor1  ##Picnic Basket 1 is undervalued or PB2 OR Djembe is overvalued
                ):  ## buy at the worst ask(most expensive) ##TODO maybe be more aggressive?
                    CheapestAsk = AskPrice("PICNIC_BASKET1", mode="min")
                    CheapestVolume = AskVolume("PICNIC_BASKET1", mode="min")
                    p1order.append(
                        Order(
                            "PICNIC_BASKET1",
                            CheapestAsk + PriceAdjustment("PICNIC_BASKET1", mode="buy"),
                            CheapestVolume,
                        )
                    )
                    if (
                        normspread2 > stdev2 * zfactor2
                    ):  # assume this means PB2 is the one that is overvalued
                        p2order.append(
                            Order(
                                "PICNIC_BASKET2",
                                BidPrice("PICNIC_BASKET2", mode="max")
                                + PriceAdjustment("PICNIC_BASKET2", mode="sell"),
                                -BidVolume("PICNIC_BASKET2", mode="max"),
                            )
                        )
                    else:  # Djembe is overvalued
                        djembeorder.append(
                            Order(
                                "DJEMBES",
                                BidPrice("DJEMBES", mode="max")
                                + PriceAdjustment("DJEMBES", mode="sell"),
                                -BidVolume("DJEMBES", mode="max"),
                            )
                        )
        UpdatePreviousPosition(state)

        result["PICNIC_BASKET1"] = p1order
        result["PICNIC_BASKET2"] = p2order
        result["CROISSANTS"] = croissantorder
        result["JAMS"] = jamorder
        result["DJEMBES"] = djembeorder
        # ------------------------------------------------------------------------------



        # ----------------------- Croissant Logic -----------------------
        # Check if CROISSANTS is in the order depths
        if Product.CROISSANTS not in state.order_depths:
            return {}, 1, ""
        # --- Set up persistent history ---
        if state.traderData:
            try:
                history = jsonpickle.decode(state.traderData)
                if not isinstance(history, dict):
                    history = {"cr_prices": []}
                elif "cr_prices" not in history:
                    history["cr_prices"] = []
            except Exception:
                history = {"cr_prices": []}
        else:
            history = {"cr_prices": []}

        cr_product = Product.CROISSANTS
        cr_depth: OrderDepth = state.order_depths[cr_product]
        cr_orders: List[Order] = []

        # Compute the mid–price for CROISSANTS.
        if cr_depth.buy_orders and cr_depth.sell_orders:
            best_bid = max(cr_depth.buy_orders.keys())
            best_ask = min(cr_depth.sell_orders.keys())
            midpoint = (best_bid + best_ask) / 2.0
            history["cr_prices"].append(midpoint)
        # Convert history to a NumPy array.
        cr_price_array = np.array(history["cr_prices"])

        # Set parameters from the configuration.
        history_len = self.params[cr_product].get("history_length", 100)
        z_threshold = self.params[cr_product].get("z_threshold")

        # Only compute the z-score if we have sufficient history.
        if len(cr_price_array) >= history_len:
            recent = cr_price_array[-history_len:]
            mean = np.mean(recent)
            std = np.std(recent)
            # Safeguard for zero standard deviation.
            if std == 0:
                std = 1.0
            z = (cr_price_array[-1] - mean) / std
            cr_position = state.position.get(cr_product, 0)
            # Use the global limit for CROISSANTS.
            limit = self.LIMIT[cr_product]
            # Trading signals:
            # If z < -z_threshold: the price is low → signal to buy.
            if z < -z_threshold:
                # We iterate over sell orders (ask prices) to capture a buy.
                for ask, qty in cr_depth.sell_orders.items():
                    # Here we check that after taking the order we don't exceed the long position limit.
                    if cr_position + (-qty) <= limit:
                        cr_orders.append(Order(cr_product, ask, -qty))
            # If z > z_threshold: the price is high → signal to sell.
            elif z > z_threshold:
                for bid, qty in cr_depth.buy_orders.items():
                    if cr_position - qty >= -limit:
                        cr_orders.append(Order(cr_product, bid, -qty))
            result[cr_product] = cr_orders
        else:
            result[cr_product] = []
        # ---------------------------------------------------------------




        # ----------------------- CONVERSIONS -----------------------
        # Convert orders to the format expected by the logger - for backteser  
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data