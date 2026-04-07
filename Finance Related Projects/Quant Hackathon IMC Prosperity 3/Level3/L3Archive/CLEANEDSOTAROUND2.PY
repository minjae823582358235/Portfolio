import json
import numpy as np
import math
from typing import Any, Dict, List, Tuple, Optional
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # Truncate items to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

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
        return [[listing.symbol, listing.product, listing.denomination] 
                for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] 
                for symbol, depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])
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
        return value[:max_length - 3] + "..."


logger = Logger()

# Product constants
RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = 'KELP'
SQUID_INK = 'SQUID_INK'
CROISSANTS = 'CROISSANTS'
PICNIC_BASKET1 = 'PICNIC_BASKET1'
PICNIC_BASKET2 = 'PICNIC_BASKET2'
JAMS = 'JAMS'
DJEMBES = 'DJEMBES'
VOLCANIC_ROCK = 'VOLCANIC_ROCK'
VOLCANIC_ROCK_VOUCHER_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
VOLCANIC_ROCK_VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
VOLCANIC_ROCK_VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
VOLCANIC_ROCK_VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
VOLCANIC_ROCK_VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'

# Position limits for each product
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
}

# Strategy parameters
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
        "take_width": 18.7017,
        "position_limit": 50,
        "min_volume_filter": 21.7543,
        "spread_edge": 1.0760,
        "default_fair_method": "vwap_with_vol_filter",
    },
    SQUID_INK: {
        "rsi_window": 106,
        "rsi_overbought": 52,
        "rsi_oversold": 41,
    },
    CROISSANTS: {
        "history_length": 83.8247,
        "z_threshold": 1.9986
    }
}

# Synthetic basket parameters
synth1Mean = -131.606  # PB1 is usually cheaper than PB2
synth1Sigma = np.round(29.05 // np.sqrt(1000), decimals=5)
s1Zscore = 0.7872
synth2Mean = 105.417
synth2Sigma = np.round(27.166 // np.sqrt(1000), decimals=5)
s2Zscore = 1.4417
diff_threshold_b1_b2 = 176.8118
KSZscore = 0.190327

# Dynamic inventory factor parameters
# Position limits for different inventory levels
INVENTORY_LEVELS = {
    'PB1': {'high': 38.8995, 'mid': 10.8195, 'low': 9.2489, 'neg': -14.6403},
    'PB2': {'high': 39.2463, 'mid': 26.8949, 'low': 5.3674, 'neg': -26.1430},
    'DJEMBES': {'high': 22.4723, 'mid': 16.7552, 'low': -4.4401, 'neg': -9.0611},
    'SQUID_INK': {'high': 26.244, 'mid': 3.57077, 'low': 1.0171, 'neg': -19.168886},
    'KELP': {'high': 35.561048, 'mid': 34.49889, 'low': 9.11303, 'neg': 1.2853198},
}

# Hold factor parameters
HOLD_FACTORS = {
    'PB1': 18.1815,
    'PB2': 3.9930,
    'DJEMBES': 1.7,
    'SQUID_INK': 28.964,
    'KELP': 28.964,
}

# General utility functions
def sort_dict(dictionary: dict):
    """Sort dictionary by keys"""
    return {key: dictionary[key] for key in sorted(dictionary)}

def volume_capability(product, mode, state: TradingState):
    """Calculate available volume capacity for a product"""
    if mode == "buy":
        return LIMIT[product] - state.position.get(product, 0)
    elif mode == "sell":
        return state.position.get(product, 0) + LIMIT[product]
    return 0

def vwap(product: str, state: TradingState) -> float:
    """Calculate volume-weighted average price"""
    if product not in state.order_depths:
        return 0
        
    vwap_value = 0
    total_amt = 0

    for prc, amt in state.order_depths[product].buy_orders.items():
        vwap_value += prc * amt
        total_amt += amt

    for prc, amt in state.order_depths[product].sell_orders.items():
        vwap_value += prc * abs(amt)
        total_amt += abs(amt)

    if total_amt == 0:
        return 0
        
    vwap_value /= total_amt
    return np.round(vwap_value, decimals=5)

def mid_price(order_depth: OrderDepth) -> float:
    """Calculate mid-price from order depth"""
    m1 = m2 = 0
    
    if order_depth.sell_orders:
        m1 = sum(price * abs(amount) for price, amount in order_depth.sell_orders.items())
        m1 /= sum(abs(amount) for amount in order_depth.sell_orders.values())
    
    if order_depth.buy_orders:
        m2 = sum(price * amount for price, amount in order_depth.buy_orders.items())
        m2 /= sum(amount for amount in order_depth.buy_orders.values())
    
    # Use whichever is available, or the average if both are
    return (m1 + m2) / 2 if (m1 and m2) else (m1 or m2)

def get_ask_price(product, mode, state: TradingState):
    """Get ask price for a product"""
    if product not in state.order_depths:
        return 0
    
    if not state.order_depths[product].sell_orders:
        return 0
        
    prices = state.order_depths[product].sell_orders.keys()
    return max(prices) if mode == "max" else min(prices)

def get_bid_price(product, mode, state: TradingState):
    """Get bid price for a product"""
    if product not in state.order_depths:
        return 1000000
    
    if not state.order_depths[product].buy_orders:
        return 1000000
        
    prices = state.order_depths[product].buy_orders.keys()
    return max(prices) if mode == "max" else min(prices)

def get_ask_volume(product, mode, state: TradingState):
    """Get ask volume for highest/lowest price"""
    if product not in state.order_depths:
        return 100
        
    price = get_ask_price(product, mode, state)
    if price == 0:
        return 100
        
    return abs(state.order_depths[product].sell_orders[price])

def get_bid_volume(product, mode, state: TradingState):
    """Get bid volume for highest/lowest price"""
    if product not in state.order_depths:
        return 100
        
    price = get_bid_price(product, mode, state)
    if price == 1000000:
        return 100
        
    return abs(state.order_depths[product].buy_orders[price])

# RAINFOREST_RESIN Strategy Functions
def resin_take_orders(
    order_depth: OrderDepth, fair_value: float, position: int, position_limit: int
) -> Tuple[List[Order], int, int]:
    """Take immediately profitable orders for resin"""
    orders = []
    buy_order_volume = 0
    sell_order_volume = 0
    
    # Check for profitable selling opportunities
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        if best_bid > fair_value:
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order(RAINFOREST_RESIN, best_bid, -quantity))
                sell_order_volume += quantity
    
    # Check for profitable buying opportunities
    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask < fair_value:
            quantity = min(best_ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order(RAINFOREST_RESIN, best_ask, quantity))
                buy_order_volume += quantity
                
    return orders, buy_order_volume, sell_order_volume

class Trader:
    def __init__(self) -> None:
        self.logger = Logger()
        self.logger.print("Initialized trader for JAMS and DJEMBES only.")
        self.pos_limits = LIMIT
        self.diff_threshold_b1_b2 = diff_threshold_b1_b2
        self.lot_size = 1

        # Position tracking
        self.previousposition = {params: 0 for params in LIMIT.keys()}
        self.position = {params: 0 for params in LIMIT.keys()}
        self.positionCounter = {params: 0 for params in LIMIT.keys()}

        # Price history tracking
        self.asset_history = {
            "RAINFOREST_RESIN": {"timestamps": [], "mid_prices": []},
            "KELP": {"timestamps": [], "mid_prices": []},
            "SQUID_INK": {"timestamps": [], "mid_prices": []}
        }

        # Pairs trading strategy setup
        self.lookbackwindow = 45
        self.strategy = SquidInkRsiStrategy(SQUID_INK, LIMIT[SQUID_INK])
    
    def update_market_data(self, product, state):
        """Store current timestamp and mid-price for tracked assets"""
        if product not in self.asset_history:
            return
            
        order_depth = state.order_depths[product]
        mid = mid_price(order_depth)
        self.asset_history[product]["timestamps"].append(state.timestamp)
        self.asset_history[product]["mid_prices"].append(mid)

    def update_position_counter(self, product, state):
        """Track how long we've held a position"""
        if product not in state.position:
            return
            
        if state.position[product] == self.previousposition[product]:
            self.positionCounter[product] += 1
        else:
            self.positionCounter[product] = 0

    def get_price_adjustment(self, product, mode, state):
        """Calculate price adjustment based on position and volume"""
        if product not in state.position:
            return 0
            
        # Price adjustment parameters by product and position size
        adjustment_params = {
            PICNIC_BASKET1: {"high": pb1high, "mid": pb1mid, "low": pb1low, "neg": pb1neg, "hold": pb1Hold},
            PICNIC_BASKET2: {"high": pb2high, "mid": pb2mid, "low": pb2low, "neg": pb2neg, "hold": pb2Hold},
            DJEMBES: {"high": djembeshigh, "mid": djembesmid, "low": djembeslow, "neg": djembesneg, "hold": djembeHold},
            SQUID_INK: {"high": squidinkhigh, "mid": squidinkmid, "low": squidinklow, "neg": squidinkneg, "hold": squidinkHold},
            KELP: {"high": kelphigh, "mid": kelpmid, "low": kelplow, "neg": kelpneg, "hold": kelpHold},
        }
        
        # No adjustment if product not in parameters
        if product not in adjustment_params:
            return 0
            
        params = adjustment_params[product]
        volume_fraction = VolumeCapability(product, mode=mode, state=state) / LIMIT[product]
        hold_premium = int(params["hold"] * self.positionCounter[product])
        
        # Add hold premium to all price levels
        high = params["high"] + hold_premium
        mid = params["mid"] + hold_premium
        low = params["low"] + hold_premium
        neg = params["neg"] + hold_premium
        
        # Direction multiplier
        factor = 1 if mode == "buy" else -1
        
        # Return price adjustment based on volume fraction
        if volume_fraction <= 0.1:
            # Special case for Picnic Basket 1
            if product == PICNIC_BASKET1:
                return int(factor * (high + 3))
            return int(factor * high)
        elif volume_fraction <= 0.2 and product in [PICNIC_BASKET1, PICNIC_BASKET2]:
            return int(factor * high)
        elif volume_fraction < 0.5:
            return int(factor * mid)
        elif volume_fraction < 1:
            return int(factor * low)
        else:
            return int(factor * neg)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # Initialize results
        result = {product: [] for product in LIMIT.keys()}
        conversions = 0
        
        # Load trader data if available
        try:
            if state.traderData:
                loaded_data = json.loads(state.traderData)
                self.strategy.load(loaded_data.get(SQUID_INK, {}))
            else:
                self.strategy.load({})
        except Exception:
            self.strategy.load({})
        
        # Update position tracking
        for product in state.position:
            self.update_position_counter(product, state)
            self.position[product] = state.position[product]
            self.update_market_data(product, state)
        
        # -------- RESIN: Simple Market Making --------
        if RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(RAINFOREST_RESIN, 0)
            resin_params = PARAMS[RAINFOREST_RESIN]
            resin_order_depth = state.order_depths[RAINFOREST_RESIN]
            resin_fair_value = resin_params["fair_value"]
            
            # Process orders in three phases: take, clear, and make
            orders_take, bo, so = resin_take_orders(
                resin_order_depth, resin_fair_value, resin_position, LIMIT[RAINFOREST_RESIN]
            )
            orders_clear, bo, so = resin_clear_orders(
                resin_order_depth, resin_position, resin_fair_value, LIMIT[RAINFOREST_RESIN], bo, so
            )
            orders_make = resin_make_orders(
                resin_order_depth, resin_fair_value, resin_position, LIMIT[RAINFOREST_RESIN], bo, so
            )
            result[RAINFOREST_RESIN] += orders_take + orders_clear + orders_make

        # -------- KELP Trading Method --------
        if KELP in state.order_depths:
            kelp_position = state.position.get(KELP, 0)
            kelp_params = PARAMS[KELP]
            kelp_order_depth = state.order_depths[KELP]
            kelp_fair = kelp_fair_value(
                kelp_order_depth, kelp_params["default_fair_method"], kelp_params["min_volume_filter"]
            )
            
            kelp_take, bo, so = kelp_take_orders(kelp_order_depth, kelp_fair, kelp_params, kelp_position)
            kelp_clear, bo, so = kelp_clear_orders(kelp_order_depth, kelp_position, kelp_params, kelp_fair, bo, so)
            kelp_make = kelp_make_orders(kelp_order_depth, kelp_fair, kelp_position, kelp_params, bo, so)
            result[KELP] = kelp_take + kelp_clear + kelp_make

        # -------- SQUID INK Strategy --------
        self.strategy.run(state)
        if self.strategy.orders:
            result[SQUID_INK] = self.strategy.orders

        # Save strategy data for next round
        trader_data_for_next_round = {SQUID_INK: self.strategy.save()}
        trader_data_encoded = json.dumps(trader_data_for_next_round, separators=(",", ":"))

        # -------- PICNIC BASKET Arbitrage --------
        if PICNIC_BASKET1 in state.order_depths:
            self._handle_picnic_basket_arbitrage(state, result)

        # -------- JAMS & DJEMBES Signal-based Trading --------
        self._handle_jams_and_djembes_trading(state, result)

        logger.flush(state, result, conversions, trader_data_encoded)
        return result, conversions, trader_data_encoded
        
    def _handle_picnic_basket_arbitrage(self, state, result):
        """Handle arbitrage between picnic baskets and their components"""
        # Calculate spreads
        spread1 = (
            vwap(PICNIC_BASKET1, state=state) - 
            1.5 * vwap(PICNIC_BASKET2, state=state) - 
            vwap(DJEMBES, state=state)
        )
        normspread1 = spread1 - synth1Mean
        
        spread2 = (
            vwap(PICNIC_BASKET2, state=state) - 
            4 * vwap(CROISSANTS, state=state) - 
            2 * vwap(JAMS, state=state)
        )
        normspread2 = spread2 - synth2Mean
        
        # Case 1: Basket 1 overvalued or PB2/Djembe undervalued
        if normspread1 > synth1Sigma * s1Zscore:
            # Sell Basket 1
            highest_bid = BidPrice(PICNIC_BASKET1, mode="max", state=state)
            highest_volume = BidVolume(PICNIC_BASKET1, mode="max", state=state)
            result[PICNIC_BASKET1].append(
                Order(
                    PICNIC_BASKET1,
                    highest_bid + self.get_price_adjustment(PICNIC_BASKET1, mode="sell", state=state),
                    -highest_volume
                )
            )
            
            # If Basket 2 is overvalued, buy it
            if normspread2 > synth2Sigma * s1Zscore:
                result[PICNIC_BASKET2].append(
                    Order(
                        PICNIC_BASKET2,
                        AskPrice(PICNIC_BASKET2, mode="min", state=state) + 
                        self.get_price_adjustment(PICNIC_BASKET2, mode="buy", state=state),
                        AskVolume(PICNIC_BASKET2, mode="min", state=state)
                    )
                )
        
        # Case 2: Basket 1 undervalued or PB2/Djembe overvalued
        elif normspread1 < -synth1Sigma * s1Zscore:
            # Buy Basket 1
            cheapest_ask = AskPrice(PICNIC_BASKET1, mode="min", state=state)
            cheapest_volume = AskVolume(PICNIC_BASKET1, mode="min", state=state)
            result[PICNIC_BASKET1].append(
                Order(
                    PICNIC_BASKET1,
                    cheapest_ask + self.get_price_adjustment(PICNIC_BASKET1, mode="buy", state=state),
                    cheapest_volume
                )
            )
            
            # If Basket 2 is overvalued, sell it
            if normspread2 > synth2Sigma * s2Zscore:
                result[PICNIC_BASKET2].append(
                    Order(
                        PICNIC_BASKET2,
                        BidPrice(PICNIC_BASKET2, mode="max", state=state) + 
                        self.get_price_adjustment(PICNIC_BASKET2, mode="sell", state=state),
                        -BidVolume(PICNIC_BASKET2, mode="max", state=state)
                    )
                )

    def _handle_jams_and_djembes_trading(self, state, result):
        """Process trading signals for JAMS and DJEMBES based on basket composition"""
        # Get market data for relevant products
        relevant_products = [CROISSANTS, JAMS, DJEMBES, PICNIC_BASKET1, PICNIC_BASKET2]
        market_data = self._get_market_data(state, relevant_products)
        
        # Skip if we don't have complete market data
        if not all(market_data["mid_price"].values()):
            self.logger.print("Insufficient market data to compute composition signal.")
            return
            
        # Calculate composition signal
        signal = self._calculate_composition_signal(market_data["mid_price"])
        
        # Place orders based on signal
        for prod in [JAMS, DJEMBES]:
            self._place_order_based_on_signal(prod, signal, state, result)
    
    def _get_market_data(self, state, products):
        """Get market data for the specified products"""
        best_bid = {}
        best_ask = {}
        mid_price = {}
        
        for prod in products:
            od = state.order_depths.get(prod)
            if not od:
                best_bid[prod] = None
                best_ask[prod] = None
                mid_price[prod] = None
                continue
                
            best_bid[prod] = max(od.buy_orders.keys()) if od.buy_orders else None
            best_ask[prod] = min(od.sell_orders.keys()) if od.sell_orders else None
            
            if best_bid[prod] is not None and best_ask[prod] is not None:
                mid_price[prod] = 0.5 * (best_bid[prod] + best_ask[prod])
            elif best_bid[prod] is not None:
                mid_price[prod] = best_bid[prod]
            elif best_ask[prod] is not None:
                mid_price[prod] = best_ask[prod]
            else:
                mid_price[prod] = None
                
        return {"best_bid": best_bid, "best_ask": best_ask, "mid_price": mid_price}
        
    def _calculate_composition_signal(self, mid_prices):
        """Calculate trading signal based on basket composition"""
        b1 = mid_prices[PICNIC_BASKET1]
        b2 = mid_prices[PICNIC_BASKET2]
        c = mid_prices[CROISSANTS]
        j = mid_prices[JAMS]
        d = mid_prices[DJEMBES]
        
        # Calculate implied basket price and difference
        implied_b1 = b2 + 2 * c + j + d
        diff_comp = b1 - implied_b1
        
        self.logger.print(f"Composition signal: BASKET1={b1:.1f}, Implied_BASKET1={implied_b1:.1f}, diff={diff_comp:.1f}")
        
        # Determine signal
        if diff_comp > self.diff_threshold_b1_b2:
            # Basket overpriced - buy components
            signal = 1
        elif diff_comp < -self.diff_threshold_b1_b2:
            # Components overpriced - sell components
            signal = -1
        else:
            signal = 0
            
        self.logger.print(f"Signal determined (for JAMS and DJEMBES): {signal}")
        return signal
        
    def _place_order_based_on_signal(self, product, signal, state, result):
        """Place orders for JAMS and DJEMBES based on signal"""
        od = state.order_depths.get(product)
        if not od:
            self.logger.print(f"No order depth for {product}, skipping.")
            return
            
        current_pos = state.position.get(product, 0)
        pos_limit = self.pos_limits[product]
        
        if signal == 1:  # Buy signal
            capacity = pos_limit - current_pos
            if capacity <= 0:
                self.logger.print(f"No buying capacity for {product} (current position: {current_pos}).")
                return
                
            trade_qty = min(self.lot_size, capacity)
            
            if od.sell_orders:
                best_ask_price = min(od.sell_orders.keys())
                price = best_ask_price + 1
                result[product].append(Order(product, price, trade_qty))
                self.logger.print(f"Placing BUY order for {product}: {trade_qty}x{price}")
            else:
                self.logger.print(f"No sell orders for {product}, cannot place BUY order.")
                
        elif signal == -1:  # Sell signal
            capacity = pos_limit + current_pos
            if capacity <= 0:
                self.logger.print(f"No selling capacity for {product} (current position: {current_pos}).")
                return
                
            trade_qty = min(self.lot_size, capacity)
            
            if od.buy_orders:
                best_bid_price = max(od.buy_orders.keys())
                price = best_bid_price - 1
                result[product].append(Order(product, price, -trade_qty))
                self.logger.print(f"Placing SELL order for {product}: {trade_qty}x{price}")
            else:
                self.logger.print(f"No buy orders for {product}, cannot place SELL order.")
        else:
            self.logger.print(f"No clear signal for {product} – no order placed.")