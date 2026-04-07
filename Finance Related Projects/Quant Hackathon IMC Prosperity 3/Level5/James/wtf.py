
""" 

1. Sell to -75 when green hits sunlight on its second intercept. ()
2. When red hits sunlight, start the buy sell from  -75 to ride the uptrend. 
3. then when orange hits sunlight on its second intercept, reset inv to 0. 
4. wait for next green hit 



if Sunlight = SunthreshUpper: and First entry is true 
    get total positions
    current position - total positions - LIMIT  #Go to biggest possible sell order IN LOCAL -> -75 



if Sunlight = SunThresLower and second entry is true 
    buy or sell whatever gets us to 0. 


if Sunlight = SunThreshTrendy: #Cashing out the sell orders IN LOCAL -> 0 
    buy 3, sell 2 to get back to 0 from -75 

    """

import logging
from typing import Dict, List, Tuple
from datamodel import Order, OrderDepth, TradingState

# Product identifier
MAGNIFICENT_MACARONS = 'MAGNIFICENT_MACARONS'
# Sell limit (absolute inventory)
LIMITS: Dict[str,int] = {MAGNIFICENT_MACARONS: 75}

class Trader:
    def __init__(self) -> None:
        # raw thresholds for triangle wave comparisons
        self.SUN_THRESH_UPPER = 50
        self.SUN_THRESH_TRENDY = 22
        self.SUN_THRESH_LOWER = -30

        # state trackers
        self.last_raw_sun: float = None
        self.upper_stage: int = 0     # 0=waiting first up,1=waiting down,2=aggressive sell
        self.trendy_stage: int = 0    # 0=waiting red cross,1=riding uptrend,2=done
        self.lower_stage: int = 0     # 0=waiting first down,1=waiting up,2=flattened

        # logger
        self.logger = logging.getLogger('SunlightStrategy')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self.logger.addHandler(handler)

    def run(self, state: TradingState) -> Tuple[Dict[str,List[Order]], int, str]:
        orders: Dict[str,List[Order]] = {}
        conversions = 0
        trader_data = ''

        # read raw triangle-wave sunlight index
        raw_sun = self._get_sunlight_index(state)

        # initialize on first tick
        if self.last_raw_sun is None:
            self.last_raw_sun = raw_sun
            return orders, conversions, trader_data

        # Phase 1: detect two zero-crossings of (raw_sun - upper_thresh)
        if self.upper_stage < 2:
            self._check_upper(raw_sun)
        # Phase 1b: aggressive sell then detect red crossing
        elif self.upper_stage == 2 and self.trendy_stage == 0:
            self._aggressive_sell(MAGNIFICENT_MACARONS,
                                  LIMITS[MAGNIFICENT_MACARONS],
                                  orders, state)
            self._check_trendy(raw_sun)
        # Phase 2: ride uptrend
        elif self.trendy_stage == 1:
            self._ride_uptrend(MAGNIFICENT_MACARONS, orders, state)
            if state.position.get(MAGNIFICENT_MACARONS, 0) >= 0:
                self.trendy_stage = 2
                self.logger.info('Completed uptrend ride; position ≥ 0.')
        # Phase 3: detect two zero-crossings of (raw_sun - lower_thresh) & flatten
        elif self.trendy_stage >= 1 and self.lower_stage < 2:
            self._check_lower(raw_sun, orders, state)

        # update last raw sunlight
        self.last_raw_sun = raw_sun
        return orders, conversions, trader_data

    def _get_sunlight_index(self, state: TradingState) -> float:
        obs = state.observations.conversionObservations.get(MAGNIFICENT_MACARONS)
        if obs is None:
            raise ValueError(f'Conversion observation missing for {MAGNIFICENT_MACARONS}')
        return obs.sunlightIndex

    def _check_upper(self, raw: float) -> None:
        # compute diffs relative to upper threshold
        last_diff = self.last_raw_sun - self.SUN_THRESH_UPPER
        curr_diff = raw - self.SUN_THRESH_UPPER
        # first upward crossing through zero
        if last_diff < 0 <= curr_diff and self.upper_stage == 0:
            self.upper_stage = 1
            self.logger.info('Upper thresh first crossing (up).')
        # second downward crossing through zero
        if last_diff > 0 >= curr_diff and self.upper_stage == 1:
            self.upper_stage = 2
            self.logger.info('Upper thresh second crossing (down); start aggressive sell.')

    def _check_trendy(self, raw: float) -> None:
        # detect upward crossing for red threshold
        last_diff = self.last_raw_sun - self.SUN_THRESH_TRENDY
        curr_diff = raw - self.SUN_THRESH_TRENDY
        if last_diff < 0 <= curr_diff and self.trendy_stage == 0:
            self.trendy_stage = 1
            self.logger.info('Red thresh crossing (up); enter uptrend ride.')

    def _check_lower(self, raw: float, orders: Dict[str,List[Order]], state: TradingState) -> None:
        # compute diffs relative to lower threshold
        last_diff = self.last_raw_sun - self.SUN_THRESH_LOWER
        curr_diff = raw - self.SUN_THRESH_LOWER
        # first downward crossing through zero
        if last_diff > 0 >= curr_diff and self.lower_stage == 0:
            self.lower_stage = 1
            self.logger.info('Lower thresh first crossing (down).')
        # second upward crossing through zero
        if last_diff < 0 <= curr_diff and self.lower_stage == 1:
            self.lower_stage = 2
            self.logger.info('Lower thresh second crossing (up); flattening.')
            self._flatten_position(MAGNIFICENT_MACARONS, orders, state)
            # reset stages for next cycle
            self.upper_stage = 0
            self.trendy_stage = 0
            self.lower_stage = 0

    def _aggressive_sell(self, product: str, limit: int,
                         orders: Dict[str,List[Order]], state: TradingState) -> None:
        book = state.order_depths.get(product)
        if book is None:
            return
        pos = state.position.get(product, 0)
        to_sell = limit + pos
        if to_sell <= 0:
            return
        # sweep bids
        for price, volume in sorted(book.buy_orders.items(), key=lambda x: -x[0]):
            if to_sell <= 0:
                break
            qty = min(volume, to_sell)
            orders.setdefault(product, []).append(Order(product, price, -qty))
            to_sell -= qty
        if to_sell > 0:
            price = max(book.buy_orders.keys())
            orders.setdefault(product, []).append(Order(product, price, -to_sell))

    def _ride_uptrend(self, product: str, orders: Dict[str,List[Order]], state: TradingState) -> None:
        book = state.order_depths.get(product)
        if book is None:
            return
        # buy 3, sell 2
        ask = min(book.sell_orders.keys())
        bid = max(book.buy_orders.keys())
        orders.setdefault(product, []).append(Order(product, ask, min(3, book.sell_orders[ask])))
        orders.setdefault(product, []).append(Order(product, bid, -min(2, book.buy_orders[bid])))

    def _flatten_position(self, product: str, orders: Dict[str,List[Order]], state: TradingState) -> None:
        book = state.order_depths.get(product)
        if book is None:
            return
        pos = state.position.get(product, 0)
        if pos < 0:
            to_buy = -pos
            for price, vol in sorted(book.sell_orders.items(), key=lambda x: x[0]):
                if to_buy <= 0:
                    break
                qty = min(vol, to_buy)
                orders.setdefault(product, []).append(Order(product, price, qty))
                to_buy -= qty
            if to_buy > 0:
                orders.setdefault(product, []).append(Order(product, min(book.sell_orders.keys()), to_buy))
        elif pos > 0:
            to_sell = pos
            for price, vol in sorted(book.buy_orders.items(), key=lambda x: -x[0]):
                if to_sell <= 0:
                    break
                qty = min(vol, to_sell)
                orders.setdefault(product, []).append(Order(product, price, -qty))
                to_sell -= qty
            if to_sell > 0:
                orders.setdefault(product, []).append(Order(product, max(book.buy_orders.keys()), -to_sell))

        
##  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #    #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  

#         prosperity3bt "/Users/jameszhao/Documents/Programs/IMC-Prosperity-3/Level5/James/wtf.py" 4 5 --no-out           
        
##  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #    #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

