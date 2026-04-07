import json
import jsonpickle
import numpy as np
import math
from collections import deque
from typing import Any, Dict, List, Tuple

from datamodel import (
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    TradingState,
)

################################################################################################################
#                                         Configuration                                                          #
################################################################################################################

SYMBOL = Symbol('JAMS')
POSITION_LIMIT = 350
LOT_SIZE = 350

################################################################################################################
#                                            Simple Logger                                                         #
################################################################################################################

class Logger:
    """Minimal JSON logger for orders and debug info."""
    def __init__(self, max_length: int = 3750) -> None:
        self.max_length = max_length
        self.buffer = []

    def log(self, **kwargs: Any) -> None:
        """Buffer a dict for this cycle."""
        self.buffer.append(kwargs)

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], extra: str = "") -> None:
        payload = {
            'timestamp': state.timestamp,
            'position': getattr(state, 'position', {}),
            'orders': [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr],
            'debug': extra,
        }
        # enforce length
        text = json.dumps(payload, cls=ProsperityEncoder, separators=(',',':'))
        if len(text) > self.max_length:
            text = text[:self.max_length-3] + '...'
        print(text)
        self.buffer.clear()

################################################################################################################
#                                    Kalman Filter Z‑Score Trader                                                   #
################################################################################################################

class KalmanZTrader:
    def __init__(
        self,
        symbol: Symbol,
        lookback: int = 100,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        lot_size: int = LOT_SIZE,
        pos_limit: int = POSITION_LIMIT,
        Q: float = 1e-5,
        R: float = 1e-2,
    ) -> None:
        self.symbol = symbol
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lot_size = lot_size
        self.pos_limit = pos_limit
        self.Q = Q
        self.R = R
        self._reset_filter()

    def _reset_filter(self):
        self.x = None
        self.P = None
        self.residuals = deque(maxlen=self.lookback)
        self.position = 0

    def _kalman_update(self, measurement: float) -> float:
        if self.x is None:
            self.x = measurement
            self.P = 1.0
        # Predict
        P_pred = self.P + self.Q
        # Update
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred
        return self.x

    def _compute_zscore(self, mid: float) -> float:
        trend = self._kalman_update(mid)
        resid = mid - trend
        self.residuals.append(resid)
        if len(self.residuals) < 2:
            return 0.0
        mean = np.mean(self.residuals)
        std = np.std(self.residuals, ddof=1)
        return (resid - mean) / std if std > 1e-6 else 0.0

    def _mid(self, od: OrderDepth) -> float:
        bids = od.buy_orders
        asks = od.sell_orders
        if not bids or not asks:
            if bids: return max(bids)
            if asks: return min(asks)
            return 0.0
        return 0.5 * (max(bids) + min(asks))

    def generate_orders(self, od: OrderDepth) -> Tuple[List[Order], float]:
        mid = self._mid(od)
        z = self._compute_zscore(mid)
        orders: List[Order] = []

        # Entry signals
        if z < -self.entry_z and self.position < self.pos_limit:
            qty = min(self.lot_size, self.pos_limit - self.position)
            orders.append(Order(self.symbol, math.floor(mid), +qty))
            self.position += qty
        elif z > self.entry_z and self.position > -self.pos_limit:
            qty = min(self.lot_size, self.pos_limit + self.position)
            orders.append(Order(self.symbol, math.ceil(mid), -qty))
            self.position -= qty
        # Exit signal
        elif abs(z) < self.exit_z and self.position != 0:
            qty = abs(self.position)
            px = math.ceil(mid) if self.position > 0 else math.floor(mid)
            sign = -1 if self.position > 0 else +1
            orders.append(Order(self.symbol, px, sign * qty))
            self.position = 0

        return orders, z

################################################################################################################
#                                       Main Trader Integration                                                    #
################################################################################################################

class Trader:
    def __init__(self) -> None:
        self.logger = Logger()
        self.kalman = KalmanZTrader(symbol=SYMBOL)

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        od = state.order_depths.get(SYMBOL)
        if od is None:
            return {SYMBOL: []}, 0, ""

        orders_list, zscore = self.kalman.generate_orders(od)
        orders = {SYMBOL: orders_list}
        debug = f"z={zscore:.2f}, pos={self.kalman.position}"
        self.logger.flush(state, orders, debug)
        return orders, 0, debug


#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  

#         prosperity3bt "Level5/James/JamJames2.py" 5 --no-out

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  # 