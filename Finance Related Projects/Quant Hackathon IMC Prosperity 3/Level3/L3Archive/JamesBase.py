from datamodel import Order, OrderDepth, TradingState, Symbol
import math
import numpy as np
import json
import jsonpickle
from typing import List, Tuple, Dict, Any
from collections import deque





# ---------------------------
# Logger 
# ---------------------------


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, traderData: str) -> None:
        output = jsonpickle.encode(
            [state.timestamp, orders, conversions, traderData, self.logs],
            unpicklable=False
        )
        print(output)
        self.logs = ""

logger = Logger()


# ---------------------------
# Product and Parameters
# ---------------------------
class Product:
    KELP = "KELP"

PARAMS = {
   
    Product.KELP: {
        "take_width": 1,
        "position_limit": 50,
        "min_volume_filter": 20,
        "spread_edge": 1,
        "default_fair_method": "vwap_with_vol_filter"
    }
}

LIMIT = {
    
    Product.KELP: 50
    
}




