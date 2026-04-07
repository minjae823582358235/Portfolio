# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:54:28 2025

@author: Mark Brezina
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def __init__(self):
        self.limit = 20
        
    def mid_price(self, order_depth):
        
        if len(order_depth.sell_orders) != 0:
            
            m1 = 0
            n1 = 0
            for i in range(len(order_depth.sell_orders)):
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[i]
                m1 = m1 + best_ask * best_ask_amount
                n1 = n1 + best_ask_amount
            
            m1 = m1 / n1
            
        if len(order_depth.buy_orders) != 0:
            
            m2 = 0
            n2 = 0
            for i in range(len(order_depth.buy_orders)):
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[i]
                
                m2 = m2 + best_bid * best_bid_amount
                n2 = n2 + best_bid_amount
            
            m2 = m2 / n2
            
        return (m1+m2)/2
    
    def run(self, state: TradingState):
        
        limit = 20
        qt = state.position["KELP"] if "KELP" in state.position else 0
        
        
        result = {}
        for product in state.order_depths:
            
            if product == "KELP":
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []
            
                mid = self.mid_price(order_depth)
                next_mid_price = 0.0015640583224964238 * mid + 2015.8956653967819
                # short-term linear regression
                # every 10 time steps, new linear regression.
                if state.timestamp % 10 == 0:
                    slope, intercept = statistics.linear_regression(x,y)
                
                bid = int(round(next_mid_price - 1.5,0))
                print("BUY", str(3) + "x", bid)
                orders.append(Order(product, bid, 3))
                
                ask = int(round(next_mid_price + 1.5,0))
                print("SELL", str(3) + "x", ask)
                orders.append(Order(product, ask, -3))
                        
                
                result[product] = orders
        
            else:
                continue
    
        traderData = "SAMPLE" 
        
        #short moving average
        #long moving average
         
        conversions = 1
        return result, conversions, traderData