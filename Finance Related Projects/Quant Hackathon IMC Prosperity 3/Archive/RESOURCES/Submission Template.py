from datamodel import OrderDepth, UserId, TradingState, Order
class Trader:
    def run(self, state: TradingState):
        result={}
        conversions=None
        traderData='Default'
        return result,conversions,traderData