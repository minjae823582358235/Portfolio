from datamodel import OrderDepth, UserId, TradingState, Order, Trade
import numpy as np
import statistics
selldict=dict
buydict=dict
traderData=str
conversions=int
product=str
result=[]
sym=str
state=TradingState
marketTrade=Trade
def ArmaForecast(data,steps):
    def find_optimal_pq(data: np.ndarray, max_lag: int = 10):

        best_p, best_q = 1, 1
        min_error = float("inf")
        
        for p in range(1, max_lag + 1):
            for q in range(1, max_lag + 1):
                model = fit_arma_model(data, p, q)
                residuals = data[p:] - forecast_arma(data, model, steps=len(data) - p)
                error = np.sum(np.abs(residuals))
                
                if error < min_error:
                    min_error = error
                    best_p, best_q = p, q
        
        return best_p, best_q

    def fit_arma_model(data: np.ndarray, p: int, q: int):

        n = len(data)
        ar_coeffs = np.zeros(p)
        ma_coeffs = np.zeros(q)
        
        # Estimate AR coefficients using least squares
        if p > 0:
            X = np.column_stack([data[i:n - p + i] for i in range(p)])
            Y = data[p:]
            ar_coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        
        # Estimate MA coefficients using moving average
        if q > 0:
            residuals = data[p:] - np.dot(X, ar_coeffs)
            ma_coeffs = [statistics.mean(residuals[i:i+q]) for i in range(len(residuals)-q)]
        
        return {"ar": ar_coeffs, "ma": ma_coeffs}

    def forecast_arma(data: np.ndarray, model, steps: int):

        forecast = list(data[-len(model["ar"]):])
        for _ in range(steps):
            ar_part = sum(model["ar"][i] * forecast[-(i+1)] for i in range(len(model["ar"])))
            ma_part = sum(model["ma"]) if len(model["ma"]) > 0 else 0
            forecast.append(ar_part + ma_part)
        
        return np.array(forecast[-steps:])
    p, q = find_optimal_pq(data, max_lag=5)
    model = fit_arma_model(data, p, q)
    forecast = forecast_arma(data, model, steps=steps)
    return forecast

def ACF(data: np.ndarray, lag: int) -> float:
    if lag < 0 or lag >= len(data):
        raise ValueError("Lag must be between 0 and len(data) - 1")
    
    data = np.asarray(data)
    mean = np.mean(data)
    var = np.var(data)
    
    if var == 0:
        return 1.0 if lag == 0 else 0.0  # Avoid division by zero
    
    n = len(data) - lag
    return np.sum((data[:n] - mean) * (data[lag:] - mean)) / (n * var)

def ARMA():
    marketTrades=state.market_trades[sym]
    sumDict={}
    nDict={}
    priceArr=[]
    trade=Trade
    for trade in marketTrades:
        if trade.timestamp not in sumDict.keys():
            sumDict[trade.timestamp]=trade.price
            nDict[trade.timestamp]=abs(trade.quantity)
        else:
            sumDict[trade.timestamp]+=trade.price
            nDict[trade.timestamp]+=abs(trade.quantity)
    sortSum={key:sumDict[key] for key in sorted(sumDict)}
    for timestamp in sortSum:
        priceArr.append(np.round(sortSum[timestamp]/nDict[timestamp],decimals=1))
    nextstep=ArmaForecast(data=priceArr,steps=1)
    for sellPrice in selldict:
        if sellPrice<nextstep:
            result.append(Order(symbol=sym,price=sellPrice,quantity=-selldict[sellPrice]))
    for buyPrice in buydict:
        if buyPrice>nextstep:
            result.append(Order(symbol=sym,price=buyPrice,quantity=-buydict[buyPrice]))

def Default():
    for sellPrice in selldict.keys():
        if sellPrice<FairPrice():
            result.append(Order(symbol=sym,price=sellPrice,quantity=-selldict[sellPrice]))
    for buyPrice in buydict.keys():
        if buyPrice>FairPrice():
            result.append(Order(symbol=sym,price=buyPrice,quantity=-buydict[buyPrice]))

def FairPrice():
    output=(selldict[-1]+buydict[0])/2 #assuming both sorted ascending
    return output

class Trader:
    def run(self, state: TradingState):
        for sym in state.order_depths:
            orderdepth=state.order_depths[sym] ##
            buydict=orderdepth.buy_orders #OrderDepth() buy spread for a given product
            selldict=orderdepth.sell_orders #OrderDepth() sell spread for a given product
            buydict={key:buydict[key] for key in sorted(buydict)} #Asending
            selldict={key:selldict[key] for key in sorted(selldict)} #Asending
            print(state.timestamp)
            print('buydict: '+str(buydict)+'\n')
            print('selldict: '+str(selldict)+'\n')
            if state.timestamp <20*100:
                Default()
                traderData='default'
                conversions=None
                return result,conversions,traderData
            traderData='ARMA Forecast'
            ARMA()
        conversions=1
        return result,conversions,traderData


