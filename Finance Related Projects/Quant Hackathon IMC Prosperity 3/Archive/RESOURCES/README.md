# Layered approach




# Avellaneda-Stoikov

Market making is a trading approach in which a participant continuously quotes both buy (bid) and sell (ask) orders, in order to capture the bid‐ask spread and earn on the distance between a middle price and the posted quotes. 
<br>
<br>
Sometimes assets stay put around a middle price, sometimes they drift. Therefore we also need a framework to capture those movements.

| Name | Symbol |
| ------------- | ------------- |
| bid - buy price quoted | $b$ |
| ask - sell price quoted | $a$ |
| bid amount - # of bids quoted | $n_b$ |
| ask amount - # of asks quoted | $n_a$ |
| drift | $\mu$ |
| standard deviation | $\sigma$ |
| risk factor | $\gamma$ |
| inventory | $q$ |
| reserve price | $r$ |
| reserve deltas | $\delta_a$ <br> $\delta_b$ |
| fair price | $v_f$ |
| wealth | pnl |
| order amount | Content Cell |


1. two sided quoting - adjust for drift - skewing quotes directionally
   - Post bids/buy at slightly below mid
   - Post asks/sells at slightly above mid
   - adjust quotes, to skew in drift direction
   	- Positive drift - up - go long over time. - raise bid price, lower ask price
   	- negative drift - down - go short over time. - Lower bid prices, raise ask prices
   - Adjust for jump risk premiums - This premium compensates the maker for the possibility that a large move occurs immediately after quoting, resulting in potential losses.
   - Predict future asks/bids to adapt quotes
   - real-time volatility forecast, GARCH, vol-of-vol heuristic to automatic adjust spread.
3. Inventory management - adjust for drift
   - Over time we build up a long or short position.
   - Managing inventory risk.
  	- Dynamically adjusting quotes
   	- Market makers might use faster inventory rebalancing (e.g., crossing the spread to flatten positions).
   	- Risk limits. To avoid ruinous losses if a jump occurs while carrying a big inventory, we put in risk limits on the maximum position size and throttle order placement, as well as reducing inventory with regards to movements in the market

$$dS_t = \mu_t dt + \sigma_t dW_t + \int J_t dN_t $$
 where $\mu_t$ and $\sigma_t$ can be time dependent. $N_t$ is a poisson process. $J_t$ is the jump size dist. We than have to solve the stochastic control/ Hamilton-jacobi-bellman context. We need to solve for proper quote adjustments. the solution becomes functions of drift, volatility and intensity.


Near‐Real‐Time Parameter Updates:
To keep up with fast-changing market conditions, the maker recalibrates $\mu_t$ and $\sigma_t$
from trade or quote arrival data and uses event‐risk flags (e.g., “major news in 1 minute, jump risk high!”) to shift spreads or pull quotes.

Advanced Market Data Filters:
Market makers incorporate short‐term volatility estimators, news sentiment, or order‐flow imbalance indicators that can signal upcoming jumps (e.g., rapid expansions of the bid‐ask spread or a liquidity “vacuum” in the order book). They can then throttle quotes or widen them significantly.


###  Simple market-making for RESIN

The mid-price between bid-asks is fairly stable around 10K.

### market-making with drift for KELP
The mid-price drifts either up or down, but moves between 2027 and 2015.

**inventory**
- upper bound of inventory position

**Reserve price**
r[n] = s[n] - q[n] * gamma * sigma**2*(T-dt*n)
r[n] = (ra[n] + rb[n])/2

r = brownian_motion - q * gamma * sigma^2 * (T-t)
delta = gamma * sigma^2 * (T-t) + 2 / gamma * math.log(1 + gamma / k)

**Reserve deltas**
delta_a = ra[n] - s[n]
delta_ask = ask - s
 
delta_b = s[n] - rb[n]
delta_bid = s - bid 

**Reserve spread**
r_spread = 2 / gamma * math.log(1+gamma/k) 

**Optimal quotes**
ra[n] = r[n] + r_spread/2
ra[n] = s[n] + math.log(1+(1-2*q[n])*coef)/gamma
rb[n] = r[n] - r_spread/2
rb[n] = s[n] + math.log(1+(-1-2*q[n])*coef)/gamma

coef = gamma**2*sigma**2/(2*w-gamma**2*q[n]**2*sigma**2)

**order consumption probability factors**
- intensities
lambda_a = A * math.exp(-k*delta_a)
lambda_b = A * math.exp(-k*delta_b)


"""
            s - mid market price
            q - difference between current size and counterparty order size
            gamma - sensitivity parameter (how much our quote should move in response to inventory changes)
            var - price variance (calculated using mid price over x rolling window)
            T - time difference but ill ignore that first
            k - order book liquidity density
            r - reservation price
            delta - optimal spread // 2
            """
            
            s = (bid_prices[0] + ask_prices[0]) / (2 * TICK_SIZE_IN_CENTS) # mid price in $
            q = 0
            gamma = 0.05
            var = 0
            k = 1
            if self.T > 0:
                self.T -= 0.002
            else:
                self.T = 0.000001

            if int(s) != 0:
                self.mid_prices.append(s)

            lookback = 10

            if len(self.mid_prices) >= lookback:
                var = np.var(self.mid_prices[-1:-lookback - 1:-1])
                q = self.position

            # Reservation pricing
            r = s - (q * gamma * var * self.T)

			# Bid ask spread
            delta = (gamma * var * self.T + (2 / gamma * math.log(1 + (gamma / k))))

            # Prices to be sent in
            new_bid_price = math.ceil((r - delta / 2)) * TICK_SIZE_IN_CENTS
            new_ask_price = math.ceil((r + delta / 2)) * TICK_SIZE_IN_CENTS

            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                self.bid_id = 0
            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                self.send_cancel_order(self.ask_id)
                self.ask_id = 0

            if self.bid_id == 0 and new_bid_price != 0 and self.position <= POSITION_LIMIT - (LOT_SIZE * len(self.bids) + LOT_SIZE):
                self.bid_id = next(self.order_ids)
                self.bid_price = new_bid_price
                self.send_insert_order(self.bid_id, Side.BUY, new_bid_price, self.BID_LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.bids.add(self.bid_id)

            if self.ask_id == 0 and new_ask_price != 0 and self.position >= -POSITION_LIMIT + (LOT_SIZE * len(self.asks) + LOT_SIZE):
                self.ask_id = next(self.order_ids)
                self.ask_price = new_ask_price
                self.send_insert_order(self.ask_id, Side.SELL, new_ask_price, self.ASK_LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.asks.add(self.ask_id)

            # Add volume at the end and prices after sending order
            self.bid_volume.append(sum(bid_volumes))
            self.ask_volume.append(sum(ask_volumes))
            self.ask_prices = np.append(self.ask_prices, ask_prices[0])
            self.bid_prices = np.append(self.bid_prices, bid_prices[0])





