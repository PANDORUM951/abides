from agent.TradingAgent import TradingAgent
from util.util import log_print

import pandas as pd
import numpy as np

# Discrete grid for the quoting epsilon. Relative to the reference half-spread:
#   epsilon = -1 -> spread_mm = 0 (quotes at the mid, so the MM effectively does nothing)
#   epsilon =  0 -> spread_mm = spread_ref (quotes at the observed best bid/ask)
#   epsilon =  1 -> spread_mm = 2 * spread_ref (quotes twice as wide)
QUOTE_EPSILONS = np.round(np.arange(-1, 1.01, 0.2), 2)

# Discrete grid for the hedge epsilon. Fraction of the current inventory to unwind.
HEDGE_EPSILONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

class RandomMarketMakerAgent(TradingAgent):
    """
    Naive market maker that randomises its quoting spread and its hedging on every wake-up.

    At each wake-up it:
        1. Cancels all of its resting limit orders.
        2. Samples one quoting epsilon (uniform over QUOTE_EPSILONS) and one hedge epsilon
           (uniform over HEDGE_EPSILONS).
        3. Hedges: sends a market order opposite to the sign of its inventory, for
           hedge_eps * |inventory| shares. Does nothing if inventory is zero or hedge_eps is zero.
        4. Quotes one bid and one ask of fixed size, at mid -/+ spread_mm, where
           spread_mm = spread_ref * (1 + quote_eps) and spread_ref is the observed half-spread.

    The hedge is sent before the new quotes so that the market order cannot match against
    the agent's own freshly placed limit orders.

    """

    def __init__(self, id, name, type, symbol, starting_cash, min_size=250, max_size=500 , wake_up_freq='1s',
                 log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol      # Symbol traded
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.wake_up_freq = wake_up_freq  # Frequency of agent wake up
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)

        if can_trade:
            self.cancelOrders()
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        """ Once the spread arrives, hedge, quote, and schedule the next wake-up """
        if self.mkt_closed: return

        self.hedge()
        self.quote()

        self.setWakeup(currentTime + self.getWakeFrequency())
        self.state = 'AWAITING_WAKEUP'

    # Strategy Logic

    def hedge(self):
        """ Unwind hedge_eps * |inventory| share with a market order opposite to the inventory sign """
        hedge_eps = self.random_state.choice(HEDGE_EPSILONS)
        inventory = self.getHoldings(self.symbol)

        hedge_qty = int(round(abs(inventory) * hedge_eps))
        if hedge_qty <= 0: return

        # Long inventory -> sell; short inventory -> buy
        is_buy = inventory > 0
        log_print("{} hedging {} shares (eps={}, inventory={})", self.name, hedge_qty, hedge_eps, inventory)
        self.placeMarketOrder(self.symbol, hedge_qty, is_buy)


    def quote(self):
        bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
        if bid is None or ask is None:
            # One-sided or empty book: no reference spread, so skip quoting this round
            log_print("{} skipping quotes: no bid/ask in book", self.name)
            return

        quote_eps = self.random_state.choice(QUOTE_EPSILONS)

        mid = (bid + ask) / 2
        spread_ref = (ask - bid) / 2
        spread_mm = spread_ref * (1 + quote_eps) 

        # epsilon = -1 -> zero spread. Do nothing instead of quoting
        if spread_mm <= 0:
            log_print("{} quote_eps={}: zero spread, not quoting", self.name, quote_eps)
            return

        bid_price = int(round(mid - spread_mm))
        ask_price = int(round(mid + spread_mm))

        order_size = round(self.random_state.randint(self.min_size, self.max_size) / 2) # order size per LOB side

        log_print("{} quoting bid {} / ask {} (eps={})", self.name, bid_price, ask_price, quote_eps)
        self.placeLimitOrder(self.symbol, order_size, True, bid_price)
        self.placeLimitOrder(self.symbol, order_size, False, bid_price)


    def cancelOrders(self):
        """ cancels all resting limit orders placed by the market maker """
        for _, order in self.orders.items():
            self.cancelOrder(order)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)