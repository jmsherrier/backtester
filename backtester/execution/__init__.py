"""Transaction-cost and slippage models.

Every change in position incurs a cost. Models here convert a vector of trades
(position deltas) into a cost charged against P&L. The default model combines a
fixed commission (bps of notional traded) with a simple slippage term.
"""
