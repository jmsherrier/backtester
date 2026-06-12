"""Data loading and universe construction.

Responsible for turning raw price files into clean, aligned return matrices
indexed by date with one column per ticker. All downstream code assumes the
data handed to it is already point-in-time correct (no survivorship-free
guarantees yet, but no forward-filled prices either).
"""

from backtester.data.loading import (
    generate_gbm_prices,
    load_prices_csv,
    prices_to_returns,
)

__all__ = [
    "generate_gbm_prices",
    "load_prices_csv",
    "prices_to_returns",
]
