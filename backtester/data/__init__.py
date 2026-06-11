"""Data loading and universe construction.

Responsible for turning raw price files into clean, aligned return matrices
indexed by date with one column per ticker. All downstream code assumes the
data handed to it is already point-in-time correct (no survivorship-free
guarantees yet, but no forward-filled prices either).
"""
