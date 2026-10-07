"""Synthetic auction fixtures shared by the auction-theory canon (SYNTHETIC).

Uniform/regular value distributions, bidder count, and a planted order
book for the double-auction module. All numbers deterministic (seeded).
"""

import numpy as np

N_BID = 4
VAL_LO, VAL_HI = 0.0, 1.0


def iid_values(seed: int, n: int = N_BID, m: int = 6000) -> np.ndarray:
    """m auctions x n bidders, uniform iid values."""
    return np.random.default_rng(seed).uniform(VAL_LO, VAL_HI, (m, n))


def uniform_bne_bid(v: np.ndarray, n: int = N_BID) -> np.ndarray:
    """Symmetric first-price BNE for U[0,1]: b(v) = (n-1)/n * v."""
    return (n - 1) / n * v


# GSP: 3 slots, click-through rates, bidder values per click
GSP_CTR = np.array([0.30, 0.15, 0.05])
GSP_VALUES = np.array([8.0, 5.0, 3.0, 1.0])
