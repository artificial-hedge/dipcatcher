"""k-double auction clearing on a planted order book.

Buy bids sorted desc, sell asks sorted asc; the k-double auction sets
price p = k*best_bid + (1-k)*best_ask at the crossing point. Bench:
cleared volume + price on an overlapping book vs no-trade (gapped) book,
and the surplus split between buyers and sellers vs naive midpoint.
"""

import numpy as np


def _clear(bids: np.ndarray, asks: np.ndarray, k: float = 0.5) -> tuple[float, int]:
    n = min(len(bids), len(asks))
    for i in range(n):
        if bids[i] < asks[i]:
            p = k * bids[i - 1] + (1 - k) * asks[i - 1] if i > 0 else 0.0
            return p, i
    p = k * bids[n - 1] + (1 - k) * asks[n - 1]
    return p, n


def _book(seed: int, gap: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mid = 50.0
    bids = np.sort(rng.normal(mid - 0.5 - gap / 2, 0.8, 40))[::-1]
    asks = np.sort(rng.normal(mid + 0.5 + gap / 2, 0.8, 40))
    return bids, asks


def bench_double_auction(seed: int = 4509) -> dict[str, float]:
    bids, asks = _book(seed, gap=0.0)
    p, vol = _clear(bids, asks)
    mid = 0.5 * (bids[0] + asks[0])
    surplus = float(np.sum(bids[:vol] - p) + np.sum(p - asks[:vol]))
    bids2, asks2 = _book(seed + 1, gap=6.0)
    p2, vol2 = _clear(bids2, asks2)
    return {
        "synthetic_da_price": p,
        "synthetic_da_volume": float(vol),
        "synthetic_da_mid_err": abs(p - mid),
        "synthetic_da_surplus": surplus,
        "synthetic_da_gap_volume": float(vol2),
        "synthetic_da_gap_price": p2,
    }
