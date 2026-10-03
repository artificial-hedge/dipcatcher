"""Shared fixture for wave-197 inventory canon — demand process +
holding/shortage cost simulation harness.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

D_RATE = 20.0  # units/week
H_COST = 0.5  # holding $/unit/week
S_COST = 5.0  # shortage $/unit
K_ORDER = 60.0  # fixed order cost


def demand(seed: int, weeks: int = 104, rate: float = D_RATE) -> FloatArray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.poisson(rate, weeks).astype(np.float64))


def simulate(policy, seed: int, weeks: int = 104) -> tuple[float, float, float]:
    """policy(inventory_position, week) -> order qty.
    Returns (avg_cost, fill_rate, n_orders). Lead time = 1 week."""
    d = demand(seed, weeks)
    inv = 0.0
    pending: list[tuple[int, float]] = []
    tot_cost = tot_hold = tot_short = 0.0
    n_orders = 0
    filled = 0.0
    for w, dem in enumerate(d):
        for aw, q in pending:
            if aw <= w:
                inv += q
        pending = [(aw, q) for aw, q in pending if aw > w]
        sell = min(inv, dem)
        inv -= sell
        filled += sell
        tot_short += S_COST * (dem - sell)
        tot_hold += H_COST * max(inv, 0.0)
        ip = inv + sum(q for _, q in pending)
        q = float(policy(ip, w))
        if q > 0:
            pending.append((w + 1, q))
            tot_cost += K_ORDER
            n_orders += 1
    avg = (tot_cost + tot_hold + tot_short) / weeks
    fill = filled / max(float(d.sum()), 1e-9)
    return avg, fill, float(n_orders)
