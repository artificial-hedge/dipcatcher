"""Synthetic queueing-network fixtures shared by the reliability canon (SYNTHETIC).

A tiny event-driven queue simulator plus deterministic network parameters
(routing matrix, service rates, visit ratios) so the product-form and MVA
modules bench against both theory values and simulation.
"""

import numpy as np

# open Jackson: arrivals gamma to node 0, routing 0->1->2 with feedback
GAMMA = np.array([2.0, 0.0, 0.0])
MU = np.array([4.0, 5.0, 6.0])
P = np.array([[0.0, 1.0, 0.0], [0.0, 0.1, 0.9], [0.0, 0.0, 0.0]])

# closed network (Gordon-Newell / BCMP): 3 nodes, service demands
CN_MU = np.array([3.0, 4.0, 5.0])
CN_VISIT = np.array([1.0, 0.8, 0.6])
CN_N = 8

# repairable system: failure rate lam, repair rate mu (2-state CTMC)
AVAIL_LAM, AVAIL_MU = 0.02, 0.5

# renewal-reward: interrenewal exp(lam), reward per cycle = mean r
RR_LAM, RR_MEAN = 0.5, 3.0

# M/G/1 + vacations: exp service + exp vacations
VQ_LAM, VQ_MU, VQ_VAC = 1.5, 3.0, 0.5


def mm1_sim(seed: int, lam: float, mu: float, horizon: float = 20000.0) -> float:
    """Event-driven M/M/1 mean queue-length estimate."""
    rng = np.random.default_rng(seed)
    t, n = 0.0, 0
    area = 0.0
    while t < horizon:
        arr = rng.exponential(1.0 / lam)
        dep = rng.exponential(1.0 / mu) if n > 0 else np.inf
        dt = min(arr, dep)
        area += n * dt
        t += dt
        n += 1 if arr < dep else -1
    return area / t


def jackson_lam() -> np.ndarray:
    """Traffic equations: lambda = gamma + P^T lambda (solve by iteration)."""
    lam = GAMMA.copy()
    for _ in range(200):
        lam = GAMMA + P.T @ lam
    return lam
