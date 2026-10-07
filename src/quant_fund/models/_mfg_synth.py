"""Synthetic fixtures shared by the game-theory canon (SYNTHETIC).

One LQ control system (for the mean-field games), one Cournot market,
one leader-follower market, one small zero-sum stochastic game, and one
congestion potential game — all deterministic parameters so every module
benches against the same planted equilibria.
"""

import numpy as np

# LQ system: dx = (a*x + b*u)dt + sigma dW, cost 0.5*(q*x^2 + r*u^2) + eta*(x-mbar)^2
LQ_A, LQ_B, LQ_SIG, LQ_Q, LQ_R, LQ_ETA, LQ_H = -0.1, 1.0, 0.2, 1.0, 1.0, 0.5, 0.02

# Cournot: inverse demand P = A - b*Q, firm marginal costs c_i
COU_A, COU_B = 100.0, 1.0
COU_C = np.array([10.0, 14.0, 20.0, 30.0])

# Stochastic game: 2 states, 2x2 simultaneous matrix games per state,
# deterministic-ish transitions with a planted discount.
SG_GAMMA = 0.9
SG_M = [
    np.array([[3.0, 1.0], [0.0, 2.0]]),  # state 0 payoff (row max, col min)
    np.array([[1.0, 4.0], [2.0, 0.0]]),  # state 1
]
SG_P = [
    np.array([[[0.7, 0.3], [0.4, 0.6]], [[0.6, 0.4], [0.3, 0.7]]]),  # s0: [i,j]->p(next)
    np.array([[[0.5, 0.5], [0.2, 0.8]], [[0.8, 0.2], [0.6, 0.4]]]),  # s1
]

# Congestion potential game: N users choose one of R routes; latency
# l_r(x) = a_r*x + b_r; potential Phi = sum_r sum_{x<=n_r} l_r(x).
POT_A = np.array([1.0, 2.0, 3.0, 4.0])
POT_B = np.array([0.0, 4.0, 8.0, 12.0])
POT_N = 24
