"""Synthetic RL-theory fixtures shared by the convergence-bounds canon (SYNTHETIC).

A small planted bandit (known gaps) and a tiny chain MDP (known values)
so every module verifies a *rate* — regret bound shapes, contraction
constants, and 1/t TD convergence — against its theory constant.
"""

import numpy as np

# K-armed bandit: arm 0 best at 0.9, rest staggered below
ARM_MEANS = np.array([0.9, 0.7, 0.55, 0.4, 0.3])
T_BANDIT = 40000

# expert losses for multiplicative weights: expert 0 edge = 0.1
EXP_P = np.array([0.6, 0.5, 0.5, 0.5, 0.5])
T_EXP = 40000

# tiny 4-state chain MDP (random walk with absorbing ends)
MDP_P = np.array(
    [
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5, 0.5],
    ]
)
MDP_R = np.array([0.0, 0.0, 0.0, 1.0])
MDP_GAMMA = 0.9
