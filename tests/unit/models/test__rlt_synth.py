"""Invariant probes for _rlt_synth (constants-only fixture)."""

import numpy as np

from quant_fund.models._rlt_synth import (
    ARM_MEANS,
    EXP_P,
    MDP_GAMMA,
    MDP_P,
    MDP_R,
    T_BANDIT,
    T_EXP,
)


def test_bandit_has_unique_best_arm():
    assert ARM_MEANS[0] == ARM_MEANS.max()
    assert (ARM_MEANS[1:] < ARM_MEANS[0]).all()
    assert ((ARM_MEANS >= 0) & (ARM_MEANS <= 1)).all()
    assert T_BANDIT > 0


def test_expert_0_has_the_edge():
    assert EXP_P[0] == EXP_P.max()  # edge +0.1 over the 0.5 baseline experts
    assert ((EXP_P > 0) & (EXP_P < 1)).all()
    assert T_EXP > 0


def test_mdp_transition_rows_are_stochastic():
    np.testing.assert_allclose(MDP_P.sum(1), 1.0)
    assert (MDP_P >= 0).all()
    assert MDP_P.shape == (4, 4) and MDP_R.shape == (4,)
    assert 0 < MDP_GAMMA < 1


def test_mdp_bellman_contraction_consistent():
    # value iteration converges; check reward-reachable state 3 > others
    v = np.zeros(4)
    for _ in range(500):
        v = MDP_R + MDP_GAMMA * MDP_P @ v
    assert v[3] > v[2] > v[1] > v[0]
    assert abs(v - (MDP_R + MDP_GAMMA * MDP_P @ v)).max() < 1e-10
