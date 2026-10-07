"""Unit tests for quant_fund.models._ex_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._ex_synth import (
    DISTRACTORS,
    GOAL,
    SIDE,
    WALLS,
    neighbors,
    null_bonus,
    q_learn,
    reward,
    s2i,
    state_feat,
)


def test_grid_topology_consistent() -> None:
    assert GOAL == (0, SIDE - 1)
    assert not (DISTRACTORS & WALLS)
    assert GOAL not in WALLS


def test_neighbors_respect_walls_and_bounds() -> None:
    for wall in WALLS:
        assert wall not in [sp for _, sp in neighbors((wall[0] - 1, wall[1]))]
    corner = neighbors((0, 0))
    assert len(corner) == 2
    assert all(0 <= sp[0] < SIDE and 0 <= sp[1] < SIDE for _, sp in corner)


def test_reward_values() -> None:
    assert reward(GOAL) == 1.0
    assert reward(next(iter(DISTRACTORS))) == 0.1
    assert reward((6, 6)) == 0.0


def test_state_feat_flags() -> None:
    xg = state_feat(GOAL)
    xd = state_feat(next(iter(DISTRACTORS)))
    assert xg[3] == 1.0 and xg[2] == 0.0
    assert xd[2] == 1.0 and xd[3] == 0.0
    assert xg[5] == 1.0


def test_s2i_bijective() -> None:
    seen = {s2i((r, c)) for r in range(SIDE) for c in range(SIDE)}
    assert seen == set(range(SIDE * SIDE))


def test_q_learn_deterministic_and_bounded() -> None:
    Q1, c1, s1 = q_learn(null_bonus, seed=0, episodes=40, steps=30)
    Q2, c2, s2 = q_learn(null_bonus, seed=0, episodes=40, steps=30)
    np.testing.assert_array_equal(Q1, Q2)
    assert (c1, s1) == (c2, s2)
    assert 0.0 <= c1 <= 1.0 and 0.0 <= s1 <= 1.0
    assert Q1.shape == (SIDE * SIDE, 4)


def test_q_learn_intrinsic_bonus_helps_coverage() -> None:
    def count_bonus(s, sp, st, ep, rng):
        return 0.3  # constant novelty-ish push

    _Q, c_explore, _s = q_learn(count_bonus, seed=1, episodes=60, steps=40)
    _Q0, c_flat, _s0 = q_learn(null_bonus, seed=1, episodes=60, steps=40)
    assert c_explore >= c_flat


def test_q_learn_rejects_vacuous_loops() -> None:
    with pytest.raises(ValueError, match="episodes"):
        q_learn(null_bonus, episodes=0)
    with pytest.raises(ValueError, match="steps"):
        q_learn(null_bonus, episodes=5, steps=0)


def test_q_learn_rejects_bad_hyperparams() -> None:
    with pytest.raises(ValueError, match="eps"):
        q_learn(null_bonus, eps=1.5)
    with pytest.raises(ValueError, match="gamma"):
        q_learn(null_bonus, gamma=-0.1)
    with pytest.raises(ValueError, match="lr"):
        q_learn(null_bonus, lr=0.0)
