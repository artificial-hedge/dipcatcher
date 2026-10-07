"""Unit tests for quant_fund.models._bx_synth."""

from __future__ import annotations

import numpy as np

from quant_fund.models._bx_synth import bandit_env, psrl_env


def test_bandit_env_deterministic_and_valid() -> None:
    mu1, sd1 = bandit_env(9)
    mu2, sd2 = bandit_env(9)
    assert np.array_equal(mu1, mu2)
    assert np.array_equal(sd1, sd2)
    assert np.all((mu1 >= 0.1) & (mu1 <= 1.0))
    assert np.all(sd1 > 0)


def test_psrl_env_transition_rows_normalized() -> None:
    P0, P1, R0, R1 = psrl_env(4)
    assert np.allclose(P0.sum(1), 1.0)
    assert np.allclose(P1.sum(1), 1.0)
    assert np.all(P0 >= 0) and np.all(P1 >= 0)
    assert np.all((R0 >= 0) & (R0 <= 1))
    assert np.all((R1 >= 0) & (R1 <= 1))


def test_psrl_env_deterministic() -> None:
    a = psrl_env(4)
    b = psrl_env(4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
