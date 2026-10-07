"""Adversarial probes for quant_fund.models._mem_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._mem_synth import synth_copy, synth_dynamics


def test_synth_copy_structure() -> None:
    rng = np.random.default_rng(0)
    x, y = synth_copy(4, 6, rng)
    assert x.shape == (4, 13, 9)
    assert y.shape == (4, 6)
    # delimiter flag set at position t
    assert (x[:, 6, 8] == 1.0).all()
    # payload one-hots match y
    assert np.array_equal(x[:, :6, :8].argmax(-1), y)


def test_synth_copy_deterministic_and_hostile() -> None:
    a = synth_copy(2, 3, np.random.default_rng(7))
    b = synth_copy(2, 3, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0])
    with pytest.raises(ValueError):
        synth_copy(0, 3, np.random.default_rng(0))
    with pytest.raises(ValueError):
        synth_copy(2, 0, np.random.default_rng(0))


def test_synth_dynamics_recursion() -> None:
    rng = np.random.default_rng(2)
    x, u, xn = synth_dynamics(3, 10, rng)
    assert x.shape == (3, 11, 2)
    assert u.shape == (3, 10, 1)
    assert xn.shape == (3, 10, 2)
    # x_next equals the tail of the trajectory
    assert np.array_equal(xn, x[:, 1:])


def test_synth_dynamics_hostile() -> None:
    with pytest.raises(ValueError):
        synth_dynamics(0, 5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        synth_dynamics(2, 0, np.random.default_rng(0))
