"""Unit tests for quant_fund.models._eb_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._eb_synth import gauss_baseline_mmd, mmd, moon_data


def test_moon_data_shapes_and_determinism() -> None:
    tr1, te1 = moon_data(seed=3, n=200)
    tr2, te2 = moon_data(seed=3, n=200)
    np.testing.assert_array_equal(tr1, tr2)
    np.testing.assert_array_equal(te1, te2)
    assert tr1.shape == (200, 2)
    assert te1.shape == (500, 2)


def test_mmd_identical_samples_near_zero() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 2))
    assert mmd(X, X) < 1e-9


def test_mmd_positive_for_different_dists() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 2))
    Y = rng.standard_normal((400, 2)) + 3.0
    assert mmd(X, Y) > mmd(X, X) + 0.1


def test_mmd_rejects_zero_bandwidth_laundering() -> None:
    # gam=0 makes every kernel value 1.0 -> MMD is exactly 0 for ANY data,
    # a fake-perfect score. Must raise instead of laundering.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2))
    Y = rng.standard_normal((50, 2)) + 100.0
    with pytest.raises(ValueError, match="gam"):
        mmd(X, Y, gam=0.0)
    with pytest.raises(ValueError, match="gam"):
        mmd(X, Y, gam=-1.0)
    with pytest.raises(ValueError, match="gam"):
        mmd(X, Y, gam=np.inf)


def test_mmd_rejects_empty_samples() -> None:
    X = np.zeros((0, 2))
    Y = np.ones((5, 2))
    with pytest.raises(ValueError, match="non-empty"):
        mmd(X, Y)


def test_gauss_baseline_mmd_deterministic() -> None:
    Xtr, Xte = moon_data(seed=1, n=300)
    assert gauss_baseline_mmd(Xtr, Xte) == gauss_baseline_mmd(Xtr, Xte)


def test_langevin_rejects_vacuous_steps() -> None:
    torch = pytest.importorskip("torch")
    from quant_fund.models._eb_synth import langevin, make_energy

    net = make_energy(torch)
    with pytest.raises(ValueError, match="steps"):
        langevin(torch, net, 16, steps=0)
    with pytest.raises(ValueError, match="step"):
        langevin(torch, net, 16, steps=5, step=0.0)
    with pytest.raises(ValueError, match="n"):
        langevin(torch, net, 0, steps=5)


def test_langevin_seeded_determinism() -> None:
    torch = pytest.importorskip("torch")
    from quant_fund.models._eb_synth import langevin, make_energy

    net = make_energy(torch)
    out1 = langevin(torch, net, 8, steps=3, seed=5)
    out2 = langevin(torch, net, 8, steps=3, seed=5)
    assert torch.equal(out1, out2)
    assert out1.shape == (8, 2)
