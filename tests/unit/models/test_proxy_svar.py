"""Tests for proxy SVAR + sign restrictions (models/proxy_svar.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.proxy_svar import (
    bench_proxy_svar,
    companion_irf,
    proxy_svar,
    sign_restrict_draws,
    synth_svar,
    var_fit,
)


@pytest.fixture
def sim():
    return synth_svar(t=400, seed=7)


def test_var_fit_shapes(sim):
    fit = var_fit(sim["y"], p=2)
    assert fit["A"].shape == (2, 4)
    assert fit["Sigma"].shape == (2, 2)
    assert fit["U"].shape[1] == 2
    assert np.all(np.linalg.eigvalsh(fit["Sigma"]) > 0)


def test_var_fit_recovers_dynamics(sim):
    fit = var_fit(sim["y"], p=1)
    a_est = fit["A"][:, :2]
    assert np.linalg.norm(a_est - sim["A"]) < 0.25


def test_companion_irf_decay():
    a = np.array([[0.5, 0.0], [0.0, 0.5]])
    psi = companion_irf(a, 2, horizon=5)
    assert psi.shape == (6, 2, 2)
    assert np.allclose(psi[0], np.eye(2))
    assert np.allclose(psi[1], a)
    assert np.allclose(psi[5][0, 0], 0.5**5)


def test_proxy_svar_relevance(sim):
    out = proxy_svar(sim["y"], sim["z"], p=1, target_var=0, horizon=8)
    assert out["first_stage_f"] > 100.0
    assert out["weak_instrument"] == 0.0
    assert out["corr_z_target"] > 0.5


def test_proxy_svar_irf_shape(sim):
    out = proxy_svar(sim["y"], sim["z"], p=1, target_var=0, horizon=8)
    irf = out["irf"]
    # shock 1: var0 up at h0, var1 down at h0, then decays
    assert irf[0, 0] == pytest.approx(1.0)
    assert irf[0, 1] < 0
    assert abs(irf[6, 0]) < abs(irf[0, 0])


def test_proxy_svar_weak_instrument(sim):
    rng = np.random.default_rng(0)
    z = rng.standard_normal(sim["y"].shape[0])
    try:
        out = proxy_svar(sim["y"], z, p=1, target_var=0, horizon=4)
        assert out["weak_instrument"] == 1.0
    except ValueError:
        pass  # outright irrelevance also acceptable


def test_proxy_svar_misaligned_z(sim):
    with pytest.raises(ValueError):
        proxy_svar(sim["y"], sim["z"][:-3], p=1, target_var=0)


def test_sign_restrict_shapes(sim):
    out = sign_restrict_draws(
        sim["y"],
        p=1,
        restrictions={(1, 0): -1, (0, 0): 1},
        n_draws=200,
        horizon=6,
        seed=1,
    )
    assert out["n_kept"] > 10
    assert out["irfs"].shape == (out["n_kept"], 7, 2)
    # every kept draw complies
    assert np.all(out["irfs"][:, 0, 1] < 0)
    assert np.all(out["irfs"][:, 0, 0] > 0)


def test_sign_restrict_set_not_point(sim):
    out = sign_restrict_draws(
        sim["y"], p=1, restrictions={(1, 0): -1}, n_draws=300, horizon=4, seed=1
    )
    assert 0.0 < out["keep_rate"] < 1.0


def test_sign_restrict_determinism(sim):
    a = sign_restrict_draws(
        sim["y"], p=1, restrictions={(1, 0): -1}, n_draws=100, horizon=4, seed=9
    )
    b = sign_restrict_draws(
        sim["y"], p=1, restrictions={(1, 0): -1}, n_draws=100, horizon=4, seed=9
    )
    assert np.array_equal(a["irfs"], b["irfs"])


def test_sign_restrict_validation(sim):
    with pytest.raises(ValueError):
        sign_restrict_draws(sim["y"], p=1, restrictions=None)
    with pytest.raises(ValueError):
        sign_restrict_draws(sim["y"], p=1, restrictions={(9, 0): 1})


def test_synth_structure():
    sim = synth_svar(t=200, seed=3)
    assert sim["y"].shape == (200, 2)
    assert sim["z"].shape == (200,)
    assert np.array_equal(sim["B0"], np.array([[1.0, 0.5], [-0.7, 1.0]]))


def test_bench_keys():
    out = bench_proxy_svar()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_irf_relerr"] < 0.4
    assert out["synthetic_first_stage_f"] > 50.0
    assert out["synthetic_weak_instrument_flagged"] == 1.0
    assert 0.0 < out["synthetic_sign_keep_rate"] < 1.0
    assert out["synthetic_determinism"] == 1.0
