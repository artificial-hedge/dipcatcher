"""Adversarial probes for pmcmc_sv."""

import numpy as np
import pytest

from quant_fund.models import pmcmc_sv as pm


def test_pf_loglik_rejects_nan_params():
    r, _ = pm.simulate_sv(120, seed=1)
    with pytest.raises(ValueError, match="finite"):
        pm.sv_pf_loglik(r, mu=np.nan, phi=0.9, sigma_eta=0.1)
    with pytest.raises(ValueError, match="finite"):
        pm.sv_pf_loglik(r, mu=-4.0, phi=np.inf, sigma_eta=0.1)


def test_jacobian_in_acceptance(monkeypatch):
    """With a constant likelihood, the ONLY thing that can reject an
    in-bounds proposal is the Jacobian correction.  Under the old code
    (alpha = exp(ll_p - ll_c) only) every in-bounds proposal is accepted
    -> acceptance ~1.0.  With the Jacobian the rate drops far below 1."""
    monkeypatch.setattr(pm, "sv_pf_loglik", lambda *a, **k: -500.0)
    r, _ = pm.simulate_sv(120, seed=2)
    res = pm.pmmh_sv(
        r,
        n_iter=300,
        n_particles=10,
        theta0=np.array([-4.0, 0.9, 0.2]),
        seed=3,
    )
    # Jacobian-only chain: proposals moving phi up get J_p < J_c -> some
    # rejects; equal-ll-accept-everything behavior would give ~0.99.
    # (pinned seed: 0.91 with the Jacobian, ~1.0 without)
    assert res.acceptance_rate < 0.96


def test_determinism():
    r, _ = pm.simulate_sv(150, seed=4)
    a = pm.pmmh_sv(r, n_iter=60, n_particles=30, seed=5)
    b = pm.pmmh_sv(r, n_iter=60, n_particles=30, seed=5)
    np.testing.assert_allclose(a.post_mean, b.post_mean)
    assert a.acceptance_rate == b.acceptance_rate


def test_posterior_in_domain():
    r, _ = pm.simulate_sv(200, seed=6)
    res = pm.pmmh_sv(r, n_iter=80, n_particles=30, seed=7)
    assert abs(res.post_mean[1]) < 1.0
    assert res.post_mean[2] > 0.0
