"""Unit tests for quant_fund.models.black_litterman."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.black_litterman import (
    bench_bl,
    bl_posterior,
    implied_returns,
    synth_bl,
)


def test_no_view_collapse() -> None:
    sigma, w_eq, _, _ = synth_bl(seed=1)
    r = bl_posterior(sigma, w_eq, np.zeros((0, 3)), np.zeros(0))
    assert np.allclose(np.asarray(r["mu_bl"]), implied_returns(sigma, w_eq))


def test_view_tilts_correct_asset() -> None:
    sigma, w_eq, p, q = synth_bl(seed=2)
    r = bl_posterior(sigma, w_eq, p, q)
    mu = np.asarray(r["mu_bl"])
    pi = np.asarray(r["pi"])
    assert mu[0] > pi[0] and mu[0] < q[0] + 0.01


def test_posterior_cov_psd() -> None:
    sigma, w_eq, p, q = synth_bl(seed=3)
    r = bl_posterior(sigma, w_eq, p, q)
    assert float(r["max_eig_neg"]) <= 1e-12


def test_input_validation() -> None:
    sigma, w_eq, p, q = synth_bl(seed=4)
    with pytest.raises(ValueError):
        bl_posterior(sigma, w_eq, p, q, tau=-1.0)
    with pytest.raises(ValueError):
        bl_posterior(sigma, w_eq, p[:, :2], q)


def test_bench_contract() -> None:
    out = bench_bl()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_bl_pull0"] > 0.0
