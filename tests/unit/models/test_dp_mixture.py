"""Tests for models/dp_mixture.py — DP Gaussian-mixture CAVI."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import digamma

from quant_fund.models.dp_mixture import (
    bench_dp_mixture,
    caviar_dpmm,
    dpmm_fit_predict,
    effective_components,
    elbo_dpmm,
    stick_break_expectations,
    synth_regimes,
)


@pytest.fixture(scope="module")
def blob() -> tuple[np.ndarray, np.ndarray]:
    return synth_regimes(240, 3, 3, 6.0, seed=0)


@pytest.fixture(scope="module")
def fit(blob):
    x, _ = blob
    return caviar_dpmm(x, k_trunc=10, alpha=1.0, n_iter=120, seed=0)


class TestGuards:
    def test_bad_inputs(self, blob):
        x, _ = blob
        with pytest.raises(ValueError):
            caviar_dpmm(np.ones((5, 2)), k_trunc=5)
        with pytest.raises(ValueError):
            caviar_dpmm(np.full((50, 2), np.nan))
        with pytest.raises(ValueError):
            caviar_dpmm(x, k_trunc=1)
        with pytest.raises(ValueError):
            caviar_dpmm(x, k_trunc=10, alpha=-1.0)
        with pytest.raises(ValueError):
            caviar_dpmm(x, k_trunc=10, n_iter=1)
        with pytest.raises(ValueError):
            effective_components(np.ones((10, 3)), thresh=0.9)
        with pytest.raises(ValueError):
            stick_break_expectations(np.ones(4), -np.ones(4))
        with pytest.raises(ValueError):
            synth_regimes(10, 2, 2, 5.0, 0)


class TestStickBreak:
    def test_expectation_math(self):
        g1 = np.array([2.0, 3.0])
        g2 = np.array([5.0, 1.0])
        elv, el1v = stick_break_expectations(g1, g2)
        np.testing.assert_allclose(elv, digamma(g1) - digamma(g1 + g2))
        np.testing.assert_allclose(el1v, digamma(g2) - digamma(g1 + g2))
        assert (elv < 0).all() and (el1v < 0).all()


class TestCAVI:
    def test_resp_normalized(self, blob, fit):
        np.testing.assert_allclose(fit.resp.sum(axis=1), 1.0, atol=1e-9)
        assert (fit.resp >= 0).all()

    def test_elbo_recorded_finite(self, fit):
        assert np.isfinite(fit.elbo_path).all()
        assert fit.elbo_path.size >= 2

    def test_elbo_roughly_monotone(self, fit):
        diffs = np.diff(fit.elbo_path)
        # CAVI is coordinate ascent — small numerical wiggles allowed
        assert (diffs >= -1e-6 * np.abs(fit.elbo_path[:-1])).mean() > 0.85

    def test_deterministic(self, blob):
        x, _ = blob
        a = caviar_dpmm(x, k_trunc=8, n_iter=60, seed=3)
        b = caviar_dpmm(x, k_trunc=8, n_iter=60, seed=3)
        np.testing.assert_array_equal(a.resp, b.resp)

    def test_seed_variation_allowed(self, blob):
        x, _ = blob
        a = caviar_dpmm(x, k_trunc=8, n_iter=60, seed=0)
        b = caviar_dpmm(x, k_trunc=8, n_iter=60, seed=1)
        # both fits are valid; responsibilities may differ under different init
        assert a.resp.shape == b.resp.shape


class TestRecovery:
    def test_recovers_three_blobs(self, blob, fit):
        _, truth = blob
        labels, _ = dpmm_fit_predict(blob[0], fit)
        ari = _ari(truth, labels)
        assert ari > 0.8

    def test_effective_component_count(self, fit):
        assert effective_components(fit.resp, thresh=0.02) == 3

    def test_predictive_density_positive(self, blob, fit):
        _, dens = dpmm_fit_predict(blob[0], fit)
        assert (dens > 0).all() and np.isfinite(dens).all()

    def test_overlap_harder_than_separated(self):
        # assignment sharpness and partition recovery degrade as blobs overlap
        x_easy, t_easy = synth_regimes(200, 3, 3, 6.0, 5)
        x_hard, t_hard = synth_regimes(200, 3, 3, 2.5, 5)
        fit_e = caviar_dpmm(x_easy, k_trunc=8, n_iter=100, seed=5)
        fit_h = caviar_dpmm(x_hard, k_trunc=8, n_iter=100, seed=5)
        l_e, _ = dpmm_fit_predict(x_easy, fit_e)
        l_h, _ = dpmm_fit_predict(x_hard, fit_h)
        assert fit_e.resp.max(axis=1).mean() > fit_h.resp.max(axis=1).mean()
        assert _ari(t_easy, l_e) > _ari(t_hard, l_h)


class TestElboFn:
    def test_finite_on_fit(self, blob, fit):
        val = elbo_dpmm(
            blob[0],
            fit.resp,
            fit.gamma1,
            fit.gamma2,
            1.0,
            np.full(fit.gamma1.size, 1.0),
            fit.mean,
            np.ones_like(fit.tau),
            np.ones_like(fit.tau),
            blob[0].mean(axis=0),
            0.01,
            1.0,
            np.ones(3),
        )
        assert np.isfinite(val)


class TestBench:
    def test_bench_keys_finite(self):
        blob = bench_dp_mixture(seed=0)
        assert blob
        for k, v in blob.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float) and np.isfinite(v)

    def test_bench_contract_no_forbidden(self):
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in bench_dp_mixture(seed=0):
            assert not forbidden.intersection(k.split("_"))

    def test_bench_science(self):
        blob = bench_dp_mixture(seed=0)
        assert blob["synthetic_ari"] > 0.8
        assert blob["synthetic_k_hat_err"] < 0.5
        assert blob["synthetic_elbo_monotone_frac"] > 0.8
        assert blob["synthetic_assign_sharpness"] > 0.9
        assert blob["synthetic_determinism"] == 1.0


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    """Tiny ARI copy for the test module (kept local, no imports)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    n = a.size
    nij = np.zeros((np.unique(a).size, np.unique(b).size))
    for i, ua in enumerate(np.unique(a)):
        for j, ub in enumerate(np.unique(b)):
            nij[i, j] = ((a == ua) & (b == ub)).sum()
    c2 = lambda v: v * (v - 1) / 2.0  # noqa: E731
    s_ij = c2(nij).sum()
    s_a = c2(nij.sum(axis=1)).sum()
    s_b = c2(nij.sum(axis=0)).sum()
    tot = n * (n - 1) / 2.0
    exp_ = s_a * s_b / tot
    den = 0.5 * (s_a + s_b) - exp_
    return float((s_ij - exp_) / den) if den > 0 else 1.0
