"""Unit tests for quant_fund.models.nested_sampling."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.nested_sampling import bench_nested_sampling, nested_sampling


def _gauss_prob():
    d = 2
    mu_l = np.array([1.0, 0.5])
    sd_l = np.array([0.5, 0.5])
    sd_p = np.array([3.0, 3.0])

    def loglike(x):
        r = (x - mu_l) / sd_l
        return float(-0.5 * np.sum(r * r))

    def logprior(x):
        r = x / sd_p
        return float(-0.5 * np.sum(r * r))

    def sp(rng):
        return rng.standard_normal(d) * sd_p

    return d, mu_l, sd_l, sd_p, loglike, logprior, sp


def test_nested_returns_keys() -> None:
    _, _, _, _, ll, lp, sp = _gauss_prob()
    out = nested_sampling(ll, lp, sp, n_live=40, n_max=800, seed=0)
    for k in ("logz", "h", "dead_points", "dead_loglike", "posterior_weights", "n_iter"):
        assert k in out


def test_dead_loglike_monotone() -> None:
    _, _, _, _, ll, lp, sp = _gauss_prob()
    out = nested_sampling(ll, lp, sp, n_live=40, n_max=800, seed=0)
    dll = np.asarray(out["dead_loglike"])
    assert np.all(np.diff(dll) >= -1e-9)


def test_posterior_weights_sum_one() -> None:
    _, _, _, _, ll, lp, sp = _gauss_prob()
    out = nested_sampling(ll, lp, sp, n_live=40, n_max=800, seed=0)
    pw = np.asarray(out["posterior_weights"])
    assert pw.sum() == pytest.approx(1.0, rel=1e-6)
    assert np.all(pw >= 0.0)


def test_posterior_mean_recovered() -> None:
    _, mu_l, sd_l, _, ll, lp, sp = _gauss_prob()
    out = nested_sampling(ll, lp, sp, n_live=80, n_max=2000, seed=1)
    dead = np.asarray(out["dead_points"])
    pw = np.asarray(out["posterior_weights"])
    mean_hat = (dead * pw[:, None]).sum(axis=0)
    assert np.linalg.norm(mean_hat - mu_l) < 0.4


def test_rejects_nonfinite_likelihood() -> None:
    def bad_ll(x):
        return float("nan")

    def lp(x):
        return 0.0

    def sp(rng):
        return rng.standard_normal(2)

    with pytest.raises(ValueError):
        nested_sampling(bad_ll, lp, sp, n_live=10, n_max=20, seed=0)


def test_bench_nested_sampling_score() -> None:
    out = bench_nested_sampling()
    assert out["score"] == pytest.approx(1.0)
    assert out["synthetic_ns_logz_err"] < 0.4
