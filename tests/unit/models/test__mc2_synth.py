"""Adversarial probes for quant_fund.models._mc2_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._mc2_synth import MU, SIGMA, errors, ess, indep_mh, logp


def test_logp_at_mode() -> None:
    assert logp(MU.copy()) == pytest.approx(0.0)
    assert logp(MU + 3.0) < logp(MU + 1.0)


def test_ess_iid_vs_correlated() -> None:
    rng = np.random.default_rng(0)
    iid = rng.standard_normal(4000)
    e_iid = ess(iid)
    assert e_iid > 3000  # nearly all effective
    ar = np.zeros(4000)
    for i in range(1, 4000):
        ar[i] = 0.98 * ar[i - 1] + rng.standard_normal() * 0.2
    e_ar = ess(ar)
    assert e_ar < e_iid


def test_ess_hostile_inputs() -> None:
    with pytest.raises(ValueError):
        ess(np.array([]))
    with pytest.raises(ValueError):
        ess(np.array([1.0]))
    with pytest.raises(ValueError):
        ess(np.ones(10), max_lag=1)


def test_errors_zero_at_truth() -> None:
    rng = np.random.default_rng(1)
    fake = rng.multivariate_normal(MU, SIGMA, 2000)
    m_err, c_err = errors(fake)
    assert m_err < 0.2
    assert c_err < 0.5


def test_errors_hostile_inputs() -> None:
    with pytest.raises(ValueError):
        errors(np.zeros((0, 2)))
    with pytest.raises(ValueError):
        errors(np.zeros((5, 3)))


def test_indep_mh_deterministic_and_shape() -> None:
    r1 = np.random.default_rng(4)
    r2 = np.random.default_rng(4)
    a = indep_mh(200, r1)
    b = indep_mh(200, r2)
    assert a.shape == (200, 2)
    assert np.array_equal(a, b)


def test_indep_mh_hostile_params() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        indep_mh(0, rng)
    with pytest.raises(ValueError):
        indep_mh(10, rng, prop_sd=0.0)
    with pytest.raises(ValueError):
        indep_mh(10, rng, prop_sd=np.inf)
