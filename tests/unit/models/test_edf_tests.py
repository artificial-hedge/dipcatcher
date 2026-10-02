"""EDF goodness-of-fit tests (KS, CvM, AD)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from quant_fund.models.edf_tests import (
    ad_statistic,
    ad_test,
    cvm_statistic,
    cvm_test,
    ks_statistic,
    ks_test,
)


def test_ks_stat_zero_for_exact_ecdf():
    x = np.linspace(0.05, 0.95, 20)
    # empirical vs its own empirical CDF: bounded diff
    d = ks_statistic(
        x,
        lambda t: (
            stats.uniform().cdf(t) * 0 + np.searchsorted(np.sort(x), t, side="right") / len(x)
        ),
    )
    assert d < 0.06


def test_ks_rejects_shifted():
    rng = np.random.default_rng(0)
    x = rng.beta(3, 1, 200)
    out = ks_test(x, stats.uniform().cdf)
    assert out["pvalue"] < 0.01
    assert out["stat"] > 0.2


def test_ks_accepts_uniform():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 400)
    out = ks_test(x, stats.uniform().cdf)
    assert out["pvalue"] > 0.05


def test_cvm_stat_matches_scipy():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 100)
    ours = cvm_statistic(x, stats.uniform().cdf)
    theirs = stats.cramervonmises(x, stats.uniform().cdf).statistic
    assert ours == pytest.approx(theirs, rel=1e-10)


def test_ad_stat_positive_and_ordered():
    rng = np.random.default_rng(3)
    good = ad_statistic(rng.uniform(0, 1, 100), stats.uniform().cdf)
    bad = ad_statistic(rng.beta(2, 5, 100), stats.uniform().cdf)
    assert bad > good


def test_ad_test_rejects_wrong_cdf():
    rng = np.random.default_rng(4)
    x = rng.beta(5, 1.5, 200)
    out = ad_test(x, stats.uniform().cdf)
    assert out["pvalue"] < 0.01


def test_input_validation():
    with pytest.raises(ValueError):
        ks_test(np.array([np.nan] * 30), stats.uniform().cdf)
    with pytest.raises(ValueError):
        cvm_test(np.ones(3), stats.uniform().cdf)
    with pytest.raises(ValueError):
        ad_test(np.full(40, np.inf), stats.uniform().cdf)
