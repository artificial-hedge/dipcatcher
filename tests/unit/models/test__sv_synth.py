"""Adversarial probes for _sv_synth (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._sv_synth import cindex, cox_ph, sv_data


def test_sv_data_deterministic():
    a = sv_data(0, n=32)
    b = sv_data(0, n=32)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_sv_data_risk_ordering():
    X, t, e, beta = sv_data(1, n=300)
    risk = X @ beta
    hi = risk > np.quantile(risk, 0.9)
    lo = risk < np.quantile(risk, 0.1)
    assert t[hi & (e == 1)].mean() < t[lo & (e == 1)].mean()


@pytest.mark.parametrize("n,d", [(0, 4), (4, 0)])
def test_sv_data_hostile(n, d):
    with pytest.raises(ValueError):
        sv_data(0, n=n, d=d)


def test_sv_data_beta_dim_mismatch():
    with pytest.raises(ValueError):
        sv_data(0, n=8, d=6, beta=np.ones(4))


def test_cindex_perfect_and_reversed():
    t = np.arange(10.0) + 1.0
    e = np.ones(10, dtype=np.int64)
    good = -t.copy()  # higher risk → shorter survival
    bad = t.copy()
    assert cindex(t, good, e) == pytest.approx(1.0)
    assert cindex(t, bad, e) == pytest.approx(0.0)


def test_cindex_ties_count_half():
    t = np.array([1.0, 2.0])
    e = np.ones(2, dtype=np.int64)
    r = np.array([0.5, 0.5])
    assert cindex(t, r, e) == pytest.approx(0.5)  # all-tied → chance, not perfect


def test_cindex_all_censored_raises():
    t = np.array([1.0, 2.0, 3.0])
    e = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError, match="no comparable"):
        cindex(t, np.zeros(3), e)


def test_cindex_length_mismatch():
    t = np.array([1.0, 2.0])
    e = np.ones(2, dtype=np.int64)
    with pytest.raises(ValueError):
        cindex(t, np.ones(3), e)


def test_cindex_nonfinite():
    t = np.array([1.0, np.nan])
    e = np.ones(2, dtype=np.int64)
    with pytest.raises(ValueError):
        cindex(t, np.ones(2), e)


def test_cox_ph_recovers_beta_direction():
    X, t, e, beta = sv_data(2, n=200)
    b = cox_ph(X, t, e, iters=300)
    cos = float(b @ beta / np.linalg.norm(b))
    assert cos > 0.8  # same direction as true beta (finite-sample + lr=0.01)


def test_cox_ph_deterministic():
    X, t, e, _ = sv_data(3, n=64)
    np.testing.assert_array_equal(cox_ph(X, t, e, iters=5), cox_ph(X, t, e, iters=5))


@pytest.mark.parametrize("iters", [0, -1])
def test_cox_ph_vacuous_iters(iters):
    X, t, e, _ = sv_data(0, n=16)
    with pytest.raises(ValueError):
        cox_ph(X, t, e, iters=iters)


def test_cox_ph_hostile_inputs():
    X, t, e, _ = sv_data(0, n=16)
    with pytest.raises(ValueError):
        cox_ph(X, t, np.ones(10, dtype=np.int64))  # length mismatch
    with pytest.raises(ValueError):
        cox_ph(X, t, e * 2)  # non-binary events
    with pytest.raises(ValueError):
        cox_ph(np.zeros((0, 3)), t[:0], e[:0])
